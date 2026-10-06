"""Web intake form and agent workbench (Starlette + Jinja2, no JavaScript build step).

    GET  /                        intake form
    POST /build                   build an agent package from the form
    GET  /agents/{slug}           package summary, ask box, eval button
    POST /agents/{slug}/ask       answer one question
    POST /agents/{slug}/eval      run the test set
    GET  /agents/{slug}/download  zip of the package
    POST /agents/{slug}/pull      download the agent's Ollama model in the background
    GET  /agents/{slug}/edit      edit form: model, rules, retrieval, documents, test set
    POST /agents/{slug}/edit      save the edits
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from . import ollama
from .agent import Agent
from .builder import Blueprint, SampleQuestion, build_agent, zip_agent
from .editor import update_agent
from .cost import CostInputs
from .evals import TestSet, run_eval, write_report
from .ingest import SUPPORTED_SUFFIXES
from .intake import CRITERIA, Criterion, Intake
from .providers import PROVIDER_CHOICES, ProviderError, choice_for, provider_from_choice
from .spec import EFFORT_PRESETS, AgentSpec, AnswerStyle, ProviderConfig, slugify

MAX_FILES = 50
MAX_FILE_BYTES = 25 * 1024 * 1024
SAMPLE_ROWS = 5


env = Environment(loader=PackageLoader("abe", "templates"), autoescape=select_autoescape(["html"]))


def render(name: str, status: int = 200, **context) -> HTMLResponse:
    return HTMLResponse(env.get_template(name).render(**context), status_code=status)


def create_app(workspace: str | Path) -> Starlette:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    def agent_dir_for(slug: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]{1,60}", slug):
            raise FileNotFoundError(slug)
        path = workspace / slug
        if not (path / "agent.yaml").exists():
            raise FileNotFoundError(slug)
        return path

    async def form(request: Request) -> Response:
        agents = sorted(p.name for p in workspace.iterdir() if (p / "agent.yaml").exists())
        ollama_models = await run_in_threadpool(ollama.list_models)
        return render(
            "form.html",
            ollama_models=ollama_models,
            ollama_host=ollama.host(),
            providers=PROVIDER_CHOICES,
            criteria=CRITERIA,
            sample_rows=range(1, SAMPLE_ROWS + 1),
            suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)),
            agents=agents,
        )

    async def build(request: Request) -> Response:
        data = await request.form(max_files=MAX_FILES, max_fields=200)
        try:
            return await _build(data)
        finally:
            await data.close()

    async def _build(data) -> Response:
        try:
            blueprint = blueprint_from_form(data)
        except ValueError as exc:
            return render("error.html", message=str(exc), status=400)

        with tempfile.TemporaryDirectory() as tmp:
            saved: list[Path] = []
            for upload in data.getlist("documents"):
                if not getattr(upload, "filename", None):
                    continue
                name = safe_filename(upload.filename)
                if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                    return render("error.html", message=f"{upload.filename}: unsupported file type.", status=400)
                content = await upload.read()
                if len(content) > MAX_FILE_BYTES:
                    return render("error.html", message=f"{upload.filename} is larger than 25 MB.", status=400)
                target = Path(tmp) / name
                target.write_bytes(content)
                saved.append(target)
            if not saved and not (workspace / blueprint.spec.slug / "docs").exists():
                return render("error.html", message="Upload at least one PDF, DOCX or TXT file.", status=400)
            try:
                build_agent(blueprint, saved, workspace)
            except ValueError as exc:
                return render("error.html", message=str(exc), status=400)
        provider = blueprint.spec.provider
        if is_ollama(provider) and provider.model:
            base = ollama.host_from_endpoint(provider.endpoint)
            installed = await run_in_threadpool(ollama.list_models, base)
            if installed is not None and not ollama.is_installed(provider.model, installed):
                ollama.start_pull(provider.model, base)
        return RedirectResponse(f"/agents/{blueprint.spec.slug}", status_code=303)

    async def show(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        context = await run_in_threadpool(_agent_context, agent_dir)
        if request.query_params.get("saved"):
            context["saved"] = last_changes.pop(slug, None) or True
        return render("agent.html", **context)

    async def ask(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        data = await request.form()
        question = str(data.get("question", "")).strip()
        context = await run_in_threadpool(_agent_context, agent_dir)
        if question:
            try:
                context["answer"] = Agent.from_directory(agent_dir).ask(question)
            except ProviderError as exc:
                context["error"] = str(exc)
                context["error_hints"] = error_hints(str(exc))
            context["question"] = question
        return render("agent.html", **context)

    async def evaluate(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        context = await run_in_threadpool(_agent_context, agent_dir)
        try:
            report = run_eval(Agent.from_directory(agent_dir), TestSet.load(agent_dir / "evals" / "testset.yaml"))
            write_report(report, agent_dir / "evals")
            context["report"] = report
        except ProviderError as exc:
            context["error"] = str(exc)
            context["error_hints"] = error_hints(str(exc))
        return render("agent.html", **context)

    async def download(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        archive = zip_agent(agent_dir)
        return FileResponse(archive, filename=archive.name, media_type="application/zip")

    last_changes: dict[str, object] = {}

    async def edit_form(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        return render("edit.html", **await run_in_threadpool(_edit_context, agent_dir))

    async def edit_save(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        data = await request.form(max_files=MAX_FILES, max_fields=200)
        try:
            old = AgentSpec.load(agent_dir / "agent.yaml")
            spec = spec_from_edit_form(data, old)
            with tempfile.TemporaryDirectory() as tmp:
                added: list[Path] = []
                for upload in data.getlist("documents"):
                    if not getattr(upload, "filename", None):
                        continue
                    name = safe_filename(upload.filename)
                    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                        raise ValueError(f"{upload.filename}: unsupported file type.")
                    content = await upload.read()
                    if len(content) > MAX_FILE_BYTES:
                        raise ValueError(f"{upload.filename} is larger than 25 MB.")
                    target = Path(tmp) / name
                    target.write_bytes(content)
                    added.append(target)
                result = await run_in_threadpool(
                    update_agent,
                    agent_dir,
                    spec,
                    add_docs=added,
                    remove_docs=[str(v) for v in data.getlist("remove_doc")],
                    testset_yaml=str(data.get("testset", "")) if data.get("testset") is not None else None,
                )
        except ValueError as exc:
            context = await run_in_threadpool(_edit_context, agent_dir)
            context["error"] = str(exc)
            return render("edit.html", status=400, **context)
        finally:
            await data.close()

        provider = spec.provider
        if is_ollama(provider) and provider.model:
            base = ollama.host_from_endpoint(provider.endpoint)
            installed = await run_in_threadpool(ollama.list_models, base)
            if installed is not None and not ollama.is_installed(provider.model, installed):
                ollama.start_pull(provider.model, base)
        last_changes[slug] = result
        return RedirectResponse(f"/agents/{slug}?saved=1", status_code=303)

    async def pull(request: Request) -> Response:
        slug = request.path_params["slug"]
        try:
            agent_dir = agent_dir_for(slug)
        except FileNotFoundError:
            return render("error.html", message=f"No agent named {slug}.", status=404)
        provider = AgentSpec.load(agent_dir / "agent.yaml").provider
        if is_ollama(provider) and provider.model:
            ollama.start_pull(provider.model, ollama.host_from_endpoint(provider.endpoint))
        return RedirectResponse(f"/agents/{slug}", status_code=303)

    return Starlette(
        routes=[
            Route("/", form),
            Route("/build", build, methods=["POST"]),
            Route("/agents/{slug}", show),
            Route("/agents/{slug}/ask", ask, methods=["POST"]),
            Route("/agents/{slug}/eval", evaluate, methods=["POST"]),
            Route("/agents/{slug}/download", download),
            Route("/agents/{slug}/pull", pull, methods=["POST"]),
            Route("/agents/{slug}/edit", edit_form),
            Route("/agents/{slug}/edit", edit_save, methods=["POST"]),
        ]
    )


def _agent_context(agent_dir: Path) -> dict:
    spec = AgentSpec.load(agent_dir / "agent.yaml")
    blueprint = Blueprint.load(agent_dir / "blueprint.yaml")
    intake = blueprint.intake if blueprint.intake and not blueprint.intake.is_empty() else None
    verdict, reason = intake.recommendation() if intake else (None, None)
    files = sorted(
        str(p.relative_to(agent_dir)).replace("\\", "/") for p in agent_dir.rglob("*") if p.is_file()
    )
    return {
        "spec": spec,
        "verdict": verdict,
        "verdict_reason": reason,
        "files": files,
        "cost_md": (agent_dir / "cost.md").read_text(encoding="utf-8"),
        "system_prompt": spec.system_prompt(),
        "ollama": _ollama_context(spec.provider),
    }


def _edit_context(agent_dir: Path) -> dict:
    spec = AgentSpec.load(agent_dir / "agent.yaml")
    testset = agent_dir / "evals" / "testset.yaml"
    docs = sorted(p.name for p in (agent_dir / "docs").iterdir() if p.is_file())
    return {
        "spec": spec,
        "docs": docs,
        "testset": testset.read_text(encoding="utf-8") if testset.exists() else "threshold: 0.8\ncases: []\n",
        "providers": PROVIDER_CHOICES,
        "ollama_models": ollama.list_models(),
        "ollama_host": ollama.host(),
        "suffixes": ", ".join(sorted(SUPPORTED_SUFFIXES)),
        "current": {
            "choice": choice_for(spec.provider),
            "model": spec.provider.model,
            "endpoint": spec.provider.endpoint,
            "api_key_env": spec.provider.api_key_env,
        },
    }


def spec_from_edit_form(data, old: AgentSpec) -> AgentSpec:
    def text(key: str, default: str = "") -> str:
        value = data.get(key)
        return default if value is None else str(value).strip()

    def number(key: str, default: float, kind=float):
        raw = text(key)
        if not raw:
            return default
        try:
            return kind(raw)
        except ValueError as exc:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a number.") from exc

    name = text("name", old.name)
    if not name:
        raise ValueError("Give the agent a name.")
    provider = provider_from_form(data).model_copy(update={
        "max_tokens": number("max_tokens", old.provider.max_tokens, int),
        "temperature": number("temperature", old.provider.temperature),
        "api_version": old.provider.api_version,
    })
    retrieval = old.retrieval.model_copy(update={
        "top_k": number("top_k", old.retrieval.top_k, int),
        "min_score": number("min_score", old.retrieval.min_score),
        "chunk_size": number("chunk_size", old.retrieval.chunk_size, int),
        "chunk_overlap": number("chunk_overlap", old.retrieval.chunk_overlap, int),
    })
    if retrieval.top_k < 1 or retrieval.chunk_size < 100 or retrieval.chunk_overlap < 0:
        raise ValueError("Passages per question must be at least 1, and chunk size at least 100 characters.")
    updated = old.model_copy(update={
        "name": name,
        "description": text("description"),
        "problem": text("problem"),
        "goal": text("goal"),
        "audience": text("audience"),
        "out_of_scope": [line.strip() for line in text("out_of_scope").splitlines() if line.strip()],
        "tone": text("tone") or AgentSpec.model_fields["tone"].default,
        "require_citations": bool(data.get("require_citations")),
        "human_review": bool(data.get("human_review")),
        "style": style_from_form(data),
        "provider": provider,
        "retrieval": retrieval,
    })
    effort = effort_from_form(data, old.effort)
    if effort != old.effort:
        # A new level resets passages and answer length to its preset; otherwise keep the fine-tuned values.
        updated = updated.with_effort(effort)
    return updated


def error_hints(message: str) -> list[str]:
    """Plain-language suggestions for the provider errors people hit most often."""
    text = message.lower()
    hints: list[str] = []
    if "out of memory" in text or "cuda error" in text:
        hints += [
            "The model didn't fit in your graphics card's memory (VRAM). A model fits when its download size plus "
            "about 2 GB is no more than your VRAM: on an 8 GB card, gemma4:e4b-it-qat or llama3.1:8b; a 26B model "
            "needs about 20 GB. See docs/choosing-a-local-model.md.",
            "Pick a smaller model or a smaller tag (for example 4b or 8b instead of 26b) with Edit agent, or "
            "'abe set-model'.",
            "Free VRAM: run 'ollama ps' to see what's loaded, 'ollama stop <model>' to unload it, and close games "
            "or other GPU-heavy programs.",
        ]
    if "could not reach" in text or "connection refused" in text or "failed to establish" in text:
        hints.append("Nothing answered at the model's address. If it's Ollama, start it and try again.")
    if "not found" in text and "model" in text:
        hints.append("The model name isn't installed. Run 'ollama list' to see installed names, or "
                     "'ollama pull <model>' to install it.")
    if "api key" in text:
        hints.append("Set the key in the same terminal before running 'abe serve', then restart it.")
    return hints


def style_from_form(data) -> AnswerStyle:
    return AnswerStyle(**{field: bool(data.get(f"style_{field}")) for field in AnswerStyle.model_fields})


def effort_from_form(data, default: str = "standard") -> str:
    effort = str(data.get("effort") or default)
    if effort not in EFFORT_PRESETS:
        raise ValueError(f"Unknown effort level: {effort}")
    return effort


def is_ollama(provider: ProviderConfig) -> bool:
    return choice_for(provider) == "ollama"


def provider_from_form(data) -> ProviderConfig:
    """Provider settings from the model fields of the build or edit form.

    When editing, an endpoint or key variable left over from the previous provider is
    replaced by the new provider's default, so switching from Azure to Ollama doesn't
    keep the Azure address.
    """
    def text(key: str) -> str:
        return str(data.get(key, "") or "").strip()

    choice = text("provider") or "azure_openai"
    endpoint, key_env = text("endpoint"), text("api_key_env")
    if text("original_provider") and choice != text("original_provider"):
        if endpoint == text("original_endpoint"):
            endpoint = ""
        if key_env == text("original_api_key_env"):
            key_env = ""
    if choice == "ollama":
        endpoint = ""
    try:
        return provider_from_choice(choice, text("model"), endpoint, key_env)
    except ProviderError as exc:
        raise ValueError(str(exc)) from exc


def _ollama_context(provider: ProviderConfig) -> dict | None:
    """What the agent page should say about the agent's local model, or None for other providers."""
    if not is_ollama(provider) or not provider.model:
        return None
    base = ollama.host_from_endpoint(provider.endpoint)
    installed = ollama.list_models(base)
    return {
        "model": provider.model,
        "host": base,
        "reachable": installed is not None,
        "installed": installed is not None and ollama.is_installed(provider.model, installed),
        "pull": ollama.pull_status(provider.model),
    }


def blueprint_from_form(data) -> Blueprint:
    def text(key: str, default: str = "") -> str:
        return str(data.get(key, default) or default).strip()

    def number(key: str, default: float) -> float:
        raw = text(key)
        try:
            return float(raw) if raw else default
        except ValueError as exc:
            raise ValueError(f"{key} must be a number.") from exc

    name = text("name")
    if not name:
        raise ValueError("Give the agent a name.")

    provider = provider_from_form(data)

    spec = AgentSpec(
        name=name,
        slug=slugify(name),
        description=text("description"),
        problem=text("problem"),
        goal=text("goal"),
        audience=text("audience"),
        out_of_scope=[line.strip() for line in text("out_of_scope").splitlines() if line.strip()],
        tone=text("tone") or AgentSpec.model_fields["tone"].default,
        require_citations=bool(data.get("require_citations")),
        human_review=bool(data.get("human_review")),
        style=style_from_form(data),
        provider=provider,
    ).with_effort(effort_from_form(data))

    def score(key: str) -> int | None:
        raw = text(key)
        return int(number(key, 0)) if raw else None

    intake = Intake(
        value_statement=text("value_statement"),
        **{key: Criterion(score=score(f"{key}_score"), notes=text(f"{key}_notes")) for key in CRITERIA},
    )
    if intake.is_empty():
        intake = None

    cost = CostInputs.estimate_for(
        spec,
        users=int(number("users", 25)),
        questions_per_user_per_day=number("questions_per_day", 5),
        price_per_million_input=number("price_in", 0),
        price_per_million_output=number("price_out", 0),
        currency=text("currency", "CAD") or "CAD",
    )

    samples = []
    for i in range(1, SAMPLE_ROWS + 1):
        question = text(f"q{i}")
        if not question:
            continue
        samples.append(
            SampleQuestion(
                question=question,
                must_mention=[m.strip() for m in text(f"q{i}_mention").split(",") if m.strip()],
                source=text(f"q{i}_source") or None,
                should_decline=bool(data.get(f"q{i}_decline")),
            )
        )

    return Blueprint(spec=spec, intake=intake, cost=cost, sample_questions=samples)


def safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    stem, suffix = Path(name).stem, Path(name).suffix
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" .") or "document"
    return f"{stem}{suffix.lower()}"
