"""Web intake form and agent workbench (Starlette + Jinja2, no JavaScript build step).

    GET  /                        intake form
    POST /build                   build an agent package from the form
    GET  /agents/{slug}           package summary, ask box, eval button
    POST /agents/{slug}/ask       answer one question
    POST /agents/{slug}/eval      run the test set
    GET  /agents/{slug}/download  zip of the package
    POST /agents/{slug}/pull      download the agent's Ollama model in the background
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
from .cost import CostInputs
from .evals import TestSet, run_eval, write_report
from .ingest import SUPPORTED_SUFFIXES
from .intake import CRITERIA, Criterion, Intake
from .providers import OPENROUTER_HEADERS, OPENROUTER_URL, ProviderError
from .spec import AgentSpec, ProviderConfig, slugify

MAX_FILES = 50
MAX_FILE_BYTES = 25 * 1024 * 1024
SAMPLE_ROWS = 5

PROVIDER_CHOICES = {
    "azure_openai": ("Azure OpenAI", "azure_openai", "", "AZURE_OPENAI_API_KEY"),
    "ollama": ("Local model (Ollama)", "openai_compatible", "", ""),
    "openrouter": ("OpenRouter (prepaid credits, many models)", "openai_compatible", OPENROUTER_URL,
                   "OPENROUTER_API_KEY"),
    "openai": ("OpenAI", "openai_compatible", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("Anthropic", "anthropic", "", "ANTHROPIC_API_KEY"),
    "extractive": ("No model: quote the best passages (offline baseline)", "extractive", "", ""),
}

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
        return render("agent.html", **await run_in_threadpool(_agent_context, agent_dir))

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


def error_hints(message: str) -> list[str]:
    """Plain-language suggestions for the provider errors people hit most often."""
    text = message.lower()
    hints: list[str] = []
    if "out of memory" in text or "cuda error" in text:
        hints += [
            "The model didn't fit in your graphics card's memory (VRAM). A model fits when its download size plus "
            "about 2 GB is no more than your VRAM: on an 8 GB card, gemma4:e4b-it-qat or llama3.1:8b; a 26B model "
            "needs about 20 GB. See docs/choosing-a-local-model.md.",
            "Pick a smaller model or a smaller tag (for example 4b or 8b instead of 26b) and change 'model' in "
            "agent.yaml.",
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


def is_ollama(provider: ProviderConfig) -> bool:
    if provider.kind != "openai_compatible":
        return False
    base = ollama.host_from_endpoint(provider.endpoint)
    return base == ollama.host() or ":11434" in base


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

    choice = text("provider", "azure_openai")
    if choice not in PROVIDER_CHOICES:
        raise ValueError(f"Unknown provider choice: {choice}")
    _label, kind, default_endpoint, default_key_env = PROVIDER_CHOICES[choice]
    if choice == "ollama":
        default_endpoint = ollama.openai_endpoint()
    provider = ProviderConfig(
        kind=kind,
        model=text("model"),
        endpoint=text("endpoint") or default_endpoint,
        api_key_env=text("api_key_env") or default_key_env,
        extra_headers=dict(OPENROUTER_HEADERS) if choice == "openrouter" else {},
    )

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
        provider=provider,
    )

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
