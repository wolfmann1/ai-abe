"""Turn a blueprint (the intake form's answers) and a set of documents into an agent package.

An agent package is a folder:

    <slug>/
      blueprint.yaml      the answers the package was built from; rebuild with `abe build`
      agent.yaml          the AgentSpec every runtime reads
      system_prompt.md    the prompt generated from the spec, for review or import elsewhere
      docs/               copies of the source documents
      index.json          the retrieval index
      evals/testset.yaml  starter test set from the sample questions
      intake.md           intake screen and recommendation
      cost.md             monthly cost model
      mcp.json            MCP client configuration snippet
      README.md           how to run this agent
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .cost import CostInputs
from .index import BM25Index
from .ingest import SUPPORTED_SUFFIXES, ingest_directory
from .intake import Intake
from .spec import AgentSpec

OUT_OF_SCOPE_PROBE = "What will the weather be on Mars next Tuesday?"


class SampleQuestion(BaseModel):
    question: str
    must_mention: list[str] = Field(default_factory=list)
    source: str | None = None
    should_decline: bool = False


class Blueprint(BaseModel):
    spec: AgentSpec
    intake: Intake = Field(default_factory=Intake)
    cost: CostInputs | None = None
    sample_questions: list[SampleQuestion] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Blueprint:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def build_agent(blueprint: Blueprint, documents: list[Path], workspace: str | Path) -> Path:
    spec = blueprint.spec
    agent_dir = Path(workspace) / spec.slug
    docs_dir = agent_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for document in documents:
        document = Path(document)
        if document.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {document.name}")
        target = docs_dir / document.name
        if document.resolve() != target.resolve():
            shutil.copy2(document, target)

    blueprint.save(agent_dir / "blueprint.yaml")
    spec.save(agent_dir / "agent.yaml")
    (agent_dir / "system_prompt.md").write_text(spec.system_prompt() + "\n", encoding="utf-8")

    index = build_index(agent_dir)
    write_testset(blueprint, agent_dir / "evals" / "testset.yaml")
    (agent_dir / "intake.md").write_text(blueprint.intake.to_markdown(spec.name), encoding="utf-8")
    cost = blueprint.cost or CostInputs.estimate_for(spec)
    (agent_dir / "cost.md").write_text(cost.to_markdown(spec.name), encoding="utf-8")
    write_mcp_config(agent_dir, spec.slug)
    (agent_dir / "README.md").write_text(package_readme(spec, index), encoding="utf-8")
    return agent_dir


def build_index(agent_dir: str | Path) -> BM25Index:
    agent_dir = Path(agent_dir)
    spec = AgentSpec.load(agent_dir / "agent.yaml")
    chunks = ingest_directory(agent_dir / "docs", spec.retrieval.chunk_size, spec.retrieval.chunk_overlap)
    index = BM25Index(chunks)
    index.save(agent_dir / "index.json")
    return index


def write_testset(blueprint: Blueprint, path: Path) -> None:
    cases = []
    for i, sample in enumerate(blueprint.sample_questions, start=1):
        if not sample.question.strip():
            continue
        case: dict = {"id": f"sample-{i}", "question": sample.question.strip()}
        if sample.should_decline:
            case["expect_decline"] = True
        else:
            case["expect_contains"] = [m for m in sample.must_mention if m.strip()]
            if sample.source:
                case["expect_source"] = sample.source
        cases.append(case)
    cases.append({"id": "out-of-scope-probe", "question": OUT_OF_SCOPE_PROBE, "expect_decline": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Starter test set generated from the intake form.\n"
        "# Add a case for every question users actually ask, and every wrong answer you find.\n"
    )
    path.write_text(
        header + yaml.safe_dump({"threshold": 0.8, "cases": cases}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_mcp_config(agent_dir: Path, slug: str) -> None:
    python = sys.executable if sys.executable else "python"
    path = str(agent_dir.resolve())
    config = {
        "mcpServers": {
            # Uses the model configured in agent.yaml for the ask tool.
            f"abe-{slug}": {"command": python, "args": ["-m", "abe", "mcp", path]},
        },
        # Alternative: retrieval only. The MCP client's own model answers, on the user's existing
        # subscription (Claude Desktop, GitHub Copilot in VS Code). Use one entry or the other, not both.
        "subscriptionAlternative": {
            f"abe-{slug}": {"command": python, "args": ["-m", "abe", "mcp", path, "--retrieval-only"]},
        },
    }
    (agent_dir / "mcp.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def package_readme(spec: AgentSpec, index: BM25Index) -> str:
    sources = "\n".join(f"- {s}" for s in index.sources()) or "- (none yet)"
    return f"""# {spec.name}

{spec.description}

Built with [ai-abe](https://github.com/wolfmann1/ai-abe). Provider: `{spec.provider.kind}`.
Knowledge base: {len(index.chunks)} passages from {len(index.sources())} documents.

{sources}

## Use it

```
abe ask {spec.slug} "your question"        # one question from the command line
abe eval {spec.slug}                        # run evals/testset.yaml and write evals/report.md
abe mcp {spec.slug}                         # serve it to an MCP client over stdio
```

Run these from the workspace folder that contains `{spec.slug}/`, or pass the full path.
`mcp.json` holds a ready-made client entry for Claude Desktop, VS Code or any MCP client.

## Change it

- Add or replace files in `docs/`, then `abe reindex {spec.slug}`.
- Edit `agent.yaml` to change the model, retrieval settings or rules; `system_prompt.md` is
  regenerated by `abe build` and shown here for review.
- Add cases to `evals/testset.yaml` before changing anything, so you can see what the change did.
"""


def zip_agent(agent_dir: str | Path) -> Path:
    agent_dir = Path(agent_dir)
    target = agent_dir.parent / f"{agent_dir.name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(agent_dir.rglob("*")):
            if path.is_file():
                zf.write(path, Path(agent_dir.name) / path.relative_to(agent_dir))
    return target
