"""Change an existing agent package in place.

Everything an agent does is driven by files in its package, so editing means
rewriting those files consistently:

    agent.yaml and blueprint.yaml   the new settings
    system_prompt.md and README.md  regenerated from them
    docs/ and index.json            documents added or removed; the index is rebuilt when
                                    documents or chunking settings change
    evals/testset.yaml              replaced if a new test set is supplied (validated first)
    cost.md                         recalculated, since retrieval settings change token counts

The agent's folder name (its slug) never changes, so MCP client entries and links keep working.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .builder import Blueprint, build_index, package_readme
from .cost import CostInputs
from .evals import TestSet
from .ingest import SUPPORTED_SUFFIXES
from .spec import AgentSpec


@dataclass
class EditResult:
    changes: list[str] = field(default_factory=list)
    reindexed: bool = False


def update_agent(
    agent_dir: str | Path,
    spec: AgentSpec,
    *,
    add_docs: list[Path] | None = None,
    remove_docs: list[str] | None = None,
    testset_yaml: str | None = None,
) -> EditResult:
    agent_dir = Path(agent_dir)
    old = AgentSpec.load(agent_dir / "agent.yaml")
    blueprint = Blueprint.load(agent_dir / "blueprint.yaml")
    spec = spec.model_copy(update={"slug": old.slug})
    result = EditResult()

    # Validate the test set before changing anything, so a typo can't leave the package half-edited.
    testset_text = None
    if testset_yaml is not None:
        testset_yaml = testset_yaml.replace("\r\n", "\n")
        current = (agent_dir / "evals" / "testset.yaml")
        if not current.exists() or testset_yaml.strip() != current.read_text(encoding="utf-8").strip():
            try:
                TestSet.model_validate(yaml.safe_load(testset_yaml))
            except Exception as exc:  # yaml or validation error, reported to the user as-is
                raise ValueError(f"The test set isn't valid: {exc}") from exc
            testset_text = testset_yaml.strip() + "\n"

    docs_dir = agent_dir / "docs"
    existing = {p.name for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES}
    remaining = (existing - {Path(n).name for n in remove_docs or []}) | {Path(d).name for d in add_docs or []}
    if not remaining:
        raise ValueError("An agent needs at least one document. Add one before removing the last.")
    docs_changed = False
    for name in remove_docs or []:
        target = docs_dir / Path(name).name
        if target.is_file() and target.parent.resolve() == docs_dir.resolve():
            target.unlink()
            result.changes.append(f"Removed {target.name}")
            docs_changed = True
    for document in add_docs or []:
        document = Path(document)
        if document.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {document.name}")
        shutil.copy2(document, docs_dir / document.name)
        result.changes.append(f"Added {document.name}")
        docs_changed = True

    if old.provider != spec.provider:
        before, after = _describe(old.provider), _describe(spec.provider)
        result.changes.append(f"Model: {before} → {after}" if before != after else "Model settings updated")
    if old.retrieval != spec.retrieval:
        result.changes.append("Retrieval settings updated")
    rules = ("name", "description", "problem", "goal", "audience", "out_of_scope", "tone",
             "require_citations", "human_review")
    if any(getattr(old, f) != getattr(spec, f) for f in rules):
        result.changes.append("Description and rules updated")
    if old.effort != spec.effort:
        result.changes.append(f"Effort: {old.effort} → {spec.effort}")
    if old.style != spec.style:
        result.changes.append("Answer style updated")

    spec.save(agent_dir / "agent.yaml")
    blueprint.spec = spec
    (agent_dir / "system_prompt.md").write_text(spec.system_prompt() + "\n", encoding="utf-8")

    chunking_changed = (old.retrieval.chunk_size, old.retrieval.chunk_overlap) != (
        spec.retrieval.chunk_size,
        spec.retrieval.chunk_overlap,
    )
    if docs_changed or chunking_changed:
        index = build_index(agent_dir)
        result.reindexed = True
    else:
        from .index import BM25Index

        index = BM25Index.load(agent_dir / "index.json")

    if testset_text is not None:
        (agent_dir / "evals").mkdir(exist_ok=True)
        (agent_dir / "evals" / "testset.yaml").write_text(testset_text, encoding="utf-8")
        result.changes.append("Test set updated")

    previous = blueprint.cost or CostInputs.estimate_for(old)
    cost = CostInputs.estimate_for(
        spec,
        users=previous.users,
        questions_per_user_per_day=previous.questions_per_user_per_day,
        working_days_per_month=previous.working_days_per_month,
        price_per_million_input=previous.price_per_million_input,
        price_per_million_output=previous.price_per_million_output,
        currency=previous.currency,
    )
    blueprint.cost = cost
    (agent_dir / "cost.md").write_text(cost.to_markdown(spec.name), encoding="utf-8")
    blueprint.save(agent_dir / "blueprint.yaml")
    (agent_dir / "README.md").write_text(package_readme(spec, index), encoding="utf-8")

    if not result.changes:
        result.changes.append("No changes")
    return result



def _describe(provider) -> str:
    from .providers import PROVIDER_CHOICES, choice_for

    choice = choice_for(provider)
    label = PROVIDER_CHOICES.get(choice, (choice,))[0].split(" (")[0]
    if choice == "extractive":
        return "no model (extractive)"
    return f"{provider.model} ({label})" if provider.model else label
