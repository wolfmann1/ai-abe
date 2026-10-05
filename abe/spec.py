"""Agent specification: the single file that describes an agent.

Everything ABE generates is derived from an AgentSpec, and everything that runs an
agent (the web app, the CLI, the MCP server, the eval harness) reads one back.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ProviderKind = Literal["azure_openai", "openai_compatible", "anthropic", "extractive"]


class ProviderConfig(BaseModel):
    """Which model answers questions, and how to reach it.

    API keys are never stored in the spec. `api_key_env` names the environment
    variable that holds the key.
    """

    kind: ProviderKind = "azure_openai"
    model: str = ""
    endpoint: str = ""
    api_version: str = "2024-10-21"
    api_key_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 800


class RetrievalConfig(BaseModel):
    chunk_size: int = Field(800, description="Target chunk length in characters.")
    chunk_overlap: int = Field(150, description="Characters repeated between adjacent chunks.")
    top_k: int = Field(4, description="Chunks passed to the model per question.")
    min_score: float = Field(
        0.5,
        description="BM25 score below which the best match is treated as no match and the agent declines.",
    )


class AgentSpec(BaseModel):
    name: str
    slug: str = ""
    description: str = ""
    problem: str = ""
    goal: str = ""
    audience: str = ""
    out_of_scope: list[str] = Field(default_factory=list)
    tone: str = "Plain and direct. Short answers first, detail after."
    require_citations: bool = True
    human_review: bool = False
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    def model_post_init(self, __context) -> None:
        if not self.slug:
            self.slug = slugify(self.name)

    def system_prompt(self) -> str:
        lines = [f"You are {self.name}."]
        if self.description:
            lines.append(self.description)
        if self.audience:
            lines.append(f"You are answering questions from: {self.audience}.")
        if self.problem:
            lines.append(f"The problem you exist to solve: {self.problem}")
        if self.goal:
            lines.append(f"A good outcome looks like: {self.goal}")
        lines.append("")
        lines.append("Rules:")
        lines.append("- Answer only from the numbered context passages supplied with each question.")
        lines.append(
            "- If the context does not contain the answer, say so plainly and suggest who or what "
            "the user should check instead. Do not guess."
        )
        if self.require_citations:
            lines.append("- Cite the passages you used with their numbers in square brackets, e.g. [1].")
        for item in self.out_of_scope:
            lines.append(f"- Out of scope, decline politely: {item}")
        if self.human_review:
            lines.append(
                "- End every answer with: 'Review required before acting on this answer.'"
            )
        lines.append(f"- Style: {self.tone}")
        return "\n".join(lines)

    @classmethod
    def load(cls, path: str | Path) -> AgentSpec:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "agent"
