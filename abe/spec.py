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
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers sent with every request, e.g. OpenRouter's app attribution headers.",
    )


class RetrievalConfig(BaseModel):
    chunk_size: int = Field(800, description="Target chunk length in characters.")
    chunk_overlap: int = Field(150, description="Characters repeated between adjacent chunks.")
    top_k: int = Field(4, description="Chunks passed to the model per question.")
    min_score: float = Field(
        0.5,
        description="BM25 score below which the best match is treated as no match and the agent declines.",
    )


Effort = Literal["quick", "standard", "thorough", "custom"]

# What each effort level sets. Passages and answer length are applied to the agent's settings when the
# level is chosen, and can still be fine-tuned afterwards; the instruction goes into the system prompt.
EFFORT_PRESETS: dict[str, dict] = {
    "quick": {
        "top_k": 3,
        "max_tokens": 300,
        "instruction": "Answer in one to three sentences.",
    },
    "standard": {"top_k": 4, "max_tokens": 800, "instruction": ""},
    "thorough": {
        "top_k": 8,
        "max_tokens": 1500,
        "instruction": (
            "Check every passage before answering. Cover each part of the question, and state anything the "
            "passages leave unanswered."
        ),
    },
    # Your own wording in `effort_instruction`; passages and answer length are left as they are.
    "custom": {"top_k": None, "max_tokens": None, "instruction": ""},
}


class AnswerStyle(BaseModel):
    """Optional instructions added to the system prompt."""

    steps: bool = Field(False, description="Give procedures as numbered steps.")
    code_blocks: bool = Field(False, description="Show commands, paths and settings as code, exactly as written.")
    version_aware: bool = Field(False, description="Say which version, product or region each part applies to.")
    flag_conflicts: bool = Field(False, description="Point out passages that disagree instead of picking one.")
    general_knowledge: bool = Field(False, description="Allow general knowledge, labelled as not from the documents.")
    custom: list[str] = Field(default_factory=list, description="Your own style instructions, one per line.")

    def instructions(self) -> list[str]:
        lines = []
        if self.steps:
            lines.append("When the answer is a procedure, give numbered steps.")
        if self.code_blocks:
            lines.append(
                "Put commands, file paths and settings in code blocks, exactly as written in the passage."
            )
        if self.version_aware:
            lines.append(
                "If the passages describe different versions, products or regions, say which one each part of "
                "the answer applies to."
            )
        if self.flag_conflicts:
            lines.append("If two passages disagree, say so and cite both rather than picking one.")
        lines.extend(line for line in self.custom if line.strip())
        return lines


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
    additional_rules: list[str] = Field(default_factory=list, description="Your own rules, one per line.")
    effort: Effort = "standard"
    effort_instruction: str = Field("", description="Your own effort instruction when effort is 'custom'.")
    prompt_override: str = Field(
        "", description="A system prompt written by hand. When set, it replaces the generated one entirely."
    )
    style: AnswerStyle = Field(default_factory=AnswerStyle)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    def model_post_init(self, __context) -> None:
        if not self.slug:
            self.slug = slugify(self.name)

    def system_prompt(self) -> str:
        if self.prompt_override.strip():
            return self.prompt_override.strip()
        return self.generated_prompt()

    def generated_prompt(self) -> str:
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
        if self.style.general_knowledge:
            lines.append("- Base your answer on the numbered context passages supplied with each question.")
            lines.append(
                "- You may add general knowledge where the passages are incomplete. Mark every such part "
                "'(Not from the documents)' so the reader can tell it apart."
            )
        else:
            lines.append("- Answer only from the numbered context passages supplied with each question.")
            lines.append(
                "- If the context does not contain the answer, say so plainly and suggest who or what "
                "the user should check instead. Do not guess."
            )
        if self.require_citations:
            lines.append("- Cite the passages you used with their numbers in square brackets, e.g. [1].")
        for item in self.out_of_scope:
            lines.append(f"- Out of scope, decline politely: {item}")
        for rule in self.additional_rules:
            if rule.strip():
                lines.append(f"- {rule.strip()}")
        if self.human_review:
            lines.append(
                "- End every answer with: 'Review required before acting on this answer.'"
            )
        for instruction in self.style.instructions():
            lines.append(f"- {instruction}")
        if self.effort == "custom":
            effort = self.effort_instruction.strip()
        else:
            effort = EFFORT_PRESETS[self.effort]["instruction"]
        if effort:
            lines.append(f"- {effort}")
        lines.append(f"- Style: {self.tone}")
        return "\n".join(lines)

    def with_effort(self, effort: str) -> AgentSpec:
        """A copy at the given effort level, with passages and answer length set to its preset."""
        preset = EFFORT_PRESETS[effort]
        if effort == "custom":
            return self.model_copy(update={"effort": effort})
        return self.model_copy(update={
            "effort": effort,
            "retrieval": self.retrieval.model_copy(update={"top_k": preset["top_k"]}),
            "provider": self.provider.model_copy(update={"max_tokens": preset["max_tokens"]}),
        })

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
