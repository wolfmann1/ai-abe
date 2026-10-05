"""The agent: retrieve passages, ask the model, return an answer with its sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .index import BM25Index
from .providers import Provider, make_provider
from .spec import AgentSpec

DECLINE_MESSAGE = (
    "I don't have enough in my knowledge base to answer that reliably. "
    "Please check with a person who owns this area."
)


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    declined: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class Agent:
    def __init__(self, spec: AgentSpec, index: BM25Index, provider: Provider | None = None):
        self.spec = spec
        self.index = index
        self.provider = provider or make_provider(spec.provider)

    @classmethod
    def from_directory(cls, directory: str | Path, provider: Provider | None = None) -> Agent:
        directory = Path(directory)
        spec = AgentSpec.load(directory / "agent.yaml")
        index = BM25Index.load(directory / "index.json")
        return cls(spec, index, provider)

    def ask(self, question: str) -> Answer:
        hits = self.index.search(question, self.spec.retrieval.top_k)
        retrieved = [{"source": c.source, "id": c.id, "score": s, "text": c.text} for c, s in hits]
        if not hits or hits[0][1] < self.spec.retrieval.min_score:
            return Answer(DECLINE_MESSAGE, retrieved=retrieved, declined=True)

        passages = [(n, chunk.text) for n, (chunk, _score) in enumerate(hits, start=1)]
        if self.provider.extractive:
            completion = self.provider.answer(question, passages)
            if not completion.text:
                return Answer(DECLINE_MESSAGE, retrieved=retrieved, declined=True)
        else:
            completion = self.provider.complete(self.spec.system_prompt(), build_user_message(question, hits))

        cited = cited_sources(completion.text, hits)
        return Answer(
            text=completion.text.strip(),
            sources=cited,
            retrieved=retrieved,
            declined=False,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )


def build_user_message(question: str, hits) -> str:
    blocks = [f"[{n}] (source: {chunk.source})\n{chunk.text}" for n, (chunk, _s) in enumerate(hits, start=1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def cited_sources(text: str, hits) -> list[str]:
    """Sources the answer cites by [n]; if it cites none, every retrieved source."""
    cited: list[str] = []
    for n, (chunk, _s) in enumerate(hits, start=1):
        if f"[{n}]" in text and chunk.source not in cited:
            cited.append(chunk.source)
    if not cited:
        for chunk, _s in hits:
            if chunk.source not in cited:
                cited.append(chunk.source)
    return cited
