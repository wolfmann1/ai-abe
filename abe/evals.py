"""Evaluation harness: run a test set against an agent and score it.

A test set is a YAML file:

    threshold: 0.8          # minimum accuracy for the run to pass (CI gate)
    cases:
      - id: disk-full
        question: What do I do when the cache volume is above 90%?
        expect_contains: ["purge", "90%"]     # every phrase must appear (case-insensitive)
        expect_source: runbook-disk-pressure.txt   # optional: must be among the cited sources
      - id: out-of-scope
        question: What is the CEO's salary?
        expect_decline: true                  # the agent must decline rather than answer

Each case passes or fails, with the reasons recorded. Retrieval is scored
separately from the answer, so a failure can be traced to the right layer: the
right passage was never found, or it was found and the model misused it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .agent import Agent


class EvalCase(BaseModel):
    id: str
    question: str
    expect_contains: list[str] = Field(default_factory=list)
    expect_source: str | None = None
    expect_decline: bool = False


class TestSet(BaseModel):
    threshold: float = 0.8
    cases: list[EvalCase]

    @classmethod
    def load(cls, path: str | Path) -> TestSet:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    retrieval_hit: bool | None
    reasons: list[str] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EvalReport:
    agent: str
    provider: str
    threshold: float
    results: list[CaseResult]

    @property
    def accuracy(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def retrieval_hit_rate(self) -> float | None:
        scored = [r.retrieval_hit for r in self.results if r.retrieval_hit is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def passed(self) -> bool:
        return self.accuracy >= self.threshold

    def avg_tokens(self) -> tuple[float, float]:
        answered = [r for r in self.results if r.input_tokens or r.output_tokens]
        if not answered:
            return 0.0, 0.0
        return (
            sum(r.input_tokens for r in answered) / len(answered),
            sum(r.output_tokens for r in answered) / len(answered),
        )

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "provider": self.provider,
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 4),
            "retrieval_hit_rate": None if self.retrieval_hit_rate is None else round(self.retrieval_hit_rate, 4),
            "passed": self.passed,
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        hit = self.retrieval_hit_rate
        lines = [
            f"# Evaluation: {self.agent}",
            "",
            f"Provider: `{self.provider}`",
            "",
            "| Measure | Result |",
            "|---|---|",
            f"| Answer accuracy | {self.accuracy:.0%} ({sum(r.passed for r in self.results)}/{len(self.results)}) |",
            f"| Retrieval hit rate | {'n/a' if hit is None else f'{hit:.0%}'} |",
            f"| Threshold | {self.threshold:.0%} |",
            f"| Result | {'PASS' if self.passed else 'FAIL'} |",
            "",
            "| Case | Result | Retrieval | Notes |",
            "|---|---|---|---|",
        ]
        for r in self.results:
            retrieval = "n/a" if r.retrieval_hit is None else ("hit" if r.retrieval_hit else "miss")
            notes = "; ".join(r.reasons).replace("|", "/") or "—"
            lines.append(f"| {r.id} | {'pass' if r.passed else 'FAIL'} | {retrieval} | {notes} |")
        return "\n".join(lines) + "\n"


def run_eval(agent: Agent, testset: TestSet) -> EvalReport:
    results = [_run_case(agent, case) for case in testset.cases]
    return EvalReport(agent.spec.name, agent.provider.kind, testset.threshold, results)


def _run_case(agent: Agent, case: EvalCase) -> CaseResult:
    answer = agent.ask(case.question)
    reasons: list[str] = []
    retrieved_sources = {r["source"] for r in answer.retrieved}
    retrieval_hit = (case.expect_source in retrieved_sources) if case.expect_source else None

    if case.expect_decline:
        if not answer.declined:
            reasons.append("expected a decline, got an answer")
    else:
        if answer.declined:
            reasons.append("declined a question it should answer")
        lowered = answer.text.lower()
        for phrase in case.expect_contains:
            if phrase.lower() not in lowered:
                reasons.append(f"missing '{phrase}'")
        if case.expect_source and case.expect_source not in answer.sources:
            reasons.append(f"did not cite {case.expect_source}")

    return CaseResult(
        id=case.id,
        question=case.question,
        passed=not reasons,
        retrieval_hit=retrieval_hit,
        reasons=reasons,
        answer=answer.text,
        sources=answer.sources,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )


def write_report(report: EvalReport, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    md = directory / "report.md"
    js = directory / "report.json"
    md.write_text(report.to_markdown(), encoding="utf-8")
    js.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return md, js
