"""Intake screen: decide whether an AI agent is worth building before building it.

Six criteria, each scored 1 (blocking) to 5 (strong):

    data_readiness         Do the documents exist, are they current, and can we use them?
    technical_feasibility  Can retrieval plus a language model plausibly answer these questions?
    business_readiness     Is there a sponsor, a user group, and a process that will change?
    minimum_accuracy       Can the users tolerate the error rate a first version will have?
    verifiability          Can a user check an answer against its source quickly?
    build_vs_buy           Is building better than an existing product or a plain search box?
"""

from __future__ import annotations

from pydantic import BaseModel, Field

CRITERIA = {
    "data_readiness": "Data readiness",
    "technical_feasibility": "Technical feasibility",
    "business_readiness": "Business readiness",
    "minimum_accuracy": "Minimum viable accuracy",
    "verifiability": "Output verifiability",
    "build_vs_buy": "Build versus buy",
}


class Criterion(BaseModel):
    score: int = Field(3, ge=1, le=5)
    notes: str = ""


class Intake(BaseModel):
    data_readiness: Criterion = Field(default_factory=Criterion)
    technical_feasibility: Criterion = Field(default_factory=Criterion)
    business_readiness: Criterion = Field(default_factory=Criterion)
    minimum_accuracy: Criterion = Field(default_factory=Criterion)
    verifiability: Criterion = Field(default_factory=Criterion)
    build_vs_buy: Criterion = Field(default_factory=Criterion)
    value_statement: str = ""

    def scores(self) -> dict[str, int]:
        return {key: getattr(self, key).score for key in CRITERIA}

    def recommendation(self) -> tuple[str, str]:
        scores = self.scores()
        blocking = [CRITERIA[k] for k, v in scores.items() if v <= 1]
        weak = [CRITERIA[k] for k, v in scores.items() if v == 2]
        average = sum(scores.values()) / len(scores)
        if blocking:
            return "Do not build yet", f"Blocked by: {', '.join(blocking)}."
        if weak:
            return (
                "Time-boxed prototype",
                f"Average {average:.1f}/5. Resolve before committing a team: {', '.join(weak)}.",
            )
        if average >= 3.5:
            return "Proceed", f"Average {average:.1f}/5 with no weak criteria."
        return "Time-boxed prototype", f"Average {average:.1f}/5. No blockers, but no strong case yet."

    def to_markdown(self, agent_name: str) -> str:
        verdict, reason = self.recommendation()
        lines = [
            f"# Intake screen: {agent_name}",
            "",
            f"**Recommendation: {verdict}.** {reason}",
            "",
        ]
        if self.value_statement:
            lines += ["## Expected value", "", self.value_statement, ""]
        lines += ["## Criteria", "", "| Criterion | Score | Notes |", "|---|---|---|"]
        for key, label in CRITERIA.items():
            item: Criterion = getattr(self, key)
            lines.append(f"| {label} | {item.score}/5 | {item.notes.replace('|', '/') or '—'} |")
        lines += [
            "",
            "Scores: 1 = blocking, 3 = workable, 5 = strong. Any 1 stops the build; any 2 limits it to a "
            "time-boxed prototype.",
            "",
        ]
        return "\n".join(lines)
