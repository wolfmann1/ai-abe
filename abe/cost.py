"""Monthly running-cost estimate for an agent.

Token prices change often and differ by provider, model and contract, so ABE ships
no prices. Enter the per-million-token rates from your provider's current price
sheet. With no rates entered the estimate reports token volume only.

Token counts start as an estimate from the retrieval settings (about four
characters per token for English text) and can be replaced with the averages
measured during an evaluation run.
"""

from __future__ import annotations

from pydantic import BaseModel

from .spec import AgentSpec

CHARS_PER_TOKEN = 4
SYSTEM_PROMPT_TOKENS = 250


class CostInputs(BaseModel):
    users: int = 25
    questions_per_user_per_day: float = 5
    working_days_per_month: int = 21
    avg_input_tokens: float = 0
    avg_output_tokens: float = 0
    price_per_million_input: float = 0.0
    price_per_million_output: float = 0.0
    currency: str = "CAD"
    measured: bool = False

    @classmethod
    def estimate_for(cls, spec: AgentSpec, **overrides) -> CostInputs:
        context_tokens = spec.retrieval.top_k * spec.retrieval.chunk_size / CHARS_PER_TOKEN
        inputs = cls(
            avg_input_tokens=round(SYSTEM_PROMPT_TOKENS + context_tokens + 40),
            avg_output_tokens=min(spec.provider.max_tokens, 300),
        )
        return inputs.model_copy(update=overrides)

    @property
    def questions_per_month(self) -> float:
        return self.users * self.questions_per_user_per_day * self.working_days_per_month

    @property
    def tokens_per_month(self) -> tuple[float, float]:
        q = self.questions_per_month
        return q * self.avg_input_tokens, q * self.avg_output_tokens

    @property
    def has_prices(self) -> bool:
        return self.price_per_million_input > 0 or self.price_per_million_output > 0

    @property
    def monthly_cost(self) -> float:
        tin, tout = self.tokens_per_month
        return tin / 1e6 * self.price_per_million_input + tout / 1e6 * self.price_per_million_output

    def to_markdown(self, agent_name: str) -> str:
        tin, tout = self.tokens_per_month
        basis = "measured during evaluation" if self.measured else "estimated from retrieval settings"
        lines = [
            f"# Cost model: {agent_name}",
            "",
            "| Input | Value |",
            "|---|---|",
            f"| Users | {self.users} |",
            f"| Questions per user per day | {self.questions_per_user_per_day:g} |",
            f"| Working days per month | {self.working_days_per_month} |",
            f"| Questions per month | {self.questions_per_month:,.0f} |",
            f"| Average input tokens per question ({basis}) | {self.avg_input_tokens:,.0f} |",
            f"| Average output tokens per question | {self.avg_output_tokens:,.0f} |",
            f"| Input tokens per month | {tin:,.0f} |",
            f"| Output tokens per month | {tout:,.0f} |",
        ]
        if self.has_prices:
            lines += [
                f"| Price per million input tokens | {self.price_per_million_input:g} {self.currency} |",
                f"| Price per million output tokens | {self.price_per_million_output:g} {self.currency} |",
                f"| **Estimated monthly model cost** | **{self.monthly_cost:,.2f} {self.currency}** |",
            ]
        else:
            lines += [
                "| Estimated monthly model cost | Enter your provider's per-million-token rates to calculate |",
            ]
        lines += [
            "",
            "Model cost only. Hosting, storage, monitoring and support time are not included.",
            "The largest lever is input tokens per question: lowering `top_k` or `chunk_size` cuts cost "
            "roughly in proportion, at some risk to accuracy. Re-run the evaluation after changing either.",
            "",
        ]
        return "\n".join(lines)
