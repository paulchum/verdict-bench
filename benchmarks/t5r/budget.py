from __future__ import annotations

import os
from dataclasses import dataclass


class FundingRequired(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    limit_usd: float
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.limit_usd <= 0.0:
            raise FundingRequired("A positive funded credit budget is required.")

    @property
    def remaining_usd(self) -> float:
        return max(self.limit_usd - self.spent_usd, 0.0)

    def check(self, estimated_next_call_usd: float = 0.0) -> None:
        projected = self.spent_usd + max(float(estimated_next_call_usd), 0.0)
        if projected > self.limit_usd + 1e-12:
            raise BudgetExceeded(
                f"Funded credit cap would be exceeded: ${projected:.6f} > ${self.limit_usd:.6f}"
            )

    def debit(self, cost_usd: float) -> None:
        cost = max(float(cost_usd), 0.0)
        self.check(cost)
        self.spent_usd += cost


def require_funded_network_run(credit_budget_usd: float) -> BudgetGuard:
    if os.environ.get("VERDICT_BENCH_FUNDED") != "1":
        raise FundingRequired(
            "Network runs are disabled. Set VERDICT_BENCH_FUNDED=1 only after credits are confirmed."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise FundingRequired("OPENAI_API_KEY is not set for the funded project.")
    return BudgetGuard(credit_budget_usd)
