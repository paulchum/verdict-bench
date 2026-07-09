from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.t5r.budget import BudgetGuard
from benchmarks.t5r.ledger import make_call_record, usage_from_message, write_call


def activate_tau_source(source_dir: Path) -> Path:
    """Expose the pinned Tau source tree without replacing any Tau modules."""
    source_root = (source_dir / "src").resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Missing Tau source package: {source_root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return source_root


def find_base_task(domain: str, task_id: str) -> tuple[Any, Any]:
    """Load a base task and environment through Tau's official registry."""
    from tau2.registry import registry  # type: ignore[import-not-found]

    environment_constructor = registry.get_env_constructor(domain)
    tasks_loader = registry.get_tasks_loader(domain)
    tasks = [
        task
        for task in tasks_loader(task_split_name="base")
        if str(task.id) == str(task_id)
    ]
    if len(tasks) != 1:
        raise ValueError(f"Expected one base task for {domain}:{task_id}, got {len(tasks)}")
    return environment_constructor, tasks[0]


def build_metered_user(
    *,
    llm: str,
    instructions: str,
    tools: list[Any] | None,
    llm_args: dict[str, Any],
    run_id: str,
    arm: str,
    domain: str,
    task_id: str,
    trial: int,
    ledger_path: Path,
    budget_guard: BudgetGuard,
) -> Any:
    """Construct the official Tau user simulator with usage metering."""
    from tau2.user.user_simulator import UserSimulator  # type: ignore[import-not-found]

    class MeteredUserSimulator(UserSimulator):
        def _generate_next_message(self, message, state):
            budget_guard.check(0.50)
            started = time.monotonic()
            try:
                user_message = super()._generate_next_message(message, state)
            except Exception as exc:
                write_call(
                    ledger_path,
                    make_call_record(
                        run_id=run_id,
                        arm=arm,
                        domain=domain,
                        task_id=task_id,
                        trial=trial,
                        decision_idx=sum(
                            1 for item in state.messages if item.role == "user"
                        ),
                        component=None,
                        context_key=None,
                        purpose="user_simulator",
                        model=llm,
                        usage=None,
                        raw_data=None,
                        provider_cost_usd=None,
                        status="error",
                        latency_seconds=time.monotonic() - started,
                        error=repr(exc),
                    ),
                )
                raise

            usage, raw_data, provider_cost = usage_from_message(user_message)
            record = make_call_record(
                run_id=run_id,
                arm=arm,
                domain=domain,
                task_id=task_id,
                trial=trial,
                decision_idx=sum(1 for item in state.messages if item.role == "user"),
                component=None,
                context_key=None,
                purpose="user_simulator",
                model=llm,
                usage=usage,
                raw_data=raw_data,
                provider_cost_usd=provider_cost,
                latency_seconds=time.monotonic() - started,
            )
            budget_guard.debit(record.billed_usd)
            write_call(ledger_path, record)
            return user_message

    return MeteredUserSimulator(
        llm=llm,
        instructions=instructions,
        tools=tools,
        llm_args=llm_args,
    )


def run_official_text_simulation(orchestrator: Any) -> Any:
    """Execute and adjudicate through Tau's official runner."""
    from tau2.runner import run_simulation  # type: ignore[import-not-found]

    return run_simulation(orchestrator)
