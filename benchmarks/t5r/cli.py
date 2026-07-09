from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.t5r.budget import (
    BudgetExceeded,
    BudgetGuard,
    FundingRequired,
    require_funded_network_run,
)
from benchmarks.t5r.certification import CertifiedEngine, evidence_floor
from benchmarks.t5r.config import (
    BENCHMARK_VERSION,
    COMPONENTS,
    DEFAULT_ACTOR_MODEL,
    DEFAULT_ARMS,
    DEFAULT_EXTERNAL_DIR,
    DEFAULT_MAX_STEPS,
    DEFAULT_PROXY_MODEL,
    DEFAULT_SEED,
    DECLARED_MODULUS_L,
    FLEET_DELTA,
    K_MAX,
    OPTIONAL_ARMS,
    RUN_ROOT,
    SCHEMA_VERSION,
    THETA,
    W_TRIAL_CAP,
    Y_STAR,
)
from benchmarks.t5r.context import stable_hash
from benchmarks.t5r.ledger import EventRecord, append_jsonl, read_jsonl, write_event
from benchmarks.t5r.pricing import price_sheet_payload
from benchmarks.t5r.report import build_report
from benchmarks.t5r.source import SourceLock, ensure_tau2_source
from benchmarks.t5r.splits import TaskSplits, build_task_splits

TUNING_ARMS = (
    "GREEDY",
    "BON-2",
    "BON-4",
    "EPS-0.15",
    "EPS-0.35",
    "EPS-0.60",
    "RAW-m0.05-w4",
    "RAW-m0.05-w8",
    "RAW-m0.15-w4",
    "RAW-m0.15-w8",
    "CERT-b0.20",
    "CERT-b0.35",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _new_run_id(label: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{label}"


def _config_payload(*, stage: str, arms: tuple[str, ...], seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "stage": stage,
        "seed": seed,
        "arms": list(arms),
        "models": {
            "actor": DEFAULT_ACTOR_MODEL,
            "user": DEFAULT_ACTOR_MODEL,
            "proxy": DEFAULT_PROXY_MODEL,
        },
        "components": list(COMPONENTS),
        "certificate": {
            "theta": THETA,
            "y_star": Y_STAR,
            "fleet_delta": FLEET_DELTA,
            "epsilon": 0.35,
            "p_fresh": 1.0,
            "k_max": K_MAX,
            "n_floor": evidence_floor(),
            "w_trial_cap": W_TRIAL_CAP,
        },
        "t8": {
            "features": ["min(turn,8)/8", "mutation_seen"],
            "metric": "l1",
            "L": DECLARED_MODULUS_L,
            "claim": "declared_modulus_not_empirically_proven",
        },
    }


def _prepare_run(
    *,
    source_dir: Path,
    run_dir: Path,
    stage: str,
    arms: tuple[str, ...],
    seed: int,
    allow_clone: bool = True,
) -> tuple[SourceLock, TaskSplits, str, str]:
    source = ensure_tau2_source(source_dir, allow_clone=allow_clone)
    splits = build_task_splits(source_dir, seed=seed, source=source)
    config = _config_payload(stage=stage, arms=arms, seed=seed)
    sampler = {
        "seed": seed,
        "domains": {
            domain: {
                "development": split.development,
                "pilot": split.pilot,
                "adjudication_reserved": split.adjudication_reserved,
            }
            for domain, split in splits.domains.items()
        },
    }
    config_hash = stable_hash(config)
    sampler_hash = stable_hash(sampler)
    source.write(run_dir / "source_lock.json")
    splits.write(run_dir / "task_splits.json")
    _write_json(run_dir / "price_sheet.json", price_sheet_payload())
    _write_json(
        run_dir / "model_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "requested_models": {
                "actor": DEFAULT_ACTOR_MODEL,
                "user": DEFAULT_ACTOR_MODEL,
                "proxy": DEFAULT_PROXY_MODEL,
            },
            "resolved_model_policy": "record_every_successful_response",
            "snapshot_lock_required": True,
        },
    )
    _write_json(run_dir / "config_manifest.json", {**config, "config_hash": config_hash})
    _write_json(run_dir / "sampler_manifest.json", {**sampler, "sampler_hash": sampler_hash})
    _write_json(
        run_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source": asdict(source),
            "config_hash": config_hash,
            "sampler_hash": sampler_hash,
        },
    )
    return source, splits, config_hash, sampler_hash


def command_doctor(args: argparse.Namespace) -> int:
    source = ensure_tau2_source(args.source_dir, allow_clone=args.fetch_source)
    splits = build_task_splits(args.source_dir, seed=args.seed, source=source)
    from benchmarks.t5r.tau_runtime import activate_tau_source

    activate_tau_source(args.source_dir)
    from tau2.registry import registry  # type: ignore[import-not-found]
    from tau2.runner import run_simulation  # type: ignore[import-not-found]

    checks = {
        "source_commit": source.commit,
        "source_patch_sha256": source.patch_sha256,
        "official_runner": run_simulation.__name__ == "run_simulation",
        "domains": {
            domain: {
                "development": len(split.development),
                "pilot": len(split.pilot),
                "adjudication_reserved": len(split.adjudication_reserved),
            }
            for domain, split in splits.domains.items()
        },
        "tau_registry_has_required_domains": all(
            domain in registry.get_domains() for domain in ("mock", "airline", "retail")
        ),
        "n_floor": evidence_floor(),
        "network_model_calls_enabled": False,
    }
    checks["ok"] = (
        checks["official_runner"]
        and checks["tau_registry_has_required_domains"]
        and checks["n_floor"] == 4
        and all(item["pilot"] in {8, 16} for item in checks["domains"].values())
    )
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["ok"] else 1


def command_freeze(args: argparse.Namespace) -> int:
    run_dir = args.run_root / (args.run_id or _new_run_id("freeze"))
    arms = tuple(DEFAULT_ARMS) + (tuple(OPTIONAL_ARMS) if args.include_cert_ei else ())
    source, splits, config_hash, sampler_hash = _prepare_run(
        source_dir=args.source_dir,
        run_dir=run_dir,
        stage="preregistration",
        arms=arms,
        seed=args.seed,
        allow_clone=args.fetch_source,
    )
    registration = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_measured_tasks",
        "claim_scope": "pilot_evidence_only",
        "no_suite_shopping": True,
        "hidden_evaluation_available_to_proxy": False,
        "source": asdict(source),
        "config_hash": config_hash,
        "sampler_hash": sampler_hash,
        "development_tasks": {
            domain: split.development for domain, split in splits.domains.items()
        },
        "pilot_tasks": {domain: split.pilot for domain, split in splits.domains.items()},
        "adjudication_policy": {
            "requires_exploration_over_greedy": True,
            "minimum_median_exploration_tokens": 1000,
            "maximum_meter_error": 0.02,
            "maximum_remaining_credit_fraction": 0.80,
        },
    }
    _write_json(run_dir / "preregistration.json", registration)
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "stage": "preregistration",
            "status": "complete",
            "billable_calls": 0,
        },
    )
    build_report(run_dir, arms)
    print(run_dir.resolve())
    return 0


def _first_mock_task_id() -> str:
    from tau2.registry import registry  # type: ignore[import-not-found]

    loader = registry.get_tasks_loader("mock")
    try:
        tasks = loader(task_split_name="base")
    except (TypeError, ValueError):
        tasks = loader()
    if not tasks:
        raise ValueError("Tau mock domain has no tasks")
    return str(tasks[0].id)


def _pilot_task_plan(splits: TaskSplits) -> list[tuple[str, str]]:
    return [
        (domain, task_id)
        for domain in ("airline", "retail")
        for task_id in splits.domains[domain].pilot
    ]


def _smoke_task_plan(splits: TaskSplits) -> list[tuple[str, str]]:
    return [
        ("mock", _first_mock_task_id()),
        ("airline", splits.domains["airline"].development[0]),
        ("retail", splits.domains["retail"].development[0]),
    ]


def _family(name: str) -> str | None:
    for prefix in ("BON-", "EPS-", "RAW-", "CERT-"):
        if name.startswith(prefix):
            return prefix[:-1]
    return None


def _select_champions(tuning_run: Path) -> tuple[str, ...]:
    summary_path = tuning_run / "summary.json"
    if not summary_path.exists():
        build_report(tuning_run, TUNING_ARMS)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("completion_matrix", {}).get("complete"):
        raise ValueError("Tuning run is incomplete; champions cannot be selected")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for arm in summary.get("arms", []):
        family = _family(arm["arm"])
        if family is not None:
            grouped.setdefault(family, []).append(arm)
    winners = []
    for family in ("BON", "EPS", "RAW", "CERT"):
        options = grouped.get(family, [])
        if not options:
            raise ValueError(f"Tuning run has no completed {family} arm")
        winner = max(
            options,
            key=lambda item: (
                item.get("success_rate") if item.get("success_rate") is not None else -1,
                -float(item.get("billed_usd", math.inf)),
                item["arm"],
            ),
        )
        winners.append(winner["arm"])
    return ("GREEDY", *winners)


def _merge_tuning_trial(tuning_run: Path, run_dir: Path, arms: tuple[str, ...]) -> None:
    """Carry the selected tuning trial into the four-trial confirmation artifact."""
    for name in ("calls.jsonl", "decisions.jsonl", "events.jsonl", "simulations.jsonl"):
        for record in read_jsonl(tuning_run / name):
            if record.get("arm") not in arms or int(record.get("trial", 0)) != 0:
                continue
            record = dict(record)
            record["source_run_id"] = record.get("run_id")
            record["run_id"] = run_dir.name
            append_jsonl(run_dir / name, record)
    for arm in arms:
        source_state = tuning_run / "state" / f"{_safe_name(arm)}_trial-0.json"
        if source_state.exists():
            (run_dir / "state").mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                source_state,
                run_dir / "state" / f"{_safe_name(arm)}_trial-0.json",
            )


def _run_gross_cost(run_dir: Path) -> float:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(
            summary.get("meter_reconciliation", {}).get(
                "gross_price_sheet_cost_usd", 0.0
            )
        )
    return sum(
        float(record.get("billed_usd", 0.0))
        for record in read_jsonl(run_dir / "calls.jsonl")
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_engine_state(run_dir: Path, arm: str, trial: int, engine: CertifiedEngine) -> None:
    _write_json(
        run_dir / "state" / f"{_safe_name(arm)}_trial-{trial}.json",
        engine.snapshot(),
    )


def _validate_recent_calls(
    ledger_path: Path, *, arm: str, domain: str, task_id: str, trial: int
) -> None:
    calls = [
        item
        for item in read_jsonl(ledger_path)
        if item.get("arm") == arm
        and item.get("domain") == domain
        and str(item.get("task_id")) == str(task_id)
        and int(item.get("trial", 0)) == trial
    ]
    successful = [item for item in calls if item.get("status") == "success"]
    if not successful:
        raise RuntimeError("No successful provider calls were recorded")
    if any(int(item.get("total_tokens", 0)) <= 0 for item in successful):
        raise RuntimeError("A successful provider call is missing usage")
    if any(not item.get("resolved_model") for item in successful):
        raise RuntimeError("A successful provider call is missing its resolved model")
    physical_ids = [item["physical_call_id"] for item in successful]
    if len(physical_ids) != len(set(physical_ids)):
        raise RuntimeError("Duplicate physical call IDs indicate a replay mismatch")


def _run_task(
    *,
    run_dir: Path,
    run_id: str,
    source_dir: Path,
    arm: str,
    domain: str,
    task_id: str,
    trial: int,
    seed: int,
    max_steps: int,
    config_hash: str,
    sampler_hash: str,
    engine: CertifiedEngine,
    raw_state: Any,
    guard: BudgetGuard,
) -> None:
    from benchmarks.t5r.agent import GatedTauAgent
    from benchmarks.t5r.tau_runtime import (
        build_metered_user,
        find_base_task,
        run_official_text_simulation,
    )
    from tau2.orchestrator.orchestrator import Orchestrator  # type: ignore[import-not-found]

    environment_constructor, task = find_base_task(domain, task_id)
    environment = environment_constructor()
    ledger_path = run_dir / "calls.jsonl"
    user = build_metered_user(
        llm=DEFAULT_ACTOR_MODEL,
        instructions=str(task.user_scenario),
        tools=environment.get_user_tools(include=task.user_tools) or None,
        llm_args={"max_tokens": 2048},
        run_id=run_id,
        arm=arm,
        domain=domain,
        task_id=task_id,
        trial=trial,
        ledger_path=ledger_path,
        budget_guard=guard,
    )
    agent = GatedTauAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        run_id=run_id,
        arm=arm,
        domain=domain,
        task_id=task_id,
        trial=trial,
        ledger_path=ledger_path,
        events_path=run_dir / "events.jsonl",
        decisions_path=run_dir / "decisions.jsonl",
        certified_engine=engine,
        raw_run_state=raw_state,
        budget_guard=guard,
        config_hash=config_hash,
        sampler_hash=sampler_hash,
        llm=DEFAULT_ACTOR_MODEL,
        proxy_llm=DEFAULT_PROXY_MODEL,
        seed=seed,
    )
    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=10,
        seed=seed,
        simulation_id=f"{run_id}:{arm}:{domain}:{task_id}:trial-{trial}",
        validate_communication=True,
    )
    simulation = run_official_text_simulation(orchestrator)
    payload = simulation.model_dump(mode="json")
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "arm": arm,
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
        }
    )
    append_jsonl(run_dir / "simulations.jsonl", payload)
    _validate_recent_calls(
        ledger_path, arm=arm, domain=domain, task_id=task_id, trial=trial
    )


def _run_network_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    arms: tuple[str, ...],
    trials: range,
    cap_usd: float,
) -> int:
    guard = require_funded_network_run(cap_usd)
    run_dir = args.run_root / (args.run_id or _new_run_id(stage))
    source, splits, config_hash, sampler_hash = _prepare_run(
        source_dir=args.source_dir,
        run_dir=run_dir,
        stage=stage,
        arms=arms,
        seed=args.seed,
    )
    linked_tuning_run = getattr(args, "linked_tuning_run", None)
    if linked_tuning_run is not None:
        _merge_tuning_trial(linked_tuning_run, run_dir, arms)
    manifest_trials = list(trials)
    if linked_tuning_run is not None:
        manifest_trials.insert(0, 0)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "stage": stage,
        "status": "running",
        "funded_credit_cap_usd": cap_usd,
        "cash_spend_authorized_usd": 0.0,
        "source": asdict(source),
        "arms": list(arms),
        "trials": manifest_trials,
        "awarded_credits_usd": getattr(args, "awarded_credits_usd", None),
        "pilot_total_cap_usd": getattr(args, "pilot_total_cap_usd", cap_usd),
        "prior_tuning_cost_usd": getattr(args, "prior_tuning_cost_usd", 0.0),
        "linked_tuning_run": (
            str(linked_tuning_run.resolve()) if linked_tuning_run is not None else None
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)

    from benchmarks.t5r.tau_runtime import activate_tau_source

    activate_tau_source(args.source_dir)
    from benchmarks.t5r.agent import RawRunState
    task_plan = _smoke_task_plan(splits) if stage == "funded-smoke" else _pilot_task_plan(splits)
    horizon = max(len(task_plan) * args.max_steps * 2, 2)
    engines = {
        (arm, trial): CertifiedEngine(seed=args.seed + trial, horizon=horizon)
        for arm in arms
        for trial in trials
    }
    raw_states = {(arm, trial): RawRunState() for arm in arms for trial in trials}
    stop_on_error = stage == "funded-smoke"

    executed_arms: list[str] = []
    try:
        for arm in arms:
            if arm == "CERT-EI" and executed_arms:
                projected_arm_cost = guard.spent_usd / len(executed_arms)
                if guard.remaining_usd < projected_arm_cost:
                    manifest.setdefault("skipped_optional_arms", []).append(
                        {
                            "arm": arm,
                            "reason": "insufficient remaining funded credit for a full matched arm",
                            "remaining_usd": guard.remaining_usd,
                            "projected_arm_cost_usd": projected_arm_cost,
                        }
                    )
                    continue
            executed_arms.append(arm)
            for trial in trials:
                for task_index, (domain, task_id) in enumerate(task_plan):
                    try:
                        _run_task(
                            run_dir=run_dir,
                            run_id=run_dir.name,
                            source_dir=args.source_dir,
                            arm=arm,
                            domain=domain,
                            task_id=task_id,
                            trial=trial,
                            seed=args.seed + trial * 10_000 + task_index,
                            max_steps=args.max_steps,
                            config_hash=config_hash,
                            sampler_hash=sampler_hash,
                            engine=engines[(arm, trial)],
                            raw_state=raw_states[(arm, trial)],
                            guard=guard,
                        )
                    except Exception as exc:
                        write_event(
                            run_dir / "events.jsonl",
                            EventRecord(
                                run_id=run_dir.name,
                                arm=arm,
                                domain=domain,
                                task_id=task_id,
                                trial=trial,
                                decision_idx=None,
                                component=None,
                                context_key=None,
                                event="TaskFailure",
                                detail=repr(exc),
                            ),
                        )
                        append_jsonl(
                            run_dir / "simulations.jsonl",
                            {
                                "schema_version": SCHEMA_VERSION,
                                "run_id": run_dir.name,
                                "arm": arm,
                                "domain": domain,
                                "task_id": task_id,
                                "trial": trial,
                                "termination_reason": "exception",
                                "reward_info": None,
                                "error": repr(exc),
                            },
                        )
                        if stop_on_error:
                            raise
                    finally:
                        _write_engine_state(
                            run_dir, arm, trial, engines[(arm, trial)]
                        )
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "stopped_on_mismatch"
        raise
    finally:
        manifest["arms"] = executed_arms
        manifest["stage_gross_price_sheet_cost_usd"] = guard.spent_usd
        manifest["gross_price_sheet_cost_usd"] = (
            guard.spent_usd + float(manifest.get("prior_tuning_cost_usd", 0.0))
        )
        _write_json(run_dir / "manifest.json", manifest)
        build_report(run_dir, tuple(executed_arms))
    print(run_dir.resolve())
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    cap = min(float(args.credit_budget_usd), 10.0)
    return _run_network_stage(
        args,
        stage="funded-smoke",
        arms=("CERT",),
        trials=range(1),
        cap_usd=cap,
    )


def command_pilot(args: argparse.Namespace) -> int:
    allowed_cap = min(100.0, 0.20 * float(args.awarded_credits_usd))
    args.pilot_total_cap_usd = allowed_cap
    args.prior_tuning_cost_usd = 0.0
    args.linked_tuning_run = None
    if args.stage == "tune":
        cap = min(float(args.credit_budget_usd), allowed_cap)
        arms = TUNING_ARMS + (("CERT-EI",) if args.include_cert_ei else ())
        trials = range(1)
        stage = "funded-pilot-tuning"
    else:
        if args.tuning_run is None:
            raise ValueError("--tuning-run is required for the confirm stage")
        arms = _select_champions(args.tuning_run)
        args.linked_tuning_run = args.tuning_run
        args.prior_tuning_cost_usd = _run_gross_cost(args.tuning_run)
        remaining_total_cap = allowed_cap - args.prior_tuning_cost_usd
        if remaining_total_cap <= 0.0:
            raise ValueError("Tuning spend exhausted the predeclared pilot credit cap")
        cap = min(float(args.credit_budget_usd), remaining_total_cap)
        trials = range(1, 4)
        stage = "funded-pilot-confirmation"
    return _run_network_stage(
        args,
        stage=stage,
        arms=tuple(arms),
        trials=trials,
        cap_usd=cap,
    )


def command_report(args: argparse.Namespace) -> int:
    from benchmarks.t5r.artifacts import generate_trace_site

    if args.provider_billed_usd is not None:
        _write_json(
            args.run_dir / "meter_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "provider_billed_usd": float(args.provider_billed_usd),
                "source": args.meter_source,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    manifest_path = args.run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    arms = tuple(manifest.get("arms") or DEFAULT_ARMS)
    payload = build_report(args.run_dir, arms)
    generate_trace_site(args.run_dir, args.run_dir / "site")
    print(json.dumps(payload["publication_gate"], indent=2, sort_keys=True))
    return 0


def command_export_tau(args: argparse.Namespace) -> int:
    simulations = read_jsonl(args.run_dir / "simulations.jsonl")
    export_dir = args.output_dir or (args.run_dir / "tau_submission")
    export_dir.mkdir(parents=True, exist_ok=True)
    for index, simulation in enumerate(simulations):
        name = "{domain}_{task}_trial-{trial}_{arm}_{index}.json".format(
            domain=_safe_name(str(simulation.get("domain", "unknown"))),
            task=_safe_name(str(simulation.get("task_id", "unknown"))),
            trial=simulation.get("trial", 0),
            arm=_safe_name(str(simulation.get("arm", "unknown"))),
            index=index,
        )
        _write_json(export_dir / "trajectories" / name, simulation)
    source_lock = args.run_dir / "source_lock.json"
    if source_lock.exists():
        shutil.copy2(source_lock, export_dir / "source_lock.json")
    _write_json(
        export_dir / "submission_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "entry_type": "modified_scaffold_custom_agent",
            "standard_verified_submission": False,
            "trajectory_count": len(simulations),
            "disclosure": (
                "Verdict Bench changes the agent scaffold and must be evaluated as a "
                "custom modified-scaffold entry under Tau submission rules."
            ),
        },
    )
    print(export_dir.resolve())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verdict-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_source_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument("--fetch-source", action="store_true")

    doctor = subparsers.add_parser("doctor", help="verify source, math, and Tau imports")
    add_source_options(doctor)
    doctor.set_defaults(handler=command_doctor)

    freeze = subparsers.add_parser("freeze", help="write the no-cost preregistration")
    add_source_options(freeze)
    freeze.add_argument("--run-root", type=Path, default=RUN_ROOT)
    freeze.add_argument("--run-id")
    freeze.add_argument("--include-cert-ei", action="store_true")
    freeze.set_defaults(handler=command_freeze)

    def add_network_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
        command.add_argument("--run-root", type=Path, default=RUN_ROOT)
        command.add_argument("--run-id")
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
        command.add_argument("--credit-budget-usd", type=float, required=True)

    smoke = subparsers.add_parser("smoke", help="run Tau mock and development tasks")
    add_network_options(smoke)
    smoke.set_defaults(handler=command_smoke)

    pilot = subparsers.add_parser("pilot", help="run funded tuning or confirmation")
    add_network_options(pilot)
    pilot.add_argument("--awarded-credits-usd", type=float, required=True)
    pilot.add_argument("--stage", choices=("tune", "confirm"), default="tune")
    pilot.add_argument("--tuning-run", type=Path)
    pilot.add_argument("--include-cert-ei", action="store_true")
    pilot.set_defaults(handler=command_pilot)

    report = subparsers.add_parser("report", help="rebuild a run report")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--provider-billed-usd", type=float)
    report.add_argument(
        "--meter-source",
        default="OpenAI project billing export",
        help="description of the provider billing evidence",
    )
    report.set_defaults(handler=command_report)

    export_tau = subparsers.add_parser("export-tau", help="package custom Tau trajectories")
    export_tau.add_argument("run_dir", type=Path)
    export_tau.add_argument("--output-dir", type=Path)
    export_tau.set_defaults(handler=command_export_tau)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (BudgetExceeded, FundingRequired) as exc:
        print(f"verdict-bench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
