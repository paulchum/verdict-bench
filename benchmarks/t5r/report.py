from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.t5r.config import DEFAULT_ARMS, SCHEMA_VERSION
from benchmarks.t5r.ledger import read_jsonl, summarize_calls
from benchmarks.t5r.pricing import price_sheet_payload
from benchmarks.t5r.stats import brier_score, paired_bootstrap_mean, wilson_interval

MATCHED_QUALITY_MARGIN = 0.05


def canonicalize_json_numbers(value: Any) -> Any:
    """Stabilize published floats across supported Python runtimes."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("published artifacts cannot contain non-finite numbers")
        normalized = round(value, 12)
        return 0.0 if normalized == 0 else normalized
    if isinstance(value, dict):
        return {key: canonicalize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_json_numbers(item) for item in value]
    if isinstance(value, tuple):
        return [canonicalize_json_numbers(item) for item in value]
    return value


@dataclass(frozen=True)
class ArmReport:
    arm: str
    task_results: int
    successes: int
    success_rate: float | None
    success_ci95_low: float | None
    success_ci95_high: float | None
    average_reward: float | None
    gross_price_sheet_cost_usd: float
    exploration_cost_usd: float
    proxy_cost_usd: float
    mean_cost_per_task_usd: float | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    exploration_tokens: int
    exploration_share: float
    median_exploration_rollout_tokens: float | None
    proxy_tokens: int
    refusals: int
    retirements: int
    task_failures: int
    missing_usage_calls: int
    missing_resolved_models: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _reward(simulation: dict[str, Any]) -> float | None:
    reward = (simulation.get("reward_info") or {}).get("reward")
    return float(reward) if reward is not None else None


def _cell(item: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(item.get("domain", "")),
        str(item.get("task_id", "")),
        int(item.get("trial", 0)),
    )


def _unique_simulations(items: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = defaultdict(dict)
    for item in items:
        result[str(item.get("arm"))][_cell(item)] = item
    return result


def _group_by_arm(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("arm"))].append(item)
    return grouped


def _event_counts(events: list[dict[str, Any]]) -> tuple[int, int]:
    refusal_names = {
        "NotSeparated",
        "Inhomogeneous",
        "ModulusTooCoarse",
        "EvidenceCensored",
    }
    refusals = sum(1 for event in events if event.get("event") in refusal_names)
    retirements = sum(
        1 for event in events if "retir" in str(event.get("event", "")).lower()
    )
    return refusals, retirements


def _engine_status(run_dir: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    declared_moduli: list[dict[str, Any]] = []
    for path in sorted((run_dir / "state").glob("*.json")):
        payload = _read_json(path)
        declared = payload.get("declared_modulus")
        if declared:
            declared_moduli.append(declared)
        for key in (payload.get("keys") or {}).values():
            counts[str(key.get("status", "Unknown"))] += 1
    return {
        "declared_modulus": (
            declared_moduli[0]
            if declared_moduli
            else {
                "features": ["min(turn,8)/8", "mutation_seen"],
                "metric": "l1",
                "L": 1.0,
                "status": "declared_not_empirically_proven",
            }
        ),
        "key_status_counts": dict(sorted(counts.items())),
        "licensed_keys": counts.get("Active", 0) + counts.get("RetiredContextKey", 0),
        "refused_keys": (
            counts.get("Inhomogeneous", 0)
            + counts.get("ModulusTooCoarse", 0)
            + counts.get("EvidenceCensored", 0)
        ),
    }


def _proxy_calibration(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    outcomes: list[int] = []
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for decision in decisions:
        candidate_scores = decision.get("candidate_scores") or {}
        bernoulli_outcomes = decision.get("bernoulli_outcomes") or {}
        for component, outcome in bernoulli_outcomes.items():
            if component not in candidate_scores:
                continue
            score = float(candidate_scores[component])
            binary = int(outcome)
            scores.append(score)
            outcomes.append(binary)
            bins[min(int(score * 5), 4)].append((score, binary))
    return {
        "kind": "seeded_bernoulli_audit_calibration",
        "note": (
            "This checks the declared Bernoulliization mechanism, not calibration to "
            "hidden Tau criteria."
        ),
        "observations": len(scores),
        "brier_score": brier_score(scores, outcomes),
        "bins": [
            {
                "low": index / 5.0,
                "high": (index + 1) / 5.0,
                "count": len(bins[index]),
                "mean_score": (
                    statistics.mean(score for score, _ in bins[index])
                    if bins[index]
                    else None
                ),
                "outcome_rate": (
                    statistics.mean(outcome for _, outcome in bins[index])
                    if bins[index]
                    else None
                ),
            }
            for index in range(5)
        ],
    }


def _completion_matrix(
    manifest: dict[str, Any],
    splits: dict[str, Any],
    arms: tuple[str, ...],
    simulations: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
) -> dict[str, Any]:
    stage = str(manifest.get("stage", ""))
    trials = [int(value) for value in manifest.get("trials", [])]
    if "pilot" in stage:
        tasks = [
            (domain, str(task_id))
            for domain, split in (splits.get("domains") or {}).items()
            for task_id in split.get("pilot", [])
        ]
    elif stage == "funded-smoke":
        tasks = [
            (domain, str(split.get("development", [""])[0]))
            for domain, split in (splits.get("domains") or {}).items()
        ]
        tasks.insert(0, ("mock", "*"))
    else:
        tasks = []
    expected_count = len(tasks) * len(trials)
    rows = []
    for arm in arms:
        actual = simulations.get(arm, {})
        if stage == "funded-smoke":
            completed = len(actual)
            missing = max(expected_count - completed, 0)
        else:
            expected = {
                (domain, task_id, trial)
                for domain, task_id in tasks
                for trial in trials
            }
            completed = len(set(actual) & expected)
            missing = len(expected - set(actual))
        rows.append(
            {
                "arm": arm,
                "expected": expected_count,
                "completed": completed,
                "missing": missing,
                "complete": expected_count > 0 and missing == 0,
            }
        )
    return {
        "expected_tasks_per_trial": len(tasks),
        "trials": trials,
        "rows": rows,
        "complete": bool(rows) and all(row["complete"] for row in rows),
    }


def _meter_reconciliation(
    calls: list[dict[str, Any]],
    *,
    dashboard: dict[str, Any],
    synthetic: bool,
) -> dict[str, Any]:
    successful = [call for call in calls if call.get("status", "success") == "success"]
    gross = sum(float(call.get("billed_usd", 0.0)) for call in successful)
    priced = [call for call in successful if call.get("provider_cost_usd") is not None]
    provider = sum(float(call["provider_cost_usd"]) for call in priced)
    coverage = len(priced) / len(successful) if successful else 0.0
    missing_usage = sum(int(call.get("total_tokens", 0)) <= 0 for call in successful)
    missing_models = sum(not call.get("resolved_model") for call in successful)
    call_level_relative_error = (
        abs(provider - gross) / max(provider, gross, 1e-12)
        if successful and coverage == 1.0
        else None
    )
    dashboard_cost = dashboard.get("provider_billed_usd")
    dashboard_relative_error = (
        abs(float(dashboard_cost) - gross) / max(float(dashboard_cost), gross, 1e-12)
        if dashboard_cost is not None
        else None
    )
    if not successful:
        status = "not_applicable"
    elif missing_usage or missing_models:
        status = "failed_call_schema"
    elif synthetic and coverage == 1.0:
        status = "[SIM]_reconciled_within_2pct"
    elif dashboard_cost is None:
        status = "pending_provider_dashboard"
    elif dashboard_relative_error is not None and dashboard_relative_error <= 0.02:
        status = "reconciled_within_2pct"
    else:
        status = "outside_2pct"
    return {
        "status": status,
        "gross_price_sheet_cost_usd": gross,
        "provider_reported_cost_usd": provider if priced else None,
        "provider_cost_coverage": coverage,
        "call_level_relative_error": call_level_relative_error,
        "provider_dashboard_cost_usd": dashboard_cost,
        "dashboard_relative_error": dashboard_relative_error,
        "dashboard_evidence": dashboard or None,
        "missing_usage_calls": missing_usage,
        "missing_resolved_models": missing_models,
    }


def _task_costs(calls: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str, int], float]]:
    result: dict[str, dict[tuple[str, str, int], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for call in calls:
        result[str(call.get("arm"))][_cell(call)] += float(call.get("billed_usd", 0.0))
    return result


def _paired_savings(
    arm_reports: list[ArmReport],
    simulations: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    costs: dict[str, dict[tuple[str, str, int], float]],
) -> dict[str, Any]:
    available = [report for report in arm_reports if report.success_rate is not None]
    certs = [report for report in available if report.arm == "CERT" or report.arm.startswith("CERT-")]
    if not certs:
        return {"target_arm": None, "comparisons": []}
    target = max(certs, key=lambda item: (item.success_rate, -item.gross_price_sheet_cost_usd))
    comparisons = []
    for baseline in available:
        if baseline.arm == target.arm:
            continue
        common = sorted(set(simulations[target.arm]) & set(simulations[baseline.arm]))
        cost_differences = [
            costs[baseline.arm].get(key, 0.0) - costs[target.arm].get(key, 0.0)
            for key in common
        ]
        quality_differences = [
            float((_reward(simulations[target.arm][key]) or 0.0) >= 0.999)
            - float((_reward(simulations[baseline.arm][key]) or 0.0) >= 0.999)
            for key in common
        ]
        comparisons.append(
            {
                "baseline_arm": baseline.arm,
                "positive_savings_means_cert_is_cheaper": True,
                "cost_savings_usd": paired_bootstrap_mean(cost_differences),
                "success_delta": paired_bootstrap_mean(quality_differences),
            }
        )
    return {"target_arm": target.arm, "comparisons": comparisons}


def _matched_quality(arm_reports: list[ArmReport]) -> dict[str, Any]:
    measured = [report for report in arm_reports if report.success_rate is not None]
    certs = [report for report in measured if report.arm == "CERT" or report.arm.startswith("CERT-")]
    if not certs:
        return {
            "target_arm": None,
            "margin": MATCHED_QUALITY_MARGIN,
            "cheapest_baseline": None,
            "matched_baselines": [],
        }
    target = max(certs, key=lambda item: (item.success_rate, -item.gross_price_sheet_cost_usd))
    matched = [
        report
        for report in measured
        if report.arm != target.arm
        and report.success_rate >= target.success_rate - MATCHED_QUALITY_MARGIN
    ]
    cheapest = min(
        matched,
        key=lambda item: (
            item.mean_cost_per_task_usd
            if item.mean_cost_per_task_usd is not None
            else float("inf")
        ),
        default=None,
    )
    return {
        "target_arm": target.arm,
        "target_success_rate": target.success_rate,
        "margin": MATCHED_QUALITY_MARGIN,
        "rule": "baseline_success >= target_success - margin",
        "matched_baselines": [report.arm for report in matched],
        "cheapest_baseline": cheapest.arm if cheapest else None,
        "cheapest_baseline_cost_per_task_usd": (
            cheapest.mean_cost_per_task_usd if cheapest else None
        ),
    }


def build_report(run_dir: Path, arms: tuple[str, ...] = DEFAULT_ARMS) -> dict[str, Any]:
    calls = read_jsonl(run_dir / "calls.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    decisions = read_jsonl(run_dir / "decisions.jsonl")
    simulation_rows = read_jsonl(run_dir / "simulations.jsonl")
    manifest = _read_json(run_dir / "manifest.json")
    splits = _read_json(run_dir / "task_splits.json")
    meter_manifest = _read_json(run_dir / "meter_manifest.json")

    calls_by_arm = _group_by_arm(calls)
    events_by_arm = _group_by_arm(events)
    simulations = _unique_simulations(simulation_rows)
    arm_reports: list[ArmReport] = []
    for arm in arms:
        totals = summarize_calls(calls, arm)
        arm_calls = calls_by_arm.get(arm, [])
        arm_simulations = list(simulations.get(arm, {}).values())
        rewards = [_reward(simulation) for simulation in arm_simulations]
        successes = sum(reward is not None and reward >= 0.999 for reward in rewards)
        interval = wilson_interval(successes, len(arm_simulations))
        known_rewards = [reward for reward in rewards if reward is not None]
        rollout_tokens = [
            int(call.get("total_tokens", 0))
            for call in arm_calls
            if call.get("purpose") == "exploration_rollout"
        ]
        refusals, retirements = _event_counts(events_by_arm.get(arm, []))
        successful_calls = [
            call for call in arm_calls if call.get("status", "success") == "success"
        ]
        exploration_cost = sum(
            float(call.get("billed_usd", 0.0))
            for call in successful_calls
            if call.get("purpose") == "exploration_rollout"
        )
        proxy_cost = sum(
            float(call.get("billed_usd", 0.0))
            for call in successful_calls
            if call.get("purpose") == "proxy_score"
        )
        arm_reports.append(
            ArmReport(
                arm=arm,
                task_results=len(arm_simulations),
                successes=successes,
                success_rate=(successes / len(arm_simulations) if arm_simulations else None),
                success_ci95_low=interval.low if interval else None,
                success_ci95_high=interval.high if interval else None,
                average_reward=(statistics.mean(known_rewards) if known_rewards else None),
                gross_price_sheet_cost_usd=totals.billed_usd,
                exploration_cost_usd=exploration_cost,
                proxy_cost_usd=proxy_cost,
                mean_cost_per_task_usd=(
                    totals.billed_usd / len(arm_simulations) if arm_simulations else None
                ),
                input_tokens=totals.input_tokens,
                cached_input_tokens=totals.cached_input_tokens,
                output_tokens=totals.output_tokens,
                reasoning_tokens=totals.reasoning_tokens,
                total_tokens=totals.total_tokens,
                exploration_tokens=totals.exploration_tokens,
                exploration_share=totals.exploration_share,
                median_exploration_rollout_tokens=(
                    statistics.median(rollout_tokens) if rollout_tokens else None
                ),
                proxy_tokens=totals.proxy_tokens,
                refusals=refusals,
                retirements=retirements,
                task_failures=sum(reward is None for reward in rewards),
                missing_usage_calls=sum(
                    int(call.get("total_tokens", 0)) <= 0 for call in successful_calls
                ),
                missing_resolved_models=sum(
                    not call.get("resolved_model") for call in successful_calls
                ),
            )
        )

    greedy = next((report for report in arm_reports if report.arm == "GREEDY"), None)
    exploring = [
        report
        for report in arm_reports
        if report.arm != "GREEDY" and report.success_rate is not None
    ]
    best_exploring = max(exploring, key=lambda item: item.success_rate, default=None)
    delta = (
        best_exploring.success_rate - greedy.success_rate
        if best_exploring is not None
        and greedy is not None
        and greedy.success_rate is not None
        else None
    )
    exploration_rollouts = [
        int(call.get("total_tokens", 0))
        for call in calls
        if call.get("purpose") == "exploration_rollout"
    ]
    median_rollout = (
        statistics.median(exploration_rollouts) if exploration_rollouts else None
    )
    completion = _completion_matrix(manifest, splits, arms, simulations)
    stage = str(manifest.get("stage", ""))
    synthetic = stage.startswith("simulation")
    meter = _meter_reconciliation(
        calls, dashboard=meter_manifest, synthetic=synthetic
    )
    matched = _matched_quality(arm_reports)
    flags = []
    if median_rollout is not None and median_rollout < 1000:
        flags.append("MaterialRolloutCostBelow1k")
    if delta is not None and delta <= 0:
        flags.append("ExplorationDidNotBeatGreedy")
    if (
        greedy is not None
        and best_exploring is not None
        and greedy.success_rate is not None
        and greedy.success_rate >= best_exploring.success_rate - MATCHED_QUALITY_MARGIN
    ):
        flags.append("GreedyInsideMatchedQualityBand")

    reserved_tasks = sum(
        len(split.get("adjudication_reserved", []))
        for split in (splits.get("domains") or {}).values()
    )
    completed_results = sum(report.task_results for report in arm_reports)
    gross_cost = meter["gross_price_sheet_cost_usd"]
    projected_adjudication_cost = (
        gross_cost / completed_results * reserved_tasks * 4 * len(arm_reports)
        if completed_results and arm_reports
        else None
    )
    awarded_credits = manifest.get("awarded_credits_usd")
    remaining_credits = (
        max(float(awarded_credits) - gross_cost, 0.0)
        if awarded_credits is not None
        else None
    )
    projection_fits = (
        projected_adjudication_cost is not None
        and remaining_credits is not None
        and projected_adjudication_cost <= 0.80 * remaining_credits
    )
    trials = {int(value) for value in manifest.get("trials", [])}
    gates = {
        "exploration_beats_greedy": delta is not None and delta > 0,
        "median_exploration_rollout_at_least_1000_tokens": (
            median_rollout is not None and median_rollout >= 1000
        ),
        "meter_reconciled_within_2pct": meter["status"] == "reconciled_within_2pct",
        "completion_matrix_complete": completion["complete"],
        "four_trials_completed": {0, 1, 2, 3}.issubset(trials),
        "projected_adjudication_within_80pct_remaining_credits": projection_fits,
    }
    measured = bool(simulation_rows) and not synthetic
    publication_gate = {
        "status": (
            "eligible_to_freeze_adjudication"
            if measured and all(gates.values())
            else "pilot_evidence_only_do_not_adjudicate"
        ),
        "checks": gates,
        "k_r1_outcome": None,
        "reason": "Only frozen adjudication may issue K-R1 PASS, FAIL, or VOID.",
        "credit_projection": {
            "awarded_credits_usd": awarded_credits,
            "remaining_credits_usd": remaining_credits,
            "reserved_tasks": reserved_tasks,
            "projected_adjudication_cost_usd": projected_adjudication_cost,
            "maximum_allowed_fraction_of_remaining": 0.80,
        },
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "[SIM] synthetic_fixture_not_empirical"
            if synthetic
            else "pilot_evidence_only_not_k_r1"
        ),
        "price_sheet": price_sheet_payload(),
        "arms": [asdict(report) for report in arm_reports],
        "workload_validity": {
            "greedy_success_rate": greedy.success_rate if greedy else None,
            "best_exploring_arm": best_exploring.arm if best_exploring else None,
            "best_exploring_success_rate": (
                best_exploring.success_rate if best_exploring else None
            ),
            "greedy_vs_exploring_delta": delta,
            "median_exploration_rollout_tokens": median_rollout,
            "matched_quality": matched,
            "pilot_flags": flags,
        },
        "paired_bootstrap": _paired_savings(
            arm_reports, simulations, _task_costs(calls)
        ),
        "proxy_calibration": _proxy_calibration(decisions),
        "t8": _engine_status(run_dir),
        "completion_matrix": completion,
        "meter_reconciliation": meter,
        "publication_gate": publication_gate,
        "artifact_counts": {
            "calls": len(calls),
            "decisions": len(decisions),
            "events": len(events),
            "simulations": len(simulation_rows),
        },
    }
    payload = canonicalize_json_numbers(payload)
    _write_report_files(run_dir, payload)
    return payload


def _write_report_files(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    workload = payload["workload_validity"]
    lines = [
        "# T5-R Tau3 Evidence Report",
        "",
        f"**Status:** {payload['status']}",
        "",
        "This is not a K-R1 adjudication. Only a separately frozen adjudication can issue PASS, FAIL, or VOID.",
        "",
        "## Workload Validity",
        "",
        f"- GREEDY success: {_fmt(workload['greedy_success_rate'])}",
        f"- Best exploring arm: {_fmt(workload['best_exploring_arm'])}",
        f"- GREEDY-to-exploring delta: {_fmt(workload['greedy_vs_exploring_delta'])}",
        f"- Median exploration rollout tokens: {_fmt(workload['median_exploration_rollout_tokens'])}",
        f"- Flags: {', '.join(workload['pilot_flags']) or 'none'}",
        "",
        "## Arm Results",
        "",
        "| Arm | Results | Success (95% CI) | Gross USD | Explore USD | Tokens | Explore share | Refusals | Retirements | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        interval = (
            f"{_fmt(arm['success_rate'])} "
            f"[{_fmt(arm['success_ci95_low'])}, {_fmt(arm['success_ci95_high'])}]"
        )
        lines.append(
            "| {arm} | {results} | {success} | {gross} | {explore} | {tokens} | {share} | {refusals} | {retirements} | {failures} |".format(
                arm=arm["arm"],
                results=arm["task_results"],
                success=interval,
                gross=_fmt(arm["gross_price_sheet_cost_usd"]),
                explore=_fmt(arm["exploration_cost_usd"]),
                tokens=arm["total_tokens"],
                share=_fmt(arm["exploration_share"]),
                refusals=arm["refusals"],
                retirements=arm["retirements"],
                failures=arm["task_failures"],
            )
        )
    lines.extend(
        [
            "",
            "## Evidence Health",
            "",
            f"- Meter: {payload['meter_reconciliation']['status']}",
            f"- Completion: {_fmt(payload['completion_matrix']['complete'])}",
            f"- T8 refused keys: {payload['t8']['refused_keys']}",
            f"- Proxy audit observations: {payload['proxy_calibration']['observations']}",
            f"- Publication gate: {payload['publication_gate']['status']}",
            "",
        ]
    )
    return "\n".join(lines)
