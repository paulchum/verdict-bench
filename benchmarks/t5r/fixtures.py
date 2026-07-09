from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from benchmarks.t5r.artifacts import generate_trace_site
from benchmarks.t5r.config import (
    DEFAULT_ACTOR_MODEL,
    DEFAULT_ARMS,
    DEFAULT_PROXY_MODEL,
    DEFAULT_SEED,
    DEFAULT_EXTERNAL_DIR,
    SCHEMA_VERSION,
)
from benchmarks.t5r.context import stable_hash
from benchmarks.t5r.ledger import append_jsonl
from benchmarks.t5r.pricing import Usage, billed_usd, price_sheet_payload
from benchmarks.t5r.report import build_report
from benchmarks.t5r.source import ensure_tau2_source
from benchmarks.t5r.splits import build_task_splits


SIM_SUCCESS_COUNTS = {
    "GREEDY": 13,
    "BON-2": 14,
    "BON-4": 15,
    "EPS-0.35": 14,
    "RAW": 15,
    "CERT": 16,
}


def generate_simulated_release(source_dir: Path, output_dir: Path) -> Path:
    """Generate deterministic non-empirical artifacts for UI and CI testing."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = ensure_tau2_source(source_dir, allow_clone=False)
    splits = build_task_splits(source_dir, seed=DEFAULT_SEED, source=source)
    tasks = [
        (domain, task_id)
        for domain in ("airline", "retail")
        for task_id in splits.domains[domain].pilot
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "sim-v0.1.0",
        "stage": "simulation-fixture",
        "status": "complete",
        "label": "[SIM]",
        "arms": list(DEFAULT_ARMS),
        "trials": [0],
        "billable_calls": 0,
        "cash_spend_authorized_usd": 0.0,
    }
    _write_json(output_dir / "manifest.json", manifest)
    source.write(output_dir / "source_lock.json")
    splits.write(output_dir / "task_splits.json")
    _write_json(output_dir / "price_sheet.json", price_sheet_payload())
    _write_json(
        output_dir / "model_manifest.json",
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

    rng = random.Random(DEFAULT_SEED)
    for arm_index, arm in enumerate(DEFAULT_ARMS):
        success_count = SIM_SUCCESS_COUNTS[arm]
        successful_indices = set(
            random.Random(f"{DEFAULT_SEED}:{arm}").sample(
                range(len(tasks)), success_count
            )
        )
        for task_index, (domain, task_id) in enumerate(tasks):
            decision_idx = task_index % 7
            component = (
                "policy_first",
                "tool_progress",
                "verify_then_commit",
            )[(task_index + arm_index) % 3]
            context = (
                f"{domain}:user:{'early' if decision_idx < 3 else 'middle'}:0:"
                f"{stable_hash(arm)}:{stable_hash('seed-300')}"
            )
            base_tokens = 1100 + 75 * arm_index + 11 * task_index
            _sim_call(
                output_dir,
                arm=arm,
                domain=domain,
                task_id=task_id,
                decision_idx=decision_idx,
                component=component,
                context=context,
                purpose="incumbent_rollout",
                model=DEFAULT_ACTOR_MODEL,
                total_tokens=base_tokens,
                ordinal=0,
            )
            candidate_scores: dict[str, float] = {}
            bernoulli_outcomes: dict[str, int] = {}
            if arm != "GREEDY":
                explore_component = (
                    "tool_progress" if component != "tool_progress" else "verify_then_commit"
                )
                exploration_tokens = base_tokens + 150 + (task_index % 5) * 80
                _sim_call(
                    output_dir,
                    arm=arm,
                    domain=domain,
                    task_id=task_id,
                    decision_idx=decision_idx,
                    component=explore_component,
                    context=context,
                    purpose="exploration_rollout",
                    model=DEFAULT_ACTOR_MODEL,
                    total_tokens=exploration_tokens,
                    ordinal=1,
                )
                score = round(0.25 + 0.7 * rng.random(), 4)
                _sim_call(
                    output_dir,
                    arm=arm,
                    domain=domain,
                    task_id=task_id,
                    decision_idx=decision_idx,
                    component=explore_component,
                    context=context,
                    purpose="proxy_score",
                    model=DEFAULT_PROXY_MODEL,
                    total_tokens=320 + task_index % 40,
                    ordinal=2,
                    score=score,
                )
                candidate_scores = {component: max(score - 0.08, 0.0), explore_component: score}
                if arm == "CERT":
                    bernoulli_outcomes = {
                        name: _sim_coin(value, f"{arm}:{domain}:{task_id}:{name}")
                        for name, value in candidate_scores.items()
                    }
                    append_jsonl(
                        output_dir / "events.jsonl",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": "sim-v0.1.0",
                            "arm": arm,
                            "domain": domain,
                            "task_id": task_id,
                            "trial": 0,
                            "decision_idx": decision_idx,
                            "component": explore_component,
                            "context_key": context,
                            "event": "NotSeparated",
                            "detail": "[SIM] pair audit remained below threshold",
                            "log_e": -0.5,
                            "threshold": 8.0,
                        },
                    )
            append_jsonl(
                output_dir / "decisions.jsonl",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "sim-v0.1.0",
                    "arm": arm,
                    "domain": domain,
                    "task_id": task_id,
                    "trial": 0,
                    "decision_idx": decision_idx,
                    "context_key": context,
                    "incumbent_component": component,
                    "challenger_component": (
                        next((name for name in candidate_scores if name != component), None)
                    ),
                    "selected_component": (
                        max(candidate_scores, key=candidate_scores.get)
                        if candidate_scores
                        else component
                    ),
                    "outcome": "[SIM] NotSeparated" if arm == "CERT" else "[SIM] selection",
                    "candidate_scores": candidate_scores,
                    "bernoulli_outcomes": bernoulli_outcomes,
                    "accept_log_e": -0.5 if arm == "CERT" else None,
                    "refute_log_e": -0.8 if arm == "CERT" else None,
                    "threshold": 8.0 if arm == "CERT" else None,
                },
            )
            append_jsonl(
                output_dir / "simulations.jsonl",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "sim-v0.1.0",
                    "arm": arm,
                    "domain": domain,
                    "task_id": task_id,
                    "trial": 0,
                    "termination_reason": "[SIM] user_stop",
                    "reward_info": {
                        "reward": 1.0 if task_index in successful_indices else 0.0,
                        "info": {"label": "[SIM] deterministic fixture"},
                    },
                    "messages": [],
                },
            )

    _write_sim_state(output_dir)
    build_report(output_dir, tuple(DEFAULT_ARMS))
    generate_trace_site(output_dir, output_dir / "site")
    return output_dir


def _sim_call(
    output_dir: Path,
    *,
    arm: str,
    domain: str,
    task_id: str,
    decision_idx: int,
    component: str,
    context: str,
    purpose: str,
    model: str,
    total_tokens: int,
    ordinal: int,
    score: float | None = None,
) -> None:
    output_tokens = max(64, total_tokens // 5)
    input_tokens = total_tokens - output_tokens
    usage = Usage(input_tokens=input_tokens, cached_input_tokens=0, output_tokens=output_tokens)
    cost = billed_usd(model, usage)
    call_id = hashlib.sha256(
        f"sim:{arm}:{domain}:{task_id}:{decision_idx}:{purpose}:{ordinal}".encode()
    ).hexdigest()[:24]
    append_jsonl(
        output_dir / "calls.jsonl",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": "sim-v0.1.0",
            "arm": arm,
            "domain": domain,
            "task_id": task_id,
            "trial": 0,
            "decision_idx": decision_idx,
            "component": component,
            "context_key": context,
            "purpose": purpose,
            "requested_model": model,
            "resolved_model": f"[SIM] {model}",
            "physical_call_id": call_id,
            "request_id": f"sim_{call_id}",
            "status": "success",
            "attempts": 1,
            "input_tokens": input_tokens,
            "cached_input_tokens": 0,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "total_tokens": total_tokens,
            "billed_usd": cost,
            "provider_cost_usd": cost,
            "latency_seconds": round(0.2 + ordinal * 0.04, 3),
            "score": score,
            "error": None,
        },
    )


def _sim_coin(score: float, event_id: str) -> int:
    digest = hashlib.sha256(f"{DEFAULT_SEED}:{event_id}".encode()).digest()
    return int(int.from_bytes(digest[:8], "big") / float(1 << 64) < score)


def _write_sim_state(output_dir: Path) -> None:
    keys: dict[str, Any] = {}
    statuses = ("EvidenceCensored", "Active", "ModulusTooCoarse", "Inhomogeneous")
    for index in range(12):
        keys[f"sim-key-{index}"] = {
            "status": statuses[index % len(statuses)],
            "stream": {"observations": 4 + index, "successes": 2 + index // 2},
            "retired": False,
        }
    _write_json(
        output_dir / "state" / "CERT_trial-0.json",
        {
            "seed": DEFAULT_SEED,
            "declared_modulus": {
                "features": ["min(turn,8)/8", "mutation_seen"],
                "metric": "l1",
                "L": 1.0,
                "status": "declared_not_empirically_proven",
            },
            "keys": keys,
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic [SIM] release")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/t5r/public_artifacts/sim-v0.1.0"),
    )
    args = parser.parse_args(argv)
    print(generate_simulated_release(args.source_dir, args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
