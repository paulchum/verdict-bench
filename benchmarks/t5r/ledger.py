from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.t5r.config import SCHEMA_VERSION
from benchmarks.t5r.pricing import billed_usd, normalize_usage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CallRecord:
    run_id: str
    arm: str
    domain: str
    task_id: str
    trial: int
    decision_idx: int | None
    component: str | None
    context_key: str | None
    purpose: str
    requested_model: str
    resolved_model: str | None
    physical_call_id: str
    request_id: str | None
    status: str
    attempts: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    billed_usd: float
    provider_cost_usd: float | None = None
    latency_seconds: float | None = None
    score: float | None = None
    error: str | None = None
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class EventRecord:
    run_id: str
    arm: str
    domain: str
    task_id: str
    trial: int
    decision_idx: int | None
    component: str | None
    context_key: str | None
    event: str
    detail: str
    log_e: float | None = None
    threshold: float | None = None
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class DecisionRecord:
    run_id: str
    arm: str
    domain: str
    task_id: str
    trial: int
    decision_idx: int
    context_key: str
    incumbent_component: str
    challenger_component: str | None
    selected_component: str
    outcome: str
    candidate_scores: dict[str, float]
    bernoulli_outcomes: dict[str, int]
    accept_log_e: float | None = None
    refute_log_e: float | None = None
    threshold: float | None = None
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class ArmTotals:
    calls: int
    billed_usd: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    exploration_tokens: int
    proxy_tokens: int

    @property
    def exploration_share(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.exploration_tokens / self.total_tokens


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _raw_usage(raw_data: dict[str, Any] | None) -> dict[str, Any]:
    return (raw_data or {}).get("usage") or {}


def _resolved_model(raw_data: dict[str, Any] | None) -> str | None:
    raw_data = raw_data or {}
    return raw_data.get("model") or (raw_data.get("response") or {}).get("model")


def _request_id(raw_data: dict[str, Any] | None) -> str | None:
    raw_data = raw_data or {}
    return raw_data.get("id") or raw_data.get("request_id")


def _reasoning_tokens(usage: dict[str, Any] | None, raw_data: dict[str, Any] | None) -> int:
    usage = usage or {}
    raw_usage = _raw_usage(raw_data)
    details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or raw_usage.get("completion_tokens_details")
        or raw_usage.get("output_tokens_details")
        or {}
    )
    return int(details.get("reasoning_tokens") or 0)


def make_call_record(
    *,
    run_id: str,
    arm: str,
    domain: str,
    task_id: str,
    decision_idx: int | None,
    component: str | None,
    purpose: str,
    model: str,
    usage: dict[str, Any] | None,
    raw_data: dict[str, Any] | None,
    provider_cost_usd: float | None,
    trial: int = 0,
    context_key: str | None = None,
    attempts: int = 1,
    status: str = "success",
    latency_seconds: float | None = None,
    score: float | None = None,
    error: str | None = None,
) -> CallRecord:
    normalized = normalize_usage(usage, raw_data)
    request_id = _request_id(raw_data)
    identity_payload = json.dumps(
        {
            "run": run_id,
            "arm": arm,
            "domain": domain,
            "task": task_id,
            "trial": trial,
            "decision": decision_idx,
            "component": component,
            "purpose": purpose,
            "request_id": request_id,
        },
        sort_keys=True,
    )
    physical_call_id = request_id or hashlib.sha256(
        identity_payload.encode("utf-8")
    ).hexdigest()[:24]
    gross_cost = billed_usd(model, normalized) if status == "success" else 0.0
    return CallRecord(
        run_id=run_id,
        arm=arm,
        domain=domain,
        task_id=task_id,
        trial=trial,
        decision_idx=decision_idx,
        component=component,
        context_key=context_key,
        purpose=purpose,
        requested_model=model,
        resolved_model=_resolved_model(raw_data),
        physical_call_id=physical_call_id,
        request_id=request_id,
        status=status,
        attempts=attempts,
        input_tokens=normalized.input_tokens,
        cached_input_tokens=normalized.cached_input_tokens,
        output_tokens=normalized.output_tokens,
        reasoning_tokens=_reasoning_tokens(usage, raw_data),
        total_tokens=normalized.total_tokens,
        billed_usd=gross_cost,
        provider_cost_usd=provider_cost_usd,
        latency_seconds=latency_seconds,
        score=score,
        error=error,
    )


def write_call(path: Path, record: CallRecord) -> None:
    append_jsonl(path, asdict(record))


def write_event(path: Path, record: EventRecord) -> None:
    append_jsonl(path, asdict(record))


def write_decision(path: Path, record: DecisionRecord) -> None:
    append_jsonl(path, asdict(record))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_calls(records: list[dict[str, Any]], arm: str | None = None) -> ArmTotals:
    selected = [record for record in records if arm is None or record["arm"] == arm]
    input_tokens = sum(int(record.get("input_tokens", 0)) for record in selected)
    cached_input_tokens = sum(
        int(record.get("cached_input_tokens", 0)) for record in selected
    )
    output_tokens = sum(int(record.get("output_tokens", 0)) for record in selected)
    reasoning_tokens = sum(int(record.get("reasoning_tokens", 0)) for record in selected)
    total_tokens = sum(int(record.get("total_tokens", 0)) for record in selected)
    billed = sum(float(record.get("billed_usd", 0.0)) for record in selected)
    exploration = sum(
        int(record.get("total_tokens", 0))
        for record in selected
        if record.get("purpose") in {"exploration_rollout", "candidate_rollout"}
    )
    proxy = sum(
        int(record.get("total_tokens", 0))
        for record in selected
        if record.get("purpose") == "proxy_score"
    )
    return ArmTotals(
        calls=len(selected),
        billed_usd=billed,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        exploration_tokens=exploration,
        proxy_tokens=proxy,
    )


def usage_from_message(
    message: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float | None]:
    usage = getattr(message, "usage", None)
    raw_data = getattr(message, "raw_data", None)
    cost = getattr(message, "cost", None)
    return usage, raw_data, cost
