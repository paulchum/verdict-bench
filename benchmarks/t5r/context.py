from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from benchmarks.t5r.certification import AuditKey
from benchmarks.t5r.config import COMPONENTS, MUTATING_TOOLS, TURN_BUCKETS


def stable_hash(payload: Mapping[str, Any] | Iterable[Any] | str) -> str:
    if isinstance(payload, str):
        encoded = payload
    else:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def turn_bucket(decision_idx: int) -> str:
    for lower, upper, label in TURN_BUCKETS:
        if decision_idx >= lower and (upper is None or decision_idx <= upper):
            return label
    raise ValueError(f"No turn bucket for {decision_idx}")


def normalized_turn(decision_idx: int) -> str:
    return f"{min(max(decision_idx, 0), 8) / 8.0:.3f}"


def _tool_name_for_result(messages: list[Any], tool_result: Any) -> str | None:
    result_id = getattr(tool_result, "id", None)
    for message in reversed(messages):
        calls = getattr(message, "tool_calls", None) or []
        for call in calls:
            if result_id is None or getattr(call, "id", None) == result_id:
                return getattr(call, "name", None)
    return None


def classify_last_event(messages: list[Any]) -> tuple[str, bool]:
    mutation_seen = False
    last_event = "user"
    for index, message in enumerate(messages):
        role = str(getattr(message, "role", ""))
        if role in {"assistant", "agent"}:
            calls = getattr(message, "tool_calls", None) or []
            mutation_seen = mutation_seen or any(
                getattr(call, "name", None) in MUTATING_TOOLS for call in calls
            )
        if role == "tool":
            tool_name = _tool_name_for_result(messages[:index], message)
            is_mutating = tool_name in MUTATING_TOOLS
            mutation_seen = mutation_seen or is_mutating
            last_event = "write_tool" if is_mutating else "read_tool"
        elif role == "user":
            last_event = "user"
    return last_event, mutation_seen


@dataclass(frozen=True)
class ContextDescriptor:
    domain: str
    last_event: str
    turn_bucket: str
    mutation_seen: bool
    config_hash: str
    sampler_hash: str
    subcell: str

    @property
    def id(self) -> str:
        return ":".join(
            (
                self.domain,
                self.last_event,
                self.turn_bucket,
                str(int(self.mutation_seen)),
                self.config_hash,
                self.sampler_hash,
            )
        )

    def keys(self, components: tuple[str, ...] = COMPONENTS) -> dict[str, AuditKey]:
        return {
            component: AuditKey(
                component=component,
                domain=self.domain,
                last_event=self.last_event,
                turn_bucket=self.turn_bucket,
                mutation_seen=self.mutation_seen,
                config_hash=self.config_hash,
                sampler_hash=self.sampler_hash,
            )
            for component in components
        }


def describe_context(
    *,
    domain: str,
    decision_idx: int,
    messages: list[Any],
    config_hash: str,
    sampler_hash: str,
) -> ContextDescriptor:
    last_event, mutation_seen = classify_last_event(messages)
    return ContextDescriptor(
        domain=domain,
        last_event=last_event,
        turn_bucket=turn_bucket(decision_idx),
        mutation_seen=mutation_seen,
        config_hash=config_hash,
        sampler_hash=sampler_hash,
        subcell=normalized_turn(decision_idx),
    )
