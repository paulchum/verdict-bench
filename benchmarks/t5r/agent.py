from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.t5r.budget import BudgetGuard
from benchmarks.t5r.certification import CertifiedEngine, EngineEvent
from benchmarks.t5r.config import COMPONENTS, DEFAULT_ACTOR_MODEL, DEFAULT_PROXY_MODEL
from benchmarks.t5r.context import describe_context
from benchmarks.t5r.ledger import (
    DecisionRecord,
    EventRecord,
    make_call_record,
    usage_from_message,
    write_call,
    write_decision,
    write_event,
)

from tau2.agent.base_agent import (  # type: ignore[import-not-found]
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (  # type: ignore[import-not-found]
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool  # type: ignore[import-not-found]
from tau2.utils.llm_utils import extract_json_from_llm_response, generate  # type: ignore[import-not-found]


COMPONENT_PROMPTS = {
    "policy_first": (
        "For this response, first apply the domain policy exactly. Ask a concise "
        "clarifying question when policy-required information is missing."
    ),
    "tool_progress": (
        "For this response, prefer a valid tool call when the visible state supports "
        "one. Never invent tool arguments."
    ),
    "verify_then_commit": (
        "For this response, verify the policy and visible facts before acting. Ask the "
        "user before an unsupported irreversible action."
    ),
}


@dataclass(frozen=True)
class ArmConfig:
    name: str
    bon_n: int = 0
    epsilon: float = 0.0
    raw_mode: bool = False
    raw_margin: float = 0.0
    raw_retirement_window: int = 4
    cert_mode: bool = False
    cert_ei_mode: bool = False
    cert_budget_rate: float = 1.0


ARM_CONFIGS = {
    "GREEDY": ArmConfig("GREEDY"),
    "BON-2": ArmConfig("BON-2", bon_n=2),
    "BON-4": ArmConfig("BON-4", bon_n=4),
    "EPS-0.35": ArmConfig("EPS-0.35", epsilon=0.35),
    "RAW": ArmConfig("RAW", raw_mode=True),
    "CERT": ArmConfig("CERT", cert_mode=True, cert_budget_rate=0.35),
    "CERT-EI": ArmConfig(
        "CERT-EI", cert_mode=True, cert_ei_mode=True, cert_budget_rate=0.35
    ),
    "HEURISTIC-v0": ArmConfig("HEURISTIC-v0"),
}


def arm_config_for(name: str) -> ArmConfig:
    if name in ARM_CONFIGS:
        return ARM_CONFIGS[name]
    epsilon_match = re.fullmatch(r"EPS-(0(?:\.\d+)?|1(?:\.0+)?)", name)
    if epsilon_match:
        return ArmConfig(name, epsilon=float(epsilon_match.group(1)))
    raw_match = re.fullmatch(r"RAW-m(0(?:\.\d+)?|1(?:\.0+)?)-w(\d+)", name)
    if raw_match:
        return ArmConfig(
            name,
            raw_mode=True,
            raw_margin=float(raw_match.group(1)),
            raw_retirement_window=int(raw_match.group(2)),
        )
    cert_match = re.fullmatch(r"CERT-b(0(?:\.\d+)?|1(?:\.0+)?)", name)
    if cert_match:
        return ArmConfig(
            name,
            cert_mode=True,
            cert_budget_rate=float(cert_match.group(1)),
        )
    raise ValueError(f"Unknown arm: {name}")


@dataclass
class Candidate:
    message: AssistantMessage
    component: str
    purpose: str
    score: float | None = None
    audit_outcome: int | None = None


@dataclass
class RawKeyStats:
    scores: list[float] = field(default_factory=list)
    retired: bool = False


@dataclass
class RawRunState:
    by_key: dict[str, RawKeyStats] = field(default_factory=dict)

    def state_for(self, key: str) -> RawKeyStats:
        return self.by_key.setdefault(key, RawKeyStats())


@dataclass
class GatedAgentState:
    system_messages: list[SystemMessage]
    messages: list[Message]
    decision_idx: int = 0


class GatedTauAgent(HalfDuplexAgent[GatedAgentState]):
    """Tau text agent whose CERT arm delegates to the T5K/T4B engine."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        *,
        run_id: str,
        arm: str,
        domain: str,
        task_id: str,
        trial: int,
        ledger_path: Path,
        events_path: Path,
        decisions_path: Path,
        certified_engine: CertifiedEngine,
        raw_run_state: RawRunState,
        budget_guard: BudgetGuard,
        config_hash: str,
        sampler_hash: str,
        llm: str = DEFAULT_ACTOR_MODEL,
        llm_args: dict[str, Any] | None = None,
        proxy_llm: str = DEFAULT_PROXY_MODEL,
        proxy_llm_args: dict[str, Any] | None = None,
        seed: int = 300,
        raw_retirement_min_samples: int = 4,
        raw_retirement_threshold: float = 0.25,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        if arm == "HEURISTIC-v0":
            raise ValueError("HEURISTIC-v0 is retained for provenance and cannot run publicly")
        self.run_id = run_id
        self.arm = arm
        self.arm_config = arm_config_for(arm)
        self.domain = domain
        self.task_id = task_id
        self.trial = trial
        self.ledger_path = ledger_path
        self.events_path = events_path
        self.decisions_path = decisions_path
        self.certified_engine = certified_engine
        self.raw_run_state = raw_run_state
        self.budget_guard = budget_guard
        self.config_hash = config_hash
        self.sampler_hash = sampler_hash
        self.llm = llm
        self.llm_args = llm_args or {"max_tokens": 2048}
        self.proxy_llm = proxy_llm
        self.proxy_llm_args = proxy_llm_args or {"max_tokens": 256}
        self.rng = random.Random(f"{seed}:{arm}:{domain}:{task_id}:{trial}")
        self.raw_retirement_min_samples = max(
            raw_retirement_min_samples, self.arm_config.raw_retirement_window
        )
        self.raw_retirement_threshold = raw_retirement_threshold

    @property
    def system_prompt(self) -> str:
        return (
            "<instructions>\n"
            "You are a customer service agent. Follow the policy exactly. In each turn, "
            "send a user message or make tool calls, never both.\n"
            "</instructions>\n"
            "<policy>\n"
            f"{self.domain_policy}\n"
            "</policy>"
        )

    def get_init_state(self, message_history: list[Message] | None = None) -> GatedAgentState:
        message_history = message_history or []
        assert all(is_valid_agent_history_message(message) for message in message_history)
        return GatedAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GatedAgentState
    ) -> tuple[AssistantMessage, GatedAgentState]:
        self._append_input(message, state)
        chosen = self._select_candidate(state)
        state.messages.append(chosen.message)
        state.decision_idx += 1
        return chosen.message, state

    def _append_input(self, message: ValidAgentInputMessage, state: GatedAgentState) -> None:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    def _context(self, state: GatedAgentState):
        return describe_context(
            domain=self.domain,
            decision_idx=state.decision_idx,
            messages=state.messages,
            config_hash=self.config_hash,
            sampler_hash=self.sampler_hash,
        )

    def _select_candidate(self, state: GatedAgentState) -> Candidate:
        context = self._context(state)
        keys = context.keys()
        config = self.arm_config

        if config.name == "GREEDY":
            candidate = self._generate_candidate(
                state, COMPONENTS[0], "incumbent_rollout", context.id
            )
            self._write_simple_decision(state, context.id, candidate, "GreedyCommit")
            return candidate

        if config.bon_n:
            candidates = [
                self._generate_candidate(
                    state,
                    COMPONENTS[offset % len(COMPONENTS)],
                    "exploration_rollout",
                    context.id,
                )
                for offset in range(config.bon_n)
            ]
            self._score_candidates(state, context.id, candidates)
            selected = max(candidates, key=lambda item: item.score if item.score is not None else -1)
            self._write_candidates_decision(state, context.id, candidates, selected, "BestOfN")
            return selected

        incumbent_component = self.certified_engine.incumbent_for(context.id)
        incumbent = self._generate_candidate(
            state, incumbent_component, "incumbent_rollout", context.id
        )

        if config.epsilon:
            if self.rng.random() >= config.epsilon:
                self._write_simple_decision(state, context.id, incumbent, "EpsilonExploit")
                return incumbent
            challenger_component = self._next_component(incumbent_component)
            challenger = self._generate_candidate(
                state, challenger_component, "exploration_rollout", context.id
            )
            candidates = [incumbent, challenger]
            self._score_candidates(state, context.id, candidates)
            selected = max(candidates, key=lambda item: item.score if item.score is not None else -1)
            self._write_candidates_decision(state, context.id, candidates, selected, "EpsilonExplore")
            return selected

        if config.raw_mode:
            candidates = [incumbent]
            for component in COMPONENTS:
                key = keys[component].id
                if component != incumbent_component and not self.raw_run_state.state_for(key).retired:
                    candidates.append(
                        self._generate_candidate(
                            state, component, "exploration_rollout", context.id
                        )
                    )
            self._score_candidates(state, context.id, candidates)
            self._update_raw_retirements(state, keys, candidates)
            incumbent_score = float(candidates[0].score)
            best = max(
                candidates,
                key=lambda item: item.score if item.score is not None else -1,
            )
            selected = (
                best
                if float(best.score) >= incumbent_score + config.raw_margin
                else candidates[0]
            )
            self._write_candidates_decision(state, context.id, candidates, selected, "RawArgmax")
            return selected

        if config.cert_mode:
            if self.rng.random() > config.cert_budget_rate:
                self._write_simple_decision(
                    state, context.id, incumbent, "CertificateBudgetExploit"
                )
                return incumbent
            challenger_component = self.certified_engine.choose_challenger(
                keys, use_chi=not config.cert_ei_mode
            )
            if challenger_component is None:
                self._write_simple_decision(state, context.id, incumbent, "NoActiveChallenger")
                return incumbent
            challenger = self._generate_candidate(
                state, challenger_component, "exploration_rollout", context.id
            )
            candidates = [incumbent, challenger]
            self._score_candidates(state, context.id, candidates)
            outcomes: dict[str, int] = {}
            for candidate in candidates:
                event_id = (
                    f"{self.run_id}:{self.arm}:{self.domain}:{self.task_id}:{self.trial}:"
                    f"{state.decision_idx}:{candidate.component}"
                )
                outcome, engine_events = self.certified_engine.observe(
                    keys[candidate.component],
                    score=float(candidate.score),
                    event_id=event_id,
                    subcell=context.subcell,
                )
                candidate.audit_outcome = outcome
                outcomes[candidate.component] = outcome
                self._write_engine_events(state, engine_events)
            decision = self.certified_engine.decide(
                keys, challenger_component=challenger_component
            )
            selected = next(
                candidate
                for candidate in candidates
                if candidate.component == decision.selected_component
            )
            write_decision(
                self.decisions_path,
                DecisionRecord(
                    run_id=self.run_id,
                    arm=self.arm,
                    domain=self.domain,
                    task_id=self.task_id,
                    trial=self.trial,
                    decision_idx=state.decision_idx,
                    context_key=context.id,
                    incumbent_component=decision.incumbent_component,
                    challenger_component=decision.challenger_component,
                    selected_component=decision.selected_component,
                    outcome=decision.outcome,
                    candidate_scores={c.component: float(c.score) for c in candidates},
                    bernoulli_outcomes=outcomes,
                    accept_log_e=decision.accept_log_e,
                    refute_log_e=decision.refute_log_e,
                    threshold=decision.threshold,
                ),
            )
            self._event(
                state,
                challenger_component,
                context.id,
                decision.outcome,
                "pair audit decision",
                max(decision.accept_log_e, decision.refute_log_e),
                decision.threshold,
            )
            return selected

        raise RuntimeError(f"Arm has no selection behavior: {self.arm}")

    def _next_component(self, incumbent: str) -> str:
        index = COMPONENTS.index(incumbent)
        return COMPONENTS[(index + 1) % len(COMPONENTS)]

    def _generate_candidate(
        self,
        state: GatedAgentState,
        component: str,
        purpose: str,
        context_key: str,
    ) -> Candidate:
        self._event(
            state,
            component,
            context_key,
            "CandidateLabelSelected",
            f"component selected before generation; purpose={purpose}",
        )
        messages: list[Message] = list(state.system_messages)
        messages.append(
            SystemMessage(
                role="system",
                content=f'<component name="{component}">{COMPONENT_PROMPTS[component]}</component>',
            )
        )
        messages.extend(state.messages)
        self.budget_guard.check(0.50)
        started = time.monotonic()
        try:
            message = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                call_name=f"t5r_{self.arm}_{purpose}",
                **self.llm_args,
            )
        except Exception as exc:
            write_call(
                self.ledger_path,
                make_call_record(
                    run_id=self.run_id,
                    arm=self.arm,
                    domain=self.domain,
                    task_id=self.task_id,
                    trial=self.trial,
                    decision_idx=state.decision_idx,
                    component=component,
                    context_key=context_key,
                    purpose=purpose,
                    model=self.llm,
                    usage=None,
                    raw_data=None,
                    provider_cost_usd=None,
                    status="error",
                    latency_seconds=time.monotonic() - started,
                    error=repr(exc),
                ),
            )
            self._event(
                state,
                component,
                context_key,
                "ProviderCallError",
                repr(exc),
            )
            raise
        usage, raw_data, provider_cost = usage_from_message(message)
        record = make_call_record(
            run_id=self.run_id,
            arm=self.arm,
            domain=self.domain,
            task_id=self.task_id,
            trial=self.trial,
            decision_idx=state.decision_idx,
            component=component,
            context_key=context_key,
            purpose=purpose,
            model=self.llm,
            usage=usage,
            raw_data=raw_data,
            provider_cost_usd=provider_cost,
            latency_seconds=time.monotonic() - started,
        )
        self.budget_guard.debit(record.billed_usd)
        write_call(self.ledger_path, record)
        return Candidate(message=message, component=component, purpose=purpose)

    def _score_candidates(
        self, state: GatedAgentState, context_key: str, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            candidate.score = self._proxy_score(state, context_key, candidate)

    def _proxy_score(
        self, state: GatedAgentState, context_key: str, candidate: Candidate
    ) -> float:
        candidate_text = self._format_candidate(candidate.message)
        recent = "\n\n".join(self._format_message(message) for message in state.messages[-8:])
        prompt = (
            "Score this candidate response from 0 to 1 using only the visible policy, "
            "visible conversation, and visible tool schema. Judge policy compliance, "
            "helpfulness, and likely progress. You have no hidden task criteria and must "
            "not infer any. Return JSON with numeric `score` and short `rationale`.\n\n"
            f"<policy>\n{self.domain_policy[:6000]}\n</policy>\n\n"
            f"<recent_conversation>\n{recent}\n</recent_conversation>\n\n"
            f"<candidate>\n{candidate_text}\n</candidate>"
        )
        self.budget_guard.check(0.10)
        started = time.monotonic()
        try:
            score_message = generate(
                model=self.proxy_llm,
                messages=[
                    SystemMessage(
                        role="system",
                        content="You are a strict visible-state proxy scorer.",
                    ),
                    UserMessage(role="user", content=prompt),
                ],
                call_name=f"t5r_{self.arm}_proxy_score",
                **self.proxy_llm_args,
            )
        except Exception as exc:
            write_call(
                self.ledger_path,
                make_call_record(
                    run_id=self.run_id,
                    arm=self.arm,
                    domain=self.domain,
                    task_id=self.task_id,
                    trial=self.trial,
                    decision_idx=state.decision_idx,
                    component=candidate.component,
                    context_key=context_key,
                    purpose="proxy_score",
                    model=self.proxy_llm,
                    usage=None,
                    raw_data=None,
                    provider_cost_usd=None,
                    status="error",
                    latency_seconds=time.monotonic() - started,
                    error=repr(exc),
                ),
            )
            self._event(
                state,
                candidate.component,
                context_key,
                "ProviderCallError",
                repr(exc),
            )
            raise
        score = self._parse_score(score_message.content)
        usage, raw_data, provider_cost = usage_from_message(score_message)
        record = make_call_record(
            run_id=self.run_id,
            arm=self.arm,
            domain=self.domain,
            task_id=self.task_id,
            trial=self.trial,
            decision_idx=state.decision_idx,
            component=candidate.component,
            context_key=context_key,
            purpose="proxy_score",
            model=self.proxy_llm,
            usage=usage,
            raw_data=raw_data,
            provider_cost_usd=provider_cost,
            latency_seconds=time.monotonic() - started,
            score=score,
        )
        self.budget_guard.debit(record.billed_usd)
        write_call(self.ledger_path, record)
        return score

    @staticmethod
    def _parse_score(content: str | None) -> float:
        if not content:
            raise ValueError("Proxy scorer returned empty content")
        payload = json.loads(extract_json_from_llm_response(content))
        score = float(payload["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Proxy score outside [0,1]: {score}")
        return score

    def _update_raw_retirements(self, state, keys, candidates) -> None:
        for candidate in candidates:
            stats = self.raw_run_state.state_for(keys[candidate.component].id)
            stats.scores.append(float(candidate.score))
            if stats.retired or len(stats.scores) < self.raw_retirement_min_samples:
                continue
            if sum(stats.scores) / len(stats.scores) < self.raw_retirement_threshold:
                stats.retired = True
                self._event(
                    state,
                    candidate.component,
                    keys[candidate.component].id,
                    "RawRetirement",
                    "uncertified plug-in mean fell below the frozen threshold",
                )

    def _write_engine_events(self, state: GatedAgentState, events: list[EngineEvent]) -> None:
        for event in events:
            self._event(
                state,
                event.component,
                event.key,
                event.event,
                event.detail,
                event.log_e,
                event.threshold,
            )

    def _write_simple_decision(
        self, state: GatedAgentState, context_key: str, candidate: Candidate, outcome: str
    ) -> None:
        write_decision(
            self.decisions_path,
            DecisionRecord(
                run_id=self.run_id,
                arm=self.arm,
                domain=self.domain,
                task_id=self.task_id,
                trial=self.trial,
                decision_idx=state.decision_idx,
                context_key=context_key,
                incumbent_component=candidate.component,
                challenger_component=None,
                selected_component=candidate.component,
                outcome=outcome,
                candidate_scores={},
                bernoulli_outcomes={},
            ),
        )

    def _write_candidates_decision(
        self,
        state: GatedAgentState,
        context_key: str,
        candidates: list[Candidate],
        selected: Candidate,
        outcome: str,
    ) -> None:
        write_decision(
            self.decisions_path,
            DecisionRecord(
                run_id=self.run_id,
                arm=self.arm,
                domain=self.domain,
                task_id=self.task_id,
                trial=self.trial,
                decision_idx=state.decision_idx,
                context_key=context_key,
                incumbent_component=candidates[0].component,
                challenger_component=(candidates[1].component if len(candidates) > 1 else None),
                selected_component=selected.component,
                outcome=outcome,
                candidate_scores={c.component: float(c.score) for c in candidates},
                bernoulli_outcomes={},
            ),
        )

    def _event(
        self,
        state: GatedAgentState,
        component: str | None,
        context_key: str | None,
        event: str,
        detail: str,
        log_e: float | None = None,
        threshold: float | None = None,
    ) -> None:
        write_event(
            self.events_path,
            EventRecord(
                run_id=self.run_id,
                arm=self.arm,
                domain=self.domain,
                task_id=self.task_id,
                trial=self.trial,
                decision_idx=state.decision_idx,
                component=component,
                context_key=context_key,
                event=event,
                detail=detail,
                log_e=log_e,
                threshold=threshold,
            ),
        )

    @staticmethod
    def _format_candidate(message: AssistantMessage) -> str:
        if message.is_tool_call():
            calls = [
                {"name": call.name, "arguments": call.arguments}
                for call in (message.tool_calls or [])
            ]
            return json.dumps({"tool_calls": calls}, sort_keys=True)
        return message.content or ""

    def _format_message(self, message: Message) -> str:
        if isinstance(message, ToolMessage):
            return f"tool: {message.content}"
        if isinstance(message, AssistantMessage):
            return f"assistant: {self._format_candidate(message)}"
        if isinstance(message, UserMessage):
            return f"user: {message.content}"
        return f"{getattr(message, 'role', 'message')}: {getattr(message, 'content', '')}"
