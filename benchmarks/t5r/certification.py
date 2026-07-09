from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from benchmarks.t5r.config import (
    COMPONENTS,
    DECLARED_MODULUS_L,
    FLEET_DELTA,
    K_MAX,
    THETA,
    W_TRIAL_CAP,
    Y_STAR,
)

LOG2 = math.log(2.0)
_NUMERIC_EPSILON = 1e-12


def kl_bernoulli(p: float, q: float) -> float:
    p = min(max(float(p), _NUMERIC_EPSILON), 1.0 - _NUMERIC_EPSILON)
    q = min(max(float(q), _NUMERIC_EPSILON), 1.0 - _NUMERIC_EPSILON)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kt_regret(n: int) -> float:
    return 0.5 * math.log(max(int(n), 1)) + LOG2


def pair_log_e(
    n_a: int, successes_a: int, n_b: int, successes_b: int
) -> float:
    """Anytime log e-value for the one-sided null mean(a) >= mean(b)."""
    if n_a < 1 or n_b < 1:
        return -math.inf
    mean_a = successes_a / n_a
    mean_b = successes_b / n_b
    if mean_a >= mean_b:
        return -(kt_regret(n_a) + kt_regret(n_b))
    pooled = (successes_a + successes_b) / (n_a + n_b)
    z = n_a * kl_bernoulli(mean_a, pooled) + n_b * kl_bernoulli(mean_b, pooled)
    return z - kt_regret(n_a) - kt_regret(n_b)


def half_null_log_e(
    n_candidate: int,
    successes_candidate: int,
    n_incumbent: int,
    successes_incumbent: int,
) -> float:
    """Lower-bounded log e-value refuting a factor-two improvement.

    The null is mean(candidate) >= (1 + mean(incumbent)) / 2. The scalar
    bisection and conservative slack are the executable TNA construction.
    """
    if n_candidate < 1 or n_incumbent < 1:
        return -math.inf
    candidate_mean = successes_candidate / n_candidate
    incumbent_mean = successes_incumbent / n_incumbent
    if candidate_mean >= (1.0 + incumbent_mean) / 2.0 - 1e-15:
        return -(kt_regret(n_candidate) + kt_regret(n_incumbent))

    low = max(2.0 * candidate_mean - 1.0, 1e-9)
    high = max(incumbent_mean, low + 1e-15)

    def derivative(value: float) -> float:
        boundary_candidate = (1.0 + value) / 2.0
        candidate_term = (
            0.5
            * n_candidate
            * (boundary_candidate - candidate_mean)
            / max(boundary_candidate * (1.0 - boundary_candidate), _NUMERIC_EPSILON)
        )
        incumbent_term = (
            n_incumbent
            * (value - incumbent_mean)
            / max(value * (1.0 - value), _NUMERIC_EPSILON)
        )
        return candidate_term + incumbent_term

    for _ in range(48):
        middle = 0.5 * (low + high)
        if derivative(middle) < 0.0:
            low = middle
        else:
            high = middle

    boundary_candidate = (1.0 + high) / 2.0
    z = n_candidate * kl_bernoulli(candidate_mean, boundary_candidate)
    z += n_incumbent * kl_bernoulli(incumbent_mean, high)
    conservative_slack = (high - low) * max(derivative(high), 0.0) + 1e-9
    return max(z - conservative_slack, 0.0) - kt_regret(
        n_candidate
    ) - kt_regret(n_incumbent)


def ebh_log_threshold(k_max: int, delta: float, executed: int) -> float:
    return math.log(k_max / (delta * (executed + 1.0)))


def ledger_log_e(n: int, drop_successes: int, boundary: float) -> float:
    if n < 1:
        return -math.inf
    empirical = drop_successes / n
    z = n * kl_bernoulli(empirical, boundary) if empirical > boundary else 0.0
    return z - kt_regret(n)


def fixed_scale_log_e(
    n: int,
    successes: int,
    boundary: float,
    *,
    direction: str,
) -> float:
    """Bound-KT one-sided W clock at a fixed Bernoulli boundary."""
    if n < 1:
        return -math.inf
    empirical = successes / n
    if direction == "below":
        z = n * kl_bernoulli(empirical, boundary) if empirical < boundary else 0.0
    elif direction == "above":
        z = n * kl_bernoulli(empirical, boundary) if empirical > boundary else 0.0
    else:
        raise ValueError(f"Unknown fixed-scale direction: {direction}")
    return z - kt_regret(n)


def evidence_floor(theta: float = THETA, delta: float = FLEET_DELTA) -> int:
    return int(
        math.ceil(
            math.log(1.0 / (2.0 * delta)) / math.log(1.0 / (1.0 - theta))
        )
    )


def bernoulliize(score: float, *, seed: int, event_id: str) -> int:
    """Convert a bounded proxy score to a reproducible Bernoulli observation."""
    score = min(max(float(score), 0.0), 1.0)
    digest = hashlib.sha256(f"{seed}:{event_id}".encode("utf-8")).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return int(uniform < score)


def truncated_ei(scores: Iterable[float], incumbent_mean: float) -> float:
    depth_v = 1.0 - min(max(incumbent_mean, 0.0), 1.0)
    values = []
    for score in scores:
        depth_u = 1.0 - min(max(float(score), 0.0), 1.0)
        values.append((depth_v - depth_u) if depth_u <= depth_v / 2.0 else 0.0)
    return sum(values) / len(values) if values else 0.0


def raw_ei(scores: Iterable[float], incumbent_mean: float) -> float:
    depth_v = 1.0 - min(max(incumbent_mean, 0.0), 1.0)
    values = [max(depth_v - (1.0 - float(score)), 0.0) for score in scores]
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class AuditKey:
    component: str
    domain: str
    last_event: str
    turn_bucket: str
    mutation_seen: bool
    config_hash: str
    sampler_hash: str

    @property
    def context_id(self) -> str:
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

    @property
    def id(self) -> str:
        return f"{self.component}:{self.context_id}"


@dataclass
class StreamStats:
    observations: int = 0
    successes: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.successes / self.observations if self.observations else 0.5


@dataclass
class WTrialState:
    observations: int = 0
    successes: int = 0


@dataclass
class ComponentKeyState:
    stream: StreamStats = field(default_factory=StreamStats)
    w_trial: WTrialState = field(default_factory=WTrialState)
    ledger_trials: int = 0
    ledger_drops: int = 0
    ledger_log_e: float = -math.inf
    retired: bool = False
    retirement_log_e: float | None = None
    subcells: dict[str, StreamStats] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineEvent:
    event: str
    key: str
    component: str
    detail: str
    log_e: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class CertifiedDecision:
    selected_component: str
    incumbent_component: str
    challenger_component: str
    outcome: str
    accept_log_e: float
    refute_log_e: float
    threshold: float


class CertifiedEngine:
    """Run-level certified routing state for the T5-R real-workload adapter."""

    def __init__(
        self,
        *,
        seed: int,
        horizon: int,
        components: tuple[str, ...] = COMPONENTS,
        theta: float = THETA,
        y_star: float = Y_STAR,
        fleet_delta: float = FLEET_DELTA,
        k_max: int = K_MAX,
        modulus_l: float = DECLARED_MODULUS_L,
        w_trial_cap: int = W_TRIAL_CAP,
    ) -> None:
        self.seed = seed
        self.horizon = max(int(horizon), 2)
        self.components = components
        self.theta = theta
        self.y_star = y_star
        self.fleet_delta = fleet_delta
        self.k_max = k_max
        self.modulus_l = modulus_l
        self.w_trial_cap = w_trial_cap
        self.delta_t = 1.0 / (self.horizon * self.horizon)
        self.audit_threshold = math.log(1.0 / self.delta_t)
        self.n_floor = evidence_floor(theta, fleet_delta)
        self._states: dict[str, ComponentKeyState] = {}
        self._keys: dict[str, AuditKey] = {}
        self._incumbents: dict[str, str] = {}
        self._retirements_executed = 0

    def state_for(self, key: AuditKey) -> ComponentKeyState:
        self._keys.setdefault(key.id, key)
        return self._states.setdefault(key.id, ComponentKeyState())

    def incumbent_for(self, context_id: str) -> str:
        return self._incumbents.setdefault(context_id, self.components[0])

    def active_components(self, keys: dict[str, AuditKey]) -> list[str]:
        active = [
            component
            for component in self.components
            if not self.state_for(keys[component]).retired
        ]
        return active

    def choose_challenger(
        self, keys: dict[str, AuditKey], *, use_chi: bool = True
    ) -> str | None:
        context_id = next(iter(keys.values())).context_id
        incumbent = self.incumbent_for(context_id)
        active = [component for component in self.active_components(keys) if component != incumbent]
        if not active:
            return None
        incumbent_mean = self.state_for(keys[incumbent]).stream.mean
        scored: list[tuple[float, int, str]] = []
        for index, component in enumerate(active):
            scores = self.state_for(keys[component]).stream.scores
            ranking = (
                truncated_ei(scores, incumbent_mean)
                if use_chi
                else raw_ei(scores, incumbent_mean)
            )
            scored.append((ranking, -index, component))
        if all(score <= 0.0 for score, _, _ in scored):
            fewest = min(
                active,
                key=lambda component: (
                    self.state_for(keys[component]).stream.observations,
                    self.components.index(component),
                ),
            )
            return fewest
        return max(scored)[2]

    def observe(
        self,
        key: AuditKey,
        *,
        score: float,
        event_id: str,
        subcell: str,
    ) -> tuple[int, list[EngineEvent]]:
        state = self.state_for(key)
        outcome = bernoulliize(score, seed=self.seed, event_id=event_id)
        state.stream.observations += 1
        state.stream.successes += outcome
        state.stream.scores.append(float(score))
        subcell_state = state.subcells.setdefault(subcell, StreamStats())
        subcell_state.observations += 1
        subcell_state.successes += outcome
        subcell_state.scores.append(float(score))

        events: list[EngineEvent] = []
        if state.retired:
            return outcome, events

        state.w_trial.observations += 1
        state.w_trial.successes += outcome
        n_w = state.w_trial.observations
        fixed_mean = 1.0 - self.y_star
        non_witness_log_e = fixed_scale_log_e(
            n_w, state.w_trial.successes, fixed_mean, direction="below"
        )
        witness_log_e = fixed_scale_log_e(
            n_w, state.w_trial.successes, fixed_mean, direction="above"
        )
        settled: int | None = None
        if non_witness_log_e >= self.audit_threshold:
            settled = 1
        elif witness_log_e >= self.audit_threshold:
            settled = 0
        elif n_w >= self.w_trial_cap:
            settled = 0

        if settled is not None:
            state.ledger_trials += 1
            state.ledger_drops += settled
            state.w_trial = WTrialState()
            boundary = 1.0 - self.theta * (1.0 - self.delta_t)
            state.ledger_log_e = ledger_log_e(
                state.ledger_trials, state.ledger_drops, boundary
            )
            events.append(
                EngineEvent(
                    event="ledger_trial",
                    key=key.id,
                    component=key.component,
                    detail=f"settled B={settled}; trials={state.ledger_trials}",
                    log_e=state.ledger_log_e,
                )
            )
            retirement_threshold = ebh_log_threshold(
                self.k_max, self.fleet_delta, self._retirements_executed
            )
            if (
                state.ledger_trials >= self.n_floor
                and state.ledger_log_e >= retirement_threshold
            ):
                state.retired = True
                state.retirement_log_e = state.ledger_log_e
                self._retirements_executed += 1
                events.append(
                    EngineEvent(
                        event="RetiredContextKey",
                        key=key.id,
                        component=key.component,
                        detail="component retired only inside the declared context key",
                        log_e=state.ledger_log_e,
                        threshold=retirement_threshold,
                    )
                )
        return outcome, events

    def decide(
        self,
        keys: dict[str, AuditKey],
        *,
        challenger_component: str,
    ) -> CertifiedDecision:
        context_id = keys[challenger_component].context_id
        incumbent_component = self.incumbent_for(context_id)
        incumbent = self.state_for(keys[incumbent_component]).stream
        challenger = self.state_for(keys[challenger_component]).stream
        accept = pair_log_e(
            incumbent.observations,
            incumbent.successes,
            challenger.observations,
            challenger.successes,
        )
        refute = half_null_log_e(
            challenger.observations,
            challenger.successes,
            incumbent.observations,
            incumbent.successes,
        )
        selected = incumbent_component
        outcome = "NotSeparated"
        if accept >= self.audit_threshold:
            selected = challenger_component
            outcome = "AcceptCertificate"
            self._incumbents[context_id] = challenger_component
        elif refute >= self.audit_threshold:
            outcome = "DropCertificate"
        return CertifiedDecision(
            selected_component=selected,
            incumbent_component=incumbent_component,
            challenger_component=challenger_component,
            outcome=outcome,
            accept_log_e=accept,
            refute_log_e=refute,
            threshold=self.audit_threshold,
        )

    def key_status(self, key: AuditKey) -> str:
        state = self.state_for(key)
        turn_values: list[float] = []
        for label in state.subcells:
            try:
                turn_values.append(float(label))
            except ValueError:
                continue
        diameter = max(turn_values) - min(turn_values) if turn_values else 0.0
        if self.modulus_l * diameter >= self.theta:
            return "ModulusTooCoarse"
        if self._inhomogeneous(state):
            return "Inhomogeneous"
        if state.ledger_trials < self.n_floor:
            return "EvidenceCensored"
        return "RetiredContextKey" if state.retired else "Active"

    def _inhomogeneous(self, state: ComponentKeyState) -> bool:
        labels = sorted(state.subcells)
        pairs = [(a, b) for index, a in enumerate(labels) for b in labels[index + 1 :]]
        threshold = ebh_log_threshold(max(len(pairs), 1), self.fleet_delta, 0)
        for left, right in pairs:
            a = state.subcells[left]
            b = state.subcells[right]
            if (
                pair_log_e(a.observations, a.successes, b.observations, b.successes)
                >= threshold
                or pair_log_e(b.observations, b.successes, a.observations, a.successes)
                >= threshold
            ):
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "horizon": self.horizon,
            "delta_t": self.delta_t,
            "audit_threshold": self.audit_threshold,
            "theta": self.theta,
            "y_star": self.y_star,
            "fleet_delta": self.fleet_delta,
            "k_max": self.k_max,
            "n_floor": self.n_floor,
            "declared_modulus": {
                "features": ["min(turn,8)/8", "mutation_seen"],
                "metric": "l1",
                "L": self.modulus_l,
                "status": "declared_not_empirically_proven",
            },
            "retirements_executed": self._retirements_executed,
            "incumbents": dict(sorted(self._incumbents.items())),
            "keys": {
                key: {
                    **asdict(state),
                    "audit_key": asdict(self._keys[key]),
                    "status": self.key_status(self._keys[key]),
                }
                for key, state in sorted(self._states.items())
            },
        }
