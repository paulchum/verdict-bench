from __future__ import annotations

import math
import json
import unittest
from pathlib import Path

from benchmarks.t5r.certification import (
    AuditKey,
    CertifiedEngine,
    StreamStats,
    bernoulliize,
    ebh_log_threshold,
    evidence_floor,
    fixed_scale_log_e,
    half_null_log_e,
    kl_bernoulli,
    ledger_log_e,
    pair_log_e,
)


def audit_key(component: str, *, domain: str = "airline", bucket: str = "early") -> AuditKey:
    return AuditKey(
        component=component,
        domain=domain,
        last_event="user",
        turn_bucket=bucket,
        mutation_seen=False,
        config_hash="config",
        sampler_hash="sampler",
    )


class CertificationTests(unittest.TestCase):
    def test_moonshot_golden_parity_vectors(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "moonshot_parity_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        for case in fixture["pair_log_e"]:
            actual = pair_log_e(case["n_a"], case["s_a"], case["n_b"], case["s_b"])
            self.assertAlmostEqual(actual, case["expected"], places=12)
        for case in fixture["half_null_log_e"]:
            actual = half_null_log_e(
                case["n_candidate"],
                case["s_candidate"],
                case["n_incumbent"],
                case["s_incumbent"],
            )
            self.assertAlmostEqual(actual, case["expected"], places=12)
        for case in fixture["ledger_log_e"]:
            actual = ledger_log_e(case["n"], case["successes"], case["boundary"])
            self.assertAlmostEqual(actual, case["expected"], places=12)
        for case in fixture["fixed_scale_w"]:
            for direction in ("below", "above"):
                actual = fixed_scale_log_e(
                    case["n"],
                    case["successes"],
                    case["boundary"],
                    direction=direction,
                )
                self.assertAlmostEqual(actual, case[direction], places=12)
        for case in fixture["floors"]:
            self.assertEqual(
                evidence_floor(case["theta"], case["delta"]), case["expected"]
            )

    def test_moonshot_default_evidence_floor_is_four(self) -> None:
        self.assertEqual(evidence_floor(0.4, 0.1), 4)

    def test_bernoulliization_is_seeded_and_extremes_are_exact(self) -> None:
        self.assertEqual(bernoulliize(0.0, seed=300, event_id="x"), 0)
        self.assertEqual(bernoulliize(1.0, seed=300, event_id="x"), 1)
        first = bernoulliize(0.43, seed=300, event_id="episode-7")
        self.assertEqual(first, bernoulliize(0.43, seed=300, event_id="episode-7"))

    def test_kt_glr_separates_strong_pair(self) -> None:
        self.assertGreater(pair_log_e(100, 5, 100, 95), 20.0)
        self.assertGreater(half_null_log_e(100, 5, 100, 95), 1.0)
        self.assertGreater(kl_bernoulli(0.1, 0.9), 1.0)

    def test_e_bh_and_ledger_are_finite(self) -> None:
        threshold = ebh_log_threshold(108, 0.1, 0)
        self.assertAlmostEqual(threshold, math.log(1080.0))
        self.assertGreater(ledger_log_e(100, 100, 0.6), threshold)

    def test_engine_persists_and_accepts_without_episode_reset(self) -> None:
        engine = CertifiedEngine(seed=300, horizon=200)
        keys = {name: audit_key(name) for name in engine.components}
        for index in range(60):
            engine.observe(
                keys["policy_first"], score=0.0, event_id=f"inc-{index}", subcell="0.000"
            )
            engine.observe(
                keys["tool_progress"], score=1.0, event_id=f"new-{index}", subcell="0.000"
            )
        decision = engine.decide(keys, challenger_component="tool_progress")
        self.assertEqual(decision.outcome, "AcceptCertificate")
        self.assertEqual(engine.incumbent_for(keys["policy_first"].context_id), "tool_progress")
        self.assertEqual(engine.state_for(keys["tool_progress"]).stream.observations, 60)

    def test_retirement_is_context_scoped(self) -> None:
        engine = CertifiedEngine(seed=300, horizon=20)
        airline = audit_key("tool_progress", domain="airline")
        retail = audit_key("tool_progress", domain="retail")
        engine.state_for(airline).retired = True
        self.assertTrue(engine.state_for(airline).retired)
        self.assertFalse(engine.state_for(retail).retired)

    def test_t8_modulus_and_inhomogeneity_refusals(self) -> None:
        engine = CertifiedEngine(seed=300, horizon=20)
        coarse = audit_key("policy_first")
        coarse_state = engine.state_for(coarse)
        coarse_state.subcells["0.000"] = StreamStats(4, 2, [])
        coarse_state.subcells["0.500"] = StreamStats(4, 2, [])
        self.assertEqual(engine.key_status(coarse), "ModulusTooCoarse")

        inhomogeneous = audit_key("tool_progress")
        inhomogeneous_state = engine.state_for(inhomogeneous)
        inhomogeneous_state.subcells["0.000"] = StreamStats(200, 0, [])
        inhomogeneous_state.subcells["0.250"] = StreamStats(200, 200, [])
        self.assertEqual(engine.key_status(inhomogeneous), "Inhomogeneous")


if __name__ == "__main__":
    unittest.main()
