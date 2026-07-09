from __future__ import annotations

import inspect
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr
from unittest.mock import patch

from benchmarks.t5r.budget import BudgetExceeded, FundingRequired, require_funded_network_run
from benchmarks.t5r.cli import main
from benchmarks.t5r.context import classify_last_event, describe_context, turn_bucket
from benchmarks.t5r.report import canonicalize_json_numbers
from benchmarks.t5r.schema import validate_run_artifacts
from benchmarks.t5r.source import ensure_tau2_source
from benchmarks.t5r.splits import build_task_splits, sample_ids
from benchmarks.t5r.stats import paired_bootstrap_mean, wilson_interval


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".external" / "tau2-bench"


class Message:
    def __init__(self, role: str, *, tool_calls=None, identifier=None) -> None:
        self.role = role
        self.tool_calls = tool_calls or []
        self.id = identifier


class ToolCall:
    def __init__(self, name: str, identifier: str = "call-1") -> None:
        self.name = name
        self.id = identifier


class ControlTests(unittest.TestCase):
    def test_exact_seed_300_split_and_reserved_development_tasks(self) -> None:
        source = ensure_tau2_source(SOURCE, allow_clone=False)
        splits = build_task_splits(SOURCE, seed=300, source=source)
        self.assertEqual(
            splits.domains["airline"].pilot,
            ["0", "22", "27", "31", "33", "38", "47", "48"],
        )
        self.assertEqual(
            splits.domains["retail"].pilot,
            ["12", "15", "17", "20", "21", "31", "42", "47", "54", "64", "73", "74", "76", "80", "110", "111"],
        )
        self.assertEqual(splits.domains["airline"].development, ["49"])
        self.assertEqual(splits.domains["retail"].development, ["113"])
        self.assertEqual(len(splits.domains["airline"].adjudication_reserved), 41)
        self.assertEqual(len(splits.domains["retail"].adjudication_reserved), 97)

    def test_sample_is_deterministic(self) -> None:
        first = sample_ids([str(i) for i in range(20)], 5, random.Random(300))
        second = sample_ids([str(i) for i in range(20)], 5, random.Random(300))
        self.assertEqual(first, second)

    def test_context_uses_frozen_mutating_allowlist(self) -> None:
        messages = [
            Message("user"),
            Message("assistant", tool_calls=[ToolCall("book_reservation")]),
            Message("tool", identifier="call-1"),
        ]
        event, mutation = classify_last_event(messages)
        self.assertEqual(event, "write_tool")
        self.assertTrue(mutation)
        descriptor = describe_context(
            domain="airline",
            decision_idx=7,
            messages=messages,
            config_hash="c",
            sampler_hash="s",
        )
        self.assertEqual(descriptor.turn_bucket, "late")
        self.assertEqual(descriptor.subcell, "0.875")
        self.assertEqual(turn_bucket(0), "early")
        self.assertEqual(turn_bucket(3), "middle")

    def test_network_budget_requires_both_switch_and_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FundingRequired):
                require_funded_network_run(1.0)
        with patch.dict(
            os.environ,
            {"VERDICT_BENCH_FUNDED": "1", "OPENAI_API_KEY": "fixture"},
            clear=True,
        ):
            guard = require_funded_network_run(1.0)
            guard.debit(0.6)
            with self.assertRaises(BudgetExceeded):
                guard.debit(0.5)

    def test_cli_reports_expected_funding_refusal_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(stderr):
            result = main(["smoke", "--credit-budget-usd", "1"])
        self.assertEqual(result, 2)
        self.assertIn("Network runs are disabled", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_statistics_are_deterministic(self) -> None:
        interval = wilson_interval(5, 10)
        self.assertIsNotNone(interval)
        self.assertLess(interval.low, 0.5)
        self.assertGreater(interval.high, 0.5)
        first = paired_bootstrap_mean([1.0, 2.0, 3.0], draws=200, seed=300)
        second = paired_bootstrap_mean([1.0, 2.0, 3.0], draws=200, seed=300)
        self.assertEqual(first, second)

    def test_published_floats_are_runtime_stable(self) -> None:
        self.assertEqual(
            canonicalize_json_numbers({"cost": 0.25309799999999993}),
            {"cost": 0.253098},
        )
        self.assertEqual(canonicalize_json_numbers(-0.0), 0.0)
        with self.assertRaises(ValueError):
            canonicalize_json_numbers(float("nan"))

    def test_proxy_prompt_source_never_mentions_evaluation_criteria(self) -> None:
        source = (ROOT / "benchmarks" / "t5r" / "agent.py").read_text(encoding="utf-8")
        proxy_block = source[source.index("def _proxy_score"):source.index("def _update_raw_retirements")]
        self.assertNotIn("evaluation_criteria", proxy_block)
        self.assertIn("no hidden task criteria", proxy_block)

    def test_component_label_is_logged_before_generation(self) -> None:
        source = (ROOT / "benchmarks" / "t5r" / "agent.py").read_text(encoding="utf-8")
        block = source[source.index("def _generate_candidate"):source.index("def _score_candidates")]
        self.assertLess(block.index("CandidateLabelSelected"), block.index("message = generate"))

    def test_synthetic_release_validates_and_is_clearly_labeled(self) -> None:
        run_dir = ROOT / "benchmarks" / "t5r" / "public_artifacts" / "sim-v0.1.0"
        self.assertEqual(validate_run_artifacts(run_dir), [])
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["status"].startswith("[SIM]"))
        self.assertIsNone(summary["publication_gate"]["k_r1_outcome"])


if __name__ == "__main__":
    unittest.main()
