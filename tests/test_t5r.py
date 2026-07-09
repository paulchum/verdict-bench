from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.t5r.ledger import append_jsonl, summarize_calls
from benchmarks.t5r.pricing import Usage, billed_usd, normalize_usage
from benchmarks.t5r.report import build_report
from benchmarks.t5r.splits import sample_ids


class T5RTests(unittest.TestCase):
    def test_sample_ids_is_deterministic_with_rng(self) -> None:
        import random

        ids = [str(i) for i in range(20)]
        rng = random.Random(300)
        first = sample_ids(ids, 5, rng)
        rng = random.Random(300)
        second = sample_ids(ids, 5, rng)
        self.assertEqual(first, second)
        self.assertEqual(first, ["0", "11", "15", "18", "19"])

    def test_billing_uses_cached_input_discount(self) -> None:
        usage = normalize_usage(
            {"prompt_tokens": 1000, "completion_tokens": 200},
            {"usage": {"prompt_tokens_details": {"cached_tokens": 400}}},
        )
        self.assertEqual(usage, Usage(input_tokens=1000, cached_input_tokens=400, output_tokens=200))
        cost = billed_usd("openai/gpt-5-mini", usage)
        expected = (600 * 0.25 + 400 * 0.025 + 200 * 2.00) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_summarize_calls_counts_exploration_and_proxy_tokens(self) -> None:
        records = [
            {"arm": "GREEDY", "purpose": "incumbent_rollout", "input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "total_tokens": 15, "billed_usd": 0.1},
            {"arm": "BON-2", "purpose": "exploration_rollout", "input_tokens": 20, "cached_input_tokens": 0, "output_tokens": 5, "total_tokens": 25, "billed_usd": 0.2},
            {"arm": "BON-2", "purpose": "proxy_score", "input_tokens": 30, "cached_input_tokens": 0, "output_tokens": 5, "total_tokens": 35, "billed_usd": 0.3},
        ]
        totals = summarize_calls(records, "BON-2")
        self.assertEqual(totals.calls, 2)
        self.assertEqual(totals.exploration_tokens, 25)
        self.assertEqual(totals.proxy_tokens, 35)
        self.assertEqual(totals.total_tokens, 60)

    def test_report_writes_summary_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            append_jsonl(
                run_dir / "calls.jsonl",
                {
                    "run_id": "r",
                    "arm": "GREEDY",
                    "domain": "airline",
                    "task_id": "0",
                    "decision_idx": 0,
                    "component": None,
                    "purpose": "incumbent_rollout",
                    "model": "openai/gpt-5-mini",
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 50,
                    "total_tokens": 150,
                    "billed_usd": 0.0001,
                },
            )
            append_jsonl(
                run_dir / "simulations.jsonl",
                {
                    "run_id": "r",
                    "arm": "GREEDY",
                    "domain": "airline",
                    "task_id": "0",
                    "reward_info": {"reward": 1.0},
                },
            )
            payload = build_report(run_dir, ("GREEDY",))
            self.assertEqual(payload["status"], "pilot_evidence_only_not_k_r1")
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertIn("not a K-R1", (run_dir / "report.md").read_text())
            loaded = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(loaded["arms"][0]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
