from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = "1.0.0"
BENCHMARK_VERSION = "0.1.0"

DEFAULT_SOURCE_REPO = "https://github.com/sierra-research/tau2-bench.git"
DEFAULT_SOURCE_REF = "v1.0.0"
DEFAULT_SOURCE_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"

DEFAULT_SEED = 300
DEFAULT_ACTOR_MODEL = "openai/gpt-5-mini-2025-08-07"
DEFAULT_PROXY_MODEL = "openai/gpt-5-nano-2025-08-07"
DEFAULT_MAX_STEPS = 80
DEFAULT_TRIALS = 1

DOMAINS = ("airline", "retail")
DEVELOPMENT_TASKS = {"airline": ["49"], "retail": ["113"]}
PILOT_TASK_COUNTS = {"airline": 8, "retail": 16}
COMPONENTS = ("policy_first", "tool_progress", "verify_then_commit")

RUN_ROOT = Path("benchmarks/t5r/runs")
DEFAULT_EXTERNAL_DIR = Path(".external/tau2-bench")
DEFAULT_TAU_PATCH = Path(
    "benchmarks/t5r/patches/tau2-v1.0.0-text-only-imports.patch"
)
DEFAULT_TAU_PATCH_SHA256 = (
    "6bbe50a97fc5c88951bf475a97893b93f1bc98f205483d3b4015b8513513ef61"
)

DEFAULT_ARMS = ("GREEDY", "BON-2", "BON-4", "EPS-0.35", "RAW", "CERT")
OPTIONAL_ARMS = ("CERT-EI",)
EXCLUDED_PUBLIC_ARMS = ("HEURISTIC-v0",)

# Pilot-frozen defaults from the T5K/T4B/T8 deployment sheet.
THETA = 0.4
Y_STAR = 0.2
FLEET_DELTA = 0.1
EXPLORATION_EPSILON = 0.35
P_FRESH = 1.0
K_MAX = 108
DECLARED_MODULUS_L = 1.0
W_TRIAL_CAP = 80

TURN_BUCKETS = ((0, 2, "early"), (3, 5, "middle"), (6, None, "late"))

MUTATING_TOOLS = frozenset(
    {
        # Airline
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
        # Retail
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)

SOURCE_URLS = {
    "tau2_bench": "https://github.com/sierra-research/tau2-bench",
    "tau2_release": "https://github.com/sierra-research/tau2-bench/releases/tag/v1.0.0",
    "tau2_cli_docs": "https://github.com/sierra-research/tau2-bench/blob/main/docs/cli-reference.md",
    "tau2_submission": "https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md",
    "openai_actor": "https://developers.openai.com/api/docs/models/gpt-5-mini",
    "openai_proxy": "https://developers.openai.com/api/docs/models/gpt-5-nano",
}
