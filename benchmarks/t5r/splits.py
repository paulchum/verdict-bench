from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.t5r.config import (
    DEFAULT_SEED,
    DEVELOPMENT_TASKS,
    DOMAINS,
    PILOT_TASK_COUNTS,
    SCHEMA_VERSION,
)
from benchmarks.t5r.source import SourceLock


@dataclass(frozen=True)
class DomainSplit:
    development: list[str]
    pilot: list[str]
    adjudication_reserved: list[str]


@dataclass(frozen=True)
class TaskSplits:
    seed: int
    domains: dict[str, DomainSplit]
    source: SourceLock | None = None

    def write(self, path: Path) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "domains": {name: asdict(split) for name, split in self.domains.items()},
            "source": asdict(self.source) if self.source is not None else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")


def stable_task_sort_key(task_id: str) -> tuple[int, str]:
    return (len(task_id), task_id)


def sample_ids(ids: list[str], count: int, rng: random.Random) -> list[str]:
    if count > len(ids):
        raise ValueError(f"Cannot sample {count} tasks from {len(ids)} ids")
    return sorted(rng.sample(sorted(ids, key=stable_task_sort_key), count), key=stable_task_sort_key)


def load_domain_task_ids(source_dir: Path, domain: str) -> list[str]:
    task_file = source_dir / "data" / "tau2" / "domains" / domain / "tasks.json"
    tasks = json.loads(task_file.read_text())
    return [str(task["id"]) for task in tasks]


def build_task_splits(
    source_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    source: SourceLock | None = None,
) -> TaskSplits:
    rng = random.Random(seed)
    domains: dict[str, DomainSplit] = {}
    for domain in DOMAINS:
        ids = sorted(load_domain_task_ids(source_dir, domain), key=stable_task_sort_key)
        pilot = sample_ids(ids, PILOT_TASK_COUNTS[domain], rng)
        pilot_set = set(pilot)
        development = DEVELOPMENT_TASKS[domain]
        if any(task_id in pilot_set for task_id in development):
            raise ValueError(f"Development tasks overlap pilot for {domain}")
        reserved = [
            task_id
            for task_id in ids
            if task_id not in pilot_set and task_id not in set(development)
        ]
        domains[domain] = DomainSplit(
            development=list(development),
            pilot=pilot,
            adjudication_reserved=reserved,
        )
    return TaskSplits(seed=seed, domains=domains, source=source)
