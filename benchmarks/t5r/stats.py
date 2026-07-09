from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Interval | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return Interval(max(0.0, center - radius), min(1.0, center + radius))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty sample")
    ordered = sorted(values)
    position = min(max(probability, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_mean(
    differences: Iterable[float],
    *,
    seed: int = 300,
    draws: int = 10_000,
) -> dict[str, float | int] | None:
    values = [float(value) for value in differences]
    if not values:
        return None
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    ]
    return {
        "pairs": len(values),
        "mean": statistics.mean(values),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "draws": draws,
        "seed": seed,
    }


def brier_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float | None:
    pairs = list(zip(probabilities, outcomes))
    if not pairs:
        return None
    return statistics.mean((float(probability) - int(outcome)) ** 2 for probability, outcome in pairs)
