# Formal Contract Surface

Verdict Bench adapts an existing formal control chain to a public workload. The adapter does not claim a new theorem.

## Episode Grain

One T5K episode is one fresh component candidate plus its post-generation proxy evaluation. The component label is selected and logged before generation. A proxy score `s` in `[0,1]` becomes a deterministic seeded `Bernoulli(s)` audit outcome.

## Context Key

Evidence is keyed by:

```text
(component, domain, last_event, turn_bucket, mutation_seen, config_hash, sampler_hash)
```

Turn buckets are `0-2`, `3-5`, and `6+`. Tool events are classified from a frozen read/write allowlist. Retirement is valid only inside the exact context key that generated its evidence.

## Pair Decisions

The accept process tests the one-sided null that the incumbent mean is at least the challenger mean. The refute process tests the half-null that the challenger mean is at least `(1 + incumbent_mean) / 2`. Both use GLR statistics with bound-KT clock corrections.

An unresolved pair returns `NotSeparated`. There is no heuristic margin fallback.

## Component Retirement

Fresh component outcomes feed serial fixed-scale W trials. Settled trial outcomes enter a component ledger. Retirement requires:

- at least the unconditional evidence floor;
- a ledger e-process crossing;
- execution through the declared e-BH fleet threshold;
- a context key licensed by the T8 declaration and tests.

Admissions are never filtered by score before entering the evidence stream.

## Routing

CERT routes with truncated expected improvement over the certified zone. Raw expected improvement is isolated to the declared `CERT-EI` comparator. Routing controls efficiency only; it cannot make an invalid certificate valid.

## T8 Refusals

The adapter declares features `(min(turn,8)/8, mutation_seen)`, L1 distance, and `L=1`. It emits:

- `Inhomogeneous`
- `ModulusTooCoarse`
- `EvidenceCensored`

when evidence cannot license a key. The declaration is always reported as declared, never empirically proven.
