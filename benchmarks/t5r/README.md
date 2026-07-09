# Verdict Bench

Verdict Bench is a public research harness for certificate-bearing routing and evidence refusal in tool-using language-model agents. It applies a posterior-free, anytime-valid control stack to the public Tau benchmark and records enough evidence to audit every candidate decision.

**Current release status:** implementation and preregistration only. The included results are deterministic `[SIM]` fixtures for CI and interface testing. They are not measured benchmark performance.

## What It Tests

Verdict Bench compares six declared agent-control arms:

- `GREEDY`
- `BON-2`
- `BON-4`
- `EPS-0.35`
- `RAW`
- `CERT`

`CERT-EI` is an optional comparator. The superseded heuristic is retained only as the provenance label `HEURISTIC-v0`; it cannot execute and is excluded from public headline tables.

The certified arm logs pairwise accept/refute evidence, fixed-scale W trials, component-ledger evidence, e-BH retirement thresholds, T8 context status, and every evidence refusal. Proxy scoring uses visible policy, conversation, and tool state only. Hidden Tau evaluation criteria are available only to Tau's offline evaluator after the trajectory is complete.

## Source and Model Locks

- Tau: `sierra-research/tau2-bench` tag `v1.0.0`
- Commit: `17e07b1da2bbc0cadfddeea36412686e0604127b`
- Actor and official user simulator: `gpt-5-mini-2025-08-07`
- Proxy: `gpt-5-nano-2025-08-07`

The repository includes a SHA-256-locked import-only patch that prevents Tau text-mode imports from requiring optional voice and knowledge dependencies. `verdict-bench doctor` verifies both the commit and the exact working diff.

## No-Spend Safety

`doctor`, `freeze`, `report`, and `export-tau` make no model calls.

`smoke` and `pilot` refuse to start unless both conditions are true:

```bash
export VERDICT_BENCH_FUNDED=1
export OPENAI_API_KEY=...
```

They also require a positive `--credit-budget-usd`. Smoke is capped at $10 in credits. Pilot is capped at the lower of $100 or 20% of the declared awarded credits. This project authorizes $0 of cash-funded API spend before credits are confirmed.

## Commands

```bash
verdict-bench doctor --fetch-source
verdict-bench freeze --fetch-source
verdict-bench smoke --credit-budget-usd 10
verdict-bench pilot --stage tune --awarded-credits-usd 1000 --credit-budget-usd 100
verdict-bench pilot --stage confirm --tuning-run PATH --awarded-credits-usd 1000 --credit-budget-usd 100
verdict-bench report RUN_DIR --provider-billed-usd AMOUNT
verdict-bench export-tau RUN_DIR
```

## Fixed Pilot Split

Seed `300` selects 8 airline and 16 retail base tasks. `airline:49` and `retail:113` are development-only and never enter pilot or adjudication. All remaining airline and retail base tasks are reserved for adjudication.

## Reporting Boundary

Pilot reports include success with 95% intervals, token classes, gross price-sheet cost, exploration cost, paired bootstrap savings, proxy audit calibration, T8 statuses, refusals, retirements, task failures, completion, and dashboard meter reconciliation.

Pilot output can only recommend whether adjudication may be frozen. It can never issue K-R1 `PASS`, `FAIL`, or `VOID`. Negative and refused outcomes remain publishable under the no-suite-shopping rule.

## Synthetic Trace Explorer

The release includes a static explorer generated from the public `[SIM]` artifact. It exposes candidate generation, proxy outcomes, call-level tokens and spend, certificates, and refusals. Every synthetic status is visibly marked `[SIM]`.

## Verification

```bash
python -m unittest discover -s tests -v
python -m benchmarks.t5r.fixtures
```

CI checks source locks, golden parity vectors, split determinism, billing, schemas, budget breaking, hidden-criteria isolation, report reproducibility, the static explorer, secret scanning, and an offline Tau mock trajectory through the official runner.

## Limitations

- No measured Tau result is included in `v0.1.0`.
- The T8 modulus `L=1` is declared, not empirically proven.
- Seeded Bernoulli audit calibration is not calibration to hidden Tau criteria.
- A Tau leaderboard submission from this code is a custom modified-scaffold submission, not a standard verified entry.
- One external reproduction is required before the result is described as externally validated.

## License

Apache License 2.0. Tau remains subject to its own repository license and terms.
