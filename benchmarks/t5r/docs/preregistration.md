# T5-R Tau Preregistration

**Version:** 0.1.0
**Status:** frozen before any measured development or pilot task
**Outcome scope:** pilot evidence only; no K-R1 outcome

## Workload

- Tau `v1.0.0`, commit `17e07b1da2bbc0cadfddeea36412686e0604127b`.
- Text-mode base tasks in airline and retail.
- Official Tau runner, environment, evaluator, and user simulator.
- Seed `300`.

Development-only tasks:

- `airline:49`
- `retail:113`

Pilot tasks:

- Airline: `0, 22, 27, 31, 33, 38, 47, 48`
- Retail: `12, 15, 17, 20, 21, 31, 42, 47, 54, 64, 73, 74, 76, 80, 110, 111`

All other airline and retail base tasks are adjudication-reserved.

## Models

- Actor and user: `openai/gpt-5-mini-2025-08-07`
- Proxy: `openai/gpt-5-nano-2025-08-07`

Every response records the provider-resolved model. A successful call without usage or resolved model identity invalidates the smoke and blocks the pilot.

## Certified Parameters

- `theta = 0.4`
- `y* = 0.2`
- fleet `delta = 0.1`
- exploration `epsilon = 0.35`
- `p_fresh = 1`
- `K_max = 108`
- `n_floor = 4`
- T8 features `(min(turn,8)/8, mutation_seen)`
- T8 metric L1 with declared `L = 1`

`L=1` is a declared modulus, not an empirically established fact.

## Tuning and Confirmation

The one-trial tuning grid is:

- BON: `2, 4`
- EPS: `0.15, 0.35, 0.60`
- RAW margin: `0.05, 0.15`
- RAW retirement window: `4, 8`
- CERT budget rate: `0.20, 0.35`

Champion selection maximizes success and breaks ties by lower gross price-sheet cost. Three additional trials are run only for selected family champions, GREEDY, and CERT. The selected tuning trial is carried into the four-trial confirmation artifact.

## Adjudication Gate

Proceed only if all conditions hold:

1. Best exploration success exceeds GREEDY.
2. Median exploration rollout is at least 1,000 billed tokens.
3. Provider dashboard cost reconciles to the price-sheet ledger within 2%.
4. Every configured arm, task, and trial has a result and every successful call has usage.
5. Projected adjudication is no more than 80% of remaining awarded credits.

Parameters and analysis must be frozen before any reserved task is touched.

## Publication Rule

Positive, negative, censored, and refused outcomes will be released. A failed gate is reported as pilot evidence only. Suite shopping and post-outcome split changes are prohibited.
