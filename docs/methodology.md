# Methodology and Reporting

## Primary Outcomes

- Tau offline success rate and Wilson 95% interval.
- Gross price-sheet cost and provider dashboard reconciliation.
- Input, cached input, output, reasoning, and total tokens.
- Exploration tokens, share, and median exploration rollout length.
- GREEDY-to-exploration success delta.
- Cheapest baseline within the frozen 5-point matched-quality margin.
- Paired task-level bootstrap cost savings and success delta.
- Proxy Bernoulli audit calibration.
- T8 licensed and refused key counts.
- Refusals, retirements, task failures, and completion matrix.

## Metering

Every actor, user, rollout, proxy, and error call is logged. Successful calls require usage, request identity, requested and resolved model, token classes, gross price-sheet cost, provider-reported call cost when available, latency, and context tags.

Call-level provider cost estimates do not satisfy the meter gate. `verdict-bench report --provider-billed-usd` records the project billing export total and computes the final discrepancy.

## Quality Isolation

Proxy prompts contain visible policy, recent visible conversation, candidate content, and visible tools. They do not receive task evaluation criteria. Tau's `reward_info.reward` is attached only by the official offline evaluator after the trajectory ends.

## Leaderboard Boundary

Verdict Bench changes prompts and orchestration. Any leaderboard package must set `submission_type` to `custom`, disclose modified prompts and control flow, include complete trajectories, and never describe itself as a standard verified submission.
