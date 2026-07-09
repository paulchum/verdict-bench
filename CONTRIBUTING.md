# Contributing

Verdict Bench accepts focused fixes, new deterministic tests, schema improvements, and independent reproduction artifacts.

## Ground Rules

- Do not add measured results without the matching preregistration, complete trajectories, and meter evidence.
- Preserve `[SIM]` labels on all synthetic fixtures and screenshots.
- Do not expose hidden Tau evaluation criteria to proxy prompts.
- Do not filter component admissions by observed proxy score before ledger entry.
- Keep durable actions scoped to the exact declared context key.
- Describe modified Tau orchestration as a custom submission.

## Local Checks

```bash
python -m unittest discover -s tests -v
node --check benchmarks/t5r/site/app.js
verdict-bench doctor --fetch-source
```

Pull requests should explain any change to frozen parameters, schemas, source locks, or report semantics. Changes after measured outcomes are observed require a new preregistration version.
