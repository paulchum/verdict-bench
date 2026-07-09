# Provenance

## Workload

- Repository: `https://github.com/sierra-research/tau2-bench.git`
- Tag: `v1.0.0`
- Commit: `17e07b1da2bbc0cadfddeea36412686e0604127b`
- Compatibility patch: `benchmarks/t5r/patches/tau2-v1.0.0-text-only-imports.patch`
- Patch SHA-256: `6bbe50a97fc5c88951bf475a97893b93f1bc98f205483d3b4015b8513513ef61`

`verdict-bench doctor` rejects a different commit, patch, or working diff.

## Models

- `openai/gpt-5-mini-2025-08-07` for actor and official Tau user simulator calls.
- `openai/gpt-5-nano-2025-08-07` for visible-state proxy calls.

Every successful call must record its provider-resolved model. Model aliases are not used in measured runs.

## Prices

Price manifests are pinned from the official model pages at freeze time. The 2026-07-09 release manifest records per-million-token input, cached-input, and output prices. Provider dashboard reconciliation remains mandatory for measured evidence.

## Synthetic Release

`benchmarks/t5r/public_artifacts/sim-v0.1.0` is generated deterministically by `python -m benchmarks.t5r.fixtures`. It contains no provider calls and is labeled `[SIM]` throughout.
