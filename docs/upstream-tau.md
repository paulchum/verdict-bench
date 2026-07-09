# Upstream Tau Compatibility Contribution

Tau `v1.0.0` imports optional voice and knowledge modules while loading the text runner. A base text installation can therefore fail on missing `scipy`, `websockets`, or `rank_bm25` before an airline or retail task starts.

The included patch:

- makes voice package exports lazy;
- imports `scipy.signal.resample_poly` only when resampling is requested;
- registers optional voice and knowledge components only when dependencies exist;
- moves voice-only runner imports into voice execution paths;
- keeps the official text runner, registry, user simulator, and evaluator intact;
- fixes an interpreter-shutdown edge in the event-loop destructor wrapper.

Tau fixed the mainline optional-dependency boundary after `v1.0.0` in
[PR #197](https://github.com/sierra-research/tau2-bench/pull/197). The pinned tag
still reproduces the missing-SciPy failure, so `verdict-bench doctor` applies and
verifies this compatibility backport until a newer Tau release is frozen.

Verdict Layer contributed
[PR #391](https://github.com/sierra-research/tau2-bench/pull/391), a subprocess
regression that blocks `scipy` and `rank_bm25` while importing the text runner.
This guards the mainline fix without duplicating it.
