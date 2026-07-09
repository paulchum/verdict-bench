# Upstream Tau Compatibility Contribution

Tau `v1.0.0` imports optional voice and knowledge modules while loading the text runner. A base text installation can therefore fail on missing `scipy`, `websockets`, or `rank_bm25` before an airline or retail task starts.

The included patch:

- makes voice package exports lazy;
- imports `scipy.signal.resample_poly` only when resampling is requested;
- registers optional voice and knowledge components only when dependencies exist;
- moves voice-only runner imports into voice execution paths;
- keeps the official text runner, registry, user simulator, and evaluator intact;
- fixes an interpreter-shutdown edge in the event-loop destructor wrapper.

`verdict-bench doctor` applies and verifies the patch. The same focused change is intended for an upstream Tau pull request; the local patch remains until a release containing the fix is pinned.
