# T5-R Tau3 Evidence Report

**Status:** [SIM] synthetic_fixture_not_empirical

This is not a K-R1 adjudication. Only a separately frozen adjudication can issue PASS, FAIL, or VOID.

## Workload Validity

- GREEDY success: 0.5417
- Best exploring arm: CERT
- GREEDY-to-exploring delta: 0.1250
- Median exploration rollout tokens: 1757.0000
- Flags: none

## Arm Results

| Arm | Results | Success (95% CI) | Gross USD | Explore USD | Tokens | Explore share | Refusals | Retirements | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GREEDY | 24 | 0.5417 [0.3507, 0.7211] | 0.0176 | 0 | 29436 | 0.0000 | 0 | 0 | 0 |
| BON-2 | 24 | 0.5833 [0.3883, 0.7553] | 0.0428 | 0.0231 | 77708 | 0.4957 | 0 | 0 | 0 |
| BON-4 | 24 | 0.6250 [0.4271, 0.7884] | 0.0449 | 0.0242 | 81308 | 0.4958 | 0 | 0 | 0 |
| EPS-0.35 | 24 | 0.5833 [0.3883, 0.7553] | 0.0471 | 0.0253 | 84908 | 0.4960 | 0 | 0 | 0 |
| RAW | 24 | 0.6250 [0.4271, 0.7884] | 0.0493 | 0.0263 | 88508 | 0.4962 | 0 | 0 | 0 |
| CERT | 24 | 0.6667 [0.4671, 0.8203] | 0.0514 | 0.0274 | 92108 | 0.4963 | 24 | 0 | 0 |

## Evidence Health

- Meter: [SIM]_reconciled_within_2pct
- Completion: False
- T8 refused keys: 9
- Proxy audit observations: 48
- Publication gate: pilot_evidence_only_do_not_adjudicate
