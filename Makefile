.PHONY: doctor test sim verify

doctor:
	verdict-bench doctor --fetch-source

test:
	python -m unittest discover -s tests -v

sim:
	python -m benchmarks.t5r.fixtures

verify: test
	node --check benchmarks/t5r/site/app.js
	python -c "from pathlib import Path; from benchmarks.t5r.schema import validate_run_artifacts; errors = validate_run_artifacts(Path('benchmarks/t5r/public_artifacts/sim-v0.1.0')); assert not errors, errors"
