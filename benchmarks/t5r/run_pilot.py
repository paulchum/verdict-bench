"""Compatibility entry point; prefer ``python -m benchmarks.t5r``."""

from benchmarks.t5r.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
