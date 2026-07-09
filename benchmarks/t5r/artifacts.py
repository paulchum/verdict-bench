from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from benchmarks.t5r.config import SCHEMA_VERSION
from benchmarks.t5r.ledger import read_jsonl


def generate_trace_site(run_dir: Path, output_dir: Path) -> Path:
    """Build a static explorer entirely from released run artifacts."""
    template_dir = Path(__file__).with_name("site")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(template_dir / name, output_dir / name)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "summary": _read_json(run_dir / "summary.json"),
        "manifest": _read_json(run_dir / "manifest.json"),
        "calls": read_jsonl(run_dir / "calls.jsonl"),
        "decisions": read_jsonl(run_dir / "decisions.jsonl"),
        "events": read_jsonl(run_dir / "events.jsonl"),
        "simulations": read_jsonl(run_dir / "simulations.jsonl"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    (output_dir / "data.json").write_text(serialized + "\n", encoding="utf-8")
    (output_dir / "data.js").write_text(
        f"window.__VERDICT_TRACE_DATA__={serialized};\n", encoding="utf-8"
    )
    return output_dir / "index.html"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
