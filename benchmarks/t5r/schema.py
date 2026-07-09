from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.t5r.ledger import read_jsonl


SCHEMA_FILES = {
    "calls.jsonl": "call-1.0.0.schema.json",
    "events.jsonl": "event-1.0.0.schema.json",
    "decisions.jsonl": "decision-1.0.0.schema.json",
    "simulations.jsonl": "task-result-1.0.0.schema.json",
    "manifest.json": "manifest-1.0.0.schema.json",
    "source_lock.json": "source-lock-1.0.0.schema.json",
    "model_manifest.json": "model-manifest-1.0.0.schema.json",
    "price_sheet.json": "price-sheet-1.0.0.schema.json",
}


def validate_run_artifacts(run_dir: Path) -> list[str]:
    """Validate public schemas and cross-record metering invariants."""
    from jsonschema import Draft202012Validator

    schema_dir = Path(__file__).with_name("schemas")
    errors: list[str] = []
    for artifact_name, schema_name in SCHEMA_FILES.items():
        artifact_path = run_dir / artifact_name
        if not artifact_path.exists():
            continue
        schema = json.loads((schema_dir / schema_name).read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        records: list[Any]
        if artifact_path.suffix == ".jsonl":
            records = read_jsonl(artifact_path)
        else:
            records = [json.loads(artifact_path.read_text(encoding="utf-8"))]
        for index, record in enumerate(records):
            for error in validator.iter_errors(record):
                location = ".".join(str(part) for part in error.absolute_path)
                errors.append(
                    f"{artifact_name}[{index}]"
                    + (f".{location}" if location else "")
                    + f": {error.message}"
                )

    calls = read_jsonl(run_dir / "calls.jsonl")
    successful = [record for record in calls if record.get("status") == "success"]
    for index, record in enumerate(successful):
        if int(record.get("total_tokens", 0)) <= 0:
            errors.append(f"calls.success[{index}]: missing usage")
        if not record.get("resolved_model"):
            errors.append(f"calls.success[{index}]: missing resolved model")
    physical_ids = [record.get("physical_call_id") for record in successful]
    if len(physical_ids) != len(set(physical_ids)):
        errors.append("calls.jsonl: duplicate successful physical_call_id")
    return errors
