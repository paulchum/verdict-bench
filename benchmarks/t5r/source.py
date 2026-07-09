from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.t5r.config import (
    DEFAULT_SOURCE_COMMIT,
    DEFAULT_SOURCE_REF,
    DEFAULT_SOURCE_REPO,
    DEFAULT_TAU_PATCH,
    DEFAULT_TAU_PATCH_SHA256,
    SCHEMA_VERSION,
    SOURCE_URLS,
)


@dataclass(frozen=True)
class SourceLock:
    repo: str
    ref: str
    commit: str
    patch_path: str
    patch_sha256: str
    source_urls: dict[str, str]
    schema_version: str = SCHEMA_VERSION

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _git_succeeds(args: list[str], cwd: Path) -> bool:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    ).returncode == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _working_diff_sha256(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def ensure_tau2_source(
    target_dir: Path,
    repo: str = DEFAULT_SOURCE_REPO,
    ref: str = DEFAULT_SOURCE_REF,
    commit: str = DEFAULT_SOURCE_COMMIT,
    patch_path: Path = DEFAULT_TAU_PATCH,
    patch_sha256: str = DEFAULT_TAU_PATCH_SHA256,
    *,
    allow_clone: bool = True,
) -> SourceLock:
    """Ensure the pinned tau2-bench checkout exists and matches the lock."""
    target_dir = target_dir.resolve()
    if not (target_dir / ".git").exists():
        if not allow_clone:
            raise FileNotFoundError(f"Missing tau2-bench checkout: {target_dir}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--depth", "1", "--branch", ref, repo, str(target_dir)])

    actual = _run_git(["rev-parse", "HEAD"], cwd=target_dir)
    if actual != commit:
        raise RuntimeError(
            f"tau2-bench checkout mismatch at {target_dir}: expected {commit}, got {actual}"
        )

    patch_path = patch_path.resolve()
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing pinned Tau patch: {patch_path}")
    actual_patch_sha256 = _sha256(patch_path)
    if actual_patch_sha256 != patch_sha256:
        raise RuntimeError(
            "Tau patch hash mismatch: "
            f"expected {patch_sha256}, got {actual_patch_sha256}"
        )

    if _git_succeeds(["apply", "--check", str(patch_path)], target_dir):
        _run_git(["apply", str(patch_path)], cwd=target_dir)
    elif not _git_succeeds(
        ["apply", "--reverse", "--check", str(patch_path)], target_dir
    ):
        raise RuntimeError(
            f"Tau checkout has changes inconsistent with {patch_path.name}"
        )

    if _working_diff_sha256(target_dir) != actual_patch_sha256:
        raise RuntimeError("Tau checkout contains changes beyond the pinned patch")

    return SourceLock(
        repo=repo,
        ref=ref,
        commit=actual,
        patch_path=patch_path.name,
        patch_sha256=actual_patch_sha256,
        source_urls=SOURCE_URLS,
    )
