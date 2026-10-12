"""Portable SHA-256 provenance manifests for empirical benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _validate_root(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Evidence root must be an existing non-symlink directory.")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden in portable evidence bundles.")


def create_evidence_manifest(
    directory: str | Path,
    *,
    git_sha: str,
    experiment_config: Mapping[str, object],
) -> dict[str, object]:
    """Write an immutable manifest of files, revision and canonical configuration."""
    root = Path(directory)
    _validate_root(root)
    if not _GIT_SHA.fullmatch(git_sha):
        raise ValueError("A full lowercase 40-character Git SHA is required.")
    manifest_path = root / "evidence_provenance.json"
    if manifest_path.exists():
        raise FileExistsError("Evidence provenance already exists.")
    config_json = json.dumps(
        experiment_config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in _files(root)
    ]
    report = {
        "schema_version": 1,
        "git_sha": git_sha,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "experiment_config": json.loads(config_json),
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def verify_evidence_manifest(directory: str | Path) -> dict[str, object]:
    """Verify complete bundle membership and byte-for-byte artifact integrity."""
    root = Path(directory)
    _validate_root(root)
    manifest_path = root / "evidence_provenance.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not _GIT_SHA.fullmatch(
        str(report.get("git_sha", ""))
    ):
        raise ValueError("Invalid evidence provenance schema or Git SHA.")
    config_json = json.dumps(
        report["experiment_config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if report.get("config_sha256") != config_hash:
        raise ValueError("Experiment configuration checksum mismatch.")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in _files(root)
        if path != manifest_path
    }
    expected = report["files"]
    if not isinstance(expected, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not _SHA256.fullmatch(str(item.get("sha256", "")))
        or type(item.get("bytes")) is not int
        for item in expected
    ):
        raise ValueError("Malformed evidence file entries.")
    paths = [item["path"] for item in expected]
    if len(set(paths)) != len(paths) or set(paths) != set(actual):
        raise ValueError("Evidence bundle membership mismatch.")
    for item in expected:
        path = actual[item["path"]]
        if path.stat().st_size != item["bytes"]:
            raise ValueError("Evidence file size mismatch.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Evidence file checksum mismatch.")
    return report
