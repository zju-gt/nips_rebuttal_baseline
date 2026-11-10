from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from api_config import load_api_config


BASELINE_REPOSITORIES = (
    "sop-world-react",
    "sop-world-flexion",
    "sop-world-AWM",
    "sop-world-Reasoning",
    "sop-world-EvoTest",
    "sop-world",
)

TAU2_DOMAIN_FILES = {
    "airline": ("db.json", "policy.md", "split_tasks.json", "tasks.json"),
    "retail": ("db.json", "policy.md", "split_tasks.json", "tasks.json"),
    "telecom": (
        "db.toml",
        "user_db.toml",
        "main_policy.md",
        "main_policy_solo.md",
        "tech_support_manual.md",
        "tech_support_workflow.md",
        "tech_support_workflow_solo.md",
        "split_tasks.json",
        "tasks.json",
    ),
}

TAU2_METADATA_FILES = ("LICENSE", "README.md", "pyproject.toml", "uv.lock")

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".test-artifacts",
    "__pycache__",
    "analysis",
    "logs",
    "results",
    "tests",
}


def _is_excluded(path: Path) -> bool:
    lowered_parts = tuple(part.lower() for part in path.parts)
    return (
        any(part in EXCLUDED_DIRECTORY_NAMES for part in lowered_parts)
        or any(part.startswith("test-output-") for part in lowered_parts)
        or path.suffix.lower() == ".pyc"
        or path.name.lower() == ".env"
        or path.name.lower().startswith(".env.")
    )


def _reject_symlink_chain(path: Path, boundary: Path) -> None:
    if path != boundary and boundary not in path.parents:
        raise ValueError(f"source path is outside its repository: {path}")
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"symlink is not allowed in bundled source: {current}")
        if current == boundary:
            return
        current = current.parent


def _require_regular_file(path: Path, boundary: Path) -> None:
    _reject_symlink_chain(path, boundary)
    if not path.is_file():
        raise ValueError(f"required source file is missing: {path}")


def _regular_files(
    root: Path,
    boundary: Path,
    allowed_suffixes: frozenset[str] | None = None,
) -> list[Path]:
    _reject_symlink_chain(root, boundary)
    if not root.is_dir():
        raise ValueError(f"required source directory is missing: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in bundled source: {path}")
        if path.is_dir() or _is_excluded(path):
            continue
        if not path.is_file():
            raise ValueError(f"non-regular source entry is not allowed: {path}")
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        files.append(path)
    return files


def _copy_tree(
    source: Path,
    destination: Path,
    boundary: Path,
    allowed_suffixes: frozenset[str] | None = None,
) -> None:
    for source_file in _regular_files(source, boundary, allowed_suffixes):
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def _copy_regular_file(source: Path, destination: Path, boundary: Path) -> None:
    _require_regular_file(source, boundary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _validate_sources(source_workspace: Path) -> None:
    for name in BASELINE_REPOSITORIES:
        repo = source_workspace / name
        _regular_files(repo / "src", source_workspace, frozenset({".py"}))
        _regular_files(repo / "prompts", source_workspace)
        _require_regular_file(repo / "src" / "tau2_runner.py", source_workspace)
    tau2 = source_workspace / "tau2-bench"
    _regular_files(tau2 / "src", source_workspace)
    _regular_files(
        tau2 / "data" / "tau2" / "user_simulator",
        source_workspace,
    )
    for filename in TAU2_METADATA_FILES:
        _require_regular_file(tau2 / filename, source_workspace)
    for domain, filenames in TAU2_DOMAIN_FILES.items():
        for filename in filenames:
            path = tau2 / "data" / "tau2" / "domains" / domain / filename
            _require_regular_file(path, source_workspace)


def _build_bundle(source_workspace: Path, build_dir: Path) -> None:
    for name in BASELINE_REPOSITORIES:
        source = source_workspace / name
        _copy_tree(
            source / "src",
            build_dir / name / "src",
            source_workspace,
            frozenset({".py"}),
        )
        _copy_tree(
            source / "prompts",
            build_dir / name / "prompts",
            source_workspace,
        )

    tau2_source = source_workspace / "tau2-bench"
    tau2_target = build_dir / "tau2-bench"
    _copy_tree(tau2_source / "src", tau2_target / "src", source_workspace)
    _copy_tree(
        tau2_source / "data" / "tau2" / "user_simulator",
        tau2_target / "data" / "tau2" / "user_simulator",
        source_workspace,
    )
    for filename in TAU2_METADATA_FILES:
        _copy_regular_file(
            tau2_source / filename,
            tau2_target / filename,
            source_workspace,
        )
    for domain, filenames in TAU2_DOMAIN_FILES.items():
        for filename in filenames:
            source_file = tau2_source / "data" / "tau2" / "domains" / domain / filename
            target = tau2_target / "data" / "tau2" / "domains" / domain / filename
            _copy_regular_file(source_file, target, source_workspace)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo: Path) -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1", "-uall"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head or None, bool(status)


def _component_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(
            f"{record['path']}\0{record['sha256']}\0{record['size']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _write_manifest(build_dir: Path, source_workspace: Path) -> Path:
    file_records: list[dict[str, Any]] = []
    for path in sorted(build_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(build_dir).as_posix()
        component = PurePosixPath(relative).parts[0]
        file_records.append(
            {
                "path": relative,
                "component": component,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    file_records.sort(key=lambda item: str(item["path"]))
    components: dict[str, dict[str, Any]] = {}
    for name in (*BASELINE_REPOSITORIES, "tau2-bench"):
        records = [item for item in file_records if item["component"] == name]
        head, dirty = _git_provenance(source_workspace / name)
        components[name] = {
            "source_path": str((source_workspace / name).resolve()),
            "git_head": head,
            "dirty": dirty,
            "content_sha256": _component_hash(records),
        }
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "files": file_records,
        "total_files": len(file_records),
        "total_bytes": sum(int(item["size"]) for item in file_records),
    }
    manifest_path = build_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _audit_secret(build_dir: Path, secret: str) -> None:
    encoded = secret.encode("utf-8")
    for path in build_dir.rglob("*"):
        if path.is_file() and encoded in path.read_bytes():
            raise ValueError("bundle contains a credential value")


def sync_vendor_bundle(
    source_workspace: Path,
    code_dir: Path,
    openai_test_path: Path,
) -> Path:
    source_workspace = source_workspace.expanduser().resolve()
    code_dir = code_dir.expanduser().resolve()
    openai_test_path = openai_test_path.expanduser().resolve()
    _validate_sources(source_workspace)
    api_config = load_api_config(openai_test_path, {})
    code_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix=".vendor-build-", dir=code_dir))
    vendor_dir = code_dir / "vendor"
    backup_dir: Path | None = None
    try:
        _build_bundle(source_workspace, build_dir)
        _audit_secret(build_dir, api_config.api_key)
        _write_manifest(build_dir, source_workspace)
        verify_vendor_bundle(build_dir)
        try:
            if vendor_dir.exists():
                backup_dir = Path(
                    tempfile.mkdtemp(prefix=".vendor-backup-", dir=code_dir)
                )
                backup_dir.rmdir()
                vendor_dir.rename(backup_dir)
            build_dir.rename(vendor_dir)
        except BaseException:
            if backup_dir is not None and backup_dir.exists() and not vendor_dir.exists():
                backup_dir.rename(vendor_dir)
            raise
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)
        return vendor_dir / "bundle_manifest.json"
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)


def verify_vendor_bundle(vendor_dir: Path) -> dict[str, Any]:
    vendor_dir = vendor_dir.expanduser().resolve()
    manifest_path = vendor_dir / "bundle_manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("bundle manifest is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("bundle manifest must contain a JSON object")
    records = value.get("files")
    if not isinstance(records, list):
        raise ValueError("bundle manifest files must be a list")
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("bundle manifest contains an invalid file record")
        raw_path = record["path"]
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("bundle manifest contains an unsafe path")
        normalized = relative.as_posix()
        if normalized in expected:
            raise ValueError("bundle manifest contains a duplicate path")
        expected[normalized] = record

    actual: dict[str, Path] = {}
    for path in vendor_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle contains a symlink")
        if not path.is_file() or path == manifest_path:
            continue
        actual[path.relative_to(vendor_dir).as_posix()] = path
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError("bundle is missing a manifest-listed file")
    unlisted = sorted(set(actual) - set(expected))
    if unlisted:
        raise ValueError("bundle contains an unlisted file")

    for relative, record in expected.items():
        path = actual[relative]
        if record.get("size") != path.stat().st_size:
            raise ValueError("bundle file size mismatch")
        if record.get("sha256") != _sha256(path):
            raise ValueError("bundle file hash mismatch")
        if record.get("component") != PurePosixPath(relative).parts[0]:
            raise ValueError("bundle file component mismatch")
    if value.get("total_files") != len(expected):
        raise ValueError("bundle total file count mismatch")
    if value.get("total_bytes") != sum(path.stat().st_size for path in actual.values()):
        raise ValueError("bundle total byte count mismatch")

    components = value.get("components")
    if not isinstance(components, dict):
        raise ValueError("bundle manifest components must be an object")
    for name, provenance in components.items():
        if not isinstance(provenance, dict):
            raise ValueError("bundle component provenance is invalid")
        component_records = [
            record for record in records if record.get("component") == name
        ]
        if provenance.get("content_sha256") != _component_hash(component_records):
            raise ValueError("bundle component hash mismatch")
    return value
