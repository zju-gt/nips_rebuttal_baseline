from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from baseline_registry import DOMAIN_ORDER


SEED = 300
VERSION = "base-sha256-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expected_split(tau2_repo: Path, domain: str) -> dict[str, Any]:
    if domain not in DOMAIN_ORDER:
        raise ValueError(f"unsupported domain: {domain}")
    source = tau2_repo / "data" / "tau2" / "domains" / domain
    tasks_path = source / "tasks.json"
    splits_path = source / "split_tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not isinstance(splits, dict):
        raise ValueError(f"invalid tau2 task data for {domain}")
    task_ids = [str(task["id"]) for task in tasks]
    base_ids = [str(item) for item in splits["base"]]
    if len(task_ids) != len(set(task_ids)) or len(base_ids) != len(set(base_ids)):
        raise ValueError(f"duplicate task IDs for {domain}")
    if not set(base_ids) <= set(task_ids):
        raise ValueError(f"base split contains missing tasks for {domain}")
    ordered = sorted(
        base_ids,
        key=lambda task_id: (
            hashlib.sha256(f"{SEED}:{domain}:{task_id}".encode()).hexdigest(),
            task_id,
        ),
    )
    n_train = (7 * len(ordered) + 5) // 10
    return {
        "version": VERSION,
        "seed": SEED,
        "domain": domain,
        "task_source_sha256": sha256_file(tasks_path),
        "base_source_sha256": sha256_file(splits_path),
        "base_ids": base_ids,
        "train_ids": ordered[:n_train],
        "val_ids": ordered[n_train:],
    }


def write_or_verify_split(tau2_repo: Path, split_dir: Path, domain: str) -> Path:
    expected = expected_split(tau2_repo, domain)
    path = split_dir / f"{domain}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"existing split differs from current data or algorithm: {path}")
    else:
        split_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_split(path: Path, *, domain: str, tau2_repo: Path) -> dict[str, Any]:
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_split(tau2_repo, domain)
    if actual != expected:
        raise ValueError(f"split does not match base tasks: {path}")
    return actual
