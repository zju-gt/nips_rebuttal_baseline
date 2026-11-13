from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train_val_split import canonical_hash, sha256_file


ARTIFACT_FILES = {
    "awm": ("workflow_memory.json",),
    "reasoningbank": ("reasoning_bank.json",),
    "living_harness": (
        "memories.json",
        "state_graph.json",
        "meta_sop.md",
        "artifacts/benchmark_prior.md",
        "artifacts/domain_prior.md",
    ),
}


def _source_for(train_dir: Path, method: str, relative: str, cycles: int) -> Path:
    if relative.startswith("artifacts/"):
        return train_dir / relative
    return train_dir / f"loop_{cycles - 1}" / relative


def export_harness(
    *,
    train_dir: Path,
    harness_dir: Path,
    method: str,
    domain: str,
    split: dict[str, Any],
    cycles: int,
    model: str,
    runner_source: Path | None = None,
) -> dict[str, Any]:
    if method not in ARTIFACT_FILES or cycles < 1:
        raise ValueError(f"invalid harness method or cycle count: {method}, {cycles}")
    if harness_dir.exists():
        raise ValueError(f"harness directory already exists: {harness_dir}")
    summary = json.loads((train_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary.get("infra_error_count", -1)) != 0:
        raise ValueError(f"training has infrastructure errors: {train_dir}")
    if int(summary.get("missing_reward_count", -1)) != 0:
        raise ValueError(f"training has missing rewards: {train_dir}")
    if int(summary.get("num_tasks", -1)) != len(split["train_ids"]):
        raise ValueError(f"training task count mismatch: {train_dir}")
    if int(summary.get("total_samples", -1)) != cycles * len(split["train_ids"]):
        raise ValueError(f"training sample count mismatch: {train_dir}")
    results = json.loads((train_dir / "results.json").read_text(encoding="utf-8"))
    task_counts = Counter(str(item["task_id"]) for item in results.get("simulations", []))
    if task_counts != Counter({task_id: cycles for task_id in split["train_ids"]}):
        raise ValueError(f"training result IDs mismatch: {train_dir}")
    sources = {name: _source_for(train_dir, method, name, cycles) for name in ARTIFACT_FILES[method]}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    for name in sources:
        if name.endswith(".json"):
            json.loads(sources[name].read_text(encoding="utf-8"))
    harness_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    for name, source in sources.items():
        target = harness_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[name] = sha256_file(target)
    manifest = {
        "method": method,
        "domain": domain,
        "model": model,
        "split_hash": canonical_hash(split),
        "train_ids_hash": canonical_hash(split["train_ids"]),
        "train_task_count": len(split["train_ids"]),
        "cycles": cycles,
        "runner_source_sha256": sha256_file(runner_source) if runner_source else None,
        "tau2_repo_commit": summary.get("tau2_repo_commit"),
        "runner_commit": summary.get("runner_commit"),
        "meta_sop_empty": not (harness_dir / "meta_sop.md").read_text(encoding="utf-8").strip() if method == "living_harness" else None,
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (harness_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def verify_harness(
    harness_dir: Path, *, method: str, domain: str, split: dict[str, Any],
    model: str | None = None, runner_source: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads((harness_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("method") != method
        or manifest.get("domain") != domain
        or manifest.get("split_hash") != canonical_hash(split)
        or manifest.get("train_ids_hash") != canonical_hash(split["train_ids"])
        or (model is not None and manifest.get("model") != model)
        or (runner_source is not None and manifest.get("runner_source_sha256") != sha256_file(runner_source))
    ):
        raise ValueError(f"harness metadata mismatch: {harness_dir}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES[method]):
        raise ValueError(f"harness file list mismatch: {harness_dir}")
    for name, digest in files.items():
        if sha256_file(harness_dir / name) != digest:
            raise ValueError(f"harness file changed: {harness_dir / name}")
    return manifest
