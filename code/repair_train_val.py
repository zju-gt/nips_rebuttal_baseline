from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_config import load_api_config
from baseline_registry import get_baseline_specs
from run_tau2_baselines import CODE_DIR, build_child_env
from train_val_artifacts import verify_harness
from train_val_split import canonical_hash, load_split


def _needs_retry(simulation: dict[str, Any]) -> bool:
    reward_info = simulation.get("reward_info")
    return (
        str(simulation.get("termination_reason", "")).lower() == "infrastructure_error"
        or not isinstance(reward_info, dict)
        or reward_info.get("reward") is None
    )


def _read_results(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    simulations = value.get("simulations")
    ids = [str(item["task_id"]) for item in simulations]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f"result task IDs do not match selection: {path}")
    return value


def _run_retry(
    *, method: str, domain: str, ids: list[str], attempt: int, run_dir: Path,
    split: dict[str, Any], workspace: Path, tau2_repo: Path, python: Path,
    api: Any, workers: int, rpm_limit: int,
) -> tuple[Path, dict[str, Any]]:
    selection = run_dir / "split" / f"{domain}.val.retry_{attempt}.json"
    expected_selection = {"domain": domain, "task_ids": ids, "split_hash": canonical_hash(split)}
    if selection.exists():
        if json.loads(selection.read_text(encoding="utf-8")) != expected_selection:
            raise ValueError(f"retry selection changed: {selection}")
    else:
        selection.write_text(json.dumps(expected_selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output = run_dir / "val_retry" / method / domain / f"attempt_{attempt}"
    results_path = output / "results.json"
    if not results_path.exists():
        spec = get_baseline_specs()[method]
        env = build_child_env(
            base_environ=os.environ,
            api_config=api,
            baseline=spec,
            domain=domain,
            output_dir=output,
            tau2_repo=tau2_repo,
            workers=min(workers, len(ids)),
            runtime_shim=CODE_DIR / "runtime_shim",
            repo_src=workspace / spec.repo_name / "src",
        )
        env.update(
            TAU2_TASK_IDS_FILE=str(selection.resolve()),
            TAU2_TRAIN_VAL_PHASE="val",
            NUM_CYCLES="1",
            PASS_HAT_K="1",
            TAU2_API_RPM_LIMIT=str(rpm_limit),
            SAMPLE_TIMEOUT_SECONDS="2400",
        )
        if method != "react":
            env["TAU2_FROZEN_HARNESS_DIR"] = str((run_dir / "harnesses" / method / domain).resolve())
        repo = workspace / spec.repo_name
        command = [str(python), str(CODE_DIR / "tau2_baseline_entry.py"), "--repo", str(repo), "--output-dir", str(output), "--method", method]
        log_path = run_dir / "logs" / f"retry_{attempt}__{method}__{domain}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"retry {method}/{domain}: {len(ids)} tasks, attempt {attempt}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code != 0:
            raise RuntimeError(f"retry failed; see {log_path}")
    return output, _read_results(results_path, ids)


def _merge(
    *, method: str, domain: str, run_dir: Path, workspace: Path, tau2_repo: Path,
    original: dict[str, Any], original_summary: dict[str, Any],
    replacements: dict[str, dict[str, Any]], retry_dirs: list[Path],
    api: Any,
) -> None:
    output = run_dir / "val" / method / domain
    results_path = output / "results.json"
    summary_path = output / "summary.json"
    if (output / "results.initial.json").exists() or (output / "summary.initial.json").exists():
        raise ValueError(f"an earlier merge already modified {output}")
    merged = dict(original)
    merged["simulations"] = [replacements.get(str(item["task_id"]), item) for item in original["simulations"]]
    if any(_needs_retry(item) for item in merged["simulations"]):
        raise ValueError("merged validation still has missing rewards or infrastructure errors")

    spec = get_baseline_specs()[method]
    source = workspace / spec.repo_name / "src"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(tau2_repo / "src"))
    env = build_child_env(
        base_environ=os.environ, api_config=api, baseline=spec, domain=domain,
        output_dir=output, tau2_repo=tau2_repo, workers=1,
        runtime_shim=CODE_DIR / "runtime_shim", repo_src=source,
    )
    os.environ.update(env)
    runner = importlib.import_module("tau2_runner")
    mods = runner.get_tau2_modules()
    result_object = mods["Results"].model_validate(merged)
    cfg = runner.load_runner_config()
    success = sum(runner.is_successful_reward(runner._simulation_reward_value(item)) for item in result_object.simulations)
    retry_duration = sum(json.loads((directory / "summary.json").read_text(encoding="utf-8")).get("duration_seconds", 0) for directory in retry_dirs)
    summary = runner.compute_summary(
        mods, result_object, cfg,
        start_time=time.time() - float(original_summary.get("duration_seconds", 0)) - float(retry_duration),
        total_samples=len(result_object.simulations),
        success_count=success,
        failure_count=len(result_object.simulations) - success,
    )
    summary.update({
        "phase": "val", "frozen": True,
        "results_path": str(results_path), "run_dir": str(output),
        "selected_task_ids_hash": original_summary.get("selected_task_ids_hash"),
        "split_hash": original_summary.get("split_hash"),
        "harness_manifest_sha256": original_summary.get("harness_manifest_sha256"),
        "retry_result_paths": [str(directory / "results.json") for directory in retry_dirs],
        "retry_api_usage_logs": [str(directory / "api_usage.jsonl") for directory in retry_dirs],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if int(summary.get("infra_error_count", -1)) != 0 or int(summary.get("missing_reward_count", -1)) != 0:
        raise ValueError("recomputed validation summary is incomplete")
    shutil.copy2(results_path, output / "results.initial.json")
    shutil.copy2(summary_path, output / "summary.initial.json")
    temporary_results = output / "results.merged.tmp.json"
    result_object.save(temporary_results, format="json")
    temporary_results.replace(results_path)
    temporary_summary = output / "summary.merged.tmp.json"
    temporary_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)
    print(f"merged {method}/{domain}: {success}/{len(result_object.simulations)} successes, no missing rewards", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retry only invalid train/val validation tasks")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", default="react", choices=("react", "awm", "reasoningbank", "living_harness"))
    parser.add_argument("--domain", required=True, choices=("airline", "retail", "telecom"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rpm-limit", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--workspace-root", type=Path, default=CODE_DIR / "vendor")
    parser.add_argument("--python", type=Path, default=CODE_DIR / ".venv" / "bin" / "python")
    args = parser.parse_args(argv)
    if args.max_attempts < 1 or args.workers < 1 or args.rpm_limit < 0:
        raise ValueError("invalid retry settings")
    run_dir = args.output_root.expanduser().resolve() / args.run_id
    workspace = args.workspace_root.expanduser().resolve()
    tau2_repo = workspace / "tau2-bench"
    split = load_split(run_dir / "split" / f"{args.domain}.json", domain=args.domain, tau2_repo=tau2_repo)
    api = load_api_config(CODE_DIR / "openai_test.py", os.environ)
    if args.method != "react":
        verify_harness(run_dir / "harnesses" / args.method / args.domain, method=args.method, domain=args.domain, split=split, model=api.model)
    output = run_dir / "val" / args.method / args.domain
    original = _read_results(output / "results.json", split["val_ids"])
    original_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    current = {str(item["task_id"]): item for item in original["simulations"]}
    if not any(_needs_retry(item) for item in current.values()):
        print("validation has no tasks requiring retry")
        return 0
    retry_dirs: list[Path] = []
    for attempt in range(1, args.max_attempts + 1):
        ids = [task_id for task_id in split["val_ids"] if _needs_retry(current[task_id])]
        if not ids:
            break
        directory, retry = _run_retry(
            method=args.method, domain=args.domain, ids=ids, attempt=attempt,
            run_dir=run_dir, split=split, workspace=workspace, tau2_repo=tau2_repo,
            python=args.python.expanduser().absolute(), api=api, workers=args.workers,
            rpm_limit=args.rpm_limit,
        )
        retry_dirs.append(directory)
        current.update({str(item["task_id"]): item for item in retry["simulations"]})
    if any(_needs_retry(item) for item in current.values()):
        raise RuntimeError("some tasks still lack rewards after all retry attempts; original results are intact")
    replacements = {task_id: item for task_id, item in current.items() if item != next(x for x in original["simulations"] if str(x["task_id"]) == task_id)}
    _merge(
        method=args.method, domain=args.domain, run_dir=run_dir, workspace=workspace,
        tau2_repo=tau2_repo, original=original, original_summary=original_summary,
        replacements=replacements, retry_dirs=retry_dirs, api=api,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
