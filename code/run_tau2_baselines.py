from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from api_config import ApiConfig, load_api_config
from baseline_registry import (
    DOMAIN_ORDER,
    METHOD_ORDER,
    BaselineSpec,
    get_baseline_specs,
    parse_selection,
)
from vendor_bundle import TAU2_DOMAIN_FILES, verify_vendor_bundle


CODE_DIR = Path(__file__).resolve().parent
ACTOR_MODEL = "gpt-5.2-2025-12-11"
USER_MODEL = "gpt-5.1-2025-11-13"
DEFAULT_SEED = 300


@dataclass(frozen=True)
class JobSpec:
    method: str
    domain: str
    repo: Path
    output_dir: Path
    log_path: Path
    env: Mapping[str, str]


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lower()
    return any(part in normalized for part in ("api_key", "authorization", "secret"))


def render_sanitized_environment(environ: Mapping[str, str]) -> dict[str, str]:
    experiment_names = {
        "DOMAIN",
        "GPU",
        "LLM_BACKEND",
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "NL_JUDGE_BASE_URL",
        "NL_JUDGE_MODEL",
        "PROC_PER_NODE",
        "PYTHONPATH",
        "START_VLLM_SERVER",
        "TAU2_API_USAGE_LOG",
        "TAU2_ASSIGNED_OUTPUT_DIR",
        "TAU2_DATA_DIR",
        "TAU2_REPO",
        "TAU2_RUN_SEED",
        "TAU2_USER_MODEL",
    }
    experiment_names.update(
        key for spec in get_baseline_specs().values() for key in spec.env
    )
    return {
        key: str(environ[key])
        for key in sorted(experiment_names)
        if key in environ and not _is_sensitive_name(key)
    }


def build_child_env(
    *,
    base_environ: Mapping[str, str],
    api_config: ApiConfig,
    baseline: BaselineSpec,
    domain: str,
    output_dir: Path,
    tau2_repo: Path,
    workers: int,
    runtime_shim: Path,
    repo_src: Path,
) -> dict[str, str]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    env = {str(key): str(value) for key, value in base_environ.items()}
    existing_pythonpath = env.get("PYTHONPATH", "")
    python_paths = [str(runtime_shim.resolve()), str(repo_src.resolve())]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env.update(
        {
            "LLM_BACKEND": "litellm_vllm",
            "LLM_MODEL_NAME": api_config.model,
            "LLM_BASE_URL": api_config.base_url,
            "LLM_API_KEY": api_config.api_key,
            "TAU2_USER_MODEL": USER_MODEL,
            "NL_JUDGE_MODEL": USER_MODEL,
            "NL_JUDGE_BASE_URL": api_config.base_url,
            "NL_JUDGE_API_KEY": api_config.api_key,
            "START_VLLM_SERVER": "0",
            "GPU": "",
            "PROC_PER_NODE": str(workers),
            "DOMAIN": domain,
            "TAU2_REPO": str(tau2_repo.resolve()),
            "TAU2_DATA_DIR": str((tau2_repo / "data").resolve()),
            "TAU2_RUN_SEED": str(DEFAULT_SEED),
            "TAU2_ASSIGNED_OUTPUT_DIR": str(output_dir.resolve()),
            "TAU2_API_USAGE_LOG": str((output_dir / "api_usage.jsonl").resolve()),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update({key: str(value) for key, value in baseline.env.items()})
    return env


def build_jobs(
    *,
    methods: Sequence[str],
    domains: Sequence[str],
    workspace_root: Path,
    run_dir: Path,
    tau2_repo: Path,
    api_config: ApiConfig,
    workers: int,
    base_environ: Mapping[str, str],
) -> list[JobSpec]:
    specs = get_baseline_specs()
    jobs: list[JobSpec] = []
    for method in methods:
        baseline = specs[method]
        repo = (workspace_root / baseline.repo_name).resolve()
        for domain in domains:
            output_dir = (run_dir / method / domain).resolve()
            jobs.append(
                JobSpec(
                    method=method,
                    domain=domain,
                    repo=repo,
                    output_dir=output_dir,
                    log_path=(run_dir / "logs" / f"{method}__{domain}.log").resolve(),
                    env=build_child_env(
                        base_environ=base_environ,
                        api_config=api_config,
                        baseline=baseline,
                        domain=domain,
                        output_dir=output_dir,
                        tau2_repo=tau2_repo,
                        workers=workers,
                        runtime_shim=CODE_DIR / "runtime_shim",
                        repo_src=repo / "src",
                    ),
                )
            )
    return jobs


def prepare_run_directory(run_dir: Path, *, resume: bool) -> Path:
    resolved = run_dir.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"run path is not a directory: {resolved}")
        if not resume:
            raise ValueError(
                f"run directory already exists; choose another --run-id or use --resume: {resolved}"
            )
        return resolved
    resolved.mkdir(parents=True)
    return resolved


def classify_resume_output(output_dir: Path) -> str:
    if not output_dir.exists():
        return "pending"
    if not output_dir.is_dir():
        raise ValueError(f"job output path is not a directory: {output_dir}")
    entries = list(output_dir.iterdir())
    if not entries:
        return "pending"
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(
            f"partial job output must be moved before resume: {output_dir}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid summary.json in resume output: {output_dir}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"summary.json must contain an object: {output_dir}")
    return "skipped"


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _job_record(job: JobSpec, status: str) -> dict[str, Any]:
    return {
        "method": job.method,
        "domain": job.domain,
        "repo": str(job.repo),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "status": status,
        "environment": render_sanitized_environment(job.env),
    }


def _manifest(
    *,
    run_id: str,
    jobs: Sequence[JobSpec],
    statuses: Sequence[str],
    api_config: ApiConfig,
    workers: int,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "actor_model": api_config.model,
        "user_model": USER_MODEL,
        "judge_model": USER_MODEL,
        "base_url": api_config.base_url,
        "seed": DEFAULT_SEED,
        "workers": workers,
        "jobs": [
            _job_record(job, status) for job, status in zip(jobs, statuses, strict=True)
        ],
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing manifest is invalid: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise ValueError(f"existing manifest is invalid: {path}")
    return value


def _reconcile_resume_manifest(
    existing: Mapping[str, Any],
    candidate: Mapping[str, Any],
    jobs: Sequence[JobSpec],
    statuses: Sequence[str],
) -> dict[str, Any]:
    for field in (
        "run_id",
        "actor_model",
        "user_model",
        "judge_model",
        "base_url",
        "seed",
        "workers",
    ):
        if existing.get(field) != candidate.get(field):
            raise ValueError(f"incompatible resume configuration: {field}")

    existing_jobs = existing.get("jobs")
    candidate_jobs = candidate.get("jobs")
    if not isinstance(existing_jobs, list) or not isinstance(candidate_jobs, list):
        raise ValueError("incompatible resume manifest jobs")

    merged_jobs = [dict(item) for item in existing_jobs if isinstance(item, Mapping)]
    if len(merged_jobs) != len(existing_jobs):
        raise ValueError("incompatible resume manifest jobs")
    indexes: dict[tuple[str, str], int] = {}
    for index, record in enumerate(merged_jobs):
        key = (str(record.get("method", "")), str(record.get("domain", "")))
        if not all(key) or key in indexes:
            raise ValueError("incompatible resume manifest job identity")
        indexes[key] = index

    candidate_by_key = {
        (str(record["method"]), str(record["domain"])): record
        for record in candidate_jobs
        if isinstance(record, Mapping)
    }
    for job, status in zip(jobs, statuses, strict=True):
        key = (job.method, job.domain)
        new_record = dict(candidate_by_key[key])
        existing_index = indexes.get(key)
        if existing_index is None:
            new_record["status"] = status
            indexes[key] = len(merged_jobs)
            merged_jobs.append(new_record)
            continue
        old_record = merged_jobs[existing_index]
        for field in ("repo", "output_dir", "log_path", "environment"):
            if old_record.get(field) != new_record.get(field):
                raise ValueError(
                    f"incompatible resume configuration for {job.method}/{job.domain}: {field}"
                )
        replacement = dict(old_record)
        replacement["status"] = status
        if status == "pending":
            replacement.pop("exit_code", None)
        merged_jobs[existing_index] = replacement

    merged = dict(existing)
    merged["dry_run"] = candidate.get("dry_run", False)
    merged["resumed_at"] = datetime.now(timezone.utc).isoformat()
    merged["jobs"] = merged_jobs
    return merged


def _redact_output(text: str, environ: Mapping[str, str]) -> str:
    redacted = text
    for name, value in environ.items():
        if _is_sensitive_name(name) and value:
            redacted = redacted.replace(str(value), "<redacted>")
    return redacted


def _valid_summary(output_dir: Path) -> bool:
    try:
        value = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict)


def run_jobs(
    jobs: Sequence[JobSpec],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    python_executable: str,
    entry_script: Path,
    continue_on_error: bool,
    terminal: TextIO | None = None,
) -> int:
    stream = sys.stdout if terminal is None else terminal
    any_failed = False
    records = manifest["jobs"]
    records_by_key = {
        (str(record["method"]), str(record["domain"])): record
        for record in records
    }
    for job in jobs:
        record = records_by_key[(job.method, job.domain)]
        if record["status"] == "skipped":
            continue
        record["status"] = "running"
        write_manifest(manifest_path, manifest)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            python_executable,
            str(entry_script),
            "--repo",
            str(job.repo),
            "--output-dir",
            str(job.output_dir),
            "--method",
            job.method,
        ]
        return_code = 1
        with job.log_path.open("w", encoding="utf-8") as log_file:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=dict(job.env),
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    safe_line = _redact_output(line, job.env)
                    log_file.write(safe_line)
                    log_file.flush()
                    stream.write(safe_line)
                    stream.flush()
                return_code = process.wait()
            except OSError as exc:
                safe_line = f"failed to start child process ({type(exc).__name__})\n"
                log_file.write(safe_line)
                stream.write(safe_line)
                stream.flush()
        succeeded = return_code == 0 and _valid_summary(job.output_dir)
        record["status"] = "succeeded" if succeeded else "failed"
        record["exit_code"] = return_code
        write_manifest(manifest_path, manifest)
        if not succeeded:
            any_failed = True
            if not continue_on_error:
                break
    all_statuses = {str(record.get("status", "")) for record in records}
    if "failed" in all_statuses:
        manifest["status"] = "failed"
    elif all_statuses & {"pending", "running"}:
        manifest["status"] = "partial"
    else:
        manifest["status"] = "succeeded"
    write_manifest(manifest_path, manifest)
    return 1 if any_failed else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all selected tau2 baselines through an OpenAI-compatible API"
    )
    parser.add_argument("--methods")
    parser.add_argument("--domains")
    parser.add_argument("--run-id")
    parser.add_argument("--workspace-root", type=Path, default=CODE_DIR / "vendor")
    parser.add_argument("--output-root", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--openai-test", type=Path, default=CODE_DIR / "openai_test.py")
    parser.add_argument("--tau2-repo", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verify-bundle", action="store_true")
    return parser.parse_args(argv)


def _resolve_python(value: str) -> str:
    if os.sep in value:
        path = Path(value).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Python executable is not executable: {path}")
        return str(path)
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"Python executable was not found: {value}")
    return resolved


def _validate_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run ID must be one non-empty path component")
    return run_id


def _validate_inputs(
    *,
    workspace_root: Path,
    methods: Sequence[str],
    tau2_repo: Path,
    openai_test: Path,
    output_root: Path,
    workers: int,
    bundled_workspace: bool,
) -> None:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if not openai_test.is_file():
        raise ValueError(f"OpenAI configuration file not found: {openai_test}")
    if not tau2_repo.is_dir():
        raise ValueError(f"tau2 repository not found: {tau2_repo}")
    specs = get_baseline_specs()
    for method in methods:
        repo = workspace_root / specs[method].repo_name
        runner = repo / "src" / "tau2_runner.py"
        if not runner.is_file():
            raise ValueError(f"baseline runner not found for {method}: {runner}")
        if bundled_workspace:
            prompts = repo / "prompts"
            if not prompts.is_dir() or not any(
                path.is_file() for path in prompts.rglob("*")
            ):
                raise ValueError(f"bundled baseline prompts are missing: {prompts}")
    if bundled_workspace:
        manifest_path = workspace_root / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"bundled workspace is missing bundle_manifest.json: {manifest_path}")
        if not (tau2_repo / "src" / "tau2").is_dir():
            raise ValueError(f"bundled tau2 source is missing: {tau2_repo / 'src/tau2'}")
        user_simulator = tau2_repo / "data" / "tau2" / "user_simulator"
        if not user_simulator.is_dir() or not any(
            path.is_file() for path in user_simulator.rglob("*")
        ):
            raise ValueError(f"bundled tau2 data is missing: {user_simulator}")
        for domain, filenames in TAU2_DOMAIN_FILES.items():
            for filename in filenames:
                path = tau2_repo / "data" / "tau2" / "domains" / domain / filename
                if not path.is_file():
                    raise ValueError(f"bundled tau2 data is missing: {path}")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output root is not a directory: {output_root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    methods = parse_selection(args.methods, METHOD_ORDER, "method")
    domains = parse_selection(args.domains, DOMAIN_ORDER, "domain")
    workspace_root = args.workspace_root.expanduser().resolve()
    bundled_workspace = workspace_root == (CODE_DIR / "vendor").resolve()
    tau2_repo = (
        args.tau2_repo.expanduser().resolve()
        if args.tau2_repo is not None
        else (workspace_root / "tau2-bench").resolve()
    )
    openai_test = args.openai_test.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    python_executable = _resolve_python(args.python)
    _validate_inputs(
        workspace_root=workspace_root,
        methods=methods,
        tau2_repo=tau2_repo,
        openai_test=openai_test,
        output_root=output_root,
        workers=args.workers,
        bundled_workspace=bundled_workspace,
    )
    if args.verify_bundle:
        verify_vendor_bundle(workspace_root)
    api_config = load_api_config(openai_test, os.environ)
    run_id = _validate_run_id(
        args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    run_dir = prepare_run_directory(output_root / run_id, resume=args.resume)
    jobs = build_jobs(
        methods=methods,
        domains=domains,
        workspace_root=workspace_root,
        run_dir=run_dir,
        tau2_repo=tau2_repo,
        api_config=api_config,
        workers=args.workers,
        base_environ=os.environ,
    )
    statuses = (
        [classify_resume_output(job.output_dir) for job in jobs]
        if args.resume
        else ["pending"] * len(jobs)
    )
    candidate_manifest = _manifest(
        run_id=run_id,
        jobs=jobs,
        statuses=statuses,
        api_config=api_config,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    manifest_path = run_dir / "manifest.json"
    if args.resume and manifest_path.is_file():
        manifest = _reconcile_resume_manifest(
            _load_manifest(manifest_path),
            candidate_manifest,
            jobs,
            statuses,
        )
        if not args.dry_run:
            write_manifest(manifest_path, manifest)
    else:
        manifest = candidate_manifest
        write_manifest(manifest_path, manifest)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "planned_jobs": sum(status == "pending" for status in statuses),
                    "skipped_jobs": sum(status == "skipped" for status in statuses),
                    "run_id": run_id,
                },
                sort_keys=True,
            )
        )
        return 0
    return run_jobs(
        jobs,
        manifest=manifest,
        manifest_path=manifest_path,
        python_executable=python_executable,
        entry_script=CODE_DIR / "tau2_baseline_entry.py",
        continue_on_error=args.continue_on_error,
    )


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
