from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
        "RESULT_BATCH_SIZE",
        "RESULT_SAVE_EVERY",
        "START_VLLM_SERVER",
        "TAU2_API_USAGE_LOG",
        "TAU2_API_RPM_LIMIT",
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
    env.setdefault("RESULT_BATCH_SIZE", "1")
    env.setdefault("RESULT_SAVE_EVERY", "1")
    if baseline.name in {"awm", "reasoningbank", "evotest"}:
        env.setdefault("TAU2_API_RPM_LIMIT", "20")
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
    if not _valid_summary(output_dir):
        raise ValueError(
            f"job output has infrastructure errors and must be moved before resume: {output_dir}"
        )
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
    parallel_jobs: int,
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
        "parallel_jobs": parallel_jobs,
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
    if existing.get("parallel_jobs", 1) != candidate.get("parallel_jobs", 1):
        raise ValueError("incompatible resume configuration: parallel_jobs")

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
    if not isinstance(value, dict):
        return False
    infra_errors = value.get("infra_error_count", 0)
    return isinstance(infra_errors, int) and infra_errors == 0


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
    from tqdm import tqdm

    stream = sys.stdout if terminal is None else terminal
    progress_stream = sys.stderr if terminal is None else terminal
    any_failed = False
    failed_jobs = 0
    records = manifest["jobs"]
    records_by_key = {
        (str(record["method"]), str(record["domain"])): record
        for record in records
    }
    skipped_jobs = sum(
        records_by_key[(job.method, job.domain)]["status"] == "skipped"
        for job in jobs
    )
    with tqdm(
        total=len(jobs),
        initial=skipped_jobs,
        desc="tau2 jobs",
        unit="job",
        dynamic_ncols=True,
        file=progress_stream,
    ) as progress:
        for job in jobs:
            record = records_by_key[(job.method, job.domain)]
            if record["status"] == "skipped":
                continue
            progress.set_postfix(job=f"{job.method}/{job.domain}", failed=failed_jobs)
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
                failed_jobs += 1
            progress.update(1)
            progress.set_postfix(job=f"{job.method}/{job.domain}", failed=failed_jobs)
            if not succeeded and not continue_on_error:
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


def _run_parallel_job(
    job: JobSpec,
    *,
    python_executable: str,
    entry_script: Path,
    active_processes: dict[tuple[str, str], subprocess.Popen[str]],
    process_lock: threading.Lock,
    stopping: threading.Event,
) -> tuple[int, bool]:
    key = (job.method, job.domain)
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
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    return_code = 1
    with job.log_path.open("w", encoding="utf-8") as log_file:
        try:
            with process_lock:
                if stopping.is_set():
                    return 130, False
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=dict(job.env),
                    bufsize=1,
                    start_new_session=True,
                )
                active_processes[key] = process
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log_file.write(_redact_output(line, job.env))
                    log_file.flush()
                return_code = process.wait()
            finally:
                with process_lock:
                    active_processes.pop(key, None)
        except OSError as exc:
            log_file.write(f"failed to start child process ({type(exc).__name__})\n")
    return return_code, return_code == 0 and _valid_summary(job.output_dir)


def _stop_parallel_processes(
    active_processes: dict[tuple[str, str], subprocess.Popen[str]],
    process_lock: threading.Lock,
    stopping: threading.Event,
) -> None:
    with process_lock:
        stopping.set()
        processes = list(active_processes.values())
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_jobs_parallel(
    jobs: Sequence[JobSpec],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    python_executable: str,
    entry_script: Path,
    continue_on_error: bool,
    parallel_jobs: int,
    terminal: TextIO | None = None,
) -> int:
    from tqdm import tqdm

    if parallel_jobs < 1:
        raise ValueError("parallel_jobs must be at least 1")
    progress_stream = sys.stderr if terminal is None else terminal
    records = manifest["jobs"]
    records_by_key = {
        (str(record["method"]), str(record["domain"])): record
        for record in records
    }
    skipped_jobs = sum(
        records_by_key[(job.method, job.domain)]["status"] == "skipped"
        for job in jobs
    )
    pending_jobs = [
        job
        for job in jobs
        if records_by_key[(job.method, job.domain)]["status"] != "skipped"
    ]
    active_processes: dict[tuple[str, str], subprocess.Popen[str]] = {}
    process_lock = threading.Lock()
    stopping = threading.Event()
    futures: dict[Future[tuple[int, bool]], JobSpec] = {}
    executor = ThreadPoolExecutor(max_workers=parallel_jobs)
    failed_jobs = 0
    next_job_index = 0
    stop_scheduling = False
    try:
        with tqdm(
            total=len(jobs),
            initial=skipped_jobs,
            desc="tau2 jobs",
            unit="job",
            dynamic_ncols=True,
            file=progress_stream,
        ) as progress:

            def submit_next_job() -> None:
                nonlocal next_job_index
                job = pending_jobs[next_job_index]
                next_job_index += 1
                record = records_by_key[(job.method, job.domain)]
                record["status"] = "running"
                write_manifest(manifest_path, manifest)
                future = executor.submit(
                    _run_parallel_job,
                    job,
                    python_executable=python_executable,
                    entry_script=entry_script,
                    active_processes=active_processes,
                    process_lock=process_lock,
                    stopping=stopping,
                )
                futures[future] = job
                progress.write(f"started {job.method}/{job.domain}", file=progress_stream)
                progress.set_postfix(running=len(futures), failed=failed_jobs)

            while next_job_index < len(pending_jobs) and len(futures) < parallel_jobs:
                submit_next_job()

            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    job = futures.pop(future)
                    record = records_by_key[(job.method, job.domain)]
                    try:
                        return_code, succeeded = future.result()
                    except Exception as exc:
                        return_code, succeeded = 1, False
                        job.log_path.parent.mkdir(parents=True, exist_ok=True)
                        with job.log_path.open("a", encoding="utf-8") as log_file:
                            log_file.write(f"job worker failed ({type(exc).__name__})\n")
                    record["status"] = "succeeded" if succeeded else "failed"
                    record["exit_code"] = return_code
                    write_manifest(manifest_path, manifest)
                    if not succeeded:
                        failed_jobs += 1
                        if not continue_on_error:
                            stop_scheduling = True
                    progress.update(1)
                    progress.write(
                        f"finished {job.method}/{job.domain}: {record['status']}",
                        file=progress_stream,
                    )
                    progress.set_postfix(running=len(futures), failed=failed_jobs)
                while (
                    not stop_scheduling
                    and next_job_index < len(pending_jobs)
                    and len(futures) < parallel_jobs
                ):
                    submit_next_job()
    except KeyboardInterrupt:
        _stop_parallel_processes(active_processes, process_lock, stopping)
        for job in jobs:
            record = records_by_key[(job.method, job.domain)]
            if record["status"] == "running":
                record["status"] = "failed"
                record["exit_code"] = -int(signal.SIGTERM)
        manifest["status"] = "failed"
        manifest["interrupted_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        return 130
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    all_statuses = {str(record.get("status", "")) for record in records}
    if "failed" in all_statuses:
        manifest["status"] = "failed"
    elif all_statuses & {"pending", "running"}:
        manifest["status"] = "partial"
    else:
        manifest["status"] = "succeeded"
    write_manifest(manifest_path, manifest)
    return 1 if failed_jobs else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all selected tau2 baselines through an OpenAI-compatible API"
    )
    parser.add_argument("--methods")
    parser.add_argument("--domains")
    parser.add_argument("--jobs", help="comma-separated method/domain pairs")
    parser.add_argument("--run-id")
    parser.add_argument("--workspace-root", type=Path, default=CODE_DIR / "vendor")
    parser.add_argument("--output-root", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--openai-test", type=Path, default=CODE_DIR / "openai_test.py")
    parser.add_argument("--tau2-repo", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verify-bundle", action="store_true")
    return parser.parse_args(argv)


def _parse_job_selection(raw: str) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for item in raw.split(","):
        parts = item.strip().split("/")
        if len(parts) != 2:
            raise ValueError(f"invalid --jobs entry: {item.strip()}")
        method, domain = (part.strip() for part in parts)
        if method not in METHOD_ORDER or domain not in DOMAIN_ORDER:
            raise ValueError(f"unknown method/domain in --jobs: {item.strip()}")
        pair = (method, domain)
        if pair in selected:
            raise ValueError(f"duplicate --jobs entry: {item.strip()}")
        selected.append(pair)
    if not selected:
        raise ValueError("--jobs must select at least one method/domain pair")
    return selected


def _resolve_python(value: str) -> str:
    if os.sep in value:
        # Keep a venv's python symlink: resolving it selects the system interpreter.
        path = Path(value).expanduser().absolute()
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
    if args.parallel_jobs < 1:
        raise ValueError("--parallel-jobs must be at least 1")
    if args.jobs is not None:
        if args.methods is not None or args.domains is not None:
            raise ValueError("--jobs cannot be combined with --methods or --domains")
        selected_pairs = _parse_job_selection(args.jobs)
        methods = list(dict.fromkeys(method for method, _ in selected_pairs))
        domains = list(dict.fromkeys(domain for _, domain in selected_pairs))
    else:
        selected_pairs = None
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
    if selected_pairs is not None:
        jobs_by_pair = {(job.method, job.domain): job for job in jobs}
        jobs = [jobs_by_pair[pair] for pair in selected_pairs]
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
        parallel_jobs=args.parallel_jobs,
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
    runner = run_jobs_parallel if args.parallel_jobs > 1 else run_jobs
    runner_options = {"parallel_jobs": args.parallel_jobs} if args.parallel_jobs > 1 else {}
    return runner(
        jobs,
        manifest=manifest,
        manifest_path=manifest_path,
        python_executable=python_executable,
        entry_script=CODE_DIR / "tau2_baseline_entry.py",
        continue_on_error=args.continue_on_error,
        **runner_options,
    )


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
