from __future__ import annotations

import json
import io
import os
import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from api_config import ApiConfig
from baseline_registry import DOMAIN_ORDER, METHOD_ORDER, get_baseline_specs


def _fake_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    for spec in get_baseline_specs().values():
        src = workspace / spec.repo_name / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "tau2_runner.py").write_text("# fake runner\n", encoding="utf-8")
        prompts = workspace / spec.repo_name / "prompts" / "prior"
        prompts.mkdir(parents=True)
        (prompts / "domain_prior_retail.md").write_text("prompt", encoding="utf-8")
    tau2_repo = workspace / "tau2-bench"
    (tau2_repo / "src" / "tau2").mkdir(parents=True)
    (tau2_repo / "src" / "tau2" / "__init__.py").write_text("", encoding="utf-8")
    for filename, content in {
        "LICENSE": "MIT",
        "README.md": "tau2",
        "pyproject.toml": "[project]\nname='tau2'\nversion='0'\n",
        "uv.lock": "lock",
    }.items():
        (tau2_repo / filename).write_text(content, encoding="utf-8")
    user_simulator = tau2_repo / "data" / "tau2" / "user_simulator"
    user_simulator.mkdir(parents=True)
    (user_simulator / "guidelines.md").write_text("guide", encoding="utf-8")
    domain_files = {
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
    for domain, filenames in domain_files.items():
        domain_dir = tau2_repo / "data" / "tau2" / "domains" / domain
        domain_dir.mkdir(parents=True)
        for filename in filenames:
            (domain_dir / filename).write_text("fixture", encoding="utf-8")
    return workspace, tau2_repo


def _api_file(tmp_path: Path) -> Path:
    path = tmp_path / "openai_test.py"
    path.write_text(
        "\n".join(
            [
                "from openai import OpenAI",
                "client = OpenAI(api_key='dry-run-secret', base_url='https://api.example.invalid/v1')",
                "client.chat.completions.create(model='gpt-5.2-2025-12-11', messages=[])",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_dry_run_writes_sanitized_eighteen_job_manifest(tmp_path: Path) -> None:
    from run_tau2_baselines import main

    workspace, tau2_repo = _fake_workspace(tmp_path)
    output_root = tmp_path / "results"
    marker = tmp_path / "subprocess-ran"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(f"#!/bin/sh\ntouch '{marker}'\n", encoding="utf-8")
    fake_python.chmod(0o755)

    exit_code = main(
        [
            "--dry-run",
            "--run-id",
            "dry-run",
            "--workspace-root",
            str(workspace),
            "--output-root",
            str(output_root),
            "--openai-test",
            str(_api_file(tmp_path)),
            "--tau2-repo",
            str(tau2_repo),
            "--python",
            str(fake_python),
        ]
    )

    assert exit_code == 0
    assert not marker.exists()
    manifest_path = output_root / "dry-run" / "manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(raw)
    assert len(manifest["jobs"]) == 18
    assert [(job["method"], job["domain"]) for job in manifest["jobs"]] == [
        (method, domain) for method in METHOD_ORDER for domain in DOMAIN_ORDER
    ]
    assert {job["status"] for job in manifest["jobs"]} == {"pending"}
    assert manifest["actor_model"] == "gpt-5.2-2025-12-11"
    assert manifest["user_model"] == "gpt-5.1-2025-11-13"
    assert manifest["judge_model"] == "gpt-5.1-2025-11-13"
    assert manifest["seed"] == 300
    assert "dry-run-secret" not in raw
    assert "api_key" not in raw.lower()


def test_build_child_environment_is_complete_and_sanitized(tmp_path: Path) -> None:
    from run_tau2_baselines import build_child_env, render_sanitized_environment

    workspace, tau2_repo = _fake_workspace(tmp_path)
    spec = get_baseline_specs()["evotest"]
    output_dir = tmp_path / "run" / "evotest" / "retail"
    base_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "/caller/path",
        "LLM_API_KEY": "old-secret",
    }
    api = ApiConfig(
        model="gpt-5.2-2025-12-11",
        base_url="https://api.example.invalid/v1",
        api_key="new-secret",
    )

    env = build_child_env(
        base_environ=base_env,
        api_config=api,
        baseline=spec,
        domain="retail",
        output_dir=output_dir,
        tau2_repo=tau2_repo,
        workers=2,
        runtime_shim=CODE_DIR / "runtime_shim",
        repo_src=workspace / spec.repo_name / "src",
    )
    rendered = json.dumps(render_sanitized_environment(env), sort_keys=True)

    assert env["LLM_BACKEND"] == "litellm_vllm"
    assert env["LLM_MODEL_NAME"] == "gpt-5.2-2025-12-11"
    assert env["TAU2_USER_MODEL"] == "gpt-5.1-2025-11-13"
    assert env["NL_JUDGE_MODEL"] == "gpt-5.1-2025-11-13"
    assert env["LLM_BASE_URL"] == "https://api.example.invalid/v1"
    assert env["LLM_API_KEY"] == "new-secret"
    assert env["NL_JUDGE_API_KEY"] == "new-secret"
    assert env["START_VLLM_SERVER"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["GPU"] == ""
    assert env["PROC_PER_NODE"] == "2"
    assert env["RESULT_BATCH_SIZE"] == "1"
    assert env["RESULT_SAVE_EVERY"] == "1"
    assert env["DOMAIN"] == "retail"
    assert env["TAU2_DATA_DIR"] == str((tau2_repo / "data").resolve())
    assert env["TAU2_RUN_SEED"] == "300"
    assert env["TAU2_ASSIGNED_OUTPUT_DIR"] == str(output_dir.resolve())
    assert env["TAU2_API_USAGE_LOG"] == str((output_dir / "api_usage.jsonl").resolve())
    assert env["TAU2_API_RPM_LIMIT"] == "20"
    assert env["EVOTEST_HINTS_ENABLED"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[:2] == [
        str((CODE_DIR / "runtime_shim").resolve()),
        str((workspace / spec.repo_name / "src").resolve()),
    ]
    assert "new-secret" not in rendered
    assert "old-secret" not in rendered
    assert "api_key" not in rendered.lower()
    assert "authorization" not in rendered.lower()
    assert '"RESULT_BATCH_SIZE": "1"' in rendered
    assert '"RESULT_SAVE_EVERY": "1"' in rendered


def test_evolve_baselines_use_async_sample_workers(tmp_path: Path) -> None:
    from run_tau2_baselines import build_child_env

    workspace, tau2_repo = _fake_workspace(tmp_path)
    api = ApiConfig(
        model="gpt-5.2-2025-12-11",
        base_url="https://api.example.invalid/v1",
        api_key="secret",
    )
    for method in ("awm", "reasoningbank", "evotest"):
        spec = get_baseline_specs()[method]
        env = build_child_env(
            base_environ={},
            api_config=api,
            baseline=spec,
            domain="telecom",
            output_dir=tmp_path / method,
            tau2_repo=tau2_repo,
            workers=16,
            runtime_shim=CODE_DIR / "runtime_shim",
            repo_src=workspace / spec.repo_name / "src",
        )
        assert env["EVOLVE_PARALLEL_MODE"] == "async"
        assert env["PROC_PER_NODE"] == "16"
        assert env["RESULT_BATCH_SIZE"] == "1"
        assert env["TAU2_API_RPM_LIMIT"] == "20"


def test_bundled_workspace_is_default_and_jobs_stay_under_code_vendor(
    tmp_path: Path,
) -> None:
    from run_tau2_baselines import _parse_args, build_jobs

    args = _parse_args([])
    assert args.workspace_root == CODE_DIR / "vendor"
    workspace = CODE_DIR / "vendor"
    jobs = build_jobs(
        methods=["react"],
        domains=["retail"],
        workspace_root=workspace,
        run_dir=tmp_path / "run",
        tau2_repo=workspace / "tau2-bench",
        api_config=ApiConfig(
            model="gpt-5.2-2025-12-11",
            base_url="https://api.example.invalid/v1",
            api_key="secret",
        ),
        workers=1,
        base_environ={},
    )

    assert jobs[0].repo == (workspace / "sop-world-react").resolve()
    assert jobs[0].env["TAU2_REPO"] == str((workspace / "tau2-bench").resolve())
    assert jobs[0].env["TAU2_DATA_DIR"] == str(
        (workspace / "tau2-bench" / "data").resolve()
    )


def _write_child(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse, json, os, sys",
                "from pathlib import Path",
                "p = argparse.ArgumentParser()",
                "p.add_argument('--repo')",
                "p.add_argument('--output-dir')",
                "p.add_argument('--method')",
                "a = p.parse_args()",
                "print('child:' + a.method + ':' + os.environ['DOMAIN'])",
                "if os.environ.get('TEST_FAIL') == '1': sys.exit(7)",
                "out = Path(a.output_dir)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'summary.json').write_text(json.dumps({'ok': True}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )


def _job(tmp_path: Path, method: str, domain: str, *, fail: bool = False):
    from run_tau2_baselines import JobSpec

    output_dir = tmp_path / method / domain
    env = os.environ.copy()
    env["DOMAIN"] = domain
    if fail:
        env["TEST_FAIL"] = "1"
    return JobSpec(
        method=method,
        domain=domain,
        repo=tmp_path / "repo",
        output_dir=output_dir,
        log_path=tmp_path / "logs" / f"{method}__{domain}.log",
        env=env,
    )


def test_execution_updates_manifest_and_creates_logs(tmp_path: Path) -> None:
    from run_tau2_baselines import run_jobs, write_manifest

    child = tmp_path / "child.py"
    _write_child(child)
    jobs = [_job(tmp_path, "react", "retail"), _job(tmp_path, "react", "airline")]
    manifest = {"jobs": [{"method": j.method, "domain": j.domain, "status": "pending"} for j in jobs]}
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)

    status = run_jobs(
        jobs,
        manifest=manifest,
        manifest_path=manifest_path,
        python_executable=sys.executable,
        entry_script=child,
        continue_on_error=False,
    )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert status == 0
    assert [item["status"] for item in saved["jobs"]] == ["succeeded", "succeeded"]
    assert all(job.log_path.is_file() for job in jobs)
    assert "child:react:retail" in jobs[0].log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("continue_on_error", "expected_statuses"),
    [
        (False, ["failed", "pending"]),
        (True, ["failed", "succeeded"]),
    ],
)
def test_execution_failure_policy(
    tmp_path: Path,
    continue_on_error: bool,
    expected_statuses: list[str],
) -> None:
    from run_tau2_baselines import run_jobs, write_manifest

    child = tmp_path / "child.py"
    _write_child(child)
    jobs = [_job(tmp_path, "react", "retail", fail=True), _job(tmp_path, "awm", "airline")]
    manifest = {"jobs": [{"method": j.method, "domain": j.domain, "status": "pending"} for j in jobs]}
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)

    status = run_jobs(
        jobs,
        manifest=manifest,
        manifest_path=manifest_path,
        python_executable=sys.executable,
        entry_script=child,
        continue_on_error=continue_on_error,
    )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert status == 1
    assert [item["status"] for item in saved["jobs"]] == expected_statuses


def test_parallel_jobs_run_independent_domains_together(tmp_path: Path) -> None:
    from run_tau2_baselines import run_jobs_parallel, write_manifest

    child = tmp_path / "barrier_child.py"
    child.write_text(
        "\n".join(
            [
                "import argparse, json, sys, time",
                "from pathlib import Path",
                "p = argparse.ArgumentParser()",
                "p.add_argument('--repo')",
                "p.add_argument('--output-dir')",
                "p.add_argument('--method')",
                "a = p.parse_args()",
                "barrier = Path(a.repo).parent / 'barrier'",
                "barrier.mkdir(exist_ok=True)",
                "(barrier / (a.method + '_' + Path(a.output_dir).name)).write_text('ready')",
                "deadline = time.monotonic() + 10",
                "while len(list(barrier.glob('*'))) < 3:",
                "    if time.monotonic() > deadline: sys.exit(9)",
                "    time.sleep(0.05)",
                "out = Path(a.output_dir)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'summary.json').write_text(json.dumps({'ok': True}))",
            ]
        ),
        encoding="utf-8",
    )
    jobs = [
        _job(tmp_path, "awm", "telecom"),
        _job(tmp_path, "reasoningbank", "retail"),
        _job(tmp_path, "evotest", "airline"),
    ]
    manifest = {
        "jobs": [
            {"method": job.method, "domain": job.domain, "status": "pending"}
            for job in jobs
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    terminal = io.StringIO()

    status = run_jobs_parallel(
        jobs,
        manifest=manifest,
        manifest_path=manifest_path,
        python_executable=sys.executable,
        entry_script=child,
        continue_on_error=True,
        parallel_jobs=3,
        terminal=terminal,
    )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert status == 0
    assert saved["status"] == "succeeded"
    assert [record["status"] for record in saved["jobs"]] == ["succeeded"] * 3
    assert "3/3" in terminal.getvalue()
    assert all(job.log_path.is_file() for job in jobs)


def test_explicit_job_selection_plans_only_requested_pairs(tmp_path: Path) -> None:
    from run_tau2_baselines import main

    workspace, tau2_repo = _fake_workspace(tmp_path)
    output_root = tmp_path / "results"
    selected = "awm/telecom,reasoningbank/retail,evotest/airline"
    assert main(
        [
            "--dry-run",
            "--jobs",
            selected,
            "--parallel-jobs",
            "3",
            "--workers",
            "1",
            "--run-id",
            "selected",
            "--workspace-root",
            str(workspace),
            "--output-root",
            str(output_root),
            "--openai-test",
            str(_api_file(tmp_path)),
            "--tau2-repo",
            str(tau2_repo),
            "--python",
            sys.executable,
        ]
    ) == 0

    manifest = json.loads((output_root / "selected" / "manifest.json").read_text())
    assert manifest["parallel_jobs"] == 3
    assert [(job["method"], job["domain"]) for job in manifest["jobs"]] == [
        ("awm", "telecom"),
        ("reasoningbank", "retail"),
        ("evotest", "airline"),
    ]


def test_resume_skips_valid_summary_and_rejects_partial_output(tmp_path: Path) -> None:
    from run_tau2_baselines import classify_resume_output

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "summary.json").write_text('{"reward": 1}', encoding="utf-8")
    assert classify_resume_output(complete) == "skipped"

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "api_usage.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        classify_resume_output(partial)
    assert (partial / "api_usage.jsonl").is_file()

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "summary.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        classify_resume_output(malformed)


def test_infrastructure_errors_make_job_summary_invalid(tmp_path: Path) -> None:
    from run_tau2_baselines import _valid_summary, classify_resume_output

    summary = tmp_path / "summary.json"
    summary.write_text('{"total_samples": 114, "infra_error_count": 114}')
    assert not _valid_summary(tmp_path)
    with pytest.raises(ValueError, match="infrastructure errors"):
        classify_resume_output(tmp_path)
    summary.write_text('{"total_samples": 114, "infra_error_count": 0}')
    assert _valid_summary(tmp_path)


def test_existing_run_directory_requires_resume(tmp_path: Path) -> None:
    from run_tau2_baselines import prepare_run_directory

    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="--resume"):
        prepare_run_directory(run_dir, resume=False)
    assert prepare_run_directory(run_dir, resume=True) == run_dir.resolve()


def test_cli_sanitizes_malformed_configuration_errors(tmp_path: Path) -> None:
    import subprocess

    workspace, tau2_repo = _fake_workspace(tmp_path)
    secret = "malformed-cli-secret"
    malformed = tmp_path / "openai_test.py"
    malformed.write_text(
        f"client = OpenAI(api_key={secret!r} base_url='https://example.invalid/v1')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / "run_tau2_baselines.py"),
            "--dry-run",
            "--run-id",
            "malformed",
            "--workspace-root",
            str(workspace),
            "--output-root",
            str(tmp_path / "results"),
            "--openai-test",
            str(malformed),
            "--tau2-repo",
            str(tau2_repo),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert secret not in result.stderr
    assert "Traceback" not in result.stderr
    assert "parse" in result.stderr.lower()


def test_resume_preserves_manifest_and_rejects_configuration_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from run_tau2_baselines import main

    workspace, tau2_repo = _fake_workspace(tmp_path)
    output_root = tmp_path / "results"
    common = [
        "--run-id",
        "resume-run",
        "--workspace-root",
        str(workspace),
        "--output-root",
        str(output_root),
        "--openai-test",
        str(_api_file(tmp_path)),
        "--tau2-repo",
        str(tau2_repo),
        "--python",
        sys.executable,
    ]
    assert main(["--dry-run", *common]) == 0
    manifest_path = output_root / "resume-run" / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = original["jobs"][0]
    first["status"] = "succeeded"
    first["exit_code"] = 0
    output_dir = Path(first["output_dir"])
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text('{"reward": 1}', encoding="utf-8")
    manifest_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
    before_resume = manifest_path.read_text(encoding="utf-8")

    assert main(["--resume", "--dry-run", "--methods", "react", *common]) == 0
    assert manifest_path.read_text(encoding="utf-8") == before_resume

    monkeypatch.setenv("LLM_MODEL_NAME", "different-model")
    with pytest.raises(ValueError, match="incompatible"):
        main(["--resume", "--methods", "react", *common])
    assert manifest_path.read_text(encoding="utf-8") == before_resume


def test_bundled_preflight_requires_manifest_and_runtime_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_tau2_baselines
    from run_tau2_baselines import main
    from vendor_bundle import sync_vendor_bundle

    workspace, _ = _fake_workspace(tmp_path)
    code_dir = tmp_path / "portable" / "code"
    code_dir.mkdir(parents=True)
    api_file = _api_file(tmp_path)
    sync_vendor_bundle(workspace, code_dir, api_file)
    monkeypatch.setattr(run_tau2_baselines, "CODE_DIR", code_dir)
    common = [
        "--dry-run",
        "--run-id",
        "bundle-preflight",
        "--output-root",
        str(tmp_path / "results"),
        "--openai-test",
        str(api_file),
        "--python",
        sys.executable,
    ]

    (code_dir / "vendor" / "bundle_manifest.json").unlink()
    with pytest.raises(ValueError, match="bundle_manifest"):
        main(common)

    sync_vendor_bundle(workspace, code_dir, api_file)
    prompt = code_dir / "vendor" / "sop-world-react" / "prompts" / "prior" / "domain_prior_retail.md"
    prompt.unlink()
    with pytest.raises(ValueError, match="prompts"):
        main(common)

    sync_vendor_bundle(workspace, code_dir, api_file)
    data_file = code_dir / "vendor" / "tau2-bench" / "data" / "tau2" / "domains" / "retail" / "tasks.json"
    data_file.unlink()
    with pytest.raises(ValueError, match="tau2 data"):
        main(common)


def test_verify_bundle_rejects_mutation_before_creating_run_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_tau2_baselines
    from run_tau2_baselines import main
    from vendor_bundle import sync_vendor_bundle

    workspace, _ = _fake_workspace(tmp_path)
    code_dir = tmp_path / "portable" / "code"
    code_dir.mkdir(parents=True)
    api_file = _api_file(tmp_path)
    sync_vendor_bundle(workspace, code_dir, api_file)
    monkeypatch.setattr(run_tau2_baselines, "CODE_DIR", code_dir)
    runner = code_dir / "vendor" / "sop-world-react" / "src" / "tau2_runner.py"
    runner.write_text("# mutated runner\n", encoding="utf-8")
    output_root = tmp_path / "results"

    with pytest.raises(ValueError, match="hash|size"):
        main(
            [
                "--dry-run",
                "--verify-bundle",
                "--run-id",
                "mutated",
                "--output-root",
                str(output_root),
                "--openai-test",
                str(api_file),
                "--python",
                sys.executable,
            ]
        )

    assert not output_root.exists()


def test_explicit_external_workspace_does_not_require_bundle_manifest(
    tmp_path: Path,
) -> None:
    from run_tau2_baselines import main

    workspace, tau2_repo = _fake_workspace(tmp_path)
    assert main(
        [
            "--dry-run",
            "--run-id",
            "external",
            "--workspace-root",
            str(workspace),
            "--output-root",
            str(tmp_path / "results"),
            "--openai-test",
            str(_api_file(tmp_path)),
            "--tau2-repo",
            str(tau2_repo),
            "--python",
            sys.executable,
        ]
    ) == 0
