from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

BASELINES = (
    "sop-world-react",
    "sop-world-flexion",
    "sop-world-AWM",
    "sop-world-Reasoning",
    "sop-world-EvoTest",
    "sop-world",
)


def _copy_code(tmp_path: Path) -> Path:
    copied = tmp_path / "transfer" / "code"
    shutil.copytree(
        CODE_DIR,
        copied,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"),
    )
    return copied


def test_copied_code_directory_plans_all_jobs_without_siblings(
    tmp_path: Path,
) -> None:
    copied = _copy_code(tmp_path)
    output_root = copied / "isolated-results"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LLM_MODEL_NAME", None)
    env.pop("LLM_BASE_URL", None)
    env.pop("LLM_API_KEY", None)

    result = subprocess.run(
        [
            "bash",
            str(copied / "run_tau2_baselines.sh"),
            "--dry-run",
            "--verify-bundle",
            "--run-id",
            "isolated",
            "--output-root",
            str(output_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (output_root / "isolated" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["jobs"]) == 18
    copied_root = copied.resolve()
    for job in manifest["jobs"]:
        assert Path(job["repo"]).is_relative_to(copied_root)
        assert Path(job["output_dir"]).is_relative_to(copied_root)
        assert Path(job["log_path"]).is_relative_to(copied_root)
        assert Path(job["environment"]["TAU2_REPO"]).is_relative_to(copied_root)
        assert Path(job["environment"]["TAU2_DATA_DIR"]).is_relative_to(copied_root)


def test_all_vendored_tau2_runners_import_without_writing_vendor(
    tmp_path: Path,
) -> None:
    from vendor_bundle import verify_vendor_bundle

    vendor = CODE_DIR / "vendor"
    verify_vendor_bundle(vendor)
    for name in BASELINES:
        output_dir = tmp_path / name
        env = os.environ.copy()
        for proxy_name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            env.pop(proxy_name, None)
        env.update(
            {
                "NO_PROXY": "*",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(CODE_DIR / "runtime_shim"),
                        str(vendor / name / "src"),
                        str(vendor / "tau2-bench" / "src"),
                    ]
                ),
                "TAU2_ASSIGNED_OUTPUT_DIR": str(output_dir),
                "TAU2_USER_MODEL": "gpt-5.1-2025-11-13",
                "TAU2_RUN_SEED": "300",
                "TAU2_API_USAGE_LOG": str(output_dir / "api_usage.jsonl"),
                "TAU2_REPO": str(vendor / "tau2-bench"),
                "TAU2_DATA_DIR": str(vendor / "tau2-bench" / "data"),
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", "import tau2_runner; print(tau2_runner.__file__)"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert str(vendor / name / "src" / "tau2_runner.py") in result.stdout
    verify_vendor_bundle(vendor)


def test_api_requirements_are_local_and_exclude_heavy_model_stack() -> None:
    requirements = CODE_DIR / "requirements-api.txt"
    lines = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    normalized = "\n".join(lines).lower()

    assert "-e ./vendor/tau2-bench" in lines
    assert any(line.lower().startswith("openai") for line in lines)
    assert any(line.lower().startswith("tqdm") for line in lines)
    for banned in ("torch", "transformers", "vllm", "cuda", "voice", "multiwoz"):
        assert banned not in normalized
