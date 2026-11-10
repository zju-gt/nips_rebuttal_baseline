from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
ENTRY = CODE_DIR / "tau2_baseline_entry.py"


def _child_env(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(CODE_DIR / "runtime_shim"),
            "TAU2_ASSIGNED_OUTPUT_DIR": str(output_dir),
            "TAU2_USER_MODEL": "gpt-5.1-2025-11-13",
            "TAU2_RUN_SEED": "300",
            "TAU2_API_USAGE_LOG": str(output_dir / "api_usage.jsonl"),
            "LLM_API_KEY": "entry-test-secret",
        }
    )
    return env


def _write_runner(repo: Path, *, write_summary: bool = True) -> None:
    src = repo / "src"
    src.mkdir(parents=True)
    summary_lines = (
        [
            "    summary = {'reward': 1.0, 'tasks': 1}",
            "    (output / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')",
            "    return summary",
        ]
        if write_summary
        else ["    return {'reward': 1.0, 'tasks': 1}"]
    )
    (src / "tau2_runner.py").write_text(
        "\n".join(
            [
                "import json",
                "from dataclasses import dataclass",
                "from pathlib import Path",
                "@dataclass(frozen=True)",
                "class RunnerConfig:",
                "    seed: int = 7",
                "    domain: str = 'retail'",
                "class TextRunConfig:",
                "    def __init__(self, **values):",
                "        self.values = values",
                "        for key, value in values.items(): setattr(self, key, value)",
                "    def model_copy(self, *, update):",
                "        return TextRunConfig(**{**self.values, **update})",
                "def build_run_dir(config): return Path('wrong-output')",
                "def load_runner_config(): return RunnerConfig()",
                "def build_tau2_config(mods, runner_cfg):",
                "    return TextRunConfig(llm_user='openai/gpt-5.2-test', llm_args_user={'reasoning': {'effort': 'medium'}})",
                "def run_benchmark():",
                "    config = load_runner_config()",
                "    output = build_run_dir(config)",
                "    output.mkdir(parents=True, exist_ok=True)",
                "    (output / 'observed_seed.txt').write_text(str(config.seed), encoding='utf-8')",
                *summary_lines,
            ]
        ),
        encoding="utf-8",
    )


def _run_entry(repo: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--repo",
            str(repo),
            "--output-dir",
            str(output_dir),
            "--method",
            "react",
        ],
        capture_output=True,
        text=True,
        env=_child_env(output_dir),
    )


def test_entry_runs_selected_repo_into_assigned_output(tmp_path: Path) -> None:
    repo = tmp_path / "baseline"
    output_dir = tmp_path / "assigned"
    _write_runner(repo)

    result = _run_entry(repo, output_dir)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "method": "react",
        "output_dir": str(output_dir.resolve()),
        "status": "succeeded",
    }
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == {
        "reward": 1.0,
        "tasks": 1,
    }
    assert (output_dir / "observed_seed.txt").read_text(encoding="utf-8") == "300"
    assert not (repo / "wrong-output").exists()


def test_entry_rejects_missing_runner_without_leaking_environment(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "missing-runner"
    repo.mkdir()

    result = _run_entry(repo, tmp_path / "assigned")

    assert result.returncode != 0
    assert "src/tau2_runner.py" in result.stderr
    assert "entry-test-secret" not in result.stderr
    assert "LLM_API_KEY" not in result.stderr


def test_entry_requires_valid_summary_file(tmp_path: Path) -> None:
    repo = tmp_path / "baseline"
    output_dir = tmp_path / "assigned"
    _write_runner(repo, write_summary=False)

    result = _run_entry(repo, output_dir)

    assert result.returncode != 0
    assert "summary.json" in result.stderr
    assert "entry-test-secret" not in result.stderr
