from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_val_artifacts import export_harness, verify_harness
from train_val_eval import _worker_spec, frozen_prompts
from train_val_split import expected_split
from runtime_shim.tau2_api_shim import ShimSettings, patch_tau2_runner
import run_train_val
import train_val_split
from repair_train_val import _needs_retry


@pytest.mark.parametrize(
    ("domain", "expected_counts"),
    [
        ("airline", (50, 35, 15)),
        ("retail", (114, 80, 34)),
        ("telecom", (114, 80, 34)),
    ],
)
def test_base_split_is_disjoint_complete_and_stable(domain: str, expected_counts: tuple[int, int, int]) -> None:
    tau2_repo = CODE_DIR / "vendor" / "tau2-bench"
    split = expected_split(tau2_repo, domain)
    assert (len(split["base_ids"]), len(split["train_ids"]), len(split["val_ids"])) == expected_counts
    assert set(split["train_ids"]).isdisjoint(split["val_ids"])
    assert set(split["train_ids"]) | set(split["val_ids"]) == set(split["base_ids"])
    assert split == expected_split(tau2_repo, domain)


def test_harness_exports_final_training_state_and_detects_changes(tmp_path: Path) -> None:
    split = {"train_ids": ["a", "b"], "val_ids": ["c"]}
    train = tmp_path / "train"
    (train / "loop_0").mkdir(parents=True)
    (train / "workflow_memory.json").write_text('{"entries": ["wrong"]}', encoding="utf-8")
    (train / "loop_0" / "workflow_memory.json").write_text('{"entries": ["final"]}', encoding="utf-8")
    summary = {"infra_error_count": 0, "missing_reward_count": 1, "num_tasks": 2, "total_samples": 2}
    (train / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (train / "results.json").write_text(json.dumps({"simulations": [{"task_id": "a"}, {"task_id": "b"}]}), encoding="utf-8")
    harness = tmp_path / "harnesses" / "awm" / "retail"
    with pytest.raises(ValueError, match="missing rewards"):
        export_harness(train_dir=train, harness_dir=harness, method="awm", domain="retail", split=split, cycles=1, model="gpt-5.2")
    summary["missing_reward_count"] = 0
    (train / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    export_harness(train_dir=train, harness_dir=harness, method="awm", domain="retail", split=split, cycles=1, model="gpt-5.2")
    assert json.loads((harness / "workflow_memory.json").read_text())["entries"] == ["final"]
    verify_harness(harness, method="awm", domain="retail", split=split)
    (harness / "workflow_memory.json").write_text('{"entries": ["changed"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_harness(harness, method="awm", domain="retail", split=split)


def test_frozen_eval_uses_saved_memory_in_prompt_but_disables_worker_updates(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "workflow_memory.json").write_text('{"entries": ["learned"]}', encoding="utf-8")
    seen = []

    def build_context(**kwargs):
        seen.append(kwargs)
        return str(kwargs["workflow_memory_snapshot"]["entries"][0])

    runner = SimpleNamespace(
        _build_runner_embed_texts=lambda cfg: None,
        _task_query_from_task=lambda task: "query",
        _safe_task_family=lambda domain, task: "family",
        _build_evolution_context_text=build_context,
    )
    cfg = SimpleNamespace(domain="retail", evolve_components=["workflow_memory"], retrieval_top_k=3)
    task = SimpleNamespace(id="one")
    prompts = frozen_prompts(runner, cfg, [task], SimpleNamespace(current_system_prompt=""), harness, "awm")
    assert prompts == {"one": "learned"}
    assert seen[0]["evolve_components"] == ["workflow_memory"]

    @dataclass
    class WorkerSpec:
        worker_id: int
        cuda_visible_devices: str
        run_dir: str
        system_prompt_prefix: str
        result_batch_size: int
        evolve_components: list[str]
        domain: str

    runner.WorkerSpec = WorkerSpec
    cfg.gpu_ids = []
    cfg.proc_per_node = 2
    worker = _worker_spec(runner, cfg, worker_id=0, run_dir=tmp_path / "val")
    assert worker.evolve_components == []
    assert worker.result_batch_size == 1


def test_task_selection_rejects_non_base_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"domain": "telecom", "task_ids": ["base-1"]}), encoding="utf-8")
    monkeypatch.setenv("TAU2_TASK_IDS_FILE", str(selection))
    module = ModuleType("fake_tau2_runner")
    module.load_runner_config = lambda: None
    module.build_tau2_config = lambda *args: None
    module.get_tasks_for_domain = lambda mods, domain: [SimpleNamespace(id="base-1")]
    settings = ShimSettings(output_dir=tmp_path, user_model="user", seed=300, usage_log=tmp_path / "usage.jsonl")
    patch_tau2_runner(module, settings)
    assert [task.id for task in module.get_tasks_for_domain({}, "telecom")] == ["base-1"]
    selection.write_text(json.dumps({"domain": "telecom", "task_ids": ["full-only"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="base split"):
        module.get_tasks_for_domain({}, "telecom")


def test_table_columns_follow_airline_retail_telecom_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    successes = {"airline": 1, "retail": 0, "telecom": 1}
    monkeypatch.setattr(run_train_val, "load_split", lambda path, **kwargs: {"val_ids": [path.stem], "train_ids": []})
    monkeypatch.setattr(run_train_val, "_summary_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(run_train_val, "verify_harness", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_val_split, "sha256_file", lambda path: "same-harness")
    for method in run_train_val.METHODS:
        for domain in run_train_val.TABLE_DOMAINS:
            job = tmp_path / "val" / method / domain
            job.mkdir(parents=True)
            (job / "summary.json").write_text(
                json.dumps({"success_count": successes[domain], "harness_manifest_sha256": "same-harness"}),
                encoding="utf-8",
            )
            (job / "results.json").write_text(json.dumps({"simulations": [{"task_id": domain}]}), encoding="utf-8")
    run_train_val._make_table(tmp_path, tmp_path)
    lines = (tmp_path / "table.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Method,Airline,Retail,Telecom,Avg"
    assert lines[1] == "GPT5.2-base,100.00,0.00,100.00,66.67"


def test_python_command_keeps_virtualenv_symlink(tmp_path: Path) -> None:
    interpreter = tmp_path / "system-python"
    interpreter.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(interpreter)
    assert run_train_val._python_command_path(venv_python) == venv_python
    assert run_train_val._python_command_path(venv_python) != interpreter


def test_retry_only_missing_rewards_and_infrastructure_errors() -> None:
    assert _needs_retry({"termination_reason": "timeout", "reward_info": None})
    assert _needs_retry({"termination_reason": "infrastructure_error", "reward_info": {"reward": 0}})
    assert not _needs_retry({"termination_reason": "user_stop", "reward_info": {"reward": 0}})
    assert not _needs_retry({"termination_reason": "user_stop", "reward_info": {"reward": 1}})
