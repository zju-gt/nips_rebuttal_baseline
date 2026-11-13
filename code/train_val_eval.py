from __future__ import annotations

import json
import os
from dataclasses import MISSING, fields
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from time import time
from types import ModuleType
from typing import Any

from train_val_artifacts import ARTIFACT_FILES
from train_val_split import canonical_hash, sha256_file


def _frozen_context(runner: ModuleType, cfg: Any, run_dir: Path, tasks: list[Any], harness: Path, method: str) -> Any:
    if method != "living_harness":
        return runner.build_context_bundle_for_run(cfg, run_dir, tasks)
    from context_builder import load_context_bundle

    return load_context_bundle(
        mode=cfg.mode,
        sop_text=cfg.sop_text,
        domain=cfg.domain,
        benchmark_prior_text=(harness / "artifacts/benchmark_prior.md").read_text(encoding="utf-8"),
        domain_prior_text=(harness / "artifacts/domain_prior.md").read_text(encoding="utf-8"),
        meta_sop_text=(harness / "meta_sop.md").read_text(encoding="utf-8"),
        artifact_dir=harness / "artifacts",
    )


def frozen_prompts(
    runner: ModuleType, cfg: Any, tasks: list[Any], context: Any, harness: Path | None, method: str
) -> dict[str, str]:
    if method == "react":
        return {str(task.id): context.current_system_prompt for task in tasks}
    assert harness is not None
    if method not in ARTIFACT_FILES:
        raise ValueError(f"unsupported method: {method}")
    memory = json.loads((harness / "memories.json").read_text(encoding="utf-8")) if method == "living_harness" else {}
    graph = json.loads((harness / "state_graph.json").read_text(encoding="utf-8")) if method == "living_harness" else {}
    extra: dict[str, Any] = {}
    if method == "awm":
        extra["workflow_memory_snapshot"] = json.loads((harness / "workflow_memory.json").read_text(encoding="utf-8"))
    elif method == "reasoningbank":
        extra["reasoning_bank_snapshot"] = json.loads((harness / "reasoning_bank.json").read_text(encoding="utf-8"))
    embed_texts = runner._build_runner_embed_texts(cfg)
    prompts: dict[str, str] = {}
    for task in tasks:
        prompts[str(task.id)] = runner._build_evolution_context_text(
            base_prompt=context.current_system_prompt,
            memory_snapshot=memory,
            state_graph_snapshot=graph,
            evolve_components=list(cfg.evolve_components),
            task_query=runner._task_query_from_task(task),
            task_family=runner._safe_task_family(cfg.domain, task),
            retrieval_top_k=cfg.retrieval_top_k,
            embed_texts=embed_texts,
            **extra,
        )
    return prompts


def _worker_spec(runner: ModuleType, cfg: Any, *, worker_id: int, run_dir: Path) -> Any:
    overrides = {
        "worker_id": worker_id,
        "cuda_visible_devices": (cfg.gpu_ids or [""])[worker_id % len(cfg.gpu_ids or [""])],
        "run_dir": str(run_dir),
        "system_prompt_prefix": "",
        "result_batch_size": 1,
        "evolve_components": [],
    }
    kwargs: dict[str, Any] = {}
    for field in fields(runner.WorkerSpec):
        if field.name in overrides:
            kwargs[field.name] = overrides[field.name]
        elif hasattr(cfg, field.name):
            kwargs[field.name] = getattr(cfg, field.name)
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"cannot construct frozen worker: missing {field.name}")
    return runner.WorkerSpec(**kwargs)


def run_frozen_benchmark(runner: ModuleType, method: str, harness: Path | None) -> dict[str, Any]:
    selection_path = os.environ.get("TAU2_TASK_IDS_FILE")
    if not selection_path:
        raise ValueError("validation requires TAU2_TASK_IDS_FILE")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if not isinstance(selection.get("split_hash"), str):
        raise ValueError("validation selection is missing split_hash")
    cfg = runner.load_runner_config()
    if cfg.pass_hat_k != 1 or cfg.num_cycles != 1:
        raise ValueError("validation requires PASS_HAT_K=1 and NUM_CYCLES=1")
    if method != "react" and harness is None:
        raise ValueError(f"validation harness required for {method}")
    mods = runner.get_tau2_modules()
    tasks = runner.get_tasks_for_domain(mods, cfg.domain)
    if not tasks:
        raise ValueError("validation selection is empty")
    run_dir = runner.build_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    context = _frozen_context(runner, cfg, run_dir, tasks, harness, method) if harness else runner.build_context_bundle_for_run(cfg, run_dir, tasks)
    prompts = frozen_prompts(runner, cfg, tasks, context, harness, method)
    config = runner.build_tau2_config(mods, cfg)
    run_info = mods["get_info"](config)
    run_info.git_commit = runner.get_git_commit(cfg.tau2_repo)
    Results = mods["Results"]
    results = Results(info=run_info, tasks=tasks, simulations=[])
    results_path = run_dir / "results.json"
    results.save(results_path, format="json")

    trial_seed = runner._job_seed_list(cfg.seed, 1)[0]
    jobs = [
        runner.Job(
            job_id=f"{task.id}-t0-s{trial_seed}",
            task_data=task.model_dump(mode="json"),
            trial=0,
            seed=trial_seed,
            system_prompt_prefix=prompts[str(task.id)],
        )
        for task in tasks
    ]
    ctx = get_context("spawn")
    task_queue, ready_queue, result_queue = ctx.Queue(), ctx.Queue(), ctx.Queue()
    count = max(1, len(cfg.gpu_ids or [""]) * cfg.proc_per_node)
    count = min(count, len(jobs))
    workers = [
        ctx.Process(
            target=runner.worker_main,
            args=(_worker_spec(runner, cfg, worker_id=i, run_dir=run_dir), task_queue, ready_queue, result_queue),
        )
        for i in range(count)
    ]
    started = time()
    try:
        for worker in workers:
            worker.start()
        startup_deadline = time() + 600
        for _ in workers:
            while True:
                remaining = startup_deadline - time()
                if remaining <= 0:
                    raise RuntimeError("frozen workers did not start within 600 seconds")
                try:
                    packet = ready_queue.get(timeout=min(5, remaining))
                    break
                except Empty:
                    failed = [worker.exitcode for worker in workers if worker.exitcode is not None]
                    if failed:
                        raise RuntimeError(f"frozen worker exited during startup: {failed}")
            if packet.get("status") != "ready":
                raise RuntimeError(f"frozen worker failed to start: {packet}")
        for job in jobs:
            task_queue.put(job)
        for _ in workers:
            task_queue.put(None)
        from tqdm import tqdm

        with tqdm(total=len(jobs), desc=f"tau2-bench/{cfg.domain} (frozen)") as progress:
            while len(results.simulations) < len(jobs):
                try:
                    packet = result_queue.get(timeout=30)
                except Empty:
                    if any(worker.exitcode not in (None, 0) for worker in workers) or all(worker.exitcode is not None for worker in workers):
                        raise RuntimeError("frozen worker exited before completing all tasks")
                    continue
                if packet.get("status") != "ok":
                    raise RuntimeError(f"frozen worker failed: {packet}")
                batch = packet.get("simulations", [])
                if not batch:
                    raise RuntimeError("frozen worker returned empty batch")
                results.simulations.extend(mods["SimulationRun"].model_validate(value) for value in batch)
                results.save(results_path, format="json")
                progress.update(len(batch))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=10)
        task_queue.close()
        ready_queue.close()
        result_queue.close()

    actual_ids = [str(sim.task_id) for sim in results.simulations]
    expected_ids = [str(task.id) for task in tasks]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError("validation result IDs differ from selected tasks")
    success = sum(runner.is_successful_reward(runner._simulation_reward_value(sim)) for sim in results.simulations)
    summary = runner.compute_summary(
        mods,
        results,
        cfg,
        start_time=started,
        total_samples=len(jobs),
        success_count=success,
        failure_count=len(jobs) - success,
    )
    summary.update(
        phase="val",
        frozen=True,
        selected_task_ids_hash=canonical_hash(expected_ids),
        split_hash=selection["split_hash"],
        sample_timeout_seconds=cfg.sample_timeout_seconds,
        api_rpm_limit=int(os.environ.get("TAU2_API_RPM_LIMIT", "0")),
        results_path=str(results_path),
        run_dir=str(run_dir),
        harness_manifest_sha256=sha256_file(harness / "artifact_manifest.json") if harness else None,
        timestamp=datetime.now().isoformat(),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
