from __future__ import annotations

import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from multiwoz_prompt_context import prepare_multiwoz_context_bundles
from evolve_state import record_evolution_event
from memory import apply_memory_update, seed_memory_snapshot, summarize_memory_snapshot
from reasoning_bank import (
    apply_reasoning_update,
    derive_reasoning_entries,
    seed_reasoning_bank,
    summarize_reasoning_bank,
)
from retrieval import (
    build_openai_compatible_embed_texts,
    render_graph_retrieval_section,
    render_memory_retrieval_section,
    render_reasoning_retrieval_section,
    retrieve_graph_updates,
    retrieve_memory_entries,
    retrieve_reasoning_entries,
)
from rewards import build_reward_signal, render_trajectory_text
from state_graph import apply_state_graph_update, seed_state_graph, summarize_state_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIWOZ_SRC = PROJECT_ROOT / "multiwoz" / "src"
RESULTS_ROOT = PROJECT_ROOT / "results" / "multiwoz"

if str(MULTIWOZ_SRC) not in sys.path:
    sys.path.insert(0, str(MULTIWOZ_SRC))

from speakrl_prompting_repro.backends import TransformersChatBackend
from speakrl_prompting_repro.config import load_config
from speakrl_prompting_repro.judge import SuccessJudge
from speakrl_prompting_repro.multiwoz_loader import MultiWOZGoalLoader
from speakrl_prompting_repro.openrouter_client import OpenRouterClient
from speakrl_prompting_repro.prompts import AGENT_SYSTEM_PROMPT
from speakrl_prompting_repro.rollout import PromptingEpisodeRunner
from speakrl_prompting_repro.simulator import UserSimulator
from speakrl_prompting_repro.tools_env import MultiWOZToolEnvironment


DEFAULTS = {
    "LLM_BACKEND": "transformers",
    "LLM_MODEL_NAME": "Qwen3.5-27B",
    "LLM_MODEL_PATH": "/sde/llm_weights/Qwen3.5-27B",
    "LLM_BASE_URL": "",
    "LLM_API_KEY": "",
    "MODE": "base_lm",
    "SOP_TEXT": "static",
    "NUM_CYCLES": 1,
    "SPLIT": "test",
    "LIMIT": 0,
    "PROC_PER_NODE": 1,
    "MAX_TURNS": 20,
    "TEMPERATURE": 0.2,
    "SEED": 42,
    "REPAIR_MAX_NEW_TOKENS": 256,
    "MAX_NEW_TOKENS": 256,
    "EVOLVE_COMPONENTS": "",
    "RESUME": "1",
    "MULTIWOZ_DEBUG_TIMING": "0",
    "EMBEDDING_MODEL": "text-embedding-v4",
    "EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "RETRIEVAL_TOP_K": 3,
}

OPENAI_BACKEND_ALIASES = {
    "openai",
    "openai_compatible",
    "openai-compatible",
    "vllm",
    "litellm_vllm",
}

_BACKEND_CACHE: dict[tuple[str, str, str, str, str, str, str], Any] = {}


@dataclass(frozen=True)
class MultiWOZRunnerConfig:
    llm_backend: str
    llm_model_name: str
    llm_model_path: str
    llm_base_url: str
    llm_api_key: str
    gpu_ids: list[str]
    proc_per_node: int
    mode: str
    sop_text: str
    num_cycles: int
    multiwoz_config: Path
    multiwoz_data_json: Path
    multiwoz_db_dir: Path
    split: str
    limit: int
    max_turns: int
    temperature: float
    seed: int
    max_new_tokens: int = DEFAULTS["MAX_NEW_TOKENS"]
    evolve_components: list[str] = field(default_factory=list)
    repair_max_new_tokens: int = DEFAULTS["REPAIR_MAX_NEW_TOKENS"]
    debug_timing: bool = False
    results_root: Path = RESULTS_ROOT
    embedding_model: str = DEFAULTS["EMBEDDING_MODEL"]
    embedding_base_url: str = DEFAULTS["EMBEDDING_BASE_URL"]
    embedding_api_key: str = ""
    retrieval_top_k: int = DEFAULTS["RETRIEVAL_TOP_K"]


def parse_str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def parse_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def parse_csv_env(name: str, default: str = "") -> list[str]:
    value = os.getenv(name, default).strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool_env(name: str, default: str = "0") -> bool:
    value = parse_str_env(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def parse_gpu_list(raw: str) -> list[str]:
    if not raw.strip():
        return [""]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_config_relative_path(config_path: Path, raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path
    return (config_path.parent / raw_path).resolve()


def _load_repro_defaults(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    repro = load_config(config_path)
    return {
        "multiwoz_data_json": _resolve_config_relative_path(
            config_path, repro.data.multiwoz_data_json
        ),
        "multiwoz_db_dir": _resolve_config_relative_path(config_path, repro.data.db_dir),
        "llm_model_name": repro.models.agent_model,
        "llm_base_url": repro.secrets.api_base_url,
        "llm_api_key": repro.secrets.api_key,
        "max_turns": repro.runtime.max_turns,
        "temperature": repro.runtime.temperature,
        "seed": repro.runtime.seed,
    }


def load_runner_config() -> MultiWOZRunnerConfig:
    default_config_path = PROJECT_ROOT / "multiwoz" / "repro_config.json"
    config_path = Path(parse_str_env("MULTIWOZ_CONFIG", str(default_config_path))).resolve()
    repro_defaults = _load_repro_defaults(config_path)

    data_json_default = str(
        repro_defaults.get("multiwoz_data_json", PROJECT_ROOT / "multiwoz" / "MULTIWOZ2.4" / "data" / "data.json")
    )
    db_dir_default = str(
        repro_defaults.get("multiwoz_db_dir", PROJECT_ROOT / "multiwoz" / "MULTIWOZ2.4" / "db")
    )

    return MultiWOZRunnerConfig(
        llm_backend=parse_str_env("LLM_BACKEND", DEFAULTS["LLM_BACKEND"]),
        llm_model_name=parse_str_env(
            "LLM_MODEL_NAME",
            str(repro_defaults.get("llm_model_name", DEFAULTS["LLM_MODEL_NAME"])),
        ),
        llm_model_path=parse_str_env("LLM_MODEL_PATH", DEFAULTS["LLM_MODEL_PATH"]),
        llm_base_url=parse_str_env(
            "LLM_BASE_URL",
            str(repro_defaults.get("llm_base_url", DEFAULTS["LLM_BASE_URL"])),
        ),
        llm_api_key=parse_str_env(
            "LLM_API_KEY",
            str(repro_defaults.get("llm_api_key", DEFAULTS["LLM_API_KEY"])),
        ),
        gpu_ids=parse_gpu_list(parse_str_env("GPU", "")),
        proc_per_node=parse_int_env("PROC_PER_NODE", DEFAULTS["PROC_PER_NODE"]),
        mode=parse_str_env("MODE", DEFAULTS["MODE"]),
        sop_text=parse_str_env("SOP_TEXT", DEFAULTS["SOP_TEXT"]),
        num_cycles=max(1, parse_int_env("NUM_CYCLES", DEFAULTS["NUM_CYCLES"])),
        multiwoz_config=config_path,
        multiwoz_data_json=Path(
            parse_str_env("MULTIWOZ_DATA_JSON", data_json_default)
        ).resolve(),
        multiwoz_db_dir=Path(
            parse_str_env("MULTIWOZ_DB_DIR", db_dir_default)
        ).resolve(),
        split=parse_str_env("SPLIT", DEFAULTS["SPLIT"]),
        limit=parse_int_env("LIMIT", DEFAULTS["LIMIT"]),
        max_turns=parse_int_env(
            "MAX_TURNS",
            int(repro_defaults.get("max_turns", DEFAULTS["MAX_TURNS"])),
        ),
        temperature=parse_float_env(
            "TEMPERATURE",
            float(repro_defaults.get("temperature", DEFAULTS["TEMPERATURE"])),
        ),
        seed=parse_int_env(
            "SEED",
            int(repro_defaults.get("seed", DEFAULTS["SEED"])),
        ),
        max_new_tokens=parse_int_env("MAX_NEW_TOKENS", DEFAULTS["MAX_NEW_TOKENS"]),
        evolve_components=parse_csv_env(
            "EVOLVE_COMPONENTS", DEFAULTS["EVOLVE_COMPONENTS"]
        ),
        repair_max_new_tokens=parse_int_env(
            "REPAIR_MAX_NEW_TOKENS", DEFAULTS["REPAIR_MAX_NEW_TOKENS"]
        ),
        debug_timing=parse_bool_env(
            "MULTIWOZ_DEBUG_TIMING", DEFAULTS["MULTIWOZ_DEBUG_TIMING"]
        ),
        embedding_model=parse_str_env("EMBEDDING_MODEL", DEFAULTS["EMBEDDING_MODEL"]),
        embedding_base_url=parse_str_env(
            "EMBEDDING_BASE_URL",
            DEFAULTS["EMBEDDING_BASE_URL"],
        ),
        embedding_api_key=parse_str_env(
            "EMBEDDING_API_KEY",
            parse_str_env("DASHSCOPE_API_KEY", ""),
        ),
        retrieval_top_k=max(
            0,
            parse_int_env("RETRIEVAL_TOP_K", DEFAULTS["RETRIEVAL_TOP_K"]),
        ),
    )


def _timing_log(config: MultiWOZRunnerConfig, stage: str, **fields: Any) -> None:
    if not config.debug_timing:
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[multiwoz-timing {now}] {stage}{(' ' + payload) if payload else ''}", flush=True)


def build_run_dir(config: MultiWOZRunnerConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.results_root / config.split / f"{config.mode}_{timestamp}"


def _resolve_worker_device_ids(
    gpu_ids: list[str],
) -> list[str]:
    if not gpu_ids:
        return [""]
    deduped_devices: list[str] = []
    seen: set[str] = set()
    for device in gpu_ids:
        normalized = str(device).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_devices.append(normalized)
    return deduped_devices or [""]


def _prepare_runtime_device_for_current_process(
    config: MultiWOZRunnerConfig,
    *,
    requested_device_id: str,
) -> str:
    backend_name = config.llm_backend.strip().lower()
    if backend_name == "transformers" and str(requested_device_id).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_device_id).strip()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        return "0"
    return requested_device_id


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def compute_summary(
    *,
    tasks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    split: str,
    backend: str,
    model_name: str,
    num_cycles: int,
    evolve_components: list[str],
    serial_execution: bool,
    total_n: int | None = None,
) -> dict[str, Any]:
    if len(tasks) == len(records):
        task_metadata = tasks
    else:
        task_metadata = [
            {
                "primary_domain": record.get("primary_domain", ""),
                "domains": record.get("domains", []),
            }
            for record in records
        ]

    paired = list(zip(task_metadata, records))
    success_values = [float(record["success"]) for _, record in paired]
    turn_values = [float(record["turns"]) for _, record in paired]
    resolved_total_n = total_n if total_n is not None else len(tasks)

    primary_domain_breakdown: dict[str, dict[str, float | int]] = {}
    for primary_domain in sorted(
        {
            str(task.get("primary_domain", "")).strip()
            for task, _ in paired
            if str(task.get("primary_domain", "")).strip()
        }
    ):
        domain_successes = [
            float(record["success"])
            for task, record in paired
            if task.get("primary_domain") == primary_domain
        ]
        primary_domain_breakdown[primary_domain] = {
            "n": len(domain_successes),
            "success_mean": _mean(domain_successes),
        }

    domain_count_breakdown: dict[str, dict[str, float | int]] = {}
    for domain_count in (1, 2, 3):
        count_successes = [
            float(record["success"])
            for task, record in paired
            if len(task.get("domains", [])) == domain_count
        ]
        domain_count_breakdown[str(domain_count)] = {
            "n": len(count_successes),
            "success_mean": _mean(count_successes),
        }

    extra_successes = [
        float(record["success"])
        for task, record in paired
        if len(task.get("domains", [])) >= 4
    ]
    if extra_successes:
        domain_count_breakdown["4_plus"] = {
            "n": len(extra_successes),
            "success_mean": _mean(extra_successes),
        }

    return {
        "n": len(paired),
        "total_n": resolved_total_n,
        "completed_n": len(paired),
        "remaining_n": max(0, resolved_total_n - len(paired)),
        "success_mean": _mean(success_values),
        "turns_mean": _mean(turn_values),
        "primary_domain_breakdown": primary_domain_breakdown,
        "domain_count_breakdown": domain_count_breakdown,
        "split": split,
        "backend": backend,
        "model_name": model_name,
        "num_cycles": num_cycles,
        "evolve_components": evolve_components,
        "serial_execution": serial_execution,
    }


def build_backend(config: MultiWOZRunnerConfig, *, device_id: str = "") -> Any:
    backend_name = config.llm_backend.strip().lower()
    if backend_name == "transformers":
        return TransformersChatBackend(
            model_name=config.llm_model_name,
            model_path=config.llm_model_path,
            device_id=device_id,
            repair_max_new_tokens=config.repair_max_new_tokens,
            max_new_tokens=config.max_new_tokens,
        )
    if backend_name in OPENAI_BACKEND_ALIASES:
        return OpenRouterClient(
            api_key=config.llm_api_key or None,
            base_url=config.llm_base_url or None,
        )
    raise ValueError(f"Unsupported LLM backend: {config.llm_backend}")


def _backend_cache_key(
    config: MultiWOZRunnerConfig,
    *,
    device_id: str = "",
) -> tuple[str, str, str, str, str, str, str]:
    return (
        config.llm_backend.strip().lower(),
        config.llm_model_name,
        config.llm_model_path,
        config.llm_base_url,
        config.llm_api_key,
        device_id,
        os.getenv("CUDA_VISIBLE_DEVICES", ""),
    )


def _get_cached_backend(config: MultiWOZRunnerConfig, *, device_id: str = "") -> Any:
    key = _backend_cache_key(config, device_id=device_id)
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        _timing_log(
            config,
            "backend_cache_hit",
            device_id=device_id or "cpu",
        )
        return cached
    start = time.perf_counter()
    backend = build_backend(config, device_id=device_id)
    _BACKEND_CACHE[key] = backend
    _timing_log(
        config,
        "backend_cache_miss",
        device_id=device_id or "cpu",
        load_seconds=f"{time.perf_counter() - start:.2f}",
    )
    return backend


def _build_episode_runner(
    config: MultiWOZRunnerConfig,
    *,
    agent_system_prompt: str,
    device_id: str = "",
    backend: Any | None = None,
) -> PromptingEpisodeRunner:
    runtime_backend = backend if backend is not None else build_backend(config, device_id=device_id)
    simulator = UserSimulator(
        client=runtime_backend,
        model=config.llm_model_name,
        temperature=config.temperature,
        seed=config.seed,
    )
    judge = SuccessJudge(
        client=runtime_backend,
        model=config.llm_model_name,
        temperature=config.temperature,
        seed=config.seed,
    )
    tools_env = MultiWOZToolEnvironment(config.multiwoz_db_dir, seed=config.seed)
    return PromptingEpisodeRunner(
        client=runtime_backend,
        agent_model=config.llm_model_name,
        simulator=simulator,
        judge=judge,
        tools_env=tools_env,
        temperature=config.temperature,
        seed=config.seed,
        max_turns=config.max_turns,
        agent_system_prompt=agent_system_prompt,
    )


def _result_record(task: Any, result: Any) -> dict[str, Any]:
    return {
        "dialogue_id": result.dialogue_id,
        "success": result.success,
        "turns": result.turns,
        "judge_reason": result.judge_reason,
        "domains": getattr(task, "domains", []),
        "primary_domain": getattr(task, "primary_domain", ""),
        "history": result.history,
        "trace": result.trace,
    }


def _infra_error_record(task: Any, exc: Exception) -> dict[str, Any]:
    return {
        "dialogue_id": getattr(task, "dialogue_id", ""),
        "success": 0,
        "turns": 0,
        "judge_reason": f"infrastructure error: {type(exc).__name__}: {exc}",
        "domains": getattr(task, "domains", []),
        "primary_domain": getattr(task, "primary_domain", ""),
        "history": [],
        "trace": {
            "infra_error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    }


def _run_one_task(
    config: MultiWOZRunnerConfig,
    task: Any,
    agent_system_prompt: str,
    device_id: str = "",
) -> dict[str, Any]:
    start = time.perf_counter()
    dialogue_id = str(getattr(task, "dialogue_id", "unknown"))
    _timing_log(
        config,
        "task_start",
        dialogue_id=dialogue_id,
        device_id=device_id or "cpu",
    )
    backend = _get_cached_backend(config, device_id=device_id)
    runner = _build_episode_runner(
        config,
        agent_system_prompt=agent_system_prompt,
        device_id=device_id,
        backend=backend,
    )
    runner_ready = time.perf_counter()
    result = runner.run(task)
    done = time.perf_counter()
    _timing_log(
        config,
        "task_done",
        dialogue_id=dialogue_id,
        device_id=device_id or "cpu",
        build_runner_seconds=f"{runner_ready - start:.2f}",
        rollout_seconds=f"{done - runner_ready:.2f}",
        total_seconds=f"{done - start:.2f}",
        success=result.success,
        turns=result.turns,
    )
    return _result_record(task, result)


def _run_one_task_safe(
    config: MultiWOZRunnerConfig,
    task: Any,
    agent_system_prompt: str,
    device_id: str = "",
) -> dict[str, Any]:
    try:
        return _run_one_task(
            config,
            task,
            agent_system_prompt,
            device_id=device_id,
        )
    except Exception as exc:  # pragma: no cover - exercised via tests using stubs
        _timing_log(
            config,
            "task_error",
            dialogue_id=str(getattr(task, "dialogue_id", "unknown")),
            device_id=device_id or "cpu",
            error_type=type(exc).__name__,
        )
        return _infra_error_record(task, exc)


def _baseline_worker_main(
    config: MultiWOZRunnerConfig,
    requested_device_id: str,
    task_queue: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    boot_start = time.perf_counter()
    runtime_device_id = _prepare_runtime_device_for_current_process(
        config,
        requested_device_id=requested_device_id,
    )
    _timing_log(
        config,
        "worker_bootstrap",
        requested_device_id=requested_device_id or "cpu",
        runtime_device_id=runtime_device_id or "cpu",
    )
    try:
        _get_cached_backend(config, device_id=runtime_device_id)
        _timing_log(
            config,
            "worker_ready",
            requested_device_id=requested_device_id or "cpu",
            runtime_device_id=runtime_device_id or "cpu",
            bootstrap_seconds=f"{time.perf_counter() - boot_start:.2f}",
        )
        ready_queue.put({"status": "ready", "device_id": requested_device_id})
    except Exception as exc:
        _timing_log(
            config,
            "worker_bootstrap_error",
            requested_device_id=requested_device_id or "cpu",
            runtime_device_id=runtime_device_id or "cpu",
            bootstrap_seconds=f"{time.perf_counter() - boot_start:.2f}",
            error_type=type(exc).__name__,
        )
        ready_queue.put(
            {
                "status": "error",
                "device_id": requested_device_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
        return

    while True:
        packet = task_queue.get()
        if packet is None:
            break
        _timing_log(
            config,
            "worker_task_received",
            requested_device_id=requested_device_id or "cpu",
            runtime_device_id=runtime_device_id or "cpu",
            index=packet.get("index", -1),
            dialogue_id=str(getattr(packet.get("task"), "dialogue_id", "unknown")),
        )
        task = packet["task"]
        record = _run_one_task_safe(
            config,
            task,
            packet["agent_system_prompt"],
            device_id=runtime_device_id,
        )
        result_queue.put(
            {
                "status": "ok",
                "index": packet["index"],
                "record": record,
            }
        )
        _timing_log(
            config,
            "worker_task_published",
            requested_device_id=requested_device_id or "cpu",
            runtime_device_id=runtime_device_id or "cpu",
            index=packet.get("index", -1),
            dialogue_id=str(record.get("dialogue_id", "unknown")),
        )


def run_baseline(
    config: MultiWOZRunnerConfig,
    tasks: list[Any],
    *,
    run_dir: Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    bench_start = time.perf_counter()
    _timing_log(
        config,
        "baseline_start",
        run_dir=str(run_dir),
        split=config.split,
        total_tasks=len(tasks),
        workers=len(_resolve_worker_device_ids(config.gpu_ids)),
    )
    bundles = prepare_multiwoz_context_bundles(
        mode=config.mode,
        sop_text=config.sop_text,
        run_dir=run_dir,
        tasks=tasks,
    )
    task_list = sorted(tasks, key=lambda task: task.dialogue_id)
    worker_device_ids = _resolve_worker_device_ids(config.gpu_ids)

    def _prompt_for(task: Any) -> str:
        bundle_prompt = bundles[task.primary_domain].current_system_prompt
        return "\n\n".join(
            part.strip()
            for part in [AGENT_SYSTEM_PROMPT, bundle_prompt]
            if str(part).strip()
        )

    records_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    total_n = len(task_list)
    task_metadata = [
        {"primary_domain": task.primary_domain, "domains": task.domains}
        for task in task_list
    ]

    record_map: dict[str, dict[str, Any]] = {}
    if resume and records_path.exists():
        try:
            existing = json.loads(records_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
        if isinstance(existing, list):
            for item in existing:
                if isinstance(item, dict):
                    dialogue_id = str(item.get("dialogue_id", "")).strip()
                    if dialogue_id:
                        record_map[dialogue_id] = item

    pending_tasks = [task for task in task_list if task.dialogue_id not in record_map]
    _timing_log(
        config,
        "baseline_task_partition",
        total_n=total_n,
        resumed_n=len(record_map),
        pending_n=len(pending_tasks),
    )

    def _is_success_record(record: dict[str, Any]) -> bool:
        try:
            return float(record.get("success", 0)) >= 0.5
        except (TypeError, ValueError):
            return False

    completed_success = sum(
        1 for record in record_map.values() if _is_success_record(record)
    )
    completed_fail = max(0, len(record_map) - completed_success)

    def _ordered_records() -> list[dict[str, Any]]:
        return [
            record_map[task.dialogue_id]
            for task in task_list
            if task.dialogue_id in record_map
        ]

    def _flush_partial() -> None:
        ordered = _ordered_records()
        _write_json_file(records_path, ordered)
        partial_summary = compute_summary(
            tasks=task_metadata,
            records=ordered,
            split=config.split,
            backend=config.llm_backend,
            model_name=config.llm_model_name,
            num_cycles=1,
            evolve_components=[],
            serial_execution=len(worker_device_ids) <= 1,
            total_n=total_n,
        )
        partial_summary["run_dir"] = str(run_dir)
        partial_summary["timestamp"] = datetime.now().isoformat()
        _write_json_file(summary_path, partial_summary)

    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(
            total=total_n,
            initial=min(total_n, len(record_map)),
            desc=f"multiwoz/{config.split}",
        )
    except Exception:
        progress = None

    def _refresh_progress_postfix() -> None:
        if progress is None:
            return
        progress.set_postfix(
            success=completed_success,
            fail=completed_fail,
            refresh=True,
        )

    try:
        _refresh_progress_postfix()
        _flush_partial()

        if len(worker_device_ids) <= 1:
            runtime_device_id = _prepare_runtime_device_for_current_process(
                config,
                requested_device_id=worker_device_ids[0],
            )
            _timing_log(
                config,
                "baseline_single_worker",
                runtime_device_id=runtime_device_id or "cpu",
                pending_n=len(pending_tasks),
            )
            for task in pending_tasks:
                record = _run_one_task_safe(
                    config,
                    task,
                    _prompt_for(task),
                    device_id=runtime_device_id,
                )
                record_map[record["dialogue_id"]] = record
                if _is_success_record(record):
                    completed_success += 1
                else:
                    completed_fail += 1
                _flush_partial()
                if progress is not None:
                    progress.update(1)
                    _refresh_progress_postfix()
            ordered_records = _ordered_records()
            _timing_log(
                config,
                "baseline_done",
                completed_n=len(ordered_records),
                total_n=total_n,
                elapsed_seconds=f"{time.perf_counter() - bench_start:.2f}",
            )
            return ordered_records

        if not pending_tasks:
            ordered_records = _ordered_records()
            _timing_log(
                config,
                "baseline_done",
                completed_n=len(ordered_records),
                total_n=total_n,
                elapsed_seconds=f"{time.perf_counter() - bench_start:.2f}",
            )
            return ordered_records

        mp_ctx = mp.get_context("spawn")
        task_queue = mp_ctx.Queue(maxsize=max(1, len(worker_device_ids) * 2))
        ready_queue = mp_ctx.Queue()
        result_queue = mp_ctx.Queue()
        workers = [
            mp_ctx.Process(
                target=_baseline_worker_main,
                args=(config, device_id, task_queue, ready_queue, result_queue),
                daemon=False,
            )
            for device_id in worker_device_ids
        ]
        for worker in workers:
            worker.start()
        _timing_log(
            config,
            "baseline_workers_started",
            worker_count=len(workers),
            devices=",".join(worker_device_ids),
        )

        ready = 0
        ready_wait_start = time.perf_counter()
        while ready < len(workers):
            packet = ready_queue.get()
            if packet.get("status") != "ready":
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                raise RuntimeError(f"MultiWOZ worker failed to start: {packet}")
            ready += 1
            _timing_log(
                config,
                "baseline_worker_ready",
                ready=f"{ready}/{len(workers)}",
                device_id=packet.get("device_id", ""),
                elapsed_seconds=f"{time.perf_counter() - ready_wait_start:.2f}",
            )

        try:
            for index, task in enumerate(pending_tasks):
                task_queue.put(
                    {
                        "index": index,
                        "task": task,
                        "agent_system_prompt": _prompt_for(task),
                    }
                )
            for _ in workers:
                task_queue.put(None)

            completed_pending = 0
            wait_start = time.perf_counter()
            last_wait_log = time.perf_counter()
            while completed_pending < len(pending_tasks):
                try:
                    packet = result_queue.get(timeout=30.0)
                except Exception:
                    alive_workers = sum(1 for worker in workers if worker.is_alive())
                    if progress is not None:
                        progress.set_postfix(
                            success=completed_success,
                            fail=completed_fail,
                            inflight=max(0, len(pending_tasks) - completed_pending),
                            alive_workers=alive_workers,
                            refresh=True,
                        )
                    now = time.perf_counter()
                    if config.debug_timing and now - last_wait_log >= 30.0:
                        _timing_log(
                            config,
                            "baseline_waiting_result",
                            completed_pending=completed_pending,
                            pending_n=len(pending_tasks),
                            alive_workers=alive_workers,
                            waited_seconds=f"{now - wait_start:.2f}",
                        )
                        last_wait_log = now
                    if alive_workers == 0:
                        raise RuntimeError(
                            "All MultiWOZ workers exited before returning results; "
                            "this usually means worker-side OOM or fatal runtime errors."
                        )
                    continue
                task = pending_tasks[int(packet.get("index", -1))]
                if packet.get("status") == "ok":
                    record = packet["record"]
                else:
                    record = _infra_error_record(
                        task,
                        RuntimeError(
                            f"worker packet error: {packet.get('error_type', 'UnknownError')}: "
                            f"{packet.get('error', 'unknown')}"
                        ),
                    )
                record_map[record["dialogue_id"]] = record
                if _is_success_record(record):
                    completed_success += 1
                else:
                    completed_fail += 1
                completed_pending += 1
                _flush_partial()
                if progress is not None:
                    progress.update(1)
                    _refresh_progress_postfix()
                _timing_log(
                    config,
                    "baseline_result_received",
                    completed_pending=completed_pending,
                    pending_n=len(pending_tasks),
                    dialogue_id=str(record.get("dialogue_id", "unknown")),
                    wait_elapsed_seconds=f"{time.perf_counter() - wait_start:.2f}",
                )
        finally:
            for worker in workers:
                worker.join(timeout=10.0)
                if worker.is_alive():
                    worker.terminate()
        ordered_records = _ordered_records()
        _timing_log(
            config,
            "baseline_done",
            completed_n=len(ordered_records),
            total_n=total_n,
            elapsed_seconds=f"{time.perf_counter() - bench_start:.2f}",
        )
        return ordered_records
    finally:
        if progress is not None:
            progress.close()


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _collect_warning_candidates(value: Any, out: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "warning",
                "not found",
                "not implemented",
                "unsupported",
                "exception",
                "traceback",
                "timeout",
                "infra",
            )
        ):
            out.append(text)
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered_key = str(key).lower()
            if lowered_key in {
                "warning",
                "warnings",
                "runtime_warning",
                "runtime_warnings",
                "log",
                "logs",
                "error",
                "errors",
                "message",
                "messages",
            }:
                _collect_warning_candidates(nested, out)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_warning_candidates(item, out)


def _record_warnings(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    _collect_warning_candidates(record.get("trace"), candidates)
    _collect_warning_candidates(record.get("warnings"), candidates)
    _collect_warning_candidates(record.get("judge_reason"), candidates)
    for message in record.get("history", []) or []:
        if isinstance(message, dict):
            _collect_warning_candidates(message.get("content"), candidates)
    deduped: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        key = text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text.strip())
    return deduped[:12]


def _memory_delta(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_conf = {
        str(entry.get("id", "")): float(entry.get("confidence", 0.0))
        for entry in (before.get("entries") or [])
        if entry.get("id")
    }
    changed: list[dict[str, Any]] = []
    for entry in after.get("entries") or []:
        entry_id = str(entry.get("id", ""))
        if not entry_id:
            continue
        confidence = float(entry.get("confidence", 0.0))
        if entry_id not in before_conf or abs(before_conf[entry_id] - confidence) > 1e-9:
            changed.append(entry)
    return changed


def _graph_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_node_conf = {
        str(node.get("id", "")): float(node.get("confidence", 0.0))
        for node in (before.get("nodes") or [])
        if node.get("id")
    }
    changed_nodes: list[dict[str, Any]] = []
    for node in after.get("nodes") or []:
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        confidence = float(node.get("confidence", 0.0))
        if node_id not in before_node_conf or abs(before_node_conf[node_id] - confidence) > 1e-9:
            changed_nodes.append(node)
    before_edges = {
        (
            str(edge.get("source", "")),
            str(edge.get("target", "")),
            str(edge.get("type", "")),
            str(edge.get("condition", "")),
        )
        for edge in (before.get("edges") or [])
    }
    changed_edges = [
        edge
        for edge in (after.get("edges") or [])
        if (
            str(edge.get("source", "")),
            str(edge.get("target", "")),
            str(edge.get("type", "")),
            str(edge.get("condition", "")),
        )
        not in before_edges
    ]
    return {
        "nodes": changed_nodes,
        "edges": changed_edges,
    }


def _render_meta_sop_snapshot(meta_sop_by_domain: dict[str, str]) -> str:
    sections: list[str] = []
    for domain, meta_sop_text in sorted(meta_sop_by_domain.items()):
        meta_sop_text = str(meta_sop_text or "").strip()
        if not meta_sop_text:
            continue
        sections.append(f"## {domain}\n\n{meta_sop_text}")
    return "\n\n".join(sections).strip()


def _build_retrieval_query(task_query: str, task_family: str) -> str:
    sections: list[str] = []
    normalized_family = str(task_family or "").strip()
    normalized_query = str(task_query or "").strip()
    if normalized_family:
        sections.append(f"Family: {normalized_family}")
    if normalized_query:
        sections.append(f"Task: {normalized_query}")
    return "\n".join(sections).strip()


def _build_evolving_agent_prompt(
    base_prompt: str,
    bundle_prompt: str,
    memory_snapshot: dict[str, Any],
    graph_snapshot: dict[str, Any],
    reasoning_bank_snapshot: dict[str, Any] | None,
    meta_sop_text: str,
    evolve_components: list[str],
    task_query: str = "",
    task_family: str = "",
    retrieval_top_k: int = 0,
    embed_texts=None,
) -> str:
    sections = [base_prompt.strip()]
    normalized_bundle_prompt = bundle_prompt.strip()
    normalized_meta_sop = meta_sop_text.strip()
    if normalized_bundle_prompt:
        sections.append(normalized_bundle_prompt)
    if (
        "meta_sop" in evolve_components
        and normalized_meta_sop
        and normalized_meta_sop not in normalized_bundle_prompt
    ):
        sections.append(f"Current Meta-SOP:\n{meta_sop_text.strip()}")
    retrieval_query = _build_retrieval_query(task_query, task_family)
    use_retrieval = bool(embed_texts and retrieval_top_k > 0 and retrieval_query.strip())
    if "memory" in evolve_components and (memory_snapshot.get("entries") or []):
        if use_retrieval:
            memory_text = render_memory_retrieval_section(
                retrieve_memory_entries(
                    memory_snapshot,
                    query_text=retrieval_query,
                    family=task_family,
                    top_k=retrieval_top_k,
                    embed_texts=embed_texts,
                )
            )
            sections.append(memory_text or summarize_memory_snapshot(memory_snapshot))
        else:
            sections.append(summarize_memory_snapshot(memory_snapshot))
    if "reasoning_bank" in evolve_components and (reasoning_bank_snapshot or {}).get("entries"):
        if use_retrieval:
            reasoning_text = render_reasoning_retrieval_section(
                retrieve_reasoning_entries(
                    reasoning_bank_snapshot or {},
                    query_text=retrieval_query,
                    family=task_family,
                    top_k=retrieval_top_k,
                    embed_texts=embed_texts,
                )
            )
            sections.append(reasoning_text or summarize_reasoning_bank(reasoning_bank_snapshot or {}))
        else:
            sections.append(summarize_reasoning_bank(reasoning_bank_snapshot or {}))
    step_nodes = [
        node for node in (graph_snapshot.get("nodes") or []) if node.get("type") == "step"
    ]
    if "state_graph" in evolve_components and step_nodes:
        if use_retrieval:
            graph_text = render_graph_retrieval_section(
                retrieve_graph_updates(
                    graph_snapshot,
                    query_text=retrieval_query,
                    family=task_family,
                    top_k=retrieval_top_k,
                    embed_texts=embed_texts,
                )
            )
            sections.append(graph_text or summarize_state_graph(graph_snapshot))
        else:
            sections.append(summarize_state_graph(graph_snapshot))
    return "\n\n".join(section for section in sections if section.strip())


def run_self_evolve(
    *,
    run_dir: Path,
    tasks: list[Any],
    split: str,
    backend: str,
    model_name: str,
    mode: str,
    sop_text: str,
    num_cycles: int,
    evolve_components: list[str],
    run_episode_fn,
) -> dict[str, Any]:
    bundles = prepare_multiwoz_context_bundles(
        mode=mode,
        sop_text=sop_text,
        run_dir=run_dir,
        tasks=tasks,
    )
    retrieval_top_k = max(0, parse_int_env("RETRIEVAL_TOP_K", DEFAULTS["RETRIEVAL_TOP_K"]))
    embed_texts = build_openai_compatible_embed_texts(
        model=parse_str_env("EMBEDDING_MODEL", DEFAULTS["EMBEDDING_MODEL"]),
        api_key=parse_str_env("EMBEDDING_API_KEY", parse_str_env("DASHSCOPE_API_KEY", "")),
        base_url=parse_str_env("EMBEDDING_BASE_URL", DEFAULTS["EMBEDDING_BASE_URL"]),
    )
    memory_snapshot = seed_memory_snapshot("multiwoz")
    graph_snapshot = seed_state_graph("multiwoz")
    reasoning_bank = seed_reasoning_bank("multiwoz")
    current_meta_sop_by_domain = {
        domain: str(bundle.meta_sop_text or "")
        for domain, bundle in bundles.items()
    }
    task_list = sorted(tasks, key=lambda item: item.dialogue_id)
    all_records: list[dict[str, Any]] = []

    for cycle_index in range(num_cycles):
        cycle_dir = run_dir / f"loop_{cycle_index}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        cycle_records: list[dict[str, Any]] = []
        cycle_events: list[dict[str, Any]] = []

        for task in task_list:
            bundle = bundles[task.primary_domain]
            current_meta_sop_text = current_meta_sop_by_domain.get(task.primary_domain, "")
            agent_prompt = _build_evolving_agent_prompt(
                AGENT_SYSTEM_PROMPT,
                bundle.current_system_prompt,
                memory_snapshot,
                graph_snapshot,
                reasoning_bank,
                current_meta_sop_text,
                evolve_components,
                task_query=json.dumps(getattr(task, "raw_goal", {}), ensure_ascii=False),
                task_family=task.primary_domain,
                retrieval_top_k=retrieval_top_k,
                embed_texts=embed_texts,
            )
            try:
                record = run_episode_fn(task, agent_prompt)
            except Exception as exc:
                record = _infra_error_record(task, exc)

            runtime_warnings = _record_warnings(record)
            family_prior = task.primary_domain
            trace_data = record.get("trace")
            has_infra_error = isinstance(trace_data, dict) and bool(trace_data.get("infra_error"))
            reward_signal = build_reward_signal(
                sample_id=record["dialogue_id"],
                task_id=record["dialogue_id"],
                task_family=family_prior,
                trajectory_text=render_trajectory_text(record.get("history", [])),
                reward_value=float(record.get("success", 0)),
                termination_reason=(
                    "INFRASTRUCTURE_ERROR"
                    if has_infra_error
                    else ("SUCCESS" if int(record.get("success", 0)) == 1 else "FAILURE")
                ),
                meta_sop_text=current_meta_sop_text,
                warnings=runtime_warnings,
                domain="multiwoz",
            )
            experience_packet = reward_signal.get("experience_packet")
            if not isinstance(experience_packet, dict):
                experience_packet = {}
            family_focus = str(experience_packet.get("family_focus") or family_prior)
            update_family = family_focus or family_prior

            memory_before = deepcopy(memory_snapshot)
            graph_before = deepcopy(graph_snapshot)
            reasoning_before = deepcopy(reasoning_bank)

            if (
                "memory" in evolve_components
                and reward_signal["outcome_tag"] == "failure"
            ):
                memory_snapshot = apply_memory_update(
                    memory_snapshot,
                    family=update_family,
                    posterior_text=reward_signal["posterior_text"],
                    outcome_tag=reward_signal["outcome_tag"],
                    source_cycle=cycle_index,
                    source_task=record["dialogue_id"],
                    reward_value=reward_signal["reward_value"],
                    meta_sop_text=current_meta_sop_text,
                    task_id=record["dialogue_id"],
                    experience_packet=experience_packet,
                )
            if "state_graph" in evolve_components:
                graph_snapshot = apply_state_graph_update(
                    graph_snapshot,
                    family=update_family,
                    posterior_text=reward_signal["posterior_text"],
                    outcome_tag=reward_signal["outcome_tag"],
                    source_cycle=cycle_index,
                    source_task=record["dialogue_id"],
                    reward_value=reward_signal["reward_value"],
                    meta_sop_text=current_meta_sop_text,
                    experience_packet=experience_packet,
                )
            if "reasoning_bank" in evolve_components:
                reasoning_bank = apply_reasoning_update(
                    reasoning_bank,
                    entries=derive_reasoning_entries(
                        experience_packet=experience_packet,
                        posterior_text=reward_signal.get("posterior_digest", reward_signal["posterior_text"]),
                        family=update_family,
                        outcome_type=reward_signal["outcome_tag"],
                    ),
                    source_cycle=cycle_index,
                    source_task=record["dialogue_id"],
                    task_id=record["dialogue_id"],
                    reward_value=reward_signal["reward_value"],
                )

            memory_delta = (
                _memory_delta(memory_before, memory_snapshot)
                if "memory" in evolve_components
                else None
            )
            graph_delta = (
                _graph_delta(graph_before, graph_snapshot)
                if "state_graph" in evolve_components
                else None
            )
            reasoning_delta = None
            if "reasoning_bank" in evolve_components:
                reasoning_delta = max(
                    0,
                    len(reasoning_bank.get("entries", []) or [])
                    - len(reasoning_before.get("entries", []) or []),
                )

            cycle_events.append(
                record_evolution_event(
                    sample_id=record["dialogue_id"],
                    task_id=record["dialogue_id"],
                    cycle_id=cycle_index,
                    task_family=update_family,
                    family_prior=family_prior,
                    family_focus=family_focus,
                    outcome_tag=reward_signal["outcome_tag"],
                    reward_value=reward_signal["reward_value"],
                    posterior_text=reward_signal["posterior_text"],
                    diagnostic_flags=experience_packet.get("diagnostic_flags", []),
                    experience_packet=experience_packet,
                    memory_delta=memory_delta,
                    graph_delta=graph_delta,
                )
            )
            if reasoning_delta:
                cycle_events[-1]["reasoning_bank_delta"] = reasoning_delta
            cycle_records.append(record)

            _write_json_file(cycle_dir / "results.json", cycle_records)
            _write_jsonl_file(cycle_dir / "events.jsonl", cycle_events)
            _write_json_file(cycle_dir / "memories.json", memory_snapshot)
            _write_json_file(cycle_dir / "state_graph.json", graph_snapshot)
            _write_json_file(cycle_dir / "reasoning_bank.json", reasoning_bank)
            meta_sop_snapshot = _render_meta_sop_snapshot(current_meta_sop_by_domain)
            (cycle_dir / "meta_sop.md").write_text(
                (meta_sop_snapshot.rstrip() + "\n") if meta_sop_snapshot else "",
                encoding="utf-8",
            )

        cycle_summary = compute_summary(
            tasks=[
                {
                    "primary_domain": task.primary_domain,
                    "domains": task.domains,
                }
                for task in task_list
            ],
            records=cycle_records,
            split=split,
            backend=backend,
            model_name=model_name,
            num_cycles=num_cycles,
            evolve_components=evolve_components,
            serial_execution=True,
            total_n=len(task_list),
        )
        cycle_summary["run_dir"] = str(cycle_dir)
        cycle_summary["timestamp"] = datetime.now().isoformat()
        cycle_summary["reasoning_entry_count"] = len(reasoning_bank.get("entries", []) or [])
        _write_json_file(cycle_dir / "summary.json", cycle_summary)
        all_records.extend(cycle_records)

    _write_json_file(run_dir / "results.json", all_records)
    summary = compute_summary(
        tasks=[],
        records=all_records,
        split=split,
        backend=backend,
        model_name=model_name,
        num_cycles=num_cycles,
        evolve_components=evolve_components,
        serial_execution=True,
        total_n=len(task_list) * num_cycles,
    )
    summary["run_dir"] = str(run_dir)
    summary["timestamp"] = datetime.now().isoformat()
    summary["reasoning_entry_count"] = len(reasoning_bank.get("entries", []) or [])
    _write_json_file(run_dir / "summary.json", summary)
    return summary


def load_tasks(config: MultiWOZRunnerConfig) -> list[Any]:
    tasks = MultiWOZGoalLoader(config.multiwoz_data_json).iter_tasks(split=config.split)
    if config.limit > 0:
        return tasks[: config.limit]
    return tasks


def _config_snapshot(config: MultiWOZRunnerConfig) -> dict[str, Any]:
    return {
        "llm_backend": config.llm_backend,
        "llm_model_name": config.llm_model_name,
        "llm_model_path": config.llm_model_path,
        "llm_base_url": config.llm_base_url,
        "gpu_ids": config.gpu_ids,
        "proc_per_node": config.proc_per_node,
        "mode": config.mode,
        "sop_text": config.sop_text,
        "num_cycles": config.num_cycles,
        "evolve_components": config.evolve_components,
        "split": config.split,
        "limit": config.limit,
        "worker_device_ids": _resolve_worker_device_ids(config.gpu_ids),
        "max_turns": config.max_turns,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "debug_timing": config.debug_timing,
        "embedding_model": config.embedding_model,
        "embedding_base_url": config.embedding_base_url,
        "retrieval_top_k": config.retrieval_top_k,
        "multiwoz_config": str(config.multiwoz_config),
        "multiwoz_data_json": str(config.multiwoz_data_json),
        "multiwoz_db_dir": str(config.multiwoz_db_dir),
    }


def main() -> None:
    config = load_runner_config()
    run_dir_override = parse_str_env("RUN_DIR", "").strip()
    run_dir = Path(run_dir_override).resolve() if run_dir_override else build_run_dir(config)
    resume_baseline = parse_bool_env("RESUME", DEFAULTS["RESUME"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_file(run_dir / "config_snapshot.json", _config_snapshot(config))

    tasks = load_tasks(config)

    if config.evolve_components:
        device_id = _prepare_runtime_device_for_current_process(
            config,
            requested_device_id=_resolve_worker_device_ids(config.gpu_ids)[0],
        )
        summary = run_self_evolve(
            run_dir=run_dir,
            tasks=tasks,
            split=config.split,
            backend=config.llm_backend,
            model_name=config.llm_model_name,
            mode=config.mode,
            sop_text=config.sop_text,
            num_cycles=max(1, config.num_cycles),
            evolve_components=config.evolve_components,
            run_episode_fn=lambda task, agent_prompt: _run_one_task(
                config,
                task,
                agent_prompt,
                device_id=device_id,
            ),
        )
        _write_json_file(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    records = run_baseline(config, tasks, run_dir=run_dir, resume=resume_baseline)
    summary = compute_summary(
        tasks=[
            {
                "primary_domain": task.primary_domain,
                "domains": task.domains,
            }
            for task in tasks
        ],
        records=records,
        split=config.split,
        backend=config.llm_backend,
        model_name=config.llm_model_name,
        num_cycles=1,
        evolve_components=[],
        serial_execution=len(_resolve_worker_device_ids(config.gpu_ids)) <= 1,
        total_n=len(tasks),
    )
    summary["run_dir"] = str(run_dir)
    summary["timestamp"] = datetime.now().isoformat()
    _write_json_file(run_dir / "results.json", records)
    _write_json_file(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
