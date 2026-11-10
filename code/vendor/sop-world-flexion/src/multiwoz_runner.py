from __future__ import annotations

import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from multiwoz_prompt_context import prepare_multiwoz_context_bundles
from evolve_state import record_evolution_event
from memory import apply_memory_update, seed_memory_snapshot, summarize_memory_snapshot
from meta_sop import build_update_context_text
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
    "WORKERS": 1,
    "PROC_PER_NODE": 1,
    "MAX_TURNS": 20,
    "TEMPERATURE": 0.2,
    "SEED": 42,
    "REPAIR_MAX_NEW_TOKENS": 256,
    "EVOLVE_COMPONENTS": "",
}

OPENAI_BACKEND_ALIASES = {
    "openai",
    "openai_compatible",
    "openai-compatible",
    "vllm",
    "litellm_vllm",
}


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
    workers: int
    max_turns: int
    temperature: float
    seed: int
    evolve_components: list[str] = field(default_factory=list)
    repair_max_new_tokens: int = DEFAULTS["REPAIR_MAX_NEW_TOKENS"]
    results_root: Path = RESULTS_ROOT


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
        workers=max(1, parse_int_env("WORKERS", DEFAULTS["WORKERS"])),
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
        evolve_components=parse_csv_env(
            "EVOLVE_COMPONENTS", DEFAULTS["EVOLVE_COMPONENTS"]
        ),
        repair_max_new_tokens=parse_int_env(
            "REPAIR_MAX_NEW_TOKENS", DEFAULTS["REPAIR_MAX_NEW_TOKENS"]
        ),
    )


def build_run_dir(config: MultiWOZRunnerConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.results_root / config.split / f"{config.mode}_{timestamp}"


def _resolve_worker_device_ids(
    gpu_ids: list[str],
    proc_per_node: int,
    workers: int,
) -> list[str]:
    devices = gpu_ids or [""]
    total_slots = max(1, len(devices) * max(1, proc_per_node))
    return [devices[index % len(devices)] for index in range(min(total_slots, max(1, workers)))]


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
        )
    if backend_name in OPENAI_BACKEND_ALIASES:
        return OpenRouterClient(
            api_key=config.llm_api_key or None,
            base_url=config.llm_base_url or None,
        )
    raise ValueError(f"Unsupported LLM backend: {config.llm_backend}")


def _build_episode_runner(
    config: MultiWOZRunnerConfig,
    *,
    agent_system_prompt: str,
    device_id: str = "",
) -> PromptingEpisodeRunner:
    backend = build_backend(config, device_id=device_id)
    simulator = UserSimulator(
        client=backend,
        model=config.llm_model_name,
        temperature=config.temperature,
        seed=config.seed,
    )
    judge = SuccessJudge(
        client=backend,
        model=config.llm_model_name,
        temperature=config.temperature,
        seed=config.seed,
    )
    tools_env = MultiWOZToolEnvironment(config.multiwoz_db_dir, seed=config.seed)
    return PromptingEpisodeRunner(
        client=backend,
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
    runner = _build_episode_runner(
        config,
        agent_system_prompt=agent_system_prompt,
        device_id=device_id,
    )
    result = runner.run(task)
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
        return _infra_error_record(task, exc)


def run_baseline(
    config: MultiWOZRunnerConfig,
    tasks: list[Any],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    bundles = prepare_multiwoz_context_bundles(
        mode=config.mode,
        sop_text=config.sop_text,
        run_dir=run_dir,
        tasks=tasks,
    )
    task_list = sorted(tasks, key=lambda task: task.dialogue_id)
    worker_device_ids = _resolve_worker_device_ids(
        config.gpu_ids,
        config.proc_per_node,
        config.workers,
    )

    def _prompt_for(task: Any) -> str:
        bundle_prompt = bundles[task.primary_domain].current_system_prompt
        return "\n\n".join(
            part.strip()
            for part in [AGENT_SYSTEM_PROMPT, bundle_prompt]
            if str(part).strip()
        )

    if config.workers <= 1:
        return [
            _run_one_task_safe(
                config,
                task,
                _prompt_for(task),
                device_id=worker_device_ids[0],
            )
            for task in task_list
        ]

    record_map: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _run_one_task_safe,
                config,
                task,
                _prompt_for(task),
                worker_device_ids[index % len(worker_device_ids)],
            ): task
            for index, task in enumerate(task_list)
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # pragma: no cover - worker crash / executor failure
                record = _infra_error_record(task, exc)
            record_map[record["dialogue_id"]] = record
    return [record_map[task.dialogue_id] for task in task_list]


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _render_meta_sop_snapshot(meta_sop_by_domain: dict[str, str]) -> str:
    sections: list[str] = []
    for domain, meta_sop_text in sorted(meta_sop_by_domain.items()):
        meta_sop_text = str(meta_sop_text or "").strip()
        if not meta_sop_text:
            continue
        sections.append(f"## {domain}\n\n{meta_sop_text}")
    return "\n\n".join(sections).strip()


def _build_evolving_agent_prompt(
    base_prompt: str,
    bundle_prompt: str,
    memory_snapshot: dict[str, Any],
    graph_snapshot: dict[str, Any],
    meta_sop_text: str,
    evolve_components: list[str],
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
    if "memory" in evolve_components and (memory_snapshot.get("entries") or []):
        sections.append(summarize_memory_snapshot(memory_snapshot))
    step_nodes = [
        node for node in (graph_snapshot.get("nodes") or []) if node.get("type") == "step"
    ]
    if "state_graph" in evolve_components and step_nodes:
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
    memory_snapshot = seed_memory_snapshot("multiwoz")
    graph_snapshot = seed_state_graph("multiwoz")
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
                current_meta_sop_text,
                evolve_components,
            )
            try:
                record = run_episode_fn(task, agent_prompt)
            except Exception as exc:
                record = _infra_error_record(task, exc)

            reward_signal = build_reward_signal(
                sample_id=record["dialogue_id"],
                task_id=record["dialogue_id"],
                task_family=task.primary_domain,
                trajectory_text=render_trajectory_text(record.get("history", [])),
                reward_value=float(record.get("success", 0)),
                termination_reason="SUCCESS" if int(record.get("success", 0)) == 1 else "FAILURE",
                meta_sop_text=current_meta_sop_text,
            )

            if "meta_sop" in evolve_components:
                current_meta_sop_text = build_update_context_text(
                    current_meta_sop_text,
                    outcome_tag=reward_signal["outcome_tag"],
                    family=task.primary_domain,
                    posterior_text=reward_signal["posterior_text"],
                )
                current_meta_sop_by_domain[task.primary_domain] = current_meta_sop_text

            if "memory" in evolve_components:
                memory_snapshot = apply_memory_update(
                    memory_snapshot,
                    family=task.primary_domain,
                    posterior_text=reward_signal["posterior_text"],
                    outcome_tag=reward_signal["outcome_tag"],
                    source_cycle=cycle_index,
                    source_task=record["dialogue_id"],
                    reward_value=reward_signal["reward_value"],
                    meta_sop_text=current_meta_sop_text,
                    task_id=record["dialogue_id"],
                )
            if "state_graph" in evolve_components:
                graph_snapshot = apply_state_graph_update(
                    graph_snapshot,
                    family=task.primary_domain,
                    posterior_text=reward_signal["posterior_text"],
                    outcome_tag=reward_signal["outcome_tag"],
                    source_cycle=cycle_index,
                    source_task=record["dialogue_id"],
                    reward_value=reward_signal["reward_value"],
                    meta_sop_text=current_meta_sop_text,
                )

            cycle_events.append(
                record_evolution_event(
                    sample_id=record["dialogue_id"],
                    task_id=record["dialogue_id"],
                    cycle_id=cycle_index,
                    task_family=task.primary_domain,
                    outcome_tag=reward_signal["outcome_tag"],
                    reward_value=reward_signal["reward_value"],
                    posterior_text=reward_signal["posterior_text"],
                    memory_delta=(memory_snapshot.get("entries") or [])[-1]
                    if memory_snapshot.get("entries")
                    else None,
                    graph_delta=(graph_snapshot.get("nodes") or [])[-1]
                    if graph_snapshot.get("nodes")
                    else None,
                )
            )
            cycle_records.append(record)

            _write_json_file(cycle_dir / "results.json", cycle_records)
            _write_jsonl_file(cycle_dir / "events.jsonl", cycle_events)
            _write_json_file(cycle_dir / "memories.json", memory_snapshot)
            _write_json_file(cycle_dir / "state_graph.json", graph_snapshot)
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
        "workers": config.workers,
        "max_turns": config.max_turns,
        "temperature": config.temperature,
        "seed": config.seed,
        "multiwoz_config": str(config.multiwoz_config),
        "multiwoz_data_json": str(config.multiwoz_data_json),
        "multiwoz_db_dir": str(config.multiwoz_db_dir),
    }


def main() -> None:
    config = load_runner_config()
    run_dir = build_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_file(run_dir / "config_snapshot.json", _config_snapshot(config))

    tasks = load_tasks(config)

    if config.evolve_components:
        device_id = _resolve_worker_device_ids(
            config.gpu_ids,
            config.proc_per_node,
            1,
        )[0]
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

    records = run_baseline(config, tasks, run_dir=run_dir)
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
        serial_execution=False,
        total_n=len(tasks),
    )
    summary["run_dir"] = str(run_dir)
    summary["timestamp"] = datetime.now().isoformat()
    _write_json_file(run_dir / "results.json", records)
    _write_json_file(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
