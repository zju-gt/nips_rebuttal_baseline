#!/usr/bin/env python3
"""Local tau2-bench runner backed by transformers.

This runner keeps tau2-bench's prompts, orchestration, and evaluator logic,
but replaces the litellm-backed model calls with a local transformers backend.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import random
import re
import signal
import sys
import time
import warnings
import uuid
from collections import Counter, OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - optional dependency is present in env.yml
    load_dotenv = None
    dotenv_values = None

from evolve_state import (
    infer_task_family,
    record_evolution_event,
)
from llm_backend import (
    apply_backend_llm_args,
    generate_backend_json_object,
    generate_backend_text,
    install_backend_runtime,
    normalize_model_name_for_backend,
)
from memory import apply_memory_update, seed_memory_snapshot, summarize_memory_snapshot
from retrieval import (
    build_openai_compatible_embed_texts,
    render_graph_retrieval_section,
    render_memory_retrieval_section,
    render_workflow_retrieval_section,
    retrieve_graph_updates,
    retrieve_memory_entries,
    retrieve_workflow_entries,
)
from rewards import build_reward_signal, render_trajectory_text
from state_graph import apply_state_graph_update, seed_state_graph, summarize_state_graph
from workflow_memory import (
    apply_workflow_memory_update,
    build_workflow_memory_signal,
    seed_workflow_memory,
    summarize_workflow_memory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "tau2-bench"

DEFAULTS = {
    "LLM_BACKEND": "transformers",
    "LLM_MODEL_NAME": "Qwen3.5-27B",
    "LLM_MODEL_PATH": "/sde/llm_weights/Qwen3.5-27B",
    "LLM_BASE_URL": "",
    "LLM_API_KEY": "",
    "PROC_PER_NODE": 1,
    "DOMAIN": "retail",
    "MODE": "base_lm",
    "SOP_TEXT": "static",
    "NUM_CYCLES": 1,
    "TAU2_REPO": "/home/duyuetian/projects/tau2-bench",
    "PASS_HAT_K": 1,
    "MAX_STEPS": 200,
    "MAX_ERRORS": 5,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 1.0,
    "SAMPLE_TIMEOUT_SECONDS": 800.0,
    "REPAIR_MAX_NEW_TOKENS": 256,
    "RESULT_BATCH_SIZE": 2,
    "RESULT_SAVE_EVERY": 2,
    "SEED": 300,
    "NL_JUDGE_MODEL": "gpt-5.1-chat-latest",
    "NL_JUDGE_BASE_URL": "https://routify.alibaba-inc.com/protocol/openai/v1",
    "EVOLVE_COMPONENTS": "",
    "EVOLVE_PARALLEL_MODE": "strict",
    "EVOLVE_BATCH_SIZE": 1,
    "EVOLVE_SYNC_ORDER": "task_index",
    "TAU2_LOG_LEVEL": "CRITICAL",
    "EMBEDDING_MODEL": "text-embedding-v4",
    "EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "RETRIEVAL_TOP_K": 3,
}


@dataclass(frozen=True)
class RunnerConfig:
    llm_backend: str
    llm_model_name: str
    llm_model_path: str
    llm_base_url: str
    llm_api_key: str
    gpu_ids: list[str]
    proc_per_node: int
    domain: str
    mode: str
    sop_text: str
    num_cycles: int
    tau2_repo: str
    pass_hat_k: int
    seed: int
    max_steps: int
    max_errors: int
    max_retries: int
    retry_delay: float
    max_new_tokens: int
    repair_max_new_tokens: int
    result_batch_size: int
    result_save_every: int
    tau2_log_level: str
    sample_timeout_seconds: float = DEFAULTS["SAMPLE_TIMEOUT_SECONDS"]
    evolve_components: list[str] = field(default_factory=list)
    evolve_parallel_mode: str = DEFAULTS["EVOLVE_PARALLEL_MODE"]
    evolve_batch_size: int = DEFAULTS["EVOLVE_BATCH_SIZE"]
    evolve_sync_order: str = DEFAULTS["EVOLVE_SYNC_ORDER"]
    results_root: Path = RESULTS_ROOT
    embedding_model: str = DEFAULTS["EMBEDDING_MODEL"]
    embedding_base_url: str = DEFAULTS["EMBEDDING_BASE_URL"]
    embedding_api_key: str = ""
    retrieval_top_k: int = DEFAULTS["RETRIEVAL_TOP_K"]


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    cuda_visible_devices: str
    llm_backend: str
    llm_model_name: str
    llm_model_path: str
    llm_base_url: str
    llm_api_key: str
    tau2_repo: str
    run_dir: str
    domain: str
    mode: str
    sop_text: str
    num_cycles: int
    system_prompt_prefix: str
    pass_hat_k: int
    seed: int
    max_steps: int
    max_errors: int
    max_retries: int
    retry_delay: float
    sample_timeout_seconds: float
    max_new_tokens: int
    repair_max_new_tokens: int
    result_batch_size: int
    result_save_every: int
    tau2_log_level: str
    evolve_components: list[str] = field(default_factory=list)
    meta_sop_text: str = ""


@dataclass(frozen=True)
class Job:
    job_id: str
    task_data: dict[str, Any]
    trial: int
    seed: int
    system_prompt_prefix: str = ""


@dataclass(frozen=True)
class EvolveSamplePlan:
    job: Job
    task_index: int
    sequence_index: int


def ensure_tau2_repo_on_path(tau2_repo: str) -> None:
    src_dir = Path(tau2_repo) / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


@contextmanager
def _sample_timeout(timeout_seconds: float):
    if timeout_seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise TimeoutError(f"Sample exceeded total timeout of {timeout_seconds:.0f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def get_git_commit(repo: str) -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
            .split("\n")[0]
        )
    except Exception:
        return "unknown"


def parse_gpu_list(raw: str) -> list[str]:
    if not raw.strip():
        return [""]
    parts = [item.strip() for item in raw.split(",")]
    return [item for item in parts if item]


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


def configure_runtime_logging(level: str) -> None:
    """Silence tau2-bench logs while keeping tqdm and fatal errors visible."""
    try:
        from loguru import logger
    except ModuleNotFoundError:
        logger = None

    normalized = level.upper().strip()
    if normalized not in {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        normalized = "CRITICAL"

    if logger is not None:
        logger.remove()
        logger.add(sys.stderr, level=normalized, colorize=True)

    logging.basicConfig(level=getattr(logging, normalized, logging.CRITICAL), force=True)
    for name in [
        "LiteLLM",
        "httpx",
        "urllib3",
        "transformers",
        "tokenizers",
        "huggingface_hub",
    ]:
        logging.getLogger(name).setLevel(getattr(logging, normalized, logging.CRITICAL))

    warnings.filterwarnings("ignore")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value


def load_local_env_files() -> None:
    if dotenv_values is None and load_dotenv is None:
        return
    for candidate in (PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env"):
        if candidate.exists():
            if dotenv_values is not None:
                _merge_env_values(dotenv_values(candidate))
            elif load_dotenv is not None:
                load_dotenv(candidate, override=False)


def _merge_env_values(values: dict[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        current = os.getenv(key)
        if current is None or not current.strip():
            os.environ[key] = value


def validate_judge_env() -> None:
    judge_model = parse_str_env("NL_JUDGE_MODEL", DEFAULTS["NL_JUDGE_MODEL"])
    judge_base_url = parse_str_env("NL_JUDGE_BASE_URL", DEFAULTS["NL_JUDGE_BASE_URL"])
    judge_api_key = parse_str_env(
        "NL_JUDGE_API_KEY",
        parse_str_env("OPENROUTER_API_KEY", ""),
    )

    needs_api_key = bool(judge_base_url.strip()) or judge_model.startswith("openrouter/")
    if needs_api_key and not judge_api_key.strip():
        raise ValueError(
            "NL_JUDGE_API_KEY is required when the judge is configured to use "
            f"a remote API endpoint (model={judge_model!r}, base_url={judge_base_url!r})."
        )


def configure_tau2_judge_defaults(tau2_config: Any) -> None:
    judge_model = parse_str_env("NL_JUDGE_MODEL", DEFAULTS["NL_JUDGE_MODEL"])
    judge_api_key = parse_str_env(
        "NL_JUDGE_API_KEY",
        parse_str_env("OPENROUTER_API_KEY", ""),
    )
    judge_base_url = parse_str_env("NL_JUDGE_BASE_URL", DEFAULTS["NL_JUDGE_BASE_URL"])

    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = judge_model
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}

    if judge_api_key.strip():
        tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_key"] = judge_api_key.strip()
    if judge_base_url.strip():
        tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS["base_url"] = judge_base_url.strip()

    imported_nl_evaluator = sys.modules.get("tau2.evaluator.evaluator_nl_assertions")
    if imported_nl_evaluator is not None:
        setattr(imported_nl_evaluator, "DEFAULT_LLM_NL_ASSERTIONS", judge_model)
        setattr(
            imported_nl_evaluator,
            "DEFAULT_LLM_NL_ASSERTIONS_ARGS",
            dict(tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS),
        )

    if judge_api_key.strip():
        os.environ.setdefault("NL_JUDGE_API_KEY", judge_api_key.strip())


def build_run_dir(config: RunnerConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.results_root / config.domain / f"{config.mode}_{ts}"


def load_domain_policy_text(
    domain: str,
    tau2_repo: str,
    *,
    strict: bool = True,
) -> str:
    ensure_tau2_repo_on_path(tau2_repo)

    try:
        from tau2.registry import registry

        env_constructor = registry.get_env_constructor(domain)
        try:
            environment = env_constructor()
        except TypeError:
            environment = env_constructor(solo_mode=False)
        policy_text = environment.get_policy()
        return str(policy_text).strip()
    except Exception:
        if strict:
            raise RuntimeError(f"Failed to load domain policy for {domain}") from None
        fallback_paths = [
            PROJECT_ROOT / "prompts" / "prior" / f"domain_prior_{domain}.md",
            PROJECT_ROOT / "prompts" / "prior" / "benchmark_prior_tau2.md",
        ]
        for path in fallback_paths:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
    return ""


def build_context_bundle_for_run(
    runner_cfg: RunnerConfig,
    run_dir: Path,
    tasks: list[Any],
) -> "ResolvedContextBundle":
    from context_builder import load_context_bundle
    from execution import (
        build_generation_callback,
        prepare_context_artifacts,
        select_representative_tasks,
    )

    task_dicts = [task.model_dump(mode="json") for task in tasks]
    representative_tasks = select_representative_tasks(runner_cfg.domain, task_dicts)
    if runner_cfg.mode == "base_lm":
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return load_context_bundle(
            mode=runner_cfg.mode,
            sop_text=runner_cfg.sop_text,
            domain=runner_cfg.domain,
            benchmark_prior_text="",
            domain_prior_text="",
            meta_sop_text="",
            artifact_dir=artifacts_dir,
        )

    generator = None
    domain_policy_text = ""
    if runner_cfg.sop_text == "dynamic":
        domain_policy_text = load_domain_policy_text(
            runner_cfg.domain,
            runner_cfg.tau2_repo,
            strict=True,
        )
        generator = build_generation_callback(
            domain=runner_cfg.domain,
            domain_policy_text=domain_policy_text,
            representative_tasks=representative_tasks,
        )
    return prepare_context_artifacts(
        mode=runner_cfg.mode,
        sop_text=runner_cfg.sop_text,
        domain=runner_cfg.domain,
        run_dir=run_dir,
        domain_policy_text=domain_policy_text,
        representative_tasks=representative_tasks,
        generator=generator,
    )


def get_tau2_modules() -> dict[str, Any]:
    ensure_tau2_repo_on_path(parse_str_env("TAU2_REPO", DEFAULTS["TAU2_REPO"]))

    import tau2.config as tau2_config

    configure_tau2_judge_defaults(tau2_config)

    from tau2.data_model.simulation import Results, SimulationRun, TextRunConfig
    from tau2.data_model.simulation import TerminationReason
    from tau2.data_model.tasks import Task
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.runner.helpers import get_info, get_tasks
    from tau2.runner.progress import run_with_retry
    from tau2.run import run_single_task
    from tau2.utils import utils as tau2_utils
    import tau2.utils.llm_utils as llm_utils

    return {
        "config": tau2_config,
        "Results": Results,
        "SimulationRun": SimulationRun,
        "TextRunConfig": TextRunConfig,
        "Task": Task,
        "EvaluationType": EvaluationType,
        "compute_metrics": compute_metrics,
        "get_tasks": get_tasks,
        "get_info": get_info,
        "run_with_retry": run_with_retry,
        "run_single_task": run_single_task,
        "tau2_utils": tau2_utils,
        "llm_utils": llm_utils,
        "TerminationReason": TerminationReason,
    }


def _extract_json_text(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()

    for start_char in ("{", "["):
        start = stripped.find(start_char)
        if start != -1:
            candidate = stripped[start:]
            decoder = json.JSONDecoder()
            try:
                obj, _ = decoder.raw_decode(candidate)
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                continue
    return None


def _safe_json_loads(text: str) -> Any:
    json_text = _extract_json_text(text)
    if json_text is None:
        raise ValueError("No JSON object found in model output")
    return json.loads(json_text)


def _build_nl_assertions_system_prompt() -> str:
    return """
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each expected outcome.
- Grade each expected outcome individually.

FORMAT
- Return exactly one valid JSON object.
- Do not wrap the response in markdown fences.
- Do not add any commentary outside the JSON object.
- The response must contain a results array with one entry per expected outcome.

Example response:
{
  "results": [
    {
      "expectedOutcome": "The agent greets the customer",
      "reasoning": "The agent says hello at the beginning of the conversation.",
      "metExpectation": true
    }
  ]
}
""".strip()


def _patch_domain_policy_prefix(
    registry: Any,
    domain: str,
    extra_policy_provider: Any,
    debug_label_provider: Any | None = None,
) -> None:
    original_get_env_constructor = registry.get_env_constructor

    def _patched_get_env_constructor(name: str) -> Any:
        env_constructor = original_get_env_constructor(name)
        if name != domain:
            return env_constructor

        def _wrapped_env_constructor(*args: Any, **kwargs: Any) -> Any:
            environment = env_constructor(*args, **kwargs)
            original_get_policy = environment.get_policy
            merged_policy_logged = False

            def _patched_get_policy() -> str:
                nonlocal merged_policy_logged
                extra_policy = str(extra_policy_provider() or "").strip()
                base_policy = str(original_get_policy()).strip()
                merged_policy = "\n\n".join(
                    [part for part in [extra_policy, base_policy] if part]
                )
                if _agent_prompt_debug_enabled() and not merged_policy_logged:
                    debug_label = ""
                    if debug_label_provider is not None:
                        debug_label = str(debug_label_provider() or "").strip()
                    _emit_agent_prompt_debug_block(
                        "MERGED_POLICY",
                        sample_id=debug_label or "unknown",
                        content=merged_policy,
                    )
                    merged_policy_logged = True
                return merged_policy

            environment.get_policy = _patched_get_policy
            return environment

        return _wrapped_env_constructor

    registry.get_env_constructor = _patched_get_env_constructor


def _simulation_messages(simulation: Any) -> list[Any]:
    if hasattr(simulation, "get_messages"):
        try:
            return list(simulation.get_messages())
        except Exception:
            pass
    messages = getattr(simulation, "messages", None)
    return list(messages or [])


def _agent_prompt_debug_enabled() -> bool:
    value = os.getenv("AGENT_PROMPT_DEBUG_IO", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    value = os.getenv("TAU2_AGENT_PROMPT_DEBUG_IO", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _emit_agent_prompt_debug_block(
    label: str,
    *,
    sample_id: str,
    content: str,
) -> None:
    print(
        "\n".join(
            [
                f"[agent-prompt-debug] {label}_BEGIN sample_id={sample_id}",
                str(content or ""),
                f"[agent-prompt-debug] {label}_END sample_id={sample_id}",
            ]
        ),
        flush=True,
    )


def _simulation_reward_value(simulation: Any) -> Optional[float]:
    reward_info = getattr(simulation, "reward_info", None)
    if reward_info is None:
        return None
    if isinstance(reward_info, dict):
        reward = reward_info.get("reward")
    else:
        reward = getattr(reward_info, "reward", None)
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _simulation_termination_reason(simulation: Any) -> str:
    termination = getattr(simulation, "termination_reason", None)
    if termination is None:
        return ""
    if hasattr(termination, "name"):
        return str(getattr(termination, "name"))
    return str(termination)


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


def _simulation_warnings(simulation: Any) -> list[str]:
    candidates: list[str] = []
    _collect_warning_candidates(getattr(simulation, "info", None), candidates)
    for message in _simulation_messages(simulation):
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content", "") or "")
        else:
            content = str(getattr(message, "content", "") or "")
        _collect_warning_candidates(content, candidates)
    deduped: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        key = text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text.strip())
    return deduped[:12]


def _task_query_from_task(task: Any) -> str:
    if task is None:
        return ""
    if hasattr(task, "model_dump"):
        try:
            task_data = task.model_dump(mode="json")
        except Exception:
            task_data = {}
    elif isinstance(task, dict):
        task_data = task
    else:
        task_data = {}
    if not isinstance(task_data, dict):
        return ""
    candidate_keys = (
        "query",
        "user_query",
        "instruction",
        "goal",
        "question",
        "description",
        "task",
    )
    for key in candidate_keys:
        value = task_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _task_query_from_simulation(simulation: Any) -> str:
    for message in _simulation_messages(simulation):
        role = ""
        content = ""
        if isinstance(message, dict):
            role = str(message.get("role", "") or "")
            content = str(message.get("content", "") or "")
        else:
            role = str(getattr(message, "role", "") or "")
            content = str(getattr(message, "content", "") or "")
        if role == "user" and content.strip():
            return content.strip()
    return ""


def _derive_task_query(task: Any, simulation: Any) -> str:
    query = _task_query_from_simulation(simulation)
    if query.strip():
        return query.strip()
    return _task_query_from_task(task)


def _build_retrieval_query(task_query: str, task_family: str) -> str:
    normalized_query = str(task_query or "").strip()
    normalized_family = str(task_family or "").strip()
    sections: list[str] = []
    if normalized_family:
        sections.append(f"Family: {normalized_family}")
    if normalized_query:
        sections.append(f"Task: {normalized_query}")
    return "\n".join(sections).strip()


def _build_runner_embed_texts(runner_cfg: RunnerConfig) -> Any:
    return build_openai_compatible_embed_texts(
        model=str(getattr(runner_cfg, "embedding_model", DEFAULTS["EMBEDDING_MODEL"])),
        api_key=str(getattr(runner_cfg, "embedding_api_key", "") or ""),
        base_url=str(getattr(runner_cfg, "embedding_base_url", DEFAULTS["EMBEDDING_BASE_URL"])),
    )


def _extractor_generate_for_config(config: Any, *, device_id: str = "") -> Any:
    def _generate(**kwargs: Any) -> dict[str, Any]:
        return generate_backend_json_object(
            config,
            messages=kwargs["messages"],
            call_name=str(kwargs.get("call_name", "experience_extract")),
            max_new_tokens=int(kwargs.get("max_new_tokens", 640)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=float(kwargs.get("top_p", 1.0)),
            top_k=int(kwargs.get("top_k", 0)),
            seed=kwargs.get("seed"),
            device_id=device_id,
        )

    return _generate


def _simulation_precomputed_reward_signal(simulation: Any) -> dict[str, Any] | None:
    info = getattr(simulation, "info", None)
    if not isinstance(info, dict):
        return None
    value = info.pop("_runner_reward_signal", None)
    if isinstance(value, dict):
        return value
    return None


def _simulation_precomputed_workflow_signal(simulation: Any) -> dict[str, Any] | None:
    info = getattr(simulation, "info", None)
    if not isinstance(info, dict):
        return None
    value = info.pop("_runner_workflow_signal", None)
    if isinstance(value, dict):
        return value
    return None


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


def _workflow_delta(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_counts = {
        str(entry.get("id", "")): int(entry.get("success_count", 0))
        for entry in (before.get("entries") or [])
        if entry.get("id")
    }
    changed: list[dict[str, Any]] = []
    for entry in after.get("entries") or []:
        entry_id = str(entry.get("id", ""))
        if not entry_id:
            continue
        success_count = int(entry.get("success_count", 0))
        if entry_id not in before_counts or before_counts[entry_id] != success_count:
            changed.append(entry)
    return changed


def _safe_task_family(domain: str, task: Any) -> str:
    try:
        task_data = task.model_dump(mode="json")
    except Exception:
        task_data = {}
    return infer_task_family(domain, task_data)


def _build_evolution_context_text(
    *,
    base_prompt: str,
    memory_snapshot: dict[str, Any],
    state_graph_snapshot: dict[str, Any],
    workflow_memory_snapshot: dict[str, Any] | None = None,
    evolve_components: list[str],
    task_query: str = "",
    task_family: str = "",
    retrieval_top_k: int = 0,
    embed_texts: Any = None,
) -> str:
    parts: list[str] = []
    if base_prompt.strip():
        parts.append(base_prompt.strip())
    retrieval_query = _build_retrieval_query(task_query, task_family)
    use_retrieval = bool(embed_texts and retrieval_top_k > 0 and retrieval_query.strip())
    if "memory" in evolve_components:
        if use_retrieval:
            retrieved_memory = retrieve_memory_entries(
                memory_snapshot,
                query_text=retrieval_query,
                family=task_family,
                top_k=retrieval_top_k,
                embed_texts=embed_texts,
            )
            memory_text = render_memory_retrieval_section(retrieved_memory)
            if memory_text.strip():
                parts.append(memory_text)
            else:
                parts.append(summarize_memory_snapshot(memory_snapshot))
        else:
            parts.append(summarize_memory_snapshot(memory_snapshot))
    if "state_graph" in evolve_components:
        if use_retrieval:
            retrieved_graph = retrieve_graph_updates(
                state_graph_snapshot,
                query_text=retrieval_query,
                family=task_family,
                top_k=retrieval_top_k,
                embed_texts=embed_texts,
            )
            graph_text = render_graph_retrieval_section(retrieved_graph)
            if graph_text.strip():
                parts.append(graph_text)
            else:
                parts.append(summarize_state_graph(state_graph_snapshot))
        else:
            parts.append(summarize_state_graph(state_graph_snapshot))
    if "workflow_memory" in evolve_components:
        workflow_snapshot = workflow_memory_snapshot or {}
        if use_retrieval:
            retrieved_workflows = retrieve_workflow_entries(
                workflow_snapshot,
                query_text=retrieval_query,
                family=task_family,
                top_k=retrieval_top_k,
                embed_texts=embed_texts,
            )
            workflow_text = render_workflow_retrieval_section(retrieved_workflows)
            if workflow_text.strip():
                parts.append(workflow_text)
            else:
                workflow_summary = summarize_workflow_memory(workflow_snapshot)
                if "(empty)" not in workflow_summary:
                    parts.append(workflow_summary)
        else:
            workflow_summary = summarize_workflow_memory(workflow_snapshot)
            if "(empty)" not in workflow_summary:
                parts.append(workflow_summary)
    return "\n\n".join(part for part in parts if str(part).strip())


def _write_json_file(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _format_tool_definitions(tools: Optional[list[Any]]) -> str:
    if not tools:
        return ""
    parts = ["<available_tools>"]
    for idx, tool in enumerate(tools, start=1):
        try:
            schema = tool.openai_schema
        except Exception:
            schema = {"name": getattr(tool, "name", f"tool_{idx}")}
        parts.append(json.dumps(schema, ensure_ascii=False, indent=2))
    parts.append("</available_tools>")
    return "\n".join(parts)


def _message_to_text(message: Any) -> str:
    AssistantMessage = None
    ToolMessage = None
    UserMessage = None
    try:
        from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    except Exception:
        pass

    if UserMessage is not None and isinstance(message, UserMessage):
        content = message.content or ""
        if message.tool_calls:
            tool_data = {
                "tool_calls": [tc.model_dump(mode="json") for tc in message.tool_calls]
            }
            return f"role=user\ncontent={content}\n{json.dumps(tool_data, ensure_ascii=False)}"
        return f"role=user\ncontent={content}"

    if AssistantMessage is not None and isinstance(message, AssistantMessage):
        content = message.content or ""
        if message.tool_calls:
            tool_data = {
                "tool_calls": [tc.model_dump(mode="json") for tc in message.tool_calls]
            }
            return f"role=assistant\ncontent={content}\n{json.dumps(tool_data, ensure_ascii=False)}"
        return f"role=assistant\ncontent={content}"

    if ToolMessage is not None and isinstance(message, ToolMessage):
        return (
            f"role=tool\nrequestor={message.requestor}\n"
            f"tool_call_id={message.id}\ncontent={message.content or ''}"
        )

    if isinstance(message, dict):
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", "") or "")
        return f"role={role}\ncontent={content}"

    return f"role={getattr(message, 'role', 'unknown')}\ncontent={getattr(message, 'content', '') or ''}"


def _build_prompt(
    messages: list[Any],
    tools: Optional[list[Any]],
    call_name: Optional[str],
    tool_choice: Optional[str],
) -> tuple[list[dict[str, str]], str]:
    SystemMessage = None
    try:
        from tau2.data_model.message import SystemMessage
    except Exception:
        pass

    def _is_system_message(message: Any) -> bool:
        if isinstance(message, dict):
            return str(message.get("role", "")).strip() == "system"
        if SystemMessage is None:
            return False
        return isinstance(message, SystemMessage)

    system_messages = [m for m in messages if _is_system_message(m)]
    non_system_messages = [m for m in messages if not _is_system_message(m)]
    system_text = "\n\n".join(
        [
            (
                m.get("content", "")
                if isinstance(m, dict)
                else getattr(m, "content", "")
            )
            or ""
            for m in system_messages
            if (
                (
                    m.get("content", "")
                    if isinstance(m, dict)
                    else getattr(m, "content", "")
                )
                or ""
            ).strip()
        ]
    ).strip()

    helper_lines = [
        "<runner_instructions>",
        "Return exactly one JSON object.",
    ]
    if tools:
        helper_lines.append(
            "Use {\"content\": \"...\"} for plain text or "
            "{\"tool_calls\": [{\"name\": \"...\", \"arguments\": {...}}]} for tool use."
        )
    else:
        helper_lines.append('Use {"content": "..."} only.')
    if tool_choice == "required":
        helper_lines.append("A tool call is required.")
    helper_lines.append("</runner_instructions>")
    helper_lines.append(_format_tool_definitions(tools))
    helper_text = "\n".join([line for line in helper_lines if line])

    full_system = "\n\n".join([part for part in [system_text, helper_text] if part])
    chat_messages: list[dict[str, str]] = []
    if full_system:
        chat_messages.append({"role": "system", "content": full_system})
    for message in non_system_messages:
        if isinstance(message, dict):
            role = str(message.get("role", "user"))
            if role == "tool":
                tool_call_id = str(message.get("tool_call_id", ""))
                content = str(message.get("content", "") or "")
                chat_messages.append(
                    {
                        "role": "tool",
                        "content": f"tool_call_id={tool_call_id}\n{content}",
                    }
                )
            else:
                content = str(message.get("content", "") or "")
                chat_messages.append({"role": role, "content": content})
            continue
        role = getattr(message, "role", "user")
        if role == "tool":
            tool_call_id = getattr(message, "id", "")
            content = getattr(message, "content", "") or ""
            chat_messages.append(
                {
                    "role": "tool",
                    "content": f"tool_call_id={tool_call_id}\n{content}",
                }
            )
        else:
            chat_messages.append({"role": role, "content": _message_to_text(message)})
    return chat_messages, full_system


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


class LocalTransformersBackend:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device_id: str,
        repair_max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device_id = device_id
        self.repair_max_new_tokens = repair_max_new_tokens
        self.tokenizer = None
        self.model = None

    @classmethod
    def load(
        cls,
        model_name: str,
        model_path: str,
        device_id: str,
        repair_max_new_tokens: int = 256,
    ) -> "LocalTransformersBackend":
        backend = cls(
            model_name=model_name,
            model_path=model_path,
            device_id=device_id,
            repair_max_new_tokens=repair_max_new_tokens,
        )
        backend._load_model()
        return backend

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        debug_timing = _env_flag_enabled("LLM_DEBUG_TIMING") or _env_flag_enabled(
            "MULTIWOZ_DEBUG_TIMING"
        )
        load_start = time.perf_counter()
        if debug_timing:
            print(
                f"[llm-timing] stage=load_model_start model={self.model_name} path={self.model_path}",
                flush=True,
            )

        tokenizer_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if debug_timing:
            print(
                "[llm-timing] "
                f"stage=load_tokenizer_done model={self.model_name} "
                f"seconds={time.perf_counter() - tokenizer_start:.3f}",
                flush=True,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        if torch.cuda.is_available():
            # Prefer direct GPU placement during load to avoid a large post-load CPU->GPU transfer.
            model_kwargs["device_map"] = "cuda:0"

        def _from_pretrained_with_dtype_compat(kwargs: dict[str, Any]) -> Any:
            try:
                return AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
            except TypeError:
                legacy_kwargs = dict(kwargs)
                if "dtype" in legacy_kwargs and "torch_dtype" not in legacy_kwargs:
                    legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
                return AutoModelForCausalLM.from_pretrained(self.model_path, **legacy_kwargs)

        model_load_start = time.perf_counter()
        used_direct_device_map = False
        try:
            model = _from_pretrained_with_dtype_compat(model_kwargs)
            used_direct_device_map = bool(model_kwargs.get("device_map"))
        except Exception as load_exc:
            if model_kwargs.get("device_map"):
                if debug_timing:
                    print(
                        "[llm-timing] "
                        f"stage=load_model_device_map_fallback model={self.model_name} "
                        f"error_type={type(load_exc).__name__}",
                        flush=True,
                    )
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("device_map", None)
                model = _from_pretrained_with_dtype_compat(fallback_kwargs)
                used_direct_device_map = False
            else:
                raise
        if debug_timing:
            print(
                "[llm-timing] "
                f"stage=load_model_weights_done model={self.model_name} "
                f"seconds={time.perf_counter() - model_load_start:.3f}",
                flush=True,
            )

        if torch.cuda.is_available() and not used_direct_device_map:
            to_cuda_start = time.perf_counter()
            model = model.to("cuda:0")
            if debug_timing:
                print(
                    "[llm-timing] "
                    f"stage=model_to_cuda_done model={self.model_name} "
                    f"seconds={time.perf_counter() - to_cuda_start:.3f}",
                    flush=True,
                )

        self._normalize_generation_config(model)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        if debug_timing:
            print(
                "[llm-timing] "
                f"stage=load_model_done model={self.model_name} "
                f"total_seconds={time.perf_counter() - load_start:.3f}",
                flush=True,
            )

    @staticmethod
    def _normalize_generation_config(model: Any) -> None:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            return
        # Greedy decoding should not carry stale sampling flags from model defaults.
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, attr):
                setattr(generation_config, attr, None)

    def _render_chat_text(
        self,
        chat_messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        assert self.tokenizer is not None
        try:
            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            rendered = []
            for msg in chat_messages:
                rendered.append(f"{msg['role'].upper()}:\n{msg['content']}\n")
            if add_generation_prompt:
                rendered.append("ASSISTANT:\n")
            return "\n".join(rendered)

    def _build_input_tensors(
        self,
        chat_messages: list[dict[str, str]],
        *,
        family: str | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        assert self.tokenizer is not None
        assert self.model is not None
        text = self._render_chat_text(chat_messages, add_generation_prompt=True)
        full_inputs = self.tokenizer(text, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            full_inputs = {k: v.to("cuda:0") for k, v in full_inputs.items()}
        return text, full_inputs

    def _forward_with_cache(
        self,
        input_ids: Any,
        *,
        past_key_values: Any = None,
        total_prompt_len: Optional[int] = None,
    ) -> tuple[Any, Any]:
        assert self.model is not None
        import torch

        if total_prompt_len is None:
            total_prompt_len = int(input_ids.shape[-1])
        attention_mask = torch.ones(
            (1, total_prompt_len),
            device=input_ids.device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            logits = outputs[0]
        next_past = getattr(outputs, "past_key_values", None)
        if next_past is None and len(outputs) > 1:
            next_past = outputs[1]
        return logits, next_past

    def _call_family(self, call_name: Optional[str]) -> str:
        if not call_name:
            return "default"
        if call_name.endswith("_repair"):
            return "repair"
        if call_name.startswith("user_") or call_name in {
            "interruption_decision",
            "backchannel_decision",
        }:
            return "user"
        if call_name.startswith("agent_") or call_name == "interface_agent_response":
            return "agent"
        if call_name.startswith("llm_judge_") or call_name in {
            "nl_assertions_eval",
            "classify_authentication",
        }:
            return "judge"
        return call_name

    def _sample_next_token(
        self,
        logits: Any,
        llm_args: dict[str, Any],
    ) -> int:
        import torch

        temperature = llm_args.get("temperature", 0.0)
        top_p = llm_args.get("top_p", 1.0)
        top_k = llm_args.get("top_k", 0)
        next_token_logits = logits[:, -1, :].squeeze(0)
        if temperature is None or float(temperature) <= 0.0:
            return int(torch.argmax(next_token_logits, dim=-1).item())

        scaled = next_token_logits / float(temperature)
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), scaled.shape[-1])
            topk_values, topk_indices = torch.topk(scaled, k)
            filtered = torch.full_like(scaled, float("-inf"))
            filtered.scatter_(0, topk_indices, topk_values)
            scaled = filtered
        probs = torch.softmax(scaled, dim=-1)
        if top_p is not None and float(top_p) < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > float(top_p)
            if torch.any(cutoff):
                cutoff[0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                probs = torch.zeros_like(probs)
                probs.scatter_(0, sorted_indices, sorted_probs)
                total = probs.sum()
                if total.item() > 0:
                    probs = probs / total
        return int(torch.multinomial(probs, num_samples=1).item())

    def _decode_generated(self, input_len: int, output_ids: Any) -> str:
        assert self.tokenizer is not None
        generated = output_ids[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_text(
        self,
        prompt_text: str,
        inputs: Any,
        llm_args: dict[str, Any],
    ) -> str:
        assert self.model is not None
        import torch

        seed = llm_args.get("seed")
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        prompt_ids = tuple(int(x) for x in inputs["input_ids"][0].tolist())
        logits, past_key_values = self._forward_with_cache(
            inputs["input_ids"],
            total_prompt_len=len(prompt_ids),
        )

        generated_ids: list[int] = []
        max_new_tokens = int(llm_args.get("max_new_tokens", 2048))
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        current_logits = logits

        for _ in range(max_new_tokens):
            next_token_id = self._sample_next_token(current_logits, llm_args)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            generated_ids.append(next_token_id)
            next_tensor = torch.tensor(
                [[next_token_id]],
                device=inputs["input_ids"].device,
                dtype=inputs["input_ids"].dtype,
            )
            current_logits, past_key_values = self._forward_with_cache(
                next_tensor,
                past_key_values=past_key_values,
                total_prompt_len=len(prompt_ids) + len(generated_ids),
            )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _parse_message_payload(
        self,
        text: str,
        tools: Optional[list[Any]],
        tool_choice: Optional[str],
    ) -> tuple[Optional[str], Optional[list[Any]]]:
        stripped = text.strip()
        if not stripped:
            return "", None

        payload: Any
        try:
            payload = _safe_json_loads(stripped)
        except Exception:
            return stripped, None

        content: Optional[str] = None
        tool_calls: Optional[list[Any]] = None
        tool_call_cls: Any = None

        def _tool_call_class() -> Any:
            nonlocal tool_call_cls
            if tool_call_cls is None:
                try:
                    from tau2.data_model.message import ToolCall as ImportedToolCall

                    tool_call_cls = ImportedToolCall
                except Exception:
                    tool_call_cls = False
            return tool_call_cls

        def _build_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
            call_cls = _tool_call_class()
            if call_cls is False:
                return {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            return call_cls(id=call_id, name=name, arguments=arguments)

        if isinstance(payload, dict):
            if "tool_calls" in payload and payload["tool_calls"]:
                tool_calls = []
                for idx, item in enumerate(payload["tool_calls"]):
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not name and isinstance(item.get("function"), dict):
                        name = item["function"].get("name")
                    arguments = item.get("arguments")
                    if arguments is None and isinstance(item.get("function"), dict):
                        arguments = item["function"].get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except Exception:
                            arguments = {"value": arguments}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    tool_calls.append(
                        _build_tool_call(
                            call_id=item.get("id", f"call_{idx}"),
                            name=name or "",
                            arguments=arguments,
                        )
                    )
            if "content" in payload and payload["content"] is not None:
                content = str(payload["content"])
            elif "text" in payload and payload["text"] is not None:
                content = str(payload["text"])
            elif "message" in payload and payload["message"] is not None:
                content = str(payload["message"])
            elif "action" in payload and payload["action"] is not None:
                tool_calls = tool_calls or []
                args = payload.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"value": args}
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(
                    _build_tool_call(
                        call_id=payload.get("id", "call_0"),
                        name=str(payload["action"]),
                        arguments=args,
                    )
                )
        elif isinstance(payload, list):
            tool_calls = []
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {"value": arguments}
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_calls.append(
                    _build_tool_call(
                        call_id=item.get("id", f"call_{idx}"),
                        name=name,
                        arguments=arguments,
                    )
                )

        if tool_calls:
            content = None
        return content, tool_calls

    def _repair_output(
        self,
        original_text: str,
        chat_messages: list[dict[str, str]],
        tools: Optional[list[Any]],
        tool_choice: Optional[str],
        llm_args: dict[str, Any],
    ) -> str:
        repair_prompt = [
            {
                "role": "system",
                "content": (
                    "Convert the following model output into a single valid JSON object.\n"
                    "Do not add commentary.\n"
                    "If tools are available, preserve tool calls in the output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original output:\n{original_text}\n\n"
                    "Return only the corrected JSON object."
                ),
            },
        ]
        repair_text, repair_inputs = self._build_input_tensors(
            repair_prompt,
            family="repair",
        )
        repair_llm_args = deepcopy(llm_args)
        repair_llm_args["temperature"] = 0.0
        repair_llm_args["top_p"] = 1.0
        repair_llm_args["top_k"] = 0
        repair_llm_args["max_new_tokens"] = self.repair_max_new_tokens
        repaired = self._generate_text(repair_text, repair_inputs, repair_llm_args)
        return repaired

    def _repair_json_object(
        self,
        original_text: str,
        messages: list[dict[str, str]],
        llm_args: dict[str, Any],
    ) -> str:
        repair_prompt = [
            {
                "role": "system",
                "content": (
                "Convert the following model output into a single valid JSON object.\n"
                    "The first non-whitespace character of your response must be '{' and the last non-whitespace character must be '}'.\n"
                    "Do not add commentary, markdown, code fences, or <think> tags.\n"
                    "Preserve the original meaning and schema intent.\n"
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original output:\n{original_text}\n\n"
                    "Return only the corrected JSON object.\n"
                    "Start immediately with '{' and end with '}'."
                ),
            },
        ]
        repair_text, repair_inputs = self._build_input_tensors(
            repair_prompt,
            family="repair",
        )
        repair_llm_args = deepcopy(llm_args)
        repair_llm_args["temperature"] = 0.0
        repair_llm_args["top_p"] = 1.0
        repair_llm_args["top_k"] = 0
        repair_llm_args["max_new_tokens"] = self.repair_max_new_tokens
        return self._generate_text(repair_text, repair_inputs, repair_llm_args)

    @staticmethod
    def _fallback_content() -> str:
        return "Please continue."

    def _coerce_message_payload(
        self,
        content: Optional[str],
        tool_calls: Optional[list[Any]],
        generated_text: str,
    ) -> tuple[Optional[str], Optional[list[Any]]]:
        normalized_text = generated_text.strip()
        normalized_content = content
        normalized_tool_calls = tool_calls

        if normalized_tool_calls is not None and not normalized_tool_calls:
            normalized_tool_calls = None

        if normalized_tool_calls is None and (
            normalized_content is None or not normalized_content.strip()
        ):
            normalized_content = normalized_text or self._fallback_content()

        return normalized_content, normalized_tool_calls

    def generate_raw_text(
        self,
        *,
        messages: list[dict[str, str]],
        call_name: Optional[str] = None,
        **llm_args: Any,
    ) -> str:
        family = self._call_family(call_name)
        _, inputs = self._build_input_tensors(messages, family=family)
        return self._generate_text("", inputs, llm_args)

    def generate_json_object(
        self,
        *,
        messages: list[dict[str, str]],
        call_name: Optional[str] = None,
        **llm_args: Any,
    ) -> dict[str, Any]:
        family = self._call_family(call_name)
        _, inputs = self._build_input_tensors(messages, family=family)
        generated_text = self._generate_text("", inputs, llm_args)
        repaired_text = ""
        used_repair = False
        failure_stage = ""
        failure_reason = ""
        payload: Any = None

        try:
            payload = _safe_json_loads(generated_text)
        except Exception as exc:
            failure_stage = "raw_parse"
            failure_reason = f"{type(exc).__name__}: {exc}"
            repaired_text = self._repair_json_object(generated_text, messages, llm_args)
            used_repair = True
            try:
                payload = _safe_json_loads(repaired_text)
                failure_stage = ""
                failure_reason = ""
            except Exception as repair_exc:
                failure_stage = "repair_parse"
                failure_reason = f"{type(repair_exc).__name__}: {repair_exc}"

        return {
            "payload": payload,
            "raw_text": generated_text,
            "repaired_text": repaired_text,
            "used_repair": used_repair,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        }

    def generate(
        self,
        *,
        messages: list[Any],
        tools: Optional[list[Any]] = None,
        tool_choice: Optional[str] = None,
        call_name: Optional[str] = None,
        **llm_args: Any,
    ) -> Any:
        AssistantMessage = None
        try:
            from tau2.data_model.message import AssistantMessage
        except Exception:
            pass

        total_start = time.perf_counter()
        family = self._call_family(call_name)
        chat_messages, prompt_text = _build_prompt(messages, tools, call_name, tool_choice)
        prompt_built = time.perf_counter()
        prompt_text, inputs = self._build_input_tensors(chat_messages)
        tensors_built = time.perf_counter()
        timing_enabled = _env_flag_enabled("LLM_DEBUG_TIMING") or _env_flag_enabled("MULTIWOZ_DEBUG_TIMING")
        if timing_enabled:
            prompt_token_count_pre = int(inputs["input_ids"].shape[-1])
            print(
                "[llm-timing] "
                f"stage=generate_start "
                f"model={self.model_name} "
                f"family={family} "
                f"call_name={call_name or 'none'} "
                f"prompt_tokens={prompt_token_count_pre}",
                flush=True,
            )
        start = time.perf_counter()
        generated_text = self._generate_text(prompt_text, inputs, llm_args)
        generation_time_seconds = time.perf_counter() - start
        generated_at = time.perf_counter()
        content, tool_calls = self._parse_message_payload(generated_text, tools, tool_choice)
        parse_done = time.perf_counter()

        if content is None and tool_calls is None:
            repaired_text = self._repair_output(generated_text, chat_messages, tools, tool_choice, llm_args)
            content, tool_calls = self._parse_message_payload(repaired_text, tools, tool_choice)
            generated_text = repaired_text
            parse_done = time.perf_counter()
        content, tool_calls = self._coerce_message_payload(content, tool_calls, generated_text)

        assert self.tokenizer is not None
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        completion_token_count = max(
            0, len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        )
        usage = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
        }

        response_payload = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "cost": 0.0,
            "usage": usage,
            "raw_data": {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "call_name": call_name,
                "prompt": prompt_text,
                "response_text": generated_text,
                "generation_time_seconds": generation_time_seconds,
            },
            "generation_time_seconds": generation_time_seconds,
        }
        if timing_enabled:
            total_done = time.perf_counter()
            print(
                "[llm-timing] "
                f"model={self.model_name} "
                f"family={family} "
                f"call_name={call_name or 'none'} "
                f"prompt_tokens={prompt_token_count} "
                f"completion_tokens={completion_token_count} "
                f"build_prompt_s={prompt_built - total_start:.3f} "
                f"build_tensors_s={tensors_built - prompt_built:.3f} "
                f"generate_s={generated_at - tensors_built:.3f} "
                f"parse_s={parse_done - generated_at:.3f} "
                f"total_s={total_done - total_start:.3f}",
                flush=True,
            )
        if AssistantMessage is None:
            return response_payload
        return AssistantMessage(**response_payload)


LOCAL_GENERATION_MODULES = [
    "tau2.agent.llm_agent",
    "tau2.user.user_simulator",
    "tau2.user.user_simulator_streaming",
    "tau2.environment.utils.interface_agent",
]


def patch_tau2_generation(generate_fn: Any) -> None:
    """Patch only the local agent/user generation paths.

    The tau2-bench judge path must keep using the original llm_utils.generate
    binding so that evaluators such as NLAssertionsEvaluator remain on the
    upstream tau2-bench implementation.
    """
    ensure_tau2_repo_on_path(parse_str_env("TAU2_REPO", DEFAULTS["TAU2_REPO"]))
    for module_name in LOCAL_GENERATION_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "generate"):
            setattr(module, "generate", generate_fn)


def build_tau2_config(mods: dict[str, Any], runner_cfg: RunnerConfig):
    TextRunConfig = mods["TextRunConfig"]
    tau2_config = mods["config"]

    llm_args_agent = deepcopy(tau2_config.DEFAULT_LLM_ARGS_AGENT)
    llm_args_user = deepcopy(tau2_config.DEFAULT_LLM_ARGS_USER)
    llm_args_agent["max_new_tokens"] = runner_cfg.max_new_tokens
    llm_args_user["max_new_tokens"] = runner_cfg.max_new_tokens
    llm_args_agent = apply_backend_llm_args(runner_cfg, llm_args_agent)
    llm_args_user = apply_backend_llm_args(runner_cfg, llm_args_user)
    llm_model_name = normalize_model_name_for_backend(runner_cfg, runner_cfg.llm_model_name)

    return TextRunConfig(
        domain=runner_cfg.domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=llm_model_name,
        llm_args_agent=llm_args_agent,
        llm_user=llm_model_name,
        llm_args_user=llm_args_user,
        num_trials=runner_cfg.pass_hat_k,
        max_steps=runner_cfg.max_steps,
        max_errors=runner_cfg.max_errors,
        seed=runner_cfg.seed,
        max_concurrency=1,
        save_to=None,
        task_split_name="base",
        auto_review=False,
    )


def get_tasks_for_domain(mods: dict[str, Any], domain: str) -> list[Any]:
    get_tasks = mods["get_tasks"]
    try:
        return get_tasks(domain, task_split_name="base")
    except TypeError:
        return get_tasks(domain, "base")


def is_successful_reward(reward: Optional[float]) -> bool:
    if reward is None:
        return False
    return abs(reward - 1.0) <= 1e-6


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    return value


def compute_dialogue_rounds(simulation: Any) -> int:
    messages = []
    if hasattr(simulation, "get_messages"):
        try:
            messages = simulation.get_messages()
        except Exception:
            messages = simulation.messages or []
    elif getattr(simulation, "messages", None) is not None:
        messages = simulation.messages

    turns = [m.turn_idx for m in messages if getattr(m, "turn_idx", None) is not None]
    if turns:
        return max(turns) + 1
    return sum(1 for m in messages if getattr(m, "role", None) != "system")


def get_simulation_step_count(simulation: Any) -> Optional[int]:
    info = getattr(simulation, "info", None)
    if not isinstance(info, dict):
        return None
    raw_value = info.get("step_count")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def attach_step_count(simulation: Any, step_count: int | None) -> Any:
    info = dict(getattr(simulation, "info", None) or {})
    info["step_count"] = step_count
    simulation.info = info
    return simulation


def execute_sample_run(
    mods: dict[str, Any],
    runner_cfg: RunnerConfig,
    task: Any,
    *,
    trial: int,
    seed: int,
    save_dir: Path,
) -> Any:
    EvaluationType = mods["EvaluationType"]
    run_single_task = mods["run_single_task"]
    SimulationRun = mods["SimulationRun"]
    TerminationReason = mods["TerminationReason"]

    config = build_tau2_config(mods, runner_cfg)
    logger = logging.getLogger(__name__)
    started_at = datetime.now()
    started_monotonic = time.monotonic()
    timeout_seconds = float(getattr(runner_cfg, "sample_timeout_seconds", DEFAULTS["SAMPLE_TIMEOUT_SECONDS"]))
    max_attempts = int(runner_cfg.max_retries) + 1

    def _run_once() -> Any:
        return run_single_task(
            config,
            task,
            seed=seed,
            evaluation_type=EvaluationType.ALL,
            save_dir=save_dir,
            verbose_logs=False,
            audio_debug=False,
            audio_taps=False,
            auto_review=False,
            review_mode="full",
        )

    def _timeout_failure(error: TimeoutError, attempts: int) -> Any:
        ended_at = datetime.now()
        return SimulationRun(
            id=str(uuid.uuid4()),
            task_id=getattr(task, "id", "unknown"),
            timestamp=started_at.isoformat(),
            start_time=started_at.isoformat(),
            end_time=ended_at.isoformat(),
            duration=max(0.0, time.monotonic() - started_monotonic),
            termination_reason=TerminationReason.TIMEOUT,
            messages=[],
            trial=trial,
            seed=seed,
            info={
                "error": str(error),
                "error_type": type(error).__name__,
                "failed_after_attempts": attempts,
                "timeout_seconds": timeout_seconds,
            },
        )

    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        elapsed = time.monotonic() - started_monotonic
        remaining = timeout_seconds - elapsed if timeout_seconds > 0 else float("inf")
        if timeout_seconds > 0 and remaining <= 0:
            return _timeout_failure(
                TimeoutError(f"Sample exceeded total timeout of {timeout_seconds:.0f}s"),
                attempt,
            )

        if attempt > 0:
            logger.warning(
                "Retry %s/%s for task %s: %s",
                attempt,
                runner_cfg.max_retries,
                getattr(task, "id", "unknown"),
                last_exception,
            )
            if runner_cfg.retry_delay > 0:
                try:
                    if timeout_seconds > 0:
                        with _sample_timeout(max(0.0, remaining)):
                            time.sleep(runner_cfg.retry_delay)
                    else:
                        time.sleep(runner_cfg.retry_delay)
                except TimeoutError as exc:
                    return _timeout_failure(exc, attempt)

        try:
            if timeout_seconds > 0:
                with _sample_timeout(
                    max(0.0, timeout_seconds - (time.monotonic() - started_monotonic))
                ):
                    result = _run_once()
            else:
                result = _run_once()
            result.trial = trial
            return result
        except TimeoutError as exc:
            return _timeout_failure(exc, attempt + 1)
        except Exception as exc:
            last_exception = exc
            if attempt >= max_attempts - 1:
                break

    ended_at = datetime.now()
    assert last_exception is not None
    return SimulationRun(
        id=str(uuid.uuid4()),
        task_id=getattr(task, "id", "unknown"),
        timestamp=started_at.isoformat(),
        start_time=started_at.isoformat(),
        end_time=ended_at.isoformat(),
        duration=max(0.0, time.monotonic() - started_monotonic),
        termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
        messages=[],
        trial=trial,
        seed=seed,
        info={
            "error": str(last_exception),
            "error_type": type(last_exception).__name__,
            "failed_after_attempts": max_attempts,
        },
    )


def compute_summary(
    mods: dict[str, Any],
    results: Any,
    runner_cfg: RunnerConfig,
    start_time: float,
    total_samples: int,
    success_count: int,
    failure_count: int,
) -> dict[str, Any]:
    compute_metrics = mods["compute_metrics"]
    metrics = compute_metrics(results)
    metrics_dict = _sanitize_json_value(metrics.model_dump())
    simulations = list(getattr(results, "simulations", []) or [])

    termination_counts = Counter(
        _simulation_termination_reason(simulation) for simulation in simulations
    )
    reward_values = [
        reward
        for reward in (_simulation_reward_value(simulation) for simulation in simulations)
        if reward is not None
    ]
    rewarded_sample_count = len(reward_values)
    missing_reward_count = max(0, len(simulations) - rewarded_sample_count)
    avg_reward_all_samples = (
        sum(reward_values) / len(simulations) if simulations else 0.0
    )

    avg_dialogue_rounds = 0.0
    if simulations:
        rounds = [compute_dialogue_rounds(sim) for sim in simulations]
        avg_dialogue_rounds = sum(rounds) / len(rounds) if rounds else 0.0

    avg_step_count = 0.0
    if simulations:
        step_counts = [
            step_count
            for step_count in (get_simulation_step_count(sim) for sim in simulations)
            if step_count is not None
        ]
        avg_step_count = sum(step_counts) / len(step_counts) if step_counts else 0.0

    duration_seconds = time.time() - start_time
    completion_rate = success_count / total_samples if total_samples else 0.0

    pass_hat_ks = metrics_dict.get("pass_hat_ks", {})
    pass_hat_focus = pass_hat_ks.get(runner_cfg.pass_hat_k)
    if pass_hat_focus is None:
        pass_hat_focus = pass_hat_ks.get(str(runner_cfg.pass_hat_k))
    pass_hat_focus = _sanitize_json_value(pass_hat_focus)
    merged_timeout_count = max(
        int(metrics_dict.get("termination_timeout", 0) or 0),
        int(termination_counts.get("timeout", 0) or 0),
        int(termination_counts.get("TIMEOUT", 0) or 0),
    )

    summary = OrderedDict(
        [
            ("benchmark", "tau2-bench"),
            ("domain", runner_cfg.domain),
            ("mode", runner_cfg.mode),
            ("sop_text", runner_cfg.sop_text),
            ("num_cycles", runner_cfg.num_cycles),
            ("model_name", runner_cfg.llm_model_name),
            ("model_path", runner_cfg.llm_model_path),
            ("pass_hat_k", runner_cfg.pass_hat_k),
            ("num_trials", runner_cfg.pass_hat_k),
            ("num_tasks", len(results.tasks)),
            ("total_samples", total_samples),
            ("success_count", success_count),
            ("failure_count", failure_count),
            ("task_completion_rate", completion_rate),
            ("avg_dialogue_rounds", avg_dialogue_rounds),
            ("avg_step_count", avg_step_count),
            ("duration_seconds", duration_seconds),
            ("avg_reward", _sanitize_json_value(metrics_dict.get("avg_reward", 0.0))),
            ("avg_reward_all_samples", avg_reward_all_samples),
            ("rewarded_sample_count", rewarded_sample_count),
            ("missing_reward_count", missing_reward_count),
            ("pass_hat_ks", pass_hat_ks),
            ("pass_hat_k_focus", pass_hat_focus),
            ("avg_agent_cost", _sanitize_json_value(metrics_dict.get("avg_agent_cost", 0.0))),
            ("total_simulations", metrics_dict.get("total_simulations", 0)),
            ("total_tasks", metrics_dict.get("total_tasks", 0)),
            ("infra_error_count", metrics_dict.get("infra_error_count", 0)),
            ("total_read_actions", metrics_dict.get("total_read_actions", 0)),
            ("correct_read_actions", metrics_dict.get("correct_read_actions", 0)),
            ("total_write_actions", metrics_dict.get("total_write_actions", 0)),
            ("correct_write_actions", metrics_dict.get("correct_write_actions", 0)),
            ("db_match_count", metrics_dict.get("db_match_count", 0)),
            ("db_mismatch_count", metrics_dict.get("db_mismatch_count", 0)),
            ("db_not_checked", metrics_dict.get("db_not_checked", 0)),
            ("auth_succeeded", metrics_dict.get("auth_succeeded", 0)),
            ("auth_failed", metrics_dict.get("auth_failed", 0)),
            ("auth_not_needed", metrics_dict.get("auth_not_needed", 0)),
            ("auth_not_checked", metrics_dict.get("auth_not_checked", 0)),
            ("termination_user_stop", metrics_dict.get("termination_user_stop", 0)),
            ("termination_agent_stop", metrics_dict.get("termination_agent_stop", 0)),
            ("termination_max_steps", metrics_dict.get("termination_max_steps", 0)),
            ("termination_timeout", merged_timeout_count),
            ("termination_error", metrics_dict.get("termination_error", 0)),
            (
                "termination_infrastructure_error",
                metrics_dict.get("termination_infrastructure_error", 0),
            ),
            (
                "sims_with_unresponsive_period",
                metrics_dict.get("sims_with_unresponsive_period", 0),
            ),
            (
                "sims_with_responsiveness_info",
                metrics_dict.get("sims_with_responsiveness_info", 0),
            ),
            (
                "agent_errors_by_severity",
                metrics_dict.get("agent_errors_by_severity", {}),
            ),
            (
                "user_errors_by_severity",
                metrics_dict.get("user_errors_by_severity", {}),
            ),
            (
                "sims_by_max_agent_severity",
                metrics_dict.get("sims_by_max_agent_severity", {}),
            ),
            (
                "sims_by_max_user_severity",
                metrics_dict.get("sims_by_max_user_severity", {}),
            ),
            (
                "sims_by_first_critical_source",
                metrics_dict.get("sims_by_first_critical_source", {}),
            ),
            (
                "agent_error_tags_by_severity",
                metrics_dict.get("agent_error_tags_by_severity", {}),
            ),
            (
                "user_error_tags_by_severity",
                metrics_dict.get("user_error_tags_by_severity", {}),
            ),
            ("tau2_agent_metrics", metrics_dict),
            (
                "dialogue_round_definition",
                "dialogue_rounds = max(turn_idx) + 1, fallback to non-system message count",
            ),
        ]
    )
    return summary


def _worker_run_job(
    mods: dict[str, Any],
    runner_cfg: RunnerConfig,
    worker_spec: WorkerSpec,
    job: Job,
) -> dict[str, Any]:
    SimulationRun = mods["SimulationRun"]
    Task = mods["Task"]
    TerminationReason = mods["TerminationReason"]

    task = Task.model_validate(job.task_data)

    try:
        result = execute_sample_run(
            mods,
            runner_cfg,
            task,
            trial=job.trial,
            seed=job.seed,
            save_dir=Path(worker_spec.run_dir),
        )
    except Exception as exc:
        now = datetime.now().isoformat()
        result = SimulationRun(
            id=job.job_id,
            task_id=task.id,
            timestamp=now,
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
            messages=[],
            trial=job.trial,
            seed=job.seed,
            info={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
    if worker_spec.evolve_components:
        trajectory_text = render_trajectory_text(_simulation_messages(result))
        runtime_warnings = _simulation_warnings(result)
        task_family_prior = _safe_task_family(worker_spec.domain, task)
        task_query = _derive_task_query(task, result)
        reward_signal = build_reward_signal(
            sample_id=job.job_id,
            trajectory_text=trajectory_text,
            reward_value=_simulation_reward_value(result),
            termination_reason=_simulation_termination_reason(result),
            meta_sop_text=worker_spec.meta_sop_text,
            task_family=task_family_prior,
            task_id=task.id,
            warnings=runtime_warnings,
            domain=worker_spec.domain,
            task_query=task_query,
            extractor_generate=_extractor_generate_for_config(
                runner_cfg,
                device_id=worker_spec.cuda_visible_devices,
            ),
        )
        info = dict(getattr(result, "info", None) or {})
        info["_runner_reward_signal"] = reward_signal
        if (
            "workflow_memory" in worker_spec.evolve_components
            and is_successful_reward(_simulation_reward_value(result))
        ):
            info["_runner_workflow_signal"] = build_workflow_memory_signal(
                sample_id=job.job_id,
                trajectory_text=trajectory_text,
                task_query=task_query,
                task_family=task_family_prior,
                domain=worker_spec.domain,
                extractor_generate=_extractor_generate_for_config(
                    runner_cfg,
                    device_id=worker_spec.cuda_visible_devices,
                ),
            )
        result.info = info
    return result.model_dump(mode="json")


def worker_main(
    worker_spec: WorkerSpec,
    task_queue: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    configure_runtime_logging(worker_spec.tau2_log_level)
    os.environ["CUDA_VISIBLE_DEVICES"] = worker_spec.cuda_visible_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    result_batch: list[dict[str, Any]] = []

    def _flush_result_batch() -> None:
        nonlocal result_batch
        if not result_batch:
            return
        result_queue.put(
            {
                "worker_id": worker_spec.worker_id,
                "status": "ok",
                "simulations": result_batch,
                "batch_size": len(result_batch),
            }
        )
        result_batch = []

    try:
        ensure_tau2_repo_on_path(worker_spec.tau2_repo)
        mods = get_tau2_modules()
        from tau2.registry import registry
        import tau2.runner.batch as runner_batch
        import tau2.runner.simulation as runner_simulation

        install_backend_runtime(worker_spec, device_id=worker_spec.cuda_visible_devices)

        original_run_simulation = runner_simulation.run_simulation

        def _run_simulation_with_step_count(*args: Any, **kwargs: Any) -> Any:
            simulation = original_run_simulation(*args, **kwargs)
            orchestrator = args[0] if args else kwargs.get("orchestrator")
            return attach_step_count(
                simulation,
                getattr(orchestrator, "step_count", None),
            )

        runner_simulation.run_simulation = _run_simulation_with_step_count
        runner_batch.run_simulation = _run_simulation_with_step_count

        ready_queue.put({"worker_id": worker_spec.worker_id, "status": "ready"})

        while True:
            job = task_queue.get()
            if job is None:
                break
            original_get_env_constructor = registry.get_env_constructor
            prompt_prefix = str(
                getattr(job, "system_prompt_prefix", "") or worker_spec.system_prompt_prefix
            ).strip()
            if _agent_prompt_debug_enabled():
                _emit_agent_prompt_debug_block(
                    "PREFIX",
                    sample_id=str(getattr(job, "job_id", "") or "unknown"),
                    content=prompt_prefix,
                )
            if prompt_prefix:
                _patch_domain_policy_prefix(
                    registry,
                    worker_spec.domain,
                    lambda fixed=prompt_prefix: fixed,
                    lambda fixed=job.job_id: fixed,
                )
            try:
                packet = _worker_run_job(
                    mods,
                    RunnerConfig(
                        llm_backend=worker_spec.llm_backend,
                        llm_model_name=worker_spec.llm_model_name,
                        llm_model_path=worker_spec.llm_model_path,
                        llm_base_url=worker_spec.llm_base_url,
                        llm_api_key=worker_spec.llm_api_key,
                        gpu_ids=[worker_spec.cuda_visible_devices],
                        proc_per_node=1,
                        domain=worker_spec.domain,
                        mode=worker_spec.mode,
                        sop_text=worker_spec.sop_text,
                        num_cycles=worker_spec.num_cycles,
                        tau2_repo=worker_spec.tau2_repo,
                        pass_hat_k=worker_spec.pass_hat_k,
                        seed=worker_spec.seed,
                        max_steps=worker_spec.max_steps,
                        max_errors=worker_spec.max_errors,
                        max_retries=worker_spec.max_retries,
                        retry_delay=worker_spec.retry_delay,
                        sample_timeout_seconds=worker_spec.sample_timeout_seconds,
                        max_new_tokens=worker_spec.max_new_tokens,
                        repair_max_new_tokens=worker_spec.repair_max_new_tokens,
                        result_batch_size=worker_spec.result_batch_size,
                        result_save_every=worker_spec.result_save_every,
                        tau2_log_level=worker_spec.tau2_log_level,
                        results_root=Path(worker_spec.run_dir).parents[2],
                    ),
                    worker_spec,
                    job,
                )
                result_batch.append(packet)
            except Exception as exc:
                now = datetime.now().isoformat()
                failure = mods["SimulationRun"](
                    id=job.job_id,
                    task_id=job.task_data.get("id", "unknown"),
                    timestamp=now,
                    start_time=now,
                    end_time=now,
                    duration=0.0,
                    termination_reason=mods["TerminationReason"].INFRASTRUCTURE_ERROR,
                    messages=[],
                    trial=job.trial,
                    seed=job.seed,
                    info={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                result_batch.append(failure.model_dump(mode="json"))
            finally:
                registry.get_env_constructor = original_get_env_constructor
            if len(result_batch) >= worker_spec.result_batch_size:
                _flush_result_batch()
        _flush_result_batch()
    except Exception as exc:
        ready_queue.put(
            {
                "worker_id": worker_spec.worker_id,
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
    finally:
        try:
            _flush_result_batch()
        except Exception:
            pass


def run_self_evolving_benchmark(
    runner_cfg: RunnerConfig,
    mods: dict[str, Any],
    run_dir: Path,
    tasks: list[Any],
    context_bundle: Any,
    run_info: Any,
    tau2_repo_commit: str,
) -> dict[str, Any]:
    Results = mods["Results"]
    SimulationRun = mods["SimulationRun"]
    TerminationReason = mods["TerminationReason"]
    bench_start = time.time()
    from tau2.registry import registry

    evolve_parallel_mode = runner_cfg.evolve_parallel_mode.strip().lower()
    if evolve_parallel_mode == "async_parallel":
        evolve_parallel_mode = "async"
    if evolve_parallel_mode not in {"strict", "batch_sync", "async"}:
        raise ValueError(
            f"Unsupported EVOLVE_PARALLEL_MODE={runner_cfg.evolve_parallel_mode!r}; "
            "supported values: strict, batch_sync, async"
        )

    use_batch_sync = evolve_parallel_mode == "batch_sync"
    use_async_parallel = evolve_parallel_mode == "async"
    sync_order = runner_cfg.evolve_sync_order.strip().lower()
    requested_workers = max(
        1,
        len(runner_cfg.gpu_ids or [""]) * runner_cfg.proc_per_node,
    )
    use_worker_pool = (use_batch_sync or use_async_parallel) and requested_workers > 1

    if sync_order not in {"task_index"}:
        raise ValueError(
            f"Unsupported EVOLVE_SYNC_ORDER={runner_cfg.evolve_sync_order!r}; "
            "supported values: task_index"
        )

    if not use_worker_pool:
        import tau2.runner.batch as runner_batch
        import tau2.runner.simulation as runner_simulation

        install_backend_runtime(
            runner_cfg,
            device_id=runner_cfg.gpu_ids[0] if runner_cfg.gpu_ids else "",
        )

        original_run_simulation = runner_simulation.run_simulation

        def _run_simulation_with_step_count(*args: Any, **kwargs: Any) -> Any:
            simulation = original_run_simulation(*args, **kwargs)
            orchestrator = args[0] if args else kwargs.get("orchestrator")
            return attach_step_count(simulation, getattr(orchestrator, "step_count", None))

        runner_simulation.run_simulation = _run_simulation_with_step_count
        runner_batch.run_simulation = _run_simulation_with_step_count

    current_memory = seed_memory_snapshot(runner_cfg.domain)
    current_graph = seed_state_graph(runner_cfg.domain)
    current_workflow_memory = seed_workflow_memory(runner_cfg.domain)
    current_meta_sop_text = context_bundle.meta_sop_text
    embed_texts = _build_runner_embed_texts(runner_cfg)

    def _plan_prompt_prefix(
        plan: EvolveSamplePlan,
        *,
        memory_snapshot: dict[str, Any],
        graph_snapshot: dict[str, Any],
        workflow_memory_snapshot: dict[str, Any],
    ) -> str:
        sample_task = mods["Task"].model_validate(plan.job.task_data)
        task_query = _task_query_from_task(sample_task)
        task_family = _safe_task_family(runner_cfg.domain, sample_task)
        return _build_evolution_context_text(
            base_prompt=context_bundle.current_system_prompt,
            memory_snapshot=memory_snapshot,
            state_graph_snapshot=graph_snapshot,
            workflow_memory_snapshot=workflow_memory_snapshot,
            evolve_components=runner_cfg.evolve_components,
            task_query=task_query,
            task_family=task_family,
            retrieval_top_k=runner_cfg.retrieval_top_k,
            embed_texts=embed_texts,
        )

    seeds = _job_seed_list(runner_cfg.seed, runner_cfg.pass_hat_k)
    all_simulations: list[Any] = []
    total_cycles = max(1, runner_cfg.num_cycles)
    total_samples = total_cycles * len(seeds) * len(tasks)
    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(
            total=total_samples,
            desc=f"tau2-bench/{runner_cfg.domain} (evolve)",
        )
    except Exception:
        progress = None

    gpu_ids = runner_cfg.gpu_ids or [""]
    worker_pool_task_queue = None
    worker_pool_result_queue = None
    worker_pool_workers: list[Any] = []
    worker_pool_spec: WorkerSpec | None = None
    progress_updates_on_receive = use_worker_pool

    if use_worker_pool:
        mp_ctx = get_context("spawn")
        queue_batch_hint = max(runner_cfg.evolve_batch_size, requested_workers)
        worker_pool_task_queue = mp_ctx.Queue(
            maxsize=max(1, queue_batch_hint * 2)
        )
        worker_pool_ready_queue = mp_ctx.Queue()
        worker_pool_result_queue = mp_ctx.Queue()
        worker_specs: list[WorkerSpec] = []
        total_workers = max(1, len(gpu_ids) * runner_cfg.proc_per_node)
        for worker_id in range(total_workers):
            gpu_id = gpu_ids[worker_id % len(gpu_ids)]
            worker_specs.append(
                WorkerSpec(
                    worker_id=worker_id,
                    cuda_visible_devices=gpu_id,
                    llm_backend=runner_cfg.llm_backend,
                    llm_model_name=runner_cfg.llm_model_name,
                    llm_model_path=runner_cfg.llm_model_path,
                    llm_base_url=runner_cfg.llm_base_url,
                    llm_api_key=runner_cfg.llm_api_key,
                    tau2_repo=runner_cfg.tau2_repo,
                    run_dir=str(run_dir),
                    domain=runner_cfg.domain,
                    mode=runner_cfg.mode,
                    sop_text=runner_cfg.sop_text,
                    num_cycles=runner_cfg.num_cycles,
                    system_prompt_prefix="",
                    pass_hat_k=runner_cfg.pass_hat_k,
                    seed=runner_cfg.seed,
                    max_steps=runner_cfg.max_steps,
                    max_errors=runner_cfg.max_errors,
                    max_retries=runner_cfg.max_retries,
                    retry_delay=runner_cfg.retry_delay,
                    sample_timeout_seconds=runner_cfg.sample_timeout_seconds,
                    max_new_tokens=runner_cfg.max_new_tokens,
                    repair_max_new_tokens=runner_cfg.repair_max_new_tokens,
                    result_batch_size=1,
                    result_save_every=runner_cfg.result_save_every,
                    tau2_log_level=runner_cfg.tau2_log_level,
                    evolve_components=list(runner_cfg.evolve_components),
                    meta_sop_text=context_bundle.meta_sop_text,
                )
            )
        worker_pool_workers = [
            mp_ctx.Process(
                target=worker_main,
                args=(
                    spec,
                    worker_pool_task_queue,
                    worker_pool_ready_queue,
                    worker_pool_result_queue,
                ),
                daemon=False,
            )
            for spec in worker_specs
        ]
        for worker in worker_pool_workers:
            worker.start()

        ready = 0
        while ready < len(worker_pool_workers):
            packet = worker_pool_ready_queue.get()
            if packet.get("status") != "ready":
                for worker in worker_pool_workers:
                    if worker.is_alive():
                        worker.terminate()
                raise RuntimeError(f"Evolve worker failed to start: {packet}")
            ready += 1
    else:
        worker_pool_spec = WorkerSpec(
            worker_id=0,
            cuda_visible_devices=gpu_ids[0],
            llm_backend=runner_cfg.llm_backend,
            llm_model_name=runner_cfg.llm_model_name,
            llm_model_path=runner_cfg.llm_model_path,
            llm_base_url=runner_cfg.llm_base_url,
            llm_api_key=runner_cfg.llm_api_key,
            tau2_repo=runner_cfg.tau2_repo,
            run_dir=str(run_dir),
            domain=runner_cfg.domain,
            mode=runner_cfg.mode,
            sop_text=runner_cfg.sop_text,
            num_cycles=runner_cfg.num_cycles,
            system_prompt_prefix="",
            pass_hat_k=runner_cfg.pass_hat_k,
            seed=runner_cfg.seed,
            max_steps=runner_cfg.max_steps,
            max_errors=runner_cfg.max_errors,
            max_retries=runner_cfg.max_retries,
            retry_delay=runner_cfg.retry_delay,
            sample_timeout_seconds=runner_cfg.sample_timeout_seconds,
            max_new_tokens=runner_cfg.max_new_tokens,
            repair_max_new_tokens=runner_cfg.repair_max_new_tokens,
            result_batch_size=1,
            result_save_every=runner_cfg.result_save_every,
            tau2_log_level=runner_cfg.tau2_log_level,
            evolve_components=list(runner_cfg.evolve_components),
            meta_sop_text=context_bundle.meta_sop_text,
        )

    try:
        for cycle in range(total_cycles):
            cycle_start = time.time()
            cycle_dir = run_dir / f"loop_{cycle}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            cycle_results = Results(info=run_info, tasks=tasks, simulations=[])
            cycle_events: list[dict[str, Any]] = []
            cycle_success_count = 0
            cycle_failure_count = 0

            cycle_memory_path = cycle_dir / "memories.json"
            cycle_graph_path = cycle_dir / "state_graph.json"
            cycle_workflow_memory_path = cycle_dir / "workflow_memory.json"
            cycle_events_path = cycle_dir / "events.jsonl"
            cycle_meta_sop_path = cycle_dir / "meta_sop.md"
            cycle_results_path = cycle_dir / "results.json"
            cycle_summary_path = cycle_dir / "summary.json"
            _write_json_file(cycle_workflow_memory_path, current_workflow_memory)

            cycle_plans: list[EvolveSamplePlan] = []
            for trial, trial_seed in enumerate(seeds):
                for task_index, task in enumerate(tasks):
                    task_data = task.model_dump(mode="json")
                    cycle_plans.append(
                        EvolveSamplePlan(
                            job=Job(
                                job_id=f"{task.id}-c{cycle}-t{trial}-s{trial_seed}",
                                task_data=task_data,
                                trial=trial,
                                seed=trial_seed,
                            ),
                            task_index=task_index,
                            sequence_index=len(cycle_plans),
                        )
                    )

            if worker_pool_spec is not None:
                worker_spec_for_cycle = WorkerSpec(
                    worker_id=worker_pool_spec.worker_id,
                    cuda_visible_devices=worker_pool_spec.cuda_visible_devices,
                    llm_backend=worker_pool_spec.llm_backend,
                    llm_model_name=worker_pool_spec.llm_model_name,
                    llm_model_path=worker_pool_spec.llm_model_path,
                    llm_base_url=worker_pool_spec.llm_base_url,
                    llm_api_key=worker_pool_spec.llm_api_key,
                    tau2_repo=worker_pool_spec.tau2_repo,
                    run_dir=str(cycle_dir),
                    domain=worker_pool_spec.domain,
                    mode=worker_pool_spec.mode,
                    sop_text=worker_pool_spec.sop_text,
                    num_cycles=worker_pool_spec.num_cycles,
                    system_prompt_prefix=worker_pool_spec.system_prompt_prefix,
                    pass_hat_k=worker_pool_spec.pass_hat_k,
                    seed=worker_pool_spec.seed,
                    max_steps=worker_pool_spec.max_steps,
                    max_errors=worker_pool_spec.max_errors,
                    max_retries=worker_pool_spec.max_retries,
                    retry_delay=worker_pool_spec.retry_delay,
                    sample_timeout_seconds=worker_pool_spec.sample_timeout_seconds,
                    max_new_tokens=worker_pool_spec.max_new_tokens,
                    repair_max_new_tokens=worker_pool_spec.repair_max_new_tokens,
                    result_batch_size=worker_pool_spec.result_batch_size,
                    result_save_every=worker_pool_spec.result_save_every,
                    tau2_log_level=worker_pool_spec.tau2_log_level,
                    evolve_components=list(worker_pool_spec.evolve_components),
                    meta_sop_text=worker_pool_spec.meta_sop_text,
                )
            else:
                worker_spec_for_cycle = None

            def _record_sample(plan: EvolveSamplePlan, simulation: Any) -> None:
                nonlocal current_memory
                nonlocal current_graph
                nonlocal current_workflow_memory
                nonlocal cycle_success_count
                nonlocal cycle_failure_count

                sample_task = mods["Task"].model_validate(plan.job.task_data)
                task_family_prior = _safe_task_family(runner_cfg.domain, sample_task)
                reward_signal = _simulation_precomputed_reward_signal(simulation)
                if not isinstance(reward_signal, dict):
                    reward_signal = build_reward_signal(
                        sample_id=plan.job.job_id,
                        trajectory_text=render_trajectory_text(_simulation_messages(simulation)),
                        reward_value=_simulation_reward_value(simulation),
                        termination_reason=_simulation_termination_reason(simulation),
                        meta_sop_text=current_meta_sop_text,
                        task_family=task_family_prior,
                        task_id=sample_task.id,
                        warnings=_simulation_warnings(simulation),
                        domain=runner_cfg.domain,
                        task_query=_derive_task_query(sample_task, simulation),
                        extractor_generate=_extractor_generate_for_config(
                            runner_cfg,
                            device_id=runner_cfg.gpu_ids[0] if runner_cfg.gpu_ids else "",
                        )
                        if not use_worker_pool
                        else None,
                    )
                experience_packet = reward_signal.get("experience_packet")
                if not isinstance(experience_packet, dict):
                    experience_packet = {}
                task_family_focus = str(
                    experience_packet.get("family_focus") or task_family_prior
                )
                update_family = task_family_focus or task_family_prior

                memory_before = deepcopy(current_memory)
                graph_before = deepcopy(current_graph)
                workflow_before = deepcopy(current_workflow_memory)

                if (
                    "memory" in runner_cfg.evolve_components
                    and reward_signal["outcome_tag"] == "failure"
                ):
                    current_memory = apply_memory_update(
                        current_memory,
                        family=update_family,
                        posterior_text=reward_signal["posterior_text"],
                        outcome_tag=reward_signal["outcome_tag"],
                        source_cycle=cycle,
                        source_task=plan.job.job_id,
                        reward_value=reward_signal["reward_value"],
                        meta_sop_text=current_meta_sop_text,
                        task_id=sample_task.id,
                        experience_packet=experience_packet,
                    )
                if "state_graph" in runner_cfg.evolve_components:
                    current_graph = apply_state_graph_update(
                        current_graph,
                        family=update_family,
                        posterior_text=reward_signal["posterior_text"],
                        outcome_tag=reward_signal["outcome_tag"],
                        source_cycle=cycle,
                        source_task=plan.job.job_id,
                        reward_value=reward_signal["reward_value"],
                        meta_sop_text=current_meta_sop_text,
                        experience_packet=experience_packet,
                    )
                if (
                    "workflow_memory" in runner_cfg.evolve_components
                    and reward_signal["outcome_tag"] == "success"
                ):
                    workflow_signal = _simulation_precomputed_workflow_signal(simulation)
                    if not isinstance(workflow_signal, dict):
                        workflow_signal = build_workflow_memory_signal(
                            sample_id=plan.job.job_id,
                            trajectory_text=render_trajectory_text(_simulation_messages(simulation)),
                            task_query=_derive_task_query(sample_task, simulation),
                            task_family=task_family_prior,
                            domain=runner_cfg.domain,
                            extractor_generate=_extractor_generate_for_config(
                                runner_cfg,
                                device_id=runner_cfg.gpu_ids[0] if runner_cfg.gpu_ids else "",
                            )
                            if not use_worker_pool
                            else None,
                        )
                    if workflow_signal.get("update_enabled"):
                        current_workflow_memory = apply_workflow_memory_update(
                            current_workflow_memory,
                            family=task_family_prior,
                            source_cycle=cycle,
                            source_task=plan.job.job_id,
                            task_id=sample_task.id,
                            task_query=_derive_task_query(sample_task, simulation),
                            workflow_items=workflow_signal.get("workflow_items", []),
                        )

                memory_delta = (
                    _memory_delta(memory_before, current_memory)
                    if "memory" in runner_cfg.evolve_components
                    else None
                )
                graph_delta = (
                    _graph_delta(graph_before, current_graph)
                    if "state_graph" in runner_cfg.evolve_components
                    else None
                )
                workflow_delta = (
                    _workflow_delta(workflow_before, current_workflow_memory)
                    if "workflow_memory" in runner_cfg.evolve_components
                    else None
                )

                event = record_evolution_event(
                    sample_id=plan.job.job_id,
                    task_id=sample_task.id,
                    cycle_id=cycle,
                    outcome_tag=reward_signal["outcome_tag"],
                    reward_value=reward_signal["reward_value"],
                    posterior_text=reward_signal["posterior_text"],
                    task_family=update_family,
                    family_prior=task_family_prior,
                    family_focus=task_family_focus,
                    diagnostic_flags=experience_packet.get("diagnostic_flags", []),
                    experience_packet=experience_packet,
                    memory_delta=memory_delta,
                    graph_delta=graph_delta,
                    workflow_delta=workflow_delta,
                )
                cycle_events.append(event)
                all_simulations.append(simulation)
                cycle_results.simulations.append(simulation)

                if is_successful_reward(reward_signal["reward_value"]):
                    cycle_success_count += 1
                else:
                    cycle_failure_count += 1

                cycle_results.save(cycle_results_path, format="json")
                _write_jsonl_file(cycle_events_path, cycle_events)
                _write_json_file(cycle_memory_path, current_memory)
                _write_json_file(cycle_graph_path, current_graph)
                _write_json_file(cycle_workflow_memory_path, current_workflow_memory)
                cycle_meta_sop_path.write_text(
                    current_meta_sop_text.rstrip() + "\n",
                    encoding="utf-8",
                )
                if progress is not None:
                    progress.set_postfix(
                        cycle=f"{cycle + 1}/{total_cycles}",
                        success=cycle_success_count,
                        failure=cycle_failure_count,
                        refresh=True,
                    )
                    if not progress_updates_on_receive:
                        progress.update(1)

            def _run_local_plan(plan: EvolveSamplePlan, prompt_prefix: str) -> Any:
                assert worker_spec_for_cycle is not None
                sample_task = mods["Task"].model_validate(plan.job.task_data)
                prompt_job = Job(
                    job_id=plan.job.job_id,
                    task_data=plan.job.task_data,
                    trial=plan.job.trial,
                    seed=plan.job.seed,
                    system_prompt_prefix=prompt_prefix,
                )
                try:
                    if prompt_prefix.strip():
                        original_get_env_constructor = registry.get_env_constructor
                        if _agent_prompt_debug_enabled():
                            _emit_agent_prompt_debug_block(
                                "PREFIX",
                                sample_id=plan.job.job_id,
                                content=prompt_prefix,
                            )
                        _patch_domain_policy_prefix(
                            registry,
                            runner_cfg.domain,
                            lambda fixed=prompt_prefix: fixed,
                            lambda fixed=plan.job.job_id: fixed,
                        )
                    else:
                        original_get_env_constructor = None
                    simulation_data = _worker_run_job(
                        mods,
                        runner_cfg,
                        worker_spec_for_cycle,
                        prompt_job,
                    )
                    return SimulationRun.model_validate(simulation_data)
                except Exception as exc:
                    now = datetime.now().isoformat()
                    return SimulationRun(
                        id=plan.job.job_id,
                        task_id=sample_task.id,
                        timestamp=now,
                        start_time=now,
                        end_time=now,
                        duration=0.0,
                        termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                        messages=[],
                        trial=plan.job.trial,
                        seed=plan.job.seed,
                        info={
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                finally:
                    if prompt_prefix.strip():
                        registry.get_env_constructor = original_get_env_constructor

            if not use_batch_sync and not use_async_parallel:
                for plan in cycle_plans:
                    simulation = _run_local_plan(
                        plan,
                        _plan_prompt_prefix(
                            plan,
                            memory_snapshot=current_memory,
                            graph_snapshot=current_graph,
                            workflow_memory_snapshot=current_workflow_memory,
                        ),
                    )
                    _record_sample(plan, simulation)
            elif use_batch_sync:
                batch_size = max(1, runner_cfg.evolve_batch_size)
                for start_idx in range(0, len(cycle_plans), batch_size):
                    batch_plans = cycle_plans[start_idx : start_idx + batch_size]
                    batch_records: list[tuple[EvolveSamplePlan, Any]] = []

                    if use_worker_pool:
                        assert worker_pool_task_queue is not None
                        assert worker_pool_result_queue is not None

                        pending: dict[tuple[str, int, int], EvolveSamplePlan] = {}
                        for plan in batch_plans:
                            pending_key = (
                                str(plan.job.task_data.get("id", "")),
                                int(plan.job.trial),
                                int(plan.job.seed),
                            )
                            pending[pending_key] = plan
                            prompt_prefix = _plan_prompt_prefix(
                                plan,
                                memory_snapshot=current_memory,
                                graph_snapshot=current_graph,
                                workflow_memory_snapshot=current_workflow_memory,
                            )
                            worker_pool_task_queue.put(
                                Job(
                                    job_id=plan.job.job_id,
                                    task_data=plan.job.task_data,
                                    trial=plan.job.trial,
                                    seed=plan.job.seed,
                                    system_prompt_prefix=prompt_prefix,
                                )
                            )

                        while len(batch_records) < len(batch_plans):
                            packet = worker_pool_result_queue.get()
                            simulations_data = packet.get("simulations")
                            if simulations_data is None:
                                simulations_data = [packet["simulation"]]
                            for simulation_data in simulations_data:
                                simulation = SimulationRun.model_validate(simulation_data)
                                key = (
                                    str(simulation.task_id),
                                    int(simulation.trial),
                                    int(simulation.seed),
                                )
                                plan = pending.pop(key, None)
                                if plan is None:
                                    raise RuntimeError(
                                        "Received unexpected simulation in batch_sync "
                                        f"mode for key={key}"
                                    )
                                batch_records.append((plan, simulation))
                                if progress is not None and progress_updates_on_receive:
                                    progress.update(1)
                                    progress.set_postfix(
                                        cycle=f"{cycle + 1}/{total_cycles}",
                                        success=cycle_success_count,
                                        failure=cycle_failure_count,
                                        batch=f"{len(batch_records)}/{len(batch_plans)}",
                                        refresh=True,
                                    )
                                if len(batch_records) >= len(batch_plans):
                                    break
                    else:
                        for plan in batch_plans:
                            simulation = _run_local_plan(
                                plan,
                                _plan_prompt_prefix(
                                    plan,
                                    memory_snapshot=current_memory,
                                    graph_snapshot=current_graph,
                                    workflow_memory_snapshot=current_workflow_memory,
                                ),
                            )
                            batch_records.append((plan, simulation))

                    batch_records.sort(
                        key=lambda item: (
                            item[0].task_index,
                            item[0].job.trial,
                            item[0].sequence_index,
                            item[0].job.job_id,
                        )
                    )
                    for plan, simulation in batch_records:
                        _record_sample(plan, simulation)
            else:
                if use_worker_pool:
                    assert worker_pool_task_queue is not None
                    assert worker_pool_result_queue is not None

                    inflight: dict[tuple[str, int, int], EvolveSamplePlan] = {}
                    next_plan_idx = 0
                    max_inflight = max(1, requested_workers)

                    def _submit_plan(plan: EvolveSamplePlan) -> None:
                        key = (
                            str(plan.job.task_data.get("id", "")),
                            int(plan.job.trial),
                            int(plan.job.seed),
                        )
                        if key in inflight:
                            raise RuntimeError(
                                f"Duplicate in-flight async evolve sample key={key}"
                            )
                        inflight[key] = plan
                        prompt_prefix = _plan_prompt_prefix(
                            plan,
                            memory_snapshot=current_memory,
                            graph_snapshot=current_graph,
                            workflow_memory_snapshot=current_workflow_memory,
                        )
                        worker_pool_task_queue.put(
                            Job(
                                job_id=plan.job.job_id,
                                task_data=plan.job.task_data,
                                trial=plan.job.trial,
                                seed=plan.job.seed,
                                system_prompt_prefix=prompt_prefix,
                            )
                        )

                    while next_plan_idx < len(cycle_plans) and len(inflight) < max_inflight:
                        _submit_plan(cycle_plans[next_plan_idx])
                        next_plan_idx += 1

                    while inflight:
                        packet = worker_pool_result_queue.get()
                        simulations_data = packet.get("simulations")
                        if simulations_data is None:
                            simulations_data = [packet["simulation"]]
                        for simulation_data in simulations_data:
                            simulation = SimulationRun.model_validate(simulation_data)
                            key = (
                                str(simulation.task_id),
                                int(simulation.trial),
                                int(simulation.seed),
                            )
                            plan = inflight.pop(key, None)
                            if plan is None:
                                raise RuntimeError(
                                    "Received unexpected simulation in async evolve mode "
                                    f"for key={key}"
                                )
                            if progress is not None and progress_updates_on_receive:
                                progress.update(1)
                                progress.set_postfix(
                                    cycle=f"{cycle + 1}/{total_cycles}",
                                    success=cycle_success_count,
                                    failure=cycle_failure_count,
                                    inflight=len(inflight),
                                    refresh=True,
                                )
                            _record_sample(plan, simulation)
                            while (
                                next_plan_idx < len(cycle_plans)
                                and len(inflight) < max_inflight
                            ):
                                _submit_plan(cycle_plans[next_plan_idx])
                                next_plan_idx += 1
                else:
                    for plan in cycle_plans:
                        simulation = _run_local_plan(plan, "")
                        _record_sample(plan, simulation)

            cycle_summary = compute_summary(
                mods,
                cycle_results,
                runner_cfg,
                start_time=cycle_start,
                total_samples=len(cycle_results.simulations),
                success_count=cycle_success_count,
                failure_count=cycle_failure_count,
            )
            cycle_summary["workflow_entry_count"] = len(
                current_workflow_memory.get("entries", [])
            )
            cycle_summary["workflow_active_count"] = len(
                [
                    entry
                    for entry in current_workflow_memory.get("entries", [])
                    if str(entry.get("status", "")) == "active"
                ]
            )
            cycle_summary["results_path"] = str(cycle_results_path)
            cycle_summary["run_dir"] = str(cycle_dir)
            cycle_summary["tau2_repo_commit"] = tau2_repo_commit
            cycle_summary["runner_commit"] = get_git_commit(str(PROJECT_ROOT))
            cycle_summary["timestamp"] = datetime.now().isoformat()
            cycle_summary_path.write_text(json.dumps(cycle_summary, indent=2, ensure_ascii=False))
    finally:
        if use_worker_pool:
            for worker in worker_pool_workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in worker_pool_workers:
                worker.join(timeout=10.0)
        if progress is not None:
            progress.close()

    results = Results(info=run_info, tasks=tasks, simulations=all_simulations)
    results_path = run_dir / "results.json"
    results.save(results_path, format="json")
    _write_json_file(run_dir / "workflow_memory.json", current_workflow_memory)

    summary = compute_summary(
        mods,
        results,
        runner_cfg,
        start_time=bench_start,
        total_samples=len(all_simulations),
        success_count=sum(
            1
            for simulation in all_simulations
            if is_successful_reward(_simulation_reward_value(simulation))
        ),
        failure_count=sum(
            1
            for simulation in all_simulations
            if not is_successful_reward(_simulation_reward_value(simulation))
        ),
    )
    summary["workflow_entry_count"] = len(current_workflow_memory.get("entries", []))
    summary["workflow_active_count"] = len(
        [
            entry
            for entry in current_workflow_memory.get("entries", [])
            if str(entry.get("status", "")) == "active"
        ]
    )
    summary["results_path"] = str(results_path)
    summary["run_dir"] = str(run_dir)
    summary["tau2_repo_commit"] = tau2_repo_commit
    summary["runner_commit"] = get_git_commit(str(PROJECT_ROOT))
    summary["timestamp"] = datetime.now().isoformat()
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def load_runner_config() -> RunnerConfig:
    load_local_env_files()
    validate_judge_env()
    tau2_log_level = parse_str_env("TAU2_LOG_LEVEL", DEFAULTS["TAU2_LOG_LEVEL"])
    configure_runtime_logging(tau2_log_level)

    llm_backend = parse_str_env("LLM_BACKEND", DEFAULTS["LLM_BACKEND"])
    llm_model_name = parse_str_env("LLM_MODEL_NAME", DEFAULTS["LLM_MODEL_NAME"])
    llm_model_path = parse_str_env("LLM_MODEL_PATH", DEFAULTS["LLM_MODEL_PATH"])
    llm_base_url = parse_str_env("LLM_BASE_URL", DEFAULTS["LLM_BASE_URL"])
    llm_api_key = parse_str_env("LLM_API_KEY", DEFAULTS["LLM_API_KEY"])
    gpu_raw = parse_str_env("GPU", "")
    proc_per_node = parse_int_env("PROC_PER_NODE", DEFAULTS["PROC_PER_NODE"])
    domain = parse_str_env("DOMAIN", DEFAULTS["DOMAIN"])
    mode = parse_str_env("MODE", DEFAULTS["MODE"])
    sop_text = parse_str_env("SOP_TEXT", DEFAULTS["SOP_TEXT"])
    num_cycles = parse_int_env("NUM_CYCLES", DEFAULTS["NUM_CYCLES"])
    tau2_repo = parse_str_env("TAU2_REPO", DEFAULTS["TAU2_REPO"])
    pass_hat_k = parse_int_env("PASS_HAT_K", DEFAULTS["PASS_HAT_K"])
    max_steps = parse_int_env("MAX_STEPS", DEFAULTS["MAX_STEPS"])
    max_errors = parse_int_env("MAX_ERRORS", DEFAULTS["MAX_ERRORS"])
    max_retries = parse_int_env("MAX_RETRIES", DEFAULTS["MAX_RETRIES"])
    retry_delay = parse_float_env("RETRY_DELAY", DEFAULTS["RETRY_DELAY"])
    sample_timeout_seconds = parse_float_env(
        "SAMPLE_TIMEOUT_SECONDS",
        DEFAULTS["SAMPLE_TIMEOUT_SECONDS"],
    )
    max_new_tokens = parse_int_env(
        "MAX_NEW_TOKENS",
        2048,
    )
    repair_max_new_tokens = parse_int_env(
        "REPAIR_MAX_NEW_TOKENS", DEFAULTS["REPAIR_MAX_NEW_TOKENS"]
    )
    result_batch_size = max(
        1, parse_int_env("RESULT_BATCH_SIZE", DEFAULTS["RESULT_BATCH_SIZE"])
    )
    result_save_every = max(
        1, parse_int_env("RESULT_SAVE_EVERY", DEFAULTS["RESULT_SAVE_EVERY"])
    )
    embedding_model = parse_str_env("EMBEDDING_MODEL", DEFAULTS["EMBEDDING_MODEL"])
    embedding_base_url = parse_str_env(
        "EMBEDDING_BASE_URL",
        DEFAULTS["EMBEDDING_BASE_URL"],
    )
    embedding_api_key = parse_str_env(
        "EMBEDDING_API_KEY",
        parse_str_env("DASHSCOPE_API_KEY", ""),
    )
    retrieval_top_k = max(
        0,
        parse_int_env("RETRIEVAL_TOP_K", DEFAULTS["RETRIEVAL_TOP_K"]),
    )
    evolve_components = parse_csv_env("EVOLVE_COMPONENTS", DEFAULTS["EVOLVE_COMPONENTS"])
    evolve_parallel_mode = parse_str_env(
        "EVOLVE_PARALLEL_MODE",
        DEFAULTS["EVOLVE_PARALLEL_MODE"],
    ).strip().lower()
    if evolve_parallel_mode == "async_parallel":
        evolve_parallel_mode = "async"
    evolve_batch_size = max(
        1,
        parse_int_env("EVOLVE_BATCH_SIZE", DEFAULTS["EVOLVE_BATCH_SIZE"]),
    )
    evolve_sync_order = parse_str_env(
        "EVOLVE_SYNC_ORDER",
        DEFAULTS["EVOLVE_SYNC_ORDER"],
    ).strip().lower()

    if evolve_parallel_mode not in {"strict", "batch_sync", "async"}:
        raise ValueError(
            f"Unsupported EVOLVE_PARALLEL_MODE={evolve_parallel_mode!r}; "
            "supported values: strict, batch_sync, async"
        )
    if evolve_sync_order not in {"task_index"}:
        raise ValueError(
            f"Unsupported EVOLVE_SYNC_ORDER={evolve_sync_order!r}; "
            "supported values: task_index"
        )

    default_seed = parse_int_env("SEED", DEFAULTS["SEED"])
    try:
        ensure_tau2_repo_on_path(tau2_repo)
        import tau2.config as tau2_config

        default_seed = int(getattr(tau2_config, "DEFAULT_SEED", default_seed))
    except ModuleNotFoundError:
        tau2_config = None

    return RunnerConfig(
        llm_backend=llm_backend,
        llm_model_name=llm_model_name,
        llm_model_path=llm_model_path,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        gpu_ids=parse_gpu_list(gpu_raw),
        proc_per_node=proc_per_node,
        domain=domain,
        mode=mode,
        sop_text=sop_text,
        num_cycles=num_cycles,
        tau2_repo=tau2_repo,
        pass_hat_k=pass_hat_k,
        seed=default_seed,
        max_steps=max_steps,
        max_errors=max_errors,
        max_retries=max_retries,
        retry_delay=retry_delay,
        sample_timeout_seconds=sample_timeout_seconds,
        max_new_tokens=max_new_tokens,
        repair_max_new_tokens=repair_max_new_tokens,
        result_batch_size=result_batch_size,
        result_save_every=result_save_every,
        evolve_components=evolve_components,
        evolve_parallel_mode=evolve_parallel_mode,
        evolve_batch_size=evolve_batch_size,
        evolve_sync_order=evolve_sync_order,
        tau2_log_level=tau2_log_level,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        retrieval_top_k=retrieval_top_k,
    )


def _job_seed_list(seed: int, num_trials: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 1_000_000) for _ in range(num_trials)]


def run_benchmark() -> dict[str, Any]:
    runner_cfg = load_runner_config()
    mods = get_tau2_modules()
    Results = mods["Results"]
    logger = logging.getLogger(__name__)

    run_dir = build_run_dir(runner_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    bench_start = time.time()

    tasks = get_tasks_for_domain(mods, runner_cfg.domain)
    context_bundle = build_context_bundle_for_run(runner_cfg, run_dir, tasks)
    config = build_tau2_config(mods, runner_cfg)
    tau2_repo_commit = get_git_commit(runner_cfg.tau2_repo)

    # Build metadata through tau2-bench helper, then align the commit hash with the tau2 repo.
    run_info = mods["get_info"](config)
    run_info.git_commit = tau2_repo_commit

    results = Results(info=run_info, tasks=tasks, simulations=[])
    results.save(results_path, format="json")

    if runner_cfg.evolve_components:
        requested_workers = max(
            1, (len(runner_cfg.gpu_ids or [""]) * runner_cfg.proc_per_node)
        )
        if runner_cfg.evolve_parallel_mode == "strict" and requested_workers > 1:
            logger.warning(
                "Evolve mode runs sample-by-sample to preserve state updates; "
                "requested %s workers (%s GPUs x %s proc_per_node), using a single worker on GPU %s.",
                requested_workers,
                len(runner_cfg.gpu_ids or [""]),
                runner_cfg.proc_per_node,
                (runner_cfg.gpu_ids[0] if runner_cfg.gpu_ids else "cpu"),
            )
        return run_self_evolving_benchmark(
            runner_cfg=runner_cfg,
            mods=mods,
            run_dir=run_dir,
            tasks=tasks,
            context_bundle=context_bundle,
            run_info=run_info,
            tau2_repo_commit=tau2_repo_commit,
        )

    total_trials = runner_cfg.pass_hat_k
    seeds = _job_seed_list(runner_cfg.seed, total_trials)
    jobs: list[Job] = []
    for trial, trial_seed in enumerate(seeds):
        for task in tasks:
            jobs.append(
                Job(
                    job_id=f"{task.id}-t{trial}-s{trial_seed}",
                    task_data=task.model_dump(mode="json"),
                    trial=trial,
                    seed=trial_seed,
                )
            )

    total_samples = len(jobs)
    success_count = 0
    failure_count = 0

    mp_ctx = get_context("spawn")
    task_queue = mp_ctx.Queue(maxsize=max(1, len(jobs)))
    ready_queue = mp_ctx.Queue()
    result_queue = mp_ctx.Queue()

    worker_specs: list[WorkerSpec] = []
    gpu_ids = runner_cfg.gpu_ids or [""]
    total_workers = max(1, len(gpu_ids) * runner_cfg.proc_per_node)
    for worker_id in range(total_workers):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        worker_specs.append(
            WorkerSpec(
                worker_id=worker_id,
                cuda_visible_devices=gpu_id,
                llm_backend=runner_cfg.llm_backend,
                llm_model_name=runner_cfg.llm_model_name,
                llm_model_path=runner_cfg.llm_model_path,
                llm_base_url=runner_cfg.llm_base_url,
                llm_api_key=runner_cfg.llm_api_key,
                tau2_repo=runner_cfg.tau2_repo,
                run_dir=str(run_dir),
                domain=runner_cfg.domain,
                mode=runner_cfg.mode,
                sop_text=runner_cfg.sop_text,
                num_cycles=runner_cfg.num_cycles,
                system_prompt_prefix=context_bundle.current_system_prompt,
                pass_hat_k=runner_cfg.pass_hat_k,
                seed=runner_cfg.seed,
                max_steps=runner_cfg.max_steps,
                max_errors=runner_cfg.max_errors,
                max_retries=runner_cfg.max_retries,
                retry_delay=runner_cfg.retry_delay,
                sample_timeout_seconds=runner_cfg.sample_timeout_seconds,
                max_new_tokens=runner_cfg.max_new_tokens,
                repair_max_new_tokens=runner_cfg.repair_max_new_tokens,
                result_batch_size=runner_cfg.result_batch_size,
                result_save_every=runner_cfg.result_save_every,
                tau2_log_level=runner_cfg.tau2_log_level,
            )
        )

    workers = [
        mp_ctx.Process(
            target=worker_main,
            args=(spec, task_queue, ready_queue, result_queue),
            daemon=False,
        )
        for spec in worker_specs
    ]

    for worker in workers:
        worker.start()

    ready = 0
    while ready < len(workers):
        packet = ready_queue.get()
        if packet.get("status") != "ready":
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            raise RuntimeError(f"Worker failed to start: {packet}")
        ready += 1

    for job in jobs:
        task_queue.put(job)
    for _ in workers:
        task_queue.put(None)

    try:
        from tqdm import tqdm

        progress = tqdm(total=total_samples, desc=f"tau2-bench/{runner_cfg.domain}")
        completed = 0
        save_every = max(1, runner_cfg.result_save_every)
        while completed < total_samples:
            packet = result_queue.get()
            simulations_data = packet.get("simulations")
            if simulations_data is None:
                simulations_data = [packet["simulation"]]

            batch_completed = 0
            for simulation_data in simulations_data:
                simulation = mods["SimulationRun"].model_validate(simulation_data)
                results.simulations.append(simulation)
                reward = (
                    getattr(simulation.reward_info, "reward", None)
                    if simulation.reward_info
                    else None
                )
                if is_successful_reward(reward):
                    success_count += 1
                else:
                    failure_count += 1
                completed += 1
                batch_completed += 1

            if completed % save_every == 0 or completed >= total_samples:
                results.save(results_path, format="json")

            progress.update(batch_completed)
            progress.set_postfix(success=success_count, failure=failure_count)
        progress.close()
    finally:
        for worker in workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                worker.terminate()

    summary = compute_summary(
        mods,
        results,
        runner_cfg,
        start_time=bench_start,
        total_samples=total_samples,
        success_count=success_count,
        failure_count=failure_count,
    )
    summary["results_path"] = str(results_path)
    summary["run_dir"] = str(run_dir)
    summary["tau2_repo_commit"] = tau2_repo_commit
    summary["runner_commit"] = get_git_commit(str(PROJECT_ROOT))
    summary["timestamp"] = datetime.now().isoformat()
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    summary = run_benchmark()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
