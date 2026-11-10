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
    install_backend_runtime,
    normalize_model_name_for_backend,
)
from memory import apply_memory_update, seed_memory_snapshot, summarize_memory_snapshot
from rewards import build_reward_signal, render_trajectory_text
from state_graph import apply_state_graph_update, seed_state_graph, summarize_state_graph

THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


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
    "NL_JUDGE_MODEL": "gpt-5.1-chat-latest",
    "NL_JUDGE_BASE_URL": "https://routify.alibaba-inc.com/protocol/openai/v1",
    "EVOLVE_COMPONENTS": "",
    "EVOLVE_PARALLEL_MODE": "strict",
    "EVOLVE_BATCH_SIZE": 1,
    "EVOLVE_SYNC_ORDER": "task_index",
    "TAU2_LOG_LEVEL": "CRITICAL",
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
    from loguru import logger

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
) -> None:
    original_get_env_constructor = registry.get_env_constructor

    def _patched_get_env_constructor(name: str) -> Any:
        env_constructor = original_get_env_constructor(name)
        if name != domain:
            return env_constructor

        def _wrapped_env_constructor(*args: Any, **kwargs: Any) -> Any:
            environment = env_constructor(*args, **kwargs)
            original_get_policy = environment.get_policy

            def _patched_get_policy() -> str:
                extra_policy = str(extra_policy_provider() or "").strip()
                base_policy = str(original_get_policy()).strip()
                return "\n\n".join(
                    [part for part in [extra_policy, base_policy] if part]
                )

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
    evolve_components: list[str],
) -> str:
    parts: list[str] = []
    if base_prompt.strip():
        parts.append(base_prompt.strip())
    if "memory" in evolve_components:
        parts.append(summarize_memory_snapshot(memory_snapshot))
    if "state_graph" in evolve_components:
        parts.append(summarize_state_graph(state_graph_snapshot))
    return "\n\n".join(part for part in parts if str(part).strip())


def _write_json_file(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _strip_private_think(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


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
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    if isinstance(message, UserMessage):
        content = _strip_private_think(message.content or "")
        if message.tool_calls:
            tool_data = {
                "tool_calls": [tc.model_dump(mode="json") for tc in message.tool_calls]
            }
            return f"role=user\ncontent={content}\n{json.dumps(tool_data, ensure_ascii=False)}"
        return f"role=user\ncontent={content}"

    if isinstance(message, AssistantMessage):
        content = _strip_private_think(message.content or "")
        if message.tool_calls:
            tool_data = {
                "tool_calls": [tc.model_dump(mode="json") for tc in message.tool_calls]
            }
            return f"role=assistant\ncontent={content}\n{json.dumps(tool_data, ensure_ascii=False)}"
        return f"role=assistant\ncontent={content}"

    if isinstance(message, ToolMessage):
        return (
            f"role=tool\nrequestor={message.requestor}\n"
            f"tool_call_id={message.id}\ncontent={message.content or ''}"
        )

    return f"role={getattr(message, 'role', 'unknown')}\ncontent={getattr(message, 'content', '') or ''}"


def _build_prompt(
    messages: list[Any],
    tools: Optional[list[Any]],
    call_name: Optional[str],
    tool_choice: Optional[str],
    enable_private_think: bool = False,
) -> tuple[list[dict[str, str]], str]:
    from tau2.data_model.message import SystemMessage

    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    system_text = "\n\n".join(
        [m.content or "" for m in system_messages if (m.content or "").strip()]
    ).strip()

    helper_lines = [
        "<runner_instructions>",
        "Return exactly one JSON object.",
    ]
    if enable_private_think:
        helper_lines.append(
            "Use ReAct-style private reasoning in one <think>...</think> block before the JSON object."
        )
        helper_lines.append("Do not include <think> content inside the JSON payload.")
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


class LocalTransformersBackend:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device_id: str,
        repair_max_new_tokens: int = 256,
        enable_private_think: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device_id = device_id
        self.repair_max_new_tokens = repair_max_new_tokens
        self.enable_private_think = enable_private_think
        self.tokenizer = None
        self.model = None

    @classmethod
    def load(
        cls,
        model_name: str,
        model_path: str,
        device_id: str,
        repair_max_new_tokens: int = 256,
        enable_private_think: bool = False,
    ) -> "LocalTransformersBackend":
        backend = cls(
            model_name=model_name,
            model_path=model_path,
            device_id=device_id,
            repair_max_new_tokens=repair_max_new_tokens,
            enable_private_think=enable_private_think,
        )
        backend._load_model()
        return backend

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        except TypeError:
            # Older transformers releases still expect torch_dtype.
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        if torch.cuda.is_available():
            model = model.to("cuda:0")

        self._normalize_generation_config(model)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

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
                from tau2.data_model.message import ToolCall as ImportedToolCall

                tool_call_cls = ImportedToolCall
            return tool_call_cls

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
                        _tool_call_class()(
                            id=item.get("id", f"call_{idx}"),
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
                    _tool_call_class()(
                        id=payload.get("id", "call_0"),
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
                    _tool_call_class()(
                        id=item.get("id", f"call_{idx}"),
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

    @staticmethod
    def _fallback_content() -> str:
        return "Please continue."

    def _coerce_message_payload(
        self,
        content: Optional[str],
        tool_calls: Optional[list[Any]],
        generated_text: str,
    ) -> tuple[Optional[str], Optional[list[Any]]]:
        normalized_text = _strip_private_think(generated_text.strip())
        normalized_content = content
        if normalized_content is not None:
            normalized_content = _strip_private_think(normalized_content)
        normalized_tool_calls = tool_calls

        if normalized_tool_calls is not None and not normalized_tool_calls:
            normalized_tool_calls = None

        if normalized_tool_calls is None and (
            normalized_content is None or not normalized_content.strip()
        ):
            normalized_content = normalized_text or self._fallback_content()

        return normalized_content, normalized_tool_calls

    def generate(
        self,
        *,
        messages: list[Any],
        tools: Optional[list[Any]] = None,
        tool_choice: Optional[str] = None,
        call_name: Optional[str] = None,
        **llm_args: Any,
    ) -> Any:
        from tau2.data_model.message import AssistantMessage

        family = self._call_family(call_name)
        chat_messages, prompt_text = _build_prompt(
            messages,
            tools,
            call_name,
            tool_choice,
            enable_private_think=self.enable_private_think,
        )
        prompt_text, inputs = self._build_input_tensors(chat_messages)
        start = time.perf_counter()
        generated_text = self._generate_text(prompt_text, inputs, llm_args)
        generation_time_seconds = time.perf_counter() - start
        content, tool_calls = self._parse_message_payload(generated_text, tools, tool_choice)

        if content is None and tool_calls is None:
            repaired_text = self._repair_output(generated_text, chat_messages, tools, tool_choice, llm_args)
            content, tool_calls = self._parse_message_payload(repaired_text, tools, tool_choice)
            generated_text = repaired_text
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

        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=0.0,
            usage=usage,
            raw_data={
                "model_name": self.model_name,
                "model_path": self.model_path,
                "call_name": call_name,
                "prompt": prompt_text,
                "response_text": generated_text,
                "generation_time_seconds": generation_time_seconds,
            },
            generation_time_seconds=generation_time_seconds,
        )


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
            (
                "termination_timeout",
                metrics_dict.get("termination_timeout", termination_counts.get("timeout", 0)),
            ),
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
            if prompt_prefix:
                _patch_domain_policy_prefix(
                    registry,
                    worker_spec.domain,
                    lambda fixed=prompt_prefix: fixed,
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
    current_meta_sop_text = context_bundle.meta_sop_text
    current_prompt_context = {
        "text": _build_evolution_context_text(
            base_prompt=context_bundle.current_system_prompt,
            memory_snapshot=current_memory,
            state_graph_snapshot=current_graph,
            evolve_components=runner_cfg.evolve_components,
        )
    }
    if not use_worker_pool and current_prompt_context["text"].strip():
        _patch_domain_policy_prefix(
            registry,
            runner_cfg.domain,
            lambda: current_prompt_context["text"],
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
            cycle_events_path = cycle_dir / "events.jsonl"
            cycle_meta_sop_path = cycle_dir / "meta_sop.md"
            cycle_results_path = cycle_dir / "results.json"
            cycle_summary_path = cycle_dir / "summary.json"

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
                )
            else:
                worker_spec_for_cycle = None

            def _record_sample(plan: EvolveSamplePlan, simulation: Any) -> None:
                nonlocal current_memory
                nonlocal current_graph
                nonlocal cycle_success_count
                nonlocal cycle_failure_count

                sample_task = mods["Task"].model_validate(plan.job.task_data)
                reward_value = _simulation_reward_value(simulation)
                termination_reason = _simulation_termination_reason(simulation)
                trajectory_text = render_trajectory_text(_simulation_messages(simulation))
                task_family = _safe_task_family(runner_cfg.domain, sample_task)
                reward_signal = build_reward_signal(
                    sample_id=plan.job.job_id,
                    trajectory_text=trajectory_text,
                    reward_value=reward_value,
                    termination_reason=termination_reason,
                    meta_sop_text=current_meta_sop_text,
                    task_family=task_family,
                    task_id=sample_task.id,
                )

                if "memory" in runner_cfg.evolve_components:
                    current_memory = apply_memory_update(
                        current_memory,
                        family=task_family,
                        posterior_text=reward_signal["posterior_text"],
                        outcome_tag=reward_signal["outcome_tag"],
                        source_cycle=cycle,
                        source_task=plan.job.job_id,
                        reward_value=reward_signal["reward_value"],
                        meta_sop_text=current_meta_sop_text,
                        task_id=sample_task.id,
                    )
                if "state_graph" in runner_cfg.evolve_components:
                    current_graph = apply_state_graph_update(
                        current_graph,
                        family=task_family,
                        posterior_text=reward_signal["posterior_text"],
                        outcome_tag=reward_signal["outcome_tag"],
                        source_cycle=cycle,
                        source_task=plan.job.job_id,
                        reward_value=reward_signal["reward_value"],
                        meta_sop_text=current_meta_sop_text,
                    )

                event = record_evolution_event(
                    sample_id=plan.job.job_id,
                    task_id=sample_task.id,
                    cycle_id=cycle,
                    outcome_tag=reward_signal["outcome_tag"],
                    reward_value=reward_signal["reward_value"],
                    posterior_text=reward_signal["posterior_text"],
                    task_family=task_family,
                    memory_delta=(current_memory.get("entries") or [])[-1]
                    if current_memory.get("entries")
                    else None,
                    graph_delta=(current_graph.get("nodes") or [])[-1]
                    if current_graph.get("nodes")
                    else None,
                )
                cycle_events.append(event)
                all_simulations.append(simulation)
                cycle_results.simulations.append(simulation)

                if is_successful_reward(reward_signal["reward_value"]):
                    cycle_success_count += 1
                else:
                    cycle_failure_count += 1

                current_prompt_context["text"] = _build_evolution_context_text(
                    base_prompt=context_bundle.current_system_prompt,
                    memory_snapshot=current_memory,
                    state_graph_snapshot=current_graph,
                    evolve_components=runner_cfg.evolve_components,
                )

                cycle_results.save(cycle_results_path, format="json")
                _write_jsonl_file(cycle_events_path, cycle_events)
                _write_json_file(cycle_memory_path, current_memory)
                _write_json_file(cycle_graph_path, current_graph)
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
                    if use_batch_sync and prompt_prefix.strip():
                        original_get_env_constructor = registry.get_env_constructor
                        _patch_domain_policy_prefix(
                            registry,
                            runner_cfg.domain,
                            lambda fixed=prompt_prefix: fixed,
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
                    if use_batch_sync and prompt_prefix.strip():
                        registry.get_env_constructor = original_get_env_constructor

            if not use_batch_sync and not use_async_parallel:
                for plan in cycle_plans:
                    simulation = _run_local_plan(plan, "")
                    _record_sample(plan, simulation)
            elif use_batch_sync:
                batch_size = max(1, runner_cfg.evolve_batch_size)
                for start_idx in range(0, len(cycle_plans), batch_size):
                    batch_plans = cycle_plans[start_idx : start_idx + batch_size]
                    batch_prompt_snapshot = current_prompt_context["text"]
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
                            worker_pool_task_queue.put(
                                Job(
                                    job_id=plan.job.job_id,
                                    task_data=plan.job.task_data,
                                    trial=plan.job.trial,
                                    seed=plan.job.seed,
                                    system_prompt_prefix=batch_prompt_snapshot,
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
                            simulation = _run_local_plan(plan, batch_prompt_snapshot)
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
                        worker_pool_task_queue.put(
                            Job(
                                job_id=plan.job.job_id,
                                task_data=plan.job.task_data,
                                trial=plan.job.trial,
                                seed=plan.job.seed,
                                system_prompt_prefix=current_prompt_context["text"],
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

    ensure_tau2_repo_on_path(tau2_repo)
    import tau2.config as tau2_config

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
        seed=tau2_config.DEFAULT_SEED,
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
