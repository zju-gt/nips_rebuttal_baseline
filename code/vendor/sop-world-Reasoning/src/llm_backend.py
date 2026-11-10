from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_ACTIVE_TRANSFORMERS_BACKEND: Any = None
_ORIGINAL_LITELLM_COMPLETION: Any = None

logger = logging.getLogger(__name__)


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _extract_json_text(text: str) -> str | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        start = stripped.find(start_char)
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except Exception:
            continue
        return json.dumps(obj, ensure_ascii=False)
    return None


def _safe_json_loads(text: str) -> Any:
    json_text = _extract_json_text(text)
    if json_text is None:
        raise ValueError("No JSON object found in extractor output")
    return json.loads(json_text)


def _is_reasoning_unsupported_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc!s} {exc!r}".lower()
    if "reasoning" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "unknown parameter",
            "unknown_parameter",
            "unrecognized request argument",
            "invalidparameter",
        )
    )


def _completion_capability_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str]:
    model = str(kwargs.get("model") or (args[0] if args else "") or "")
    base_url = str(
        kwargs.get("base_url")
        or kwargs.get("api_base")
        or kwargs.get("api_base_url")
        or ""
    )
    return model, base_url


def _without_reasoning(kwargs: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(kwargs)
    stripped.pop("reasoning", None)
    return stripped


def completion_with_reasoning_fallback(
    completion_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Retry once without reasoning when an OpenAI-compatible route rejects it.

    Some proxies accept GPT-5.2 model names but still reject the Responses-style
    reasoning parameter. The downgrade is intentionally limited to the immediate
    retry for this failed call; later calls try reasoning again.
    """
    if "reasoning" not in kwargs:
        return completion_fn(*args, **kwargs)

    try:
        return completion_fn(*args, **kwargs)
    except Exception as exc:
        if not _is_reasoning_unsupported_error(exc):
            raise
        capability_key = _completion_capability_key(args, kwargs)
        logger.warning(
            "LLM backend rejected reasoning for model=%s base_url=%s; "
            "retrying this call once without reasoning.",
            capability_key[0] or "<unknown>",
            capability_key[1] or "<default>",
        )
        return completion_fn(*args, **_without_reasoning(kwargs))


def install_litellm_reasoning_fallback() -> None:
    """Patch litellm completion calls, including tau2 modules that imported it."""
    try:
        import litellm
    except Exception:
        return

    global _ORIGINAL_LITELLM_COMPLETION
    current_completion = getattr(litellm, "completion", None)
    if current_completion is None:
        return

    if getattr(current_completion, "_sop_reasoning_fallback", False):
        patched_completion = current_completion
        original_completion = getattr(
            current_completion,
            "_sop_original_completion",
            current_completion,
        )
    else:
        original_completion = current_completion

        def patched_completion(*args: Any, **kwargs: Any) -> Any:
            return completion_with_reasoning_fallback(
                original_completion,
                *args,
                **kwargs,
            )

        setattr(patched_completion, "_sop_reasoning_fallback", True)
        setattr(patched_completion, "_sop_original_completion", original_completion)
        litellm.completion = patched_completion

    _ORIGINAL_LITELLM_COMPLETION = original_completion

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("tau2."):
            continue
        if getattr(module, "completion", None) is original_completion:
            setattr(module, "completion", patched_completion)


@dataclass(frozen=True)
class TransformersBackendRuntime:
    model_name: str
    model_path: str
    device_id: str
    repair_max_new_tokens: int
    backend: str = "transformers"

    def install(self) -> Any:
        from tau2_runner import LocalTransformersBackend, patch_tau2_generation

        backend = LocalTransformersBackend.load(
            model_name=self.model_name,
            model_path=self.model_path,
            device_id=self.device_id,
            repair_max_new_tokens=self.repair_max_new_tokens,
        )

        def _local_generate(*args: Any, **kwargs: Any) -> Any:
            if args:
                kwargs.setdefault("model", args[0])
            return backend.generate(
                messages=kwargs.pop("messages"),
                tools=kwargs.pop("tools", None),
                tool_choice=kwargs.pop("tool_choice", None),
                call_name=kwargs.pop("call_name", None),
                **kwargs,
            )

        patch_tau2_generation(_local_generate)
        global _ACTIVE_TRANSFORMERS_BACKEND
        _ACTIVE_TRANSFORMERS_BACKEND = backend
        return backend

    def apply_llm_args(self, llm_args: dict[str, Any]) -> dict[str, Any]:
        return dict(llm_args)


@dataclass(frozen=True)
class NativeLitellmVllmBackendRuntime:
    base_url: str
    api_key: str | None
    reasoning_effort: str = ""
    backend: str = "litellm_vllm"

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("LLM_BASE_URL is required for litellm_vllm backend")

    def install(self) -> None:
        install_litellm_reasoning_fallback()
        return None

    def apply_llm_args(self, llm_args: dict[str, Any]) -> dict[str, Any]:
        adapted = dict(llm_args)
        adapted["base_url"] = self.base_url
        if self.api_key and self.api_key.strip():
            adapted["api_key"] = self.api_key
        if self.reasoning_effort.strip():
            adapted["reasoning"] = {"effort": self.reasoning_effort}
        return adapted


def build_backend_runtime(
    config: Any,
    *,
    device_id: str = "",
) -> TransformersBackendRuntime | NativeLitellmVllmBackendRuntime:
    backend = str(_get_config_value(config, "llm_backend", "transformers")).strip()
    if backend == "transformers":
        return TransformersBackendRuntime(
            model_name=str(_get_config_value(config, "llm_model_name", "")),
            model_path=str(_get_config_value(config, "llm_model_path", "")),
            device_id=device_id,
            repair_max_new_tokens=int(_get_config_value(config, "repair_max_new_tokens", 256)),
        )
    if backend == "litellm_vllm":
        return NativeLitellmVllmBackendRuntime(
            base_url=str(_get_config_value(config, "llm_base_url", "")),
            api_key=_get_config_value(config, "llm_api_key", None),
            reasoning_effort=str(
                _get_config_value(config, "llm_reasoning_effort", "")
            ).strip(),
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


def _completion_reasoning_kwargs(
    runtime: NativeLitellmVllmBackendRuntime,
) -> dict[str, Any]:
    effort = str(runtime.reasoning_effort or "").strip()
    if not effort:
        return {}
    return {"reasoning": {"effort": effort}}


def apply_backend_llm_args(
    config: Any,
    llm_args: dict[str, Any],
    *,
    device_id: str = "",
) -> dict[str, Any]:
    runtime = build_backend_runtime(config, device_id=device_id)
    return runtime.apply_llm_args(llm_args)


def normalize_model_name_for_backend(config: Any, model_name: str) -> str:
    backend = str(_get_config_value(config, "llm_backend", "transformers")).strip()
    if backend != "litellm_vllm":
        return model_name
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def install_backend_runtime(
    config: Any,
    *,
    device_id: str = "",
) -> TransformersBackendRuntime | NativeLitellmVllmBackendRuntime:
    runtime = build_backend_runtime(config, device_id=device_id)
    runtime.install()
    return runtime


def _normalize_structured_generation_result(
    *,
    payload: Any | None,
    raw_text: str,
    repaired_text: str = "",
    used_repair: bool = False,
    failure_stage: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "payload": payload,
        "raw_text": str(raw_text or "").strip(),
        "repaired_text": str(repaired_text or "").strip(),
        "used_repair": bool(used_repair),
        "failure_stage": str(failure_stage or "").strip(),
        "failure_reason": str(failure_reason or "").strip(),
    }


def generate_backend_text(
    config: Any,
    *,
    messages: list[dict[str, str]],
    call_name: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device_id: str = "",
) -> str:
    runtime = build_backend_runtime(config, device_id=device_id)
    if isinstance(runtime, TransformersBackendRuntime):
        from tau2_runner import LocalTransformersBackend

        global _ACTIVE_TRANSFORMERS_BACKEND
        backend = _ACTIVE_TRANSFORMERS_BACKEND
        if backend is None:
            backend = LocalTransformersBackend.load(
                model_name=runtime.model_name,
                model_path=runtime.model_path,
                device_id=runtime.device_id,
                repair_max_new_tokens=runtime.repair_max_new_tokens,
            )
            _ACTIVE_TRANSFORMERS_BACKEND = backend
        return backend.generate_raw_text(
            messages=messages,
            call_name=call_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

    from litellm import completion

    response = completion_with_reasoning_fallback(
        completion,
        model=normalize_model_name_for_backend(
            config,
            str(_get_config_value(config, "llm_model_name", "")),
        ),
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        base_url=runtime.base_url,
        api_key=runtime.api_key,
        seed=seed,
        **_completion_reasoning_kwargs(runtime),
    )
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "") or "").strip()


def generate_backend_json_object(
    config: Any,
    *,
    messages: list[dict[str, str]],
    call_name: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device_id: str = "",
) -> dict[str, Any]:
    runtime = build_backend_runtime(config, device_id=device_id)
    if isinstance(runtime, TransformersBackendRuntime):
        from tau2_runner import LocalTransformersBackend

        global _ACTIVE_TRANSFORMERS_BACKEND
        backend = _ACTIVE_TRANSFORMERS_BACKEND
        if backend is None:
            backend = LocalTransformersBackend.load(
                model_name=runtime.model_name,
                model_path=runtime.model_path,
                device_id=runtime.device_id,
                repair_max_new_tokens=runtime.repair_max_new_tokens,
            )
            _ACTIVE_TRANSFORMERS_BACKEND = backend
        return backend.generate_json_object(
            messages=messages,
            call_name=call_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

    from litellm import completion

    raw_text = ""
    repaired_text = ""
    try:
        response = completion_with_reasoning_fallback(
            completion,
            model=normalize_model_name_for_backend(
                config,
                str(_get_config_value(config, "llm_model_name", "")),
            ),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            base_url=runtime.base_url,
            api_key=runtime.api_key,
            seed=seed,
            response_format={"type": "json_object"},
            **_completion_reasoning_kwargs(runtime),
        )
    except Exception as exc:
        return _normalize_structured_generation_result(
            payload=None,
            raw_text=raw_text,
            repaired_text=repaired_text,
            failure_stage="completion",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    try:
        raw_text = str(response.choices[0].message.content or "").strip()
    except Exception:
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            return _normalize_structured_generation_result(
                payload=None,
                raw_text=raw_text,
                repaired_text=repaired_text,
                failure_stage="completion",
                failure_reason="No choices returned by completion API",
            )
        message = choices[0].get("message", {})
        raw_text = str(message.get("content", "") or "").strip()

    try:
        payload = _safe_json_loads(raw_text)
    except Exception as exc:
        return _normalize_structured_generation_result(
            payload=None,
            raw_text=raw_text,
            repaired_text=repaired_text,
            failure_stage="raw_parse",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    return _normalize_structured_generation_result(
        payload=payload,
        raw_text=raw_text,
    )
