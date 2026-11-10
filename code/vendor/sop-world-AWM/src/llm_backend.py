from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ACTIVE_TRANSFORMERS_BACKEND: Any = None


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
    backend: str = "litellm_vllm"

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("LLM_BASE_URL is required for litellm_vllm backend")

    def install(self) -> None:
        return None

    def apply_llm_args(self, llm_args: dict[str, Any]) -> dict[str, Any]:
        adapted = dict(llm_args)
        adapted["base_url"] = self.base_url
        if self.api_key and self.api_key.strip():
            adapted["api_key"] = self.api_key
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
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


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

    response = completion(
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
        response = completion(
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
