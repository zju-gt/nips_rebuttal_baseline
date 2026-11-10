from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


@dataclass(frozen=True)
class TransformersBackendRuntime:
    model_name: str
    model_path: str
    device_id: str
    repair_max_new_tokens: int
    backend: str = "transformers"

    def install(self) -> None:
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
