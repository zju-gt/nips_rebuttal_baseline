from __future__ import annotations

import copy
import dataclasses
import fcntl
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


@dataclass(frozen=True)
class ShimSettings:
    output_dir: Path
    user_model: str
    seed: int
    usage_log: Path
    rpm_limit: int = 0


_TARGET_MODULES = frozenset({"tau2_runner", "litellm"})
_IMPORT_HOOK_SENTINEL = "_tau2_api_shim_import_hook"


def _settings_from_environment(environ: Mapping[str, str]) -> ShimSettings | None:
    names = (
        "TAU2_ASSIGNED_OUTPUT_DIR",
        "TAU2_USER_MODEL",
        "TAU2_RUN_SEED",
        "TAU2_API_USAGE_LOG",
    )
    present = [name for name in names if environ.get(name)]
    if not present:
        return None
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise RuntimeError(
            "incomplete tau2 runtime shim environment: missing "
            + ", ".join(missing)
        )
    try:
        seed = int(environ["TAU2_RUN_SEED"])
    except ValueError as exc:
        raise ValueError("TAU2_RUN_SEED must be an integer") from exc
    try:
        rpm_limit = int(environ.get("TAU2_API_RPM_LIMIT", "0"))
    except ValueError as exc:
        raise ValueError("TAU2_API_RPM_LIMIT must be an integer") from exc
    if rpm_limit < 0:
        raise ValueError("TAU2_API_RPM_LIMIT must not be negative")
    return ShimSettings(
        output_dir=Path(environ["TAU2_ASSIGNED_OUTPUT_DIR"]).expanduser().resolve(),
        user_model=environ["TAU2_USER_MODEL"],
        seed=seed,
        usage_log=Path(environ["TAU2_API_USAGE_LOG"]).expanduser().resolve(),
        rpm_limit=rpm_limit,
    )


def _replace_seed(config: Any, seed: int) -> Any:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, seed=seed)
    if hasattr(config, "seed"):
        cloned = copy.copy(config)
        try:
            setattr(cloned, "seed", seed)
        except Exception as exc:
            raise TypeError("runner config seed cannot be replaced") from exc
        return cloned
    raise TypeError("runner config does not expose a seed field")


def _normalized_openai_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError("TAU2_USER_MODEL must not be empty")
    if "/" in normalized:
        return normalized
    return f"openai/{normalized}"


def _is_sensitive_metadata_name(name: Any) -> bool:
    normalized = str(name).lower().replace("-", "_")
    return normalized in {
        "api_key",
        "apikey",
        "authorization",
        "proxy_authorization",
        "access_token",
        "secret",
        "token",
    } or normalized.endswith(("_api_key", "_secret", "_access_token"))


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _sanitize_metadata_value(item)
            for key, item in value.items()
            if not _is_sensitive_metadata_name(key)
        }
    if isinstance(value, list):
        return [_sanitize_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_metadata_value(item) for item in value)
    return value


def _sanitize_info_metadata(info: Any) -> Any:
    info_copy = getattr(info, "model_copy", None)
    if not callable(info_copy):
        raise TypeError("tau2 Info must provide model_copy(update=...)")
    updates: dict[str, Any] = {}
    for field_name in ("agent_info", "user_info"):
        component = getattr(info, field_name, None)
        component_copy = getattr(component, "model_copy", None)
        if component is None or not callable(component_copy):
            continue
        llm_args = getattr(component, "llm_args", None)
        updates[field_name] = component_copy(
            update={"llm_args": _sanitize_metadata_value(llm_args)}
        )
    return info_copy(update=updates)


def patch_tau2_runner(module: ModuleType, settings: ShimSettings) -> None:
    if getattr(module, "_tau2_api_shim_patched", False):
        return

    original_load = getattr(module, "load_runner_config")
    original_build = getattr(module, "build_tau2_config")
    original_get_modules = getattr(module, "get_tau2_modules", None)
    original_get_tasks = getattr(module, "get_tasks_for_domain", None)

    def build_run_dir(config: Any) -> Path:
        del config
        return settings.output_dir

    @wraps(original_load)
    def load_runner_config(*args: Any, **kwargs: Any) -> Any:
        return _replace_seed(original_load(*args, **kwargs), settings.seed)

    @wraps(original_build)
    def build_tau2_config(*args: Any, **kwargs: Any) -> Any:
        config = original_build(*args, **kwargs)
        model_copy = getattr(config, "model_copy", None)
        if not callable(model_copy):
            raise TypeError("tau2 TextRunConfig must provide model_copy(update=...)")

        current_args = getattr(config, "llm_args_user", None)
        if current_args is None and hasattr(config, "values"):
            current_args = config.values.get("llm_args_user", {})
        if not isinstance(current_args, dict):
            raise TypeError("tau2 TextRunConfig.llm_args_user must be a mapping")
        user_args = dict(current_args)
        user_args.pop("reasoning", None)
        return model_copy(
            update={
                "llm_user": _normalized_openai_model(settings.user_model),
                "llm_args_user": user_args,
            }
        )

    if callable(original_get_modules):

        @wraps(original_get_modules)
        def get_tau2_modules(*args: Any, **kwargs: Any) -> Any:
            modules = original_get_modules(*args, **kwargs)
            if not isinstance(modules, Mapping):
                raise TypeError("get_tau2_modules() must return a mapping")
            get_info = modules.get("get_info")
            if not callable(get_info):
                return modules

            @wraps(get_info)
            def sanitized_get_info(*info_args: Any, **info_kwargs: Any) -> Any:
                return _sanitize_info_metadata(get_info(*info_args, **info_kwargs))

            patched_modules = dict(modules)
            patched_modules["get_info"] = sanitized_get_info
            return patched_modules

        module.get_tau2_modules = get_tau2_modules

    module.build_run_dir = build_run_dir
    module.load_runner_config = load_runner_config
    module.build_tau2_config = build_tau2_config
    task_ids_file = os.environ.get("TAU2_TASK_IDS_FILE", "")
    if task_ids_file and callable(original_get_tasks):
        @wraps(original_get_tasks)
        def get_tasks_for_domain(mods: Any, domain: str) -> list[Any]:
            selection = json.loads(Path(task_ids_file).read_text(encoding="utf-8"))
            if selection.get("domain") != domain:
                raise ValueError("task selection domain mismatch")
            ids = selection.get("task_ids")
            if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
                raise ValueError("task selection must contain nonempty string task_ids")
            if len(ids) != len(set(ids)):
                raise ValueError("task selection contains duplicate IDs")
            base_tasks = original_get_tasks(mods, domain)
            by_id = {str(task.id): task for task in base_tasks}
            if len(by_id) != len(base_tasks) or not set(ids) <= set(by_id):
                raise ValueError("task selection must be a subset of the base split")
            return [by_id[task_id] for task_id in ids]

        module.get_tasks_for_domain = get_tasks_for_domain
    module._tau2_api_shim_patched = True


def append_usage_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _pace_api_request(settings: ShimSettings) -> None:
    """Space requests from all workers of one job using a shared file lock."""
    if settings.rpm_limit <= 0:
        return
    path = settings.output_dir / ".api_next_request"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.read(fd, 64).decode("ascii").strip()
        next_at = float(raw) if raw else 0.0
        now = time.monotonic()
        slot = max(now, next_at)
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"{slot + 60.0 / settings.rpm_limit:.9f}\n".encode("ascii"))
    finally:
        os.close(fd)
    if slot > now:
        time.sleep(slot - now)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_record(
    *,
    kwargs: Mapping[str, Any],
    started_at: float,
    status: str,
    response: Any = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    usage = _value(response, "usage") if response is not None else None
    prompt_details = _value(usage, "prompt_tokens_details")
    completion_details = _value(usage, "completion_tokens_details")
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": str(kwargs.get("model", "")),
        "call_name": str(kwargs.get("call_name", "") or ""),
        "latency_seconds": max(0.0, time.perf_counter() - started_at),
    }
    if response is not None:
        record.update(
            {
                "response_id": str(_value(response, "id", "") or ""),
                "prompt_tokens": int(_value(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    _value(usage, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(_value(usage, "total_tokens", 0) or 0),
                "cached_tokens": int(
                    _value(prompt_details, "cached_tokens", 0) or 0
                ),
                "reasoning_tokens": int(
                    _value(completion_details, "reasoning_tokens", 0) or 0
                ),
            }
        )
    if error is not None:
        record["error_type"] = type(error).__name__
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            record["status_code"] = status_code
    return record


def patch_litellm(module: ModuleType, settings: ShimSettings) -> None:
    original = getattr(module, "completion")
    if getattr(original, "_tau2_usage_wrapper", False):
        return
    unsupported_reasoning: set[tuple[str, str]] = set()

    @wraps(original)
    def completion(*args: Any, **kwargs: Any) -> Any:
        capability_key = (
            str(kwargs.get("model") or (args[0] if args else "") or ""),
            str(kwargs.get("base_url") or kwargs.get("api_base") or ""),
        )
        call_kwargs = dict(kwargs)
        if capability_key in unsupported_reasoning:
            call_kwargs.pop("reasoning", None)
        for attempt in range(4):
            _pace_api_request(settings)
            started_at = time.perf_counter()
            try:
                response = original(*args, **call_kwargs)
            except Exception as exc:
                append_usage_record(
                    settings.usage_log,
                    _usage_record(
                        kwargs=call_kwargs,
                        started_at=started_at,
                        status="failed",
                        error=exc,
                    ),
                )
                error_text = f"{type(exc).__name__}: {exc!s}".lower()
                if (
                    "reasoning" in call_kwargs
                    and getattr(exc, "status_code", None) == 400
                    and "reasoning" in error_text
                    and any(
                        marker in error_text
                        for marker in (
                            "unknown parameter",
                            "unknown_parameter",
                            "unrecognized request argument",
                            "invalidparameter",
                        )
                    )
                ):
                    unsupported_reasoning.add(capability_key)
                if (
                    settings.rpm_limit > 0
                    and getattr(exc, "status_code", None) == 429
                    and attempt < 3
                ):
                    time.sleep(60)
                    continue
                raise
            append_usage_record(
                settings.usage_log,
                _usage_record(
                    kwargs=call_kwargs,
                    started_at=started_at,
                    status="succeeded",
                    response=response,
                ),
            )
            return response
        raise AssertionError("unreachable rate-limit retry state")

    completion._tau2_usage_wrapper = True
    completion._tau2_original_completion = original
    module.completion = completion


def _patch_module(module: ModuleType, settings: ShimSettings) -> None:
    if module.__name__ == "tau2_runner":
        patch_tau2_runner(module, settings)
    elif module.__name__ == "litellm":
        patch_litellm(module, settings)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(
        self,
        wrapped: importlib.abc.Loader,
        settings: ShimSettings,
    ) -> None:
        self._wrapped = wrapped
        self._settings = settings

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def get_resource_reader(self, fullname: str) -> Any:
        get_reader = getattr(self._wrapped, "get_resource_reader", None)
        return get_reader(fullname) if get_reader is not None else None

    def exec_module(self, module: ModuleType) -> None:
        exec_module = getattr(self._wrapped, "exec_module", None)
        if exec_module is None:
            raise ImportError(f"loader for {module.__name__} cannot execute modules")
        exec_module(module)
        _patch_module(module, self._settings)


class _PatchingFinder(importlib.abc.MetaPathFinder):
    _tau2_api_shim_import_hook = True

    def __init__(self, settings: ShimSettings) -> None:
        self._settings = settings

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del target
        if fullname not in _TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _PatchingLoader):
            return spec
        spec.loader = _PatchingLoader(spec.loader, self._settings)
        return spec


def install_from_environment(environ: Mapping[str, str] | None = None) -> None:
    settings = _settings_from_environment(os.environ if environ is None else environ)
    if settings is None:
        return

    for module_name in _TARGET_MODULES:
        module = sys.modules.get(module_name)
        if isinstance(module, ModuleType):
            _patch_module(module, settings)

    if any(getattr(finder, _IMPORT_HOOK_SENTINEL, False) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _PatchingFinder(settings))

