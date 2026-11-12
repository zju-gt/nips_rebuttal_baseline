from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@dataclass(frozen=True)
class _RunnerConfig:
    seed: int
    domain: str


class _TextRunConfig:
    def __init__(self, **values: object) -> None:
        self.values = dict(values)

    def model_copy(self, *, update: dict[str, object]) -> "_TextRunConfig":
        return _TextRunConfig(**{**self.values, **update})


def test_patch_tau2_runner_redirects_output_seed_and_user_model(
    tmp_path: Path,
) -> None:
    from runtime_shim.tau2_api_shim import ShimSettings, patch_tau2_runner

    module = ModuleType("tau2_runner")
    module.build_run_dir = lambda config: Path("wrong-output")
    module.load_runner_config = lambda: _RunnerConfig(seed=7, domain="retail")
    module.build_tau2_config = lambda mods, runner_cfg: _TextRunConfig(
        llm_agent="openai/gpt-5.2-test",
        llm_user="openai/gpt-5.2-test",
        llm_args_user={
            "base_url": "https://example.invalid/v1",
            "api_key": "secret-value",
            "reasoning": {"effort": "medium"},
            "temperature": 0.2,
        },
    )
    settings = ShimSettings(
        output_dir=tmp_path / "assigned",
        user_model="gpt-5.1-2025-11-13",
        seed=300,
        usage_log=tmp_path / "usage.jsonl",
    )

    patch_tau2_runner(module, settings)

    runner_config = module.load_runner_config()
    tau2_config = module.build_tau2_config({}, runner_config)
    assert module.build_run_dir(runner_config) == tmp_path / "assigned"
    assert runner_config == _RunnerConfig(seed=300, domain="retail")
    assert tau2_config.values["llm_agent"] == "openai/gpt-5.2-test"
    assert tau2_config.values["llm_user"] == "openai/gpt-5.1-2025-11-13"
    assert tau2_config.values["llm_args_user"] == {
        "base_url": "https://example.invalid/v1",
        "api_key": "secret-value",
        "temperature": 0.2,
    }


def test_patch_tau2_runner_sanitizes_native_result_metadata_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime_shim.tau2_api_shim import ShimSettings, patch_tau2_runner

    tau2_src = Path(__file__).resolve().parents[3] / "tau2-bench" / "src"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    sys.path.insert(0, str(tau2_src))
    try:
        from tau2.data_model.simulation import AgentInfo, Info, Results, UserInfo

        secret = "native-result-secret"
        runtime_config = SimpleNamespace(
            llm_args_agent={
                "api_key": secret,
                "base_url": "https://example.invalid/v1",
            },
            llm_args_user={
                "api_key": secret,
                "base_url": "https://example.invalid/v1",
            },
        )
        info = Info.model_construct(
            git_commit="test",
            num_trials=1,
            max_steps=1,
            max_errors=1,
            user_info=UserInfo(
                implementation="user_simulator",
                llm="openai/gpt-5.1-test",
                llm_args=dict(runtime_config.llm_args_user),
            ),
            agent_info=AgentInfo(
                implementation="llm_agent",
                llm="openai/gpt-5.2-test",
                llm_args=dict(runtime_config.llm_args_agent),
            ),
            environment_info=None,
            seed=300,
        )
        module = ModuleType("tau2_runner")
        module.build_run_dir = lambda config: tmp_path
        module.load_runner_config = lambda: _RunnerConfig(seed=7, domain="retail")
        module.build_tau2_config = lambda mods, runner_cfg: _TextRunConfig(
            llm_user="openai/gpt-5.2-test",
            llm_args_user={"api_key": secret},
        )
        module.get_tau2_modules = lambda: {"get_info": lambda config: info}
        settings = ShimSettings(
            output_dir=tmp_path,
            user_model="gpt-5.1-2025-11-13",
            seed=300,
            usage_log=tmp_path / "usage.jsonl",
        )

        patch_tau2_runner(module, settings)
        sanitized_info = module.get_tau2_modules()["get_info"](runtime_config)
        result_path = tmp_path / "results.json"
        Results.model_construct(
            info=sanitized_info,
            tasks=[],
            simulations=[],
            simulation_index=None,
        ).save(result_path, format="json")
        raw = result_path.read_text(encoding="utf-8")

        assert secret not in raw
        assert "api_key" not in raw
        assert runtime_config.llm_args_agent["api_key"] == secret
        assert runtime_config.llm_args_user["api_key"] == secret
    finally:
        sys.path.remove(str(tau2_src))


def test_usage_wrapper_records_tokens_without_request_content(tmp_path: Path) -> None:
    from runtime_shim.tau2_api_shim import ShimSettings, patch_litellm

    response = SimpleNamespace(
        id="response-123",
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        ),
    )
    module = ModuleType("litellm")
    module.completion = lambda *args, **kwargs: response
    usage_path = tmp_path / "usage.jsonl"
    settings = ShimSettings(
        output_dir=tmp_path / "output",
        user_model="gpt-5.1-2025-11-13",
        seed=300,
        usage_log=usage_path,
    )
    patch_litellm(module, settings)

    actual = module.completion(
        model="openai/gpt-5.2-test",
        call_name="posterior_generator",
        messages=[{"role": "user", "content": "private prompt"}],
        tools=[{"private": "tool payload"}],
        api_key="secret-value",
    )

    assert actual is response
    raw = usage_path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert record["status"] == "succeeded"
    assert record["model"] == "openai/gpt-5.2-test"
    assert record["call_name"] == "posterior_generator"
    assert record["response_id"] == "response-123"
    assert record["prompt_tokens"] == 120
    assert record["completion_tokens"] == 30
    assert record["total_tokens"] == 150
    assert record["cached_tokens"] == 20
    assert record["reasoning_tokens"] == 7
    assert record["latency_seconds"] >= 0
    assert "private prompt" not in raw
    assert "tool payload" not in raw
    assert "secret-value" not in raw
    assert "api_key" not in raw
    assert "authorization" not in raw.lower()


def test_usage_wrapper_records_sanitized_failure_and_reraises(
    tmp_path: Path,
) -> None:
    from runtime_shim.tau2_api_shim import ShimSettings, patch_litellm

    class RateLimitFailure(RuntimeError):
        status_code = 429

    failure = RateLimitFailure("private response body secret-value")
    module = ModuleType("litellm")

    def _fail(*args: object, **kwargs: object) -> object:
        raise failure

    module.completion = _fail
    usage_path = tmp_path / "usage.jsonl"
    settings = ShimSettings(
        output_dir=tmp_path / "output",
        user_model="gpt-5.1-2025-11-13",
        seed=300,
        usage_log=usage_path,
    )
    patch_litellm(module, settings)

    with pytest.raises(RateLimitFailure) as exc_info:
        module.completion(
            model="openai/gpt-5.2-test",
            messages=[{"role": "user", "content": "private prompt"}],
            api_key="secret-value",
        )

    assert exc_info.value is failure
    raw = usage_path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert record["status"] == "failed"
    assert record["error_type"] == "RateLimitFailure"
    assert record["status_code"] == 429
    assert "private response body" not in raw
    assert "private prompt" not in raw
    assert "secret-value" not in raw


def test_rate_limited_wrapper_retries_http_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime_shim.tau2_api_shim import ShimSettings, patch_litellm

    class RateLimitFailure(RuntimeError):
        status_code = 429

    attempts = 0
    waits: list[float] = []
    response = SimpleNamespace(id="recovered", usage=None)

    def _complete(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitFailure("RPM exceeded")
        return response

    monkeypatch.setattr("runtime_shim.tau2_api_shim.time.sleep", waits.append)
    module = ModuleType("litellm")
    module.completion = _complete
    settings = ShimSettings(
        output_dir=tmp_path / "output",
        user_model="gpt-5.1-2025-11-13",
        seed=300,
        usage_log=tmp_path / "usage.jsonl",
        rpm_limit=20,
    )
    patch_litellm(module, settings)

    assert module.completion(model="openai/test") is response
    assert attempts == 2
    assert 60 in waits
    assert settings.output_dir.joinpath(".api_next_request").exists()
    records = [json.loads(line) for line in settings.usage_log.read_text().splitlines()]
    assert [record["status"] for record in records] == ["failed", "succeeded"]


def test_usage_wrapper_remembers_unsupported_reasoning(tmp_path: Path) -> None:
    from runtime_shim.tau2_api_shim import ShimSettings, patch_litellm

    class UnsupportedReasoning(RuntimeError):
        status_code = 400

    requests: list[dict[str, object]] = []
    response = SimpleNamespace(id="accepted", usage=None)

    def _complete(*args: object, **kwargs: object) -> object:
        requests.append(dict(kwargs))
        if "reasoning" in kwargs:
            raise UnsupportedReasoning("Unknown parameter: reasoning")
        return response

    module = ModuleType("litellm")
    module.completion = _complete
    settings = ShimSettings(
        output_dir=tmp_path / "output",
        user_model="gpt-5.1-2025-11-13",
        seed=300,
        usage_log=tmp_path / "usage.jsonl",
    )
    patch_litellm(module, settings)
    call = {
        "model": "openai/gpt-5.2",
        "base_url": "https://example.invalid/v1",
        "reasoning": {"effort": "medium"},
    }

    with pytest.raises(UnsupportedReasoning):
        module.completion(**call)
    assert module.completion(**call) is response
    assert "reasoning" in requests[0]
    assert "reasoning" not in requests[1]
    assert "reasoning" in call


def test_sitecustomize_patches_spawned_interpreters(tmp_path: Path) -> None:
    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    (fake_modules / "tau2_runner.py").write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "from pathlib import Path",
                "@dataclass(frozen=True)",
                "class RunnerConfig:",
                "    seed: int",
                "    domain: str",
                "class TextRunConfig:",
                "    def __init__(self, **values):",
                "        self.values = values",
                "        for key, value in values.items(): setattr(self, key, value)",
                "    def model_copy(self, *, update):",
                "        return TextRunConfig(**{**self.values, **update})",
                "def build_run_dir(config): return Path('wrong-output')",
                "def load_runner_config(): return RunnerConfig(seed=7, domain='retail')",
                "def build_tau2_config(mods, runner_cfg):",
                "    return TextRunConfig(llm_user='openai/gpt-5.2-test', llm_args_user={'base_url': 'https://example.invalid/v1', 'api_key': 'secret-value', 'reasoning': {'effort': 'medium'}})",
            ]
        ),
        encoding="utf-8",
    )
    (fake_modules / "litellm.py").write_text(
        "\n".join(
            [
                "from types import SimpleNamespace",
                "def completion(*args, **kwargs):",
                "    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5, prompt_tokens_details=SimpleNamespace(cached_tokens=1), completion_tokens_details=SimpleNamespace(reasoning_tokens=1))",
                "    return SimpleNamespace(id='child-response', usage=usage)",
            ]
        ),
        encoding="utf-8",
    )
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "\n".join(
            [
                "import json",
                "import litellm",
                "import tau2_runner",
                "config = tau2_runner.load_runner_config()",
                "tau2_config = tau2_runner.build_tau2_config({}, config)",
                "response = litellm.completion(model='openai/gpt-5.2-test', call_name='child-call', messages=[{'role': 'user', 'content': 'private'}], api_key='secret-value')",
                "print(json.dumps({'seed': config.seed, 'output': str(tau2_runner.build_run_dir(config)), 'user_model': tau2_config.llm_user, 'has_reasoning': 'reasoning' in tau2_config.llm_args_user, 'response_id': response.id}))",
            ]
        ),
        encoding="utf-8",
    )
    assigned_output = tmp_path / "assigned-output"
    usage_log = tmp_path / "usage.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(CODE_DIR / "runtime_shim"), str(fake_modules)]
            ),
            "TAU2_ASSIGNED_OUTPUT_DIR": str(assigned_output),
            "TAU2_USER_MODEL": "gpt-5.1-2025-11-13",
            "TAU2_RUN_SEED": "300",
            "TAU2_API_USAGE_LOG": str(usage_log),
        }
    )

    first = subprocess.run(
        [sys.executable, str(child_script)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(first.stdout)
    assert payload == {
        "seed": 300,
        "output": str(assigned_output),
        "user_model": "openai/gpt-5.1-2025-11-13",
        "has_reasoning": False,
        "response_id": "child-response",
    }

    children = [
        subprocess.Popen(
            [sys.executable, str(child_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(3)
    ]
    for child in children:
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert json.loads(stdout)["seed"] == 300

    records = [
        json.loads(line)
        for line in usage_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert all(record["response_id"] == "child-response" for record in records)
    assert all(record["total_tokens"] == 5 for record in records)

