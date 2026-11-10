from __future__ import annotations

import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _write_api_file(
    path: Path,
    *,
    key_expr: str = '"secret-value"',
    base_url: str = "https://example.invalid/v1",
) -> Path:
    sentinel = path.with_name("executed.txt")
    path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from openai import OpenAI",
                f'Path(r"{sentinel}").write_text("executed")',
                "client = OpenAI(",
                f"    api_key={key_expr},",
                f"    base_url={base_url!r},",
                ")",
                "completion = client.chat.completions.create(",
                '    model="gpt-5.2-test",',
                '    messages=[{"role": "user", "content": "hello"}],',
                ")",
            ]
        ),
        encoding="utf-8",
    )
    return sentinel


def test_loads_literal_config_without_executing_source(tmp_path: Path) -> None:
    from api_config import load_api_config

    source = tmp_path / "openai_test.py"
    sentinel = _write_api_file(source)

    config = load_api_config(source, {})

    assert config.model == "gpt-5.2-test"
    assert config.base_url == "https://example.invalid/v1"
    assert config.api_key == "secret-value"
    assert not sentinel.exists()


def test_environment_overrides_take_precedence(tmp_path: Path) -> None:
    from api_config import load_api_config

    source = tmp_path / "openai_test.py"
    _write_api_file(source)

    config = load_api_config(
        source,
        {
            "LLM_MODEL_NAME": "env-model",
            "LLM_BASE_URL": "https://env.example/v1",
            "LLM_API_KEY": "env-secret",
        },
    )

    assert config.model == "env-model"
    assert config.base_url == "https://env.example/v1"
    assert config.api_key == "env-secret"


def test_partial_environment_override_keeps_file_values(tmp_path: Path) -> None:
    from api_config import load_api_config

    source = tmp_path / "openai_test.py"
    _write_api_file(source)

    config = load_api_config(source, {"LLM_MODEL_NAME": "env-model"})

    assert config.model == "env-model"
    assert config.base_url == "https://example.invalid/v1"
    assert config.api_key == "secret-value"


def test_rejects_dynamic_key_expression_without_exposing_source_value(
    tmp_path: Path,
) -> None:
    from api_config import load_api_config

    source = tmp_path / "openai_test.py"
    _write_api_file(source, key_expr='get_key("do-not-print-this")')

    with pytest.raises(ValueError) as exc_info:
        load_api_config(source, {})

    assert "string literal" in str(exc_info.value)
    assert "do-not-print-this" not in str(exc_info.value)


def test_public_serialization_and_repr_redact_api_key(tmp_path: Path) -> None:
    from api_config import load_api_config

    source = tmp_path / "openai_test.py"
    _write_api_file(source)

    config = load_api_config(source, {})

    assert config.public_dict() == {
        "model": "gpt-5.2-test",
        "base_url": "https://example.invalid/v1",
    }
    assert "api_key" not in config.public_dict()
    assert "secret-value" not in repr(config)


def test_rejects_syntax_errors_without_echoing_source_credentials(
    tmp_path: Path,
) -> None:
    from api_config import load_api_config

    secret = "syntax-error-secret"
    source = tmp_path / "openai_test.py"
    source.write_text(
        f"client = OpenAI(api_key={secret!r} base_url='https://example.invalid/v1')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_api_config(source, {})

    assert secret not in str(exc_info.value)
    assert "parse" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:password@example.invalid/v1",
        "https://example.invalid/v1?api_key=query-secret",
    ],
)
def test_rejects_endpoint_urls_containing_credentials(
    tmp_path: Path,
    base_url: str,
) -> None:
    from api_config import load_api_config

    source = tmp_path / "openai_test.py"
    _write_api_file(source, base_url=base_url)

    with pytest.raises(ValueError) as exc_info:
        load_api_config(source, {})

    assert "password" not in str(exc_info.value)
    assert "query-secret" not in str(exc_info.value)
    assert "credential" in str(exc_info.value).lower()

