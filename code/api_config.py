from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qsl, urlparse


@dataclass(frozen=True, repr=False)
class ApiConfig:
    model: str
    base_url: str
    api_key: str

    def public_dict(self) -> dict[str, str]:
        return {"model": self.model, "base_url": self.base_url}

    def __repr__(self) -> str:
        return (
            "ApiConfig("
            f"model={self.model!r}, base_url={self.base_url!r}, api_key='<redacted>')"
        )


def _literal_string(node: ast.AST | None, field: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError(f"{field} must be a string literal")
    value = node.value.strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _single(values: list[str], field: str) -> str:
    unique = list(dict.fromkeys(values))
    if len(unique) != 1:
        raise ValueError(f"expected exactly one literal {field}")
    return unique[0]


def load_api_config(
    openai_test_path: Path,
    environ: Mapping[str, str],
) -> ApiConfig:
    source = Path(openai_test_path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(openai_test_path))
    except SyntaxError:
        raise ValueError("could not parse OpenAI configuration") from None

    keys: list[str] = []
    base_urls: list[str] = []
    models: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        if name.endswith("OpenAI"):
            if "api_key" in keywords:
                keys.append(_literal_string(keywords["api_key"], "api_key"))
            if "base_url" in keywords:
                base_urls.append(_literal_string(keywords["base_url"], "base_url"))
        if name.endswith("chat.completions.create") and "model" in keywords:
            models.append(_literal_string(keywords["model"], "model"))

    model = str(environ.get("LLM_MODEL_NAME", "")).strip() or _single(
        models, "model"
    )
    base_url = str(environ.get("LLM_BASE_URL", "")).strip() or _single(
        base_urls, "base_url"
    )
    api_key = str(environ.get("LLM_API_KEY", "")).strip() or _single(
        keys, "api_key"
    )
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    sensitive_query_names = {
        "api_key",
        "apikey",
        "access_token",
        "authorization",
        "key",
        "secret",
        "token",
    }
    query_names = {name.lower() for name, _ in parse_qsl(parsed.query)}
    if parsed.username is not None or parsed.password is not None or (
        query_names & sensitive_query_names
    ):
        raise ValueError("base_url must not contain credentials")

    return ApiConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

