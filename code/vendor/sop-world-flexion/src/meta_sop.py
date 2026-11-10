from __future__ import annotations

import re
from typing import Iterable


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clip_words(text: str, limit: int = 12) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _unique_lines(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def build_update_context_text(
    current_text: str,
    *,
    outcome_tag: str,
    family: str,
    posterior_text: str,
) -> str:
    base_text = current_text.strip()
    if outcome_tag == "infra_error":
        return base_text

    lines = [line.rstrip() for line in base_text.splitlines()]
    bullets = [line for line in lines if line.lstrip().startswith("-")]

    if outcome_tag == "success":
        bullets.append(f"- Keep the {family} rule short and repeatable.")
    else:
        bullets.append(f"- For {family}, add the missing step before acting.")

    hint = _clip_words(posterior_text, 10)
    if hint:
        bullets.append(f"- Update from posterior: {hint}")

    merged = _unique_lines([line for line in lines if not line.lstrip().startswith("-")] + bullets)
    return "\n".join(merged).strip()
