from __future__ import annotations

import re
from typing import Any, Optional


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clip_words(text: str, limit: int) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def render_trajectory_text(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or getattr(message, "type", None) or "message"
        content = getattr(message, "content", "")
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        content_text = _normalize_text(str(content))
        if content_text:
            lines.append(f"{role}: {content_text}")
    return "\n".join(lines)


def build_outcome_tag(reward_value: Optional[float], termination_reason: str | None) -> str:
    if reward_value is not None and abs(float(reward_value) - 1.0) <= 1e-6:
        return "success"

    normalized_reason = (termination_reason or "").upper()
    if "INFRA" in normalized_reason:
        return "infra_error"
    return "failure"


def build_posterior_text(
    *,
    outcome_tag: str,
    reward_value: Optional[float],
    meta_sop_text: str,
    trajectory_text: str,
    task_family: str | None = None,
) -> str:
    family_text = task_family or "task"
    meta_hint = _clip_words(meta_sop_text, 8)
    if outcome_tag == "success":
        text = f"Reinforce the short {family_text} rule."
    elif outcome_tag == "infra_error":
        text = "Treat this as infrastructure noise; keep task logic unchanged."
    else:
        text = f"Add the missing {family_text} step and keep it short."
    if meta_hint:
        text = f"{text} Follow: {meta_hint}"
    return _clip_words(text, 40)


def build_reward_signal(
    *,
    sample_id: str,
    trajectory_text: str,
    reward_value: Optional[float],
    termination_reason: str | None,
    meta_sop_text: str,
    task_family: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    outcome_tag = build_outcome_tag(reward_value, termination_reason)
    posterior_text = build_posterior_text(
        outcome_tag=outcome_tag,
        reward_value=reward_value,
        meta_sop_text=meta_sop_text,
        trajectory_text=trajectory_text,
        task_family=task_family,
    )
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "task_family": task_family,
        "reward_value": reward_value,
        "termination_reason": termination_reason,
        "outcome_tag": outcome_tag,
        "posterior_text": posterior_text,
        "trajectory_text": _clip_words(trajectory_text, 80),
    }

