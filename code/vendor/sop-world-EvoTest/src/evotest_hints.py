from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def seed_hint_bank() -> dict[str, Any]:
    return {
        "version": 1,
        "tasks": {},
    }


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clip_words(text: str, limit: int = 24) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _extract_reusable_lesson(text: str) -> str:
    normalized = str(text or "")
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("reusable lesson:"):
            return _clip_words(line.split(":", 1)[1], 24)
    return _clip_words(normalized, 24)


def _push_hint(
    sink: list[dict[str, Any]],
    seen: set[str],
    *,
    text: str,
    source: str,
    score: float,
) -> None:
    normalized = _clip_words(text, 24)
    if not normalized:
        return
    key = normalized.lower()
    if key in seen:
        return
    seen.add(key)
    sink.append(
        {
            "text": normalized,
            "source": source,
            "score": float(score),
        }
    )


def build_task_hints_from_signal(
    reward_signal: dict[str, Any],
    *,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    if not isinstance(reward_signal, dict):
        return []
    outcome_tag = _normalize_text(reward_signal.get("outcome_tag", "")).lower()
    if outcome_tag == "infra_error":
        return []

    packet = reward_signal.get("experience_packet")
    if not isinstance(packet, dict):
        packet = {}

    hints: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in packet.get("memory_items", []) or []:
        if not isinstance(item, dict):
            continue
        recovery_action = _normalize_text(item.get("recovery_action", ""))
        missing_step = _normalize_text(item.get("missing_step", ""))
        experience = _normalize_text(item.get("experience", ""))
        if recovery_action:
            _push_hint(
                hints,
                seen,
                text=recovery_action,
                source="recovery_action",
                score=3.0,
            )
        if missing_step:
            _push_hint(
                hints,
                seen,
                text=f"Do not skip: {missing_step}",
                source="missing_step",
                score=2.5,
            )
        if experience:
            _push_hint(
                hints,
                seen,
                text=experience,
                source="experience",
                score=2.0,
            )

    workflow = packet.get("workflow")
    if isinstance(workflow, dict):
        critical_checks = workflow.get("critical_checks", []) or []
        for check in critical_checks[:2]:
            _push_hint(
                hints,
                seen,
                text=check,
                source="critical_check",
                score=2.5,
            )
        state_sequence = [
            _normalize_text(state)
            for state in workflow.get("state_sequence", []) or []
            if _normalize_text(state)
        ]
        if len(state_sequence) >= 2:
            _push_hint(
                hints,
                seen,
                text=f"Preferred flow: {' -> '.join(state_sequence[:6])}",
                source="workflow_sequence",
                score=1.75 if outcome_tag == "failure" else 2.25,
            )

    posterior_digest = _normalize_text(reward_signal.get("posterior_digest", ""))
    posterior_text = _normalize_text(reward_signal.get("posterior_text", ""))
    lesson_text = posterior_digest or _extract_reusable_lesson(posterior_text)
    if lesson_text:
        _push_hint(
            hints,
            seen,
            text=lesson_text,
            source="posterior",
            score=1.5,
        )

    if outcome_tag == "success":
        hints = [
            {
                **hint,
                "score": float(hint.get("score", 0.0)) + 0.5,
            }
            for hint in hints
        ]

    hints.sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("text", "")))
    return hints[: max(1, int(max_items))]


def update_hint_bank(
    hint_bank: dict[str, Any],
    *,
    task_id: str,
    reward_signal: dict[str, Any],
    max_items: int = 3,
) -> dict[str, Any]:
    updated = deepcopy(hint_bank or seed_hint_bank())
    updated.setdefault("version", 1)
    tasks = dict(updated.get("tasks", {}))

    task_key = _normalize_text(task_id)
    if not task_key:
        updated["tasks"] = tasks
        return updated

    task_state = deepcopy(tasks.get(task_key, {}))
    task_state.setdefault("task_id", task_key)
    task_state.setdefault("task_family", reward_signal.get("task_family"))
    task_state.setdefault("failure_count", 0)
    task_state.setdefault("success_count", 0)
    task_state.setdefault("hints", [])
    task_state["task_family"] = reward_signal.get("task_family") or task_state.get("task_family")
    task_state["last_sample_id"] = reward_signal.get("sample_id")
    task_state["last_outcome_tag"] = reward_signal.get("outcome_tag")

    outcome_tag = _normalize_text(reward_signal.get("outcome_tag", "")).lower()
    if outcome_tag == "success":
        task_state["success_count"] = int(task_state.get("success_count", 0)) + 1
    elif outcome_tag == "failure":
        task_state["failure_count"] = int(task_state.get("failure_count", 0)) + 1

    existing_by_text = {
        _normalize_text(item.get("text", "")).lower(): deepcopy(item)
        for item in task_state.get("hints", []) or []
        if isinstance(item, dict) and _normalize_text(item.get("text", ""))
    }
    for hint in build_task_hints_from_signal(reward_signal, max_items=max_items):
        key = _normalize_text(hint.get("text", "")).lower()
        if not key:
            continue
        current = existing_by_text.get(key)
        if current is None:
            current = {
                "text": hint["text"],
                "source": hint["source"],
                "score": float(hint.get("score", 0.0)),
                "count": 1,
            }
        else:
            current["score"] = max(
                float(current.get("score", 0.0)),
                float(hint.get("score", 0.0)),
            ) + 0.25
            current["count"] = int(current.get("count", 1)) + 1
            current["source"] = hint["source"]
        current["last_outcome_tag"] = outcome_tag
        existing_by_text[key] = current

    merged_hints = list(existing_by_text.values())
    merged_hints.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -int(item.get("count", 0)),
            item.get("text", ""),
        )
    )
    task_state["hints"] = merged_hints[: max(1, int(max_items))]
    tasks[task_key] = task_state
    updated["tasks"] = tasks
    return updated


def render_task_hint_block(
    hint_bank: dict[str, Any],
    *,
    task_id: str,
    max_items: int = 3,
) -> str:
    tasks = (hint_bank or {}).get("tasks", {})
    if not isinstance(tasks, dict):
        return ""
    task_state = tasks.get(task_id)
    if not isinstance(task_state, dict):
        return ""

    hints = [
        item
        for item in task_state.get("hints", []) or []
        if isinstance(item, dict) and _normalize_text(item.get("text", ""))
    ][: max(1, int(max_items))]
    if not hints:
        return ""

    family = _normalize_text(task_state.get("task_family", ""))
    failure_count = int(task_state.get("failure_count", 0))
    success_count = int(task_state.get("success_count", 0))
    lines = ["EvoTest Task Hints (learned from previous episodes of the same task):"]
    if family or failure_count or success_count:
        lines.append(
            f"Task family: {family or 'unknown'} | failures={failure_count} | successes={success_count}"
        )
    for index, hint in enumerate(hints, start=1):
        lines.append(f"{index}. {hint['text']}")
    lines.append("Apply these hints only when they match the current state and tool evidence.")
    return "\n".join(lines)


def build_task_local_prompt_prefix(
    *,
    base_prompt: str,
    hint_bank: dict[str, Any],
    task_id: str,
    max_items: int = 3,
) -> str:
    hint_block = render_task_hint_block(
        hint_bank,
        task_id=task_id,
        max_items=max_items,
    )
    parts = [part for part in [base_prompt.strip(), hint_block.strip()] if part]
    return "\n\n".join(parts)
