from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clip_words(text: str, limit: int = 24) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def build_reasoning_retrieval_text(entry: dict[str, Any]) -> str:
    title = _normalize_text(entry.get("title", ""))
    applicability = _normalize_text(entry.get("applicability", ""))
    lesson = _normalize_text(entry.get("lesson", ""))
    family = _normalize_text(entry.get("family", ""))
    outcome_type = _normalize_text(entry.get("outcome_type", ""))
    sections: list[str] = []
    if title:
        sections.append(f"Title: {title}")
    if applicability:
        sections.append(f"Applicability: {applicability}")
    if lesson:
        sections.append(f"Lesson: {lesson}")
    if family:
        sections.append(f"Family: {family}")
    if outcome_type:
        sections.append(f"Outcome: {outcome_type}")
    return "\n".join(sections).strip()


def build_reasoning_entry(
    *,
    title: str,
    lesson: str,
    applicability: str,
    family: str,
    outcome_type: str,
) -> dict[str, str]:
    normalized_title = _normalize_text(title)
    normalized_lesson = _normalize_text(lesson)
    normalized_family = _normalize_text(family) or "default"
    normalized_outcome = _normalize_text(outcome_type).lower() or "failure"
    normalized_applicability = _normalize_text(applicability)
    if not normalized_applicability:
        normalized_applicability = f"when handling similar {normalized_family} cases"
    return {
        "title": normalized_title,
        "lesson": normalized_lesson,
        "applicability": normalized_applicability,
        "family": normalized_family,
        "outcome_type": normalized_outcome,
    }


def _reasoning_semantic_hash(entry: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        (
            f"{_normalize_text(entry.get('family', ''))}:"
            f"{_normalize_text(entry.get('title', ''))}:"
            f"{_normalize_text(entry.get('applicability', ''))}:"
            f"{_normalize_text(entry.get('lesson', ''))}"
        ).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _ensure_reasoning_bank_shape(
    bank: dict[str, Any] | None,
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    normalized = deepcopy(bank or {})
    normalized.setdefault("version", 1)
    if domain is not None:
        normalized.setdefault("domain", domain)
    normalized.setdefault("families", [])
    normalized.setdefault("entries", [])
    normalized.setdefault("family_index", {})
    normalized.setdefault("events", [])
    return normalized


def seed_reasoning_bank(domain: str) -> dict[str, Any]:
    return _ensure_reasoning_bank_shape({}, domain=domain)


def _reasoning_from_memory_item(
    item: dict[str, Any],
    *,
    fallback_family: str,
    outcome_type: str,
) -> dict[str, str] | None:
    title = _normalize_text(item.get("title", ""))
    lesson = _normalize_text(item.get("experience", ""))
    family = _normalize_text(item.get("family", "")) or fallback_family
    applicability = _normalize_text(item.get("applicability", ""))
    if not applicability:
        failure_class = _normalize_text(item.get("failure_class", ""))
        missing_step = _normalize_text(item.get("missing_step", ""))
        recovery_action = _normalize_text(item.get("recovery_action", ""))
        if failure_class and missing_step:
            applicability = f"when {failure_class} appears and {missing_step} is missing"
        elif recovery_action:
            applicability = f"when the next best action is {recovery_action}"
        else:
            applicability = f"when handling similar {family} cases"
    if not title or not lesson:
        return None
    return build_reasoning_entry(
        title=title,
        lesson=lesson,
        applicability=applicability,
        family=family,
        outcome_type=outcome_type,
    )


def derive_reasoning_entries(
    *,
    experience_packet: dict[str, Any] | None,
    posterior_text: str,
    family: str,
    outcome_type: str,
) -> list[dict[str, str]]:
    normalized_family = _normalize_text(family) or "default"
    normalized_outcome = _normalize_text(outcome_type).lower() or "failure"
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    if isinstance(experience_packet, dict):
        for item in experience_packet.get("reasoning_items", []) or []:
            if not isinstance(item, dict):
                continue
            entry = build_reasoning_entry(
                title=str(item.get("title", "")),
                lesson=str(item.get("lesson", "")),
                applicability=str(item.get("applicability", "")),
                family=str(item.get("family", normalized_family)),
                outcome_type=str(item.get("outcome_type", normalized_outcome)),
            )
            key = (
                entry["family"].lower(),
                entry["title"].lower(),
                entry["applicability"].lower(),
                entry["lesson"].lower(),
            )
            if entry["title"] and entry["lesson"] and key not in seen:
                seen.add(key)
                entries.append(entry)

        for item in experience_packet.get("memory_items", []) or []:
            if not isinstance(item, dict):
                continue
            entry = _reasoning_from_memory_item(
                item,
                fallback_family=str(experience_packet.get("family_focus", normalized_family))
                or normalized_family,
                outcome_type=normalized_outcome,
            )
            if entry is None:
                continue
            key = (
                entry["family"].lower(),
                entry["title"].lower(),
                entry["applicability"].lower(),
                entry["lesson"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)

    if entries:
        return entries[:3]

    normalized_posterior = _normalize_text(posterior_text)
    if not normalized_posterior:
        return []
    return [
        build_reasoning_entry(
            title=_clip_words(normalized_posterior, 8),
            lesson=_clip_words(normalized_posterior, 32),
            applicability=f"when handling similar {normalized_family} cases",
            family=normalized_family,
            outcome_type=normalized_outcome,
        )
    ]


def apply_reasoning_update(
    bank: dict[str, Any] | None,
    *,
    entries: list[dict[str, Any]],
    source_cycle: int,
    source_task: str,
    task_id: str | None = None,
    reward_value: float | None = None,
) -> dict[str, Any]:
    updated = _ensure_reasoning_bank_shape(bank)
    stored_entries = list(updated.get("entries", []))
    family_index = dict(updated.get("family_index", {}))
    event_ids: list[tuple[str, str]] = []

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = build_reasoning_entry(
            title=str(raw_entry.get("title", "")),
            lesson=str(raw_entry.get("lesson", "")),
            applicability=str(raw_entry.get("applicability", "")),
            family=str(raw_entry.get("family", "default")),
            outcome_type=str(raw_entry.get("outcome_type", "failure")),
        )
        if not entry["title"] or not entry["lesson"]:
            continue
        entry_hash = _reasoning_semantic_hash(entry)
        entry_id = f"{entry['family']}:{entry_hash}"
        existing = next((item for item in stored_entries if item.get("id") == entry_id), None)
        confidence_gain = 1.0 if entry["outcome_type"] == "failure" else 0.75
        if existing is None:
            record = {
                "id": entry_id,
                "content_hash": entry_hash,
                "title": entry["title"],
                "lesson": entry["lesson"],
                "applicability": entry["applicability"],
                "family": entry["family"],
                "outcome_type": entry["outcome_type"],
                "retrieval_text": build_reasoning_retrieval_text(entry),
                "source_cycle": source_cycle,
                "source_task": source_task,
                "task_id": task_id,
                "reward_value": reward_value,
                "confidence": confidence_gain,
                "status": "active",
            }
            stored_entries.append(record)
            family_entries = list(family_index.get(entry["family"], []))
            if entry_id not in family_entries:
                family_entries.append(entry_id)
                family_index[entry["family"]] = family_entries
        else:
            existing["title"] = entry["title"]
            existing["lesson"] = entry["lesson"]
            existing["applicability"] = entry["applicability"]
            existing["family"] = entry["family"]
            existing["outcome_type"] = entry["outcome_type"]
            existing["retrieval_text"] = build_reasoning_retrieval_text(entry)
            existing["last_cycle"] = source_cycle
            existing["last_task"] = source_task
            existing["task_id"] = task_id
            existing["reward_value"] = reward_value
            existing["confidence"] = float(existing.get("confidence", 1.0) or 1.0) + confidence_gain
        event_ids.append((entry["family"], entry_hash))

    updated["entries"] = stored_entries
    updated["family_index"] = family_index
    families = list(updated.get("families", []))
    for entry in stored_entries:
        family = str(entry.get("family", ""))
        if family and family not in families:
            families.append(family)
    updated["families"] = families
    for family, content_hash in event_ids:
        updated.setdefault("events", []).append(
            {
                "family": family,
                "source_cycle": source_cycle,
                "source_task": source_task,
                "task_id": task_id,
                "reward_value": reward_value,
                "content_hash": content_hash,
            }
        )
    return updated


def summarize_reasoning_bank(bank: dict[str, Any], limit_per_family: int = 2) -> str:
    entries = bank.get("entries") or []
    families = bank.get("families") or []
    if not entries:
        return "Reasoning hints: (empty)"
    lines = ["Reasoning hints:"]
    for family in families:
        family_entries = [entry for entry in entries if entry.get("family") == family]
        if not family_entries:
            continue
        lines.append(f"- {family}:")
        for entry in family_entries[:limit_per_family]:
            title = _normalize_text(entry.get("title", ""))
            applicability = _normalize_text(entry.get("applicability", ""))
            lesson = _normalize_text(entry.get("lesson", ""))
            if title and lesson:
                if applicability:
                    lines.append(f"  - {title} [{applicability}]: {lesson}")
                else:
                    lines.append(f"  - {title}: {lesson}")
    return "\n".join(lines)
