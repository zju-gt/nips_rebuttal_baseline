from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from evolve_state import build_default_memory_snapshot


def seed_memory_snapshot(domain: str) -> dict[str, Any]:
    return build_default_memory_snapshot(domain)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clip_words(text: str, limit: int = 24) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _content_hash(family: str, content: str) -> str:
    digest = hashlib.sha256(f"{family}:{_normalize_text(content)}".encode("utf-8")).hexdigest()
    return digest[:12]


def _ensure_memory_shape(memory: dict[str, Any], domain: str | None = None) -> dict[str, Any]:
    normalized = deepcopy(memory or {})
    normalized.setdefault("version", 1)
    if domain is not None:
        normalized.setdefault("domain", domain)
    normalized.setdefault("families", [])
    normalized.setdefault("entries", [])
    normalized.setdefault("family_index", {})
    normalized.setdefault("events", [])
    return normalized


def apply_memory_update(
    memory: dict[str, Any],
    *,
    family: str,
    posterior_text: str,
    outcome_tag: str,
    source_cycle: int,
    source_task: str,
    reward_value: float | None = None,
    meta_sop_text: str = "",
    task_id: str | None = None,
) -> dict[str, Any]:
    updated = _ensure_memory_shape(memory)
    content = _clip_words(posterior_text, 28)
    entry_hash = _content_hash(family, content)
    entry_id = f"{family}:{entry_hash}"
    family_index = dict(updated.get("family_index", {}))
    entries = list(updated.get("entries", []))

    existing = None
    for entry in entries:
        if entry.get("id") == entry_id or (
            entry.get("family") == family and entry.get("content_hash") == entry_hash
        ):
            existing = entry
            break

    if existing is not None:
        existing["confidence"] = float(existing.get("confidence", 1.0)) + 1.0
        existing["last_cycle"] = source_cycle
        existing["last_task"] = source_task
        existing["outcome_tag"] = outcome_tag
        if reward_value is not None:
            existing["reward_value"] = reward_value
        existing["meta_sop_hint"] = _clip_words(meta_sop_text, 8) if meta_sop_text else ""
    else:
        entry = {
            "id": entry_id,
            "title": _clip_words(content, 8),
            "content": content,
            "family": family,
            "content_hash": entry_hash,
            "source_cycle": source_cycle,
            "source_task": source_task,
            "task_id": task_id,
            "confidence": 1.0 if reward_value is None or reward_value <= 0 else 1.5,
            "outcome_tag": outcome_tag,
            "reward_value": reward_value,
            "meta_sop_hint": _clip_words(meta_sop_text, 8) if meta_sop_text else "",
            "status": "active",
        }
        entries.append(entry)
        family_entries = list(family_index.get(family, []))
        family_entries.append(entry_id)
        family_index[family] = family_entries

    updated["entries"] = entries
    updated["family_index"] = family_index
    updated.setdefault("events", []).append(
        {
            "family": family,
            "source_cycle": source_cycle,
            "source_task": source_task,
            "task_id": task_id,
            "outcome_tag": outcome_tag,
            "reward_value": reward_value,
            "content_hash": entry_hash,
        }
    )
    families = list(updated.get("families", []))
    if family not in families:
        families.append(family)
        updated["families"] = families
    return updated


def summarize_memory_snapshot(memory: dict[str, Any], limit_per_family: int = 2) -> str:
    families = memory.get("families") or []
    entries = memory.get("entries") or []
    if not entries:
        return "Memory: (empty)"

    lines = ["Memory:"]
    for family in families:
        family_entries = [entry for entry in entries if entry.get("family") == family]
        if not family_entries:
            continue
        lines.append(f"- {family}:")
        for entry in family_entries[:limit_per_family]:
            lines.append(f"  - {entry.get('content', '').strip()}")
    if len(lines) == 1:
        for entry in entries[:limit_per_family]:
            lines.append(f"- {entry.get('family', 'task')}: {entry.get('content', '').strip()}")
    return "\n".join(lines)

