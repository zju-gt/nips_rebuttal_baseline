from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from evolve_state import build_default_memory_snapshot
from retrieval import build_memory_retrieval_text


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


def _semantic_hash(family: str, title: str, experience: str) -> str:
    digest = hashlib.sha256(
        f"{family}:{_normalize_text(title)}:{_normalize_text(experience)}".encode("utf-8")
    ).hexdigest()
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


def _packet_allows_update(experience_packet: dict[str, Any] | None) -> bool:
    if not isinstance(experience_packet, dict):
        return False
    if "update_enabled" in experience_packet:
        return bool(experience_packet.get("update_enabled"))
    source = _normalize_text(str(experience_packet.get("extractor_source", ""))).lower()
    if source:
        return source == "prompt"
    return True


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
    experience_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = _ensure_memory_shape(memory)
    if outcome_tag != "failure":
        return updated
    family_index = dict(updated.get("family_index", {}))
    entries = list(updated.get("entries", []))
    packet_items = []
    packet_present = isinstance(experience_packet, dict)
    if packet_present and _packet_allows_update(experience_packet):
        candidate_items = experience_packet.get("memory_items")
        if isinstance(candidate_items, list):
            packet_items = [item for item in candidate_items if isinstance(item, dict)]

    event_hashes: list[tuple[str, str]] = []
    if packet_items:
        for item in packet_items:
            item_family = _normalize_text(str(item.get("family", family))) or family
            title = _normalize_text(str(item.get("title", "")))
            experience = _normalize_text(str(item.get("experience", "")))
            if not title or not experience:
                continue
            entry_hash = _semantic_hash(item_family, title, experience)
            entry_id = f"{item_family}:{entry_hash}"
            existing = None
            for entry in entries:
                if entry.get("id") == entry_id:
                    existing = entry
                    break

            confidence_gain = 1.25 if outcome_tag == "success" else 1.0
            retrieval_text = build_memory_retrieval_text(
                {
                    "title": title,
                    "experience": experience,
                    "family": item_family,
                }
            )
            if existing is not None:
                existing["confidence"] = float(existing.get("confidence", 1.0)) + confidence_gain
                existing["last_cycle"] = source_cycle
                existing["last_task"] = source_task
                existing["task_id"] = task_id
                existing["outcome_tag"] = outcome_tag
                existing["reward_value"] = reward_value
                existing["meta_sop_hint"] = _clip_words(meta_sop_text, 8) if meta_sop_text else ""
                existing["title"] = title
                existing["experience"] = experience
                existing["content"] = experience
                existing["retrieval_text"] = retrieval_text
            else:
                entry = {
                    "id": entry_id,
                    "title": title,
                    "experience": experience,
                    "content": experience,
                    "family": item_family,
                    "content_hash": entry_hash,
                    "retrieval_text": retrieval_text,
                    "source_cycle": source_cycle,
                    "source_task": source_task,
                    "task_id": task_id,
                    "confidence": confidence_gain,
                    "outcome_tag": outcome_tag,
                    "reward_value": reward_value,
                    "meta_sop_hint": _clip_words(meta_sop_text, 8) if meta_sop_text else "",
                    "status": "active",
                }
                entries.append(entry)
                family_entries = list(family_index.get(item_family, []))
                if entry_id not in family_entries:
                    family_entries.append(entry_id)
                    family_index[item_family] = family_entries
            event_hashes.append((item_family, entry_hash))

        if not event_hashes:
            packet_items = []

    if not packet_items:
        if packet_present:
            return updated
        experience = _normalize_text(posterior_text)
        title = _clip_words(experience, 8)
        content = experience
        entry_hash = _content_hash(family, content)
        entry_id = f"{family}:{entry_hash}"
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
                "title": title,
                "experience": experience,
                "content": content,
                "family": family,
                "content_hash": entry_hash,
                "retrieval_text": build_memory_retrieval_text(
                    {
                        "title": title,
                        "experience": experience,
                        "family": family,
                    }
                ),
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
        event_hashes = [(family, entry_hash)]

    updated["entries"] = entries
    updated["family_index"] = family_index
    for event_family, content_hash in event_hashes:
        updated.setdefault("events", []).append(
            {
                "family": event_family,
                "source_cycle": source_cycle,
                "source_task": source_task,
                "task_id": task_id,
                "outcome_tag": outcome_tag,
                "reward_value": reward_value,
                "content_hash": content_hash,
            }
        )
    families = list(updated.get("families", []))
    for entry in entries:
        entry_family = str(entry.get("family", ""))
        if entry_family and entry_family not in families:
            families.append(entry_family)
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
            title = _normalize_text(str(entry.get("title", "")))
            experience = _normalize_text(str(entry.get("experience", "") or entry.get("content", "")))
            if title and experience:
                lines.append(f"  - {title}: {experience}")
            elif experience:
                lines.append(f"  - {experience}")
    if len(lines) == 1:
        for entry in entries[:limit_per_family]:
            title = _normalize_text(str(entry.get("title", "")))
            experience = _normalize_text(str(entry.get("experience", "") or entry.get("content", "")))
            text = f"{title}: {experience}" if title and experience else experience
            lines.append(f"- {entry.get('family', 'task')}: {text}")
    return "\n".join(lines)
