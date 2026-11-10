from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from evolve_state import DOMAIN_TASK_FAMILIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts" / "evolution"
WORKFLOW_EXTRACTOR_MAX_NEW_TOKENS = 2048
_PROMPT_CACHE: dict[str, str] = {}

ExtractorGenerateFn = Callable[..., Any]

_DEFAULT_WORKFLOW_STEPS: dict[str, dict[str, list[str]]] = {
    "retail": {
        "authentication": [
            "resolve customer identity",
            "validate user record with fallback attributes",
            "continue only after identity is grounded",
        ],
        "status": [
            "resolve customer or order identity",
            "fetch order details",
            "communicate status with exact order facts",
        ],
        "exchange_modify": [
            "resolve customer or order identity",
            "fetch order details and relevant items",
            "check exchange or modification eligibility",
            "confirm the replacement item or change request",
            "execute the exchange or modification action",
        ],
        "variant": [
            "identify the target product",
            "fetch product details and variants",
            "confirm the requested variant before acting",
        ],
        "payment": [
            "resolve order identity",
            "inspect payment method or refund route",
            "confirm the supported payment action",
            "execute the payment update or refund step",
        ],
    },
    "airline": {
        "cancellation": [
            "ground the reservation facts",
            "check cancellation eligibility and penalties",
            "confirm refund path and exact totals",
            "execute the cancellation action",
        ],
        "modification": [
            "freeze the current reservation goal",
            "ground reservation facts and target changes",
            "price the modification before committing",
            "execute the reservation update",
        ],
        "compensation": [
            "ground the disrupted itinerary facts",
            "check compensation eligibility",
            "compute the supported certificate or compensation amount",
            "issue the compensation action",
        ],
        "booking": [
            "clarify travel requirements",
            "search and compare valid flight options",
            "confirm the selected itinerary and passenger facts",
            "execute the booking action",
        ],
        "verification": [
            "ground the traveler or reservation identity",
            "check the relevant eligibility or coverage facts",
            "communicate the verified outcome",
        ],
    },
    "telecom": {
        "mobile_data": [
            "ground the line and data issue",
            "diagnose connectivity prerequisites in order",
            "apply the supported network or data fix",
            "confirm the expected connectivity outcome",
        ],
        "mms": [
            "ground the line and messaging issue",
            "verify MMS-specific prerequisites",
            "apply the supported MMS troubleshooting step",
            "confirm the expected MMS outcome",
        ],
        "roaming": [
            "ground the roaming context and line",
            "verify roaming eligibility and settings",
            "apply the supported roaming fix or explanation",
        ],
        "billing": [
            "ground the line and billing state",
            "inspect the balance or suspension facts",
            "apply the supported billing resolution",
            "confirm the updated account state",
        ],
        "apn_reboot": [
            "ground the line and network issue",
            "check APN and device-side prerequisites",
            "apply the supported network reset or APN fix",
            "confirm the expected connectivity outcome",
        ],
    },
}


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clip_words(text: str, limit: int) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _load_prompt_template(filename: str) -> str:
    cached = _PROMPT_CACHE.get(filename)
    if cached is not None:
        return cached
    text = (PROMPTS_ROOT / filename).read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[filename] = text
    return text


def _extract_json_text(text: str) -> str | None:
    stripped = _normalize_text(text)
    if not stripped:
        return None
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate
    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        start = stripped.find(start_char)
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except Exception:
            continue
        return json.dumps(obj, ensure_ascii=False)
    return None


def _safe_json_loads(text: str) -> Any:
    json_text = _extract_json_text(text)
    if json_text is None:
        raise ValueError("No JSON object found in workflow extractor output")
    return json.loads(json_text)


def _family_choices(domain: str, family_prior: str) -> list[str]:
    normalized_domain = _normalize_text(domain).lower()
    families = list(DOMAIN_TASK_FAMILIES.get(normalized_domain, []))
    if family_prior and family_prior not in families:
        families.append(family_prior)
    return families or [family_prior or "default"]


def seed_workflow_memory(domain: str) -> dict[str, Any]:
    families = DOMAIN_TASK_FAMILIES.get(domain.strip().lower(), [])
    return {
        "version": 1,
        "domain": domain,
        "families": families,
        "entries": [],
        "family_index": {},
        "events": [],
    }


def _ensure_workflow_memory_shape(
    memory: dict[str, Any],
    domain: str | None = None,
) -> dict[str, Any]:
    normalized = deepcopy(memory or {})
    normalized.setdefault("version", 1)
    if domain is not None:
        normalized.setdefault("domain", domain)
    normalized.setdefault("families", [])
    normalized.setdefault("entries", [])
    normalized.setdefault("family_index", {})
    normalized.setdefault("events", [])
    return normalized


def _workflow_entry_hash(
    family: str,
    title: str,
    applicability: str,
    workflow: list[str],
) -> str:
    digest = hashlib.sha256(
        (
            f"{family}:{_normalize_text(title)}:{_normalize_text(applicability)}:"
            f"{'|'.join(_normalize_text(step) for step in workflow)}"
        ).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _sanitize_workflow_items(
    value: Any,
    *,
    domain: str,
    family_prior: str,
) -> list[dict[str, Any]]:
    allowed_families = set(_family_choices(domain, family_prior))
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        family = _normalize_text(item.get("family", family_prior)) or family_prior
        if family not in allowed_families:
            family = family_prior
        title = _clip_words(_normalize_text(item.get("title", "")), 12)
        task_signature = _clip_words(_normalize_text(item.get("task_signature", "")), 24)
        applicability = _clip_words(_normalize_text(item.get("applicability", "")), 32)
        raw_steps = item.get("workflow", [])
        if not isinstance(raw_steps, list):
            raw_steps = []
        steps: list[str] = []
        seen_steps: set[str] = set()
        for raw_step in raw_steps:
            step = _clip_words(_normalize_text(raw_step), 12)
            if not step:
                continue
            key = step.lower()
            if key in seen_steps:
                continue
            seen_steps.add(key)
            steps.append(step)
            if len(steps) >= 6:
                break
        if not title or not task_signature or not applicability or len(steps) < 2:
            continue
        dedupe_key = (family, title.lower(), applicability.lower(), tuple(step.lower() for step in steps))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            {
                "title": title,
                "task_signature": task_signature,
                "workflow": steps,
                "applicability": applicability,
                "family": family,
            }
        )
        if len(normalized) >= 2:
            break
    return normalized


def build_workflow_retrieval_text(item: dict[str, Any]) -> str:
    title = _normalize_text(item.get("title", ""))
    applicability = _normalize_text(item.get("applicability", ""))
    task_signature = _normalize_text(item.get("task_signature", ""))
    workflow = [
        _normalize_text(step)
        for step in item.get("workflow", [])
        if _normalize_text(step)
    ]
    sections: list[str] = []
    if title:
        sections.append(f"Title: {title}")
    if applicability:
        sections.append(f"Applicability: {applicability}")
    if task_signature:
        sections.append(f"Task Signature: {task_signature}")
    if workflow:
        sections.append(f"Workflow: {' -> '.join(workflow)}")
    return "\n".join(sections).strip()


def summarize_workflow_memory(memory: dict[str, Any], limit_per_family: int = 1) -> str:
    entries = list(memory.get("entries") or [])
    if not entries:
        return "Workflow memory: (empty)"
    lines = ["Workflow memory:"]
    for family in memory.get("families") or []:
        family_entries = [
            entry
            for entry in entries
            if _normalize_text(entry.get("family", "")) == _normalize_text(family)
            and str(entry.get("status", "")) == "active"
        ]
        if not family_entries:
            continue
        lines.append(f"- {family}:")
        for entry in family_entries[:limit_per_family]:
            lines.append(f"  - {entry['title']}: {' -> '.join(entry['workflow'])}")
    if len(lines) == 1:
        for entry in entries[:limit_per_family]:
            lines.append(f"- {entry['family']}: {entry['title']}")
    return "\n".join(lines)


def apply_workflow_memory_update(
    memory: dict[str, Any],
    *,
    family: str,
    source_cycle: int,
    source_task: str,
    task_id: str | None,
    task_query: str,
    workflow_items: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    updated = _ensure_workflow_memory_shape(memory)
    normalized_items = _sanitize_workflow_items(
        workflow_items,
        domain=str(updated.get("domain", "")),
        family_prior=family,
    )
    if not normalized_items:
        return updated

    entries = list(updated.get("entries", []))
    family_index = dict(updated.get("family_index", {}))
    for item in normalized_items:
        entry_hash = _workflow_entry_hash(
            item["family"],
            item["title"],
            item["applicability"],
            item["workflow"],
        )
        entry_id = f"{item['family']}:{entry_hash}"
        existing = next((entry for entry in entries if entry.get("id") == entry_id), None)
        if existing is None:
            existing = {
                "id": entry_id,
                "family": item["family"],
                "title": item["title"],
                "task_signature": item["task_signature"],
                "workflow": list(item["workflow"]),
                "applicability": item["applicability"],
                "retrieval_text": build_workflow_retrieval_text(item),
                "source_cycle": source_cycle,
                "success_count": 1,
                "source_tasks": [source_task],
                "task_ids": [task_id] if task_id else [],
                "last_task_query": _clip_words(task_query, 40),
                "status": "candidate",
            }
            entries.append(existing)
            family_entries = list(family_index.get(item["family"], []))
            if entry_id not in family_entries:
                family_entries.append(entry_id)
                family_index[item["family"]] = family_entries
        else:
            existing["success_count"] = int(existing.get("success_count", 1)) + 1
            existing["last_task_query"] = _clip_words(task_query, 40)
            source_tasks = list(existing.get("source_tasks", []))
            if source_task not in source_tasks:
                source_tasks.append(source_task)
            existing["source_tasks"] = source_tasks
            task_ids = list(existing.get("task_ids", []))
            if task_id and task_id not in task_ids:
                task_ids.append(task_id)
            existing["task_ids"] = task_ids
        if int(existing.get("success_count", 0)) >= 2:
            existing["status"] = "active"
        updated["events"].append(
            {
                "family": item["family"],
                "entry_id": entry_id,
                "source_cycle": source_cycle,
                "source_task": source_task,
                "task_id": task_id,
                "task_query": _clip_words(task_query, 40),
                "status": existing["status"],
                "success_count": existing["success_count"],
            }
        )

    updated["entries"] = entries
    updated["family_index"] = family_index
    families = list(updated.get("families", []))
    for entry in entries:
        entry_family = str(entry.get("family", ""))
        if entry_family and entry_family not in families:
            families.append(entry_family)
    updated["families"] = families
    return updated


def _task_query_text(task_query: str, trajectory_text: str) -> str:
    normalized_query = _normalize_text(task_query)
    if normalized_query:
        return _clip_words(normalized_query, 80)
    for raw_line in str(trajectory_text or "").splitlines():
        line = _normalize_text(raw_line)
        if line.lower().startswith("user:"):
            return _clip_words(line.split(":", 1)[1], 80)
    return "(not available)"


def _trajectory_evidence(trajectory_text: str) -> str:
    lines = [_normalize_text(line) for line in str(trajectory_text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "(empty)"
    return "\n".join(lines[:20])


def _normalize_extractor_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and any(
        key in result
        for key in (
            "payload",
            "raw_text",
            "repaired_text",
            "used_repair",
            "failure_stage",
            "failure_reason",
        )
    ):
        payload = result.get("payload")
        raw_text = str(result.get("raw_text", "") or "").strip()
        if payload is None and raw_text:
            try:
                payload = _safe_json_loads(raw_text)
            except Exception:
                payload = None
        return {
            "payload": payload,
            "raw_text": raw_text,
            "used_repair": bool(result.get("used_repair", False)),
            "failure_stage": str(result.get("failure_stage", "") or ""),
            "failure_reason": str(result.get("failure_reason", "") or ""),
        }
    raw_text = str(result or "").strip()
    payload = None
    if raw_text:
        try:
            payload = _safe_json_loads(raw_text)
        except Exception:
            payload = None
    return {
        "payload": payload,
        "raw_text": raw_text,
        "used_repair": False,
        "failure_stage": "",
        "failure_reason": "",
    }


def _build_workflow_extractor_messages(
    *,
    task_query: str,
    trajectory_text: str,
    family_prior: str,
    domain: str,
) -> list[dict[str, str]]:
    family_options = ", ".join(_family_choices(domain, family_prior))
    return [
        {
            "role": "system",
            "content": _load_prompt_template("workflow_success_extractor.md"),
        },
        {
            "role": "user",
            "content": (
                "Return exactly one valid JSON object matching the schema in the system instruction.\n"
                "The first non-whitespace character of your response must be '{' and the last must be '}'.\n\n"
                f"Task Query:\n{task_query}\n\n"
                f"Weak Family Prior:\n{family_prior}\n\n"
                f"Allowed Family Labels:\n{family_options}\n\n"
                f"Trajectory Evidence:\n{_trajectory_evidence(trajectory_text)}"
            ),
        },
    ]


def _fallback_workflow_items(
    *,
    domain: str,
    family: str,
    task_query: str,
) -> list[dict[str, Any]]:
    steps = list(_DEFAULT_WORKFLOW_STEPS.get(domain, {}).get(family, []))
    if not steps:
        return []
    task_signature = _task_query_text(task_query, "")
    return [
        {
            "title": _clip_words(
                f"{family.replace('_', ' ').title()} success workflow",
                10,
            ),
            "task_signature": task_signature,
            "workflow": steps,
            "applicability": _clip_words(
                f"Use for similar {family.replace('_', ' ')} requests when the same task pattern appears.",
                28,
            ),
            "family": family,
        }
    ]


def build_workflow_memory_signal(
    *,
    sample_id: str,
    trajectory_text: str,
    task_query: str,
    task_family: str,
    domain: str,
    extractor_generate: ExtractorGenerateFn | None = None,
) -> dict[str, Any]:
    normalized_domain = _normalize_text(domain).lower()
    normalized_family = _normalize_text(task_family) or "default"
    normalized_query = _task_query_text(task_query, trajectory_text)
    prompt_result: dict[str, Any] | None = None
    workflow_items: list[dict[str, Any]] = []
    extractor_source = "fallback"

    if extractor_generate is not None and _normalize_text(trajectory_text):
        prompt_result = _normalize_extractor_result(
            extractor_generate(
                messages=_build_workflow_extractor_messages(
                    task_query=normalized_query,
                    trajectory_text=trajectory_text,
                    family_prior=normalized_family,
                    domain=normalized_domain,
                ),
                call_name="workflow_success",
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=WORKFLOW_EXTRACTOR_MAX_NEW_TOKENS,
            )
        )
        payload = prompt_result.get("payload")
        if isinstance(payload, dict):
            workflow_items = _sanitize_workflow_items(
                payload.get("workflow_items", []),
                domain=normalized_domain,
                family_prior=normalized_family,
            )
            if workflow_items:
                extractor_source = "prompt"

    if not workflow_items:
        workflow_items = _fallback_workflow_items(
            domain=normalized_domain,
            family=normalized_family,
            task_query=normalized_query,
        )

    return {
        "sample_id": sample_id,
        "workflow_items": workflow_items,
        "family_focus": normalized_family,
        "extractor_source": extractor_source,
        "extractor_used_repair": bool((prompt_result or {}).get("used_repair", False)),
        "extractor_failure_stage": str((prompt_result or {}).get("failure_stage", "") or ""),
        "extractor_failure_reason": str((prompt_result or {}).get("failure_reason", "") or ""),
        "extractor_raw_text": str((prompt_result or {}).get("raw_text", "") or ""),
        "update_enabled": bool(workflow_items),
    }
