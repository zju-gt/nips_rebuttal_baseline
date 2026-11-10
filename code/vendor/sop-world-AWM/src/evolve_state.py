from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DOMAIN_TASK_FAMILIES: dict[str, list[str]] = {
    "airline": ["cancellation", "modification", "compensation", "booking", "verification"],
    "retail": ["authentication", "status", "exchange_modify", "variant", "payment"],
    "telecom": ["mobile_data", "mms", "roaming", "billing", "apn_reboot"],
    "multiwoz": ["restaurant", "hotel", "train", "attraction", "taxi"],
}

FAMILY_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "airline": {
        "cancellation": ("cancel", "cancellation", "refund"),
        "modification": ("modify", "change", "rebook", "rebooking"),
        "compensation": ("compensation", "voucher", "delay", "delayed"),
        "booking": ("book", "booking", "reservation", "seat"),
        "verification": ("verify", "eligibility", "membership", "insurance"),
    },
    "retail": {
        "authentication": ("authenticate", "authentication", "email", "zip"),
        "status": ("order status", "pending", "delivered", "status"),
        "exchange_modify": ("exchange", "modify", "one-shot", "one shot"),
        "variant": ("variant", "product details", "product_id"),
        "payment": ("gift card", "refund", "payment", "credit card", "paypal"),
    },
    "telecom": {
        "mobile_data": ("mobile data", "data", "internet", "slow"),
        "mms": ("mms", "picture", "message", "photo"),
        "roaming": ("roaming", "abroad"),
        "billing": ("billing", "suspension", "overdue", "payment"),
        "apn_reboot": ("apn", "reboot", "network preference"),
    },
    "multiwoz": {
        "restaurant": ("restaurant", "food", "table", "reservation"),
        "hotel": ("hotel", "room", "parking", "stars"),
        "train": ("train", "departure", "arrive", "leave"),
        "attraction": ("attraction", "museum", "park", "theatre"),
        "taxi": ("taxi", "car", "pickup", "destination"),
    },
}

ACTION_FAMILY_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "airline": {
        "cancellation": ("cancel_reservation",),
        "modification": (
            "update_reservation_flights",
            "update_reservation_baggages",
            "update_reservation_passengers",
            "update_reservation",
        ),
        "compensation": ("create_certificate", "compensation", "voucher", "certificate"),
        "booking": ("search_direct_flight", "book_reservation"),
        "verification": ("get_user_details", "get_reservation_details"),
    },
    "retail": {
        "authentication": ("find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"),
        "status": ("get_order_details",),
        "exchange_modify": (
            "exchange_delivered_order_items",
            "modify_pending_order_items",
            "modify_pending_order_address",
            "return_delivered_order_items",
            "cancel_pending_order",
        ),
        "variant": ("get_product_details",),
        "payment": ("modify_pending_order_payment", "calculate"),
    },
    "telecom": {},
    "multiwoz": {},
}


def _task_text(task: dict[str, Any]) -> str:
    return json.dumps(task, ensure_ascii=False, sort_keys=True).lower()


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _task_text_sections(task: dict[str, Any]) -> list[tuple[str, int]]:
    sections: list[tuple[str, int]] = []
    description = task.get("description", {}) if isinstance(task, dict) else {}
    instructions = (
        task.get("user_scenario", {}).get("instructions", {})
        if isinstance(task, dict)
        else {}
    )
    evaluation = task.get("evaluation_criteria", {}) if isinstance(task, dict) else {}

    for key, weight in (("purpose", 4), ("notes", 3), ("relevant_policies", 2)):
        text = _normalize_text(description.get(key, ""))
        if text:
            sections.append((text.lower(), weight))

    for key, weight in (
        ("reason_for_call", 8),
        ("task_instructions", 5),
        ("known_info", 1),
        ("unknown_info", 1),
    ):
        text = _normalize_text(instructions.get(key, ""))
        if text:
            sections.append((text.lower(), weight))

    action_names = [
        _normalize_text(action.get("name", "")).lower()
        for action in evaluation.get("actions", [])
        if isinstance(action, dict) and _normalize_text(action.get("name", ""))
    ]
    if action_names:
        sections.append((" ".join(action_names), 10))

    for key, weight in (("communicate_info", 3), ("nl_assertions", 4)):
        values = evaluation.get(key, [])
        if isinstance(values, list):
            text = " ".join(
                _normalize_text(value).lower()
                for value in values
                if _normalize_text(value)
            )
            if text:
                sections.append((text, weight))

    if not sections:
        sections.append((_task_text(task), 1))
    return sections


def infer_task_family(domain: str, task: dict[str, Any]) -> str:
    normalized_domain = domain.strip().lower()
    domain_families = FAMILY_KEYWORDS.get(normalized_domain, {})
    if not domain_families:
        families = DOMAIN_TASK_FAMILIES.get(normalized_domain)
        if families:
            return families[0]
        return "default"

    scores = {family: 0 for family in domain_families}
    for text, weight in _task_text_sections(task):
        for family, keywords in domain_families.items():
            for keyword in keywords:
                if keyword in text:
                    scores[family] += weight

    action_hints = ACTION_FAMILY_HINTS.get(normalized_domain, {})
    actions = (
        task.get("evaluation_criteria", {}).get("actions", [])
        if isinstance(task, dict)
        else []
    )
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_name = _normalize_text(action.get("name", "")).lower()
        if not action_name:
            continue
        for family, hints in action_hints.items():
            if any(hint in action_name for hint in hints):
                scores[family] = scores.get(family, 0) + 12

    best_family, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_family
    families = DOMAIN_TASK_FAMILIES.get(normalized_domain)
    if families:
        return families[0]
    return "default"


def build_default_memory_snapshot(domain: str) -> dict[str, Any]:
    families = DOMAIN_TASK_FAMILIES.get(domain.strip().lower(), [])
    return {
        "version": 1,
        "domain": domain,
        "families": families,
        "entries": [],
        "family_index": {},
        "events": [],
    }


def build_default_state_graph(domain: str) -> dict[str, Any]:
    families = DOMAIN_TASK_FAMILIES.get(domain.strip().lower(), [])
    nodes = [
        {
            "id": f"{domain}:root",
            "type": "root",
            "label": domain,
            "family": None,
        }
    ]
    edges: list[dict[str, Any]] = []
    for family in families:
        node_id = f"{domain}:{family}"
        nodes.append(
            {
                "id": node_id,
                "type": "family",
                "label": family,
                "family": family,
            }
        )
        edges.append({"source": f"{domain}:root", "target": node_id, "type": "family"})
    return {
        "version": 1,
        "root_domain": domain,
        "families": families,
        "nodes": nodes,
        "edges": edges,
    }


def record_evolution_event(
    *,
    sample_id: str,
    cycle_id: int,
    outcome_tag: str,
    reward_value: float | None,
    posterior_text: str,
    task_family: str | None = None,
    memory_delta: Any | None = None,
    graph_delta: Any | None = None,
    workflow_delta: Any | None = None,
    task_id: str | None = None,
    family_prior: str | None = None,
    family_focus: str | None = None,
    diagnostic_flags: list[str] | None = None,
    experience_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "cycle_id": cycle_id,
        "task_family": task_family,
        "family_prior": family_prior,
        "family_focus": family_focus,
        "outcome_tag": outcome_tag,
        "reward_value": reward_value,
        "posterior_text": posterior_text,
        "diagnostic_flags": list(diagnostic_flags or []),
        "experience_packet": experience_packet,
        "memory_delta": memory_delta,
        "graph_delta": graph_delta,
        "workflow_delta": workflow_delta,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
