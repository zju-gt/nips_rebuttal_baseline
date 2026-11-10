from __future__ import annotations

import json
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


def _task_text(task: dict[str, Any]) -> str:
    return json.dumps(task, ensure_ascii=False, sort_keys=True).lower()


def infer_task_family(domain: str, task: dict[str, Any]) -> str:
    normalized_domain = domain.strip().lower()
    text = _task_text(task)
    domain_families = FAMILY_KEYWORDS.get(normalized_domain, {})
    for family, keywords in domain_families.items():
        if any(keyword in text for keyword in keywords):
            return family
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
    task_id: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "cycle_id": cycle_id,
        "task_family": task_family,
        "outcome_tag": outcome_tag,
        "reward_value": reward_value,
        "posterior_text": posterior_text,
        "memory_delta": memory_delta,
        "graph_delta": graph_delta,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
