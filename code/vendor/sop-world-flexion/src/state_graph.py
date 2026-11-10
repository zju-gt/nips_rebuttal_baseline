from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from evolve_state import build_default_state_graph


def seed_state_graph(domain: str) -> dict[str, Any]:
    return build_default_state_graph(domain)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clip_words(text: str, limit: int = 16) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _step_id(domain: str, family: str, label: str) -> str:
    digest = hashlib.sha256(f"{domain}:{family}:{_normalize_text(label)}".encode("utf-8")).hexdigest()
    return f"{domain}:{family}:{digest[:12]}"


def _ensure_graph_shape(graph: dict[str, Any], domain: str | None = None) -> dict[str, Any]:
    normalized = deepcopy(graph or {})
    normalized.setdefault("version", 1)
    if domain is not None:
        normalized.setdefault("root_domain", domain)
    normalized.setdefault("families", [])
    normalized.setdefault("nodes", [])
    normalized.setdefault("edges", [])
    return normalized


def _ensure_family_layers(graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    root_domain = graph.get("root_domain", "domain")
    family_nodes: dict[str, str] = {}

    root_id = f"{root_domain}:root"
    if not any(node.get("id") == root_id for node in nodes):
        nodes.insert(
            0,
            {
                "id": root_id,
                "type": "root",
                "label": root_domain,
                "family": None,
            },
        )

    for family in graph.get("families", []):
        node_id = f"{root_domain}:{family}"
        family_nodes[family] = node_id
        if not any(node.get("id") == node_id for node in nodes):
            nodes.append(
                {
                    "id": node_id,
                    "type": "family",
                    "label": family,
                    "family": family,
                }
            )
        if not any(edge.get("source") == root_id and edge.get("target") == node_id for edge in edges):
            edges.append({"source": root_id, "target": node_id, "type": "family"})

    graph["nodes"] = nodes
    graph["edges"] = edges
    return graph, family_nodes


def apply_state_graph_update(
    graph: dict[str, Any],
    *,
    family: str,
    posterior_text: str,
    outcome_tag: str,
    source_cycle: int,
    source_task: str,
    reward_value: float | None = None,
    meta_sop_text: str = "",
) -> dict[str, Any]:
    updated = _ensure_graph_shape(graph)
    families = list(updated.get("families", []))
    if family not in families:
        families.append(family)
        updated["families"] = families

    updated, family_nodes = _ensure_family_layers(updated)
    family_node_id = family_nodes.get(family, f"{updated.get('root_domain', 'domain')}:{family}")

    label = _clip_words(posterior_text, 16)
    if meta_sop_text:
        label = f"{label} | { _clip_words(meta_sop_text, 6) }"
    node_id = _step_id(str(updated.get("root_domain", "domain")), family, label)

    nodes = list(updated.get("nodes", []))
    edges = list(updated.get("edges", []))
    existing = None
    for node in nodes:
        if node.get("id") == node_id:
            existing = node
            break

    if existing is not None:
        existing["confidence"] = float(existing.get("confidence", 1.0)) + 1.0
        existing["last_cycle"] = source_cycle
        existing["last_task"] = source_task
        existing["outcome_tag"] = outcome_tag
        if reward_value is not None:
            existing["reward_value"] = reward_value
    else:
        nodes.append(
            {
                "id": node_id,
                "type": "step",
                "label": label,
                "family": family,
                "content": _clip_words(posterior_text, 28),
                "source_cycle": source_cycle,
                "source_task": source_task,
                "outcome_tag": outcome_tag,
                "reward_value": reward_value,
                "confidence": 1.0,
            }
        )
        if not any(edge.get("source") == family_node_id and edge.get("target") == node_id for edge in edges):
            edges.append({"source": family_node_id, "target": node_id, "type": "step"})

    updated["nodes"] = nodes
    updated["edges"] = edges
    return updated


def summarize_state_graph(graph: dict[str, Any], limit_per_family: int = 2) -> str:
    families = graph.get("families") or []
    nodes = graph.get("nodes") or []
    step_nodes = [node for node in nodes if node.get("type") == "step"]
    if not step_nodes:
        return "State graph: (empty)"

    lines = ["State graph:"]
    for family in families:
        family_nodes = [node for node in step_nodes if node.get("family") == family]
        if not family_nodes:
            continue
        lines.append(f"- {family}:")
        for node in family_nodes[:limit_per_family]:
            lines.append(f"  - {node.get('label', '').strip()}")
    if len(lines) == 1:
        for node in step_nodes[:limit_per_family]:
            lines.append(f"- {node.get('family', 'task')}: {node.get('label', '').strip()}")
    return "\n".join(lines)

