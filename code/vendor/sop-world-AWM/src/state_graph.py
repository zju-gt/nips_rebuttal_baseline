from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from evolve_state import build_default_state_graph
from retrieval import build_graph_retrieval_text


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


def _packet_allows_update(experience_packet: dict[str, Any] | None) -> bool:
    if not isinstance(experience_packet, dict):
        return False
    if "update_enabled" in experience_packet:
        return bool(experience_packet.get("update_enabled"))
    source = _normalize_text(str(experience_packet.get("extractor_source", ""))).lower()
    if source:
        return source == "prompt"
    return True


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


def _ensure_step_node(
    graph: dict[str, Any],
    *,
    family: str,
    label: str,
    source_cycle: int,
    source_task: str,
    outcome_tag: str,
    reward_value: float | None,
    content: str,
) -> str:
    normalized_label = _normalize_text(label)
    node_id = _step_id(str(graph.get("root_domain", "domain")), family, normalized_label)
    nodes = list(graph.get("nodes", []))
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
        if content.strip():
            existing["content"] = _normalize_text(content)
    else:
        nodes.append(
            {
                "id": node_id,
                "type": "step",
                "label": normalized_label,
                "family": family,
                "content": _normalize_text(content),
                "source_cycle": source_cycle,
                "source_task": source_task,
                "outcome_tag": outcome_tag,
                "reward_value": reward_value,
                "confidence": 1.0,
            }
        )
    graph["nodes"] = nodes
    return node_id


def _has_edge(
    edges: list[dict[str, Any]],
    *,
    source: str,
    target: str,
    edge_type: str,
    condition: str = "",
) -> bool:
    for edge in edges:
        if (
            edge.get("source") == source
            and edge.get("target") == target
            and edge.get("type") == edge_type
            and _normalize_text(str(edge.get("condition", ""))) == _normalize_text(condition)
        ):
            return True
    return False


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
    experience_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = _ensure_graph_shape(graph)
    packet_updates: list[dict[str, Any]] = []
    packet_present = isinstance(experience_packet, dict)
    if packet_present and _packet_allows_update(experience_packet):
        candidate_updates = experience_packet.get("graph_updates")
        if isinstance(candidate_updates, list):
            packet_updates = [item for item in candidate_updates if isinstance(item, dict)]

    applied_packet_update = False
    if packet_updates:
        for graph_update in packet_updates:
            op = _normalize_text(str(graph_update.get("op", ""))).lower()
            if op not in {"insert_before", "insert_after", "strengthen_edge"}:
                continue
            update_family = _normalize_text(str(graph_update.get("family", family))) or family
            anchor_state = _normalize_text(str(graph_update.get("anchor_state", "")))
            state = _normalize_text(str(graph_update.get("state", "")))
            condition = _normalize_text(str(graph_update.get("condition", "")))
            if not anchor_state or not state:
                continue

            families = list(updated.get("families", []))
            if update_family not in families:
                families.append(update_family)
                updated["families"] = families
            updated, family_nodes = _ensure_family_layers(updated)
            family_node_id = family_nodes.get(
                update_family,
                f"{updated.get('root_domain', 'domain')}:{update_family}",
            )
            edges = list(updated.get("edges", []))

            anchor_id = _ensure_step_node(
                updated,
                family=update_family,
                label=anchor_state,
                source_cycle=source_cycle,
                source_task=source_task,
                outcome_tag=outcome_tag,
                reward_value=reward_value,
                content=anchor_state,
            )
            state_id = _ensure_step_node(
                updated,
                family=update_family,
                label=state,
                source_cycle=source_cycle,
                source_task=source_task,
                outcome_tag=outcome_tag,
                reward_value=reward_value,
                content=state,
            )

            if not _has_edge(edges, source=family_node_id, target=anchor_id, edge_type="step"):
                edges.append({"source": family_node_id, "target": anchor_id, "type": "step"})
            if not _has_edge(edges, source=family_node_id, target=state_id, edge_type="step"):
                edges.append({"source": family_node_id, "target": state_id, "type": "step"})

            if op == "insert_before":
                source_id, target_id = state_id, anchor_id
            else:
                source_id, target_id = anchor_id, state_id

            if not _has_edge(
                edges,
                source=source_id,
                target=target_id,
                edge_type=op,
                condition=condition,
            ):
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "type": op,
                        "family": update_family,
                        "state": state,
                        "anchor_state": anchor_state,
                        "condition": condition,
                        "retrieval_text": build_graph_retrieval_text(
                            {
                                "family": update_family,
                                "op": op,
                                "state": state,
                                "anchor_state": anchor_state,
                                "condition": condition,
                            }
                        ),
                    }
                )
            updated["edges"] = edges
            applied_packet_update = True

    if not applied_packet_update:
        if packet_present:
            return updated
        families = list(updated.get("families", []))
        if family not in families:
            families.append(family)
            updated["families"] = families

        updated, family_nodes = _ensure_family_layers(updated)
        family_node_id = family_nodes.get(family, f"{updated.get('root_domain', 'domain')}:{family}")

        label = _clip_words(posterior_text, 16)
        if meta_sop_text:
            label = f"{label} | { _clip_words(meta_sop_text, 6) }"
        node_id = _ensure_step_node(
            updated,
            family=family,
            label=label,
            source_cycle=source_cycle,
            source_task=source_task,
            outcome_tag=outcome_tag,
            reward_value=reward_value,
            content=posterior_text,
        )
        edges = list(updated.get("edges", []))
        if not _has_edge(edges, source=family_node_id, target=node_id, edge_type="step"):
            edges.append({"source": family_node_id, "target": node_id, "type": "step"})
        updated["edges"] = edges

    return updated


def summarize_state_graph(graph: dict[str, Any], limit_per_family: int = 2) -> str:
    families = graph.get("families") or []
    nodes = graph.get("nodes") or []
    step_nodes = [node for node in nodes if node.get("type") == "step"]
    if not step_nodes:
        return "State graph: (empty)"

    lines = ["State graph:"]
    structured_edges = [
        edge
        for edge in (graph.get("edges") or [])
        if edge.get("type") in {"insert_before", "insert_after", "strengthen_edge"}
        and _normalize_text(str(edge.get("state", "")))
        and _normalize_text(str(edge.get("anchor_state", "")))
    ]
    for family in families:
        family_edges = [edge for edge in structured_edges if edge.get("family") == family]
        if family_edges:
            lines.append(f"- {family}:")
            for edge in family_edges[:limit_per_family]:
                lines.append(
                    f"  - {edge.get('state', '').strip()} | {edge.get('type', '').strip()} {edge.get('anchor_state', '').strip()}"
                )
                if _normalize_text(str(edge.get("condition", ""))):
                    lines.append(f"    when: {edge.get('condition', '').strip()}")
            continue
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
