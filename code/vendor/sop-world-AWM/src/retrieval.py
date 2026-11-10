from __future__ import annotations

import hashlib
import math
from typing import Any, Callable


EmbedTextsFn = Callable[[list[str]], list[list[float]]]

_EMBED_CACHE: dict[tuple[str, str, str], list[float]] = {}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def build_memory_retrieval_text(item: dict[str, Any]) -> str:
    title = _normalize_text(str(item.get("title", "")))
    experience = _normalize_text(str(item.get("experience", "")))
    sections: list[str] = []
    if title:
        sections.append(f"Title: {title}")
    if experience:
        sections.append(f"Experience: {experience}")
    return "\n".join(sections).strip()


def build_graph_retrieval_text(item: dict[str, Any]) -> str:
    state = _normalize_text(str(item.get("state", "")))
    anchor_state = _normalize_text(str(item.get("anchor_state", "")))
    op = _normalize_text(str(item.get("op", "")))
    condition = _normalize_text(str(item.get("condition", "")))
    sections: list[str] = []
    if state:
        sections.append(f"State: {state}")
    if op and anchor_state:
        sections.append(f"Placement: {op} {anchor_state}")
    elif anchor_state:
        sections.append(f"Anchor State: {anchor_state}")
    if condition:
        sections.append(f"Condition: {condition}")
    return "\n".join(sections).strip()


def build_workflow_retrieval_text(item: dict[str, Any]) -> str:
    title = _normalize_text(str(item.get("title", "")))
    applicability = _normalize_text(str(item.get("applicability", "")))
    task_signature = _normalize_text(str(item.get("task_signature", "")))
    workflow = [
        _normalize_text(str(step))
        for step in item.get("workflow", [])
        if _normalize_text(str(step))
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


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return -1.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def _rank_records(
    records: list[dict[str, Any]],
    *,
    query_text: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query_text)
    if not normalized_query or top_k <= 0 or not records:
        return []
    payloads = [normalized_query, *[str(record.get("retrieval_text", "") or "") for record in records]]
    vectors = embed_texts(payloads)
    if len(vectors) != len(payloads):
        raise ValueError("embed_texts returned a mismatched number of vectors")
    query_vector = vectors[0]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        score = _cosine_similarity(query_vector, vectors[index + 1])
        scored.append((score, index, record))
    scored.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )
    return [record for _, _, record in scored[:top_k]]


def _retrieve_family_scoped(
    records: list[dict[str, Any]],
    *,
    query_text: str,
    family: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
) -> list[dict[str, Any]]:
    normalized_family = _normalize_text(family)
    if top_k <= 0:
        return []

    same_family = [
        record
        for record in records
        if _normalize_text(str(record.get("family", ""))) == normalized_family
    ]
    fallback_family = [
        record
        for record in records
        if _normalize_text(str(record.get("family", ""))) != normalized_family
    ]

    results = _rank_records(
        same_family,
        query_text=query_text,
        top_k=top_k,
        embed_texts=embed_texts,
    )
    remaining = top_k - len(results)
    if remaining > 0:
        selected_ids = {str(item.get("id", "")) for item in results}
        cross_family = [
            record
            for record in _rank_records(
                fallback_family,
                query_text=query_text,
                top_k=remaining,
                embed_texts=embed_texts,
            )
            if str(record.get("id", "")) not in selected_ids
        ]
        results.extend(cross_family[:remaining])
    return results[:top_k]


def retrieve_memory_entries(
    memory: dict[str, Any],
    *,
    query_text: str,
    family: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in memory.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        retrieval_text = str(entry.get("retrieval_text", "") or "")
        if not retrieval_text.strip():
            retrieval_text = build_memory_retrieval_text(entry)
        record = dict(entry)
        record["retrieval_text"] = retrieval_text
        records.append(record)
    return _retrieve_family_scoped(
        records,
        query_text=query_text,
        family=family,
        top_k=top_k,
        embed_texts=embed_texts,
    )


def retrieve_graph_updates(
    graph: dict[str, Any],
    *,
    query_text: str,
    family: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if str(edge.get("type", "")) not in {"insert_before", "insert_after", "strengthen_edge"}:
            continue
        state = _normalize_text(str(edge.get("state", "")))
        anchor_state = _normalize_text(str(edge.get("anchor_state", "")))
        if not state or not anchor_state:
            continue
        retrieval_text = str(edge.get("retrieval_text", "") or "")
        if not retrieval_text.strip():
            retrieval_text = build_graph_retrieval_text(edge)
        record = dict(edge)
        record["retrieval_text"] = retrieval_text
        records.append(record)
    return _retrieve_family_scoped(
        records,
        query_text=query_text,
        family=family,
        top_k=top_k,
        embed_texts=embed_texts,
    )


def retrieve_workflow_entries(
    workflow_memory: dict[str, Any],
    *,
    query_text: str,
    family: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in workflow_memory.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        retrieval_text = str(entry.get("retrieval_text", "") or "")
        if not retrieval_text.strip():
            retrieval_text = build_workflow_retrieval_text(entry)
        record = dict(entry)
        record["retrieval_text"] = retrieval_text
        records.append(record)

    active_records = [
        record for record in records if _normalize_text(str(record.get("status", ""))) == "active"
    ]
    candidate_records = [
        record for record in records if _normalize_text(str(record.get("status", ""))) != "active"
    ]

    results = _retrieve_family_scoped(
        active_records,
        query_text=query_text,
        family=family,
        top_k=top_k,
        embed_texts=embed_texts,
    )
    remaining = top_k - len(results)
    if remaining > 0:
        selected_ids = {str(item.get("id", "")) for item in results}
        for record in _retrieve_family_scoped(
            candidate_records,
            query_text=query_text,
            family=family,
            top_k=remaining,
            embed_texts=embed_texts,
        ):
            if str(record.get("id", "")) in selected_ids:
                continue
            results.append(record)
            if len(results) >= top_k:
                break
    return results[:top_k]


def render_memory_retrieval_section(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = ["Relevant memory:"]
    for entry in entries:
        title = _normalize_text(str(entry.get("title", "")))
        experience = _normalize_text(str(entry.get("experience", "") or entry.get("content", "")))
        family = _normalize_text(str(entry.get("family", "")))
        prefix = f"[{family}] " if family else ""
        if title and experience:
            lines.append(f"- {prefix}{title}: {experience}")
        elif experience:
            lines.append(f"- {prefix}{experience}")
    return "\n".join(lines)


def render_graph_retrieval_section(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["Relevant workflow states:"]
    for item in items:
        family = _normalize_text(str(item.get("family", "")))
        op = _normalize_text(str(item.get("op", "")))
        state = _normalize_text(str(item.get("state", "")))
        anchor_state = _normalize_text(str(item.get("anchor_state", "")))
        condition = _normalize_text(str(item.get("condition", "")))
        prefix = f"[{family}] " if family else ""
        base = f"- {prefix}state: {state}"
        if op and anchor_state:
            base = f"{base} | {op} {anchor_state}"
        lines.append(base)
        if condition:
            lines.append(f"  condition: {condition}")
    return "\n".join(lines)


def render_workflow_retrieval_section(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["Relevant successful workflows:"]
    for item in items:
        family = _normalize_text(str(item.get("family", "")))
        title = _normalize_text(str(item.get("title", "")))
        applicability = _normalize_text(str(item.get("applicability", "")))
        workflow = [
            _normalize_text(str(step))
            for step in item.get("workflow", [])
            if _normalize_text(str(step))
        ]
        prefix = f"[{family}] " if family else ""
        if title:
            lines.append(f"- {prefix}{title}")
        if applicability:
            lines.append(f"  applicability: {applicability}")
        if workflow:
            lines.append(f"  workflow: {' -> '.join(workflow)}")
    return "\n".join(lines)


def build_openai_compatible_embed_texts(
    *,
    model: str,
    api_key: str,
    base_url: str,
) -> EmbedTextsFn | None:
    normalized_model = _normalize_text(model)
    normalized_key = _normalize_text(api_key)
    normalized_base_url = _normalize_text(base_url)
    if not normalized_model or not normalized_key or not normalized_base_url:
        return None

    def _embed_texts(texts: list[str]) -> list[list[float]]:
        normalized_texts = [_normalize_text(text) for text in texts]
        results: list[list[float] | None] = [None] * len(normalized_texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for index, text in enumerate(normalized_texts):
            cache_key = (
                normalized_model,
                normalized_base_url,
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
            cached = _EMBED_CACHE.get(cache_key)
            if cached is not None:
                results[index] = list(cached)
            else:
                missing_texts.append(text)
                missing_indices.append(index)

        if missing_texts:
            from openai import OpenAI

            client = OpenAI(api_key=normalized_key, base_url=normalized_base_url)
            response = client.embeddings.create(
                model=normalized_model,
                input=missing_texts,
            )
            returned_vectors = [list(item.embedding) for item in response.data]
            if len(returned_vectors) != len(missing_texts):
                raise ValueError("Embedding API returned a mismatched number of vectors")
            for text, index, vector in zip(missing_texts, missing_indices, returned_vectors, strict=False):
                cache_key = (
                    normalized_model,
                    normalized_base_url,
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
                _EMBED_CACHE[cache_key] = list(vector)
                results[index] = list(vector)

        return [list(vector or []) for vector in results]

    return _embed_texts
