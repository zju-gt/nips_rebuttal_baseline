from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any, Callable


EmbedTextsFn = Callable[[list[str]], list[list[float]]]

_EMBED_CACHE: dict[tuple[str, str, str], list[float]] = {}
_LOGGER = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def build_memory_retrieval_text(item: dict[str, Any]) -> str:
    title = _normalize_text(str(item.get("title", "")))
    applicability = _normalize_text(str(item.get("applicability", "")))
    failure_class = _normalize_text(str(item.get("failure_class", "")))
    missing_step = _normalize_text(str(item.get("missing_step", "")))
    recovery_action = _normalize_text(str(item.get("recovery_action", "")))
    experience = _normalize_text(str(item.get("experience", "")))
    source_type = _normalize_text(str(item.get("source_type", "")))
    sections: list[str] = []
    if title:
        sections.append(f"Title: {title}")
    if applicability:
        sections.append(f"Applicability: {applicability}")
    if failure_class:
        sections.append(f"Failure Class: {failure_class}")
    if missing_step:
        sections.append(f"Missing Step: {missing_step}")
    if recovery_action:
        sections.append(f"Recovery Action: {recovery_action}")
    if experience:
        sections.append(f"Experience: {experience}")
    if source_type:
        sections.append(f"Source Type: {source_type}")
    return "\n".join(sections).strip()


def build_graph_retrieval_text(item: dict[str, Any]) -> str:
    state = _normalize_text(str(item.get("state", "")))
    anchor_state = _normalize_text(str(item.get("anchor_state", "")))
    op = _normalize_text(str(item.get("op", "")))
    condition = _normalize_text(str(item.get("condition", "")))
    status = _normalize_text(str(item.get("status", "")))
    precondition = _normalize_text(str(item.get("precondition", "")))
    commit_gate = _normalize_text(str(item.get("commit_gate", "")))
    postcondition = _normalize_text(str(item.get("postcondition", "")))
    critical_checks = [
        _normalize_text(value) for value in item.get("critical_checks", []) or [] if _normalize_text(value)
    ]
    sections: list[str] = []
    if state:
        sections.append(f"State: {state}")
    if op and anchor_state:
        sections.append(f"Placement: {op} {anchor_state}")
    elif anchor_state:
        sections.append(f"Anchor State: {anchor_state}")
    if condition:
        sections.append(f"Condition: {condition}")
    if status:
        sections.append(f"Status: {status}")
    if precondition:
        sections.append(f"Precondition: {precondition}")
    if commit_gate:
        sections.append(f"Commit Gate: {commit_gate}")
    if postcondition:
        sections.append(f"Postcondition: {postcondition}")
    if critical_checks:
        sections.append(f"Critical Checks: {' | '.join(critical_checks[:3])}")
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


def _lexical_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", _normalize_text(text).lower()) if token}


def _lexical_similarity(query_text: str, retrieval_text: str) -> float:
    query_terms = _lexical_terms(query_text)
    retrieval_terms = _lexical_terms(retrieval_text)
    if not query_terms or not retrieval_terms:
        return 0.0
    overlap = len(query_terms & retrieval_terms)
    if overlap <= 0:
        return 0.0
    # Prefer overlap count first, with a light normalization so longer texts do not dominate.
    return float(overlap) / math.sqrt(len(query_terms) * len(retrieval_terms))


def _rank_records_lexical(
    records: list[dict[str, Any]],
    *,
    query_text: str,
    top_k: int,
    priority_fn: Callable[[dict[str, Any]], tuple[float, ...]] | None = None,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query_text)
    if not normalized_query or top_k <= 0 or not records:
        return []
    scored: list[tuple[tuple[float, ...], float, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        retrieval_text = str(record.get("retrieval_text", "") or "")
        score = _lexical_similarity(normalized_query, retrieval_text)
        priority = priority_fn(record) if priority_fn is not None else tuple()
        scored.append((priority, score, index, record))
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            -item[2],
        ),
        reverse=True,
    )
    return [record for _, _, _, record in scored[:top_k]]


def _rank_records(
    records: list[dict[str, Any]],
    *,
    query_text: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
    priority_fn: Callable[[dict[str, Any]], tuple[float, ...]] | None = None,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query_text)
    if not normalized_query or top_k <= 0 or not records:
        return []
    payloads = [normalized_query, *[str(record.get("retrieval_text", "") or "") for record in records]]
    try:
        vectors = embed_texts(payloads)
    except Exception as exc:
        _LOGGER.warning(
            "Embedding retrieval failed; falling back to lexical ranking: %s: %s",
            type(exc).__name__,
            exc,
        )
        return _rank_records_lexical(
            records,
            query_text=normalized_query,
            top_k=top_k,
            priority_fn=priority_fn,
        )
    if len(vectors) != len(payloads):
        _LOGGER.warning(
            "Embedding retrieval returned %s vectors for %s texts; falling back to lexical ranking.",
            len(vectors),
            len(payloads),
        )
        return _rank_records_lexical(
            records,
            query_text=normalized_query,
            top_k=top_k,
            priority_fn=priority_fn,
        )
    query_vector = vectors[0]
    scored: list[tuple[tuple[float, ...], float, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        score = _cosine_similarity(query_vector, vectors[index + 1])
        priority = priority_fn(record) if priority_fn is not None else tuple()
        scored.append((priority, score, index, record))
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            -item[2],
        ),
        reverse=True,
    )
    return [record for _, _, _, record in scored[:top_k]]


def _retrieve_family_scoped(
    records: list[dict[str, Any]],
    *,
    query_text: str,
    family: str,
    top_k: int,
    embed_texts: EmbedTextsFn,
    priority_fn: Callable[[dict[str, Any]], tuple[float, ...]] | None = None,
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
        priority_fn=priority_fn,
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
                priority_fn=priority_fn,
            )
            if str(record.get("id", "")) not in selected_ids
        ]
        results.extend(cross_family[:remaining])
    return results[:top_k]


def _memory_priority(record: dict[str, Any]) -> tuple[float, ...]:
    source_type = _normalize_text(str(record.get("source_type", ""))).lower()
    priority = 2.0 if source_type == "failure" else 1.0 if source_type == "failure_informed_success" else 0.0
    confidence = float(record.get("confidence", 0.0) or 0.0)
    return (priority, confidence)


def _workflow_priority(record: dict[str, Any]) -> tuple[float, ...]:
    status = _normalize_text(str(record.get("status", ""))).lower()
    priority = 2.0 if status == "candidate" else 1.0 if status == "stable" else 0.0
    repair_origin = 1.0 if bool(record.get("repair_origin", False)) else 0.0
    confidence = float(record.get("confidence", 0.0) or 0.0)
    return (priority, repair_origin, confidence)


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
        priority_fn=_memory_priority,
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
        priority_fn=_workflow_priority,
    )


def render_memory_retrieval_section(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = ["Relevant memory:"]
    for entry in entries:
        title = _normalize_text(str(entry.get("title", "")))
        experience = _normalize_text(str(entry.get("experience", "") or entry.get("content", "")))
        family = _normalize_text(str(entry.get("family", "")))
        source_type = _normalize_text(str(entry.get("source_type", "")))
        prefix_parts = [part for part in (source_type, family) if part]
        prefix = f"[{'/'.join(prefix_parts)}] " if prefix_parts else ""
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
        status = _normalize_text(str(item.get("status", "")))
        op = _normalize_text(str(item.get("op", "")))
        state = _normalize_text(str(item.get("state", "")))
        anchor_state = _normalize_text(str(item.get("anchor_state", "")))
        condition = _normalize_text(str(item.get("condition", "")))
        prefix_parts = [part for part in (status, family) if part]
        prefix = f"[{'/'.join(prefix_parts)}] " if prefix_parts else ""
        base = f"- {prefix}state: {state}"
        if op and anchor_state:
            base = f"{base} | {op} {anchor_state}"
        lines.append(base)
        if condition:
            lines.append(f"  condition: {condition}")
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
    batch_size = max(1, int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "10") or "10"))

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
            from openai_test import OpenAI

            client = OpenAI(api_key=normalized_key, base_url=normalized_base_url)
            for start in range(0, len(missing_texts), batch_size):
                batch_texts = missing_texts[start : start + batch_size]
                batch_indices = missing_indices[start : start + batch_size]
                response = client.embeddings.create(
                    model=normalized_model,
                    input=batch_texts,
                )
                returned_vectors = [list(item.embedding) for item in response.data]
                if len(returned_vectors) != len(batch_texts):
                    raise ValueError("Embedding API returned a mismatched number of vectors")
                for text, index, vector in zip(
                    batch_texts,
                    batch_indices,
                    returned_vectors,
                    strict=False,
                ):
                    cache_key = (
                        normalized_model,
                        normalized_base_url,
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                    _EMBED_CACHE[cache_key] = list(vector)
                    results[index] = list(vector)

        return [list(vector or []) for vector in results]

    return _embed_texts
