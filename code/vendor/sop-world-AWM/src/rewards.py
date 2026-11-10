from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from evolve_state import DOMAIN_TASK_FAMILIES, FAMILY_KEYWORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts" / "evolution"
EXTRACTOR_MAX_NEW_TOKENS = 4096
EXTRACTOR_REPAIR_MAX_NEW_TOKENS = 2048
_PROMPT_CACHE: dict[str, str] = {}

ExtractorGenerateFn = Callable[..., Any]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clip_words(text: str, limit: int) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _truncate_debug_text(text: str, limit: int = 1200) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _extractor_debug_enabled() -> bool:
    return _env_flag_enabled("EXTRACTOR_DEBUG_IO") or _env_flag_enabled(
        "TAU2_EXTRACTOR_DEBUG_IO"
    )


def _render_debug_messages(messages: list[dict[str, str]]) -> str:
    sections: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "message") or "message").strip()
        content = str(message.get("content", "") or "")
        sections.append(f"[{index}] role={role}\n{content}")
    return "\n\n".join(sections).strip()


def _emit_extractor_debug_block(
    label: str,
    *,
    sample_id: str,
    outcome_tag: str,
    content: str,
) -> None:
    print(
        "\n".join(
            [
                f"[extractor-debug] {label}_BEGIN sample_id={sample_id} outcome={outcome_tag}",
                str(content or ""),
                f"[extractor-debug] {label}_END sample_id={sample_id} outcome={outcome_tag}",
            ]
        ),
        flush=True,
    )


def _extract_json_text(text: str) -> Optional[str]:
    stripped = (text or "").strip()
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
        raise ValueError("No JSON object found in extractor output")
    return json.loads(json_text)


def _load_prompt_template(filename: str) -> str:
    cached = _PROMPT_CACHE.get(filename)
    if cached is not None:
        return cached
    path = PROMPTS_ROOT / filename
    text = path.read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[filename] = text
    return text


def render_trajectory_text(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or getattr(message, "type", None) or "message"
        content = getattr(message, "content", "")
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        content_text = _normalize_text(str(content))
        if content_text:
            lines.append(f"{role}: {content_text}")
    return "\n".join(lines)


def build_outcome_tag(reward_value: Optional[float], termination_reason: str | None) -> str:
    if reward_value is not None and abs(float(reward_value) - 1.0) <= 1e-6:
        return "success"

    normalized_reason = (termination_reason or "").upper()
    if "INFRA" in normalized_reason:
        return "infra_error"
    return "failure"


def _normalize_warnings(warnings: list[Any] | None) -> list[str]:
    if not warnings:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        text = _normalize_text(str(warning))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized[:12]


def _family_choices(domain: str | None, family_prior: str) -> list[str]:
    normalized_domain = (domain or "").strip().lower()
    families = list(DOMAIN_TASK_FAMILIES.get(normalized_domain, []))
    if family_prior and family_prior not in families:
        families.append(family_prior)
    return families or [family_prior or "default"]


def _task_query_text(task_query: str | None, trajectory_text: str) -> str:
    normalized = _normalize_text(task_query or "")
    if normalized:
        return _clip_words(normalized, 80)

    for line in trajectory_text.splitlines():
        if line.lower().startswith("user:"):
            return _clip_words(line.split(":", 1)[1], 80)
    return "(not available)"


def _warnings_block(warnings: list[str]) -> str:
    if not warnings:
        return "- (none)"
    return "\n".join(f"- {warning}" for warning in warnings)


def _trajectory_lines(trajectory_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in str(trajectory_text or "").splitlines():
        normalized = _normalize_text(raw_line)
        if normalized:
            lines.append(normalized)
    return lines


def _looks_salient_trajectory_line(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith("tool:"):
        return True
    return any(
        token in lowered
        for token in (
            "not found",
            "failed",
            "failure",
            "error",
            "timeout",
            "cannot",
            "can't",
            "unable",
            "invalid",
            "warning",
            "delivered",
            "pending",
            "refund",
            "cancel",
            "exchange",
            "modify",
            "return",
            "variant",
            "reservation",
            "booking",
        )
    )


def _build_evidence_bundle(trajectory_text: str) -> str:
    lines = _trajectory_lines(trajectory_text)
    if not lines:
        return "(empty)"

    selected: list[str] = []
    seen: set[str] = set()

    def _add(line: str) -> None:
        key = line.lower()
        if not line or key in seen:
            return
        seen.add(key)
        selected.append(line)

    user_lines = [line for line in lines if line.lower().startswith("user:")]
    assistant_lines = [line for line in lines if line.lower().startswith("assistant:")]
    salient_lines = [line for line in lines if _looks_salient_trajectory_line(line)]
    final_lines = lines[-8:]

    if user_lines:
        _add(user_lines[0])
    if assistant_lines:
        _add(assistant_lines[0])
    for line in salient_lines[:8]:
        _add(line)
    for line in final_lines:
        _add(line)

    if len(selected) < min(6, len(lines)):
        for line in lines[:6]:
            _add(line)
            if len(selected) >= min(6, len(lines)):
                break

    bundle_lines = selected[:16]
    return "\n".join(bundle_lines)


def _build_extractor_messages(
    *,
    outcome_tag: str,
    task_query: str,
    trajectory_text: str,
    termination_reason: str | None,
    family_prior: str,
    warnings: list[str],
    meta_sop_text: str,
    domain: str | None,
) -> list[dict[str, str]]:
    prompt_name = (
        "experience_success_extractor.md" if outcome_tag == "success" else "experience_failure_extractor.md"
    )
    family_options = ", ".join(_family_choices(domain, family_prior))
    system_prompt = _load_prompt_template(prompt_name)
    evidence_bundle = _build_evidence_bundle(trajectory_text)
    user_prompt = (
        "Return exactly one JSON object that matches the schema in the system instruction.\n"
        "The first non-whitespace character of your response must be '{' and the last non-whitespace character must be '}'.\n"
        "Do not output prose, markdown, code fences, analysis, a Thinking Process section, or tags such as <think>.\n"
        "Do not say things like 'Let me', 'Here is', or 'I will'.\n"
        "Do not emit ```json fences.\n"
        "Prefer concrete reusable lessons and workflow fixes whenever the trajectory supports them.\n\n"
        f"Task Query:\n{task_query}\n\n"
        f"Weak Family Prior:\n{family_prior or 'default'}\n\n"
        f"Allowed Family Labels:\n{family_options}\n\n"
        f"Termination Reason:\n{termination_reason or '(none)'}\n\n"
        f"Meta-SOP:\n{_clip_words(meta_sop_text, 120) or '(empty)'}\n\n"
        f"Runtime Warnings:\n{_warnings_block(warnings)}\n\n"
        f"Evidence Bundle:\n{evidence_bundle}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_repair_messages(raw_text: str, outcome_tag: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Convert the extractor output into exactly one valid JSON object.\n"
                "The first non-whitespace character of your response must be '{' and the last non-whitespace character must be '}'.\n"
                "Do not output analysis, markdown, code fences, explanations, a Thinking Process section, or tags such as <think>.\n"
                "Do not emit ```json fences.\n"
                "Discard narration and keep only the structured content.\n"
                "If some fields cannot be recovered, keep the schema valid by using an empty string or empty arrays.\n"
                "Return JSON only.\n\n"
                "Required schema:\n"
                "{\n"
                '  "family_focus": "string",\n'
                '  "warning_assessment": [\n'
                '    {\n'
                '      "warning_text": "string",\n'
                '      "category": "actionable_support | environment_limit | infra_noise",\n'
                '      "supports": "lookup_failure | missing_precondition | none"\n'
                "    }\n"
                "  ],\n"
                '  "memory_items": [\n'
                '    {\n'
                '      "title": "string",\n'
                '      "experience": "string",\n'
                '      "family": "string"\n'
                "    }\n"
                "  ],\n"
                '  "graph_updates": [\n'
                '    {\n'
                '      "op": "insert_before | insert_after | strengthen_edge",\n'
                '      "family": "string",\n'
                '      "state": "string",\n'
                '      "anchor_state": "string",\n'
                '      "condition": "string"\n'
                "    }\n"
                "  ],\n"
                '  "diagnostic_flags": ["string"]\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Outcome type: {outcome_tag}\n\n"
                f"Original extractor output:\n{raw_text}\n\n"
                "Return exactly one valid JSON object that matches the required schema.\n"
                "Start immediately with '{' and end with '}'."
            ),
        },
    ]


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
        repaired_text = str(result.get("repaired_text", "") or "").strip()
        used_repair = bool(result.get("used_repair", False))
        failure_stage = str(result.get("failure_stage", "") or "").strip()
        failure_reason = str(result.get("failure_reason", "") or "").strip()
        if payload is None:
            for candidate_text in (repaired_text, raw_text):
                if not candidate_text:
                    continue
                try:
                    payload = _safe_json_loads(candidate_text)
                    if candidate_text == repaired_text:
                        used_repair = True
                    failure_stage = ""
                    failure_reason = ""
                    break
                except Exception:
                    continue
        return {
            "payload": payload,
            "raw_text": raw_text,
            "repaired_text": repaired_text,
            "used_repair": used_repair,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        }

    return {
        "payload": None,
        "raw_text": str(result or "").strip(),
        "repaired_text": "",
        "used_repair": False,
        "failure_stage": "",
        "failure_reason": "",
    }


def _sanitize_warning_assessment(
    value: Any,
    warnings: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    warnings_by_key = {_normalize_text(item).lower(): item for item in warnings}
    normalized: list[dict[str, str]] = []
    diagnostic_flags: list[str] = []
    allowed_categories = {"actionable_support", "environment_limit", "infra_noise"}
    allowed_supports = {"lookup_failure", "missing_precondition", "none"}

    if not isinstance(value, list):
        value = []

    for item in value[: len(warnings)]:
        if not isinstance(item, dict):
            continue
        raw_warning_text = _normalize_text(str(item.get("warning_text", "")))
        warning_text = warnings_by_key.get(raw_warning_text.lower(), raw_warning_text)
        if not warning_text or warning_text.lower() not in warnings_by_key:
            continue
        category = _normalize_text(str(item.get("category", ""))).lower()
        supports = _normalize_text(str(item.get("supports", ""))).lower()
        if category not in allowed_categories:
            category = "infra_noise"
        if supports not in allowed_supports:
            supports = "none"
        normalized.append(
            {
                "warning_text": warning_text,
                "category": category,
                "supports": supports,
            }
        )

    if not normalized:
        for warning in warnings:
            normalized.append(
                {
                    "warning_text": warning,
                    "category": "infra_noise",
                    "supports": "none",
                }
            )

    if any(item["category"] == "environment_limit" for item in normalized):
        diagnostic_flags.append("environment_limit_detected")
    if any(item["category"] == "infra_noise" for item in normalized):
        diagnostic_flags.append("infra_noise_detected")
    return normalized, diagnostic_flags


def _sanitize_memory_items(
    value: Any,
    *,
    family_focus: str,
    max_items: int,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        title = _normalize_text(str(item.get("title", "")))
        experience = _normalize_text(str(item.get("experience", "")))
        family = _normalize_text(str(item.get("family", family_focus))) or family_focus
        if not title or not experience:
            continue
        key = (family, title.lower(), experience.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "title": title,
                "experience": experience,
                "family": family,
            }
        )
        if len(normalized) >= max_items:
            break
    return normalized


def _sanitize_graph_updates(
    value: Any,
    *,
    family_focus: str,
    max_items: int,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    allowed_ops = {"insert_before", "insert_after", "strengthen_edge"}
    for item in value:
        if not isinstance(item, dict):
            continue
        op = _normalize_text(str(item.get("op", ""))).lower()
        family = _normalize_text(str(item.get("family", family_focus))) or family_focus
        anchor_state = _normalize_text(str(item.get("anchor_state", "")))
        state = _normalize_text(str(item.get("state", "")))
        condition = _normalize_text(str(item.get("condition", "")))
        if op not in allowed_ops or not anchor_state or not state:
            continue
        key = (family, op, anchor_state.lower(), state.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "op": op,
                "family": family,
                "anchor_state": anchor_state,
                "state": state,
                "condition": condition,
            }
        )
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_experience_packet(
    payload: Any,
    *,
    outcome_tag: str,
    family_prior: str,
    warnings: list[str],
    termination_reason: str | None,
    domain: str | None,
    has_trajectory_evidence: bool,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    families = set(_family_choices(domain, family_prior))
    family_focus = _normalize_text(str(payload.get("family_focus", ""))) or family_prior
    diagnostic_flags = [
        _normalize_text(str(flag)).lower()
        for flag in payload.get("diagnostic_flags", [])
        if _normalize_text(str(flag))
    ]
    if family_focus not in families:
        family_focus = family_prior
        if "family_focus_fallback" not in diagnostic_flags:
            diagnostic_flags.append("family_focus_fallback")

    warning_assessment, warning_flags = _sanitize_warning_assessment(
        payload.get("warning_assessment", []),
        warnings,
    )
    for flag in warning_flags:
        if flag not in diagnostic_flags:
            diagnostic_flags.append(flag)

    max_memory_items = 0 if outcome_tag == "success" else 3
    max_graph_updates = 1 if outcome_tag == "success" else 2
    memory_items = _sanitize_memory_items(
        payload.get("memory_items", []),
        family_focus=family_focus,
        max_items=max_memory_items,
    )
    graph_updates = _sanitize_graph_updates(
        payload.get("graph_updates", []),
        family_focus=family_focus,
        max_items=max_graph_updates,
    )

    if outcome_tag == "infra_error":
        memory_items = []
        graph_updates = []
        if "infra_error" not in diagnostic_flags:
            diagnostic_flags.append("infra_error")

    if not has_trajectory_evidence:
        memory_items = []
        graph_updates = []
        if any(item.get("category") == "actionable_support" for item in warning_assessment):
            if "warning_without_trajectory_evidence" not in diagnostic_flags:
                diagnostic_flags.append("warning_without_trajectory_evidence")

    return {
        "case_type": outcome_tag,
        "family_prior": family_prior,
        "family_focus": family_focus,
        "warning_assessment": warning_assessment,
        "memory_items": memory_items,
        "graph_updates": graph_updates,
        "diagnostic_flags": diagnostic_flags,
        "termination_reason": termination_reason or "",
    }


def _prompt_driven_experience_packet(
    *,
    sample_id: str,
    outcome_tag: str,
    task_query: str,
    trajectory_text: str,
    termination_reason: str | None,
    task_family: str | None,
    warnings: list[str],
    domain: str | None,
    meta_sop_text: str,
    extractor_generate: ExtractorGenerateFn,
) -> dict[str, Any]:
    family_prior = task_family or "default"
    messages = _build_extractor_messages(
        outcome_tag=outcome_tag,
        task_query=task_query,
        trajectory_text=trajectory_text,
        termination_reason=termination_reason,
        family_prior=family_prior,
        warnings=warnings,
        meta_sop_text=meta_sop_text,
        domain=domain,
    )
    if _extractor_debug_enabled():
        _emit_extractor_debug_block(
            "PROMPT",
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            content=_render_debug_messages(messages),
        )

    initial_result = extractor_generate(
        messages=messages,
        call_name=f"experience_{outcome_tag}",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=EXTRACTOR_MAX_NEW_TOKENS,
    )
    extractor_result = _normalize_extractor_result(initial_result)
    if _extractor_debug_enabled():
        _emit_extractor_debug_block(
            "RAW_OUTPUT",
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            content=str(extractor_result.get("raw_text", "") or ""),
        )

    if extractor_result["payload"] is None and not isinstance(initial_result, dict):
        raw_text = extractor_result["raw_text"]
        try:
            extractor_result["payload"] = _safe_json_loads(raw_text)
        except Exception as exc:
            extractor_result["failure_stage"] = "raw_parse"
            extractor_result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            repaired_result = extractor_generate(
                messages=_build_repair_messages(raw_text, outcome_tag),
                call_name=f"experience_{outcome_tag}_repair",
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=EXTRACTOR_REPAIR_MAX_NEW_TOKENS,
            )
            repaired_text = str(repaired_result or "").strip()
            extractor_result["repaired_text"] = repaired_text
            extractor_result["used_repair"] = True
            if _extractor_debug_enabled():
                _emit_extractor_debug_block(
                    "REPAIRED_OUTPUT",
                    sample_id=sample_id,
                    outcome_tag=outcome_tag,
                    content=repaired_text,
                )
            try:
                extractor_result["payload"] = _safe_json_loads(repaired_text)
                extractor_result["failure_stage"] = ""
                extractor_result["failure_reason"] = ""
            except Exception as repair_exc:
                extractor_result["failure_stage"] = "repair_parse"
                extractor_result["failure_reason"] = (
                    f"{type(repair_exc).__name__}: {repair_exc}"
                )

    extractor_result["packet"] = _normalize_experience_packet(
        extractor_result.get("payload"),
        outcome_tag=outcome_tag,
        family_prior=family_prior,
        warnings=warnings,
        termination_reason=termination_reason,
        domain=domain,
        has_trajectory_evidence=bool(_normalize_text(trajectory_text)),
    )
    return extractor_result


def _packet_has_actionable_warning(packet: dict[str, Any]) -> bool:
    warning_assessment = packet.get("warning_assessment")
    if not isinstance(warning_assessment, list):
        return False
    return any(
        isinstance(item, dict) and item.get("category") == "actionable_support"
        for item in warning_assessment
    )


def _packet_update_enabled(packet: dict[str, Any]) -> bool:
    if _normalize_text(str(packet.get("extractor_source", ""))).lower() != "prompt":
        return False

    termination_reason = _normalize_text(str(packet.get("termination_reason", ""))).upper()
    if "TIMEOUT" in termination_reason or "INFRA" in termination_reason:
        return False

    if _normalize_text(str(packet.get("case_type", ""))).lower() == "infra_error":
        return False

    diagnostic_flags = {
        _normalize_text(str(flag)).lower()
        for flag in packet.get("diagnostic_flags", [])
        if _normalize_text(str(flag))
    }
    if "extractor_fallback_used" in diagnostic_flags:
        return False
    if "warning_without_trajectory_evidence" in diagnostic_flags:
        return False

    actionable_warning = _packet_has_actionable_warning(packet)
    if any(
        flag in diagnostic_flags
        for flag in ("infra_noise_detected", "environment_limit_detected")
    ) and not actionable_warning:
        return False

    memory_items = packet.get("memory_items")
    graph_updates = packet.get("graph_updates")
    return bool(memory_items or graph_updates)


def _fallback_classify_warning(warning_text: str) -> tuple[str, str]:
    text = warning_text.lower()
    if any(
        token in text
        for token in (
            "not implemented",
            "unsupported",
            "environment limit",
            "evaluator limit",
            "seats release not implemented",
        )
    ):
        return "environment_limit", "none"
    if any(
        token in text
        for token in (
            "simulation loop exited",
            "traceback",
            "exception",
            "infra",
            "oom",
            "connection reset",
            "timeout",
        )
    ):
        return "infra_noise", "none"
    if any(
        token in text
        for token in (
            "not found",
            "missing",
            "invalid id",
            "unknown id",
            "identifier",
        )
    ):
        if "not found" in text:
            return "actionable_support", "lookup_failure"
        return "actionable_support", "missing_precondition"
    return "infra_noise", "none"


def _fallback_infer_family_focus(
    *,
    domain: str | None,
    family_prior: str,
    trajectory_text: str,
    warnings_assessment: list[dict[str, str]],
) -> tuple[str, bool]:
    normalized_domain = (domain or "").strip().lower()
    family_map = FAMILY_KEYWORDS.get(normalized_domain, {})
    if not family_map:
        return family_prior, True

    text = trajectory_text.lower()
    warning_text = " ".join(item.get("warning_text", "").lower() for item in warnings_assessment)
    scores: dict[str, int] = {}
    for family, keywords in family_map.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 2
            if keyword in warning_text:
                score += 1
        scores[family] = score

    best_family = max(scores, key=scores.get) if scores else family_prior
    best_score = scores.get(best_family, 0)
    if best_score <= 0:
        return family_prior, True
    return best_family, False


def _fallback_failure_memory_items(
    *,
    family_focus: str,
    trajectory_text: str,
    warnings_assessment: list[dict[str, str]],
) -> list[dict[str, str]]:
    text = trajectory_text.lower()
    items: list[dict[str, str]] = []

    has_lookup_evidence = any(
        token in text
        for token in (
            "not found",
            "lookup",
            "identifier",
            "verify",
            "validation",
            "email",
            "order",
            "product",
            "user",
        )
    )
    actionable_lookup = any(
        item.get("category") == "actionable_support"
        and item.get("supports") in {"lookup_failure", "missing_precondition"}
        for item in warnings_assessment
    )

    if has_lookup_evidence or actionable_lookup:
        items.append(
            {
                "title": "Resolve identity before detail lookup",
                "experience": (
                    "When entity lookup fails or the identifier is incomplete, "
                    "validate identity with fallback attributes before querying detail records."
                ),
                "family": family_focus,
            }
        )

    asks_detail_early = any(token in text for token in ("detail", "details", "book", "cancel", "modify"))
    lacks_validation = not any(token in text for token in ("verify", "validated", "confirm", "precondition"))
    if asks_detail_early and lacks_validation:
        items.append(
            {
                "title": "Gate actions with precondition checks",
                "experience": (
                    "Before detail or action steps, confirm required entities and constraints "
                    "instead of moving directly to an irreversible operation."
                ),
                "family": family_focus,
            }
        )

    if not items and _normalize_text(trajectory_text):
        items.append(
            {
                "title": "Patch missing workflow step",
                "experience": (
                    "When the task fails despite valid user intent, identify and insert the "
                    "missing prerequisite step before repeating the workflow."
                ),
                "family": family_focus,
            }
        )
    return items[:3]


def _fallback_failure_graph_updates(
    *,
    family_focus: str,
    trajectory_text: str,
    warnings_assessment: list[dict[str, str]],
) -> list[dict[str, str]]:
    text = trajectory_text.lower()
    updates: list[dict[str, str]] = []
    actionable_lookup = any(
        item.get("category") == "actionable_support"
        and item.get("supports") in {"lookup_failure", "missing_precondition"}
        for item in warnings_assessment
    )
    has_lookup_evidence = actionable_lookup or any(
        token in text for token in ("not found", "identifier", "lookup", "verify")
    )
    if has_lookup_evidence:
        updates.append(
            {
                "op": "insert_before",
                "family": family_focus,
                "anchor_state": "get_details",
                "state": "resolve_identity",
                "condition": "identifier comes from user text or lookup has not been validated",
            }
        )

    asks_detail_early = any(token in text for token in ("detail", "details", "book", "cancel", "modify"))
    lacks_validation = not any(token in text for token in ("verify", "validated", "confirm"))
    if asks_detail_early and lacks_validation:
        updates.append(
            {
                "op": "insert_before",
                "family": family_focus,
                "anchor_state": "execute_action",
                "state": "validate_preconditions",
                "condition": "about to call detail/action operation",
            }
        )

    if not updates and _normalize_text(trajectory_text):
        updates.append(
            {
                "op": "strengthen_edge",
                "family": family_focus,
                "anchor_state": "collect_requirements",
                "state": "execute_action",
                "condition": "requirements are complete and validated",
            }
        )
    return updates[:2]


def _fallback_success_memory_items(
    *,
    family_focus: str,
    trajectory_text: str,
) -> list[dict[str, str]]:
    text = trajectory_text.lower()
    items: list[dict[str, str]] = []
    if any(token in text for token in ("verify", "validated", "confirm")):
        items.append(
            {
                "title": "Keep validation-first flow",
                "experience": (
                    "When the task reaches an action stage, confirm key identifiers and "
                    "constraints before the final action instead of assuming the path is obvious."
                ),
                "family": family_focus,
            }
        )
    if not items and _normalize_text(trajectory_text):
        items.append(
            {
                "title": "Reinforce stable successful flow",
                "experience": (
                    "For similar requests in the same family, reuse the concise step order that "
                    "already led to a successful resolution."
                ),
                "family": family_focus,
            }
        )
    return items[:2]


def _fallback_success_graph_updates(
    *,
    family_focus: str,
    trajectory_text: str,
) -> list[dict[str, str]]:
    text = trajectory_text.lower()
    if any(token in text for token in ("verify", "validated", "confirm")) and any(
        token in text for token in ("detail", "book", "cancel", "modify")
    ):
        return [
            {
                "op": "strengthen_edge",
                "family": family_focus,
                "anchor_state": "validate_preconditions",
                "state": "execute_action",
                "condition": "validation completed",
            }
        ]
    return []


def _fallback_experience_packet(
    *,
    outcome_tag: str,
    trajectory_text: str,
    termination_reason: str | None,
    task_family: str | None,
    warnings: list[str],
    domain: str | None,
) -> dict[str, Any]:
    family_prior = task_family or "default"
    warning_assessment: list[dict[str, str]] = []
    diagnostic_flags: list[str] = []

    for warning_text in warnings:
        category, supports = _fallback_classify_warning(warning_text)
        warning_assessment.append(
            {
                "warning_text": warning_text,
                "category": category,
                "supports": supports,
            }
        )
        if category == "environment_limit" and "environment_limit_detected" not in diagnostic_flags:
            diagnostic_flags.append("environment_limit_detected")
        if category == "infra_noise" and "infra_noise_detected" not in diagnostic_flags:
            diagnostic_flags.append("infra_noise_detected")

    family_focus, used_family_fallback = _fallback_infer_family_focus(
        domain=domain,
        family_prior=family_prior,
        trajectory_text=trajectory_text,
        warnings_assessment=warning_assessment,
    )
    if used_family_fallback:
        diagnostic_flags.append("family_focus_fallback")

    has_trajectory_evidence = bool(_normalize_text(trajectory_text))
    memory_items: list[dict[str, str]] = []
    graph_updates: list[dict[str, str]] = []

    if outcome_tag == "infra_error":
        diagnostic_flags.append("infra_error")
    elif has_trajectory_evidence:
        if outcome_tag == "failure":
            memory_items = _fallback_failure_memory_items(
                family_focus=family_focus,
                trajectory_text=trajectory_text,
                warnings_assessment=warning_assessment,
            )
            graph_updates = _fallback_failure_graph_updates(
                family_focus=family_focus,
                trajectory_text=trajectory_text,
                warnings_assessment=warning_assessment,
            )
        else:
            graph_updates = _fallback_success_graph_updates(
                family_focus=family_focus,
                trajectory_text=trajectory_text,
            )
    elif any(item.get("category") == "actionable_support" for item in warning_assessment):
        diagnostic_flags.append("warning_without_trajectory_evidence")

    return {
        "case_type": outcome_tag,
        "family_prior": family_prior,
        "family_focus": family_focus,
        "warning_assessment": warning_assessment,
        "memory_items": memory_items,
        "graph_updates": graph_updates,
        "diagnostic_flags": diagnostic_flags,
        "termination_reason": termination_reason or "",
    }


def build_experience_packet(
    *,
    sample_id: str,
    outcome_tag: str,
    trajectory_text: str,
    termination_reason: str | None,
    task_family: str | None,
    warnings: list[Any] | None,
    domain: str | None,
    task_query: str = "",
    meta_sop_text: str = "",
    extractor_generate: ExtractorGenerateFn | None = None,
) -> dict[str, Any]:
    normalized_warnings = _normalize_warnings(warnings)
    prompt_attempt: dict[str, Any] | None = None
    if (
        extractor_generate is not None
        and outcome_tag != "infra_error"
        and _normalize_text(trajectory_text)
    ):
        prompt_attempt = _prompt_driven_experience_packet(
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            task_query=_task_query_text(task_query, trajectory_text),
            trajectory_text=trajectory_text,
            termination_reason=termination_reason,
            task_family=task_family,
            warnings=normalized_warnings,
            domain=domain,
            meta_sop_text=meta_sop_text,
            extractor_generate=extractor_generate,
        )
        packet = prompt_attempt.get("packet")
        if isinstance(packet, dict):
            packet = dict(packet)
            packet["extractor_source"] = "prompt"
            packet["extractor_used_repair"] = bool(prompt_attempt.get("used_repair", False))
            packet["extractor_failure_stage"] = str(
                prompt_attempt.get("failure_stage", "") or ""
            )
            packet["extractor_failure_reason"] = str(
                prompt_attempt.get("failure_reason", "") or ""
            )
            packet["extractor_raw_text"] = _truncate_debug_text(
                str(prompt_attempt.get("raw_text", "") or "")
            )
            packet["extractor_repaired_text"] = _truncate_debug_text(
                str(prompt_attempt.get("repaired_text", "") or "")
            )
            packet["update_enabled"] = _packet_update_enabled(packet)
            return packet

    packet = _fallback_experience_packet(
        outcome_tag=outcome_tag,
        trajectory_text=trajectory_text,
        termination_reason=termination_reason,
        task_family=task_family,
        warnings=normalized_warnings,
        domain=domain,
    )
    if extractor_generate is not None and "extractor_fallback_used" not in packet["diagnostic_flags"]:
        packet["diagnostic_flags"].append("extractor_fallback_used")
    packet["extractor_source"] = "fallback"
    packet["extractor_used_repair"] = bool(
        prompt_attempt and prompt_attempt.get("used_repair", False)
    )
    packet["extractor_failure_stage"] = str(
        (prompt_attempt or {}).get("failure_stage", "") or ""
    )
    packet["extractor_failure_reason"] = str(
        (prompt_attempt or {}).get("failure_reason", "") or ""
    )
    packet["extractor_raw_text"] = _truncate_debug_text(
        str((prompt_attempt or {}).get("raw_text", "") or "")
    )
    packet["extractor_repaired_text"] = _truncate_debug_text(
        str((prompt_attempt or {}).get("repaired_text", "") or "")
    )
    packet["update_enabled"] = False
    return packet


def build_posterior_text(
    *,
    outcome_tag: str,
    reward_value: Optional[float],
    meta_sop_text: str,
    trajectory_text: str,
    task_family: str | None = None,
    experience_packet: dict[str, Any] | None = None,
) -> str:
    if isinstance(experience_packet, dict):
        memory_items = experience_packet.get("memory_items")
        if isinstance(memory_items, list) and memory_items:
            first_item = memory_items[0]
            if isinstance(first_item, dict):
                title = _normalize_text(str(first_item.get("title", "")))
                experience = _normalize_text(str(first_item.get("experience", "")))
                text = ": ".join(part for part in (title, experience) if part)
                if text:
                    return _clip_words(text, 40)

        graph_updates = experience_packet.get("graph_updates")
        if isinstance(graph_updates, list) and graph_updates:
            first_update = graph_updates[0]
            if isinstance(first_update, dict):
                op = _normalize_text(str(first_update.get("op", "")))
                anchor_state = _normalize_text(str(first_update.get("anchor_state", "")))
                state = _normalize_text(str(first_update.get("state", "")))
                condition = _normalize_text(str(first_update.get("condition", "")))
                if op and anchor_state and state:
                    text = f"{op} {state} around {anchor_state}"
                    if condition:
                        text = f"{text} when {condition}"
                    return _clip_words(text, 40)

    family_text = task_family or "task"
    meta_hint = _clip_words(meta_sop_text, 8)
    if outcome_tag == "success":
        text = f"Reinforce the short {family_text} rule."
    elif outcome_tag == "infra_error":
        text = "Treat this as infrastructure noise; keep task logic unchanged."
    else:
        text = f"Add the missing {family_text} step and keep it short."
    if meta_hint:
        text = f"{text} Follow: {meta_hint}"
    return _clip_words(text, 40)


def build_reward_signal(
    *,
    sample_id: str,
    trajectory_text: str,
    reward_value: Optional[float],
    termination_reason: str | None,
    meta_sop_text: str,
    task_family: str | None = None,
    task_id: str | None = None,
    warnings: list[Any] | None = None,
    domain: str | None = None,
    task_query: str = "",
    extractor_generate: ExtractorGenerateFn | None = None,
) -> dict[str, Any]:
    outcome_tag = build_outcome_tag(reward_value, termination_reason)
    experience_packet = build_experience_packet(
        sample_id=sample_id,
        outcome_tag=outcome_tag,
        trajectory_text=trajectory_text,
        termination_reason=termination_reason,
        task_family=task_family,
        warnings=warnings,
        domain=domain,
        task_query=task_query,
        meta_sop_text=meta_sop_text,
        extractor_generate=extractor_generate,
    )
    posterior_text = build_posterior_text(
        outcome_tag=outcome_tag,
        reward_value=reward_value,
        meta_sop_text=meta_sop_text,
        trajectory_text=trajectory_text,
        task_family=task_family,
        experience_packet=experience_packet,
    )
    experience_packet["posterior_text"] = posterior_text
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "task_family": task_family,
        "reward_value": reward_value,
        "termination_reason": termination_reason,
        "outcome_tag": outcome_tag,
        "posterior_text": posterior_text,
        "experience_packet": experience_packet,
        "trajectory_text": _clip_words(trajectory_text, 80),
    }
