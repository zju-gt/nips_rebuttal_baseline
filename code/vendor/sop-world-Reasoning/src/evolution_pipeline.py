from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from evolve_state import DOMAIN_TASK_FAMILIES, FAMILY_KEYWORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts" / "evolution"
_PROMPT_CACHE: dict[str, str] = {}

GenerateFn = Callable[..., Any]


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clip_words(text: str, limit: int) -> str:
    words = _normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "."


def _truncate_debug_text(text: str, limit: int = 2000) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _extract_json_text(text: str) -> Optional[str]:
    stripped = str(text or "").strip()
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
    text = (PROMPTS_ROOT / filename).read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[filename] = text
    return text


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _extractor_debug_enabled() -> bool:
    return _env_flag_enabled("EXTRACTOR_DEBUG_IO") or _env_flag_enabled(
        "TAU2_EXTRACTOR_DEBUG_IO"
    )


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


def _render_debug_messages(messages: list[dict[str, str]]) -> str:
    sections: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "message") or "message").strip()
        content = str(message.get("content", "") or "")
        sections.append(f"[{index}] role={role}\n{content}")
    return "\n\n".join(sections).strip()


def _normalize_warnings(warnings: list[Any] | None) -> list[str]:
    if not warnings:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        text = _normalize_text(warning)
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


def _warnings_block(warnings: list[str]) -> str:
    if not warnings:
        return "- (none)"
    return "\n".join(f"- {warning}" for warning in warnings)


def _reward_block(reward_breakdown: dict[str, Any] | None) -> str:
    if not isinstance(reward_breakdown, dict) or not reward_breakdown:
        return "- reward details not available"
    lines: list[str] = []
    for key, value in reward_breakdown.items():
        key_text = _normalize_text(key)
        if not key_text:
            continue
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = _normalize_text(value)
        if not value_text:
            continue
        lines.append(f"- {key_text}: {_clip_words(value_text, 40)}")
    return "\n".join(lines) if lines else "- reward details not available"


def _normalize_reflexion_summary(reflexion_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(reflexion_summary, dict):
        return {}
    memory_items = [
        _normalize_text(item)
        for item in reflexion_summary.get("memory", []) or []
        if _normalize_text(item)
    ][:5]
    attempts = [
        item
        for item in reflexion_summary.get("attempts", []) or []
        if isinstance(item, dict)
    ][:5]
    return {
        "enabled": bool(reflexion_summary.get("enabled")),
        "trials_used": int(reflexion_summary.get("trials_used", 0) or 0),
        "memory": memory_items,
        "attempts": attempts,
    }


def _reflexion_block(reflexion_summary: dict[str, Any] | None) -> str:
    summary = _normalize_reflexion_summary(reflexion_summary)
    if not summary.get("enabled"):
        return "- reflexion not used"
    lines = [f"- trials_used: {summary.get('trials_used', 0)}"]
    memory = summary.get("memory", [])
    if memory:
        lines.append("- reflexion_memory:")
        lines.extend(f"  - {item}" for item in memory[:3])
    attempts = summary.get("attempts", [])
    for attempt in attempts[:2]:
        termination = _normalize_text(attempt.get("termination_reason", "")) or "(none)"
        outcome = _normalize_text(attempt.get("outcome_tag", "")) or "(none)"
        lines.append(f"- prior_attempt: outcome={outcome} termination={termination}")
    return "\n".join(lines)


def _build_action_skeleton_text(action_skeleton: str, trajectory_text: str) -> str:
    normalized = str(action_skeleton or "").strip()
    if normalized:
        return normalized

    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in str(trajectory_text or "").splitlines():
        line = _normalize_text(raw_line)
        lowered = line.lower()
        if not line:
            continue
        if lowered.startswith("tool:"):
            key = line.lower()
            if key not in seen:
                seen.add(key)
                lines.append(f"- {line}")
        elif lowered.startswith("assistant:") and any(
            token in lowered
            for token in (
                "verify",
                "check",
                "lookup",
                "search",
                "confirm",
                "book",
                "cancel",
                "modify",
                "refund",
                "return",
                "exchange",
            )
        ):
            snippet = _clip_words(line.split(":", 1)[1], 14)
            key = snippet.lower()
            if key not in seen:
                seen.add(key)
                lines.append(f"- assistant: {snippet}")
        if len(lines) >= 6:
            break
    return "\n".join(lines) if lines else "- (not available)"


def build_evidence_bundle(trajectory_text: str) -> str:
    lines: list[str] = []
    for raw_line in str(trajectory_text or "").splitlines():
        normalized = _normalize_text(raw_line)
        if normalized:
            lines.append(normalized)
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

    for line in lines[:3]:
        _add(line)
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("tool:") or any(
            token in lowered
            for token in (
                "not found",
                "failed",
                "failure",
                "error",
                "timeout",
                "cannot",
                "unable",
                "invalid",
                "warning",
                "refund",
                "cancel",
                "exchange",
                "modify",
                "confirm",
                "verify",
            )
        ):
            _add(line)
    for line in lines[-6:]:
        _add(line)
    return "\n".join(selected[:18])


def _normalize_generation_result(result: Any) -> dict[str, Any]:
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
        return {
            "payload": result.get("payload"),
            "raw_text": str(result.get("raw_text", "") or "").strip(),
            "repaired_text": str(result.get("repaired_text", "") or "").strip(),
            "used_repair": bool(result.get("used_repair", False)),
            "failure_stage": str(result.get("failure_stage", "") or "").strip(),
            "failure_reason": str(result.get("failure_reason", "") or "").strip(),
        }
    if isinstance(result, dict):
        return {
            "payload": result,
            "raw_text": json.dumps(result, ensure_ascii=False),
            "repaired_text": "",
            "used_repair": False,
            "failure_stage": "",
            "failure_reason": "",
        }
    return {
        "payload": None,
        "raw_text": str(result or "").strip(),
        "repaired_text": "",
        "used_repair": False,
        "failure_stage": "",
        "failure_reason": "",
    }


def _call_text_generator(
    *,
    extractor_generate: GenerateFn,
    messages: list[dict[str, str]],
    call_name: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> dict[str, Any]:
    result = _normalize_generation_result(
        extractor_generate(
            messages=messages,
            call_name=call_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    )
    if result["payload"] is not None and not result["raw_text"]:
        result["raw_text"] = json.dumps(result["payload"], ensure_ascii=False)
    return result


def _build_json_repair_messages(
    *,
    raw_text: str,
    schema_name: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Convert the model output into exactly one valid JSON object.\n"
                "Return JSON only.\n"
                "Do not output analysis, markdown, code fences, or prose.\n"
                "If some fields are missing, keep the schema valid with empty strings, false, null, or empty arrays.\n\n"
                f"Target schema name: {schema_name}\n"
                f"Target schema:\n{schema_text}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original output:\n{raw_text}\n\n"
                "Return one valid JSON object matching the target schema."
            ),
        },
    ]


def _call_json_generator(
    *,
    extractor_generate: GenerateFn,
    messages: list[dict[str, str]],
    call_name: str,
    max_new_tokens: int,
    repair_max_new_tokens: int,
    schema_name: str,
    schema_text: str,
) -> dict[str, Any]:
    result = _call_text_generator(
        extractor_generate=extractor_generate,
        messages=messages,
        call_name=call_name,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )

    payload = result.get("payload")
    if payload is None:
        raw_text = str(result.get("raw_text", "") or "")
        try:
            payload = _safe_json_loads(raw_text)
            result["payload"] = payload
        except Exception as exc:
            result["failure_stage"] = result.get("failure_stage") or f"{call_name}_parse"
            result["failure_reason"] = result.get("failure_reason") or (
                f"{type(exc).__name__}: {exc}"
            )
            repair_messages = _build_json_repair_messages(
                raw_text=raw_text,
                schema_name=schema_name,
                schema_text=schema_text,
            )
            repair_result = _call_text_generator(
                extractor_generate=extractor_generate,
                messages=repair_messages,
                call_name=f"{call_name}_repair",
                max_new_tokens=repair_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            repaired_text = str(repair_result.get("raw_text", "") or "")
            try:
                payload = _safe_json_loads(repaired_text)
                result["payload"] = payload
                result["repaired_text"] = repaired_text
                result["used_repair"] = True
                result["failure_stage"] = ""
                result["failure_reason"] = ""
            except Exception as repair_exc:
                result["repaired_text"] = repaired_text
                result["used_repair"] = True
                result["failure_stage"] = f"{call_name}_repair_parse"
                result["failure_reason"] = f"{type(repair_exc).__name__}: {repair_exc}"
    return result


def _posterior_generator_messages(
    *,
    outcome_tag: str,
    task_query: str,
    family_prior: str,
    warnings: list[str],
    termination_reason: str | None,
    reward_value: float | None,
    reward_breakdown: dict[str, Any] | None,
    trajectory_text: str,
    meta_sop_text: str,
    action_skeleton: str,
    reflexion_summary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    system_prompt = _load_prompt_template("posterior_generator.md")
    user_prompt = (
        f"Outcome Tag:\n{outcome_tag}\n\n"
        f"Task Query:\n{task_query or '(not available)'}\n\n"
        f"Family Prior:\n{family_prior or 'default'}\n\n"
        f"Termination Reason:\n{termination_reason or '(none)'}\n\n"
        f"Reward Value:\n{reward_value if reward_value is not None else '(none)'}\n\n"
        f"Reward Breakdown:\n{_reward_block(reward_breakdown)}\n\n"
        f"Runtime Warnings:\n{_warnings_block(warnings)}\n\n"
        f"Action Skeleton:\n{_build_action_skeleton_text(action_skeleton, trajectory_text)}\n\n"
        f"Reflexion Summary:\n{_reflexion_block(reflexion_summary)}\n\n"
        f"Meta-SOP:\n{meta_sop_text.strip() or '(empty)'}\n\n"
        f"Trajectory:\n{trajectory_text.strip() or '(empty)'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _memory_extractor_messages(
    *,
    outcome_tag: str,
    family_prior: str,
    family_choices: list[str],
    posterior_text: str,
    meta_sop_text: str,
    warnings: list[str],
    reflexion_summary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    system_prompt = _load_prompt_template("memory_extractor.md")
    user_prompt = (
        f"Outcome Tag:\n{outcome_tag}\n\n"
        f"Family Prior:\n{family_prior or 'default'}\n\n"
        f"Allowed Family Labels:\n{', '.join(family_choices)}\n\n"
        f"Reflexion Summary:\n{_reflexion_block(reflexion_summary)}\n\n"
        f"Runtime Warnings:\n{_warnings_block(warnings)}\n\n"
        f"Meta-SOP:\n{meta_sop_text.strip() or '(empty)'}\n\n"
        f"Posterior:\n{posterior_text.strip() or '(empty)'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _workflow_extractor_messages(
    *,
    outcome_tag: str,
    family_prior: str,
    family_choices: list[str],
    posterior_text: str,
    meta_sop_text: str,
    warnings: list[str],
) -> list[dict[str, str]]:
    system_prompt = _load_prompt_template("workflow_extractor.md")
    user_prompt = (
        f"Outcome Tag:\n{outcome_tag}\n\n"
        f"Family Prior:\n{family_prior or 'default'}\n\n"
        f"Allowed Family Labels:\n{', '.join(family_choices)}\n\n"
        f"Runtime Warnings:\n{_warnings_block(warnings)}\n\n"
        f"Meta-SOP:\n{meta_sop_text.strip() or '(empty)'}\n\n"
        f"Posterior:\n{posterior_text.strip() or '(empty)'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _memory_schema_text() -> str:
    return json.dumps(
        {
            "memory_items": [
                {
                    "title": "string",
                    "applicability": "string",
                    "failure_class": "string",
                    "missing_step": "string",
                    "recovery_action": "string",
                    "family": "string",
                    "source_type": "failure | failure_informed_success",
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _workflow_schema_text() -> str:
    return json.dumps(
        {
            "workflow": {
                "family": "string",
                "precondition": "string",
                "state_sequence": ["string"],
                "critical_checks": ["string"],
                "commit_gate": "string",
                "postcondition": "string",
                "tool_hints": ["string"],
                "status": "candidate | stable",
                "repair_origin": True,
            }
        },
        ensure_ascii=False,
        indent=2,
    )


def _extract_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"(?im)^{re.escape(heading)}\s*:\s*(.*?)\s*(?=^[A-Za-z][A-Za-z /-]*:\s*|\Z)",
        re.DOTALL,
    )
    match = pattern.search(str(text or ""))
    if not match:
        return ""
    return _normalize_text(match.group(1))


def extract_posterior_digest(posterior_text: str) -> str:
    reusable_lesson = _extract_section(posterior_text, "Reusable Lesson")
    if reusable_lesson:
        return _clip_words(reusable_lesson, 36)
    pivot = _extract_section(posterior_text, "Failure/Success Pivot")
    if pivot:
        return _clip_words(pivot, 36)
    return _clip_words(str(posterior_text or ""), 36)


def _compose_memory_experience(item: dict[str, str]) -> str:
    applicability = _normalize_text(item.get("applicability", ""))
    missing_step = _normalize_text(item.get("missing_step", ""))
    recovery_action = _normalize_text(item.get("recovery_action", ""))
    pieces = [piece for piece in (applicability, missing_step, recovery_action) if piece]
    return " | ".join(pieces)


def _normalize_memory_items(
    value: Any,
    *,
    family_prior: str,
    outcome_tag: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value[:3]:
        if not isinstance(item, dict):
            continue
        family = _normalize_text(item.get("family", family_prior)) or family_prior
        title = _normalize_text(item.get("title", ""))
        applicability = _normalize_text(item.get("applicability", ""))
        failure_class = _normalize_text(item.get("failure_class", ""))
        missing_step = _normalize_text(item.get("missing_step", ""))
        recovery_action = _normalize_text(item.get("recovery_action", ""))
        source_type = _normalize_text(item.get("source_type", "")).lower()
        if source_type not in {"failure", "failure_informed_success"}:
            source_type = "failure" if outcome_tag == "failure" else "failure_informed_success"
        if not title or not recovery_action:
            continue
        normalized_item = {
            "title": title,
            "applicability": applicability,
            "failure_class": failure_class,
            "missing_step": missing_step,
            "recovery_action": recovery_action,
            "family": family,
            "source_type": source_type,
        }
        normalized_item["experience"] = _compose_memory_experience(normalized_item)
        key = (
            family.lower(),
            title.lower(),
            normalized_item["experience"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(normalized_item)
    return normalized


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = _normalize_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _normalize_workflow(
    value: Any,
    *,
    family_prior: str,
    outcome_tag: str,
) -> dict[str, Any] | None:
    if isinstance(value, dict) and "workflow" in value:
        value = value.get("workflow")
    if value is None or value == "":
        return None
    if not isinstance(value, dict):
        return None
    family = _normalize_text(value.get("family", family_prior)) or family_prior
    precondition = _normalize_text(value.get("precondition", ""))
    state_sequence = _normalize_string_list(value.get("state_sequence", []), limit=8)
    critical_checks = _normalize_string_list(value.get("critical_checks", []), limit=6)
    commit_gate = _normalize_text(value.get("commit_gate", ""))
    postcondition = _normalize_text(value.get("postcondition", ""))
    tool_hints = _normalize_string_list(value.get("tool_hints", []), limit=6)
    status = _normalize_text(value.get("status", "")).lower()
    if status not in {"candidate", "stable"}:
        status = "candidate" if outcome_tag == "failure" else "stable"
    repair_origin = bool(value.get("repair_origin", False))
    if len(state_sequence) < 2:
        return None
    return {
        "family": family,
        "precondition": precondition,
        "state_sequence": state_sequence,
        "critical_checks": critical_checks,
        "commit_gate": commit_gate,
        "postcondition": postcondition,
        "tool_hints": tool_hints,
        "status": status,
        "repair_origin": repair_origin,
    }


def workflow_to_graph_updates(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(workflow, dict):
        return []
    states = workflow.get("state_sequence") or []
    if len(states) < 2:
        return []
    condition = workflow.get("commit_gate") or workflow.get("precondition") or ""
    updates: list[dict[str, Any]] = []
    for anchor_state, state in zip(states, states[1:], strict=False):
        updates.append(
            {
                "op": "strengthen_edge",
                "family": workflow.get("family", ""),
                "anchor_state": anchor_state,
                "state": state,
                "condition": condition,
                "status": workflow.get("status", ""),
                "repair_origin": bool(workflow.get("repair_origin", False)),
                "precondition": workflow.get("precondition", ""),
                "critical_checks": list(workflow.get("critical_checks", []) or []),
                "commit_gate": workflow.get("commit_gate", ""),
                "postcondition": workflow.get("postcondition", ""),
                "tool_hints": list(workflow.get("tool_hints", []) or []),
            }
        )
    return updates[:4]


def _warning_classification(warning_text: str) -> tuple[str, str]:
    text = warning_text.lower()
    if any(
        token in text
        for token in (
            "not implemented",
            "unsupported",
            "environment limit",
            "evaluator limit",
        )
    ):
        return "environment_limit", "none"
    if any(
        token in text
        for token in ("traceback", "exception", "infra", "timeout", "connection")
    ):
        return "infra_noise", "none"
    if any(token in text for token in ("not found", "missing", "invalid", "unknown")):
        return "actionable_support", "lookup_failure"
    return "infra_noise", "none"


def _fallback_family_focus(
    *,
    domain: str | None,
    family_prior: str,
    trajectory_text: str,
    warnings: list[str],
) -> str:
    normalized_domain = (domain or "").strip().lower()
    family_map = FAMILY_KEYWORDS.get(normalized_domain, {})
    if not family_map:
        return family_prior
    scores = {family: 0 for family in family_map}
    haystacks = [trajectory_text.lower(), " ".join(warnings).lower()]
    for family, keywords in family_map.items():
        for haystack in haystacks:
            for keyword in keywords:
                if keyword in haystack:
                    scores[family] += 1
    best_family, best_score = max(scores.items(), key=lambda item: item[1])
    return best_family if best_score > 0 else family_prior


def _fallback_failure_memory_items(
    *,
    family_focus: str,
    trajectory_text: str,
    warnings: list[str],
) -> list[dict[str, str]]:
    text = trajectory_text.lower()
    warning_text = " ".join(warnings).lower()
    items: list[dict[str, str]] = []
    if any(token in f"{text} {warning_text}" for token in ("not found", "identifier", "lookup")):
        items.append(
            {
                "title": "Resolve identity before detail lookup",
                "applicability": "Use when the request depends on an order, reservation, account, or user lookup.",
                "failure_class": "lookup_failure",
                "missing_step": "Identity or record resolution was not completed before detail retrieval.",
                "recovery_action": "Back up to an alternate identity resolution step before calling detail or mutation tools.",
                "family": family_focus,
                "source_type": "failure",
            }
        )
    if any(token in text for token in ("book", "cancel", "modify", "exchange", "refund")):
        items.append(
            {
                "title": "Gate irreversible actions with a precondition check",
                "applicability": "Use when the next step mutates state, charges money, or commits a return/cancel/change.",
                "failure_class": "missing_precondition",
                "missing_step": "The workflow moved toward execution before confirming eligibility, status, or required fields.",
                "recovery_action": "Insert an explicit validation gate immediately before the mutation step.",
                "family": family_focus,
                "source_type": "failure",
            }
        )
    if not items:
        items.append(
            {
                "title": "Patch the missing prerequisite step",
                "applicability": "Use when the trajectory almost completes the task but fails on sequence or gating.",
                "failure_class": "workflow_gap",
                "missing_step": "A prerequisite step was skipped or collapsed too early.",
                "recovery_action": "Recover the prerequisite step first, then continue the remaining plan in order.",
                "family": family_focus,
                "source_type": "failure",
            }
        )
    return _normalize_memory_items(items, family_prior=family_focus, outcome_tag="failure")


def _fallback_success_memory_items(
    *,
    family_focus: str,
    posterior_text: str,
    reflexion_summary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    summary = _normalize_reflexion_summary(reflexion_summary)
    if not summary.get("memory"):
        return []
    lesson = extract_posterior_digest(posterior_text) or "Reuse the repaired branch that succeeded after earlier failure."
    items = [
        {
            "title": "Keep the repaired branch that converted failure into success",
            "applicability": "Use when the current task resembles an episode that required a mid-course correction before success.",
            "failure_class": "failure_informed_success",
            "missing_step": "An earlier failed branch exposed a prerequisite or ordering error.",
            "recovery_action": lesson,
            "family": family_focus,
            "source_type": "failure_informed_success",
        }
    ]
    return _normalize_memory_items(
        items,
        family_prior=family_focus,
        outcome_tag="success",
    )


def _fallback_workflow(
    *,
    outcome_tag: str,
    family_focus: str,
    trajectory_text: str,
    warnings: list[str],
    reflexion_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    text = trajectory_text.lower()
    if any(token in f"{text} {' '.join(warnings).lower()}" for token in ("not found", "identifier", "lookup")):
        states = [
            "collect_request",
            "resolve_identity",
            "validate_preconditions",
            "execute_task_action",
            "confirm_outcome",
        ]
    elif any(token in text for token in ("book", "cancel", "modify", "refund", "return", "exchange")):
        states = [
            "collect_request",
            "validate_preconditions",
            "prepare_action",
            "execute_task_action",
            "confirm_outcome",
        ]
    else:
        states = [
            "collect_request",
            "validate_preconditions",
            "execute_task_action",
            "confirm_outcome",
        ]
    workflow = {
        "family": family_focus,
        "precondition": "Task intent and required identifiers are established.",
        "state_sequence": states,
        "critical_checks": [
            "Do not execute the action before the prerequisite entity or eligibility check passes.",
        ],
        "commit_gate": "Only commit after the final required fields and constraints are verified.",
        "postcondition": "The requested task outcome is reached and communicated clearly.",
        "tool_hints": [],
        "status": "candidate" if outcome_tag == "failure" else "stable",
        "repair_origin": bool(_normalize_reflexion_summary(reflexion_summary).get("memory")),
    }
    return _normalize_workflow(
        workflow,
        family_prior=family_focus,
        outcome_tag=outcome_tag,
    )


def _fallback_posterior_text(
    *,
    outcome_tag: str,
    task_query: str,
    family_focus: str,
    trajectory_text: str,
    warnings: list[str],
    reward_value: float | None,
    reflexion_summary: dict[str, Any] | None,
    memory_items: list[dict[str, str]],
    workflow: dict[str, Any] | None,
) -> str:
    lines = [
        f"Task Goal: {task_query or '(not available)'}",
        f"Observed Procedure: {_clip_words(build_evidence_bundle(trajectory_text), 120)}",
        "Validated Facts: "
        + (
            _clip_words(
                " | ".join(warnings[:2]) if warnings else "No explicit runtime warning was recorded.",
                40,
            )
        ),
        f"Failure/Success Pivot: outcome={outcome_tag} reward={reward_value if reward_value is not None else '(none)'}",
        "Missing Step or Critical Decision: "
        + (
            memory_items[0]["missing_step"]
            if memory_items
            else "The key missing step could not be localized more precisely."
        ),
        "Reusable Lesson: "
        + (
            memory_items[0]["recovery_action"]
            if memory_items
            else "Keep the successful ordering and re-verify prerequisites before mutation."
        ),
        "Candidate Workflow Signal: "
        + (
            " -> ".join(workflow.get("state_sequence", []))
            if isinstance(workflow, dict)
            else f"collect_request -> validate_preconditions -> execute_{family_focus}"
        ),
    ]
    summary = _normalize_reflexion_summary(reflexion_summary)
    if summary.get("memory"):
        lines.append(f"Reflexion Note: {_clip_words(summary['memory'][-1], 24)}")
    return "\n".join(lines).strip()


def _packet_update_enabled(packet: dict[str, Any]) -> bool:
    termination_reason = _normalize_text(packet.get("termination_reason", "")).upper()
    if "TIMEOUT" in termination_reason or "INFRA" in termination_reason:
        return False
    if _normalize_text(packet.get("case_type", "")).lower() == "infra_error":
        return False
    flags = {
        _normalize_text(flag).lower()
        for flag in packet.get("diagnostic_flags", [])
        if _normalize_text(flag)
    }
    if "warning_without_trajectory_evidence" in flags:
        return False
    return bool(packet.get("memory_items") or packet.get("graph_updates") or packet.get("workflow"))


def _build_prompt_packet(
    *,
    sample_id: str,
    outcome_tag: str,
    trajectory_text: str,
    termination_reason: str | None,
    task_family: str | None,
    warnings: list[str],
    domain: str | None,
    task_query: str,
    meta_sop_text: str,
    reward_value: float | None,
    reward_breakdown: dict[str, Any] | None,
    action_skeleton: str,
    reflexion_summary: dict[str, Any] | None,
    extractor_generate: GenerateFn,
    posterior_max_new_tokens: int,
    memory_extract_max_new_tokens: int,
    workflow_extract_max_new_tokens: int,
    repair_max_new_tokens: int,
) -> dict[str, Any] | None:
    family_prior = task_family or "default"
    family_choices = _family_choices(domain, family_prior)

    posterior_messages = _posterior_generator_messages(
        outcome_tag=outcome_tag,
        task_query=task_query,
        family_prior=family_prior,
        warnings=warnings,
        termination_reason=termination_reason,
        reward_value=reward_value,
        reward_breakdown=reward_breakdown,
        trajectory_text=trajectory_text,
        meta_sop_text=meta_sop_text,
        action_skeleton=action_skeleton,
        reflexion_summary=reflexion_summary,
    )
    if _extractor_debug_enabled():
        _emit_extractor_debug_block(
            "POSTERIOR_PROMPT",
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            content=_render_debug_messages(posterior_messages),
        )
    posterior_result = _call_text_generator(
        extractor_generate=extractor_generate,
        messages=posterior_messages,
        call_name="posterior_generator",
        max_new_tokens=posterior_max_new_tokens,
    )
    posterior_text = str(posterior_result.get("raw_text", "") or "").strip()
    if _extractor_debug_enabled():
        _emit_extractor_debug_block(
            "POSTERIOR_RAW_OUTPUT",
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            content=posterior_text,
        )
    if not posterior_text:
        return None

    memory_items: list[dict[str, str]] = []
    memory_result: dict[str, Any] | None = None
    should_extract_memory = outcome_tag == "failure" or bool(
        _normalize_reflexion_summary(reflexion_summary).get("memory")
    )
    if should_extract_memory:
        memory_messages = _memory_extractor_messages(
            outcome_tag=outcome_tag,
            family_prior=family_prior,
            family_choices=family_choices,
            posterior_text=posterior_text,
            meta_sop_text=meta_sop_text,
            warnings=warnings,
            reflexion_summary=reflexion_summary,
        )
        if _extractor_debug_enabled():
            _emit_extractor_debug_block(
                "MEMORY_PROMPT",
                sample_id=sample_id,
                outcome_tag=outcome_tag,
                content=_render_debug_messages(memory_messages),
            )
        memory_result = _call_json_generator(
            extractor_generate=extractor_generate,
            messages=memory_messages,
            call_name="memory_extractor",
            max_new_tokens=memory_extract_max_new_tokens,
            repair_max_new_tokens=repair_max_new_tokens,
            schema_name="memory_items",
            schema_text=_memory_schema_text(),
        )
        if _extractor_debug_enabled():
            _emit_extractor_debug_block(
                "MEMORY_RAW_OUTPUT",
                sample_id=sample_id,
                outcome_tag=outcome_tag,
                content=str(memory_result.get("raw_text", "") or ""),
            )
        if memory_result.get("payload") is None:
            return None
        memory_items = _normalize_memory_items(
            memory_result.get("payload", {}).get("memory_items", []),
            family_prior=family_prior,
            outcome_tag=outcome_tag,
        )

    workflow_messages = _workflow_extractor_messages(
        outcome_tag=outcome_tag,
        family_prior=family_prior,
        family_choices=family_choices,
        posterior_text=posterior_text,
        meta_sop_text=meta_sop_text,
        warnings=warnings,
    )
    if _extractor_debug_enabled():
        _emit_extractor_debug_block(
            "WORKFLOW_PROMPT",
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            content=_render_debug_messages(workflow_messages),
        )
    workflow_result = _call_json_generator(
        extractor_generate=extractor_generate,
        messages=workflow_messages,
        call_name="workflow_extractor",
        max_new_tokens=workflow_extract_max_new_tokens,
        repair_max_new_tokens=repair_max_new_tokens,
        schema_name="workflow",
        schema_text=_workflow_schema_text(),
    )
    if _extractor_debug_enabled():
        _emit_extractor_debug_block(
            "WORKFLOW_RAW_OUTPUT",
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            content=str(workflow_result.get("raw_text", "") or ""),
        )
    if workflow_result.get("payload") is None:
        return None
    workflow = _normalize_workflow(
        workflow_result.get("payload"),
        family_prior=family_prior,
        outcome_tag=outcome_tag,
    )
    family_focus = (
        str((workflow or {}).get("family", "") or "")
        or (memory_items[0]["family"] if memory_items else family_prior)
    )
    graph_updates = workflow_to_graph_updates(workflow)

    warning_assessment = []
    for warning in warnings:
        category, supports = _warning_classification(warning)
        warning_assessment.append(
            {
                "warning_text": warning,
                "category": category,
                "supports": supports,
            }
        )

    diagnostic_flags = ["prompt_pipeline_used"]
    if not trajectory_text.strip() and warnings:
        diagnostic_flags.append("warning_without_trajectory_evidence")
        memory_items = []
        workflow = None
        graph_updates = []

    repaired_outputs = [
        text
        for text in (
            str((memory_result or {}).get("repaired_text", "") or ""),
            str(workflow_result.get("repaired_text", "") or ""),
        )
        if text
    ]
    raw_output_chunks = [
        f"POSTERIOR\n{posterior_text}",
    ]
    if memory_result is not None:
        raw_output_chunks.append(f"MEMORY\n{str(memory_result.get('raw_text', '') or '')}")
    raw_output_chunks.append(f"WORKFLOW\n{str(workflow_result.get('raw_text', '') or '')}")

    packet = {
        "case_type": outcome_tag,
        "family_prior": family_prior,
        "family_focus": family_focus,
        "warning_assessment": warning_assessment,
        "memory_items": memory_items,
        "workflow": workflow,
        "graph_updates": graph_updates,
        "diagnostic_flags": diagnostic_flags,
        "termination_reason": termination_reason or "",
        "posterior_text": posterior_text,
        "posterior_digest": extract_posterior_digest(posterior_text),
        "extractor_source": "prompt",
        "extractor_used_repair": bool(
            (memory_result or {}).get("used_repair", False)
            or workflow_result.get("used_repair", False)
        ),
        "extractor_failure_stage": "",
        "extractor_failure_reason": "",
        "extractor_raw_text": _truncate_debug_text("\n\n".join(raw_output_chunks)),
        "extractor_repaired_text": _truncate_debug_text("\n\n".join(repaired_outputs)),
    }
    packet["update_enabled"] = _packet_update_enabled(packet)
    return packet


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
    reward_value: float | None = None,
    reward_breakdown: dict[str, Any] | None = None,
    action_skeleton: str = "",
    reflexion_summary: dict[str, Any] | None = None,
    extractor_generate: GenerateFn | None = None,
    posterior_max_new_tokens: int = 2048,
    memory_extract_max_new_tokens: int = 1024,
    workflow_extract_max_new_tokens: int = 1024,
    repair_max_new_tokens: int = 512,
) -> dict[str, Any]:
    normalized_warnings = _normalize_warnings(warnings)
    family_prior = task_family or "default"
    family_focus = _fallback_family_focus(
        domain=domain,
        family_prior=family_prior,
        trajectory_text=trajectory_text,
        warnings=normalized_warnings,
    )

    if (
        extractor_generate is not None
        and outcome_tag != "infra_error"
        and _normalize_text(trajectory_text)
    ):
        packet = _build_prompt_packet(
            sample_id=sample_id,
            outcome_tag=outcome_tag,
            trajectory_text=trajectory_text,
            termination_reason=termination_reason,
            task_family=task_family,
            warnings=normalized_warnings,
            domain=domain,
            task_query=task_query,
            meta_sop_text=meta_sop_text,
            reward_value=reward_value,
            reward_breakdown=reward_breakdown,
            action_skeleton=action_skeleton,
            reflexion_summary=reflexion_summary,
            extractor_generate=extractor_generate,
            posterior_max_new_tokens=posterior_max_new_tokens,
            memory_extract_max_new_tokens=memory_extract_max_new_tokens,
            workflow_extract_max_new_tokens=workflow_extract_max_new_tokens,
            repair_max_new_tokens=repair_max_new_tokens,
        )
        if packet is not None:
            return packet

    memory_items: list[dict[str, str]]
    if outcome_tag == "failure":
        memory_items = _fallback_failure_memory_items(
            family_focus=family_focus,
            trajectory_text=trajectory_text,
            warnings=normalized_warnings,
        )
    else:
        memory_items = _fallback_success_memory_items(
            family_focus=family_focus,
            posterior_text="",
            reflexion_summary=reflexion_summary,
        )

    workflow = None
    if _normalize_text(trajectory_text):
        workflow = _fallback_workflow(
            outcome_tag=outcome_tag,
            family_focus=family_focus,
            trajectory_text=trajectory_text,
            warnings=normalized_warnings,
            reflexion_summary=reflexion_summary,
        )
    graph_updates = workflow_to_graph_updates(workflow)

    warning_assessment = []
    diagnostic_flags: list[str] = ["extractor_fallback_used"]
    for warning in normalized_warnings:
        category, supports = _warning_classification(warning)
        warning_assessment.append(
            {
                "warning_text": warning,
                "category": category,
                "supports": supports,
            }
        )

    if not _normalize_text(trajectory_text) and normalized_warnings:
        diagnostic_flags.append("warning_without_trajectory_evidence")
        memory_items = []
        workflow = None
        graph_updates = []

    posterior_text = _fallback_posterior_text(
        outcome_tag=outcome_tag,
        task_query=task_query,
        family_focus=family_focus,
        trajectory_text=trajectory_text,
        warnings=normalized_warnings,
        reward_value=reward_value,
        reflexion_summary=reflexion_summary,
        memory_items=memory_items,
        workflow=workflow,
    )
    packet = {
        "case_type": outcome_tag,
        "family_prior": family_prior,
        "family_focus": family_focus,
        "warning_assessment": warning_assessment,
        "memory_items": memory_items,
        "workflow": workflow,
        "graph_updates": graph_updates,
        "diagnostic_flags": diagnostic_flags,
        "termination_reason": termination_reason or "",
        "posterior_text": posterior_text,
        "posterior_digest": extract_posterior_digest(posterior_text),
        "extractor_source": "fallback",
        "extractor_used_repair": False,
        "extractor_failure_stage": "prompt_pipeline",
        "extractor_failure_reason": "prompt pipeline unavailable or incomplete",
        "extractor_raw_text": "",
        "extractor_repaired_text": "",
    }
    packet["update_enabled"] = _packet_update_enabled(packet)
    return packet
