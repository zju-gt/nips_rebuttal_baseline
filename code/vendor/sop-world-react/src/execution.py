from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from context_builder import ResolvedContextBundle, load_context_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
STATIC_PRIOR_DIR = PROMPTS_ROOT / "prior"
STATIC_META_SOP_DIR = PROMPTS_ROOT / "meta_sop"
GENERATION_DIR = PROMPTS_ROOT / "generation"

GenerationCallable = Callable[[str], str]


DOMAIN_FAMILY_KEYWORDS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "airline": [
        ("cancellation", ("cancel", "cancellation", "refund", "refunds")),
        ("modification", ("modify", "change", "rebook", "rebooking")),
        ("compensation", ("compensation", "delay", "delayed", "voucher")),
        ("booking", ("book", "booking", "reservation", "passengers")),
        ("verification", ("verify", "eligibility", "membership", "insurance")),
    ],
    "retail": [
        ("authentication", ("authenticate", "authentication", "email", "zip")),
        ("status", ("order status", "pending", "delivered", "status")),
        ("exchange_modify", ("exchange", "modify", "one-shot", "one shot")),
        ("variant", ("variant", "product details", "product_id")),
        ("payment", ("gift card", "refund", "payment", "credit card", "paypal")),
    ],
    "telecom": [
        ("mobile_data", ("mobile data", "data", "internet", "slow")),
        ("mms", ("mms", "picture", "group message")),
        ("roaming", ("roaming", "abroad")),
        ("billing", ("billing", "suspension", "overdue", "payment")),
        ("apn_reboot", ("apn", "reboot", "network preference")),
    ],
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _render_template(template_text: str, mapping: dict[str, str]) -> str:
    rendered = template_text
    for key, value in mapping.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
        rendered = rendered.replace(f"{{{{{key.upper()}}}}}", value)
    return rendered


def _task_identifier(task: dict[str, Any]) -> str:
    raw_id = task.get("id")
    if raw_id is None:
        raw_id = task.get("task_id")
    if raw_id is None:
        raw_id = task.get("name")
    return str(raw_id if raw_id is not None else json.dumps(task, ensure_ascii=False))


def _task_search_text(task: dict[str, Any]) -> str:
    return json.dumps(task, ensure_ascii=False, sort_keys=True).lower()


def _family_matches(domain: str, task_text: str) -> set[str]:
    families = DOMAIN_FAMILY_KEYWORDS.get(domain, [])
    matched: set[str] = set()
    for family, keywords in families:
        if any(keyword in task_text for keyword in keywords):
            matched.add(family)
    return matched


def select_representative_tasks(
    domain: str,
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    ordered_tasks = sorted(tasks, key=_task_identifier)
    candidate_families = {
        family for family, _ in DOMAIN_FAMILY_KEYWORDS.get(domain, [])
    }
    if not candidate_families:
        return ordered_tasks[: min(5, len(ordered_tasks))]

    annotated: list[tuple[dict[str, Any], set[str]]] = []
    for task in ordered_tasks:
        annotated.append((task, _family_matches(domain, _task_search_text(task))))

    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    remaining = annotated[:]

    while remaining and len(selected) < min(5, len(ordered_tasks)):
        best_index = 0
        best_score = (-1, -1)
        best_id = ""
        for index, (task, matched_families) in enumerate(remaining):
            uncovered = matched_families - covered
            score = (len(uncovered), len(matched_families))
            task_id = _task_identifier(task)
            if score > best_score or (score == best_score and task_id < best_id):
                best_score = score
                best_id = task_id
                best_index = index
        task, matched_families = remaining.pop(best_index)
        selected.append(task)
        covered |= matched_families
        if covered >= candidate_families:
            break

    if len(selected) < min(5, len(ordered_tasks)):
        seen_ids = {_task_identifier(task) for task in selected}
        for task in ordered_tasks:
            if _task_identifier(task) in seen_ids:
                continue
            selected.append(task)
            if len(selected) >= min(5, len(ordered_tasks)):
                break

    return selected


def _load_generation_template(name: str) -> str:
    template_path = GENERATION_DIR / name
    if template_path.exists():
        return _read_text(template_path)
    fallback_templates = {
        "benchmark_prior_generation.md": (
            "# Benchmark Prior Generation\n\n"
            "Input domain policy:\n{{DOMAIN_POLICY}}\n\n"
            "Representative tasks:\n{{REPRESENTATIVE_TASKS}}\n\n"
            "Write a short benchmark prior that keeps only universal rules.\n"
        ),
        "domain_prior_generation.md": (
            "# Domain Prior Generation\n\n"
            "Input domain policy:\n{{DOMAIN_POLICY}}\n\n"
            "Benchmark prior:\n{{BENCHMARK_PRIOR}}\n\n"
            "Representative tasks:\n{{REPRESENTATIVE_TASKS}}\n\n"
            "Write a short domain prior that keeps only domain-specific rules.\n"
        ),
        "meta_sop_generation.md": (
            "# Meta-SOP Generation\n\n"
            "Benchmark prior:\n{{BENCHMARK_PRIOR}}\n\n"
            "Domain prior:\n{{DOMAIN_PRIOR}}\n\n"
            "Representative tasks:\n{{REPRESENTATIVE_TASKS}}\n\n"
            "Write a concise top-level meta-SOP for future state-graph and memory use.\n"
        ),
    }
    if name not in fallback_templates:
        raise FileNotFoundError(template_path)
    return fallback_templates[name]


def _format_representative_tasks(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "- (none)"
    lines: list[str] = []
    for task in tasks:
        task_id = _task_identifier(task)
        purpose = ""
        description = task.get("description")
        if isinstance(description, dict):
            purpose = str(description.get("purpose") or "")
        instructions = ""
        user_scenario = task.get("user_scenario")
        if isinstance(user_scenario, dict):
            raw_instructions = user_scenario.get("instructions")
            if isinstance(raw_instructions, dict):
                instructions = str(raw_instructions.get("reason_for_call") or "")
        pieces = [piece for piece in [purpose, instructions] if piece.strip()]
        summary = " | ".join(pieces)
        lines.append(f"- {task_id}: {summary}" if summary else f"- {task_id}")
    return "\n".join(lines)


def _format_domain_policy(domain_policy_text: str) -> str:
    return domain_policy_text.strip() or "(empty)"


def _fallback_generation(
    *,
    heading: str,
    domain: str,
    domain_policy_text: str,
    representative_tasks: list[dict[str, Any]],
    benchmark_prior_text: str = "",
    domain_prior_text: str = "",
) -> str:
    task_lines = _format_representative_tasks(representative_tasks)
    sections = [f"# {heading}", f"Domain: {domain}"]
    if domain_policy_text.strip():
        sections.extend(["## Domain Policy", _format_domain_policy(domain_policy_text)])
    if benchmark_prior_text.strip():
        sections.extend(["## Benchmark Prior", benchmark_prior_text.strip()])
    if domain_prior_text.strip():
        sections.extend(["## Domain Prior", domain_prior_text.strip()])
    sections.extend(["## Representative Tasks", task_lines])
    return "\n\n".join(sections).strip() + "\n"


def build_generation_callback(
    *,
    domain: str,
    domain_policy_text: str,
    representative_tasks: list[dict[str, Any]],
) -> GenerationCallable:
    def _generator(prompt_text: str) -> str:
        header = prompt_text.splitlines()[0].strip() if prompt_text else ""
        if header == "@@STAGE=benchmark_prior_generation.md":
            return _fallback_generation(
                heading="Benchmark Prior",
                domain=domain,
                domain_policy_text=domain_policy_text,
                representative_tasks=representative_tasks,
            )
        if header == "@@STAGE=domain_prior_generation.md":
            return _fallback_generation(
                heading="Domain Prior",
                domain=domain,
                domain_policy_text=domain_policy_text,
                representative_tasks=representative_tasks,
            )
        if header == "@@STAGE=meta_sop_generation.md":
            return _fallback_generation(
                heading="Meta-SOP",
                domain=domain,
                domain_policy_text=domain_policy_text,
                representative_tasks=representative_tasks,
            )
        raise RuntimeError(f"Unknown generation stage marker: {header}")

    return _generator


def _generate_document(
    *,
    template_name: str,
    heading: str,
    domain: str,
    domain_policy_text: str,
    representative_tasks: list[dict[str, Any]],
    benchmark_prior_text: str = "",
    domain_prior_text: str = "",
    generator: GenerationCallable | None = None,
) -> str:
    template_text = _load_generation_template(template_name)
    rendered_prompt = _render_template(
        template_text,
        {
            "domain": domain,
            "domain_policy": _format_domain_policy(domain_policy_text),
            "representative_tasks": _format_representative_tasks(representative_tasks),
            "benchmark_prior": benchmark_prior_text.strip(),
            "domain_prior": domain_prior_text.strip(),
        },
    )
    if generator is not None:
        staged_prompt = f"@@STAGE={template_name}\n{rendered_prompt}"
        generated = generator(staged_prompt).strip()
        if not generated:
            raise RuntimeError(f"Generation callback returned empty output for {template_name}")
        return generated.rstrip() + "\n"
    return _fallback_generation(
        heading=heading,
        domain=domain,
        domain_policy_text=domain_policy_text,
        representative_tasks=representative_tasks,
        benchmark_prior_text=benchmark_prior_text,
        domain_prior_text=domain_prior_text,
    )


def generate_benchmark_prior(
    *,
    domain: str,
    domain_policy_text: str,
    representative_tasks: list[dict[str, Any]],
    generator: GenerationCallable | None = None,
) -> str:
    return _generate_document(
        template_name="benchmark_prior_generation.md",
        heading="Benchmark Prior",
        domain=domain,
        domain_policy_text=domain_policy_text,
        representative_tasks=representative_tasks,
        generator=generator,
    )


def generate_domain_prior(
    *,
    domain: str,
    domain_policy_text: str,
    benchmark_prior_text: str,
    representative_tasks: list[dict[str, Any]],
    generator: GenerationCallable | None = None,
) -> str:
    return _generate_document(
        template_name="domain_prior_generation.md",
        heading="Domain Prior",
        domain=domain,
        domain_policy_text=domain_policy_text,
        representative_tasks=representative_tasks,
        benchmark_prior_text=benchmark_prior_text,
        generator=generator,
    )


def generate_meta_sop(
    *,
    domain: str,
    benchmark_prior_text: str,
    domain_prior_text: str,
    representative_tasks: list[dict[str, Any]],
    generator: GenerationCallable | None = None,
) -> str:
    return _generate_document(
        template_name="meta_sop_generation.md",
        heading="Meta-SOP",
        domain=domain,
        domain_policy_text="",
        representative_tasks=representative_tasks,
        benchmark_prior_text=benchmark_prior_text,
        domain_prior_text=domain_prior_text,
        generator=generator,
    )


def _static_asset_path(kind: str, domain: str) -> Path:
    if kind == "benchmark_prior":
        return STATIC_PRIOR_DIR / "benchmark_prior_tau2.md"
    if kind == "domain_prior":
        return STATIC_PRIOR_DIR / f"domain_prior_{domain}.md"
    if kind == "meta_sop":
        return STATIC_META_SOP_DIR / f"meta_sop_{domain}.md"
    raise ValueError(f"Unknown prompt asset kind: {kind}")


def prepare_context_artifacts(
    *,
    mode: str,
    sop_text: str,
    domain: str,
    run_dir: Path,
    domain_policy_text: str,
    representative_tasks: list[dict[str, Any]],
    generator: GenerationCallable | None = None,
) -> ResolvedContextBundle:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if sop_text == "static":
        benchmark_prior_text = _read_text(_static_asset_path("benchmark_prior", domain))
        domain_prior_text = _read_text(_static_asset_path("domain_prior", domain))
        meta_sop_text = ""
        if mode in {"meta_sop", "meta_sop_prior"}:
            meta_sop_text = _read_text(_static_asset_path("meta_sop", domain))
    elif sop_text == "dynamic":
        benchmark_prior_text = generate_benchmark_prior(
            domain=domain,
            domain_policy_text=domain_policy_text,
            representative_tasks=representative_tasks,
            generator=generator,
        )
        domain_prior_text = generate_domain_prior(
            domain=domain,
            domain_policy_text=domain_policy_text,
            benchmark_prior_text=benchmark_prior_text,
            representative_tasks=representative_tasks,
            generator=generator,
        )
        meta_sop_text = ""
        if mode in {"meta_sop", "meta_sop_prior"}:
            meta_sop_text = generate_meta_sop(
                domain=domain,
                benchmark_prior_text=benchmark_prior_text,
                domain_prior_text=domain_prior_text,
                representative_tasks=representative_tasks,
                generator=generator,
            )
    else:
        raise ValueError(f"Unsupported SOP_TEXT: {sop_text}")

    _write_text(artifacts_dir / "benchmark_prior.md", benchmark_prior_text)
    _write_text(artifacts_dir / "domain_prior.md", domain_prior_text)
    if mode in {"meta_sop", "meta_sop_prior"}:
        _write_text(artifacts_dir / "meta_sop.md", meta_sop_text)

    return load_context_bundle(
        mode=mode,
        sop_text=sop_text,
        domain=domain,
        benchmark_prior_text=benchmark_prior_text,
        domain_prior_text=domain_prior_text,
        meta_sop_text=meta_sop_text,
        artifact_dir=artifacts_dir,
    )
