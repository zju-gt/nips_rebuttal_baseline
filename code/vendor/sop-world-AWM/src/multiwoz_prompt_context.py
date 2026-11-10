from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from context_builder import ResolvedContextBundle, load_context_bundle
from execution import (
    generate_benchmark_prior,
    generate_domain_prior,
    generate_meta_sop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
PRIOR_DIR = PROMPTS_ROOT / "prior"
META_SOP_DIR = PROMPTS_ROOT / "meta_sop"
MULTIWOZ_SRC = PROJECT_ROOT / "multiwoz" / "src"
MULTIWOZ_DOMAINS = ("restaurant", "hotel", "train", "attraction", "taxi")

if str(MULTIWOZ_SRC) not in sys.path:
    sys.path.insert(0, str(MULTIWOZ_SRC))

from speakrl_prompting_repro.prompts import AGENT_SYSTEM_PROMPT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def select_representative_tasks(
    tasks: list[Any],
    primary_domain: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    selected = [
        task for task in tasks if getattr(task, "primary_domain", "") == primary_domain
    ]
    selected = sorted(selected, key=lambda task: getattr(task, "dialogue_id", ""))[:limit]
    return [
        {
            "id": getattr(task, "dialogue_id", ""),
            "primary_domain": primary_domain,
            "goal": getattr(task, "raw_goal", {}),
        }
        for task in selected
    ]


def _static_text(kind: str, primary_domain: str) -> str:
    if kind == "benchmark_prior":
        return _read_text(PRIOR_DIR / "benchmark_prior_multiwoz.md")
    if kind == "domain_prior":
        return _read_text(PRIOR_DIR / f"domain_prior_{primary_domain}.md")
    if kind == "meta_sop":
        return _read_text(META_SOP_DIR / f"meta_sop_{primary_domain}.md")
    raise ValueError(f"Unsupported static asset kind: {kind}")


def prepare_multiwoz_context_bundles(
    *,
    mode: str,
    sop_text: str,
    run_dir: Path,
    tasks: list[Any],
    generator=None,
) -> dict[str, ResolvedContextBundle]:
    bundles: dict[str, ResolvedContextBundle] = {}
    domains = sorted(
        {
            getattr(task, "primary_domain", "")
            for task in tasks
            if getattr(task, "primary_domain", "") in MULTIWOZ_DOMAINS
        }
    )

    for primary_domain in domains:
        artifact_dir = run_dir / "artifacts" / primary_domain
        representative_tasks = select_representative_tasks(tasks, primary_domain)

        if sop_text == "static":
            benchmark_prior_text = _static_text("benchmark_prior", primary_domain)
            domain_prior_text = _static_text("domain_prior", primary_domain)
            meta_sop_text = (
                _static_text("meta_sop", primary_domain)
                if mode in {"meta_sop", "meta_sop_prior"}
                else ""
            )
        elif sop_text == "dynamic":
            benchmark_prior_text = generate_benchmark_prior(
                domain=primary_domain,
                domain_policy_text=AGENT_SYSTEM_PROMPT,
                representative_tasks=representative_tasks,
                generator=generator,
            )
            domain_prior_text = generate_domain_prior(
                domain=primary_domain,
                domain_policy_text=AGENT_SYSTEM_PROMPT,
                benchmark_prior_text=benchmark_prior_text,
                representative_tasks=representative_tasks,
                generator=generator,
            )
            meta_sop_text = ""
            if mode in {"meta_sop", "meta_sop_prior"}:
                meta_sop_text = generate_meta_sop(
                    domain=primary_domain,
                    benchmark_prior_text=benchmark_prior_text,
                    domain_prior_text=domain_prior_text,
                    representative_tasks=representative_tasks,
                    generator=generator,
                )
        else:
            raise ValueError(f"Unsupported SOP_TEXT: {sop_text}")

        _write_text(artifact_dir / "benchmark_prior.md", benchmark_prior_text)
        _write_text(artifact_dir / "domain_prior.md", domain_prior_text)
        if mode in {"meta_sop", "meta_sop_prior"}:
            _write_text(artifact_dir / "meta_sop.md", meta_sop_text)

        bundles[primary_domain] = load_context_bundle(
            mode=mode,
            sop_text=sop_text,
            domain=primary_domain,
            benchmark_prior_text=benchmark_prior_text,
            domain_prior_text=domain_prior_text,
            meta_sop_text=meta_sop_text,
            artifact_dir=artifact_dir,
        )

    return bundles
