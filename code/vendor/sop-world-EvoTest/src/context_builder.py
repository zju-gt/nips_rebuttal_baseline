from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedContextBundle:
    mode: str
    sop_text: str
    domain: str
    benchmark_prior_text: str
    domain_prior_text: str
    meta_sop_text: str
    current_system_prompt: str
    artifact_dir: Path


def build_system_prompt(
    mode: str,
    benchmark_prior_text: str,
    domain_prior_text: str,
) -> str:
    normalized_mode = mode.strip()
    if normalized_mode in {"base_lm", "meta_sop"}:
        return ""
    if normalized_mode not in {"prior", "meta_sop_prior"}:
        raise ValueError(f"Unsupported mode: {mode}")

    parts = [benchmark_prior_text.strip(), domain_prior_text.strip()]
    return "\n\n".join(part for part in parts if part)


def load_context_bundle(
    *,
    mode: str,
    sop_text: str,
    domain: str,
    benchmark_prior_text: str,
    domain_prior_text: str,
    meta_sop_text: str,
    artifact_dir: Path,
) -> ResolvedContextBundle:
    return ResolvedContextBundle(
        mode=mode,
        sop_text=sop_text,
        domain=domain,
        benchmark_prior_text=benchmark_prior_text,
        domain_prior_text=domain_prior_text,
        meta_sop_text=meta_sop_text,
        current_system_prompt=build_system_prompt(
            mode, benchmark_prior_text, domain_prior_text
        ),
        artifact_dir=artifact_dir,
    )

