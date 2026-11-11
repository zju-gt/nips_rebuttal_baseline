from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


METHOD_ORDER = (
    "react",
    "reflexion",
    "awm",
    "reasoningbank",
    "evotest",
    "living_harness",
)
DOMAIN_ORDER = ("retail", "airline", "telecom")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    repo_name: str
    env: Mapping[str, str]
    stateful: bool


def _spec(
    name: str,
    repo_name: str,
    env: dict[str, str],
    *,
    stateful: bool,
) -> BaselineSpec:
    return BaselineSpec(
        name=name,
        repo_name=repo_name,
        env=MappingProxyType(dict(env)),
        stateful=stateful,
    )


def get_baseline_specs() -> dict[str, BaselineSpec]:
    specs = (
        _spec(
            "react",
            "sop-world-react",
            {
                "MODE": "base_lm",
                "SOP_TEXT": "static",
                "NUM_CYCLES": "1",
                "PASS_HAT_K": "1",
                "EVOLVE_COMPONENTS": "",
            },
            stateful=False,
        ),
        _spec(
            "reflexion",
            "sop-world-flexion",
            {
                "MODE": "base_lm",
                "SOP_TEXT": "static",
                "NUM_CYCLES": "1",
                "PASS_HAT_K": "1",
                "EVOLVE_COMPONENTS": "",
                "REFLEXION_ENABLED": "1",
                "REFLEXION_MAX_TRIALS": "3",
                "REFLEXION_MEMORY_SIZE": "3",
            },
            stateful=False,
        ),
        _spec(
            "awm",
            "sop-world-AWM",
            {
                "MODE": "base_lm",
                "SOP_TEXT": "static",
                "NUM_CYCLES": "1",
                "PASS_HAT_K": "1",
                "EVOLVE_COMPONENTS": "workflow_memory",
                "EVOLVE_PARALLEL_MODE": "async",
            },
            stateful=True,
        ),
        _spec(
            "reasoningbank",
            "sop-world-Reasoning",
            {
                "MODE": "base_lm",
                "SOP_TEXT": "static",
                "NUM_CYCLES": "1",
                "PASS_HAT_K": "1",
                "EVOLVE_COMPONENTS": "reasoning_bank",
                "EVOLVE_PARALLEL_MODE": "async",
            },
            stateful=True,
        ),
        _spec(
            "evotest",
            "sop-world-EvoTest",
            {
                "MODE": "prior",
                "SOP_TEXT": "static",
                "NUM_CYCLES": "1",
                "PASS_HAT_K": "3",
                "EVOLVE_COMPONENTS": "memory,state_graph",
                "EVOLVE_PARALLEL_MODE": "async",
                "EVOTEST_HINTS_ENABLED": "1",
                "EVOTEST_HINT_MAX_ITEMS": "3",
            },
            stateful=True,
        ),
        _spec(
            "living_harness",
            "sop-world",
            {
                "MODE": "prior",
                "SOP_TEXT": "dynamic",
                "NUM_CYCLES": "3",
                "PASS_HAT_K": "1",
                "EVOLVE_COMPONENTS": "meta_sop,memory,state_graph",
                "EVOLVE_PARALLEL_MODE": "strict",
            },
            stateful=True,
        ),
    )
    return {spec.name: spec for spec in specs}


def parse_selection(
    raw: str | None,
    allowed: Sequence[str],
    label: str,
) -> list[str]:
    allowed_values = tuple(allowed)
    if raw is None:
        return list(allowed_values)

    selected: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if not value or value in selected:
            continue
        if value not in allowed_values:
            raise ValueError(f"unknown {label}: {value}")
        selected.append(value)

    if not selected:
        raise ValueError(f"{label} selection must not be empty")
    return selected

