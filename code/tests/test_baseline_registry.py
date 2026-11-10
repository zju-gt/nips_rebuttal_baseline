from __future__ import annotations

import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def test_registry_matches_exp1_method_matrix() -> None:
    from baseline_registry import METHOD_ORDER, get_baseline_specs

    specs = get_baseline_specs()

    assert METHOD_ORDER == (
        "react",
        "reflexion",
        "awm",
        "reasoningbank",
        "evotest",
        "living_harness",
    )
    assert tuple(specs) == METHOD_ORDER
    assert {name: spec.repo_name for name, spec in specs.items()} == {
        "react": "sop-world-react",
        "reflexion": "sop-world-flexion",
        "awm": "sop-world-AWM",
        "reasoningbank": "sop-world-Reasoning",
        "evotest": "sop-world-EvoTest",
        "living_harness": "sop-world",
    }

    assert dict(specs["react"].env) == {
        "MODE": "base_lm",
        "SOP_TEXT": "static",
        "NUM_CYCLES": "1",
        "PASS_HAT_K": "1",
        "EVOLVE_COMPONENTS": "",
    }
    assert dict(specs["reflexion"].env) == {
        "MODE": "base_lm",
        "SOP_TEXT": "static",
        "NUM_CYCLES": "1",
        "PASS_HAT_K": "1",
        "EVOLVE_COMPONENTS": "",
        "REFLEXION_ENABLED": "1",
        "REFLEXION_MAX_TRIALS": "3",
        "REFLEXION_MEMORY_SIZE": "3",
    }
    assert dict(specs["awm"].env) == {
        "MODE": "base_lm",
        "SOP_TEXT": "static",
        "NUM_CYCLES": "1",
        "PASS_HAT_K": "1",
        "EVOLVE_COMPONENTS": "workflow_memory",
        "EVOLVE_PARALLEL_MODE": "strict",
    }
    assert dict(specs["reasoningbank"].env) == {
        "MODE": "base_lm",
        "SOP_TEXT": "static",
        "NUM_CYCLES": "1",
        "PASS_HAT_K": "1",
        "EVOLVE_COMPONENTS": "reasoning_bank",
        "EVOLVE_PARALLEL_MODE": "strict",
    }
    assert dict(specs["evotest"].env) == {
        "MODE": "prior",
        "SOP_TEXT": "static",
        "NUM_CYCLES": "1",
        "PASS_HAT_K": "3",
        "EVOLVE_COMPONENTS": "memory,state_graph",
        "EVOLVE_PARALLEL_MODE": "strict",
        "EVOTEST_HINTS_ENABLED": "1",
        "EVOTEST_HINT_MAX_ITEMS": "3",
    }
    assert dict(specs["living_harness"].env) == {
        "MODE": "prior",
        "SOP_TEXT": "dynamic",
        "NUM_CYCLES": "3",
        "PASS_HAT_K": "1",
        "EVOLVE_COMPONENTS": "meta_sop,memory,state_graph",
        "EVOLVE_PARALLEL_MODE": "strict",
    }
    assert specs["react"].stateful is False
    assert specs["reflexion"].stateful is False
    assert all(
        specs[name].stateful
        for name in ("awm", "reasoningbank", "evotest", "living_harness")
    )


def test_parse_selection_defaults_to_allowed_order() -> None:
    from baseline_registry import METHOD_ORDER, parse_selection

    assert parse_selection(None, METHOD_ORDER, "method") == list(METHOD_ORDER)


def test_parse_selection_normalizes_and_deduplicates() -> None:
    from baseline_registry import METHOD_ORDER, parse_selection

    assert parse_selection(
        " reflexion,react, reflexion ", METHOD_ORDER, "method"
    ) == ["reflexion", "react"]


@pytest.mark.parametrize("raw", ["unknown", "react,unknown"])
def test_parse_selection_rejects_unknown_values(raw: str) -> None:
    from baseline_registry import METHOD_ORDER, parse_selection

    with pytest.raises(ValueError, match="unknown method"):
        parse_selection(raw, METHOD_ORDER, "method")


def test_parse_selection_rejects_empty_explicit_selection() -> None:
    from baseline_registry import METHOD_ORDER, parse_selection

    with pytest.raises(ValueError, match="method selection must not be empty"):
        parse_selection(" , ", METHOD_ORDER, "method")

