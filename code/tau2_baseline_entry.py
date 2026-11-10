from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


class EntryError(RuntimeError):
    pass


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one tau2 baseline job")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    return parser.parse_args(argv)


def _import_runner(repo: Path) -> ModuleType:
    runner_path = repo / "src" / "tau2_runner.py"
    if not runner_path.is_file():
        raise EntryError(f"selected repository is missing src/tau2_runner.py: {repo}")
    sys.path.insert(0, str(runner_path.parent))
    try:
        return importlib.import_module("tau2_runner")
    except Exception as exc:
        raise EntryError(
            f"failed to import tau2_runner ({type(exc).__name__})"
        ) from exc


def _load_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EntryError("runner did not create summary.json") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EntryError("runner created an invalid summary.json") from exc
    if not isinstance(value, dict):
        raise EntryError("runner summary.json must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = args.repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    runner = _import_runner(repo)
    run_benchmark = getattr(runner, "run_benchmark", None)
    if not callable(run_benchmark):
        raise EntryError("tau2_runner does not provide run_benchmark()")
    try:
        returned_summary = run_benchmark()
    except Exception as exc:
        raise EntryError(f"tau2 runner failed ({type(exc).__name__})") from exc
    if not isinstance(returned_summary, Mapping):
        raise EntryError("tau2 runner did not return a summary object")
    _load_summary(output_dir)
    print(
        json.dumps(
            {
                "method": args.method,
                "output_dir": str(output_dir),
                "status": "succeeded",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EntryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
