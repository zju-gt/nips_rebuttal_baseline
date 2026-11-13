from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from api_config import load_api_config
from baseline_registry import DOMAIN_ORDER, get_baseline_specs, parse_selection
from run_tau2_baselines import CODE_DIR, build_child_env
from train_val_artifacts import export_harness, verify_harness
from train_val_split import canonical_hash, load_split, write_or_verify_split


METHODS = ("react", "awm", "reasoningbank", "living_harness")
TRAIN_METHODS = METHODS[1:]
LABELS = {
    "react": "GPT5.2-base",
    "awm": "AWM",
    "reasoningbank": "ReasoningBank",
    "living_harness": "Living-Harness",
}
TABLE_DOMAINS = ("airline", "retail", "telecom")


def _python_command_path(path: Path) -> Path:
    # Resolving the venv's python symlink selects the system interpreter.
    return path.expanduser().absolute()


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="70/30 train/val experiment on tau2 base tasks")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("split", "train", "val", "table", "all"), required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--domains", default=",".join(DOMAIN_ORDER))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--workspace-root", type=Path, default=CODE_DIR / "vendor")
    parser.add_argument("--tau2-repo", type=Path)
    parser.add_argument("--openai-test", type=Path, default=CODE_DIR / "openai_test.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _selection_file(split_dir: Path, split: dict[str, Any], phase: str, *, dry_run: bool) -> Path:
    path = split_dir / f"{split['domain']}.{phase}.json"
    expected = {"domain": split["domain"], "task_ids": split[f"{phase}_ids"], "split_hash": canonical_hash(split)}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"task selection changed: {path}")
    elif not dry_run:
        path.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _summary_ok(output_dir: Path, expected_tasks: int, expected_samples: int, *, frozen: bool) -> bool:
    path = output_dir / "summary.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return (
        int(value.get("infra_error_count", -1)) == 0
        and int(value.get("num_tasks", -1)) == expected_tasks
        and int(value.get("total_samples", -1)) == expected_samples
        and int(value.get("missing_reward_count", -1)) == 0
        and (not frozen or value.get("frozen") is True)
    )


def _run_one(
    *, method: str, domain: str, phase: str, run_dir: Path, selection: Path,
    split: dict[str, Any], workspace: Path, tau2_repo: Path, api: Any,
    python: Path, workers: int, dry_run: bool,
) -> None:
    spec = get_baseline_specs()[method]
    output_dir = run_dir / phase / method / domain
    cycles = 3 if phase == "train" and method == "living_harness" else 1
    expected_tasks = len(split[f"{phase}_ids"])
    harness = run_dir / "harnesses" / method / domain
    runner_source = workspace / spec.repo_name / "src" / "tau2_runner.py"
    if phase == "val" and method != "react" and not dry_run:
        verify_harness(harness, method=method, domain=domain, split=split, model=api.model, runner_source=runner_source)
    if output_dir.exists():
        if not _summary_ok(output_dir, expected_tasks, expected_tasks * cycles, frozen=phase == "val"):
            raise ValueError(f"partial or invalid job directory; preserve it and use a new run ID: {output_dir}")
        if phase == "train" and not harness.exists() and not dry_run:
            export_harness(
                train_dir=output_dir, harness_dir=harness, method=method, domain=domain,
                split=split, cycles=cycles, model=api.model,
                runner_source=runner_source,
            )
        if phase == "train" and harness.exists():
            verify_harness(harness, method=method, domain=domain, split=split, model=api.model, runner_source=runner_source)
        print(f"skip complete {phase} {method}/{domain}", flush=True)
        return
    print(f"{phase} {method}/{domain}: {expected_tasks} tasks x {cycles} cycle(s)", flush=True)
    if dry_run:
        return
    env = build_child_env(
        base_environ=os.environ,
        api_config=api,
        baseline=spec,
        domain=domain,
        output_dir=output_dir,
        tau2_repo=tau2_repo,
        workers=workers,
        runtime_shim=CODE_DIR / "runtime_shim",
        repo_src=workspace / spec.repo_name / "src",
    )
    env.update(
        TAU2_TASK_IDS_FILE=str(selection.resolve()),
        TAU2_TRAIN_VAL_PHASE=phase,
        NUM_CYCLES=str(cycles),
        PASS_HAT_K="1",
    )
    env.setdefault("TAU2_API_RPM_LIMIT", "20")
    if phase == "val":
        env.setdefault("SAMPLE_TIMEOUT_SECONDS", "2400")
    if phase == "val" and method != "react":
        env["TAU2_FROZEN_HARNESS_DIR"] = str(harness.resolve())
    command = [
        str(python), str(CODE_DIR / "tau2_baseline_entry.py"),
        "--repo", str(workspace / spec.repo_name),
        "--output-dir", str(output_dir),
        "--method", method,
    ]
    log_path = run_dir / "logs" / f"{phase}__{method}__{domain}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0 or not _summary_ok(output_dir, expected_tasks, expected_tasks * cycles, frozen=phase == "val"):
        raise RuntimeError(f"{phase} {method}/{domain} failed or had infrastructure errors; see {log_path}")
    if phase == "train":
        export_harness(
            train_dir=output_dir, harness_dir=harness, method=method, domain=domain,
            split=split, cycles=cycles, model=api.model,
            runner_source=runner_source,
        )
    elif method != "react":
        verify_harness(harness, method=method, domain=domain, split=split, model=api.model, runner_source=runner_source)


def _make_table(run_dir: Path, tau2_repo: Path) -> None:
    splits = {
        domain: load_split(run_dir / "split" / f"{domain}.json", domain=domain, tau2_repo=tau2_repo)
        for domain in TABLE_DOMAINS
    }
    rows = []
    for method in METHODS:
        rates = []
        for domain in TABLE_DOMAINS:
            output_dir = run_dir / "val" / method / domain
            expected = len(splits[domain]["val_ids"])
            if not _summary_ok(output_dir, expected, expected, frozen=True):
                raise ValueError(f"validation incomplete: {method}/{domain}")
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            simulations = results.get("simulations", [])
            ids = [str(item["task_id"]) for item in simulations]
            if len(ids) != expected or len(set(ids)) != expected or set(ids) != set(splits[domain]["val_ids"]):
                raise ValueError(f"validation result IDs mismatch: {method}/{domain}")
            if method != "react":
                harness = run_dir / "harnesses" / method / domain
                verify_harness(harness, method=method, domain=domain, split=splits[domain])
                from train_val_split import sha256_file
                if summary.get("harness_manifest_sha256") != sha256_file(harness / "artifact_manifest.json"):
                    raise ValueError(f"validation used a different harness: {method}/{domain}")
            rates.append(100.0 * int(summary["success_count"]) / expected)
        rows.append((LABELS[method], *rates, sum(rates) / len(rates)))
    header = "Method,Airline,Retail,Telecom,Avg"
    csv = "\n".join([header, *(f"{row[0]},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f}" for row in rows)]) + "\n"
    markdown = "| Method | Airline | Retail | Telecom | Avg |\n| --- | ---: | ---: | ---: | ---: |\n"
    markdown += "".join(f"| {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | {row[4]:.2f} |\n" for row in rows)
    (run_dir / "table.csv").write_text(csv, encoding="utf-8")
    (run_dir / "table.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if args.workers < 1 or not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
        raise ValueError("invalid workers or run ID")
    methods = parse_selection(args.methods, METHODS, "method")
    domains = parse_selection(args.domains, DOMAIN_ORDER, "domain")
    workspace = args.workspace_root.expanduser().resolve()
    tau2_repo = (args.tau2_repo or workspace / "tau2-bench").expanduser().resolve()
    run_dir = args.output_root.expanduser().resolve() / args.run_id
    if args.phase == "table":
        _make_table(run_dir, tau2_repo)
        return 0
    split_dir = run_dir / "split"
    splits: dict[str, dict[str, Any]] = {}
    for domain in domains:
        if args.dry_run:
            from train_val_split import expected_split
            splits[domain] = expected_split(tau2_repo, domain)
        else:
            path = write_or_verify_split(tau2_repo, split_dir, domain)
            splits[domain] = load_split(path, domain=domain, tau2_repo=tau2_repo)
        print(f"split {domain}: {len(splits[domain]['train_ids'])} train, {len(splits[domain]['val_ids'])} val")
    if not args.dry_run and set(domains) == set(DOMAIN_ORDER):
        manifest_path = split_dir / "split_manifest.json"
        manifest = {
            "version": "base-sha256-v1",
            "seed": 300,
            "domains": {
                domain: {
                    "split_hash": canonical_hash(splits[domain]),
                    "base": len(splits[domain]["base_ids"]),
                    "train": len(splits[domain]["train_ids"]),
                    "val": len(splits[domain]["val_ids"]),
                }
                for domain in TABLE_DOMAINS
            },
        }
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"split manifest changed: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.phase == "split":
        return 0
    api = load_api_config(args.openai_test.expanduser().resolve(), os.environ)
    phases = ("train", "val") if args.phase == "all" else (args.phase,)
    for phase in phases:
        selected_methods = [method for method in methods if phase != "train" or method in TRAIN_METHODS]
        for method in selected_methods:
            for domain in domains:
                selection = _selection_file(split_dir, splits[domain], phase, dry_run=args.dry_run)
                _run_one(
                    method=method, domain=domain, phase=phase, run_dir=run_dir,
                    selection=selection, split=splits[domain], workspace=workspace,
                    tau2_repo=tau2_repo, api=api, python=_python_command_path(args.python),
                    workers=args.workers, dry_run=args.dry_run,
                )
    if args.phase == "all" and set(methods) == set(METHODS) and set(domains) == set(DOMAIN_ORDER) and not args.dry_run:
        _make_table(run_dir, tau2_repo)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
