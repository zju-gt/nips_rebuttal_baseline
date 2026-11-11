# Experiment results

This directory contains the original run outputs committed on 2026-11-11 for
analysis. It includes completed, interrupted, failed, and dry-run directories.
The active run `exp1-evolve-async-20261111-062736` was still running when this
version was captured, so its ReasoningBank/retail files are partial.

Use each run's `manifest.json` and each completed job's `summary.json` to check
status and `infra_error_count` before comparing rewards. `api_usage.jsonl`
contains per-call model, token usage, latency, and error status for token and
cost analysis. Raw `results.json`, evolution state, events, artifacts, and logs
are preserved alongside them.

The `.api_next_request` files used internally for API rate limiting are not
experiment results and are excluded from this version.
