#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python environment not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

RUN_ID="exp1-evolve-retail-first-$(date -u +%Y%m%d-%H%M%S)"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export RESULT_BATCH_SIZE=1
export RESULT_SAVE_EVERY=1

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$HOME/.config"
  printf '%s\n' "$RUN_ID" > "$HOME/.config/nips_rebuttal_tau2_last_run"
fi

PYTHON_BIN="$PYTHON_BIN" bash run_tau2_baselines.sh \
  --python "$PYTHON_BIN" \
  --jobs reasoningbank/retail,evotest/retail,reasoningbank/airline,reasoningbank/telecom,evotest/airline,evotest/telecom \
  --workers 8 \
  --verify-bundle "$@" \
  --run-id "$RUN_ID"
