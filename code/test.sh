#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python environment not found: %s\nCreate it with Python 3.12 and install requirements-api.txt (see README.md).\n' "$PYTHON_BIN" >&2
  exit 1
fi
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done
if [[ "$DRY_RUN" == 0 ]] && ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = ("tau2", "loguru", "litellm", "openai", "tqdm")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"Missing Python modules: {', '.join(missing)}", file=sys.stderr)
    print("Install requirements-api.txt into the selected Python environment.", file=sys.stderr)
    raise SystemExit(1)
PY
then
  exit 1
fi

RUN_ID="exp1-react-gpt52-$(date -u +%Y%m%d-%H%M%S)"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export RESULT_BATCH_SIZE=1
export RESULT_SAVE_EVERY=1
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$HOME/.config"
  printf '%s\n' "$RUN_ID" > "$HOME/.config/nips_rebuttal_tau2_last_run"
fi

# --methods react,reflexion,awm,reasoningbank,evotest,living_harness
PYTHON_BIN="$PYTHON_BIN" bash run_tau2_baselines.sh \
  --python "$PYTHON_BIN" \
  --methods reflexion,awm,reasoningbank,evotest \
  --domains retail,airline,telecom \
  --workers 16 \
  --continue-on-error \
  --verify-bundle "$@" \
  --run-id "$RUN_ID"
