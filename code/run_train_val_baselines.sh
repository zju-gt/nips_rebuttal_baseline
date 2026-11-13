#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-train-val-base-300-v1}"
WORKERS="${2:-10}"
export TAU2_API_RPM_LIMIT="${TAU2_API_RPM_LIMIT:-20}"
export SAMPLE_TIMEOUT_SECONDS="${SAMPLE_TIMEOUT_SECONDS:-2400}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$TAU2_API_RPM_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'WORKERS and TAU2_API_RPM_LIMIT must be positive integers\n' >&2
    exit 1
fi

RUN_DIR="$SCRIPT_DIR/results/$RUN_ID"
mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/logs/baselines_pipeline.log") 2>&1

for method in awm reasoningbank living_harness; do
    for domain in airline retail telecom; do
        train_dir="$RUN_DIR/train/$method/$domain"
        harness_dir="$RUN_DIR/harnesses/$method/$domain"
        if [[ -f "$train_dir/summary.json" ]]; then
            incomplete=$("$SCRIPT_DIR/.venv/bin/python" - "$train_dir/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(summary.get("missing_reward_count", 0)) + int(summary.get("infra_error_count", 0)))
PY
)
            if (( incomplete > 0 )); then
                attempt=1
                archive="$RUN_DIR/train_attempts/$method/$domain/attempt_$attempt"
                while [[ -e "$archive" ]]; do
                    ((attempt += 1))
                    archive="$RUN_DIR/train_attempts/$method/$domain/attempt_$attempt"
                done
                mkdir -p "$archive"
                mv "$train_dir" "$archive/train"
                if [[ -d "$harness_dir" ]]; then
                    mv "$harness_dir" "$archive/harness"
                fi
                if [[ -f "$RUN_DIR/logs/train__${method}__${domain}.log" ]]; then
                    mv "$RUN_DIR/logs/train__${method}__${domain}.log" "$archive/train.log"
                fi
                printf 'Archived incomplete training at %s; rerunning with %ss timeout\n' \
                    "$archive" "$SAMPLE_TIMEOUT_SECONDS"
            fi
        fi

        printf '\n== %s/%s: train and save harness ==\n' "$method" "$domain"
        bash "$SCRIPT_DIR/run_train_val.sh" \
            --run-id "$RUN_ID" --phase train --methods "$method" \
            --domains "$domain" --workers "$WORKERS"

        test -f "$harness_dir/artifact_manifest.json"
        "$SCRIPT_DIR/.venv/bin/python" - "$train_dir/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if int(summary.get("missing_reward_count", -1)) or int(summary.get("infra_error_count", -1)):
    raise SystemExit("training ended with missing rewards or infrastructure errors")
PY

        printf '\n== %s/%s: frozen validation ==\n' "$method" "$domain"
        if ! bash "$SCRIPT_DIR/run_train_val.sh" \
            --run-id "$RUN_ID" --phase val --methods "$method" \
            --domains "$domain" --workers "$WORKERS"; then
            printf 'Validation needs repair; retrying only invalid tasks for %s/%s\n' "$method" "$domain"
            last_attempt=0
            for prior in "$RUN_DIR/val_retry/$method/$domain"/attempt_*; do
                [[ -d "$prior" ]] || continue
                attempt_number="${prior##*_}"
                if [[ "$attempt_number" =~ ^[1-9][0-9]*$ ]] && (( attempt_number > last_attempt )); then
                    last_attempt="$attempt_number"
                fi
            done
            max_attempts=$((last_attempt + 3))
            "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/repair_train_val.py" \
                --run-id "$RUN_ID" --method "$method" --domain "$domain" \
                --workers "$WORKERS" --rpm-limit "$TAU2_API_RPM_LIMIT" \
                --max-attempts "$max_attempts"
            bash "$SCRIPT_DIR/run_train_val.sh" \
                --run-id "$RUN_ID" --phase val --methods "$method" \
                --domains "$domain" --workers "$WORKERS"
        fi
    done
done

bash "$SCRIPT_DIR/run_train_val.sh" --run-id "$RUN_ID" --phase table
printf 'All train/val baseline jobs completed: %s\n' "$RUN_ID"
