# Unified tau2 API baseline runner

This directory is a self-contained, API-only Exp1 bundle for tau2-bench. It includes the runtime source and prompts for all six methods, the tau2-bench package, and the retail, airline, and telecom text data. Copying this `code/` directory is sufficient; sibling `sop-world*` or `tau2-bench` repositories are not required on the Alibaba server.

The supported runtime is Python 3.12, matching the bundled tau2-bench package metadata.

## Model roles

- Actor and evolution calls: `gpt-5.2-2025-12-11`
- tau2 user simulator and natural-language judge: `gpt-5.1-2025-11-13`
- Common seed: `300`

The endpoint and credential are loaded statically from `openai_test.py`; that file is parsed with Python AST and is never imported. These environment variables can override individual values on the Alibaba server:

```bash
export LLM_MODEL_NAME=gpt-5.2-2025-12-11
export LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export LLM_API_KEY=...
```

Credentials are passed only through child-process environments. They are omitted from manifests, rendered job configuration, commands, terminal output, and API usage records.

## Install after transfer

Transfer the entire `code/` directory through your normal secure channel. Do not copy only individual runner files: `vendor/`, `runtime_shim/`, and the shell/Python entry points are all required.

On the Alibaba server:

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
```

`requirements-api.txt` installs the bundled tau2 package in editable mode plus the small direct dependencies used by the six baseline trees. It intentionally excludes local-model, CUDA, voice, and MultiWOZ stacks.

Before a paid run, verify the transferred bundle and plan all 18 jobs without making an API call:

```bash
bash run_tau2_baselines.sh \
  --dry-run \
  --verify-bundle \
  --run-id transfer-check
```

## Commands

From `/home/duyuetian/projects` or any other directory:

```bash
bash sop-world-rebuttal/code/run_tau2_baselines.sh
```

Validate the complete 18-job matrix without starting a baseline or making an API call:

```bash
bash sop-world-rebuttal/code/run_tau2_baselines.sh \
  --dry-run \
  --verify-bundle \
  --run-id exp1-check
```

Run a subset:

```bash
bash sop-world-rebuttal/code/run_tau2_baselines.sh \
  --methods react,reflexion,evotest \
  --domains retail,telecom \
  --run-id exp1-subset
```

AWM, ReasoningBank, and EvoTest use asynchronous sample scheduling. Within one
method/domain job, up to `--workers` samples run at once. When a sample finishes,
its evolution update is applied and the next sample is submitted immediately
with the latest available state. Samples already in flight keep the state
snapshot they received when submitted. This avoids waiting for the slowest
sample in a fixed batch.

To rerun AWM/telecom and all ReasoningBank and EvoTest domains in sequence,
with eight sample workers per job:

```bash
bash run_tau2_baselines.sh \
  --jobs awm/telecom,reasoningbank/retail,reasoningbank/airline,reasoningbank/telecom,evotest/retail,evotest/airline,evotest/telecom \
  --workers 8 \
  --run-id evolve-retry
```

`run_evolve_remaining.sh` runs that seven-job selection with bundle verification
and a timestamped run ID. Jobs run one at a time; samples within each job run
asynchronously. Their API calls share a 20 requests/minute limit, configurable
with `TAU2_API_RPM_LIMIT`. HTTP 429 responses are retried after a pause. A job
with infrastructure errors is marked failed, and later jobs are not started.

`run_retail_first.sh` selects the unfinished ReasoningBank and EvoTest jobs,
running both retail domains before airline and telecom. It uses a new run ID
and keeps prior results intact. When the API rejects the `reasoning` parameter,
the runtime remembers that limitation for later calls to the same model route.

Useful options:

- `--verify-bundle`: recompute every vendored file hash and reject missing, added, or modified files before planning jobs.
- `--workspace-root`: explicit external directory containing the six `sop-world*` repositories and `tau2-bench`; by default the runner uses `code/vendor`.
- `--tau2-repo`: explicit tau2-bench checkout override; by default it is `<workspace-root>/tau2-bench`.
- `--output-root`: central result root; defaults to `code/results`.
- `--workers`: sets `PROC_PER_NODE`; defaults to `1`. AWM, ReasoningBank, and EvoTest use it for asynchronous sample workers.
- `--parallel-jobs`: runs independent method/domain jobs concurrently; defaults to `1`.
- `--jobs`: select exact `method/domain` pairs instead of a methods × domains matrix.
- Active runs show a tqdm bar for completed method/domain jobs, including the current job and failure count.
- Each worker sends and saves completed samples individually (`RESULT_BATCH_SIZE=1` and `RESULT_SAVE_EVERY=1` by default).
- `--continue-on-error`: continue to later jobs after a failed job.
- `--resume`: skip only job directories whose `summary.json` is valid.
- `--python`: Python executable used for child jobs.

Use `--help` for the complete CLI.

## Baseline matrix

The default order is method-major and then domain-minor:

1. ReAct
2. Reflexion
3. AWM
4. ReasoningBank
5. EvoTest
6. Living-Harness

Each runs `retail`, `airline`, and `telecom`. Method-specific mode, trial, memory, hint, SOP, and evolution flags are centralized in `baseline_registry.py`.

## 70/30 train/val experiment

The separate `run_train_val.sh` entry point splits only the existing tau2 `base`
tasks: Airline 35/15, Retail 80/34, and Telecom 80/34. It trains AWM,
ReasoningBank, and Living-Harness on the train IDs, exports a separate final
harness for each method/domain, and evaluates all four methods on the val IDs.
Validation renders prompts from the saved harness while disabling evolution
updates in the sample workers. It checks task IDs, missing rewards, and harness
hashes before producing `table.csv` and `table.md`.
Validation uses a 20 requests/minute API limit and a 2400-second per-sample
timeout by default. A throttled run can otherwise exhaust the older
800-second wall-clock limit while a dialogue waits for its next request.

```bash
cd code
bash run_train_val.sh --run-id train-val-base-300-v1 --phase all --dry-run
bash run_train_val.sh --run-id train-val-base-300-v1 --phase split
```

Run each domain and phase in the order shown in
[`docs/tau2_train_val_experiment_plan.md`](../docs/tau2_train_val_experiment_plan.md).
The script runs method/domain jobs sequentially; `--workers` controls sample
concurrency within a job. Existing Exp1 results remain under their own run IDs.

If a completed validation job has missing rewards or infrastructure errors,
retry only those task IDs and retain the original files:

```bash
.venv/bin/python repair_train_val.py \
  --run-id train-val-base-300-v1 --method react --domain retail
```

Retry results and API usage are stored under `val_retry/`; the original full
results and summary are copied to `results.initial.json` and
`summary.initial.json` before the repaired aggregate replaces them.

Memory-based methods use the repositories' existing DashScope embedding support when `DASHSCOPE_API_KEY` or `EMBEDDING_API_KEY` is available. Otherwise, their existing lexical fallback remains active.

## Refreshing the vendor bundle

Run synchronization only on the development machine where the seven source repositories are available:

```bash
cd /home/duyuetian/projects
PYTHONDONTWRITEBYTECODE=1 python3 \
  sop-world-rebuttal/code/sync_vendor_bundle.py \
  --source-workspace /home/duyuetian/projects
```

The synchronizer copies the current working-tree runtime files, writes `vendor/bundle_manifest.json`, audits the generated tree for the configured API key, verifies all hashes, and atomically replaces the previous generated bundle. The six method trees remain independent so their approved behaviors are not deduplicated or rewritten.

The generated bundle is approximately 32 MB. It excludes Git history, `.env*`, tests, caches, logs, prior results, voice assets, MultiWOZ data, local model weights, and development-only tau2 assets. To validate an existing bundle without running a baseline or making an API call:

```bash
bash run_tau2_baselines.sh \
  --dry-run \
  --verify-bundle \
  --run-id bundle-check
```

For local comparison against the original checkouts, override the default workspace explicitly:

```bash
bash run_tau2_baselines.sh \
  --workspace-root /home/duyuetian/projects \
  --tau2-repo /home/duyuetian/projects/tau2-bench \
  --dry-run \
  --run-id external-check
```

## Outputs and resume safety

```text
code/results/<run-id>/
  manifest.json
  logs/
    react__retail.log
  react/
    retail/
      summary.json
      api_usage.jsonl
      ... native baseline artifacts ...
  ...
```

The manifest is written before execution and atomically updated after every state transition. A job is `pending`, `running`, `succeeded`, `failed`, or `skipped`.

An existing run directory is rejected unless `--resume` is supplied. Resume skips a job only when its `summary.json` parses as a JSON object. A non-empty partial directory or malformed summary is rejected and left untouched; the runner never deletes paid-run artifacts automatically.

Resume also loads the existing manifest and rejects incompatible model, endpoint, worker, path, or method-environment changes. Unselected job records are preserved. `--resume --dry-run` validates and reports the plan without rewriting the existing manifest.

## API usage records

Every logical LiteLLM attempt appends one JSON object to the job's `api_usage.jsonl`. Successful records can contain:

- timestamp, status, model, and call name;
- latency and provider response ID;
- prompt, completion, and total tokens;
- cached input and reasoning tokens when provided.

Failed records contain only the exception class and numeric status code when available. Prompts, messages, tools, authorization headers, API keys, and exception response bodies are never stored.

## Verification

The test suite uses fake repositories, fake modules, and local child processes. It does not issue paid or network API requests:

```bash
cd /home/duyuetian/projects
PYTHONDONTWRITEBYTECODE=1 pytest -q sop-world-rebuttal/code/tests
```
