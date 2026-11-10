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

Useful options:

- `--verify-bundle`: recompute every vendored file hash and reject missing, added, or modified files before planning jobs.
- `--workspace-root`: explicit external directory containing the six `sop-world*` repositories and `tau2-bench`; by default the runner uses `code/vendor`.
- `--tau2-repo`: explicit tau2-bench checkout override; by default it is `<workspace-root>/tau2-bench`.
- `--output-root`: central result root; defaults to `code/results`.
- `--workers`: sets `PROC_PER_NODE`; defaults to `1`. Stateful methods retain strict update ordering.
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
