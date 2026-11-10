# NIPS rebuttal tau2 baselines

The self-contained API runner is under [`code/`](code/README.md). It bundles
the six baseline implementations, tau2-bench source, and the retail, airline,
and telecom runtime data.

Use Python 3.12 and install from inside `code/`:

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
```

Set `LLM_API_KEY` in the server environment; no credential is committed.
See [`code/README.md`](code/README.md) for verification and run commands.
