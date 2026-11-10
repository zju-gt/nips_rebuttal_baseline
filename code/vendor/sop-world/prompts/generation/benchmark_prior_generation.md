# Benchmark Prior Generation

Write a compact `benchmark_prior.md` for tau2-bench.

Inputs:
- Domain policy:
  {{DOMAIN_POLICY}}
- Representative tasks:
  {{REPRESENTATIVE_TASKS}}

Rules:
- Output markdown only.
- Keep only benchmark-wide controls.
- Make the prior short, failure-driven, and reusable across domains.
- Include the core rules for one tool call per turn, verify before acting, refusal, and exact numbers.
