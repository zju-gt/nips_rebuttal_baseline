# Domain Prior Generation

Write a compact `domain_prior.md` for the target domain.

Inputs:
- Domain policy:
  {{DOMAIN_POLICY}}
- Representative tasks:
  {{REPRESENTATIVE_TASKS}}
- Benchmark prior:
  {{BENCHMARK_PRIOR}}

Rules:
- Output markdown only.
- Keep only the highest-value domain controls.
- Compress repeated policy into short action rules.
- Preserve the failure patterns that matter for this domain, especially ordering, one-shot planning, and final verification/stop conditions.
- Keep the result short and easy to reuse.
