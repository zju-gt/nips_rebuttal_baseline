# Meta-SOP Generation

Write a compact `meta_sop.md` for the target domain.

Inputs:
- Domain policy:
  {{DOMAIN_POLICY}}
- Representative tasks:
  {{REPRESENTATIVE_TASKS}}
- Benchmark prior:
  {{BENCHMARK_PRIOR}}
- Domain prior:
  {{DOMAIN_PRIOR}}

Rules:
- Output markdown only.
- Keep the controller immutable and short.
- Focus on ordering, gating, verification, and refusal behavior.
- Do not re-expand the full policy text.
- Preserve the top-level control logic that helps later memory/state-graph updates, especially task freezing, one-shot planning, and stop conditions.
