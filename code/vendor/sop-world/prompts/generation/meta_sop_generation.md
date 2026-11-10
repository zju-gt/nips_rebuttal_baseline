# Meta-SOP Generation

Write `meta_sop.md` for the target domain.

This Meta-SOP is for evolution-time analysis only. It is not the actor prompt.

Inputs:
- Domain policy:
  {{DOMAIN_POLICY}}
- Representative tasks:
  {{REPRESENTATIVE_TASKS}}
- Benchmark prior:
  {{BENCHMARK_PRIOR}}
- Domain prior:
  {{DOMAIN_PRIOR}}

Output requirements:
- Output markdown only.
- Write three sections with these exact headings:
  `## 1. Core Procedure And Domain Laws`
  `## 2. When To Use This Meta-SOP`
  `## 3. How To Evolve The Harness`

Rules:
- Focus on ordering, gating, verification, branch switching, commit gates, and stop conditions.
- Keep it compact but extractor-usable; do not rewrite the full policy text.
- The first section should capture durable control laws and common failure pivots.
- The second section should describe the task families or situations where this Meta-SOP should shape posterior/memory/workflow extraction.
- The third section should explain how memory, workflow/state-graph evolution, and reward interpretation should improve over time.
- Favor concrete domain rules over generic agent slogans.
