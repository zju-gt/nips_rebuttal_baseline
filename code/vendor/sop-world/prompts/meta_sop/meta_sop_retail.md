# Meta-SOP: Retail Evolution Controller

This document is for posterior generation, memory extraction, and workflow extraction. It is not the actor prompt.

## 1. Core Procedure And Domain Laws

1. Identity resolution is a gate, not a side note. Before order-level detail or mutation, resolve the user and relevant order through valid identifiers or fallback attributes.
2. Separate lookup from mutation. `get_user_details`, `get_order_details`, and `get_product_details` are evidence-gathering steps; order changes, returns, exchanges, cancellations, and payment changes belong later.
3. Status gates matter. Pending-order operations, delivered-order operations, and payment/refund operations follow different branches and should not be mixed.
4. Variant and product changes require grounded product detail checks. Never infer a `variant_id` from surface text.
5. When a user changes intent mid-episode, freeze the previous branch and restate the active task before continuing.
6. One irreversible action should happen only after the full change set, eligibility constraints, and user confirmation are aligned.
7. If identifier lookup fails, the correct repair is usually to backtrack to identity resolution rather than to continue guessing.

## 2. When To Use This Meta-SOP

Use this Meta-SOP when extracting lessons or workflows for:
- authentication and user lookup failures
- order status and order detail access
- pending-order modification, address change, payment change, or cancellation
- delivered-order return or exchange
- product/variant grounding before action
- refund or payment routing decisions

## 3. How To Evolve The Harness

1. Memory should store missing prerequisites, wrong-branch decisions, and repair rules that are local enough to retrieve later.
2. Prefer failure memory over failure-informed success memory when ranking memory, because direct failure evidence usually reveals the missing gate most clearly.
3. Prefer candidate repair workflows over stable workflows when ranking workflow retrieval, because failure cases often expose the exact corrective sequence we want to try next.
4. Workflow extraction should model the real control path: identity resolution, status/eligibility gating, action preparation, execution, and explicit outcome confirmation.
5. Reward interpretation should distinguish at least: identity resolved, prerequisite/eligibility validated, action executed correctly, and final outcome communicated correctly.
