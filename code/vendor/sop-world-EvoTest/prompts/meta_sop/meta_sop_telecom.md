# Meta-SOP: Telecom Evolution Controller

This document is for posterior generation, memory extraction, and workflow extraction. It is not the actor prompt.

## 1. Core Procedure And Domain Laws

1. Start by grounding the customer, line, and current service state before attempting a fix.
2. Treat troubleshooting as an ordered workflow, not a bag of isolated tips. Baseline connectivity checks usually come before advanced MMS, roaming, or APN branches.
3. Multiple faults may coexist. Do not prematurely stop after the first apparent fix if the task still requires broader validation.
4. Roaming and APN issues often require paired control steps: backend setting plus device-side change, then verification.
5. Reboot, toggle, or reconnect steps are meaningful workflow states and should not be skipped in extraction.
6. Verification after each meaningful intervention is part of the workflow, not an optional epilogue.
7. If baseline checks are green and the targeted capability still fails, escalation is often the correct terminal branch.

## 2. When To Use This Meta-SOP

Use this Meta-SOP when extracting lessons or workflows for:
- mobile data troubleshooting
- MMS send/receive failures
- roaming enablement and international-use issues
- billing, suspension, and account-payment related service blocks
- APN, reboot, or network-preference recovery paths

## 3. How To Evolve The Harness

1. Memory should capture branch-specific missing checks, wrong troubleshooting order, forgotten reboots/toggles, and incorrect early exits.
2. Prefer failure memory over failure-informed success memory when ranking memory, because telecom failures often pinpoint the exact skipped check or missing branch.
3. Prefer candidate repair workflows over stable workflows when ranking workflow retrieval, because troubleshooting gains often come from corrected sequences discovered after failure.
4. Workflow extraction should preserve the real control chain: establish baseline, isolate branch, apply intervention, verify, and escalate only when the verified path is exhausted.
5. Reward interpretation should distinguish baseline verification, branch-specific correction, post-fix verification, and correct escalation behavior.
