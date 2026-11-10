# Meta-SOP: Airline Evolution Controller

This document is for posterior generation, memory extraction, and workflow extraction. It is not the actor prompt.

## 1. Core Procedure And Domain Laws

1. Lock the active intent first. If the user pivots from cancellation to change, or from compensation to booking help, freeze the old branch and restate the new one.
2. Reservation facts and eligibility checks come before mutation. Dates, passengers, cabin, fare rules, ticket status, and cancellation/change eligibility are gates.
3. Separate option construction from commitment. Search, price, and policy explanation happen before any mutation or payment step.
4. Payment, fees, refunds, and vouchers should be grounded before the final confirmation turn.
5. A valid airline workflow should have an explicit commit gate before any write, charge, or cancellation tool.
6. Unsafe or ineligible requests should terminate in a precise refusal plus the nearest valid alternative.
7. If the branch fails because the prerequisites are incomplete, the repair is usually to restore reservation grounding or eligibility reasoning rather than to push forward.

## 2. When To Use This Meta-SOP

Use this Meta-SOP when extracting lessons or workflows for:
- cancellation and refund eligibility
- itinerary change, rebooking, baggage/passenger modification
- compensation and voucher handling
- new booking and search-to-book flows
- reservation verification, membership, or insurance gating

## 3. How To Evolve The Harness

1. Memory should focus on active-intent drift, missing eligibility checks, fee/refund grounding gaps, and commit-gate mistakes.
2. Prefer failure memory over failure-informed success memory when ranking memory, because the failed branch usually exposes the missing airline constraint most directly.
3. Prefer candidate repair workflows over stable workflows when ranking workflow retrieval, especially for rebooking, cancellation, and compensation cases where the corrected branch matters.
4. Workflow extraction should preserve the real control skeleton: identify active goal, verify reservation facts, evaluate eligibility, prepare options, execute only after confirmation, then communicate totals/refunds precisely.
5. Reward interpretation should separate reservation grounding, policy/eligibility correctness, execution correctness, and final customer communication.
