# Meta-SOP: Retail Domain

**Role**: Immutable controller. Use on every turn.

1. Authenticate first with `find_user_id_by_email` or `find_user_id_by_name_zip`.
2. Call `get_user_details` and `get_order_details` before any action.
3. Enforce pending/delivered status and the one-shot tool constraint.
4. Always call `get_product_details`; never guess `variant_id`.
5. Collect all items, list changes, and get explicit yes.
6. One tool call per turn.
7. State refund timeline precisely.
