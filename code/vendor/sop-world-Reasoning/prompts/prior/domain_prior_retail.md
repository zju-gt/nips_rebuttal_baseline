# Domain Prior: Retail

- Authenticate first; a bare `user_id` is not enough.
- Pending orders can be canceled or modified; delivered orders can be returned or exchanged.
- Exchange/modify tools are one-shot: collect every item and option first, then call once.
- Always fetch product details; never guess `variant_id` or product/item mapping.
- State the action list and refund timeline precisely, then get explicit yes.
