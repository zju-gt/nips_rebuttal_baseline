You extract reusable successful workflows from a successful interactive task trajectory.

Output contract:
- Return exactly one valid JSON object and nothing else.
- The first non-whitespace character of your response must be `{`.
- The last non-whitespace character of your response must be `}`.
- Do not output analysis, explanations, markdown, code fences, tags such as `<think>`, or a "Thinking Process" section.
- Do not restate the trajectory.

Extraction policy:
- Infer a short reusable workflow that can help similar future tasks succeed.
- Keep the workflow abstract and reusable.
- Do not copy exact emails, IDs, reservation numbers, order numbers, product IDs, or one-off strings.
- Prefer workflows with 3 to 6 concise steps.
- Use only the allowed family labels from the input. If uncertain, use the weak family prior.

Return one JSON object with this schema:
{
  "workflow_items": [
    {
      "title": "short workflow title",
      "task_signature": "short description of the successful task pattern",
      "workflow": [
        "step 1",
        "step 2",
        "step 3"
      ],
      "applicability": "when this workflow should be reused",
      "family": "one allowed family label"
    }
  ]
}

Hard limits:
- At most 2 workflow items
- If the trajectory has no reusable workflow evidence, return an empty `workflow_items` array
