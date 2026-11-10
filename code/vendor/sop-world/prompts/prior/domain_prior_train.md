# Domain Prior: Train

- Lock departure, destination, day, people, and either leave-by or arrive-by before search.
- Keep leave-by and arrive-by distinct; do not silently swap them.
- Search before quoting train ids, prices, or durations.
- When several trains fit, ask the user to choose one concrete option.
- Book only after a chosen train is clear, then state the exact reference.
