# Domain Prior: Airline

- Classify the request first: book, modify, cancel, compensate.
- Freeze the current goal when the user changes topic.
- Before any write or payment step, collect reservation id, dates, segments, cabin, passengers, and payment limits.
- Keep cabin and passenger count consistent; do not split one reservation across cabins.
- For multi-step changes, compute totals first, then submit once.
- For cancel/compensation, verify eligibility and refund path first.
