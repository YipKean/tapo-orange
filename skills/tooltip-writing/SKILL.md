---
name: tooltip-writing
description: Write and update tooltip copy for this project UI. Use when adding or revising entries in `scripts/i18n/tooltips.json`, or when UI controls in local Tkinter tools need concise operator-facing help text that matches this repo's style.
---

# Tooltip Writing

Store tooltip copy in `scripts/i18n/tooltips.json`, not inline in Python UI code.

Write in plain operator language first. Prefer clarity over internal implementation detail.

Keep each tooltip focused on one job:

* explain what the control does
* explain when to use it, if needed
* explain an important tradeoff or gotcha only when it materially helps

Prefer short 1-3 sentence tooltips.

Use this default structure:

1. First sentence: say what the control does.
2. Optional second sentence: say when to turn it on or off.
3. Optional third sentence: warn about a meaningful tradeoff or gotcha.

Avoid unexplained jargon when simpler wording works.

Avoid repeating the literal field label unless it improves clarity.

For advanced tuning fields, explain the practical effect rather than only the metric name.

When behavior differs between heuristic mode and classifier mode, say that explicitly so operators do not misread legacy field names.

Keep tooltip wording stable and reusable. Put one-off experiment notes, temporary caveats, and long troubleshooting detail in docs instead of tooltip text.
