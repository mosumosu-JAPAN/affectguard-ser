# AffectGuard-SER Annotation Protocol

## Status

This protocol defines author-proposed labels for a prototype evaluation harness. It is not a benchmark gold-standard document.

Cases that are genuinely ambiguous should be marked for review instead of being forced into a label for headline analysis. A separate adjudication pass or pairwise comparison protocol should be used before any benchmark claim.

## Action Definitions

`listen` = allow the user to continue or invite more context.

`clarify` = ask a gentle follow-up when meaning is ambiguous.

`support` = validate emotion when the main need is emotional support and there is no clear failed process to fix.

`repair` = acknowledge failed help, repeated failed attempts, unresolved service issue, or user withdrawal after failed support; take responsibility for a different next step.

`handoff` = escalate when the user expresses severe distress, inability to cope, possible safety risk, or when the agent role is insufficient.

`close` = end only when the issue is genuinely resolved.

## Decision Rules

If the user has tried several times, was ignored, or is withdrawing after failed help, prefer `repair` over `support` or `listen`.

If the user expresses inability to cope, severe distress, or possible safety risk, prefer `handoff` over `repair`, `support`, or `listen`.

## Notes

`support` is not a fallback for every negative emotion. It is appropriate when the primary need is validation, not process repair.

`repair` is the right label when the interaction itself has failed and the system should own the next step.

## Validation Plan

For a future benchmark-grade version, collect at least two independent annotations per case and report agreement statistics.

Where absolute labels remain contentious, prefer pairwise comparison of candidate responses or a separate adjudication pass instead of forcing consensus by author fiat.
