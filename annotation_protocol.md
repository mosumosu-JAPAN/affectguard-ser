# AffectGuard-SER Annotation Protocol

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
