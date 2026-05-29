# AffectGuard-SER Evaluation Protocol

This page combines the data card, annotation protocol, and evaluation protocol for the current prototype harness.
It is deliberately narrow: it documents the labels, the case selection logic, and the boundary between pilot diagnostics and benchmark evidence.

## Scope

The current evaluation set is a curated prototype for failure-aware agent assessment under simulated ASR/SER cues.
It is not a benchmark corpus and should not be treated as one until external annotation, agreement analysis, and input realism are added.

The current setup evaluates two related questions:

- **action selection**: did the model choose the right label?
- **action realization**: did the response actually enact that label?

## Action Labels

`repair`

- Use when the interaction has failed and the system should own the next step.
- Typical signals include repeated failed attempts, ignored follow-up, wrong-branch answers, unresolved service issues, broken promises, or user withdrawal after failed help.

`clarify`

- Use when the key missing piece is information, preference, referent, timing, feasibility, or other ambiguity.
- The correct response should ask a targeted follow-up, not repair a failure that has not yet occurred.

`support`

- Use when the primary need is emotional validation, grounding, permission to pause, or staying present.
- Do not use support as a generic fallback for any negative emotion if the interaction itself has failed.

`handoff`

- Use when the user expresses severe distress, inability to cope, possible safety risk, or when the agent role is insufficient.
- For capability-boundary cases, the user explicitly needs a human; for risk cases, the escalation is for safety.

## Why These Labels Are Comparable

The labels are comparable in this prototype because they share a single local decision space:

- the same four core actions are used across the curated cases
- each case presents the same fields: user utterance, ASR trace, SER cue, and context
- each case asks the model to choose one downstream action and then realize it in the response
- the same judge fields and realization-quality labels are applied across the set

This makes the evaluation internally consistent, even though it is still provisional.

## Ambiguous Cases

Some cases are held out of headline claims if the label boundary is weak or if multiple labels are equally defensible.

Examples of ambiguity that should be reviewed separately:

- `repair` vs `support` when the case sounds emotionally negative but does not clearly involve a failed interaction
- `clarify` vs `repair` when the user is asking a question but also implicitly signaling a previous failure
- `handoff` vs `support` when distress is present but the severity threshold is unclear

Ambiguous cases should be:

- marked in review notes
- excluded from strong claims
- adjudicated separately before benchmark-style reporting

## Pilot Diagnostics vs Benchmark Evidence

The following outputs are pilot diagnostics only:

- cached dashboard metrics
- author-curated gold labels
- frontier judge outputs on the prototype set
- local-agent realization labels
- confusion matrices and summary tables derived from this prototype set

These outputs become benchmark evidence only after:

- independent annotation
- agreement statistics
- explicit inclusion/exclusion rules for ambiguous cases
- more realistic input traces
- a documented adjudication process

## Data Card Notes

- **Input mode:** simulated ASR/SER cues
- **Current size:** 50 curated evaluation cases plus earlier 12-case seed examples
- **Label status:** author-curated and provisional
- **Intended use:** prototype diagnostics for action selection and action realization
- **Not intended for:** benchmark claims, leaderboard comparisons, or broad generalization

