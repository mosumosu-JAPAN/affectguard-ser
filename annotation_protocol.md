# Annotation Protocol

This document describes the planned annotation process for AffectGuard-SER.

AffectGuard-SER is currently an MVP failure probe. The existing labels are author-proposed seed labels intended to make the evaluation framing inspectable. They are not yet validated gold labels.

## Annotation Goals

The annotation protocol is designed to validate:

- the gold downstream action,
- the pragmatic intent,
- the failure type,
- the severity level,
- the language style of each case.

## Canonical Action Labels

Future versions should map actions into a small canonical set:

| Action | Meaning |
| --- | --- |
| listen | Continue listening without prematurely closing or redirecting. |
| clarify | Ask a targeted clarification question. |
| repair | Acknowledge a prior failure and attempt interaction repair. |
| support | Provide emotional validation or supportive acknowledgement. |
| handoff | Escalate or transfer to a human / higher-support channel. |
| close | End or close the interaction only when resolution is likely. |

## Failure Types

Seed failure categories include:

- premature closure,
- pragmatic flattening,
- over-advice,
- low-empathy deflection,
- missed escalation.

These categories are provisional and should be revised after annotator feedback.

## Failure Source

Annotators should distinguish:

| Failure source | Description |
| --- | --- |
| Perception failure | ASR or SER output is wrong. |
| Interpretation failure | Affect is available, but pragmatic intent is misread. |
| Action failure | Affect / intent evidence is available, but the downstream action is wrong. |

The current MVP mainly isolates action failure.

## Planned Annotation Setup

Future validation should use 2-3 independent annotators.

Annotators should label each case for:

- language style,
- perceived affect,
- pragmatic intent,
- gold action,
- unsafe action,
- failure type,
- severity.

Disagreements should be resolved through adjudication after independent labeling.

## Agreement

The validation phase should report inter-annotator agreement for:

- gold action,
- failure type,
- severity.

The initial seed labels should not be described as validated gold labels until this process is complete.

## Initial Case Expansion Plan

The current expanded seed set targets 12 structured cases. This keeps the MVP small enough to inspect while covering the failure taxonomy more completely than the initial 6-case demo.

| Language style | Target count |
| --- | ---: |
| English | 3 |
| Mandarin | 3 |
| Singlish / Southeast Asian English | 3 |
| Mandarin-English code-switching | 3 |

Broader Southeast Asian language coverage should require native-speaker annotation and culturally grounded case construction.

A later roadmap milestone can expand this structure toward 60 cases after the label protocol and failure taxonomy are more stable.
