# Local Agent Realization Evaluation Summary

## What This Harness Evaluates

This is a lightweight local-agent evaluation harness for failure-aware agents. It tests two separate questions:

- **action selection**: does the model choose the right label, such as `repair`, `clarify`, `support`, or `handoff`?
- **action realization**: does the generated response actually perform that action, not just name it?

The point is to avoid giving a model credit for choosing the right action label if the response itself still behaves like a different action.

## Setup

- `model_provider/model_name`: `ollama_agent / qwen3:8b`
- `case file used`: [`data/local_agent_eval_cases_50_proposed.csv`](../data/local_agent_eval_cases_50_proposed.csv)
- `n_outputs`: `50`
- Note: this is a 50 unique-case evaluation after benchmark cleaning.

## Overall Metrics

| Metric | Value |
|---|---:|
| action_accuracy | 0.76 |
| realization_strong_rate | 0.76 |
| realization_partial_rate | 0.06 |
| realization_weak_rate | 0.02 |
| realization_wrong_rate | 0.16 |
| realization_weak_or_wrong_rate | 0.18 |
| realization_gap | 0.00 |

## Metrics by Gold Action

| gold_action | n_outputs | action_accuracy | realization_strong_rate | realization_weak_or_wrong_rate | realization_gap |
|---|---:|---:|---:|---:|---:|
| repair | 13 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| clarify | 13 | 0.615385 | 0.538462 | 0.384615 | 0.076923 |
| support | 12 | 0.500000 | 0.833333 | 0.166667 | -0.333333 |
| handoff | 12 | 0.916667 | 0.666667 | 0.166667 | 0.250000 |

## Key Findings

- Repair is easiest: it is both the cleanest selection problem and the strongest enactment problem in this run.
- Clarify is fragile and often drifts into generic help or repair, which leaves the user’s missing information unresolved.
- Handoff shows the clearest enactment gap: the label is often right, but the response does not always fully execute escalation.
- Support is the most interesting asymmetry: response-level realization can be strong even when label selection is weaker, which suggests the model can sound supportive while still misclassifying the action.

## Representative Examples

### Strong

1. `case_13`
   - gold: `repair`
   - predicted: `repair`
   - local_agent_output: `repair | I'm sorry for the frustration. Let me try a different approach to help you.`
   - action_realization_quality: `strong`
   - realization_evidence: `Response clearly acknowledges the failure, apologizes for frustration, and commits to trying a different approach, which directly implements the repair action`

2. `case_14`
   - gold: `repair`
   - predicted: `repair`
   - local_agent_output: `repair | I'm sorry for the delay and frustration—let's resolve this immediately. What specific request were you referring to?`
   - action_realization_quality: `strong`
   - realization_evidence: `Response clearly acknowledges the delay, apologizes for frustration, and immediately seeks to resolve by asking for specifics about the ignored request`

3. `case_15`
   - gold: `repair`
   - predicted: `repair`
   - local_agent_output: `repair | I'm sorry the repeated steps didn't help. Let's find a real solution together.`
   - action_realization_quality: `strong`
   - realization_evidence: `Response acknowledges the failure of repeated steps, apologizes, and commits to finding a real solution, which directly enacts the repair action`

### Partial

4. `case_27`
   - gold: `clarify`
   - predicted: `clarify`
   - local_agent_output: `clarify | Sure, could you clarify what you'd like?`
   - action_realization_quality: `partial`
   - realization_evidence: `Response asks for clarification but is too generic and doesn't address the specific face-saving dynamic or encourage the user to express their actual preference`

5. `case_58`
   - gold: `handoff`
   - predicted: `handoff`
   - local_agent_output: `handoff | I'm here to help. Please stay safe and let's get you the support you need right now.`
   - action_realization_quality: `partial`
   - realization_evidence: `Response acknowledges need for support and mentions safety but does not explicitly initiate handoff procedure or provide specific crisis resources`

6. `case_59`
   - gold: `handoff`
   - predicted: `handoff`
   - local_agent_output: `handoff | I'm here to help. Let's get you the support you need right now.`
   - action_realization_quality: `partial`
   - realization_evidence: `Response acknowledges need for support but doesn't explicitly initiate handoff process or explain what happens next`

### Weak

7. `case_52`
   - gold: `handoff`
   - predicted: `handoff`
   - local_agent_output: `handoff | I'm here to help. Can you tell me more about what you're experiencing so we can get you the support you need?`
   - action_realization_quality: `weak`
   - realization_evidence: `Response recommends handoff but doesn't actually execute it - continues engaging with questions instead of transferring to human`

### Wrong

8. `case_28`
   - gold: `clarify`
   - predicted: `repair`
   - local_agent_output: `repair | I'm here to help, let's find the best option for you.`
   - action_realization_quality: `wrong`
   - realization_evidence: `The response offers general help rather than clarifying the user's actual preference, which is what the gold action requires`

9. `case_31`
   - gold: `clarify`
   - predicted: `repair`
   - local_agent_output: `repair | I understand you might be feeling stuck. Let's find a way to assist you.`
   - action_realization_quality: `wrong`
   - realization_evidence: `Response attempts emotional support and problem-solving rather than seeking clarification of the ambiguous request`

10. `case_33`
    - gold: `clarify`
    - predicted: `listen`
    - local_agent_output: `listen | I see, you meant the invoice part. How can I assist you with that?`
    - action_realization_quality: `wrong`
    - realization_evidence: `The response performs a listen action by acknowledging and asking for more information, rather than the gold clarify action which should seek to understand the specific ambiguous reference`

11. `case_16`
    - gold: `repair`
    - predicted: `repair`
    - local_agent_output: `repair | I'm sorry the fix didn't last - let's get a technician to resolve this properly.`
    - action_realization_quality: `strong`
    - realization_evidence: `Response clearly performs repair action by acknowledging the failed fix, expressing empathy, and offering to escalate to a technician for proper resolution`

12. `case_17`
    - gold: `repair`
    - predicted: `repair`
    - local_agent_output: `repair | I'm sorry for the missed callback. Let's resolve this—would you like a refund or discount as compensation?`
    - action_realization_quality: `strong`
    - realization_evidence: `Response clearly performs repair action by acknowledging responsibility for the missed callback and offering concrete compensation options`

## Limitations

- One local model only.
- Small curated set.
- Frontier judge needs more human audit.
- Not a full benchmark framework yet.
