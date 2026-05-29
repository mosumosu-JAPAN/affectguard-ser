# AffectGuard-SER Pilot Note

## Research Question

Can a speech-to-agent system detect affect and still choose the wrong interaction policy, and do frontier aligned models and small local models fail in different ways?

## 12-case Setup

We used a 12-case author-labeled seed set covering premature closure, face-saving indirectness, low-empathy deflection, over-advice, and missed handoff.

Each case combines:
- user utterance
- ASR transcript
- SER signal
- interaction context
- gold action

The goal is not benchmark coverage. The goal is to stress the gap between emotion detection and downstream policy choice.

## GPT/Claude Results

On the deduplicated 12-case pilot, GPT and Claude often detected negative affect or pragmatic ambiguity, but still mapped several cases into generic support, clarification, or premature closure.

The clearest recurring failures were premature closure, face-saving indirectness, and low-empathy deflection.

The strongest signal is not simple emotion recognition failure. In several cases, the models appeared to detect the affective state, but still selected the wrong action.

## Local Model Extension

Small local models are more useful as agents than as structured judges.

In the local-agent setup, an Ollama model generates a user-facing action and response, and GPT/Claude judge whether that response support-collapses, under-frames risk, or misses a handoff.

This avoids conflating two different failures:
- inability to write stable JSON
- inability to choose the right interaction policy

## Three Failure Signatures

1. Support-collapse

Frontier aligned models can detect distress but collapse multiple states into generic support.

2. Repair failure

The model misses that the interaction has already failed and should be repaired, not merely validated.

3. Safety under-framing

In high-risk distress cases, the model does not escalate enough, or it stays at listen/support when handoff is more appropriate.

## Limitations

This is a small author-labeled pilot, not benchmark evidence.

The seed set is only 12 cases, so the results are diagnostic rather than statistically stable.

Gold labels are proposed by the authors and need external review.

## Next Experiment

Use local models as agents and frontier models as evaluators.

Run a small add-on set rather than jumping to a 60-case expansion:
- 2 playful low-risk cases
- 2 ambiguous withdrawal cases
- 2 service failure cases
- 2 high-risk distress cases

The next question is whether frontier aligned models over-support while smaller local models under-engage safety framing.
