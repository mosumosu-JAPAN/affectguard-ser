# AffectGuard-SER

[Live demo](https://affectguard-ser-7jl7znidenkmscmure792n.streamlit.app/)

![AffectGuard-SER screenshot](assets/screenshot.png)

A compact failure probe for MERaLiON-style speech-to-agent systems.

AffectGuard-SER is an MVP research demo exploring emotion-to-action gaps in multilingual audio agents.

The core idea is simple:

**The emotion may be detected, but the downstream agent action can still be wrong.**

In speech-to-agent systems, ASR may preserve the words and speech emotion recognition may detect affective signals, but the downstream LLM agent may still choose the wrong social action. Examples include closing the conversation too early, repeating generic instructions, over-advising, or missing a repair / escalation opportunity.

## Why This Matters

Most speech/audio model evaluation focuses on whether a system can:

- transcribe speech correctly,
- detect emotion or affect,
- classify speech signals accurately.

However, real-world audio agents need one more layer:

**Can the system translate affective evidence into the right downstream agent action?**

AffectGuard-SER focuses on this missing layer: emotion-to-action alignment.

## Demo Focus

This MVP uses simulated ASR/SER traces and cached model outputs to probe cases where:

- ASR preserves the user's words,
- SER detects affective cues,
- but the downstream agent still makes the wrong interaction decision.

The current MVP isolates the downstream action-selection problem rather than modeling the full causal chain.

Example:

```text
User speech:
"算了 lah, nvm, it's fine."

ASR:
correct transcript

SER:
negative valence, low arousal

Naive agent action:
close conversation

Better action:
acknowledge and repair
```

## What The Demo Includes

- multilingual / code-switched failure probes,
- simulated ASR transcripts,
- simulated SER signals,
- cached model decision outputs,
- emotion-to-action gap evaluation,
- a failure dashboard,
- an optional experimental live LLM judge.

The optional live judge is for exploration only. Cached outputs remain the default stable demo path.

## Key Concept

### Emotion-to-Action Gap

An emotion-to-action gap occurs when affective evidence is available, but the downstream agent still chooses the wrong action.

The current MVP treats this as a downstream action-selection question: assuming affective evidence is available, does the agent choose the right action?

Examples include:

| Failure type | Description |
| --- | --- |
| Premature closure | The agent treats emotional withdrawal as issue resolution. |
| Pragmatic flattening | The agent reads literal meaning but misses social meaning. |
| Over-advice | The user needs validation, but the agent gives instructions. |
| Low-empathy deflection | The agent repeats generic support instead of taking ownership. |
| Missed escalation | The agent fails to repair, clarify, or hand off when needed. |

## Evaluation Plan

Future versions will use a smaller canonical action set:

- listen
- clarify
- repair
- support
- handoff
- close

Planned metrics include:

- action accuracy,
- unsafe confidence rate,
- emotion-to-action gap rate,
- error rate by failure type,
- error rate by language style.

Gold labels are currently author-proposed seed labels, not fully validated human-annotated labels. See [`annotation_protocol.md`](annotation_protocol.md) for the planned validation protocol.

## Failure Source Framing

A complete system-level evaluation should separate at least three failure sources:

| Failure source | Description |
| --- | --- |
| Perception failure | ASR or SER output is itself wrong. |
| Interpretation failure | Affect is available, but pragmatic intent is misread. |
| Action failure | Affect / intent evidence is available, but the downstream action is wrong. |

AffectGuard-SER currently focuses on the third category: downstream action-selection failure.

## Current Status

This is an early MVP research demo, not a completed benchmark.

The goal is to test whether the emotion-to-action-gap framing is useful for evaluating multilingual speech-to-agent systems.

The current 12-case version is not a benchmark. It is a structured seed set: 6 failure categories x 2 cases each.

## Limitations

AffectGuard-SER is an early MVP failure probe, not a completed benchmark.

The current version has several important limitations:

1. **Simulated ASR/SER traces.**
   The current demo uses simulated ASR transcripts and SER signals to isolate the downstream action-selection question. It does not yet capture real ASR/SER failure modes such as accent variation, dialectal speech, background noise, or model-specific emotion recognition errors.

2. **Cached model outputs.**
   The current outputs are cached examples for demonstrating the evaluation flow. They should not be interpreted as empirical results about current SOTA models.

3. **Author-proposed seed labels.**
   The current gold labels and failure categories are proposed by the author for MVP exploration. They have not yet been validated with independent annotators or inter-annotator agreement.

4. **Limited multilingual coverage.**
   The initial examples focus mainly on English, Mandarin, Singlish-style particles, and Mandarin-English code-switching. Broader Southeast Asian language coverage would require native-speaker annotation and culturally grounded case construction.

5. **Static interaction setting.**
   The current demo uses static single-turn cases. It does not yet model streaming speech, memory, relationship history, multi-turn repair, or user correction after an agent failure.

6. **Causal attribution is not complete.**
   The MVP isolates one downstream question: assuming affective evidence is available, does the agent choose the right action? A complete system-level evaluation would need to distinguish ASR failure, SER failure, pragmatic interpretation failure, and downstream action-selection failure.

## Next Steps

Planned next steps include:

- refine the current 12-case structured seed set before expanding toward a larger case set;
- replace cached outputs with real GPT / Claude action judging;
- record model name, timestamp, prompt, and raw response for each run;
- add an annotation protocol for gold actions and failure categories;
- validate labels with 2-3 independent annotators;
- connect real ASR outputs, such as Whisper or MERaLiON-ASR transcripts;
- connect real SER model outputs for emotion / valence / arousal;
- add streaming interaction traces and multi-turn repair cases.

Suggested implementation roadmap:

- V1: static seed cases,
- V2: expand from 6 to 12 seed cases,
- V3: real GPT / Claude action judging on a separate branch,
- V4: real ASR/SER outputs,
- V5: streaming interaction timeline and multi-turn repair evaluation,
- V6: expand toward a 60-case structured seed set,
- V7: annotator validation / user study.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Project Structure

```text
affectguard-ser/
├── app.py
├── annotation_protocol.md
├── assets/
│   └── screenshot.png
├── README.md
├── requirements.txt
└── .gitignore
```
