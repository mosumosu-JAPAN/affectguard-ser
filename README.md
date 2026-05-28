# AffectGuard-SER

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

Examples include:

| Failure type | Description |
| --- | --- |
| Premature closure | The agent treats emotional withdrawal as issue resolution. |
| Pragmatic flattening | The agent reads literal meaning but misses social meaning. |
| Over-advice | The user needs validation, but the agent gives instructions. |
| Low-empathy deflection | The agent repeats generic support instead of taking ownership. |
| Missed escalation | The agent fails to repair, clarify, or hand off when needed. |

## Current Status

This is an early MVP research demo, not a completed benchmark.

The goal is to test whether the emotion-to-action-gap framing is useful for evaluating multilingual speech-to-agent systems.

Future extensions could include:

- real ASR model outputs,
- real speech emotion recognition outputs,
- MERaLiON-style model outputs,
- audio upload support,
- streaming interaction traces,
- larger multilingual evaluation sets.

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
├── README.md
├── requirements.txt
└── .gitignore
```
