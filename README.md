# AffectGuard-SER

AffectGuard-SER is an MVP research demo and compact failure probe for multilingual speech-to-agent systems.

It explores a narrow question: when ASR preserves the words and speech emotion recognition detects affect, does the downstream agent still choose the right action?

## What It Tests

The demo focuses on emotion-to-action gaps:

- ASR may preserve the user utterance.
- SER may detect affective signals.
- The downstream LLM agent may still close, deflect, over-advise, or miss a repair/escalation opportunity.

This is not a completed benchmark. It uses simulated ASR/SER traces and cached model outputs to test whether this evaluation framing is useful for multilingual speech-to-agent systems.

## Demo Flow

The Streamlit app includes:

- Failure probes for multilingual and code-switched utterances.
- Cached model decision audits comparing generic and failure-aware behavior.
- A compact failure dashboard.
- An interaction timeline for real-time policy evaluation.
- An optional experimental live LLM judge for structured action assessment.

Cached outputs remain the default demo path. The live judge is optional and experimental.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Project Status

This is an early prototype for research discussion, not a production system or finalized evaluation suite.

## Contact / About

Built by Jayde Zhang  
NTU CCDS  
Focus: LLM agents, social-affective evaluation, failure-aware AI
