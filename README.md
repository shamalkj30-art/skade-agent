# skade-agent

A small agent that takes a customer's claim description in Norwegian — text or voice — and turns it into a structured claim ready for a downstream system, plus a drafted customer response grounded in policy clauses.

I built this to test what LangGraph could actually do on a workflow I know well. I've worked in skadebehandling at Gjensidige for over seven years, and the thing that always struck me is how much routing happens behind the scenes for what looks like a single customer message — figure out what kind of claim it is, pull the right clauses, decide if there's enough info to act on, draft a response, surface the documentation the customer still needs to send in. That's a clean fit for an agent graph.

**Live demo:** https://huggingface.co/spaces/Shamalkj30/skade-agent

---

## What it does

You give it a claim like *"Jeg krasjet bilen min i går i Storgata"* — text or audio. It:

1. **Transcribes** if it was voice (ElevenLabs Scribe).
2. **Extracts** structured fields — date, location, parties, claim type, estimated value — using a Pydantic schema, so we get a typed object back, not a string we have to parse.
3. **Validates** the extraction. If the description is too vague to act on, it short-circuits to a polite "ask for more info" branch instead of guessing.
4. **Classifies** the claim into product line (bil / innbo / reise / person / annet).
5. **Looks up** relevant policy clauses for that product.
6. **Drafts** a Norwegian customer response grounded in those clauses, plus a list of concrete next steps.

The final output is clean JSON ready for a downstream system, plus a text response the customer reads.

## Why a graph and not a chain

A linear chain works for the happy path. But two things in real claim intake aren't on the happy path:

- **Some claims are too vague to act on.** *"Det skjedde noe"* isn't a claim. A good system asks for more info instead of fabricating one.
- **Different claim types follow different paths.** Bil and reise pull from different clauses and have different tone.

LangGraph makes both of these first-class. The `validate` node can short-circuit to `clarification`. The `classify` node routes everything that follows. Every step appears in LangSmith with its inputs and outputs, so when something misbehaves I know exactly which step on which input.

```
START → transcribe → extract → validate ──► classify → lookup_policy → draft_response → END
                                      │
                                      └──► clarification ────────────────────────────► END
```

## Stack

Python · LangGraph · Claude (Sonnet 4.6 for the customer-facing draft, Haiku 4.5 for the routing nodes) · ElevenLabs Scribe (STT) · Pydantic · Streamlit · LangSmith

Picking two model sizes was deliberate — the routing and extraction nodes do simple structured-output work, Haiku is fast and roughly a tenth of the cost. The customer-facing draft is the one node that really benefits from Sonnet's quality.

## Things I actually hit building this

- **The model hallucinated dates.** First run, the model wrote *"den 9. januar 2024"* when the customer said *"i går"*. Claude doesn't know what today is — it has to be told. One-line fix: inject `date.today().isoformat()` into the extraction prompt as ground truth. The kind of thing you only catch by running real inputs through it.
- **LangSmith routes EU accounts to a different cluster.** Spent a while debugging 403 errors before I realised my account was on `eu.api.smith.langchain.com`, not the US default. Had to set the endpoint explicitly.
- **Refusal detection had to be loosened.** I initially checked for an exact refusal phrase in the response; Claude paraphrased anyway. Switched to a small set of signal phrases that catch paraphrased refusals.

## What I'd build next

- **Real RAG over Gjensidige vilkår.** Right now the policy lookup is a small in-memory list of synthetic clauses. The interface — `lookup(skadetype, query) → [clause]` — is designed so the agent can swap to the [insurance-rag](https://github.com/shamalkj30-art/insurance-rag) service unchanged.
- **Human-in-the-loop approval.** Drafted responses should land in a saksbehandler inbox for review before being sent. LangGraph's `interrupt_before` is the natural fit.
- **Suspicious-claim signal.** The validate node is the right place to add a "this looks unusual — escalate" branch.

## Run it locally

```bash
git clone https://github.com/shamalkj30-art/skade-agent.git
cd skade-agent
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY, ELEVENLABS_API_KEY

# CLI
python src/main.py "Jeg krasjet bilen min i går i Oslo."

# Web UI
streamlit run src/app.py

# Eval
python eval/run_eval.py
```

## Layout

```
skade-agent/
├── src/skade_agent/
│   ├── state.py      # typed ClaimState (Pydantic)
│   ├── nodes.py      # individual node functions
│   ├── graph.py      # LangGraph wiring + conditional edges
│   ├── policies.py   # in-memory policy clause store (demo)
│   └── voice.py      # ElevenLabs Scribe transcription
├── src/main.py       # CLI runner
├── src/app.py        # Streamlit UI
├── eval/
│   ├── scenarios.jsonl
│   └── run_eval.py   # classification + clarification accuracy
└── tests/test_smoke.py
```

## Notes

All sample claims and policy clauses in this repo are synthetic. Nothing here is customer data or internal Gjensidige material.
