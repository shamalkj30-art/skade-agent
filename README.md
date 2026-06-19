# skade-agent — claims-intake agent for Norwegian insurance

A multi-step **LangGraph** agent that takes a customer's claim description
(voice or text), pulls it apart into structured fields, decides which
policy line applies, retrieves the relevant clauses, and drafts a Norwegian
customer response — refusing to guess when the input is too vague.

**Live demo:** _<add Streamlit / HF Spaces URL>_
**Trace:** _<paste a LangSmith run URL after first invocation>_

---

## Why a graph and not a chain

A linear chain (extract → classify → respond) would work for the happy path,
but real claim intake has two awkward facts:

1. **Some claims are too vague to act on.** A good system refuses to guess
   and asks for more information instead of fabricating a confident response.
2. **Different claim types follow different paths.** Motor and travel claims
   pull from different policy clauses, and the response tone differs.

LangGraph makes both of these first-class:

- The `validate` node can short-circuit the flow to `clarification` instead
  of letting the model fake an answer.
- The `classify` node routes the rest of the graph deterministically.
- Every node appears as a discrete step in LangSmith — when something
  misbehaves in production it's immediately obvious *which* step on *which*
  input.

```
START → transcribe → extract → validate ──► classify ──► lookup_policy ──► draft_response ──► END
                                       │
                                       └──► clarification ─────────────────────────────────► END
```

## What it does, in one example

Input (Norwegian, free text):
> *"Jeg krasjet bilen min i går kveld i Storgata. Stor bulk på høyre side,
> ingen personer skadet."*

After the graph runs, the state contains:

```json
{
  "extracted": {
    "skadedato": null,
    "sted": "Storgata",
    "beskrivelse": "Kollisjon med parkert bil, stor bulk på høyre side",
    "skadetype_antatt": "bil",
    "parter": []
  },
  "skadetype": "bil",
  "policy_clauses": [
    { "source": "Bilforsikring §3.2", "text": "Egenandel ved kollisjon..." }
  ],
  "response_draft": "Takk for at du meldte saken... [Norwegian response grounded in §3.2]",
  "next_steps": [
    "Send bilder av begge bilene",
    "Bytt forsikringsdetaljer med motpart"
  ]
}
```

That JSON is what a downstream system would consume — the customer just
sees `response_draft`.

## Stack

**Python · LangGraph · LangChain · Anthropic Claude (Sonnet 4.6 + Haiku 4.5) · ElevenLabs Scribe · Pydantic · Streamlit · LangSmith**

Model selection is intentional:
- **Claude Haiku 4.5** for extraction & classification — fast and cheap,
  accurate enough for structured tasks. Most calls in the graph hit Haiku.
- **Claude Sonnet 4.6** for the customer-facing Norwegian response —
  quality of the only output the customer sees matters more than
  per-call cost.
- **ElevenLabs Scribe** for voice transcription — same vendor as the rest
  of our voice tooling, removing a separate OpenAI dependency.

## What I'd build next (limitations)

- **Real RAG over Gjensidige vilkår.** Today the policy lookup is a tiny
  in-memory list of synthetic clauses. The interface — `lookup(skadetype,
  query) → [clause]` — is designed so the agent can swap to the
  [`insurance-rag`](https://github.com/shamalkj30-art/insurance-rag) service
  unchanged.
- **Human-in-the-loop.** Drafted responses should land in a saksbehandler
  inbox for approval before being sent — LangGraph's `interrupt_before` is
  the natural fit.
- **Fraud / unusual-claim signal.** The validate node is the right place
  to add a "this looks suspicious — escalate" branch.
- **Eval on real conversations.** The current eval covers classification
  and clarification triggers; an LLM-as-judge pass on response quality
  would close the loop.

## Run it locally

```bash
git clone https://github.com/shamalkj30-art/skade-agent.git
cd skade-agent
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, LANGSMITH_API_KEY

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
├── src/
│   ├── skade_agent/
│   │   ├── state.py        # typed ClaimState (Pydantic)
│   │   ├── nodes.py        # individual node functions
│   │   ├── graph.py        # LangGraph wiring + conditional edges
│   │   ├── policies.py     # tiny in-memory policy clause store
│   │   └── voice.py        # Whisper transcription
│   ├── main.py             # CLI runner
│   └── app.py              # Streamlit UI
├── eval/
│   ├── scenarios.jsonl     # test cases (claim → expected route)
│   └── run_eval.py         # measures classification + clarification accuracy
├── data/sample_claims/     # example Norwegian claims for quick testing
└── tests/test_smoke.py     # structural tests (no API calls)
```

## Notes on data

All example claims and policy clauses are **synthetic**. Nothing in this
repo contains customer data or proprietary Gjensidige material.
