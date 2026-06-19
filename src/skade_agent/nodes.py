"""
nodes.py — the individual node functions that the LangGraph wires together.

Each node:
  - takes a ClaimState
  - returns a dict of just the fields it changes
  - is independently testable (no graph framework imports beyond state)

This separation matters because it keeps the graph file (graph.py) about
*structure* and these nodes about *behaviour* — easy to argue about each
in isolation.
"""

from __future__ import annotations

from datetime import date

from langchain_anthropic import ChatAnthropic

from skade_agent import policies, voice
from skade_agent.state import ClaimState, ClaimType, ExtractedClaim


def _today() -> str:
    """Today's date as an ISO string.

    Injected into prompts so the model can resolve relative dates like
    'i går' correctly. Without this, the LLM fabricates a date from its
    training window — a real bug we hit on the first run.
    """
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Model handles
#
# WHY two model tiers:
# - Claude Haiku 4.5 for extraction and classification — these are routine
#   structured-output tasks; Haiku is fast and cheap and plenty accurate.
# - Claude Sonnet 4.6 for the final customer-facing Norwegian response —
#   quality of the only output the customer sees matters more than
#   per-call cost. Sonnet handles Norwegian fluently with the right tone.
#
# This cheap-for-routing / capable-for-user-output split is the standard
# cost-vs-quality lever in production LLM systems. Reserved Opus 4.7 for
# cases where Sonnet visibly underperforms (none observed yet on this task).
# ---------------------------------------------------------------------------

def _mini() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=1024,
    )


def _capable() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0.2,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def transcribe_node(state: ClaimState) -> dict:
    """If the input is voice, transcribe to text. Otherwise passthrough."""
    if state.voice_file_path:
        transcript = voice.transcribe(state.voice_file_path, language=state.language)
        return {"transcript": transcript}
    # No voice — the raw text IS the transcript.
    return {"transcript": state.raw_text or ""}


def extract_node(state: ClaimState) -> dict:
    """
    Pull structured fields out of the free-text claim using a typed schema.

    WHY with_structured_output(ExtractedClaim):
    - We get a Pydantic object back, not a string we have to parse and pray.
    - The LLM is constrained to the schema; malformed responses fail loudly
      instead of silently corrupting downstream state.
    """
    if not state.transcript:
        return {"extracted": None}

    llm = _mini().with_structured_output(ExtractedClaim)
    prompt = (
        f"I dag er det {_today()}. Bruk dette som referanse for relative "
        "datoer som 'i går', 'forrige uke' eller 'sist mandag'.\n\n"
        "Du er en skadebehandler hos et norsk forsikringsselskap. "
        "Les kundens beskrivelse av hendelsen og hent ut strukturerte felter. "
        "Hvis et felt ikke er nevnt, la det stå tomt — IKKE gjett.\n\n"
        f"Kundens beskrivelse:\n{state.transcript}"
    )
    extracted = llm.invoke(prompt)
    return {"extracted": extracted}


def classify_node(state: ClaimState) -> dict:
    """
    Decide which insurance product line this claim belongs to.

    Why separate from extract: classification drives ROUTING in the graph
    (which policy clauses to retrieve), so it lives in its own node so the
    routing logic is explicit and observable in LangSmith.
    """
    if not state.extracted:
        return {"skadetype": "annet"}

    # Trust the extractor's guess if it's confident; otherwise reclassify
    # with a focused prompt. Cheap, deterministic, easy to defend.
    if state.extracted.skadetype_antatt:
        return {"skadetype": state.extracted.skadetype_antatt}

    llm = _mini()
    prompt = (
        "Klassifiser skaden i én av disse kategoriene: "
        "bil, innbo, reise, person, annet. "
        "Svar med KUN ett ord.\n\n"
        f"Beskrivelse: {state.extracted.beskrivelse}"
    )
    raw = llm.invoke(prompt).content.strip().lower()
    valid: tuple[ClaimType, ...] = ("bil", "innbo", "reise", "person", "annet")
    skadetype: ClaimType = raw if raw in valid else "annet"  # type: ignore[assignment]
    return {"skadetype": skadetype}


def validate_node(state: ClaimState) -> dict:
    """
    Sanity-check the extraction. If critical fields are missing the agent
    should ask for clarification instead of fabricating a response.

    WHY this exists as its own node:
    - Refusal-to-guess is the single most important behaviour in a
      customer-facing AI system. Making it a first-class node means it's
      visible in the trace, easy to test in eval, and easy to tune.
    """
    if not state.extracted:
        return {
            "needs_clarification": True,
            "clarification_reason": "Klarte ikke å hente ut informasjon fra beskrivelsen.",
        }

    if len(state.extracted.beskrivelse.strip()) < 15:
        return {
            "needs_clarification": True,
            "clarification_reason": "Beskrivelsen er for kort til å behandle saken.",
        }

    return {"needs_clarification": False}


def lookup_policy_node(state: ClaimState) -> dict:
    """Fetch policy clauses relevant to this claim."""
    if not state.skadetype or not state.extracted:
        return {"policy_clauses": []}
    clauses = policies.lookup(state.skadetype, state.extracted.beskrivelse)
    return {"policy_clauses": clauses}


def draft_response_node(state: ClaimState) -> dict:
    """
    Draft a Norwegian customer response, grounded in the retrieved clauses.

    WHY we include the clauses verbatim in the prompt and instruct the model
    to cite them: same pattern as the RAG project — answers must be
    traceable to source text or the system can't be trusted.
    """
    if not state.extracted:
        return {"response_draft": "", "next_steps": []}

    clauses_block = "\n".join(
        f"[{i+1}] {c.source}: {c.text}" for i, c in enumerate(state.policy_clauses)
    ) or "(ingen relevante vilkår funnet)"

    system = (
        "Du er en empatisk og presis skadebehandler hos et norsk "
        "forsikringsselskap. Du skriver ALDRI noe som ikke kan begrunnes "
        "i vilkårene under. Hvis vilkårene ikke gir svar, sier du det rett ut "
        "og henviser saken til menneskelig saksbehandler."
    )
    user = (
        f"Skadetype: {state.skadetype}\n"
        f"Strukturert sammendrag: {state.extracted.model_dump_json(indent=2)}\n\n"
        f"Relevante vilkår:\n{clauses_block}\n\n"
        "Skriv et kort svar (3-5 setninger) på norsk til kunden som:\n"
        "1. Bekrefter at vi har mottatt saken.\n"
        "2. Forklarer hva som typisk skjer videre basert på vilkårene.\n"
        "3. Lister konkrete neste steg som kunden må gjøre (f.eks. "
        "politianmeldelse, dokumentasjon).\n\n"
        "Returner svaret som vanlig tekst, ikke JSON."
    )
    response = _capable().invoke([("system", system), ("human", user)]).content

    # Extract next-steps as a quick list pass for downstream automation.
    steps_llm = _mini()
    steps_raw = steps_llm.invoke(
        "List de konkrete handlingene kunden må gjøre, én per linje, "
        "uten nummerering eller punktum. Hvis ingen, returner tom streng.\n\n"
        f"Svar:\n{response}"
    ).content
    next_steps = [s.strip("-• ").strip() for s in steps_raw.splitlines() if s.strip()]

    return {"response_draft": response, "next_steps": next_steps}


def clarification_node(state: ClaimState) -> dict:
    """When validation fails, generate a polite Norwegian request for more info."""
    reason = state.clarification_reason or "Vi trenger litt mer informasjon."
    msg = (
        "Takk for at du tok kontakt. For at vi skal kunne behandle saken din "
        f"trenger vi litt mer informasjon: {reason} "
        "Kan du beskrive hva som skjedde i mer detalj — gjerne hvor og når "
        "hendelsen fant sted?"
    )
    return {"response_draft": msg, "next_steps": ["Send utfyllende beskrivelse"]}
