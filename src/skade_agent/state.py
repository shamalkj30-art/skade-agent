"""
state.py — the typed state that flows through the LangGraph.

WHY a typed state object (instead of just passing dicts around):
- LangGraph merges partial updates from each node back into a single state.
- A typed schema means each node declares exactly what it reads and writes,
  which makes the graph itself self-documenting AND lets us catch bugs early.
- Pydantic gives us validation + JSON serialisation for free, so the final
  state can be dumped straight to a downstream system or logged.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


ClaimType = Literal["bil", "innbo", "reise", "person", "annet"]


class ExtractedClaim(BaseModel):
    """Structured fields the LLM pulls out of the free-text claim description."""

    skadedato: Optional[str] = Field(
        None,
        description="Dato for hendelsen i ISO-format (YYYY-MM-DD) hvis nevnt.",
    )
    sted: Optional[str] = Field(None, description="Hvor hendelsen skjedde.")
    beskrivelse: str = Field(
        ..., description="En kort, faktisk oppsummering av hva som skjedde."
    )
    skadetype_antatt: Optional[ClaimType] = Field(
        None,
        description="Beste gjetning på skadetype basert på beskrivelsen.",
    )
    parter: list[str] = Field(
        default_factory=list,
        description="Personer eller kjøretøy involvert, hvis nevnt.",
    )
    skadeverdi_estimat_nok: Optional[float] = Field(
        None, description="Anslått skadebeløp i NOK hvis kunden nevnte det."
    )


class PolicyClause(BaseModel):
    """A single retrieved policy clause used to ground the response."""

    source: str
    text: str


class ClaimState(BaseModel):
    """
    The full state object that flows through every node.

    Fields are Optional because each node only fills in the slice it owns —
    LangGraph merges partial updates from each node back into this shape.
    """

    # ---- inputs ----
    raw_text: Optional[str] = None
    voice_file_path: Optional[str] = None  # set when input is audio
    language: str = "no"  # Norwegian by default

    # ---- after transcribe ----
    transcript: Optional[str] = None

    # ---- after extract ----
    extracted: Optional[ExtractedClaim] = None

    # ---- after classify ----
    skadetype: Optional[ClaimType] = None

    # ---- after validate ----
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None

    # ---- after policy lookup ----
    policy_clauses: list[PolicyClause] = Field(default_factory=list)

    # ---- after draft ----
    response_draft: Optional[str] = None
    next_steps: list[str] = Field(default_factory=list)

    # ---- meta ----
    trace_id: Optional[str] = None  # for LangSmith correlation
