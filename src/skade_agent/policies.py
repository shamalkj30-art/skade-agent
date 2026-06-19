"""
policies.py — a tiny in-memory "policy lookup" used by the agent.

WHY in-memory and not the full RAG pipeline from insurance-rag:
- This project's point is the AGENT FLOW, not the retrieval. A flat list of
  synthetic clauses with simple keyword/embedding matching is enough to
  demonstrate the routing → grounding step without coupling this repo to a
  vector store of real PDFs.
- For production at Gjensidige, this would be swapped for the real
  insurance-rag service called as a tool. The interface (a function taking
  a claim type + free-text and returning ranked clauses) stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from skade_agent.state import ClaimType, PolicyClause


@dataclass(frozen=True)
class _Clause:
    skadetype: ClaimType
    source: str
    text: str


# Synthetic clauses — NOT real Gjensidige terms. Public-domain phrasing,
# inspired by typical Norwegian motor/contents/travel policies.
_CLAUSES: tuple[_Clause, ...] = (
    _Clause(
        "bil",
        "Bilforsikring §3.2",
        "Egenandel ved kollisjon med annet kjøretøy er normalt 6 000 NOK med "
        "mindre forsikringstaker har valgt redusert egenandel.",
    ),
    _Clause(
        "bil",
        "Bilforsikring §4.1",
        "Skader som skjer under påvirkning av alkohol eller andre rusmidler "
        "dekkes ikke. Selskapet kan kreve regress.",
    ),
    _Clause(
        "bil",
        "Bilforsikring §5.3",
        "Skade på vilt (elg, rådyr, rein, hjort) dekkes uten egenandel "
        "dersom hendelsen meldes til politiet innen 24 timer.",
    ),
    _Clause(
        "innbo",
        "Innboforsikring §2.4",
        "Tyveri fra bolig dekkes inntil forsikringssum. Egenandel er 4 000 NOK. "
        "Politianmeldelse er en forutsetning for utbetaling.",
    ),
    _Clause(
        "innbo",
        "Innboforsikring §3.1",
        "Vannskade som følge av lekkasje fra rørledning, varmtvannsbereder eller "
        "vaskemaskin dekkes. Skade som skyldes manglende vedlikehold er unntatt.",
    ),
    _Clause(
        "innbo",
        "Innboforsikring §6.2",
        "Sykkel er dekket inntil 30 000 NOK når den var fastlåst med godkjent "
        "lås på tyveritidspunktet.",
    ),
    _Clause(
        "reise",
        "Reiseforsikring §1.5",
        "Avbestilling dekkes ved akutt sykdom, ulykke eller dødsfall hos den "
        "forsikrede eller nær familie. Legeerklæring må fremlegges.",
    ),
    _Clause(
        "reise",
        "Reiseforsikring §2.3",
        "Forsinket bagasje gir rett til erstatning på inntil 3 000 NOK per "
        "person etter 4 timers forsinkelse. Kvitteringer for nødvendige innkjøp "
        "må oppbevares.",
    ),
    _Clause(
        "reise",
        "Reiseforsikring §4.2",
        "Tap av reisedokumenter og kontanter dekkes inntil 5 000 NOK. "
        "Tapet må meldes lokalt politi.",
    ),
    _Clause(
        "person",
        "Personforsikring §2.1",
        "Ulykkesforsikring dekker varig medisinsk invaliditet etter ulykke. "
        "Erstatning beregnes som andel av forsikringssum.",
    ),
)


def lookup(skadetype: ClaimType, query: str, top_k: int = 3) -> list[PolicyClause]:
    """
    Retrieve the top-k clauses for a given claim type.

    Simple scoring: filter to the matching skadetype, then rank by overlap
    of query tokens with clause text. Good enough to demonstrate the pattern;
    swap for an embedding store when this connects to real vilkår.
    """
    candidates = [c for c in _CLAUSES if c.skadetype == skadetype]
    if not candidates:
        candidates = list(_CLAUSES)  # fallback so we never return empty

    query_tokens = {t.lower() for t in query.split() if len(t) > 3}

    def score(clause: _Clause) -> int:
        text_tokens = {t.lower() for t in clause.text.split()}
        return len(query_tokens & text_tokens)

    ranked = sorted(candidates, key=score, reverse=True)[:top_k]
    return [PolicyClause(source=c.source, text=c.text) for c in ranked]
