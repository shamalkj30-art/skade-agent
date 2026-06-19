"""Smoke tests — no network, no API keys. Just structural sanity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_imports():
    import skade_agent  # noqa: F401
    from skade_agent import build_graph, ClaimState  # noqa: F401
    from skade_agent import nodes, policies, voice, graph, state  # noqa: F401


def test_state_defaults():
    from skade_agent.state import ClaimState
    s = ClaimState()
    assert s.language == "no"
    assert s.policy_clauses == []
    assert s.needs_clarification is False


def test_policy_lookup_returns_clauses():
    from skade_agent.policies import lookup
    clauses = lookup("bil", "kollisjon")
    assert len(clauses) > 0
    assert all(c.source for c in clauses)


def test_graph_compiles():
    from skade_agent.graph import build_graph
    g = build_graph()
    # Compiled graph should expose invoke / stream
    assert hasattr(g, "invoke")
    assert hasattr(g, "stream")
