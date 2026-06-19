"""
graph.py — wires the nodes into a LangGraph state machine.

WHY a state graph (not a simple chain or a single mega-prompt):

1. Conditional routing is first-class. The `validate` step can short-circuit
   the flow when the claim is too vague to answer — sending the user to a
   clarification node instead of fabricating a response.

2. Observability. Each node shows up as a discrete step in LangSmith with
   its own inputs/outputs. When something goes wrong in production it is
   immediately clear WHICH step failed and on WHICH state.

3. Each node is independently testable and swappable. Want to plug in
   a different LLM provider for the draft node only? Change one function —
   the graph wiring doesn't move.

4. Same code path runs whether the input is voice or text. The transcribe
   node is the seam — branching at that single point keeps the rest of the
   graph identical.

Flow:
        START
          │
     transcribe          (voice → text, or passthrough)
          │
       extract           (free-text → ExtractedClaim)
          │
      validate           (do we have enough to proceed?)
          │
   ┌──────┴───────┐
   │              │
needs_clarif?   ok
   │              │
clarification   classify → lookup_policy → draft_response
   │              │
   └──────┬───────┘
          │
         END
"""

from langgraph.graph import END, START, StateGraph

from skade_agent.nodes import (
    classify_node,
    clarification_node,
    draft_response_node,
    extract_node,
    lookup_policy_node,
    transcribe_node,
    validate_node,
)
from skade_agent.state import ClaimState


def _route_after_validate(state: ClaimState) -> str:
    """Conditional edge: short-circuit to clarification when state is too thin."""
    return "clarification" if state.needs_clarification else "classify"


def build_graph():
    """Build and compile the agent graph."""
    g = StateGraph(ClaimState)

    g.add_node("transcribe", transcribe_node)
    g.add_node("extract", extract_node)
    g.add_node("validate", validate_node)
    g.add_node("classify", classify_node)
    g.add_node("lookup_policy", lookup_policy_node)
    g.add_node("draft_response", draft_response_node)
    g.add_node("clarification", clarification_node)

    g.add_edge(START, "transcribe")
    g.add_edge("transcribe", "extract")
    g.add_edge("extract", "validate")

    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"classify": "classify", "clarification": "clarification"},
    )

    g.add_edge("classify", "lookup_policy")
    g.add_edge("lookup_policy", "draft_response")
    g.add_edge("draft_response", END)
    g.add_edge("clarification", END)

    return g.compile()
