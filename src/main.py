"""
main.py — CLI runner. Useful for quick checks and shows the full state object.

    python src/main.py "Jeg krasjet bilen min i går i Oslo, ingen andre involvert."
    python src/main.py --voice data/uploads/claim.mp3
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from skade_agent import build_graph
from skade_agent.state import ClaimState


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the skade-agent on one claim.")
    parser.add_argument("text", nargs="?", help="Claim description (Norwegian).")
    parser.add_argument("--voice", help="Path to an audio file instead of text.")
    args = parser.parse_args()

    if not args.text and not args.voice:
        parser.error("Provide either a text claim or --voice <file>")

    initial = ClaimState(raw_text=args.text, voice_file_path=args.voice)

    graph = build_graph()
    final = graph.invoke(initial)

    # graph.invoke returns the merged state as a dict-shaped object;
    # round-trip through ClaimState to get a clean JSON dump.
    final_state = ClaimState.model_validate(final)
    print(final_state.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
