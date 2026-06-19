"""
run_eval.py — measure how reliably the agent does the right thing.

We test two things, both of which directly correspond to production
failure modes:

  1. CLASSIFICATION — when the customer's claim is clearly one type
     (bil / innbo / reise / person), does the agent route it correctly?
     Wrong routing means wrong policy clauses means wrong customer answer.

  2. CLARIFICATION-INSTEAD-OF-GUESSING — when the input is too vague to
     act on, does the agent ask for more info, or does it fabricate
     a confident response? This is the "hallucination" failure mode
     for an agent that takes ACTIONS, not just generates text.

Run:
    python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skade_agent import build_graph  # noqa: E402
from skade_agent.state import ClaimState  # noqa: E402

load_dotenv()

SCENARIOS = Path(__file__).parent / "scenarios.jsonl"


def main() -> None:
    cases = [json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()]
    graph = build_graph()

    classify_total = 0
    classify_correct = 0
    clarif_total = 0
    clarif_correct = 0

    for case in cases:
        initial = ClaimState(raw_text=case["claim"])
        final_raw = graph.invoke(initial)
        final = ClaimState.model_validate(final_raw)

        if case["expect_clarification"]:
            clarif_total += 1
            ok = final.needs_clarification
            clarif_correct += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}  (clarification expected, "
                  f"got needs_clarification={final.needs_clarification})")
        else:
            classify_total += 1
            ok = final.skadetype == case["expected_skadetype"]
            classify_correct += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}  "
                  f"expected={case['expected_skadetype']}  got={final.skadetype}")

    print("\n" + "=" * 50)
    if classify_total:
        print(f"Classification accuracy: {classify_correct}/{classify_total} "
              f"= {classify_correct / classify_total:.0%}")
    if clarif_total:
        print(f"Clarification trigger:   {clarif_correct}/{clarif_total} "
              f"= {clarif_correct / clarif_total:.0%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
