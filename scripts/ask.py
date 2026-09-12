"""
Phase 4 entry point: ask a question, get an answer grounded strictly in
your ingested papers, with citations back to paper + page.

Usage:
    python scripts/ask.py "What backbone does WeedNet-X use?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.answer import answer_question  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Q: {question}\n")

    result = answer_question(question)

    print(result.answer)

    if result.sources:
        print("\nSources retrieved:")
        for s in result.sources:
            print(f"  [{s.index}] {s.citation()} ({s.section}) — distance {s.distance:.3f}")

    if result.refused:
        print(
            "\n(No answer generated — either nothing relevant enough was "
            "found, or the model determined the sources don't actually "
            "answer this. Sources above, if any, are what came closest.)"
        )


if __name__ == "__main__":
    main()
