"""
Phase 3 sanity check: runs a real semantic search query against your
embedded papers and prints the top results, so you can eyeball retrieval
quality before Phase 4 builds strict-grounded answer generation on top
of it.

Usage:
    python scripts/search_test.py "your query here"
    python scripts/search_test.py                    # uses a default sample query

Note on the "distance" numbers printed: Chroma's default index uses
squared L2 distance, not cosine similarity directly — lower means more
similar, but it isn't on a familiar 0-1 scale. Since embeddings are
normalized (see embed.py), ranking order is unaffected either way; this
script just doesn't pretend the number is something it isn't.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.vectorstore import count, query  # noqa: E402


def main() -> None:
    total = count()
    if total == 0:
        print("No chunks in the vector store yet — run scripts/embed.py first.")
        return

    query_text = " ".join(sys.argv[1:]) or "lightweight weed detection on edge devices"
    print(f"Searching {total} chunks for: {query_text!r}")
    print("(lower distance = more similar)\n")

    results = query(query_text, n_results=5)
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, dists), start=1
    ):
        snippet = doc[:220].replace("\n", " ")
        pages = (
            f"p{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"p{meta['page_start']}-{meta['page_end']}"
        )
        print(
            f"[{rank}] distance={dist:.4f}  {meta['paper_name']}  "
            f"[{meta['section']}, {pages}]"
        )
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    main()
