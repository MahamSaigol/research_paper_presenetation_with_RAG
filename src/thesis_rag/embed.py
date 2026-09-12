"""
Wraps the local embedding model (BAAI/bge-small-en-v1.5 via
sentence-transformers). No API key, no account — consistent with the
Ollama choice for classification: everything in this pipeline runs
locally except the papers you feed it.

Design choice: BGE's query/passage asymmetry is handled here, not left to
callers to remember.
    BGE models are trained so that queries and the passages they search
    over are embedded slightly differently: a query gets an instruction
    prefix ("Represent this sentence for searching relevant passages: ")
    telling the model "this is a search query", while passages/documents
    get no prefix at all. Using the wrong one for either side doesn't
    error — it just quietly makes retrieval worse, which is a much harder
    bug to notice than a crash. Centralizing this as two differently-named
    functions (embed_query vs embed_passages) makes it impossible for
    calling code to accidentally use the wrong convention.

Design choice: the model is loaded once and cached at module level.
    Loading a sentence-transformers model involves reading weights off
    disk (or downloading them, on first use) — expensive enough that you
    don't want to pay it per chunk. A module-level cache means the first
    call in a process loads it, everything after reuses the same instance.
"""

from __future__ import annotations

from .config import settings

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported lazily so that code paths that never need embeddings
        # (e.g. just running Phase 1/2) don't pay the import cost of
        # sentence-transformers/torch at all.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embeds chunk text for storage. No instruction prefix — see module docstring."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embeds a search query. Adds BGE's required instruction prefix — see module docstring."""
    model = _get_model()
    embedding = model.encode(
        [_QUERY_INSTRUCTION + text], normalize_embeddings=True
    )
    return embedding[0].tolist()


if __name__ == "__main__":
    # Smoke test: confirms the model downloads/loads and produces
    # sensible-looking embeddings before you wire it into the full
    # pipeline. First run downloads ~130MB from Hugging Face.
    vecs = embed_passages(["Weeds compete with crops for nutrients and light."])
    print(f"Passage embedding dim: {len(vecs[0])}")
    q = embed_query("How do weeds affect crop yield?")
    print(f"Query embedding dim: {len(q)}")
