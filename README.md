# Thesis Literature Review RAG

A Retrieval-Augmented Generation (RAG) system for organizing and querying
my MS thesis literature review — a lightweight Vision-Language-Action
model for weed detection on edge devices.

## What this does

- **Ingests** PDF research papers from a local folder (with an optional
  Microsoft Graph API / OneDrive mode kept in the codebase, see below)
- **Extracts** text with two-column-aware reading order (PyMuPDF), so
  academic two-column layouts don't get scrambled
- **Detects section boundaries by visual formatting** (font size, bold,
  or ALL-CAPS-with-Roman-numeral patterns) rather than guessing headings
  from a keyword list — validated against papers from IEEE, Springer,
  Elsevier, WACV/ICCV, and arXiv with very different heading conventions
- **Classifies each section** into a canonical category (Introduction,
  Related Work, Methodology, Experimental Setup, Results, Discussion,
  Limitations, Conclusion, References) using a local LLM via Ollama —
  no API key or account required
- **Hash-based dedup** so re-running only processes new or changed papers
- **Embeds chunks locally** (BAAI/bge-small-en-v1.5) into a persistent
  Chroma vector store for semantic search

## Status

- [x] Phase 1 — Ingestion (local-folder default; optional Graph API mode)
- [x] Phase 2 — Extraction, section-aware chunking, hash-based dedup
- [x] Phase 3 — Embeddings + vector store
- [x] Phase 4 — Strict-grounded retrieval with citations (answers only
      from ingested papers, explicit refusal when nothing relevant is found)
- [ ] Phase 5 — Bibliography constraint filtering (year, impact factor,
      publisher) as query-time metadata filters
- [x] Phase 6 — Auto-drafted 6-slide per-paper summaries (Problem,
      Methodology, Experimental Setup, Results, Contributions, Limitations)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Put your PDFs in papers/, or point LOCAL_PAPERS_DIR in .env elsewhere.

ollama serve
ollama pull llama3.2:3b

python scripts/ingest.py        # extract + chunk + classify sections
python scripts/embed.py         # embed chunks into the vector store
python scripts/search_test.py "your query here"   # sanity-check retrieval
python scripts/ask.py "your question here"         # grounded Q&A with citations
python scripts/draft_slides.py "paper name"        # draft 6 slides as .pptx
```

## Design notes

Every non-obvious design decision — why PyMuPDF over pypdf, why
visual-formatting-based heading detection instead of keyword matching,
why a local LLM for section classification, why BGE for embeddings, why
Chroma — is documented as a docstring at the top of the relevant module
under `src/thesis_rag/`, along with the trade-offs considered and, in
several cases, what broke when an earlier approach was tried against
real papers.
