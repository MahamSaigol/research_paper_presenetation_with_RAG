"""
Centralized, typed configuration.

Design choice: pydantic-settings over plain os.environ / python-dotenv calls
scattered through the codebase.
  - Fails fast and loudly at startup if a required value is missing, instead
    of surfacing as a cryptic KeyError deep inside an HTTP call later.
  - Gives every other module a single typed `settings` object to import,
    rather than re-reading environment variables in multiple places.

Design choice: `ingestion_source` as an explicit switch, not two separate
codebases.
    Phase 1 originally targeted Graph API only. In practice, Azure AD app
    registration turned out to depend on account/tenant conditions outside
    this project's control (see README "Ingestion source" section for the
    full story). Rather than treat local-folder ingestion as a lesser
    fallback bolted on afterward, it's a first-class, equally-supported
    mode selected by one config value — Phase 2 onward (chunking, dedup,
    embedding) is written against a common `list[PaperFile]` shape and
    doesn't care which mode produced it. Graph API mode stays in the repo,
    fully working, for anyone (including future-me, on a less locked-down
    Microsoft account) who wants to switch back.
"""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ingestion_source: Literal["local_folder", "graph_api"] = "local_folder"

    # --- local_folder mode ---
    local_papers_dir: str = "./papers"

    # --- Phase 2: extraction, chunking, dedup ---
    data_dir: str = "./data"  # holds manifest.json and chunks/
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # --- Phase 2: heading classification (Ollama, local, no API key) ---
    classify_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    # --- Phase 3: embeddings + vector store ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    chroma_dir: str = "./data/chroma"
    collection_name: str = "papers"

    # --- Phase 4: strict-grounded answer generation ---
    answer_model: str = "llama3.1:8b"
    answer_timeout_seconds: int = 180
    # Squared-L2 distance cutoff (normalized embeddings) above which the
    # top retrieved chunk is considered "not actually relevant" and the
    # question is refused before ever reaching the LLM. Rough starting
    # point, not empirically tuned — see answer.py docstring.
    relevance_distance_threshold: float = 1.0
    retrieval_k: int = 6

    # --- graph_api mode (all optional unless ingestion_source=graph_api) ---
    graph_client_id: str | None = None
    graph_tenant: str = "common"
    graph_scopes: str = "Files.Read"
    onedrive_papers_folder: str = "Thesis/Literature"
    token_cache_path: str = ".token_cache.bin"

    @model_validator(mode="after")
    def _require_graph_client_id_when_needed(self) -> "Settings":
        # Only enforce this when graph_api mode is actually selected, so
        # local-folder users never need Graph-related env vars at all.
        if self.ingestion_source == "graph_api" and not self.graph_client_id:
            raise ValueError(
                "GRAPH_CLIENT_ID is required when INGESTION_SOURCE=graph_api. "
                "Set it in .env, or switch INGESTION_SOURCE=local_folder."
            )
        return self

    @property
    def authority(self) -> str:
        return f"https://login.microsoftonline.com/{self.graph_tenant}"

    @property
    def scopes_list(self) -> list[str]:
        return self.graph_scopes.split()


settings = Settings()