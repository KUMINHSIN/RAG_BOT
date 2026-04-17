from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    embedding_provider: str
    llm_provider: str
    embedding_model: str
    llm_model: str
    raw_docs_dir: Path
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int


def get_settings() -> Settings:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()

    return Settings(
        google_api_key=api_key,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "google"),
        llm_provider=os.getenv("LLM_PROVIDER", "google"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        raw_docs_dir=Path(os.getenv("RAW_DOCS_DIR", "./data/raw")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "./data/chroma")),
        collection_name=os.getenv("COLLECTION_NAME", "thesis_docs"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
    )
