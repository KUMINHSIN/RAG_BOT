from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.ingest import IngestionError, build_or_refresh_index


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _check_config() -> None:
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env")
    if not settings.embedding_model:
        raise RuntimeError("EMBEDDING_MODEL is empty")
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is empty")

    raw_dir = settings.raw_docs_dir
    if not raw_dir.exists():
        raise RuntimeError(f"RAW_DOCS_DIR not found: {raw_dir}")

    pdf_count = len(list(raw_dir.rglob("*.pdf")))
    if pdf_count == 0:
        raise RuntimeError(f"No PDF files found in {raw_dir}")

    _ok(f"Config loaded; found {pdf_count} PDF file(s) in {raw_dir}")


def _check_embedding() -> None:
    settings = get_settings()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
    vec = embeddings.embed_query("health check")
    if not vec:
        raise RuntimeError("Embedding returned an empty vector")
    _ok(f"Embedding model works: {settings.embedding_model} (dim={len(vec)})")


def _check_llm() -> None:
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )
    response = llm.invoke("Please reply with exactly: OK")
    text = str(response.content).strip()
    if not text:
        raise RuntimeError("LLM returned empty content")
    _ok(f"LLM model works: {settings.llm_model} (sample={text[:40]})")


def _check_index() -> None:
    try:
        stats = build_or_refresh_index()
    except IngestionError as exc:
        raise RuntimeError(str(exc)) from exc

    if stats["documents"] <= 0 or stats["chunks"] <= 0:
        raise RuntimeError(f"Unexpected index stats: {stats}")

    _ok(
        f"Index build works: {stats['documents']} document(s), {stats['chunks']} chunk(s)"
    )


def _check_retrieval(smoke_question: str) -> None:
    settings = get_settings()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retrieval_k},
    )

    docs = retriever.invoke(smoke_question)
    if not docs:
        raise RuntimeError("Retriever returned no documents")

    _ok(f"Retrieval works: returned {len(docs)} document chunk(s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pre-demo health checks for Thesis RAG Bot"
    )
    parser.add_argument(
        "--question",
        default="請簡短說明這份文件的重點",
        help="Smoke-test question for the QA pipeline",
    )
    args = parser.parse_args()

    print("=== Thesis RAG Pre-Demo Check ===")

    checks = [
        ("Config & PDF", _check_config),
        ("Embedding API", _check_embedding),
        ("LLM API", _check_llm),
        ("Index Build", _check_index),
        ("Retrieval Pipeline", lambda: _check_retrieval(args.question)),
    ]

    for name, fn in checks:
        print(f"\n-- {name} --")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _fail(f"{name} failed: {exc}")
            print("\nResult: FAILED")
            return 1

    print("\nResult: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
