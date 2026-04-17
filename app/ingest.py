from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings


class IngestionError(RuntimeError):
    pass


def _require_google_api_key() -> str:
    settings = get_settings()
    if not settings.google_api_key:
        raise IngestionError("Missing GOOGLE_API_KEY. Please add it to your .env file before building the index.")
    return settings.google_api_key


def _load_pdf_documents(raw_docs_dir: Path) -> list[Document]:
    loader = PyPDFDirectoryLoader(str(raw_docs_dir), recursive=True)
    docs = loader.load()
    for doc in docs:
        source = doc.metadata.get("source", "")
        doc.metadata["source"] = str(Path(source).name)
    return docs


def _split_documents(docs: Iterable[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(list(docs))


def build_or_refresh_index() -> dict[str, int]:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    if not settings.raw_docs_dir.exists():
        raise IngestionError(f"Source folder not found: {settings.raw_docs_dir}")

    docs = _load_pdf_documents(settings.raw_docs_dir)
    if not docs:
        raise IngestionError(f"No PDF documents found in {settings.raw_docs_dir}")

    chunks = _split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=_require_google_api_key(),
    )

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )

    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    return {
        "documents": len(docs),
        "chunks": len(chunks),
    }
