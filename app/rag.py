from __future__ import annotations

from typing import Any

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from app.config import get_settings
from app.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from app.query_bridge import expand_query_for_retrieval


def _format_qa_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()

    if "quota" in lowered or "resourceexhausted" in lowered or "429" in lowered:
        return RuntimeError(
            "Gemini 配額已達上限（429）。\n"
            "你可以改用其他可用模型、等待配額重置，或啟用計費方案。"
        )

    if "api key" in lowered or "permission" in lowered or "unauth" in lowered:
        return RuntimeError("API Key 驗證失敗，請檢查 .env 的 GOOGLE_API_KEY 是否正確且仍有效。")

    return RuntimeError(f"RAG 查詢失敗：{message}")


def build_qa_chain() -> RetrievalQA:
    settings = get_settings()

    if not settings.google_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Please add it to your .env file before asking questions.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_TEMPLATE),
        ]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def ask_question(question: str) -> dict[str, Any]:
    chain = build_qa_chain()
    settings = get_settings()
    retrieval_query = expand_query_for_retrieval(
        question=question,
        rules_path=settings.query_bridge_rules_path,
        enabled=settings.query_bridge_enabled,
    )

    try:
        response = chain.invoke({"query": retrieval_query})
    except Exception as exc:  # noqa: BLE001
        raise _format_qa_error(exc) from exc

    sources = []
    for doc in response.get("source_documents", []):
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:500],
            }
        )

    return {
        "answer": response.get("result", ""),
        "sources": sources,
    }
