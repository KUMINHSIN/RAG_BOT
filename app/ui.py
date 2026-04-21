from __future__ import annotations

import streamlit as st

from app.ingest import IngestionError
from app.config import get_settings
from app.ingest import build_or_refresh_index
from app.rag import ask_question


PROJECT_TAGLINE = "可追溯來源的畢專 RAG 問答展示"

DEMO_QUESTIONS = [
    "專題主要是做什麼？",
    "照片轉成音樂中間的依據是什麼？",
    "主要技術有哪些？",
    "生成音樂後這個 app 可以做什麼？",
]

SHOWCASE_POINTS = [
    (
        "Source-grounded",
        "可追溯來源",
        "每則回答都附原文片段與頁碼，強調可追溯性。",
    ),
    (
        "RAG pipeline",
        "完整檢索流程",
        "PDF ingestion、chunking、vector retrieval、RetrievalQA 一條龍。",
    ),
    (
        "Evaluation-ready",
        "可量化品質",
        "已預留 RAGAS 評估與 demo health check。",
    ),
]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-cream: #fffdf7;
            --bg-sky: #e8f6ff;
            --bg-sky-2: #d7efff;
            --card: rgba(255, 255, 255, 0.84);
            --card-strong: rgba(255, 255, 255, 0.94);
            --line: rgba(102, 163, 255, 0.18);
            --text: #1f2a37;
            --text-soft: #4b5563;
            --sky: #5bbce9;
            --sky-dark: #2f8fd8;
            --yellow: #fff2b2;
            --yellow-2: #ffe59a;
            --shadow: 0 18px 42px rgba(84, 131, 179, 0.16);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(91, 188, 233, 0.22), transparent 26%),
                radial-gradient(circle at top right, rgba(255, 242, 178, 0.42), transparent 24%),
                linear-gradient(180deg, var(--bg-cream) 0%, var(--bg-sky) 52%, var(--bg-cream) 100%);
            color: var(--text);
            font-family: "Noto Sans TC", "Inter", sans-serif;
        }
        .hero {
            padding: 2rem 2rem 1.5rem 2rem;
            border: 1px solid var(--line);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255, 253, 247, 0.98), rgba(255, 242, 178, 0.58));
            box-shadow: var(--shadow);
        }
        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(91, 188, 233, 0.16);
            color: var(--sky-dark);
            font-size: 0.82rem;
            letter-spacing: 0.02em;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            background: var(--card);
            border: 1px solid var(--line);
            min-height: 118px;
            box-shadow: 0 12px 28px rgba(91, 188, 233, 0.08);
        }
        .metric-title {
            font-size: 0.8rem;
            color: var(--sky-dark);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.25rem;
        }
        .metric-sub {
            color: var(--text-soft);
            font-size: 0.92rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text);
            margin: 1.25rem 0 0.75rem 0;
        }
        .showcase-box {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            background: var(--card-strong);
            border: 1px solid var(--line);
            margin-bottom: 0.8rem;
            box-shadow: 0 12px 28px rgba(91, 188, 233, 0.08);
        }
        .showcase-kicker {
            color: var(--sky-dark);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .showcase-title {
            color: var(--text);
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .showcase-text {
            color: var(--text-soft);
            line-height: 1.55;
        }
        .source-card {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 24px rgba(91, 188, 233, 0.06);
        }
        .source-meta {
            color: var(--sky-dark);
            font-size: 0.82rem;
            margin-bottom: 0.3rem;
        }
        .source-content {
            color: var(--text);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.96) !important;
            color: var(--text) !important;
            border: 1px solid rgba(91, 188, 233, 0.28) !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .stButton button {
            background: linear-gradient(135deg, var(--sky), var(--yellow-2)) !important;
            color: #17324d !important;
            border: 1px solid rgba(47, 143, 216, 0.22) !important;
            box-shadow: 0 10px 20px rgba(91, 188, 233, 0.16) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        .stButton button:hover {
            filter: brightness(1.02);
            transform: translateY(-1px);
        }
        .stMarkdown, .stCaption, .stText, label, p, li {
            color: var(--text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric(title: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _showcase_point(kicker: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="showcase-box">
            <div class="showcase-kicker">{kicker}</div>
            <div class="showcase-title">{title}</div>
            <div class="showcase-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Thesis RAG Bot", page_icon="📘", layout="wide")
_inject_styles()

settings = get_settings()

if not settings.google_api_key:
    st.warning("我還沒有讀到你的 GOOGLE_API_KEY，請先在 .env 填入，這樣我才能幫你建立索引和回答問題。")

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Portfolio Showcase · Thesis-RAG-GCP</div>
        <h1 style="margin:0 0 0.4rem 0; font-size: 2.35rem; color: #1f2a37;">📘 Thesis RAG Bot</h1>
        <p style="margin:0; color:#4b5563; font-size:1.02rem; line-height:1.7; max-width: 880px;">
            {PROJECT_TAGLINE}。這個系統讓參觀者能直接詢問畢業專題內容，
            回答會附上來源片段與頁碼，能有效降低幻覺並提升展示可信度。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

metric_cols = st.columns(3)
with metric_cols[0]:
    _metric("Core Value", "Source-backed", "每則回答都能回溯到原文片段與頁碼")
with metric_cols[1]:
    _metric("RAG Stack", "PDF → Chroma → QA", "完整展示 ingestion、retrieval、generation 流程")
with metric_cols[2]:
    _metric("Demo Ready", "Health Check", "已內建 pre-demo 檢查與展示腳本")

st.markdown('<div class="section-title">展示亮點</div>', unsafe_allow_html=True)
feature_cols = st.columns(3)
for col, (kicker, title, text) in zip(feature_cols, SHOWCASE_POINTS):
    with col:
        _showcase_point(kicker, title, text)

with st.sidebar:
    st.markdown("### 索引管理")
    st.caption("你可以先按這個重建索引，讓系統讀取最新的 PDF。")
    if st.button("重建向量索引", use_container_width=True):
        try:
            with st.spinner("正在讀取 PDF 並建立索引..."):
                stats = build_or_refresh_index()
        except IngestionError as exc:
            st.error(str(exc))
        else:
            st.success(f"完成：{stats['documents']} 份文件，{stats['chunks']} 個 chunks")

    st.markdown("### Demo Questions")
    st.caption("你可以直接點下面的問題，或自己輸入想問的內容。")
    for demo_question in DEMO_QUESTIONS:
        if st.button(demo_question, key=f"demo_{demo_question}", use_container_width=True):
            st.session_state["question"] = demo_question

    st.markdown("### 本次設定")
    st.code(
        f"""Embedding: {settings.embedding_model}
LLM: {settings.llm_model}
Chunk size: {settings.chunk_size}
Overlap: {settings.chunk_overlap}
Top-k: {settings.retrieval_k}""",
        language="text",
    )

question = st.text_input(
    "請輸入問題",
    placeholder="例如：你們的模型訓練流程是什麼？",
    key="question",
)

if question:
    try:
        with st.spinner("檢索與生成中..."):
            result = ask_question(question)
    except Exception as exc:
        st.error(f"查詢失敗：{exc}")
    else:
        st.markdown('<div class="section-title">回答</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-content">{result['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">引用來源</div>', unsafe_allow_html=True)
        if not result["sources"]:
            st.info("沒有找到來源文件，請先建立索引或調整問題。")
        else:
            for idx, src in enumerate(result["sources"], start=1):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-meta">來源 {idx} · {src['source']} · page {src['page']}</div>
                        <div class="source-content">{src['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
