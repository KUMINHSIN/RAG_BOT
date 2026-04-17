# Thesis-RAG-GCP

A portfolio-ready RAG system for graduation-project Q&A with source-backed answers.

## What This Project Does
This project turns a set of graduation-project PDFs into a question-answering assistant that can be shown to visitors at an exhibition or used as a strong internship portfolio piece.

The system answers questions about the project only from retrieved source chunks, reducing hallucinations and making every answer traceable to the original documents.

## Why It Stands Out
- Source-grounded answers with direct citations
- PDF ingestion and chunking optimized for project documentation
- RetrievalQA pipeline built on LangChain + Chroma
- Google Gemini models for both embeddings and generation
- Streamlit showcase UI designed for live demo and portfolio presentation
- Pre-demo health check script to verify the system before showing it to others
- RAGAS-ready evaluation workflow for quality measurement

## Demo Flow
1. Put project PDFs into `data/raw/`
2. Build the vector index
3. Open the Streamlit showcase
4. Ask visitors' questions about the project
5. Show answer + original source chunks + page numbers

## Key Technical Highlights
- PDF loader: `PyPDFDirectoryLoader`
- Text splitting: `RecursiveCharacterTextSplitter`
- Vector store: `Chroma`
- Retrieval: `MMR` retriever
- Answer chain: `RetrievalQA` with `return_source_documents=True`
- LLM: `gemini-2.5-flash`
- Embeddings: `models/gemini-embedding-001`
- Quality workflow: `scripts/pre_demo_check.py` + `evaluation/run_ragas.py`

## Repository Structure
```text
app/            Core app code: config, ingestion, retrieval, prompts, UI
scripts/        CLI helpers for indexing, QA, and demo checks
evaluation/     Sample evaluation set and RAGAS runner
data/raw/       Source PDFs
data/chroma/    Local vector index persistence
```

## Quick Start

### 1. Create and activate the environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure secrets
```bash
copy .env.example .env
```
Add your `GOOGLE_API_KEY` to `.env`.

### 3. Add source documents
Place your thesis PDFs under `data/raw/`.

### 4. Build the index
```bash
python scripts/build_index.py
```

### 5. Run the showcase UI
```bash
streamlit run app/ui.py
```

### 6. Run the pre-demo check
```bash
python scripts/pre_demo_check.py
```

Optional smoke-test question:
```bash
python scripts/pre_demo_check.py --question "你們的系統核心目標是什麼？"
```

## Recommended Live Demo Script
Use these questions during an interview or exhibition:
- 你們專題的核心目標是什麼？
- 這個系統如何降低 LLM 幻覺？
- 你們的資料切片與檢索策略是什麼？
- 如果文件裡沒有答案，系統會怎麼做？

## Evaluation
Update `evaluation/sample_eval_set.csv`, then run:
```bash
python evaluation/run_ragas.py
```

Recommended metrics:
- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

## Deployment Notes
The project includes a Cloud Run-ready Dockerfile. Example deployment flow:
```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/thesis-rag:latest

gcloud run deploy thesis-rag \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/thesis-rag:latest \
  --platform managed \
  --region REGION \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars GOOGLE_API_KEY=YOUR_KEY
```

## Interview Talking Points
- I built a source-grounded RAG assistant for real exhibition visitors.
- I implemented PDF chunking, retrieval, and citation-backed generation.
- I added a pre-demo health check to reduce presentation failures.
- I validated model and retrieval behavior with a repeatable workflow.
- I designed the project to be both demo-friendly and deployment-ready.

## Current Known Constraints
- Free-tier Gemini quotas can be hit during repeated demo runs.
- Python 3.9 works, but Python 3.10+ is recommended for smoother Google library support.
- Chroma telemetry warnings can appear and do not affect the main demo flow.

## Next Improvements
- Add hybrid retrieval (BM25 + vectors)
- Add structured logging for latency and token usage
- Add authentication for public deployment
- Add context precision/recall charts to the evaluation report
