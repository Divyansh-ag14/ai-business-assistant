# AI Business Assistant

A document Q&A app with voice and text input. You put PDFs in a `data/` folder; the app embeds and indexes them, then answers your questions using a multi-step RAG pipeline (intent → decompose → retrieve → generate). Voice is transcribed with Whisper; the UI is a Streamlit chat interface with optional “Behind the Scenes” views of intent and sources.

---

## Features

- **Voice or text input** — Speak via mic (Whisper) or type; same pipeline for both.
- **Multi-step RAG** — Intent classification, query decomposition, FAISS retrieval, then GPT-4 answer generation.
- **Document-backed** — All PDFs in `data/` are chunked, embedded, and used for retrieval only.
- **Transparency** — “Behind the Scenes” shows intent, sub-queries, documents used, and retrieved chunks.

---

## Technical Architecture

### High-level flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER                                                                   │
│  Voice (mic) → Whisper → text     OR     Text input                     │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ query
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STREAMLIT APP (app.py)                                                 │
│  • load_dotenv() → OPENAI_API_KEY                                       │
│  • client = OpenAI() (Whisper + future use)                             │
│  • run_agent(query) → agent.py                                          │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RAG PIPELINE (agent.py — LangGraph StateGraph)                          │
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐ │
│   │  classify    │───▶│  decompose   │───▶│  retrieve    │───▶│answer  │ │
│   │  (GPT-3.5)   │    │  (GPT-3.5)   │    │  (FAISS)     │    │(GPT-4) │ │
│   └──────────────┘    └──────────────┘    └──────────────┘    └────────┘ │
│        │                     │                    │                │     │
│   intent_info           sub_queries           context            answer  │
│   (JSON)                (list)               (concatenated      (str)    │
│                                                chunks)                   │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA (at startup)                                                      │
│  data/*.pdf → PyPDFLoader → RecursiveCharacterTextSplitter              │
│            → OpenAIEmbeddings → FAISS.from_documents()                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| Layer | Technology | Role |
|-------|------------|------|
| **UI** | Streamlit, streamlit-mic-recorder | Chat UI, voice capture, session state, display of answers and “Behind the Scenes”. |
| **Speech** | OpenAI Whisper (via `openai` SDK) | Voice → text before sending to the agent. |
| **Orchestration** | LangGraph (`StateGraph`) | Defines the DAG: classify → decompose → retrieve → answer; passes state between nodes. |
| **LLMs** | LangChain OpenAI (GPT-3.5-turbo, GPT-4) | Intent, decomposition, and final answer generation. |
| **Embeddings** | LangChain `OpenAIEmbeddings` | Embed chunks at index time and queries at run time (not used for reranking in current code). |
| **Vector store** | FAISS (in-memory) | Built once at startup from chunked PDFs; similarity search in `retrieve_node`. |
| **Documents** | PyPDFLoader, pypdf | Load PDFs from `data/`; text-only (no OCR). |
| **Chunking** | RecursiveCharacterTextSplitter | 1000 chars, 150 overlap. |
| **Config / secrets** | python-dotenv, `.env` | `OPENAI_API_KEY`; no keys in code. |

### Pipeline nodes (agent.py)

1. **classify** — User query → JSON with `intent`, `reasoning`, `sectors`, `quarters` (used for labelling; retrieval is generic similarity).
2. **decompose** — Query → list of sub-queries (or single query); used to run multiple retrievals.
3. **retrieve** — For each sub-query: FAISS `similarity_search_with_score`, take top chunks, merge into `context`; collect `retrieved_docs` and `retrieved_chunks` for the UI.
4. **answer** — Prompt with `query` + `context`; GPT-4 generates answer (e.g. Executive Summary, Detailed Insights, Strategic Implications).

### State (LangGraph)

- **Input:** `query`, plus empty/default `sub_queries`, `intent_info`, `context`, `answer`, `retrieved_docs`, `retrieved_chunks`.
- **Output:** Same keys filled; `run_agent()` returns `answer`, `intent_info`, `sub_queries`, `retrieved_docs`, `retrieved_chunks` for the UI.

### Caching

- **InMemoryCache** (LangChain) — Caches LLM responses by input to avoid duplicate API calls during a session.

---

## Other details

### Prerequisites

- **Python 3.10+**
- **OpenAI API key** (GPT, embeddings, Whisper)

### Project structure

| Path | Purpose |
|------|--------|
| `app.py` | Streamlit app: title, sidebar, voice/text input, example buttons, `run_agent()`, conversation history, “Behind the Scenes” expander. |
| `agent.py` | Document load/chunk, FAISS build, LangGraph graph and nodes, `run_agent(query)`. |
| `data/` | Directory of PDFs to index (required; app expects at least one `.pdf`). |
| `.env` | `OPENAI_API_KEY` (copy from `.env.example`; do not commit). |
| `requirements.txt` | Python dependencies (see below). |

### Setup

1. Clone or download the project; `cd` into its directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
5. Add PDFs: place text-based PDF files in `data/`. The app loads all `.pdf` files at startup.

### Run

```bash
streamlit run app.py
```

Open the URL in the terminal (e.g. `http://localhost:8501`).

### Dependencies (summary)

- **App / UI:** streamlit, streamlit-mic-recorder, python-dotenv  
- **LLM / embeddings:** openai, langchain, langchain-openai, langchain-community, langchain-text-splitters, langgraph  
- **Vector / docs:** faiss-cpu, chromadb, pypdf, tiktoken  
- **Audio:** soundfile (for mic recorder)

### Limitations

- **PDFs:** Only PDFs with extractable text are supported; image-only/scanned PDFs need OCR elsewhere.
- **Index:** FAISS is in-memory and rebuilt on every app start; no persistent vector DB.
- **Intent labels:** Classifier uses a fixed schema (e.g. overview, sales_saas, sales_fmcg); retrieval is generic semantic search and works for any question type.


