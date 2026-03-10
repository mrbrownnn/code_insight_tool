# 🔍 Code Insight Tool

AI-powered codebase understanding and search tool. Index your source code, then ask questions about it using natural language.

## ✨ Features

- **AST-Based Code Parsing** — Tree-sitter parses Python, JavaScript, Java into structured chunks
- **Smart Chunking** — Code split by function/class/method (not arbitrary lines)
- **Semantic Search** — UniXcoder embeddings for code-aware vector search
- **100% Self-Hosted** — All data stays local: code, embeddings, LLM inference
- **Streamlit UI** — Web-based interface with progress tracking

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| UI | Streamlit |
| Code Parsing | Tree-sitter |
| Embeddings | UniXcoder (`microsoft/unixcoder-base`) |
| Vector DB | Qdrant (Docker) |
| Metadata DB | SQLite |
| LLM | Qwen2.5-Coder via Ollama (self-hosted) |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) NVIDIA GPU for faster embedding & LLM

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd code-insight-tool

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy env template
copy .env.example .env
```

### 2. Start Infrastructure

```bash
# Start Qdrant + Ollama
docker compose up -d

# Pull LLM model (first time only)
docker exec -it code-insight-ollama ollama pull qwen2.5-coder
```

### 3. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Index a Codebase

1. Go to **📥 Ingest** page
2. Enter a Git URL or local folder path
3. Click **🚀 Start Indexing**
4. Wait for the progress bar to complete

## 📁 Project Structure

```
code-insight-tool/
├── app.py                  # Streamlit entry point
├── config.py               # Settings (pydantic-settings)
├── docker-compose.yml      # Qdrant + Ollama
│
├── core/
│   ├── ingestion/          # Git → Filter → Parse → Chunk
│   │   ├── git_handler.py
│   │   ├── file_filter.py
│   │   ├── ast_parser.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── embedding/          # UniXcoder embeddings
│   │   ├── embedder.py
│   │   └── batch_processor.py
│   ├── retrieval/          # (Phase 2)
│   ├── generation/         # (Phase 2)
│   └── intelligence/       # (Phase 3)
│
├── storage/
│   ├── vector_store.py     # Qdrant wrapper
│   └── metadata_store.py   # SQLite metadata
│
├── ui/pages/
│   └── ingest.py           # Ingestion UI
│
└── utils/
    ├── logger.py
    ├── hash_utils.py
    └── token_counter.py
```

## 🔒 Security

- **No external API calls for code** — all embeddings generated locally
- **Self-hosted LLM** via Ollama — code never leaves your machine
- **Local vector storage** — Qdrant runs in Docker, data on local volumes

## 📋 Roadmap

- [x] **Phase 1** — Foundation (ingestion, embedding, storage, UI)
- [ ] **Phase 2** — RAG & Chat (search, context expansion, Q&A)
- [ ] **Phase 3** — Intelligence (topic generation, architecture analysis)
- [ ] **Phase 4** — Polish (UX, VS Code deeplinks, testing)

## License

MIT
