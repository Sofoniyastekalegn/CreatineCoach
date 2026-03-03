# Creatine Coach – RAG Chatbot Demo

PDF Q&A chatbot using RAG: Flask + Pinecone + Gemini + Voyage

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# Run
flask --app app run --debug
# Open http://localhost:5000
```

## How It Works
1. **Upload PDF** → Chunked (~800 chars, 200 overlap) → Embeddings (Voyage) → Pinecone
2. **Ask question** → Retrieve top-6 chunks → Gemini generates grounded answer with citations
3. **Session memory** (JSON files) enables follow-up questions

## API Endpoints
- `POST /api/session` – new conversation
- `POST /api/upload_pdf` – index PDF (returns `doc_id`)
- `POST /api/chat` – ask question (`{session_id, doc_id, question}`)

The assistant only answers using retrieved passages—no hallucinations.
