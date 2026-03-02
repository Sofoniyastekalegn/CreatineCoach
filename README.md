## Creatine Coach – RAG Chatbot Demo

A small end‑to‑end RAG demo that answers questions about the paper **“Creatine Supplementation for Muscle Growth”** using:

- **Backend**: Python + Flask
- **Embeddings**: Voyage AI (e.g. `voyage-4`)
- **Vector DB**: Pinecone
- **LLM**: Google Gemini (`gemini-2.5-flash`)

The chatbot:

- Ingests a PDF of the creatine paper
- Stores chunk embeddings in Pinecone
- Retrieves top‑K relevant chunks for each question
- Uses Gemini to generate human‑sounding, grounded answers
- Maintains **session‑based conversation memory** in JSON files

---

### 1. Setup

#### 1.1. Create and activate a virtualenv (recommended)

```bash
cd creatinecoach
python -m venv .venv
.venv\Scripts\activate  # on Windows PowerShell
```

#### 1.2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 1.3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env
```

Then edit `.env`:

```text
VOYAGE_API_KEY=pa-...       # your Voyage AI key
PINECONE_API_KEY=pcsk_...   # your Pinecone key
PINECONE_INDEX=https://...  # Pinecone index host URL
GEMINI_API_KEY=AIza...      # your Gemini API key
```

> **Note:** Keys are loaded via `python-dotenv`. Do **not** hard‑code secrets in the code.

---

### 2. Running the app

From the project root (`creatinecoach`):

```bash
flask --app app run --debug
# or
python app.py
```

Then open `http://localhost:5000` in your browser.

---

### 3. Using the web UI

1. **Create a session**
   - Click **“Create New Session”**.
   - A session ID is generated and stored; this session will have its own memory.

2. **Upload and index the PDF**
   - Use the **“Upload & Index PDF”** card to select the research paper PDF  
     (e.g. `Creatine Supplimentation for Muscle Growth.pdf`).
   - The backend parses the PDF, chunks it, creates embeddings via Voyage, and upserts into Pinecone.
   - The UI shows a `doc_id` (used as the Pinecone namespace) and how many chunks were indexed.

3. **Chat**
   - Ask questions like:
     - “How does creatine affect muscle hypertrophy?”
     - “What does the paper say about dosing protocols?”
     - “Is creatine safe according to this study?”
   - The assistant will:
     - Retrieve top‑K relevant chunks for your question.
     - Pass those as context to Gemini.
     - Answer **only** based on the retrieved passages.
     - Use session memory so follow‑ups like “what about dosage?” make sense.

---

### 4. API Endpoints (REST)

All endpoints live in `app.py`.

- **POST `/api/session`**
  - Creates a new conversation session.
  - Response: `{ "session_id": "<uuid>" }`

- **GET `/api/session/<session_id>`**
  - Returns stored messages for that session.

- **POST `/api/upload_pdf`**
  - `multipart/form-data` with field `file` (PDF).
  - Parses the PDF, chunks, embeds with Voyage, and upserts into Pinecone.
  - Response: `{ "doc_id": "<uuid>", "chunks_indexed": <int> }`

- **POST `/api/chat`**
  - JSON body:
    ```json
    {
      "session_id": "<uuid>",
      "doc_id": "<doc_id from upload>",
      "question": "What does the paper say about dosing?"
    }
    ```
  - Response:
    ```json
    {
      "session_id": "<uuid>",
      "answer": "... grounded answer ...",
      "citations": [
        { "chunk_index": 1, "page": 3, "score": 0.91 },
        { "chunk_index": 2, "page": 5, "score": 0.88 }
      ]
    }
    ```

---

### 5. RAG design details

#### 5.1. Chunking strategy

- **Extraction**: The PDF is read with `pypdf.PdfReader`, page by page.
- **Chunking**:
  - Each page’s text is normalized (newlines collapsed, excess spaces removed).
  - A **sliding window** over the text string creates chunks of ~**800 characters** with **200‑character overlap**.
  - This is a simple but effective strategy:
    - Chunks are long enough to capture full paragraphs.
    - Overlap helps when relevant sentences sit on the boundary between chunks.

Each chunk is stored with:

- A unique `id`
- `page` number
- `text` content

#### 5.2. Retrieval settings

- **Embeddings**:
  - Model: `voyage-4`
  - `input_type="document"` for chunks and `input_type="query"` for user questions.
- **Vector store**:
  - Pinecone index (configured by `PINECONE_INDEX` host).
  - Each PDF upload is stored in its own **namespace** = `doc_id` (per‑document isolation).
- **Query**:
  - `top_k = 6`
  - Returns scores and metadata (`page`, `text`) for each match.
  - Retrieved chunks are formatted into a context block and passed to Gemini.

If retrieval fails or returns empty context, the app falls back to a safe message explaining that it can’t generate an answer from the paper excerpts.

#### 5.3. Conversation memory

- Each session is stored as a JSON file under `sessions/`:
  - `<session_id>.json`
  - Shape:
    ```json
    {
      "session_id": "...",
      "created_at": "...",
      "messages": [
        { "role": "user", "content": "..." },
        { "role": "assistant", "content": "..." }
      ]
    }
    ```
- On each `/api/chat` call:
  - The last few (up to 6) messages are threaded into the prompt as a short conversation history block.
  - This allows context‑aware follow‑up questions like “what about dosage?” after asking a broader question.

---

### 6. Grounding & hallucination control

The Gemini prompt explicitly instructs the model to:

- Answer **only** using the provided excerpts from the PDF.
- If the information isn’t clearly supported:
  - Respond with:  
    “I don’t see strong evidence for that in the provided paper. Here’s what the paper does cover…”
  - And then briefly summarize whatever relevant information is available.
- Use light inline citations such as `(page X, chunk Y)` when leaning on specific passages.

This keeps the assistant tightly grounded in the paper while still sounding human and helpful.

---

### 7. Notes / Extensions

- **Voice chat**: You can layer voice input/output on top (e.g. browser Web Speech API or a TTS/ASR service) and send recognized text into `/api/chat`.
- **Multiple PDFs**: The design already supports multiple documents by using different `doc_id` namespaces in Pinecone; you’d just track the active `doc_id` per session.

