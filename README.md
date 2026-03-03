Creatine Coach – Short README

Creatine Coach is a tiny RAG demo that answers questions about the paper "Creatine Supplementation for Muscle Growth". You upload the PDF once, it gets indexed, and then you can chat with a bot that stays grounded in the paper instead of making things up.

1. Quick setup

- Create venv and install (from project root):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

- Environment variables

Copy `.env.example` to `.env` and fill in your keys for Voyage, Pinecone and Gemini (see example fields in `.env.example`). Do not commit real keys.

2. Run the app

```bash
flask --app app run --debug
# or
python app.py
```

Open `http://localhost:5000` in your browser.

3. How to use

- Create a session.
- Upload the creatine PDF to index it.
- Ask questions about the paper. The bot answers using retrieved excerpts, with light citations, and remembers a short conversation history.

For more implementation details (chunking, retrieval, memory), read the code in `app.py` and related modules.

