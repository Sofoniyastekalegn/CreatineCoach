import os
import uuid
import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from pypdf import PdfReader

import voyageai
from pinecone import Pinecone
from google import genai


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

load_dotenv()


def get_env_or_error(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def init_clients():
    voyage_api_key = get_env_or_error("VOYAGE_API_KEY")
    pinecone_api_key = get_env_or_error("PINECONE_API_KEY")
    pinecone_index_host = get_env_or_error("PINECONE_INDEX")
    gemini_api_key = get_env_or_error("GEMINI_API_KEY")

    voyage_client = voyageai.Client(api_key=voyage_api_key)

    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(host=pinecone_index_host)

    gemini_client = genai.Client(api_key=gemini_api_key)

    return voyage_client, pinecone_index, gemini_client


def chunk_text(text: str, page_number: int, chunk_size: int = 800, overlap: int = 200):
    """
    Simple word-based sliding window chunking.
    ~800 characters per chunk with 200-character overlap for better recall.
    """
    cleaned = " ".join(text.replace("\n", " ").split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk_text_value = cleaned[start:end]
        if not chunk_text_value.strip():
            break
        chunk_id = str(uuid.uuid4())
        chunks.append(
            {
                "id": chunk_id,
                "page": page_number,
                "text": chunk_text_value,
            }
        )
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(pdf_path: str):
    reader = PdfReader(pdf_path)
    all_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        page_number = i + 1
        page_chunks = chunk_text(text, page_number=page_number)
        all_chunks.extend(page_chunks)
    return all_chunks


def embed_chunks(voyage_client, chunks, model: str = "voyage-4"):
    texts = [c["text"] for c in chunks]
    if not texts:
        return []
    response = voyage_client.embed(
        texts=texts,
        model=model,
        input_type="document",
    )
    embeddings = response.embeddings
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def upsert_chunks(pinecone_index, chunks, namespace: str):
    vectors = []
    for idx, c in enumerate(chunks):
        if "embedding" not in c:
            continue
        vectors.append(
            {
                "id": c["id"],
                "values": c["embedding"],
                "metadata": {
                    "page": c["page"],
                    "text": c["text"],
                    "doc_namespace": namespace,
                },
            }
        )
    if not vectors:
        return 0
    result = pinecone_index.upsert(vectors=vectors, namespace=namespace)
    return result.get("upsertedCount", 0)


def query_chunks(voyage_client, pinecone_index, question: str, namespace: str, top_k: int = 6):
    embed = voyage_client.embed(
        texts=[question],
        model="voyage-4",
        input_type="query",
    )
    query_vec = embed.embeddings[0]
    result = pinecone_index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    matches = result.matches or []
    retrieved = []
    for m in matches:
        md = m.metadata or {}
        retrieved.append(
            {
                "id": m.id,
                "score": m.score,
                "page": md.get("page"),
                "text": md.get("text"),
            }
        )
    return retrieved


def load_session(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return {"session_id": session_id, "created_at": datetime.utcnow().isoformat(), "messages": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session_data):
    path = os.path.join(SESSION_DIR, f"{session_data['session_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)


def build_rag_prompt(question: str, retrieved_chunks, history_messages):
    history_snippets = []
    for msg in history_messages[-6:]:
        role = msg.get("role")
        content = msg.get("content")
        history_snippets.append(f"{role.upper()}: {content}")
    history_block = "\n".join(history_snippets) if history_snippets else "No prior conversation."

    context_lines = []
    for i, c in enumerate(retrieved_chunks):
        page = c.get("page")
        text = c.get("text", "")
        context_lines.append(f"[Chunk {i+1}, page {page}] {text}")
    context_block = "\n\n".join(context_lines) if context_lines else "No relevant passages were retrieved."

    prompt = f"""
You are a science-based assistant that answers questions ONLY using information from the provided research paper excerpts about creatine supplementation and muscle growth.

If the answer is not clearly supported by the excerpts, you MUST say:
"I don’t see strong evidence for that in the provided paper. Here’s what the paper does cover..."
and then briefly summarize relevant information instead of guessing.

Always:
- Be concise, clear, and human-sounding.
- Mention creatine-specific topics like performance, hypertrophy, safety, mechanisms, and dosing only when they are supported by the excerpts.
- Include light inline citations like (page X, chunk Y) when you rely on a passage.

Conversation so far:
{history_block}

Retrieved excerpts from the paper:
{context_block}

User question:
{question}

Now write your answer grounded strictly in the excerpts above.
"""
    return prompt.strip()


def call_gemini(gemini_client, prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            }
        ],
    )
    return getattr(response, "text", "").strip()


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session", methods=["POST"])
def create_session():
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    save_session(session_data)
    return jsonify({"session_id": session_id})


@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id):
    try:
        data = load_session(session_id)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        voyage_client, pinecone_index, _ = init_clients()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    file.save(save_path)

    try:
        chunks = extract_pdf_chunks(save_path)
        chunks = embed_chunks(voyage_client, chunks)
        upserted = upsert_chunks(pinecone_index, chunks, namespace=doc_id)
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {e}"}), 500

    return jsonify({"doc_id": doc_id, "chunks_indexed": upserted})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    question = data.get("question", "").strip()
    doc_id = data.get("doc_id")

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400
    if not doc_id:
        return jsonify({"error": "doc_id (PDF namespace) is required"}), 400

    try:
        voyage_client, pinecone_index, gemini_client = init_clients()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    session_data = load_session(session_id)
    history = session_data.get("messages", [])

    try:
        retrieved_chunks = query_chunks(
            voyage_client,
            pinecone_index,
            question=question,
            namespace=doc_id,
            top_k=6,
        )
    except Exception as e:
        return jsonify({"error": f"Retrieval error: {e}"}), 500

    prompt = build_rag_prompt(question, retrieved_chunks, history)

    try:
        answer = call_gemini(gemini_client, prompt)
    except Exception as e:
        return jsonify({"error": f"Generation error: {e}"}), 500

    if not answer:
        answer = (
            "I wasn’t able to generate an answer from the provided paper excerpts. "
            "It may be that the relevant information is missing or could not be retrieved."
        )

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    session_data["messages"] = history
    save_session(session_data)

    citations = []
    for idx, c in enumerate(retrieved_chunks):
        citations.append(
            {
                "chunk_index": idx + 1,
                "page": c.get("page"),
                "score": c.get("score"),
            }
        )

    return jsonify(
        {
            "session_id": session_id,
            "answer": answer,
            "citations": citations,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

