import os
import uuid
import json
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from google import genai

from create_retriever import get_retriever
from tools import UPLOAD_DIR, SESSION_DIR, ingest_pdf_to_pinecone, new_session_dict


load_dotenv()

app = FastAPI(title="CreatineCoach FastAPI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def load_session(session_id: str) -> Dict[str, Any]:
    path = session_path(session_id)
    if not os.path.exists(path):
        return new_session_dict(session_id)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(data: Dict[str, Any]) -> None:
    path = session_path(data["session_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(question: str, history: List[Dict[str, str]], matches: List[Dict[str, Any]]) -> str:
    history_lines = [
        f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
    ]
    history_block = "\n".join(history_lines) if history_lines else "No prior conversation."

    context_lines = []
    for i, m in enumerate(matches):
        page = m.get("page")
        text = m.get("text", "")
        context_lines.append(f"[Chunk {i+1}, page {page}] {text}")
    context_block = "\n\n".join(context_lines) if context_lines else "No relevant passages were retrieved."

    return f"""
You are a science-based assistant that answers questions ONLY using information from the provided research paper excerpts about creatine supplementation and muscle growth.

If the answer is not clearly supported by the excerpts, you MUST say:
"I don’t see strong evidence for that in the provided paper. Here’s what the paper does cover..."
and then briefly summarize relevant information instead of guessing.

Conversation so far:
{history_block}

Retrieved excerpts from the paper:
{context_block}

User question:
{question}

Now write your answer grounded strictly in the excerpts above.
""".strip()


@app.post("/api/session")
async def create_session_fastapi():
    session_id = str(uuid.uuid4())
    data = new_session_dict(session_id)
    save_session(data)
    return {"session_id": session_id}


@app.post("/api/upload_pdf")
async def upload_pdf_fastapi(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    finally:
        await file.close()

    try:
        upserted = await ingest_pdf_to_pinecone(save_path, namespace=doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    return {"doc_id": doc_id, "chunks_indexed": upserted}


@app.post("/api/chat")
async def chat_fastapi(payload: Dict[str, Any]):
    session_id = payload.get("session_id")
    question = (payload.get("question") or "").strip()
    doc_id = payload.get("doc_id")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required")

    retriever = get_retriever()

    try:
        query_vec = await retriever.embed_query(question)
        result = await retriever.query(query_vec, namespace=doc_id, top_k=6)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    matches = []
    for m in result.matches or []:
        md = m.metadata or {}
        matches.append(
            {
                "id": m.id,
                "score": m.score,
                "page": md.get("page"),
                "text": md.get("text"),
            }
        )

    history = load_session(session_id).get("messages", [])
    prompt = build_prompt(question, history, matches)

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        answer = getattr(response, "text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    if not answer:
        answer = (
            "I wasn’t able to generate an answer from the provided paper excerpts. "
            "It may be that the relevant information is missing or could not be retrieved."
        )

    session = load_session(session_id)
    msgs = session.get("messages", [])
    msgs.append({"role": "user", "content": question})
    msgs.append({"role": "assistant", "content": answer})
    session["messages"] = msgs
    save_session(session)

    return JSONResponse(
        {
            "session_id": session_id,
            "answer": answer,
            "citations": [
                {"chunk_index": i + 1, "page": m.get("page"), "score": m.get("score")}
                for i, m in enumerate(matches)
            ],
        }
    )

