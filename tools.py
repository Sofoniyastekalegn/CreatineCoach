import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from pypdf import PdfReader

from create_retriever import get_retriever


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

load_dotenv()


def chunk_text(text: str, page_number: int, chunk_size: int = 800, overlap: int = 200):
    cleaned = " ".join(text.replace("\n", " ").split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end]
        if not chunk.strip():
            break
        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "page": page_number,
                "text": chunk,
            }
        )
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    all_chunks: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        page_number = i + 1
        page_chunks = chunk_text(text, page_number=page_number)
        all_chunks.extend(page_chunks)
    return all_chunks


async def ingest_pdf_to_pinecone(pdf_path: str, namespace: str) -> int:
    retriever = get_retriever()
    chunks = extract_pdf_chunks(pdf_path)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = await retriever.embed_documents(texts)

    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        vectors.append(
            {
                "id": chunk["id"],
                "values": emb,
                "metadata": {
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "doc_namespace": namespace,
                },
            }
        )

    return await retriever.upsert_vectors(vectors=vectors, namespace=namespace)


def new_session_dict(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }

