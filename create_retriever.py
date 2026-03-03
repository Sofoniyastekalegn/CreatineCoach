import os
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone


load_dotenv()


class VoyagePineconeRetriever:
    """
    Small async-friendly wrapper around Voyage embeddings + Pinecone.
    Uses thread pool offloading under the hood because the SDKs are sync.
    """

    def __init__(self):
        voyage_key = os.getenv("VOYAGE_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_host = os.getenv("PINECONE_INDEX")

        if not voyage_key or not pinecone_key or not pinecone_host:
            raise RuntimeError("Missing VOYAGE_API_KEY, PINECONE_API_KEY, or PINECONE_INDEX")

        self._voyage = voyageai.Client(api_key=voyage_key)
        self._pc = Pinecone(api_key=pinecone_key)
        self._index = self._pc.Index(host=pinecone_host)

    @property
    def index(self):
        return self._index

    async def embed_documents(self, texts: List[str], model: str = "voyage-4") -> List[List[float]]:
        def _embed():
            return self._voyage.embed(texts=texts, model=model, input_type="document").embeddings

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _embed)

    async def embed_query(self, text: str, model: str = "voyage-4") -> List[float]:
        def _embed():
            return self._voyage.embed(texts=[text], model=model, input_type="query").embeddings[0]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _embed)

    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str,
    ) -> int:
        def _upsert():
            return self._index.upsert(vectors=vectors, namespace=namespace)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _upsert)
        return result.get("upsertedCount", 0)

    async def query(
        self,
        vector: List[float],
        namespace: str,
        top_k: int = 6,
    ):
        def _query():
            return self._index.query(
                vector=vector, top_k=top_k, include_metadata=True, namespace=namespace
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _query)


def get_retriever() -> VoyagePineconeRetriever:
    return VoyagePineconeRetriever()

