"""Embeddings (local, free) + ChromaDB vector index with metadata/access filter."""
import chromadb
from sentence_transformers import SentenceTransformer
from .ingest import load_policies, chunk

_model = None
def embed(texts):
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model.encode(texts, normalize_embeddings=True).tolist()

class Retriever:
    def __init__(self, window=2, stride=1):
        self.client = chromadb.EphemeralClient()
        name = f"pp_w{window}_s{stride}"
        try: self.client.delete_collection(name)
        except Exception: pass
        self.col = self.client.create_collection(name, metadata={"hnsw:space": "cosine"})
        self.chunks = [c for d in load_policies() for c in chunk(d, window, stride)]
        self.col.add(ids=[c["chunk_id"] for c in self.chunks],
                     documents=[c["text"] for c in self.chunks],
                     metadatas=[{k: v for k, v in c.items() if k != "text"} for c in self.chunks],
                     embeddings=embed([f"{c['title']}. {c['text']}" for c in self.chunks]))

    def search(self, q, k=4, allowed_access=("internal", "public")):
        r = self.col.query(query_embeddings=embed([q]), n_results=min(k, len(self.chunks)),
                           where={"access": {"$in": list(allowed_access)}})
        return [dict(text=d, score=round(1 - dist, 4), **m)
                for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
