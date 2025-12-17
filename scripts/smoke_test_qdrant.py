import os
import requests
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "deer_scholar_arxiv")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

def ollama_embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    # 兼容 embeddings: [[...]] 或 embedding: [...]
    if "embeddings" in data:
        return data["embeddings"][0]
    if "embedding" in data:
        return data["embedding"]
    raise ValueError(f"Unexpected Ollama response keys: {list(data.keys())}")

def main():
    q = "KV cache quantization for LLM inference"
    vec = ollama_embed(q)

    client = QdrantClient(url=QDRANT_URL)
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=5,
        with_payload=True,
    )

    print("Top hits:")
    for i, h in enumerate(hits, 1):
        payload = h.payload or {}
        title = payload.get("title") or payload.get("paper_title") or ""
        arxiv_id = payload.get("arxiv_id") or payload.get("id") or ""
        print(f"{i}. score={h.score:.4f}  arxiv_id={arxiv_id}  title={title}")

if __name__ == "__main__":
    main()
