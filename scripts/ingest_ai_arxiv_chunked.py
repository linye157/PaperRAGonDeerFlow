# scripts/ingest_ai_arxiv_chunked.py
import os
import uuid
import argparse
from typing import List, Dict, Any

import requests
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct


def ollama_embed(texts: List[str], base_url: str, model: str, timeout: int = 60) -> List[List[float]]:
    """
    Call Ollama embeddings endpoint:
    POST {base_url}/api/embeddings  {"model": "...", "prompt": "..."}
    """
    vectors: List[List[float]] = []
    for t in texts:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=timeout,
        )
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "deer_scholar_arxiv"))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"))
    ap.add_argument("--limit", type=int, default=20000, help="ingest first N rows for fast iteration")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    # 1) Load dataset
    ds = load_dataset("jamescalam/ai-arxiv-chunked", split="train")  # 41.6k rows
    if args.limit and args.limit < len(ds):
        ds = ds.select(range(args.limit))

    # 2) Init Qdrant
    qc = QdrantClient(url=args.qdrant_url)

    # 3) Infer embedding dim with a probe
    probe_vec = ollama_embed(["dimension probe"], args.ollama_url, args.embed_model)[0]
    dim = len(probe_vec)

    # 4) Create / recreate collection
    if args.recreate:
        if qc.collection_exists(args.collection):
            qc.delete_collection(args.collection)
    if not qc.collection_exists(args.collection):
        qc.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # 5) Upsert points
    buf_points: List[PointStruct] = []
    buf_texts: List[str] = []

    def flush():
        nonlocal buf_points, buf_texts
        if not buf_points:
            return
        vecs = ollama_embed(buf_texts, args.ollama_url, args.embed_model)
        for p, v in zip(buf_points, vecs):
            p.vector = v
        qc.upsert(collection_name=args.collection, points=buf_points)
        buf_points, buf_texts = [], []

    for row in ds:
        # Key fields from HF viewer: id, chunk-id, title, summary, chunk, authors, categories, primary_category, published, updated, source, ...
        arxiv_id = row.get("id")
        chunk_id = row.get("chunk-id")
        chunk_text = row.get("chunk") or ""
        title = row.get("title") or ""
        summary = row.get("summary") or ""

        # We index chunk as the searchable unit; store rich metadata as payload.
        doc_text = f"Title: {title}\nAbstract: {summary}\nChunk: {chunk_text}".strip()

        payload: Dict[str, Any] = {
            "arxiv_id": arxiv_id,
            "chunk_id": chunk_id,
            "title": title,
            "summary": summary,
            "categories": row.get("categories"),
            "primary_category": row.get("primary_category"),
            "authors": row.get("authors"),
            "published": row.get("published"),
            "updated": row.get("updated"),
            "pdf_url": row.get("source"),
            "comment": row.get("comment"),
        }

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{arxiv_id}:{chunk_id}"))
        buf_points.append(PointStruct(id=point_id, vector=[], payload=payload))
        buf_texts.append(doc_text)

        if len(buf_points) >= args.batch_size:
            flush()

    flush()
    print(f"[OK] Ingested {len(ds)} rows into Qdrant collection: {args.collection}")


if __name__ == "__main__":
    main()
