# project/scripts/topic_classifier.py
"""
Assigns a single high-level topic label to each article using zero-shot
classification (MNLI). Writes `topic` and `topic_score` into each Qdrant payload.

Model: facebook/bart-large-mnli (works well for short titles).
"""
import time
import json
from pathlib import Path
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from transformers import pipeline

# CONFIG
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
BATCH = 256
MIN_CONFIDENCE = 0.69  # if below, set topic = "misc"

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"

LABELS = [
    "politics",
    "world",
    "india",
    "economy",
    "business",
    "technology",
    "science",
    "health",
    "sports",
    "entertainment",
    "crime",
    "education",
    "environment",
    "lifestyle",
    "opinion",
    "misc"
]

MODEL = "facebook/bart-large-mnli"

def ensure_collection(client: QdrantClient):
    cols = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in cols:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

def fetch_points(client: QdrantClient):
    pts = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not page:
            break
        pts.extend(page)
        if offset is None:
            break
    return pts

def main():
    t0 = time.time()
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    classifier = pipeline("zero-shot-classification", model=MODEL, device=-1)  # CPU default; device=0 to use GPU
    print("Loaded MNLI classifier")

    points = fetch_points(client)
    print(f"Fetched {len(points)} points for classification")

    # Prepare mapping: Qdrant id -> title or fallback to url or empty
    items = []
    for p in points:
        payload = p.payload or {}
        title = payload.get("title") or payload.get("url") or ""
        items.append((p.id, title, payload))

    # Batch classify
    updates = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i+BATCH]
        texts = [t for (_id, t, _pl) in batch]
        if not texts:
            continue
        res = classifier(texts, candidate_labels=LABELS, truncation=True)

        # transformer pipeline returns list when input is list
        for (qid, txt, pl), out in zip(batch, res):
            labels = out["labels"]
            scores = out["scores"]
            best_label = labels[0]
            best_score = float(scores[0])
            if best_score < MIN_CONFIDENCE:
                assigned = "misc"
            else:
                assigned = best_label

            payload_update = {"topic": assigned, "topic_score": best_score}
            updates.append((qid, payload_update))

    # write updates in batches
    for j in range(0, len(updates), 500):
        seg = updates[j:j+500]
        id_list = [u[0] for u in seg]
        # Aggregated payload updates: qdrant set_payload supports same payload for many ids,
        # but here payload differs per id. So we call set_payload per id sequentially.
        for qid, payload in seg:
            client.set_payload(collection_name=COLLECTION_NAME, payload=payload, points=[qid])

        print(f"  wrote batch {j}/{len(updates)}")

    dur = time.time() - t0
    print(f"Topic classification done. items={len(items)}, time={dur:.1f}s")

if __name__ == "__main__":
    main()
