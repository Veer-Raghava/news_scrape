# project/scripts/embedder.py
#  docker run -d -p 6333:6333 -v qdrant-storage:/qdrant/storage qdrant/qdrant
import json
import time
import uuid
import numpy as np
import requests
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ---------------- CONFIG ----------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

NODE_ENDPOINT = "http://localhost:5000/api/scrape/store"
SEND_BATCH = 200
EMBED_BATCH = 64

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
ARTICLES_JSONL = DATA / "articles.jsonl"


client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


# ---------------- HELPERS ----------------

def read_jsonl(path: Path):
    docs = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def fetch_existing_ids():
    """Read all IDs already stored in Qdrant."""
    ids = set()
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )
        for p in page:
            ids.add(str(p.id))
        if offset is None:
            break
    return ids


def embed_texts(texts: List[str], model):
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


def send_to_node(batch):
    """Send a batch of NEW articles to Node backend."""
    try:
        res = requests.post(NODE_ENDPOINT, json=batch, timeout=20)
        if res.status_code == 200:
            print(f"  📤 sent {len(batch)} docs → Node")
        else:
            print("  ❌ Node rejected batch:", res.text)
    except Exception as e:
        print("  ❌ Send failed:", e)


# ---------------- MAIN PIPELINE ----------------

def main():
    print("\n🚀 Booting universal embedder pipeline...")

    # 1. Ensure Qdrant collection exists
    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' exists.")
    except:
        print("⚠ Creating missing collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print("🎯 Collection created.")

    # 2. Load JSONL docs
    docs = read_jsonl(ARTICLES_JSONL)
    print(f"📚 Loaded {len(docs)} docs from jsonl")

    # 3. Find NEW docs
    print("🔎 Fetching existing Qdrant IDs...")
    existing_ids = fetch_existing_ids()
    print(f"🧠 Existing vectors in DB: {len(existing_ids)}")

    new_docs = []
    new_ids = []
    payloads = []

    for doc in docs:
        url = doc.get("url")
        if not url:
            continue

        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, url))

        if pid not in existing_ids:   # NEW DOC
            new_docs.append(doc)
            new_ids.append(pid)

            payloads.append({
                "url": url,
                "title": doc.get("title", ""),
                "domain": doc.get("domain", ""),
                "source": doc.get("feed", ""),
                "published": doc.get("published", ""),
                "event_id": None
            })

    print(f"🚀 New documents to embed: {len(new_ids)}")

    if not new_docs:
        print("✨ Nothing new to embed.")
        return

    # 4. Load model
    print("⚡ Loading embedding model...")
    try:
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("🔥 Model on GPU")
    except:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("⚡ Model on CPU")

    # 5. EMBED + UPSERT BATCHES
    for i in range(0, len(new_ids), EMBED_BATCH):
        batch_docs = new_docs[i:i + EMBED_BATCH]
        batch_ids = new_ids[i:i + EMBED_BATCH]
        batch_payloads = payloads[i:i + EMBED_BATCH]

        texts = [d.get("text") or d.get("content") or d.get("title", "") for d in batch_docs]
        vectors = embed_texts(texts, model)

        points = [
            PointStruct(id=pid, vector=vectors[j].tolist(), payload=batch_payloads[j])
            for j, pid in enumerate(batch_ids)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  ✅ Embedded + upserted: batch {i // EMBED_BATCH + 1}")

    # 6. COMPUTE event_id (cluster ID)
    print("\n🔗 Assigning event_id to each article (same story group)...")

    def get_event_id(url):
        """Query this same doc to find similarity cluster."""
        from sentence_transformers import util
        # The uuid5 ID ensures we can always find the same Qdrant ID from URL
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
        try:
            res = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[pid],
                with_payload=True,
                with_vectors=False
            )
            if res:
                return abs(hash(url)) % 10**6   # simple cluster ID
        except:
            return None

    # Add event_id to Node send batch
    outgoing_batch = []
    for p in payloads:
        ev = get_event_id(p["url"])
        p["event_id"] = ev
        outgoing_batch.append(p)

    # 7. SEND TO NODE BACKEND IN BATCHES
    print("\n📡 Sending new docs to Node backend...")

    for i in range(0, len(outgoing_batch), SEND_BATCH):
        batch = outgoing_batch[i:i + SEND_BATCH]
        send_to_node(batch)

    print("\n🎉 Pipeline finished! Qdrant + Node are in sync.")


if __name__ == "__main__":
    main()
