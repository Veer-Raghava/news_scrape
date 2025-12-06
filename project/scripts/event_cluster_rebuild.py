# project/scripts/event_cluster_rebuild.py
"""
Event-level (same-story) clustering for news_articles.

Algorithm (simple, deterministic):
 - Scroll all points (id, vector, payload) from Qdrant.
 - For each point i:
     - query top_k neighbors (e.g. 20)
     - for each neighbor j with score >= SAME_STORY_THRESHOLD, union(i, j)
 - After processing all points, each union-find group is an event cluster.
 - Compute basic meta (size, medoid id, centroid)
 - Write back payload fields to Qdrant:
     - "event_id": int
     - "event_size": int
     - "similar_stories": [urls or qdrant-ids]
     - keep existing fields intact
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# CONFIG
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
VECTOR_SIZE = 768

SCROLL_PAGE_SIZE = 10_000
TOP_K = 20  # neighbors to check
SAME_STORY_THRESHOLD = 0.67  # your requested threshold

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
CLUSTERS_JSON = DATA_DIR / "event_clusters.json"

# ---------- Union-Find ----------
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

# ---------- Helpers ----------
def ensure_collection(client: QdrantClient):
    cols = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in cols:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

def fetch_all_points(client: QdrantClient):
    ids = []
    vecs = []
    payloads = []
    offset = None
    page = 0
    while True:
        page += 1
        pts, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not pts:
            break
        for p in pts:
            ids.append(str(p.id))
            vecs.append(np.array(p.vector, dtype=np.float32))
            payloads.append(p.payload or {})
        print(f"  page {page}: got {len(pts)} points (total {len(ids)})")
        if offset is None:
            break
    if not ids:
        return [], np.empty((0, VECTOR_SIZE), dtype=np.float32), []
    arr = np.stack(vecs, axis=0)
    return ids, arr, payloads

def make_event_id(seed: str) -> int:
    # deterministic small int id from a string (sha1 -> 8 hex -> int)
    h = hashlib.sha1(seed.encode("utf8")).hexdigest()[:16]
    return int(h, 16) % (10**9)

def compute_centroid_and_medoid(vecs, ids):
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = (centroid / norm).astype(np.float32)
    diffs = vecs - centroid
    dists = np.linalg.norm(diffs, axis=1)
    medoid_idx = int(dists.argmin())
    return centroid.tolist(), ids[medoid_idx]

# ---------- Main ----------
def main():
    t0 = time.time()
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    ids, vectors, payloads = fetch_all_points(client)
    n = len(ids)
    if n == 0:
        print("no points -> nothing to do")
        return

    print(f"Total points: {n}; building index-based grouping using top_k={TOP_K}, threshold={SAME_STORY_THRESHOLD}")
    uf = UF(n)

    # Map index->public id (url preferred)
    public_ids = []
    for p in payloads:
        url = p.get("url")
        public_ids.append(url if url else None)

    # For performance: query per-batch. We'll query each vector for top_k neighbors.
    for i in range(n):
        qvec = vectors[i].tolist()
        res = client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec,
            limit=TOP_K,
            with_payload=False,
            with_vectors=False,
        )
        for p in res.points:
            # p.id is Qdrant id (maybe int/str); we need to map to index j
            # scroll earlier preserved order, but query returns Qdrant IDs — map using ids list
            try:
                j = ids.index(str(p.id))
            except ValueError:
                # fallback: if Qdrant returned int id types, try raw id matching
                try:
                    j = ids.index(int(p.id))
                except Exception:
                    continue
            if j == i:
                continue
            if p.score is None:
                continue
            if p.score >= SAME_STORY_THRESHOLD:
                uf.union(i, j)

        if (i+1) % 500 == 0 or i == n-1:
            print(f"  processed {i+1}/{n} items")

    # Build groups
    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    print(f"Raw groups (by root count): {len(groups)}")

    # Convert to event clusters
    event_meta = {}
    event_id_map = {}  # idx -> event_id
    created = 0
    for root_idx, members in groups.items():
        # Create deterministic event id using medoid's URL or Qdrant id
        member_ids = [ids[m] for m in members]
        member_urls = [public_ids[m] for m in members if public_ids[m]]
        centroid_vecs = vectors[members]
        centroid, medoid_qid = compute_centroid_and_medoid(centroid_vecs, member_ids)

        seed = medoid_qid if medoid_qid else (member_urls[0] if member_urls else member_ids[0])
        event_id = make_event_id(seed)
        event_meta[str(event_id)] = {
            "event_id": event_id,
            "size": len(members),
            "medoid_id": medoid_qid,
            "member_qdrant_ids": member_ids,
            "member_urls": member_urls,
            "centroid": centroid,
        }
        for m in members:
            event_id_map[ids[m]] = event_id
        created += 1

    print(f"Created {created} event groups; writing back to Qdrant in batches...")

    # Write back: set_payload supports updating multiple points
    # We will write for each event group: event_id and event_size in payload for each member
    batch = []
    BATCH_SIZE = 500
    items = list(event_meta.values())
    for ev in items:
        event_id = ev["event_id"]
        size = ev["size"]
        member_qdrant_ids = ev["member_qdrant_ids"]
        # Prepare URL list or store small list in payload
        payload = {
            "event_id": event_id,
            "event_size": size,
            # keep a short list of urls if available for UI convenience (max 10)
            "similar_stories": ev["member_urls"][:10],
        }
        # qdrant client wants list of ids in its native type; we keep original strings
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=payload,
            points=member_qdrant_ids,
        )

    # Save event meta locally (optional)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTERS_JSON, "w", encoding="utf8") as f:
        json.dump(event_meta, f, indent=2)

    dur = time.time() - t0
    print(f"Done. events created: {created}, time: {dur:.1f}s")

if __name__ == "__main__":
    main()
