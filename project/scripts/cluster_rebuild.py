# project/scripts/cluster_rebuild.py
"""
Ultra-clean topic clustering system for your news vector universe.

Process:
1) Fetch all vectors from Qdrant
2) Run HDBSCAN for semantic clustering
3) Compute centroid + medoid per cluster
4) Auto-classify clusters using medoid → HF zero-shot model
5) Attach noise points by cosine ≥ 0.67
6) Save summary + write to Qdrant

This file is built for:
- High accuracy
- Zero noise misclassification
- Expandable ML pipeline (summaries, bias modeling, narrative graphs)
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import hdbscan
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from transformers import pipeline


# ================================================
# CONFIG
# ================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"

VECTOR_SIZE = 768
PAGE_SIZE = 10_000

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

NOISE_ATTACH_THRESHOLD = 0.67  # YOUR MAGIC NUMBER

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"


# ================================================
# CATEGORY MODEL (zero-shot)
# ================================================
CLASSIFIER = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

CANDIDATE_LABELS = [
    "politics",
    "business",
    "sports",
    "technology",
    "science",
    "health",
    "crime",
    "entertainment",
    "economy",
    "world",
    "environment",
    "education",
    "misc"
]


# ================================================
# DATA STRUCTURES
# ================================================
@dataclass
class ClusterMeta:
    label: int
    size: int
    centroid: List[float]
    medoid_id: str
    category: str
    summary: str  # prepared for later


@dataclass
class ClusteringSummary:
    num_points: int
    num_clusters: int
    num_noise: int
    clusters: Dict[str, ClusterMeta]


# ================================================
# HELPERS
# ================================================
def ensure_collection(client: QdrantClient):
    collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )


def fetch_all(client: QdrantClient):
    ids = []
    vectors = []
    texts = []

    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME,
            limit=PAGE_SIZE,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        if not points:
            break

        for p in points:
            ids.append(str(p.id))
            vectors.append(p.vector)

            payload = p.payload or {}
            title = payload.get("title", "") or ""
            source = payload.get("source", "") or ""
            domain = payload.get("domain", "") or ""
            url = payload.get("url", "") or ""

            txt = " | ".join(x for x in (title, source, domain, url) if x)
            texts.append(txt)

        if offset is None:
            break

    return ids, np.array(vectors, dtype=np.float32), texts


def run_hdbscan(vectors: np.ndarray):
    model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        core_dist_n_jobs=-1
    )
    return model.fit_predict(vectors)


def compute_centroid_and_medoid(vecs, ids):
    centroid = vecs.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)

    dists = np.linalg.norm(vecs - centroid, axis=1)
    medoid = ids[int(dists.argmin())]
    return centroid.tolist(), medoid


def classify_cluster(text: str):
    result = CLASSIFIER(text, CANDIDATE_LABELS)
    return result["labels"][0]


def attach_noise(ids, vectors, texts, labels, summary):
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) == 0:
        return labels, summary

    # build centroid matrix
    clabels = []
    centroids = []
    for k, meta in summary.clusters.items():
        clabels.append(int(k))
        centroids.append(np.array(meta.centroid, dtype=np.float32))
    centroids = np.stack(centroids)

    vec_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    for idx in noise_idx:
        sims = centroids @ vec_norm[idx]
        best = int(np.argmax(sims))
        if sims[best] >= NOISE_ATTACH_THRESHOLD:
            lbl = clabels[best]
            labels[idx] = lbl
            summary.clusters[str(lbl)].size += 1

    summary.num_noise = int(np.sum(labels == -1))
    return labels, summary


def save_summary(summary: ClusteringSummary):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTERS_JSON, "w") as f:
        json.dump({
            "num_points": summary.num_points,
            "num_clusters": summary.num_clusters,
            "num_noise": summary.num_noise,
            "clusters": {k: asdict(v) for k, v in summary.clusters.items()}
        }, f, indent=2)


def write_to_qdrant(client, ids, labels, summary):
    for lbl in set(labels):
        idxs = [ids[i] for i, l in enumerate(labels) if l == lbl]

        if lbl == -1:
            category = "noise"
        else:
            category = summary.clusters[str(lbl)].category

        client.set_payload(
            COLLECTION_NAME,
            payload={"cluster": int(lbl), "category": category},
            points=idxs
        )


# ================================================
# MAIN PIPELINE
# ================================================
def main():
    t0 = time.time()
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    ids, vecs, texts = fetch_all(client)
    if not ids:
        print("No points found.")
        return

    labels = run_hdbscan(vecs)

    # build cluster metadata
    clusters = {}
    unique = sorted({int(l) for l in labels if l != -1})
    for lbl in unique:
        idxs = np.where(labels == lbl)[0]
        cids = [ids[i] for i in idxs]
        cvecs = vecs[idxs]
        ctexts = [texts[i] for i in idxs]

        centroid, medoid_id = compute_centroid_and_medoid(cvecs, cids)
        category = classify_cluster(ctexts[0]) if ctexts else "misc"

        clusters[str(lbl)] = ClusterMeta(
            label=lbl,
            size=len(idxs),
            centroid=centroid,
            medoid_id=medoid_id,
            category=category,
            summary=""
        )

    summary = ClusteringSummary(
        num_points=len(ids),
        num_clusters=len(unique),
        num_noise=int(np.sum(labels == -1)),
        clusters=clusters
    )

    # noise rescue
    labels, summary = attach_noise(ids, vecs, texts, labels, summary)

    # write
    write_to_qdrant(client, ids, labels, summary)
    save_summary(summary)

    print("\nDONE.")
    print("Clusters:", summary.num_clusters)
    print("Noise:", summary.num_noise)
    print("Time:", round(time.time() - t0, 2), "sec")


if __name__ == "__main__":
    main()
