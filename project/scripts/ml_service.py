# project/scripts/ml_service.py
# Run with:
# uvicorn project.scripts.ml_service:app --reload --port 8000

from typing import List, Optional, Dict
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer


# ------------------ CONFIG ------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

DEFAULT_SAME_STORY_THRESHOLD = 0.82
DEFAULT_CLUSTER_ASSIGN_THRESHOLD = 0.72

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"

app = FastAPI(title="News ML Service", version="2.2.0")


# ------------------ Pydantic Schemas ------------------

class HealthResponse(BaseModel):
    status: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    title: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    published: Optional[str] = None
    cluster: Optional[int] = None


class SameStoryByIdRequest(BaseModel):
    id: str
    limit: int = 25
    min_score: float = DEFAULT_SAME_STORY_THRESHOLD


class SameStoryByTextRequest(BaseModel):
    text: str
    limit: int = 25
    min_score: float = DEFAULT_SAME_STORY_THRESHOLD


class SameStoryResponse(BaseModel):
    anchor: SearchResult
    items: List[SearchResult]
    took_seconds: float
    took_minutes: float


class ClusterAssignRequest(BaseModel):
    text: str
    min_score: float = DEFAULT_CLUSTER_ASSIGN_THRESHOLD


class ClusterAssignResponse(BaseModel):
    has_clusters: bool
    assigned_cluster: int
    best_score: float
    num_clusters_considered: int


# ------------------ Startup ------------------

@app.on_event("startup")
def startup_event():

    print("🌍 booting ML service for all souls…")

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists
    collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        print(f"⚠ Missing collection '{COLLECTION_NAME}'. Creating…")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print("✅ Collection created")

    # Load embedding model
    try:
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("🔥 Model on GPU")
    except Exception:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("⚡ Model on CPU")

    # Load topic clusters (optional)
    cluster_centroids = None
    cluster_labels = None

    if CLUSTERS_JSON.exists():
        try:
            with open(CLUSTERS_JSON, "r", encoding="utf8") as f:
                data = json.load(f)

            clusters = data.get("clusters", {})
            centroids = []
            labels = []

            for label_str, meta in clusters.items():
                labels.append(int(label_str))
                centroids.append(np.array(meta["centroid"], dtype=np.float32))

            if centroids:
                cluster_centroids = np.stack(centroids)
                cluster_labels = np.array(labels)

            print(f"✅ Loaded {len(labels)} cluster centroids")
        except Exception as e:
            print("⚠ Failed loading clusters.json:", e)

    app.state.client = client
    app.state.model = model
    app.state.cluster_centroids = cluster_centroids
    app.state.cluster_labels = cluster_labels

    print("✅ ML service awake and alive")


# ------------------ Helpers ------------------

def embed_text(text: str) -> np.ndarray:
    model = app.state.model
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype(np.float32)


def scoredpoint_to_result(p: ScoredPoint) -> SearchResult:
    payload = p.payload or {}
    url = payload.get("url")

    return SearchResult(
        id=url if url else str(p.id),
        score=float(p.score),
        title=payload.get("title"),
        url=url,
        source=payload.get("source"),
        domain=payload.get("domain"),
        published=payload.get("published"),
        cluster=payload.get("cluster"),
    )


def fetch_anchor_by_url(client: QdrantClient, url: str) -> ScoredPoint:

    flt = Filter(
        must=[FieldCondition(key="url", match=MatchValue(value=url))]
    )

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        raise HTTPException(404, "URL not found in Qdrant")

    p = points[0]
    return ScoredPoint(
        id=p.id,
        payload=p.payload,
        vector=p.vector,
        score=1.0,
        version=0
    )


# ------------------ ROUTES ------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


# ====== 1) SEMANTIC SEARCH ======

@app.post("/search_text", response_model=List[SearchResult])
def search_text(body: SearchRequest):
    client = app.state.client

    vec = embed_text(body.query).tolist()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=body.limit,
        with_payload=True,
        with_vectors=False
    )

    return [scoredpoint_to_result(p) for p in res.points]


# ====== 2) SAME STORY BY URL ======

@app.post("/same_story_by_id", response_model=SameStoryResponse)
def same_story_by_id(body: SameStoryByIdRequest):

    client = app.state.client

    anchor = fetch_anchor_by_url(client, body.id)
    anchor_vec = np.array(anchor.vector, dtype=np.float32)

    t0 = time.time()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=anchor_vec.tolist(),
        limit=body.limit,
        with_payload=True,
        with_vectors=False
    )

    hits = [
        p for p in res.points
        if p.score is not None and p.score >= body.min_score
    ]
    hits.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in hits]

    took = time.time() - t0

    return SameStoryResponse(
        anchor=scoredpoint_to_result(anchor),
        items=items,
        took_seconds=round(took, 3),
        took_minutes=round(took / 60, 3),
    )


# ====== 3) SAME STORY BY TEXT ======

@app.post("/same_story_by_text", response_model=SameStoryResponse)
def same_story_by_text(body: SameStoryByTextRequest):

    client = app.state.client
    vec = embed_text(body.text)

    t0 = time.time()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=body.limit,
        with_payload=True,
        with_vectors=False
    )

    hits = [
        p for p in res.points
        if p.score is not None and p.score >= body.min_score
    ]
    hits.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in hits]

    took = time.time() - t0

    return SameStoryResponse(
        anchor=SearchResult(id="query-text", score=1.0, title=body.text),
        items=items,
        took_seconds=round(took, 3),
        took_minutes=round(took / 60, 3),
    )


# ====== 4) CLUSTER ASSIGNMENT ======

@app.post("/route_cluster", response_model=ClusterAssignResponse)
def route_cluster(body: ClusterAssignRequest):

    centroids = app.state.cluster_centroids
    labels = app.state.cluster_labels

    if centroids is None:
        return ClusterAssignResponse(
            has_clusters=False,
            assigned_cluster=-1,
            best_score=0.0,
            num_clusters_considered=0,
        )

    vec = embed_text(body.text)
    sims = centroids @ vec

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_label = int(labels[best_idx])

    if best_score < body.min_score:
        assigned = -1
    else:
        assigned = best_label

    return ClusterAssignResponse(
        has_clusters=True,
        assigned_cluster=assigned,
        best_score=best_score,
        num_clusters_considered=len(labels),
    )


# ====== 5) EVENT ID LOOKUP (YOUR NEW ENDPOINT) ======

class EventIdRequest(BaseModel):
    url: str


@app.post("/event-id")
def get_event_id(body: EventIdRequest):

    client = app.state.client

    flt = Filter(
        must=[
            FieldCondition(
                key="url",
                match=MatchValue(value=body.url)
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        return {"event_id": None}

    payload = points[0].payload or {}
    return {"event_id": payload.get("event_id")}
print("REGISTERED ROUTES:")
for r in app.routes:
    print(r.path)
