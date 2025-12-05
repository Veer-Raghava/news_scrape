# project/scripts/cluster_rebuild.py
"""
Topic-aware clustering job for the news vector universe.

Modes (auto-detected):

1) INITIAL FULL CLUSTERING
   - Triggered when data/clusters.json does NOT exist.
   - Fetches ALL points (id, vector, payload) from Qdrant.
   - Runs HDBSCAN on full set.
   - Computes centroid + medoid per cluster.
   - Infers a high-level TOPIC CATEGORY for each cluster
     (with sub-flavour like sports.cricket, cine.bollywood, economy.agri, etc.).
   - Tries to "rescue" HDBSCAN noise points by attaching them
     to the nearest centroid if similarity is high enough.
   - Writes `cluster` (int) and `category` (str) into each point's payload.
   - Saves metadata to data/clusters.json.

2) NOISE-ONLY RE-CLUSTERING
   - Triggered when clusters.json exists.
   - Fetches ONLY points with cluster == -1 (noise/unassigned) or missing.
   - Runs HDBSCAN on this subset.
   - For each new cluster, assigns NEW global cluster IDs
     (continuing from previous max label).
   - Infers topic category for each new cluster.
   - Again tries to attach any remaining noise to nearest centroid.
   - Writes labels + categories back into Qdrant.
   - Updates clusters.json with appended cluster metadata.

→ Use this for broad topic buckets like politics.india / sports.cricket / economy.agri / cine.bollywood…
→ Use the ML service /same_story_* endpoints for per-event "same news, many publishers".
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hdbscan
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ----------------- CONFIG ----------------- #

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
VECTOR_SIZE = 768
SCROLL_PAGE_SIZE = 10_000

# HDBSCAN config → tune if needed
# smaller min_cluster_size → more clusters, fewer points in noise
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# After HDBSCAN, try to attach noise points to nearest centroid
# using cosine similarity on normalized vectors
NOISE_ATTACH_THRESHOLD = 0.68  # SEMANTIC ATTACH threshold

BASE = Path(__file__).resolve().parents[1]  # project/
DATA_DIR = BASE / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"

# ----------------- TOPIC CATEGORY DEFINITIONS ----------------- #
# The trick here:
# - We use very specific keywords, avoid super-generic ones like "crisis" or "strike".
# - We lean on domain/source signals (bbc_sport, *_business, etc.) by
#   including them in the text we score (title + source + domain).
#
# Each category gets a list of "strong hints". We later aggregate scores
# across all titles in a cluster and pick the strongest.

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    # politics / governance / public policy
    "politics.india": [
        "lok sabha", "rajya sabha", "mla", "mp ",
        "bjp", "congress", "aap ", "shiv sena",
        "chief minister", "cm ", "assembly election",
        "state election", "parliament", "raj bhavan",
        "governor", "cabinet reshuffle", "model code of conduct",
        "election commission", "eci ",
        "new delhi", "union minister",
    ],
    "politics.global": [
        "white house", "us election", "presidential election",
        "senate", "congress ", "republican", "democrat",
        "labour party", "conservative party", "tory",
        "european union", "eu ", "downing street",
        "prime minister of", "chancellor of",
    ],

    # geopolitics / war / diplomacy / climate summits
    "geopolitics.conflict": [
        "border clash", "ceasefire", "airstrike", "air strike",
        "missile", "rocket fire", "shelling", "troops",
        "gaza", "israel", "palestine", "west bank",
        "ukraine", "russia", "nato", "iran", "iraq", "syria",
        "hezbollah", "hamas", "taliban",
        "rebel group", "militant group",
    ],
    "geopolitics.climate": [
        "cop30", "cop 30", "cop29", "climate summit",
        "climate talks", "climate finance", "carbon market",
        "deforestation", "loss and damage", "paris agreement",
        "un climate", "unfccc",
    ],

    # economy / markets / business
    "economy.macro": [
        "gdp growth", "gdp contracts", "inflation",
        "fiscal deficit", "current account deficit",
        "recession", "slowdown", "macro data",
        "industrial output", "cpi ", "wpi ",
        "unemployment rate", "jobs data",
        "monetary policy committee", "mpc meeting",
        "central bank", "interest rate decision",
    ],
    "economy.markets": [
        "sensex", "nifty", "dow jones", "nasdaq",
        "stock market", "equity market", "shares jump",
        "shares fall", "stock surges", "stock plunges",
        "bond yields", "treasury yields", "rupee",
        "forex reserves", "currency market",
        "livemint_latest", "moneycontrol",
    ],
    "economy.agri": [
        "cotton", "ginning mill", "paddy", "rice", "wheat",
        "farmers", "minimum support price", "msp ",
        "procurement centre", "mandi", "crop failure",
        "agri export", "fertiliser subsidy",
        "sugarcane", "soybean", "oilseed",
        "cci chairman", "agriculture department",
    ],
    "business.corporate": [
        "ipo", "rights issue", "fpo ", "acquisition",
        "merger", "stake sale", "takeover bid",
        "earnings", "quarterly results", "q1 results",
        "q2 results", "q3 results", "q4 results",
        "profit rises", "profit falls", "loss widens",
        "loss narrows", "startup", "valuation",
        "funding round", "series a", "series b",
        "unicorn", "board of directors", "ceo resigns",
        "cfo resigns",
    ],
    "business.crypto": [
        "bitcoin", "ethereum", "crypto exchange",
        "stablecoin", "defi", "nft ", "blockchain",
        "web3", "token sale",
    ],

    # tech / science / health
    "tech.ai": [
        "ai model", "artificial intelligence",
        "machine learning", "neural network", "deep learning",
        "chatbot", "openai", "gpt", "llm", "generative ai",
        "fine-tuning", "prompt engineering", "inference api",
    ],
    "tech.consumer": [
        "iphone", "android phone", "smartphone",
        "laptop", "macbook", "windows update", "ios",
        "android 15", "playstation", "xbox",
        "gaming console", "headphones", "earbuds",
        "smartwatch", "wearable", "gadget review",
    ],
    "tech.cyber": [
        "data breach", "ransomware", "hacked", "cyber attack",
        "credential stuffing", "malware", "phishing",
        "security patch", "vulnerability", "zero-day",
        "bug bounty", "leaked database",
    ],
    "science.space": [
        "launch vehicle", "satellite", "rocket",
        "mission control", "isro", "nasa", "spacex",
        "chandrayaan", "lunar mission", "mars mission",
        "space telescope", "orbiter", "lander",
    ],
"health.medical": [
        "vaccine", "virus", "covid", "covid-19", "covid19",
        "infection", "hospitalised", "hospitalized",
        "intensive care", "icu", "icu ward",
        "outbreak", "epidemic", "pandemic",
        "health ministry", "health department",
        "who ", "world health organization",
        "diabetes", "hypertension", "cardiac arrest",
        "heart attack", "stroke", "cancer",
        "insulin", "antibiotics", "heart drugs",
        "drug shortage", "medicine shortage",
        "health-wellness", "health/wellness",
    ],
    "science.research": [
        "study finds", "researchers", "scientists",
        "peer reviewed", "peer-reviewed", "journal",
        "published in", "trial", "experiment", "randomised",
    ],

    # sports (with sub-types)
    "sports.cricket": [
        "cricket", "test match", "odi", "t20", "t-20",
        "ipl ", "wpl ", "wicket", "run chase", "run-chase",
        "century", "half-century", "fifty off",
        "bowler", "batsman", "all-rounder",
        "batting lineup", "bowling attack",
        "bazball", "ashes", "over rate",
        "bbL ", "psl ",
        "bbc_sport", "espncricinfo", "cricbuzz",
    ],
  "sports.football": [
        "football", "soccer", "la liga", "serie a",
        "premier league", "epl ", "bundesliga",
        "champions league", "europa league",
        "goalkeeper", "goalkeeping",
        "goal", "brace", "hat-trick", "hat trick",
        "striker", "midfielder", "defender", "full-back",
        "red card", "yellow card", "penalty shootout",
        "stoppage time winner",
        # club / team hints
        "fc ", "fc,", "man utd", "liverpool", "newcastle",
        "real madrid", "barcelona", "psg", "bayern",
        "lionesses", "tottenham", "spurs", "richarlison",
    ],
    "sports.other": [
        "tennis", "grand slam", "badminton", "shuttler",
        "olympics", "asian games", "athletics", "marathon",
        "sprinter", "wrestling", "kabaddi", "hockey",
        "shooting world cup", "10m air pistol", "10m air rifle",
        "air pistol", "air rifle", "pistol event", "rifle event",
        "squash", "boxing bout",
    ],

    # entertainment / culture / celebrity
    "cine.bollywood": [
        "bollywood", "hindi film", "karan johar",
        "salman khan", "katrina kaif", "alia bhatt",
        "shah rukh", "srk", "box office collection",
        "opening weekend", "song release", "trailer out",
        "item number", "masala entertainer",
    ],
    "cine.south": [
        "tollywood", "telugu film", "tamil film",
        "kollywood", "mollywood", "sandalwood",
        "pan-india release", "mass entertainer",
    ],
    "cine.hollywood": [
        "hollywood", "oscars", "academy awards",
        "golden globes", "marvel", "dc universe",
        "netflix original", "hbo series", "prime video",
        "disney+", "hulu series",
    ],
    "culture.pop": [
        "concert", "tour dates", "music festival",
        "music video", "rapper", "pop star",
        "album release", "streaming numbers",
        "billboard chart", "grammy", "grammys",
    ],
    "celebrity.life": [
        "postpones wedding", "wedding postponed",
        "wedding delayed", "marries", "ties the knot",
        "engagement", "fiancé", "wife", "husband",
        "relationship rumours", "dating rumours",
        "viral selfie", "instagram post",
        "nazar is real", "smriti mandhana",
        "ex-wife", "ex husband", "private ceremony",
    ],
    # crime / domestic / terror
    "crime.domestic": [
        # generic violent crime
        "murder", "killed", "shot dead", "stabbing", "stabbed",
        "brutal stabbing", "assaulted", "assault", "lynched",
        "kidnapped", "kidnapping", "abducted", "abduction",
        "minor girl", "rape", "sexual assault",
        # policing / legal
        "fir registered", "fir filed", "fir ", "chargesheet",
        "charge-sheet", "custody", "police remand",
        "police custody", "police case",
        # fraud / scam
        "fraud worth", "embezzlement", "scam", "ponzi",
        # gun / shooting crime
        "mall shooting", "school shooting", "mass shooting",
        "active shooter", "open fire", "opened fire",
        "fired shots", "gunman", "gunfire",
        "shot at", "shot in the", "gunshot",
        # cyber-crime-ish
        "cyber crime", "cybercrime", "online fraud", "upi fraud",
    ],
    "crime.terror": [
        "terrorist", "terror group", "terror outfit", "terror module",
        "sleeper cell", "bomb blast", "serial blasts",
        "ied", "improvised explosive device",
        "naxal", "maoist", "red corridor",
        "extremist outfit", "ulfa", "isis",
        "islamic state", "jihadist", "fidayeen",
        "suicide bomber", "terror attack",
    ],

    # accidents / disasters (non-terror)
    "accident.disaster": [
        # air / train / road
        "plane crash", "jet crash", "fighter jet crash",
        "aircraft crash", "helicopter crash",
        "chopper crash", "air crash",
        "train accident", "train derailment",
        "derailed near", "bus overturns", "bus crash",
        "road mishap", "road accident", "car crash",
        "pile-up", "pile up",
        # bridges / buildings / industrial
        "building collapse", "bridge collapse",
        "under-construction bridge", "industrial accident",
        "factory fire", "warehouse fire",
        # lorry / truck
        "truck overturns", "lorry overturns",
        "lorry skids off", "container lorry", "tanker overturns",
        "skids off road", "vehicle skids off",
        # disasters / fire
        "massive fire breaks out", "fire breaks out at",
        "warehouse blaze", "factory blaze",
        "ship capsizes", "boat capsizes",
    ],

    # environment / weather / climate impacts
    "env.weather": [
        "rainfall", "heavy rain", "flood", "flash flood",
        "landslide", "cyclone", "storm", "typhoon",
        "heatwave", "cold wave", "monsoon", "drought",
        "orange alert", "red alert", "yellow alert",
    ],
    "env.climate": [
        "climate crisis", "global warming", "net zero",
        "carbon emissions", "greenhouse gas",
        "renewable energy", "solar plant", "wind farm",
        "climate action plan",
    ],

    # society / education / social media etc.
    "society.education": [
        "university", "college", "students union",
        "exam cancelled", "exam postponed", "results declared",
        "board exam", "entrance test", "scholarship",
        "campus protest", "ugc ", "ugc net", "neet ",
        "jee main", "jee advanced",
    ],
    "society.social": [
        "viral video", "social media backlash",
        "trolled on", "instagram reel", "reels trend",
        "x (formerly twitter)", "tweeted", "retweet",
        "facebook post", "influencer", "content creator",
        "internet reacts", "meme fest",
        "horoscope", "zodiac", "aries", "taurus", "gemini",
        "cancer ", "leo ", "virgo ", "libra ", "scorpio ",
        "sagittarius", "capricorn", "aquarius", "pisces",
    ],

    # accidents / disasters (non-terror)
    "accident.disaster": [
        "plane crash", "jet crash", "fighter jet crash",
        "aircraft crash", "helicopter crash",
        "train accident", "train derailment",
        "bus overturns", "bus crash", "road mishap",
        "pile-up", "building collapse", "bridge collapse",
        "industrial accident", "factory fire",
    ],
}

CATEGORY_FALLBACK = "misc"

# Lower number = higher priority when counts & scores tie
CATEGORY_PRIORITY_ORDER: Dict[str, int] = {
    # high-salience “this is clearly about X”
    "crime.terror": 1,
    "crime.domestic": 2,
    "accident.disaster": 3,

    "sports.cricket": 10,
    "sports.football": 11,
    "sports.other": 12,

    "celebrity.life": 15,
    "cine.bollywood": 16,
    "cine.south": 17,
    "cine.hollywood": 18,
    "culture.pop": 19,

    "health.medical": 20,
    "science.space": 21,
    "science.research": 22,

    "economy.agri": 25,
    "economy.markets": 26,
    "economy.macro": 27,
    "business.corporate": 28,
    "business.crypto": 29,

    "politics.india": 40,
    "politics.global": 41,
    "geopolitics.conflict": 42,
    "geopolitics.climate": 43,

    "env.weather": 50,
    "env.climate": 51,

    "society.education": 60,
    "society.social": 61,

    "misc": 99,
}
DEFAULT_CATEGORY_PRIORITY = 80



# ----------------- DATA STRUCTURES ----------------- #

@dataclass
class ClusterMeta:
    label: int
    size: int
    centroid: List[float]
    medoid_id: str
    category: str  # high-level topic label (maybe with subcategory)


@dataclass
class ClusteringSummary:
    num_points: int
    num_clusters: int
    num_noise: int
    clusters: Dict[str, ClusterMeta]  # key: label as string


# ----------------- COMMON HELPERS ----------------- #

def ensure_collection(client: QdrantClient) -> None:
    collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Creating empty...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Created empty collection '{COLLECTION_NAME}'.")
    else:
        info = client.get_collection(COLLECTION_NAME)
        size = info.config.params.vectors.size
        if size != VECTOR_SIZE:
            print(f"❗ Warning: collection dim={size}, expected {VECTOR_SIZE}.")
        print(f"✅ Collection '{COLLECTION_NAME}' exists.")


def run_hdbscan(vectors: np.ndarray) -> np.ndarray:
    print("🧠 Running HDBSCAN over topic space...")
    t0 = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(vectors)

    dur = time.time() - t0
    dur_min = dur / 60
    n_clusters = len(set(int(l) for l in labels if l != -1))
    n_noise = int(np.sum(labels == -1))

    print(f"✅ HDBSCAN done in {dur:.1f}s ({dur_min:.2f} min)")
    print(f"   → clusters (excluding noise): {n_clusters}")
    print(f"   → noise points: {n_noise}")

    return labels


def compute_centroid_and_medoid(
    cluster_vectors: np.ndarray,
    cluster_ids: List[str],
) -> Tuple[List[float], str]:
    centroid = cluster_vectors.mean(axis=0)
    # normalize for cosine
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = (centroid / norm).astype(np.float32)

    diffs = cluster_vectors - centroid
    dists = np.linalg.norm(diffs, axis=1)
    medoid_idx = int(dists.argmin())
    medoid_id = cluster_ids[medoid_idx]
    return centroid.tolist(), medoid_id


# ----------------- CATEGORY INFERENCE ----------------- #

def score_text_categories(text: str) -> Dict[str, int]:
    """
    Return simple keyword-based scores per category for a title/text.
    We *only* look for substring matches; no fancy NLP here.

    NOTE: we deliberately do NOT include ultra-generic words like "crisis", "strike"
    in crime/terror so that cotton crisis & worker strikes don't become "crime.terror".
    """
    text_l = text.lower()
    scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS.keys()}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                scores[cat] += 1
    return scores


def infer_cluster_category(texts: List[str]) -> str:
    """
    Decide a cluster's category based on per-article votes.

    Steps:
      - For each article text, compute category scores
      - Take that article's best category (if any)
      - Count how many articles picked each category
      - Aggregate scores for tie-breaking
      - Then choose with:
          1) max article count
          2) max total score
          3) highest priority (lower number wins)
    """
    if not texts:
        return CATEGORY_FALLBACK

    # how many articles chose this category as "best"
    cat_title_count: Dict[str, int] = {
        cat: 0 for cat in CATEGORY_KEYWORDS.keys()
    }
    # sum of best-scores across titles
    cat_score_sum: Dict[str, int] = {
        cat: 0 for cat in CATEGORY_KEYWORDS.keys()
    }

    for t in texts:
        scores = score_text_categories(t)
        if not scores:
            continue

        best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score <= 0:
            continue

        cat_title_count[best_cat] += 1
        cat_score_sum[best_cat] += best_score

    # filter to categories that actually got at least one “vote”
    active = [
        (cat, cat_title_count[cat], cat_score_sum[cat])
        for cat in CATEGORY_KEYWORDS.keys()
        if cat_title_count[cat] > 0
    ]
    if not active:
        return CATEGORY_FALLBACK

    def sort_key(item):
        cat, count, score_sum = item
        priority = CATEGORY_PRIORITY_ORDER.get(cat, DEFAULT_CATEGORY_PRIORITY)
        # We want:
        #  - more titles first
        #  - then more score
        #  - then lower priority number wins
        return (-count, -score_sum, priority)

    active.sort(key=sort_key)
    best_cat = active[0][0]
    return best_cat


# ----------------- FULL FETCH (INITIAL) ----------------- #

def fetch_all_points(
    client: QdrantClient,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Fetch ALL points' ids, vectors, and topic-texts from Qdrant.

    "topic-text" = title + source + domain (joined) so category
    inference can also use which section/domain it came from.
    """
    print("🔎 Fetching ALL points from Qdrant (ids + vectors + titles/source/domain)...")
    ids: List[str] = []
    vecs: List[List[float]] = []
    topic_texts: List[str] = []

    offset = None
    page_idx = 0

    while True:
        page_idx += 1
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for p in points:
            ids.append(str(p.id))
            vecs.append(p.vector)

            payload = p.payload or {}
            title = payload.get("title", "") or ""
            source = payload.get("source", "") or ""
            domain = payload.get("domain", "") or ""
            url = payload.get("url", "") or ""
            topic_text = " | ".join(
                part for part in (title, source, domain, url) if part
            )
            topic_texts.append(topic_text)


        print(f"  📦 Page {page_idx}: {len(points)} points (total {len(ids)})")
        if offset is None:
            break

    if not ids:
        print("🛑 No points found. Nothing to cluster.")
        return [], np.empty((0, VECTOR_SIZE), dtype=np.float32), []

    arr = np.array(vecs, dtype=np.float32)
    print(f"✅ Total points: {len(ids)}, dim={arr.shape[1]}")
    return ids, arr, topic_texts


def build_full_summary(
    ids: List[str],
    vectors: np.ndarray,
    topic_texts: List[str],
    labels: np.ndarray,
) -> ClusteringSummary:
    print("📊 Building full clustering summary with topic categories...")
    meta: Dict[str, ClusterMeta] = {}

    unique_labels = sorted({int(l) for l in labels if l != -1})
    num_points = len(ids)
    num_clusters = len(unique_labels)
    num_noise = int(np.sum(labels == -1))

    for label in unique_labels:
        mask = (labels == label)
        idxs = np.where(mask)[0]
        cluster_ids = [ids[i] for i in idxs]
        cluster_vecs = vectors[mask]
        cluster_texts = [topic_texts[i] for i in idxs]

        centroid, medoid_id = compute_centroid_and_medoid(cluster_vecs, cluster_ids)
        category = infer_cluster_category(cluster_texts)

        cm = ClusterMeta(
            label=label,
            size=len(cluster_ids),
            centroid=centroid,
            medoid_id=medoid_id,
            category=category,
        )
        meta[str(label)] = cm
        print(
            f"   • cluster {label:>3}: size={cm.size:>4}, "
            f"category={cm.category:<20}, medoid_id={cm.medoid_id[:8]}..."
        )

    summary = ClusteringSummary(
        num_points=num_points,
        num_clusters=num_clusters,
        num_noise=num_noise,
        clusters=meta,
    )
    print(f"✅ Full summary → clusters={num_clusters}, noise={num_noise}")
    return summary


# ----------------- NOISE FETCH (OUTLIERS ONLY) ----------------- #

def fetch_noise_points(
    client: QdrantClient,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Fetch points that currently have cluster == -1 (or no cluster).

    Again we build "topic-text" as title+source+domain for category inference.
    """
    print("🔎 Fetching NOISE / unassigned points (cluster == -1 or missing)...")
    ids: List[str] = []
    vecs: List[List[float]] = []
    topic_texts: List[str] = []

    offset = None
    page_idx = 0

    while True:
        page_idx += 1
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for p in points:
                payload = p.payload or {}
                title = payload.get("title", "") or ""
                source = payload.get("source", "") or ""
                domain = payload.get("domain", "") or ""
                url = payload.get("url", "") or ""
                topic_text = " | ".join(
                    part for part in (title, source, domain, url) if part
                )
                topic_texts.append(topic_text)


        print(
            f"  📦 Page {page_idx}: scanned {len(points)} points "
            f"(noise so far {len(ids)})"
        )
        if offset is None:
            break

    if not ids:
        print("🛑 No noise/unassigned points found.")
        return [], np.empty((0, VECTOR_SIZE), dtype=np.float32), []

    arr = np.array(vecs, dtype=np.float32)
    print(f"✅ Noise points: {len(ids)}, dim={arr.shape[1]}")
    return ids, arr, topic_texts


# ----------------- CLUSTERS.JSON I/O ----------------- #

def load_existing_summary() -> Optional[ClusteringSummary]:
    if not CLUSTERS_JSON.exists():
        return None

    with open(CLUSTERS_JSON, "r", encoding="utf8") as f:
        data = json.load(f)

    clusters_dict: Dict[str, ClusterMeta] = {}
    for label, meta in data.get("clusters", {}).items():
        clusters_dict[label] = ClusterMeta(
            label=meta["label"],
            size=meta["size"],
            centroid=meta["centroid"],
            medoid_id=meta["medoid_id"],
            category=meta.get("category", CATEGORY_FALLBACK),
        )

    # num_points is mostly informative; we keep previous value
    return ClusteringSummary(
        num_points=data.get("num_points", 0),
        num_clusters=len(clusters_dict),
        num_noise=data.get("num_noise", 0),
        clusters=clusters_dict,
    )


def save_summary(summary: ClusteringSummary) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_points": summary.num_points,
        "num_clusters": summary.num_clusters,
        "num_noise": summary.num_noise,
        "clusters": {
            label: asdict(meta)
            for label, meta in summary.clusters.items()
        },
    }
    with open(CLUSTERS_JSON, "w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 Wrote cluster metadata to {CLUSTERS_JSON}")


# ----------------- WRITE LABELS TO QDRANT ----------------- #

def write_labels_to_qdrant(
    client: QdrantClient,
    ids: List[str],
    labels: np.ndarray,
    summary: ClusteringSummary,
) -> None:
    """
    Update Qdrant payload with:
      - cluster: int
      - category: str (topic)
    Noise (cluster=-1) gets category="noise".
    """
    print("📝 Writing cluster labels + categories into Qdrant payloads...")

    label_to_ids: Dict[int, List[str]] = {}
    for point_id, label in zip(ids, labels):
        label_to_ids.setdefault(int(label), []).append(point_id)

    for label, group_ids in label_to_ids.items():
        if not group_ids:
            continue

        if label == -1:
            category = "noise"
        else:
            meta = summary.clusters.get(str(label))
            category = meta.category if meta is not None else CATEGORY_FALLBACK

        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"cluster": label, "category": category},
            points=group_ids,
        )
        print(
            f"   • cluster={label:>3}, category={category:<20} "
            f"→ updated {len(group_ids)} points"
        )

    print("✅ Finished updating Qdrant payloads.")


# ----------------- NOISE ATTACH TO CENTROIDS ----------------- #

def attach_noise_to_centroids(
    ids: List[str],
    vectors: np.ndarray,
    topic_texts: List[str],  # not used right now, kept for possible future logic
    labels: np.ndarray,
    summary: ClusteringSummary,
) -> Tuple[np.ndarray, ClusteringSummary]:
    """
    HDBSCAN will mark some points as -1 (noise).
    Here we try to attach part of that noise to nearest existing centroids
    if cosine similarity is high enough (NOISE_ATTACH_THRESHOLD).

    This keeps noise small without going fully crazy.
    """
    if not summary.clusters:
        return labels, summary

    noise_mask = (labels == -1)
    num_noise = int(np.sum(noise_mask))
    if num_noise == 0:
        return labels, summary

    print(f"🔧 Trying to rescue {num_noise} noise points by centroid similarity...")

    # Build centroid matrix
    cluster_labels = []
    centroids = []
    for lbl_str, meta in summary.clusters.items():
        cluster_labels.append(int(lbl_str))
        centroids.append(np.array(meta.centroid, dtype=np.float32))
    centroids = np.stack(centroids, axis=0)  # (C, D)
    cluster_labels = np.array(cluster_labels, dtype=np.int32)

    # Normalize input vectors (they should already be normalized, but be safe)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vec_normed = vectors / norms

    noise_indices = np.where(noise_mask)[0]
    attached = 0

    for idx in noise_indices:
        v = vec_normed[idx]
        sims = centroids @ v
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= NOISE_ATTACH_THRESHOLD:
            new_label = int(cluster_labels[best_idx])
            labels[idx] = new_label
            meta = summary.clusters.get(str(new_label))
            if meta:
                meta.size += 1
            attached += 1

    new_noise = int(np.sum(labels == -1))
    summary.num_noise = new_noise

    print(f"   → attached {attached} points; noise left: {new_noise}")
    return labels, summary


# ----------------- OUTLIER RECLUSTER MERGE ----------------- #

def extend_summary_with_noise_clusters(
    prev: ClusteringSummary,
    noise_ids: List[str],
    noise_vectors: np.ndarray,
    noise_texts: List[str],
    noise_labels: np.ndarray,
) -> Tuple[ClusteringSummary, np.ndarray]:
    """
    Given previous summary + HDBSCAN labels over noise subset,
    create new global cluster labels and update summary.
    """
    print("📊 Extending summary with new topic clusters from noise subset...")

    if len(noise_ids) == 0:
        print("🛑 No noise points to extend.")
        return prev, np.array([], dtype=np.int32)

    unique_local = sorted({int(l) for l in noise_labels if l != -1})
    if not unique_local:
        print("ℹ️ HDBSCAN on noise produced only noise again. No new clusters.")
        new_global_labels = np.full_like(noise_labels, -1, dtype=np.int32)
        new_summary = ClusteringSummary(
            num_points=prev.num_points,
            num_clusters=prev.num_clusters,
            num_noise=int(np.sum(new_global_labels == -1)),
            clusters=prev.clusters,
        )
        return new_summary, new_global_labels

    existing_labels = [int(k) for k in prev.clusters.keys()] if prev.clusters else []
    start_label = (max(existing_labels) + 1) if existing_labels else 0

    local_to_global: Dict[int, int] = {}
    for offset_idx, local_label in enumerate(unique_local):
        local_to_global[local_label] = start_label + offset_idx

    new_clusters: Dict[str, ClusterMeta] = dict(prev.clusters) if prev.clusters else {}

    for local_label in unique_local:
        mask = (noise_labels == local_label)
        idxs = np.where(mask)[0]
        cluster_ids = [noise_ids[i] for i in idxs]
        cluster_vecs = noise_vectors[mask]
        cluster_texts = [noise_texts[i] for i in idxs]

        centroid, medoid_id = compute_centroid_and_medoid(cluster_vecs, cluster_ids)
        category = infer_cluster_category(cluster_texts)

        global_label = local_to_global[local_label]
        cm = ClusterMeta(
            label=global_label,
            size=len(cluster_ids),
            centroid=centroid,
            medoid_id=medoid_id,
            category=category,
        )
        new_clusters[str(global_label)] = cm
        print(
            f"   • new cluster {global_label:>3}: size={cm.size:>4}, "
            f"category={cm.category:<20}, medoid_id={cm.medoid_id[:8]}..."
        )

    global_labels = []
    for lbl in noise_labels:
        if lbl == -1:
            global_labels.append(-1)
        else:
            global_labels.append(local_to_global[int(lbl)])
    global_labels = np.array(global_labels, dtype=np.int32)

    still_noise = int(np.sum(global_labels == -1))
    new_num_clusters = prev.num_clusters + len(unique_local)

    new_summary = ClusteringSummary(
        num_points=prev.num_points,
        num_clusters=new_num_clusters,
        num_noise=still_noise,
        clusters=new_clusters,
    )

    print(f"✅ Extended summary → total clusters={new_num_clusters}, noise={still_noise}")
    return new_summary, global_labels


# ----------------- MAIN ----------------- #

def main() -> None:
    print(
        f"\n🌌 Topic cluster maintenance for '{COLLECTION_NAME}' "
        f"on {QDRANT_HOST}:{QDRANT_PORT}"
    )
    t0 = time.time()

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    existing_summary = load_existing_summary()

    # MODE 1: INITIAL FULL CLUSTERING
    if existing_summary is None:
        print("🚀 No clusters.json found → running INITIAL full topic clustering...")
        ids, vectors, topic_texts = fetch_all_points(client)
        if len(ids) == 0:
            return

        labels = run_hdbscan(vectors)
        summary = build_full_summary(ids, vectors, topic_texts, labels)

        # try to attach noise to centroids
        labels, summary = attach_noise_to_centroids(
            ids, vectors, topic_texts, labels, summary
        )

        save_summary(summary)
        write_labels_to_qdrant(client, ids, labels, summary)

        dur = time.time() - t0
        dur_min = dur / 60
        print(f"\n🎉 Initial topic clustering complete in {dur:.1f}s ({dur_min:.2f} min)")
        print(f"   points:   {summary.num_points}")
        print(f"   clusters: {summary.num_clusters}")
        print(f"   noise:    {summary.num_noise}")
        print("🌠 Done.\n")
        return

    # MODE 2: OUTLIER-ONLY RE-CLUSTERING
    print("🔁 clusters.json found → topic-clustering NOISE points only...")
    noise_ids, noise_vectors, noise_texts = fetch_noise_points(client)
    if len(noise_ids) == 0:
        print("✅ No unassigned/noise points. Nothing to do.")
        return

    noise_labels_local = run_hdbscan(noise_vectors)
    new_summary, noise_labels_global = extend_summary_with_noise_clusters(
        existing_summary,
        noise_ids,
        noise_vectors,
        noise_texts,
        noise_labels_local,
    )

    # again, try to attach any remaining noise in this subset
    noise_labels_global, new_summary = attach_noise_to_centroids(
        noise_ids, noise_vectors, noise_texts, noise_labels_global, new_summary
    )

    write_labels_to_qdrant(client, noise_ids, noise_labels_global, new_summary)
    save_summary(new_summary)

    dur = time.time() - t0
    dur_min = dur / 60
    print(f"\n🎉 Noise topic-clustering complete in {dur:.1f}s ({dur_min:.2f} min)")
    print(f"   points:   {new_summary.num_points}")
    print(f"   clusters: {new_summary.num_clusters}")
    print(f"   noise:    {new_summary.num_noise}")
    print("🌠 Done.\n")


if __name__ == "__main__":
    main()