# project/scripts/scraper_to_mongo.py
import os
import sys
import json
import time
import hashlib
from urllib.parse import urljoin, urlparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, InsertOne, errors
from tqdm import tqdm
from dateutil import parser as dateparser

# -------- CONFIG ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("NEWS_DB", "newsdb")
COLL_NAME = os.getenv("ART_COLL", "articles")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
QUEUE_FILE = os.path.join(DATA_DIR, "queue.jsonl")
SCRAPED_FILE = os.path.join(DATA_DIR, "scraped_urls.jsonl")

HEADERS = {"User-Agent": "Mozilla/5.0 (NewsScraper/1.0)"}
REQUEST_TIMEOUT = 10
BATCH_SIZE = 100    # bulk insert batch size
BAD_EXT = [".mp4", "/video", "/videos"]

# how many images candidate urls to store per article
MAX_IMAGE_CANDIDATES = 8

os.makedirs(DATA_DIR, exist_ok=True)

# -------- Mongo ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLL_NAME]

# ensure unique index on url
try:
    col.create_index("url", unique=True)
except Exception:
    pass

# -------- helpers ----------
def is_bad_url(u: str) -> bool:
    if not u: 
        return True
    low = u.lower()
    return any(ext in low for ext in BAD_EXT)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_url(base: str, src: str) -> str:
    if not src:
        return None
    src = src.strip()
    if src.startswith("data:") or src.startswith("javascript:"):
        return None
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    return urljoin(base, src)

def extract_candidate_images(soup: BeautifulSoup, page_url: str) -> List[str]:
    candidates = []

    # 1) og:image / twitter:image / meta image
    meta_keys = [("property", "og:image"), ("name", "twitter:image"), ("name", "image")]
    for attr, key in meta_keys:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            u = normalize_url(page_url, tag["content"])
            if u and u not in candidates:
                candidates.append(u)

    # 2) ld+json "image"
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
            # supports multiple shapes
            if isinstance(data, dict):
                img = data.get("image")
                if isinstance(img, str):
                    u = normalize_url(page_url, img)
                    if u and u not in candidates: candidates.append(u)
                elif isinstance(img, list):
                    for it in img:
                        u = normalize_url(page_url, it)
                        if u and u not in candidates: candidates.append(u)
            elif isinstance(data, list):
                for node in data:
                    if isinstance(node, dict) and node.get("image"):
                        img = node.get("image")
                        if isinstance(img, str):
                            u = normalize_url(page_url, img)
                            if u and u not in candidates: candidates.append(u)
        except Exception:
            continue

    # 3) article/main containers images
    containers = soup.select("article, main, .article, .post, .entry-content, .story-body") or [soup]
    for c in containers:
        for img in c.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            u = normalize_url(page_url, src)
            if u and u not in candidates:
                candidates.append(u)
            if len(candidates) >= MAX_IMAGE_CANDIDATES:
                break
        if len(candidates) >= MAX_IMAGE_CANDIDATES:
            break

    # 4) fallback: any <img> on page
    if not candidates:
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            u = normalize_url(page_url, src)
            if u and u not in candidates:
                candidates.append(u)
            if len(candidates) >= MAX_IMAGE_CANDIDATES:
                break

    return candidates

def clean_text_from_soup(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    cleaned = [p for p in paras if len(p.split()) > 5]
    if not cleaned:
        # fallback to larger text blocks
        text = soup.get_text(" ", strip=True)
        return text.strip() if text else ""
    return "\n\n".join(cleaned)

def parse_published(entry: dict) -> str:
    # take RSS published or try to parse
    published = entry.get("published") or entry.get("pubDate") or entry.get("updated") or ""
    try:
        if published:
            dt = dateparser.parse(published)
            return dt.isoformat()
    except Exception:
        pass
    return published

# -------- main scrape function ----------
def scrape_queue_to_mongo(limit=None):
    if not os.path.exists(QUEUE_FILE):
        print("no queue file:", QUEUE_FILE)
        return

    # read queue lines
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if limit:
        lines = lines[:limit]

    ops = []
    saved = 0
    skipped = 0

    for line in tqdm(lines, desc="queue"):
        try:
            item = json.loads(line)
        except Exception:
            skipped += 1
            continue

        url = item.get("link") or item.get("url")
        if not url or is_bad_url(url):
            skipped += 1
            continue

        # quick check: if url already in DB skip
        if col.find_one({"url": url}, {"_id": 1}):
            skipped += 1
            continue

        # fetch html
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            html = r.text
        except Exception as e:
            # network fail — skip and leave for next poll / manual retry
            skipped += 1
            continue

        soup = BeautifulSoup(html, "html.parser")

        title = (soup.title.string or "").strip() if soup.title else (item.get("title") or "")

        text = clean_text_from_soup(soup)
        if not text or len(text.split()) < 40:
            # too short or dynamic page; you could fallback to playwright here
            skipped += 1
            continue

        img_candidates = extract_candidate_images(soup, url)
        # build images array of subdocs
        images = []
        for idx, u in enumerate(img_candidates):
            images.append({
                "source_url": u,
                "priority": idx+1,
                "status": "pending",   # pending | downloaded | failed
                "attempts": 0,
                "local_path": None,
                "s3_url": None,
                "bytes": None
            })

        doc = {
            "url": url,
            "feed": item.get("feed"),
            "title": title or item.get("title"),
            "text": text,
            "word_count": len(text.split()),
            "published": parse_published(item),
            "domain": urlparse(url).netloc,
            "images": images,
            "meta": {},   # reserved for future og/twitter meta
            "scraped_at": datetime.utcnow()
        }

        # queue for bulk insert
        ops.append(InsertOne(doc))

        # flush batch
        if len(ops) >= BATCH_SIZE:
            try:
                col.bulk_write(ops, ordered=False)
            except errors.BulkWriteError as bwe:
                # duplicates might throw but we continue
                pass
            ops = []

        # append to backup scraped_urls.jsonl immediately (best-effort)
        try:
            with open(SCRAPED_FILE, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({"url": url}) + "\n")
        except Exception:
            pass

        saved += 1

    # final flush
    if ops:
        try:
            col.bulk_write(ops, ordered=False)
        except errors.BulkWriteError:
            pass

    # clear queue after a successful run
    open(QUEUE_FILE, "w").close()

    print(f"\nDONE — saved: {saved}, skipped: {skipped}")

if __name__ == "__main__":
    scrape_queue_to_mongo()
