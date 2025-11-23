import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")

QUEUE = os.path.join(DATA, "queue.jsonl")
OUTPUT = os.path.join(DATA, "articles.jsonl")
SCRAPED = os.path.join(DATA, "scraped_urls.jsonl")

BAD_EXT = [".mp4", "/video", "/videos"]


def load_scraped():
    scraped = set()
    if os.path.exists(SCRAPED):
        with open(SCRAPED, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    scraped.add(json.loads(line)["url"])
                except:
                    pass
    return scraped


def add_scraped(url):
    with open(SCRAPED, "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url}) + "\n")


def is_bad(url):
    return any(ext in url.lower() for ext in BAD_EXT)


# ------------------------------------------
# CLEAN BOILERPLATE LINES
# ------------------------------------------
def clean_boilerplate(text):
    BAD_PHRASES = [
        "Posts from this topic will be added",
        "Posts from this author will be added",
        "A free daily digest",
        "Sign up for Verge Deals",
        "This is the title for the native ad",
        "If you buy something from a Verge link",
        "Vox Media may earn a commission",
        "Read our review",
        "Read our hands on"
    ]

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cleaned = []

    for line in lines:
        if any(p in line for p in BAD_PHRASES):
            continue
        if lines.count(line) > 1:
            if line not in cleaned:
                cleaned.append(line)
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def extract_text(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        html = r.text
    except:
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    cleaned = [p for p in paras if len(p.split()) > 5]

    if len(cleaned) < 3:
        return None

    text = "\n".join(cleaned)
    text = clean_boilerplate(text)  # <<< ONLY ADDITIONAL LINE
    return text


def scrape():
    print("\n🔥 SCRAPER STARTED\n")

    saved = 0
    skipped = 0
    scraped_urls = load_scraped()

    with open(QUEUE, "r", encoding="utf-8") as q, \
         open(OUTPUT, "a", encoding="utf-8") as out:

        for line in q:
            item = json.loads(line)

            url = item.get("link") or item.get("url")  # FIX FOR KEY ERROR

            if url in scraped_urls or is_bad(url):
                continue

            print(f"→ Scraping: {url}")
            text = extract_text(url)

            if not text or len(text.split()) < 80:
                print("   ❌ TOO SHORT — NOT SAVED")
                continue

            rec = {
                "url": url,
                "title": item.get("title"),
                "text": text,
                "word_count": len(text.split()),
                "feed": item.get("feed"),
                "published": item.get("published"),
                "scraped_at": time.time(),
                "domain": urlparse(url).netloc
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            add_scraped(url)          # << MOVED HERE
            scraped_urls.add(url)     # << AFTER SAVE
            saved += 1

            print(f"   ✔ SAVED ({rec['word_count']} words)")

    open(QUEUE, "w").close()
    print("\n🧹 queue cleared!")
    print(f"\n🎉 DONE — saved: {saved}, skipped: {skipped}")


if __name__ == "__main__":
    scrape()
