import feedparser
import json
import os
from datetime import datetime
 #python -m project.scripts.rss_ingest
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


from feeds.feeds_list import FEEDS
 
# BASE = project_root
BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")

os.makedirs(DATA, exist_ok=True)

QUEUE = os.path.join(DATA, "queue.jsonl")
KNOWN = os.path.join(DATA, "known_urls.jsonl")


def load_known():
    known = set()
    if os.path.exists(KNOWN):
        with open(KNOWN, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    known.add(json.loads(line)["url"])
                except:
                    pass
    return known


def add_known(url):
    with open(KNOWN, "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url}) + "\n")


def normalize(feed_name, entry):
    link = entry.get("link") or entry.get("id")
    return {
        "feed": feed_name,
        "link": link,
        "title": entry.get("title", "").strip(),
        "published": entry.get("published", ""),
        "discovered_at": datetime.utcnow().isoformat()
    }


def poll_feeds():
    known = load_known()
    new_count = 0
    checked = 0

    with open(QUEUE, "a", encoding="utf-8") as q:
        for name, url in FEEDS.items():
            feed = feedparser.parse(url)

            for entry in feed.entries:
                checked += 1

                item = normalize(name, entry)
                link = item["link"]

                if not link or link in known:
                    continue

                # NEW URL → add to queue + mark known
                q.write(json.dumps(item, ensure_ascii=False) + "\n")
                add_known(link)
                known.add(link)
                new_count += 1

                print(f"[{name}] NEW → {link}")

    print(f"\n✓ POLL DONE — checked: {checked}, new: {new_count}")


if __name__ == "__main__":
    poll_feeds()
