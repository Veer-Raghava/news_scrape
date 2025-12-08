import json
import requests

NODE_URL = "http://localhost:5000/api/scrape/store"
JSONL = "project/data/articles.jsonl"

articles = []

with open(JSONL, "r", encoding="utf8") as f:
    for line in f:
        if line.strip():
            articles.append(json.loads(line))

print(f"Sending {len(articles)} articles to Node…")

res = requests.post(NODE_URL, json=articles)
print(res.status_code, res.text)
