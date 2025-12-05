from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
collection = "news_articles"

offset = None
batch = 5000

print("🧹 Removing `category` from all payloads...")

while True:
    points, offset = client.scroll(
        collection_name=collection,
        limit=batch,
        offset=offset,
        with_payload=True,
    )
    if not points:
        break

    ids = [p.id for p in points]
    client.set_payload(
        collection_name=collection,
        payload={"category": None},
        points=ids,
    )

    print(f"  cleaned {len(ids)} points...")

    if offset is None:
        break

print("✅ Done. All categories removed.")
