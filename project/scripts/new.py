from qdrant_client import QdrantClient
client = QdrantClient("localhost", port=6333)

points, _ = client.scroll(
    collection_name="news_articles",
    limit=200,
    with_payload=True,
    with_vectors=False
)

for p in points:
    print(p.payload.get("url"))
