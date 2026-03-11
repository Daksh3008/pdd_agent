from pinecone import Pinecone, ServerlessSpec


class VectorStore:
    """Thin wrapper around a Pinecone index."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self._pc = Pinecone(api_key=api_key)

        if not self._pc.has_index(index_name):
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self._index = self._pc.Index(index_name)

    def upsert(self, vectors: list[tuple]) -> None:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[i : i + batch_size])

    def query(self, vector: list[float], top_k: int = 5) -> dict:
        return self._index.query(
            vector=vector, top_k=top_k, include_metadata=True
        )

    def stats(self) -> dict:
        return self._index.describe_index_stats()
