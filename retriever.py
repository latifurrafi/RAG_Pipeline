# retriever.py

from config import TOP_K, SIMILARITY_THRESHOLD


class Retriever:

    def __init__(self, vector_store, embedder):

        self.vector_store = vector_store
        self.embedder = embedder

    def get_top_k(self, query):

        query_lower = query.lower()

        if any(word in query_lower for word in ["list", "all", "show", "give me all"]):
            return min(200, TOP_K * 4)  # Use more for list queries but respect config

        if any(word in query_lower for word in ["how many", "count", "total"]):
            return min(200, TOP_K * 4)

        return TOP_K

    def retrieve(self, query, similarity_threshold=None):

        if similarity_threshold is None:
            similarity_threshold = SIMILARITY_THRESHOLD

        top_k = self.get_top_k(query)

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            top_k,
            similarity_threshold
        )

        # Return just the documents (backward compatible)
        return [r['document'] for r in results]