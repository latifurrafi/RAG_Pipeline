# retriever.py

from config import TOP_K, SIMILARITY_THRESHOLD


class Retriever:

    def __init__(self, vector_store, embedder):

        self.vector_store = vector_store
        self.embedder = embedder

    def get_top_k(self, query):

        query_lower = query.lower()

        # For list queries, use more documents but cap at reasonable limit
        if any(word in query_lower for word in ["list", "all", "show", "give me all"]):
            return min(TOP_K * 2, 100)  # Cap at 100 instead of 200

        if any(word in query_lower for word in ["how many", "count", "total"]):
            return min(TOP_K * 2, 100)

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