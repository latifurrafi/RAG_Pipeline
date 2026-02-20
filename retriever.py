# retriever.py

class Retriever:

    def __init__(self, vector_store, embedder):

        self.vector_store = vector_store
        self.embedder = embedder

    def get_top_k(self, query):

        query = query.lower()

        if any(word in query for word in ["list", "all", "show", "give me all"]):
            return 200

        if any(word in query for word in ["how many", "count", "total"]):
            return 200

        return 5

    def retrieve(self, query):

        top_k = self.get_top_k(query)

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        return results