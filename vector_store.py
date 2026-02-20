# vector_store.py

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        # Use InnerProduct (cosine similarity) instead of L2 distance
        # This works better for normalized embeddings
        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add_documents(self, documents, embeddings):

        self.documents.extend(documents)

        # Normalize embeddings for cosine similarity
        embeddings = np.array(embeddings).astype("float32").copy()
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding, top_k, similarity_threshold=0.5):

        # Normalize query embedding (make a copy to avoid modifying original)
        query_embedding = query_embedding.astype("float32").copy()
        faiss.normalize_L2(query_embedding)

        # Search returns similarities (higher is better for InnerProduct)
        # InnerProduct returns cosine similarity when vectors are normalized
        similarities, indices = self.index.search(query_embedding, top_k)

        # Filter by similarity threshold and return results with scores
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= similarity_threshold:
                results.append({
                    'document': self.documents[idx],
                    'similarity': float(sim)
                })

        return results