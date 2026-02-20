# vector_store.py

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatL2(dimension)

        self.documents = []

    def add_documents(self, documents, embeddings):

        self.documents.extend(documents)

        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, top_k):

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k
        )

        results = [self.documents[i] for i in indices[0]]

        return results