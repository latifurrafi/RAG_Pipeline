# embedder.py

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, documents):

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, query):

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        return embedding