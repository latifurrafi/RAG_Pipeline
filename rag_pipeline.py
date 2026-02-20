# rag_pipeline.py

from csv_loader import load_csv_documents
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from ollama_client import OllamaClient
import pandas as pd
from config import DATA_PATH


class RAGPipeline:

    def __init__(self):

        self.df = pd.read_csv(DATA_PATH)

        print("Loading documents...")
        self.documents = load_csv_documents()

        print("Initializing embedder...")
        self.embedder = Embedder()

        print("Creating embeddings...")
        embeddings = self.embedder.embed_documents(self.documents)

        dimension = embeddings.shape[1]

        print("Building vector store...")
        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(
            self.documents,
            embeddings
        )

        self.retriever = Retriever(
            self.vector_store,
            self.embedder
        )

        self.llm = OllamaClient()

        print("RAG system ready")

    def structured_query(self, query):

        if "cse" in query.lower():

            cse_students = self.df[
                self.df["Department"].str.contains("CSE", case=False, na=False)
            ]

            return cse_students.to_string()

        return None

    def ask(self, query):

        structured_result = self.structured_query(query)

        if structured_result:
            return structured_result

        retrieved_docs = self.retriever.retrieve(query)

        context = "\n\n".join(retrieved_docs)

        answer = self.llm.generate(context, query)

        return answer