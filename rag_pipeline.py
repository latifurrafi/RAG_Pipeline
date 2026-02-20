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
        """
        Handle very specific structured queries that require full CSV access.
        Only triggers for explicit requests to avoid returning entire datasets.
        """
        query_lower = query.lower().strip()
        
        # Only trigger for very explicit requests like "list all cse students" or "show me all cse"
        # Not for questions like "who are cse students?" which should use RAG
        explicit_list_patterns = [
            "list all cse",
            "show all cse",
            "give me all cse",
            "display all cse",
            "all cse students",
            "all cse data"
        ]
        
        if "cse" in query_lower and any(pattern in query_lower for pattern in explicit_list_patterns):
            cse_students = self.df[
                self.df["Department"].str.contains("CSE", case=False, na=False)
            ]
            
            # Limit output to prevent overwhelming responses
            if len(cse_students) > 50:
                return f"Found {len(cse_students)} CSE students. Showing first 50:\n\n" + cse_students.head(50).to_string()
            
            return cse_students.to_string()

        return None

    def ask(self, query):

        structured_result = self.structured_query(query)

        if structured_result:
            return structured_result

        retrieved_docs = self.retriever.retrieve(query)

        if not retrieved_docs:
            return "I couldn't find any relevant information matching your query. Please try rephrasing your question or using different keywords."

        # Limit context size to prevent overwhelming the LLM
        MAX_CONTEXT_DOCS = 20  # Maximum documents to include in context
        total_found = len(retrieved_docs)
        if len(retrieved_docs) > MAX_CONTEXT_DOCS:
            retrieved_docs = retrieved_docs[:MAX_CONTEXT_DOCS]

        # Format context with clear separators
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # Add note if we truncated results
        if total_found > MAX_CONTEXT_DOCS:
            context += f"\n\n[Note: Found {total_found} relevant results, showing top {MAX_CONTEXT_DOCS} most relevant]"

        answer = self.llm.generate(context, query)

        return answer