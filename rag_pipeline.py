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

    def _detect_requested_columns(self, query):
        """
        Detect which columns the user wants from the query.
        Returns a list of column names to include, or None for all columns.
        """
        query_lower = query.lower()
        
        # Common column name mappings - maps keywords to possible column name patterns
        column_keywords = {
            'id': ['id', 'student id', 'student_id', 'roll', 'roll number', 'rollno', 'studentid'],
            'name': ['name', 'student name', 'student_name', 'full name', 'fullname', 'studentname'],
            'email': ['email', 'e-mail', 'mail', 'email address'],
            'department': ['department', 'dept', 'branch'],
            'phone': ['phone', 'mobile', 'contact', 'phone number', 'phonenumber'],
            'address': ['address', 'location'],
            'cgpa': ['cgpa', 'gpa', 'grade', 'cgpa'],
            'year': ['year', 'batch', 'semester', 'sem']
        }
        
        requested_cols = []
        
        # Check for explicit column mentions in query
        for keyword, aliases in column_keywords.items():
            for alias in aliases:
                if alias in query_lower:
                    # Find matching column in dataframe (case-insensitive, partial match)
                    for col in self.df.columns:
                        col_lower = col.lower()
                        alias_normalized = alias.replace(' ', '_').replace('-', '_')
                        
                        # Check if column name contains the alias or vice versa
                        if (alias_normalized in col_lower or 
                            col_lower in alias_normalized or
                            alias in col_lower or
                            col_lower.replace('_', ' ').replace('-', ' ') == alias):
                            if col not in requested_cols:
                                requested_cols.append(col)
                                break
        
        # If specific columns detected, return them; otherwise return None (all columns)
        return requested_cols if requested_cols else None
    
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
            "all cse data",
            "list cse",
            "show cse",
            "cse list",
            "cse students list"
        ]
        
        if "cse" in query_lower and any(pattern in query_lower for pattern in explicit_list_patterns):
            cse_students = self.df[
                self.df["Department"].str.contains("CSE", case=False, na=False)
            ]
            
            # Detect which columns user wants
            requested_cols = self._detect_requested_columns(query)
            
            # If specific columns requested, use them; otherwise use all
            if requested_cols:
                # Ensure requested columns exist in the dataframe
                valid_cols = [col for col in requested_cols if col in cse_students.columns]
                
                # If no exact matches, try fuzzy matching
                if not valid_cols:
                    query_lower = query.lower()
                    if 'id' in query_lower:
                        id_cols = [col for col in cse_students.columns if 'id' in col.lower()]
                        if id_cols:
                            valid_cols.extend(id_cols[:1])  # Take first match
                    if 'name' in query_lower:
                        name_cols = [col for col in cse_students.columns if 'name' in col.lower()]
                        if name_cols:
                            valid_cols.extend(name_cols[:1])  # Take first match
                
                # Apply column filtering if we found valid columns
                if valid_cols:
                    cse_students = cse_students[valid_cols]
            
            # Limit output to prevent overwhelming responses
            if len(cse_students) > 50:
                return f"Found {len(cse_students)} CSE students. Showing first 50:\n\n" + cse_students.head(50).to_string(index=False)
            
            return cse_students.to_string(index=False)

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