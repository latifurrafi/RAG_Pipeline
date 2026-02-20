# ollama_client.py

import requests
from config import OLLAMA_URL, OLLAMA_MODEL


class OllamaClient:

    def generate(self, context, query):

        if not context or not context.strip():
            return "I couldn't find relevant information in the database to answer your question. Please try rephrasing your query."

        # Determine if this is a list/summary query
        is_list_query = any(word in query.lower() for word in ["list", "all", "show", "give me", "what are"])
        
        if is_list_query and len(context.split("\n\n")) > 10:
            instruction = """- Provide a SUMMARY or OVERVIEW, not a complete list
- Highlight key patterns, trends, or notable items
- If listing items, limit to the most important/relevant ones (max 10-15 items)
- Use bullet points or numbered lists for clarity"""
        else:
            instruction = """- Be concise and accurate
- If asked about specific data (names, numbers, etc.), extract them directly from the context
- Format your answer clearly"""

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question using ONLY the information provided in the context below
- If the context doesn't contain enough information to answer the question, say so clearly
{instruction}
- Do NOT repeat the entire context - synthesize and summarize the relevant information

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json().get("response", "Error: No response from model")
            return result
        except requests.exceptions.RequestException as e:
            return f"Error connecting to LLM: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"