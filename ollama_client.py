# ollama_client.py

import requests
from config import OLLAMA_URL, OLLAMA_MODEL


class OllamaClient:

    def generate(self, context, query):

        prompt = f"""
You are an assistant. Answer only using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()["response"]

        return result