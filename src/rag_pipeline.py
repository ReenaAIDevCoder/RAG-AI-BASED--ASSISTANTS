from dotenv import load_dotenv
import os
import sys

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

from langchain_groq import ChatGroq
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self):
        # Initialize vector store
        self.store = VectorStore()
        self.store.load()

        # Debug: Check API key
        api_key = os.getenv("GROQ_API_KEY")
        print("DEBUG API KEY:", api_key)

        if not api_key:
            raise ValueError(" GROQ_API_KEY not found. Check your .env file!")

        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    def run(self, query):
        # Retrieve context
        results = self.store.query(query, k=3)

        context = "\n\n".join([r["metadata"]["text"] for r in results])

        # Prompt
        prompt = f"""
Answer the question based on context:

{context}

Question: {query}
"""

        # LLM response
        response = self.llm.invoke(prompt)

        return response.content