import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self):
        self.store = FaissVectorStore()
        self.store.load()

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="gemma2-9b-it"
        )

    def search_and_summarize(self, query):
        results = self.store.query(query)

        context = "\n\n".join([r["text"] for r in results])

        prompt = f"""
Answer using context:

{context}

Question: {query}
"""

        response = self.llm.invoke(prompt)
        return response.content