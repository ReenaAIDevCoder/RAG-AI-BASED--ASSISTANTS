from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

class LLMPipeline:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="gemma2-9b-it"
        )

    def run(self, query):
        prompt = f"""
Answer the following question clearly:

{query}
"""

        response = self.llm.invoke(prompt)
        return response.content