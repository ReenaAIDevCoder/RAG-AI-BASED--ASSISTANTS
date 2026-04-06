from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self):
        # lightweight model (RAM friendly)
        self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, query):
        return self.model.encode([query])[0]