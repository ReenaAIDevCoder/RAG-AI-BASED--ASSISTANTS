from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embeddings import EmbeddingPipeline


class VectorStore:
    def __init__(self):
        self.embedding = EmbeddingPipeline()

        # text splitter (RAM safe)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # 👈 reduce memory load
            chunk_overlap=50
        )

        self.db = None

    def create(self, documents):
        print("Creating vector store...")

        # split documents
        texts = []
        metadatas = []

        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            texts.extend(chunks)
            metadatas.extend([{"text": chunk} for chunk in chunks])

        # embeddings
        embeddings = self.embedding.embed_documents(texts)

        # store in chroma
        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding,
            metadatas=metadatas,
            persist_directory="db"
        )

        # self.db.persist()
        # print("Vector DB created ✅")

    def load(self):
        self.db = Chroma(
            persist_directory="db",
            embedding_function=self.embedding
        )

    def query(self, query, k=3):
        docs = self.db.similarity_search(query, k=k)

        return [
            {"metadata": {"text": doc.page_content}}
            for doc in docs
        ]