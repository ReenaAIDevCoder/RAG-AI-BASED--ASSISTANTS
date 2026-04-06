from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdfs(folder_path):
    docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())

    return docs