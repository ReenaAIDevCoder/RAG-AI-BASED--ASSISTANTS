from langchain_core.documents import Document

def load_text(text):
    return [Document(page_content=text)]