import json
from langchain_core.documents import Document

def load_jsonl(file_path):
    docs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text") or data.get("content") or str(data)
            docs.append(Document(page_content=text))

    return docs