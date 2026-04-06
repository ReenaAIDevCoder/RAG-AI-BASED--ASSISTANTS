print("STARTED...")

from src.pdf_loader import load_pdfs
from src.url_loader import load_urls
from src.text_loader import load_text
from src.json_loader import load_jsonl

from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline

if __name__ == "__main__":

    print("Select Data Source:")
    print("1. PDF")
    print("2. URL")
    print("3. TEXT")
    print("4. JSONL")

    choice = input("Enter choice: ")

    if choice == "1":
        docs = load_pdfs("data")

    elif choice == "2":
        urls = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://opencv.org/about/",
            "https://en.wikipedia.org/wiki/Computer_vision",
            "https://www.ibm.com/topics/mlops",
            "https://aws.amazon.com/what-is/mlops/",
            "https://www.geeksforgeeks.org/machine-learning/",
            "https://www.geeksforgeeks.org/artificial-intelligence/",
            "https://www.geeksforgeeks.org/computer-vision/"

        ]
        docs = load_urls(urls)

    elif choice == "3":
        docs = load_text("data/notes.txt")

    elif choice == "4":
        docs = load_jsonl("data/books.jsonl")

    else:
        print("Invalid choice")
        exit()

    store = VectorStore()
    store.create(docs)

    print("Vector DB ready ")

    rag = RAGPipeline()

    while True:
        query = input("\nAsk Question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = rag.run(query)
        print("\nAnswer:\n", answer)