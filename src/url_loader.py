from langchain_community.document_loaders import WebBaseLoader

def load_urls(urls):
    loader = WebBaseLoader(urls)
    return loader.load()


# Use like this
docs = load_urls([
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://opencv.org/about/"
])