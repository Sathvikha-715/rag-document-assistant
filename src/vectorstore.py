import uuid
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

_embeddings = None


def get_embeddings():
    """Load the embedding model once and reuse it for every upload."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def create_vectorstore(chunks):
    """
    Build a FRESH in-memory vector store for each uploaded document.

    - No persist_directory, so nothing is reloaded from an old database.
    - A unique collection name, so chunks from different PDFs never mix.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"doc_{uuid.uuid4().hex}",
    )