from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_vectorstore(chunks):
    """
    Create a Chroma vector database from document chunks.
    """

    if not chunks:
        raise ValueError(
            "No readable text was found in the PDF. "
            "Please upload a text-based PDF."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore


def load_vectorstore():
    """
    Load an existing Chroma database.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )