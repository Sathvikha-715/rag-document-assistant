from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_vectorstore(chunks):

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

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vectorstore