from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Loading PDF...")

loader = PyPDFLoader("docs/ai pdf.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages.")

print("\nSplitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

print(f"Created {len(chunks)} chunks.")

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nEmbeddings stored successfully!")

print("\nTesting retrieval...")

query = "What is Artificial Intelligence?"

results = vectorstore.similarity_search(query, k=2)

print("\nTop Matching Results:\n")

for i, doc in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result {i}")
    print("=" * 60)
    print(doc.page_content[:400])
    print()