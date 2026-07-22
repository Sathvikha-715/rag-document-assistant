from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.rag_chain import ask_question

print("=" * 50)
print("📄 RAG Document Assistant")
print("=" * 50)

print("\nLoading PDF...")

documents = load_pdf("docs/ai pdf.pdf")

print(f"Loaded {len(documents)} pages.")

print("\nSplitting document into chunks...")

chunks = split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("\nCreating vector database...")

vectorstore = create_vectorstore(chunks)

print("✅ Vector database ready!")

print("\nYou can now ask questions about the PDF.")
print("Type 'exit' to quit.\n")

while True:

    question = input("Question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer, sources = ask_question(vectorstore, question)

    print("\n" + "=" * 50)
    print("Answer:\n")
    print(answer)

    print("\nSources Used:")

    pages = set()

    for doc in sources:
        page = doc.metadata.get("page", 0) + 1
        pages.add(page)

    print(", ".join(f"Page {p}" for p in sorted(pages)))

    print("=" * 50 + "\n")