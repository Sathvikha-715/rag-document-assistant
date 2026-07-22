print("Program started")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/ai pdf.pdf")
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")
print("\nFirst page preview:\n")
print(pages[0].page_content[:500])