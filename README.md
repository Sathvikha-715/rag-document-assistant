# DocMind AI – RAG-Based Document Assistant

DocMind AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their contents. The application retrieves the most relevant document sections using semantic search and generates grounded responses with source citations using Groq's Llama 3 model.

## Live Demo

https://rag-document-assistant-z84rv6u7qzbug4dyxfmvjy.streamlit.app/

## GitHub Repository

https://github.com/Sathvikha-715/rag-document-assistant

---

## Features

- Upload and process PDF documents
- Semantic search using vector embeddings
- Context-aware question answering
- Source page citations
- Fast retrieval with ChromaDB
- Interactive Streamlit interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq (Llama 3)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- ChromaDB
- PyPDFLoader

---

## How It Works

1. Upload a PDF document.
2. Extract and split the text into chunks.
3. Generate embeddings and store them in ChromaDB.
4. Retrieve relevant chunks based on the user's query.
5. Generate a grounded response using Groq's Llama 3.
6. Display the answer along with source page citations.

---

## Installation

```bash
git clone https://github.com/Sathvikha-715/rag-document-assistant.git
cd rag-document-assistant

python -m venv venv

# Windows
venv\Scripts\activate

# Git Bash
source venv/Scripts/activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
rag-document-assistant/
│
├── app.py
├── src/
├── docs/
├── chroma_db/
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Multiple PDF support
- Conversation memory
- Docker containerization
- OCR support for scanned PDFs

---

## Author

**Sathvikha Reddy**

B.Tech – Artificial Intelligence and Data Science

GitHub: https://github.com/Sathvikha-715