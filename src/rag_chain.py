import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompts import RAG_PROMPT

# -----------------------------
# Load environment and validate
# -----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file as:\n"
        "GROQ_API_KEY=your_api_key_here"
    )

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.3
)


def ask_question(vectorstore, question, k=4):
    """
    Retrieves relevant chunks from the vectorstore and generates
    a grounded answer using the LLM.

    Args:
        vectorstore: The Chroma vectorstore to search.
        question: The user's question as a string.
        k: Number of top relevant chunks to retrieve.

    Returns:
        Tuple of (answer: str, docs: list) — the generated answer
        and the source documents used to ground it.
    """

    if not question or not question.strip():
        return "Please enter a valid question.", []

    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception as e:
        return f"Error retrieving relevant content: {e}", []

    if not docs:
        return (
            "I couldn't find relevant information in the document "
            "to answer that question. Try rephrasing it.",
            []
        )

    context = "\n\n".join(doc.page_content for doc in docs)

    try:
        prompt = RAG_PROMPT.invoke({
            "context": context,
            "question": question
        })
        response = llm.invoke(prompt)
    except Exception as e:
        return f"Error generating answer: {e}", docs

    return response.content, docs