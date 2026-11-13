import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompts import RAG_PROMPT

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file as:\n"
        "GROQ_API_KEY=your_api_key_here"
    )

# Model name lives in .env so a retired model never breaks the code again
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(model=MODEL_NAME, temperature=0)

SUMMARY_WORDS = (
    "summarize", "summarise", "summary", "overview",
    "whole document", "entire document", "main points", "key points",
)
LIST_WORDS = ("list", "tools", "all the", "name the", "which are")


def is_list_request(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in LIST_WORDS)
REFUSAL_PHRASES = (
    "couldn't find", "could not find", "not in the document",
    "no relevant information",
)


def is_summary_request(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in SUMMARY_WORDS)


def ask_question(vectorstore, question, k=8):
    """
    Ask a question against the vector store and return
    a grounded answer with source documents.
    """
    if not question or not question.strip():
        return "Please enter a valid question.", []

    try:
        if is_summary_request(question):
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 20, "fetch_k": 40},
            )
        elif is_list_request(question):
            # List questions are spread across many chunks, so fetch more
            retriever = vectorstore.as_retriever(search_kwargs={"k": 16})
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        docs = retriever.invoke(question)

    except Exception as e:
        return f"Error retrieving relevant content: {e}", []

    if not docs:
        return (
            "I couldn't find relevant information in the document "
            "to answer that question. Try rephrasing it.",
            [],
        )

    # Keep chunks in page order so summaries read in sequence
    docs = sorted(docs, key=lambda d: d.metadata.get("page", 0))

    context = "\n\n".join(
        doc.page_content.strip()
        for doc in docs
        if doc.page_content.strip()
    )

    if not context:
        return "The retrieved sections contained no readable text.", []

    try:
        prompt = RAG_PROMPT.invoke({"context": context, "question": question})
        response = llm.invoke(prompt)
    except Exception as e:
        return f"Error generating answer: {e}", docs

    answer = response.content

    # If the model refused, don't show misleading page sources
    if any(phrase in answer.lower() for phrase in REFUSAL_PHRASES):
        return answer, []

    return answer, docs