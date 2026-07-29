import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompts import RAG_PROMPT
from pydantic import SecretStr
# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file as:\\n"
        "GROQ_API_KEY=your_api_key_here"
    )

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(api_key),
    temperature=0.3
)


def ask_question(vectorstore, question, k=4):
    """
    Ask a question against the vector store and return
    a grounded answer with source documents.
    """

    if not question or not question.strip():
        return "Please enter a valid question.", []

    try:
        # New LangChain retrieval API
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        docs = retriever.invoke(question)

    except Exception as e:
        return f"Error retrieving relevant content: {e}", []

    if not docs:
        return (
            "I couldn't find relevant information in the document "
            "to answer that question. Try rephrasing it.",
            []
        )

    # Remove empty chunks
    context = "\\n\\n".join(
        doc.page_content.strip()
        for doc in docs
        if doc.page_content.strip()
    )

    if not context:
        return (
            "The retrieved sections contained no readable text.",
            docs
        )

    try:
        prompt = RAG_PROMPT.invoke({
            "context": context,
            "question": question
        })

        response = llm.invoke(prompt)

    except Exception as e:
        return f"Error generating answer: {e}", docs

    return response.content, docs