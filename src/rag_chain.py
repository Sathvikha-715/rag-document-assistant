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

# Larger output budget so reasoning models still have room to write the answer
llm = ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=4096)

SUMMARY_WORDS = (
    "summarize", "summarise", "summary", "overview",
    "whole document", "entire document", "main points", "key points",
)
LIST_WORDS = ("list", "tools", "all the", "name the", "which are")
REFUSAL_PHRASES = (
    "couldn't find", "could not find", "not in the document",
    "no relevant information",
)


def is_summary_request(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in SUMMARY_WORDS)


def is_list_request(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in LIST_WORDS)


def extract_text(response) -> str:
    """Safely get the answer text, whether content is a string or a list."""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    return (content or "").strip()


def build_context(docs) -> str:
    return "\n\n".join(
        doc.page_content.strip()
        for doc in docs
        if doc.page_content.strip()
    )


def generate(context: str, question: str) -> str:
    prompt = RAG_PROMPT.invoke({"context": context, "question": question})
    return extract_text(llm.invoke(prompt))


def ask_question(vectorstore, question, k=8):
    """
    Ask a question against the vector store and return
    a grounded answer with source documents.
    """
    if not question or not question.strip():
        return "Please enter a valid question.", []

    try:
        if is_summary_request(question):
            # Summaries need broad coverage: pick many diverse chunks (MMR)
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 20, "fetch_k": 40},
            )
        elif is_list_request(question):
            # List questions are spread across many chunks, so fetch more
            retriever = vectorstore.as_retriever(search_kwargs={"k": 16})
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        ranked_docs = retriever.invoke(question)

    except Exception as e:
        return f"Error retrieving relevant content: {e}", []

    if not ranked_docs:
        return (
            "I couldn't find relevant information in the document "
            "to answer that question. Try rephrasing it.",
            [],
        )

    # Keep chunks in page order so summaries read in sequence
    docs = sorted(ranked_docs, key=lambda d: d.metadata.get("page", 0))
    context = build_context(docs)

    if not context:
        return "The retrieved sections contained no readable text.", []

    try:
        answer = generate(context, question)

        # Retry once with fewer, most-relevant chunks if the model returned nothing
        if not answer:
            docs = ranked_docs[:8]
            answer = generate(build_context(docs), question)

        if not answer:
            return (
                "The model returned an empty response. Please try asking again.",
                [],
            )

    except Exception as e:
        return f"Error generating answer: {e}", docs

    # If the model refused, don't show misleading page sources
    if any(phrase in answer.lower() for phrase in REFUSAL_PHRASES):
        return answer, []

        # Cite only the most relevant chunks, not everything that was retrieved
    n_cite = 5 if (is_summary_request(question) or is_list_request(question)) else 3
    try:
        cited = vectorstore.similarity_search(answer, k=n_cite)
    except Exception:
        cited = ranked_docs[:n_cite]

    return answer, cited