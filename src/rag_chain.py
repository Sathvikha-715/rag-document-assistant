import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from src.prompts import RAG_PROMPT

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(prompt)

    return response.content, docs