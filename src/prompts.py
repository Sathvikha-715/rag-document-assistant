from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are an intelligent AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context,
reply:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
)