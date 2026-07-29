import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY not found in .env file.")
    print("Create a .env file in the project root with:")
    print("GROQ_API_KEY=your_api_key_here")
    sys.exit(1)

# -----------------------------
# Initialize the LLM
# -----------------------------
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3
    )
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    sys.exit(1)

# -----------------------------
# Test the connection
# -----------------------------
test_question = "What is Artificial Intelligence?"

try:
    print(f"Sending test question: {test_question}\n")
    response = llm.invoke(test_question)

    print("Response received successfully:\n")
    print(response.content)

    print(f"\n--- Metadata ---")
    print(f"Model: {llm.model_name}")
    if hasattr(response, "response_metadata"):
        token_usage = response.response_metadata.get("token_usage", {})
        print(f"Tokens used: {token_usage.get('total_tokens', 'N/A')}")

except Exception as e:
    print(f"Error calling Groq API: {e}")
    print("Check that your API key is valid and you have remaining quota.")
    sys.exit(1)