import os
import streamlit as st
from dotenv import load_dotenv
from retriever import retriever  # Importing TXT retriever
from langchain_google_genai import GoogleGenerativeAI  # Gemini AI for final response

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

# Initialize Gemini 1.5 Flash AI
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Streamlit UI
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")

st.title("🌱 AgriGPT - Your Agriculture Chatbot")
st.write("Ask anything about agriculture, farming, or crop management.")

# User input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        retrieved_text = retriever.retrieve_relevant_text(query)
        
        # Construct Gemini AI prompt
        prompt = f"""
        You are an expert in agriculture. Answer the following question based on the query.
        
        **Reference Text (from documents):**  
        {retrieved_text}

        **User Question:** {query}

        **Instructions:**  
        - Provide a **concise** response by default.
        - Give **detailed explanations** only if the user explicitly asks for it (e.g., "Explain in detail").
        - If the reference text is insufficient, provide a general answer.
        """

        # Generate response using Gemini 1.5 Flash
        response = gemini.invoke(prompt)

        if response:
            st.write("**Answer:**")
            st.write(response)
        else:
            st.error("Failed to generate a response.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Google Gemini 1.5 Flash & Local Document Retrieval")
