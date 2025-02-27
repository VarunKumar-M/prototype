import os
import streamlit as st
from dotenv import load_dotenv
from retriever import retriever  
from langchain_google_genai import GoogleGenerativeAI  

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
        # Construct Gemini AI prompt
        prompt = f"""
        You are a friendly and knowledgeable agricultural assistant. Respond in a natural, engaging, and conversational tone.
        
        *User Question:*  
        {query}
        
        *Guidelines for Response:*
        - Keep it conversational and engaging, like ChatGPT.
        - Answer naturally without over-explaining unless asked.
        - Be concise but helpful, avoiding robotic or overly formal tones.
        """

        # Generate response using Gemini 1.5 Flash
        response = gemini.invoke(prompt)

        if response:
            st.write(response)
        else:
            st.error("Failed to generate a response.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Google Gemini 1.5 Flash")
