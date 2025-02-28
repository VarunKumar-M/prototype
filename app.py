import os
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from retriever import TXTDataRetriever
from langchain_google_genai import GoogleGenerativeAI

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file.")

# Initialize Gemini AI
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Initialize Text Data Retriever (FAISS)
retriever = TXTDataRetriever()

# Streamlit UI Configuration
st.set_page_config(page_title="AgriGPT", page_icon="🌾", layout="wide")
st.title("🌾 AgriGPT - Your Expert Agricultural Assistant")

# **Language Selection**
lang_options = {
    "Auto (Detect)": "auto",
    "English": "en", "Hindi (हिंदी)": "hi", "Kannada (ಕನ್ನಡ)": "kn",
    "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te", "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu", "Malayalam (മലയാളം)": "ml", "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Bengali (বাংলা)": "bn", "Urdu (اردو)": "ur"
}
selected_lang = st.selectbox("Choose your language:", list(lang_options.keys()))

# Memory Limited to Last 5 Conversations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        # Retrieve relevant hints from FAISS (used subtly)
        retrieved_texts = retriever.retrieve_relevant_text(query)
        abstract_hint = retrieved_texts if retrieved_texts else ""

        # Detect or Set Language
        if lang_options[selected_lang] == "auto":
            try:
                response_lang = detect(query)
            except:
                response_lang = "en"
        else:
            response_lang = lang_options[selected_lang]

        # Limit Memory to Last 5 Messages
        history = "\n".join(st.session_state.chat_history[-5:])

        # **Generate Response with Gemini in Full Control**
        prompt = f"""
        You are **AgriGPT**, a professional agricultural assistant.  
        Your goal is to provide **structured, intelligent, and precise answers**.  

        ### **Guidelines:**
        - Always maintain a **formal, expert, and structured tone**.  
        - Responses should be **logically sound and contextually relevant**.  
        - Use the retrieved FAISS data as a **subtle reference**, but never quote it explicitly.  
        - Ensure responses feel like a **real expert** is answering.  

        ### **Subtle Background Knowledge (Not Quoted Directly):**  
        {abstract_hint}

        ### **User Query (Language: {response_lang}):**  
        {query}
        """

        response = gemini.invoke(prompt)

        if response:
            st.markdown("### 🎓 AgriGPT Response:")
            st.markdown(response.strip())

            st.session_state.chat_history.append(f"User ({response_lang}): {query}\nAgriGPT ({response_lang}): {response.strip()}")
            st.session_state.chat_history = st.session_state.chat_history[-5:]

        else:
            st.error("Unable to generate a response at the moment.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Gemini 1.5 Flash & FAISS RAG")