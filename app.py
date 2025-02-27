import os
import streamlit as st
from dotenv import load_dotenv
from retriever import retriever  
from langchain_google_genai import GoogleGenerativeAI  
from langdetect import detect

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

# Initialize Gemini 1.5 Flash AI
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Streamlit UI Configuration
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")

st.title("🌱 AgriGPT - Logical & Smart Agriculture Assistant")
st.write("Ask about farming, crops, or agriculture, and get well-reasoned, structured, and logical responses.")

# **Language Selection**
lang_options = {
    "Auto (Detect Language)": "auto",
    "English": "en", "Hindi (हिंदी)": "hi", "Kannada (ಕನ್ನಡ)": "kn",
    "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te", "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu", "Malayalam (മലയാളം)": "ml", "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Bengali (বাংলা)": "bn", "Urdu (اردو)": "ur", "Spanish (Español)": "es",
    "French (Français)": "fr", "German (Deutsch)": "de", "Chinese (中文)": "zh",
    "Arabic (العربية)": "ar", "Portuguese (Português)": "pt", "Russian (Русский)": "ru"
}
selected_lang = st.selectbox("Choose your language:", list(lang_options.keys()))

# **Memory Limited to Last 5 Conversations**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **User Input**
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        retrieved_text = retriever.retrieve_relevant_text(query)

        # **Use context only if it's relevant**
        context = retrieved_text if retrieved_text and len(retrieved_text.split()) > 5 else ""

        # **Detect or Set Language**
        if lang_options[selected_lang] == "auto":
            try:
                response_lang = detect(query)
            except:
                response_lang = "en"  # Default to English if detection fails
        else:
            response_lang = lang_options[selected_lang]

        # **Limit Memory to Last 5 Messages**
        history = "\n".join(st.session_state.chat_history[-5:])

        # **Logical Reasoning & Thoughtful Processing**
        prompt = f"""
        You are **AgriGPT**, an intelligent, professional, and logical agricultural assistant.  
        Your goal is to provide well-reasoned, structured, and logically sound answers.  
        You **do not generate random answers**—instead, you **analyze the question**, apply reasoning, and give a **precise yet justified** response.

        ### **How You Should Respond:**
        - **Think before answering.**  
        - **Break down complex topics** step by step.  
        - **Justify why a particular answer is correct.**  
        - **Do not assert things blindly**—explain logically.  
        - **Acknowledge uncertainty when needed**, instead of making up responses.  

        ### **Response Structuring Rules:**
        - Start with **a direct, well-reasoned answer**.  
        - Use **logical steps, structured points, or headings** to explain.  
        - **If relevant, guide the user to think critically.**  
        - **Do not include unnecessary details**—stay on point.  

        ### **Previous Conversations Context:**
        {history}

        ### **User Query (Language: {response_lang}):**  
        {query}

        {context}

        ### **Response Output Guidelines:**
        - **Direct, logically justified answer**.  
        - **Structured breakdown if needed**.  
        - **Intelligent guidance** rather than just providing an answer.  
        """

        # **Generate Response**
        response = gemini.invoke(prompt)

        if response:
            st.markdown("### 🎓 AgriGPT Response:")
            st.markdown(response.strip())

            # **Update Chat History (Limit to Last 5 Messages)**
            st.session_state.chat_history.append(f"User ({response_lang}): {query}\nAgriGPT ({response_lang}): {response.strip()}")
            st.session_state.chat_history = st.session_state.chat_history[-5:]

        else:
            st.error("Unable to generate a response at the moment.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Gemini 1.5 Flash & Logical Agricultural Knowledge")