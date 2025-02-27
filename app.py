import os
import streamlit as st
from dotenv import load_dotenv
from retriever import retriever  
from langchain_google_genai import GoogleGenerativeAI  
from langdetect import detect

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

# Initialize Gemini AI
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Streamlit UI Setup
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")
st.title("🌱 AgriGPT - Your Smart Agriculture Assistant")
st.write("Ask anything about farming, crops, or government schemes in your preferred language!")

# Language Selection
lang_options = {
    "Auto (Detect Language)": "auto", "English": "en", "Hindi (हिंदी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn", "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr", "Gujarati (ગુજરાતી)": "gu", "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa", "Bengali (বাংলা)": "bn", "Urdu (اردو)": "ur",
    "Spanish (Español)": "es", "French (Français)": "fr", "German (Deutsch)": "de",
    "Chinese (中文)": "zh", "Arabic (العربية)": "ar", "Portuguese (Português)": "pt",
    "Russian (Русский)": "ru",
}
selected_lang = st.selectbox("Choose your language:", list(lang_options.keys()))

# Maintain Last 5 Conversations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Query Input
query = st.text_input("Enter your query:")

if st.button("Ask") and query:
    retrieved_text = retriever.retrieve_relevant_text(query)
    context = retrieved_text if retrieved_text and len(retrieved_text.split()) > 5 else ""

    # Determine Response Language
    response_lang = detect(query) if lang_options[selected_lang] == "auto" else lang_options[selected_lang]

    # Keep Memory to Last 5 Messages
    history = "\n".join(st.session_state.chat_history[-5:])

    # Optimized Prompt
    prompt = f"""
    You are AgriGPT, a professional, structured, and intelligent agricultural assistant.
    - Provide **precise, accurate, and structured** responses.
    - Respond in **{response_lang}** with a natural, professional tone.
    - **Use context to enhance Gemini’s answer but let Gemini dominate**.
    - Keep responses **smart, direct, and without unnecessary details**.

    *Recent Conversation Context:*  
    {history}

    *User Query (Language: {response_lang}):*  
    {query}

    {context}

    *Response Guidelines:*
    - **Start with a clear, concise answer.**
    - **Use bullet points or steps for clarity when needed.**
    - **Avoid unnecessary explanations—be precise.**
    - **Highlight key takeaways with bold text.**

    Answer in **{response_lang}** with professionalism and clarity.
    """

    # Generate Response
    response = gemini.invoke(prompt)

    if response:
        st.markdown("### 🎓 AgriGPT Response:")
        st.markdown(response.strip())

        # Update Chat History (Limit to 5 Messages)
        st.session_state.chat_history.append(f"User ({response_lang}): {query}\nAgriGPT ({response_lang}): {response.strip()}")
        st.session_state.chat_history = st.session_state.chat_history[-5:]
    else:
        st.error("Unable to generate a response.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by AgriGPT")