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

# Streamlit UI
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")

st.title("🌱 AgriGPT - Your Multilingual Agriculture Expert")
st.write("Ask anything about agriculture, farming, or crop management in your preferred language!")

# **Language Selection**
lang_options = {
    "Auto (Detect Language)": "auto",
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Bengali (বাংলা)": "bn",
    "Urdu (اردو)": "ur",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Chinese (中文)": "zh",
    "Arabic (العربية)": "ar",
    "Portuguese (Português)": "pt",
    "Russian (Русский)": "ru",
}
selected_lang = st.selectbox("Choose your language:", list(lang_options.keys()))

# **Memory Limited to Last 5 Conversations**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **User Input**
query = st.text_input("Enter your query:")

if st.button("Ask"):
    if query:
        retrieved_text = retriever.retrieve_relevant_text(query)

        # **Abstract Context Use (No Explicit Mention of Dataset)**
        context = retrieved_text if retrieved_text and len(retrieved_text.split()) > 5 else ""

        # **Determine Response Language**
        if lang_options[selected_lang] == "auto":
            try:
                response_lang = detect(query)
            except:
                response_lang = "en"  # Default to English if detection fails
        else:
            response_lang = lang_options[selected_lang]

        # **Keep Memory to Last 5 Messages**
        history = "\n".join(st.session_state.chat_history[-5:])

        # **Multilingual Prompt with Structured Response Formatting**
        prompt = f"""
        You are a **multilingual agricultural expert** providing **well-structured, professional, and insightful responses**.
        - Offer **concise yet authoritative** guidance.
        - Respond in **{response_lang}**, maintaining a natural, conversational style.
        - Use **prior exchanges (last 5 messages) for coherence**.
        - Maintain a **fluent, structured, and professional tone**.

        *Recent Conversation for Context:*  
        {history}

        *User Query (Language: {response_lang}):*  
        {query}
        
        {context}

        *Guidelines for Response Formatting:*
        - **Use headings, bullet points, and clear outlines** for readability.
        - **Break down complex topics into steps or categories**.
        - **Highlight key takeaways with bold text**.
        - **Use structured insights** for credibility.

        Provide the response in **{response_lang}**, ensuring it is **structured, easy to read, and professional**.
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
st.caption("🤖 Powered by AgriGPT")