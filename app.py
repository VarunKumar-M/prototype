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

# Streamlit UI Configuration
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")

# App Title
st.title("🌱 AgriGPT - Your Smart Agriculture Assistant")
st.write("Talk to me about farming, crops, or anything related to agriculture!")

# Store Chat History for Realistic Flow
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
query = st.text_input("Ask me anything:")

if st.button("Ask"):
    if query:
        # Retrieve relevant context
        retrieved_text = retriever.retrieve_relevant_text(query)
        context = retrieved_text if retrieved_text and len(retrieved_text.split()) > 5 else ""

        # Maintain Chat History for Conversational Flow
        history = "\n".join(st.session_state.chat_history[-15:])  # Stores last 15 exchanges

        # Construct Prompt for AI
        prompt = f"""
        You are a *friendly, knowledgeable agricultural assistant* who speaks *just like a real human*.
        - *Remember past conversations* so replies feel natural.
        - *Sound engaging*, as if talking to a friend.
        - *Use context* intelligently, never repeating information unnecessarily.

        **Conversation So Far:**  
        {history}

        **User's Question:**  
        {query}
        
        {context}

        **How You Should Answer:**
        - Speak *naturally, like a human* (not robotic).
        - If it's a follow-up, *connect it to past answers*.
        - Be *concise yet engaging* – *don't over-explain* unless asked.
        - If something is unclear, *ask the user for clarification*.
        - If no relevant info is found, *give a thoughtful, general answer*.

        Keep your tone *warm, friendly, and intelligent* – like a real expert in agriculture.
        """

        # Generate Response
        response = gemini.invoke(prompt)

        if response:
            st.write("### 🤖 AgriGPT says:")
            st.write(response.strip())

            # Save Conversation History
            st.session_state.chat_history.append(f"User: {query}\nAgriGPT: {response.strip()}")

        else:
            st.error("Oops! I couldn't generate a response. Try again.")

    else:
        st.warning("Type something first!")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Google Gemini 1.5 Flash & Local Agricultural Knowledge")
