

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

        # Check retrieved text quality
        if retrieved_text and len(retrieved_text.split()) > 5:
            reference_text = f"Relevant Context:\n{retrieved_text}"
        else:
            reference_text = ""

        # Construct Gemini AI prompt
        prompt = f"""
        You are a friendly and knowledgeable agricultural assistant. Respond in a natural, engaging, and conversational tone.
        
        *User Question:*  
        {query}
        
        {reference_text}
        
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
st.caption("🤖 Powered by Google Gemini 1.5 Flash & Local Document Retrieval")
