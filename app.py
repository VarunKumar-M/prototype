import os
import faiss
import numpy as np
import streamlit as st
import requests
from dotenv import load_dotenv
from googletrans import Translator
from langdetect import detect
from langchain_google_genai import GoogleGenerativeAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeather API Key

if not GEMINI_API_KEY or not WEATHER_API_KEY:
    raise ValueError("API keys are missing. Set them in a .env file.")

# Initialize AI & Embedding Model
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
translator = Translator()

# Streamlit UI Configuration
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")
st.title("🌱 AgriGPT - Smart Agriculture Assistant with RAG & Weather Insights")

# **FAISS-Based Retrieval**
class FAISSRetriever:
    def __init__(self):
        self.index = None
        self.texts = []

    def build_index(self, data):
        """Builds FAISS index for fast text retrieval."""
        self.texts = data
        embeddings = np.array([embed_model.encode(text) for text in data]).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        """Retrieves the most relevant text based on FAISS search."""
        if self.index is None:
            return ""
        query_embedding = np.array([embed_model.encode(query)]).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return " ".join([self.texts[i] for i in indices[0] if i < len(self.texts)])

# Load agricultural dataset
agriculture_data = [
    "Crop rotation improves soil fertility.",
    "Drip irrigation helps conserve water.",
    "Paddy requires a warm climate and standing water.",
    "Wheat grows best in temperate climates.",
    "Organic farming reduces chemical usage.",
    "Government schemes like PM-KISAN provide financial aid to farmers.",
    "Weather conditions affect pest control methods.",
]
retriever = FAISSRetriever()
retriever.build_index(agriculture_data)

# **Get User's Location & Weather**
def get_location():
    try:
        ip_info = requests.get("https://ipinfo.io/json").json()
        city, region, country = ip_info["city"], ip_info["region"], ip_info["country"]
        lat, lon = ip_info["loc"].split(",")
        return city, region, country, lat, lon
    except:
        return None, None, None, None, None

def get_weather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        weather_data = requests.get(url).json()
        temp = weather_data["main"]["temp"]
        condition = weather_data["weather"][0]["description"].title()
        return temp, condition
    except:
        return None, None

city, region, country, lat, lon = get_location()
weather_temp, weather_condition = get_weather(lat, lon)

if city and weather_temp:
    st.info(f"📍 **Location:** {city}, {region}, {country}")
    st.info(f"🌤 **Current Weather:** {weather_temp}°C, {weather_condition}")

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

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        retrieved_text = retriever.retrieve(query)

        # **Use context only if it's relevant**
        context = retrieved_text if retrieved_text and len(retrieved_text.split()) > 5 else ""

        # **Detect or Set Language**
        if lang_options[selected_lang] == "auto":
            try:
                response_lang = detect(query)
            except:
                response_lang = "en"
        else:
            response_lang = lang_options[selected_lang]

        # **Limit Memory to Last 5 Messages**
        history = "\n".join(st.session_state.chat_history[-5:])

        # **Generate Response**
        prompt = f"""
        You are **AgriGPT**, an intelligent, professional, and structured agricultural assistant.  
        Your goal is to provide **logically sound, structured, and context-aware** answers.  

        ### **How You Should Respond:**
        - **Analyze the query logically** before answering.  
        - **Break down complex topics step by step.**  
        - **Provide location-based and weather-based advice if applicable.**  
        - **Justify recommendations with facts.**  

        ### **User's Location & Weather:**
        - **Location:** {city}, {region}, {country}  
        - **Weather:** {weather_temp}°C, {weather_condition}  

        ### **Relevant Context from RAG:**  
        {context}

        ### **User Query (Language: {response_lang}):**  
        {query}

        ### **Response Output Guidelines:**
        - **Direct, well-structured, logically justified answer**.  
        - **Provide bullet points or step-by-step format**.  
        - **Adapt response length based on query complexity.**  
        """

        response = gemini.invoke(prompt)

        if response:
            translated_response = translator.translate(response.strip(), dest=response_lang).text

            st.markdown("### 🎓 AgriGPT Response:")
            st.markdown(translated_response)

            st.session_state.chat_history.append(f"User ({response_lang}): {query}\nAgriGPT ({response_lang}): {translated_response}")
            st.session_state.chat_history = st.session_state.chat_history[-5:]

        else:
            st.error("Unable to generate a response at the moment.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Gemini 1.5 Flash, FAISS RAG, Google Translate & OpenWeather")