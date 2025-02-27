import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI  
from langdetect import detect
import requests

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

# Initialize Gemini 1.5 Flash AI
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Streamlit UI Configuration
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")

st.title("🌱 AgriGPT - Smart Agriculture Assistant")
st.write("Get farming insights, soil conditions, temperature, and more.")

# **Location, Soil & Temperature Data**
def get_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        city = res.get("city", "Unknown")
        loc = res.get("loc", "0,0").split(",")
        return city, float(loc[0]), float(loc[1])
    except:
        return "Unknown", 0, 0

def get_temperature(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        res = requests.get(url).json()
        return res["current_weather"]["temperature"]
    except:
        return "N/A"

def get_soil_info(lat, lon):
    url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}"
    try:
        res = requests.get(url).json()
        soil_type = res["properties"]["classification"]["dominant"].get("WRB", "Unknown Soil Type")
        return soil_type
    except:
        return "Data not available"

# Get user location, soil type, and temperature
city, lat, lon = get_location()
temperature = get_temperature(lat, lon)
soil_type = get_soil_info(lat, lon)

st.markdown(f"📍 **Location:** {city}  \n🌡 **Temperature:** {temperature}°C  \n🌱 **Soil Type:** {soil_type}")

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

# **User Input**
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        # **Detect or Set Language**
        if lang_options[selected_lang] == "auto":
            try:
                response_lang = detect(query)
            except:
                response_lang = "en"  # Default to English if detection fails
        else:
            response_lang = lang_options[selected_lang]

        # **Logical Reasoning & Thoughtful Processing**
        prompt = f"""
        You are **AgriGPT**, an intelligent, professional, and logically sound AI assistant.  
        You **analyze every query intelligently** and generate responses based on context.  
        You **never produce random responses** but **always provide well-reasoned answers**.

        ### **How You Should Respond:**
        - **Think logically before answering.**  
        - **If it's a professional farming query, provide a structured, well-reasoned response.**  
        - **If it's a greeting or casual talk, respond in a natural, human-like way but remain professional.**  
        - **If the question is unprofessional or irrelevant, handle it with intelligence and professionalism.**  
        - **Always give responses in bullet points unless a conversational tone is necessary.**  

        ### **Additional Information Based on User's Location:**
        - 📍 **Location:** {city}  
        - 🌡 **Temperature:** {temperature}°C  
        - 🌱 **Soil Type:** {soil_type}  

        ### **User Query (Language: {response_lang}):**  
        {query}

        ### **Response Output Guidelines:**
        - **Always respond in bullet points if informative.**  
        - **If the query is a greeting or casual talk, respond appropriately but intelligently.**  
        - **Maintain professionalism and logical reasoning in every response.**  
        """

        # **Generate Response**
        response = gemini.invoke(prompt)

        if response:
            st.markdown("### 🎓 AgriGPT Response:")
            st.markdown(response.strip())
        else:
            st.error("Unable to generate a response at the moment.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Gemini 1.5 Flash, FAO SoilGrids, Open-Meteo, & OpenStreetMap")