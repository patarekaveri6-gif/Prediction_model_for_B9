import streamlit as st
import pandas as pd
import joblib
import time
from streamlit_lottie import st_lottie
import requests

# Page Configuration
st.set_page_config(page_title="AI Impact Predictor", page_icon="🤖", layout="centered")

# Custom CSS for Animation and Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_dot7jrnx.json")

# Header Section
st.title("🤖 AI Usage & Academic Impact")
st.write("Predict the level of impact AI tools have on student performance.")
if lottie_ai:
    st_lottie(lottie_ai, height=200)

# Load the model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Input Form
with st.expander("📝 Enter Student Details", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate", "PhD"])
        city = st.selectbox("City Type", ["Metropolitan", "Urban", "Rural"])

    with col2:
        ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Gemini", "Claude", "Notion AI", "Other"])
        usage = st.slider("Daily Usage (Hours)", 0.0, 24.0, 2.0)
        purpose = st.selectbox("Primary Purpose", ["Studying", "Research", "Coding", "General Inquiry"])
        impact_grades = st.selectbox("Perceived Impact on Grades", ["Positive", "Neutral", "Negative"])

# Create DataFrame for prediction
# Note: Ensure the values match the encoding used during training (e.g., if you used LabelEncoder)
input_data = pd.DataFrame([[
    age, gender, education, city, ai_tool, usage, purpose, impact_grades
]], columns=['Age', 'Gender', 'Education_Level', 'City', 'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades'])

# Prediction Button
if st.button("✨ Predict Impact Level"):
    with st.spinner('Analyzing patterns...'):
        time.sleep(1.5) # Artificial delay for "animated" feel
        try:
            prediction = model.predict(input_data)[0]
            
            # Displaying result with color coding
            st.markdown("---")
            if prediction == "High":
                st.success(f"### Predicted Impact Level: **{prediction}** 🚀")
            elif prediction == "Medium":
                st.info(f"### Predicted Impact Level: **{prediction}** 📊")
            else:
                st.warning(f"### Predicted Impact Level: **{prediction}** 📉")
                
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Check if your categorical variables (like 'Gender') need numerical encoding.")
