import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings

# Suppress TensorFlow metric warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Load model (compile=False is fine for prediction-only use)
model = tf.keras.models.load_model("best_crop_model.h5", compile=False)

# Label list (should match training label order)
labels = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes',
    'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
    'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

st.set_page_config(page_title="🌾 Crop Recommendation System", page_icon="🌱", layout="centered")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: url('https://github.com/Nityam2305/Crop_Recommendation/raw/5aa2b8f9fa2f6f0f8515df76d832d9104995a632/img1.jpg') no-repeat center center fixed;
            background-size: cover;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: rgba(255,255,255,0);
        }
        .main {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
            h1, h2, h3, h4, h5, h6, p, label, .stTextInput label, .stNumberInput label {
            font-weight: bold !important;
            color: #111 !important;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
        .stMarkdown, .stTextInput, .stNumberInput {
            font-weight: bold !important;
            color: #111 !important;
        }
        .stAlert, .stSuccess, .stInfo {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Crop Recommendation System")

st.markdown("""
#### Enter the soil and climate parameters below to get the best crop recommendation.
---
""")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("🧪 Nitrogen (N)", min_value=0)
        P = st.number_input("🧪 Phosphorus (P)", min_value=0)
        K = st.number_input("🧪 Potassium (K)", min_value=0)
        pH = st.number_input("🧪 pH Level", format="%.2f")

    with col2:
        temp = st.number_input("🌡️ Temperature (°C)", format="%.2f")
        humidity = st.number_input("💧 Humidity (%)", format="%.2f")
        rainfall = st.number_input("🌧️ Rainfall (mm)", format="%.2f")

if st.button("🌿 Predict Crop"):
    input_data = np.array([[N, P, K, temp, humidity, pH, rainfall]])
    prediction = model.predict(input_data, verbose=0)
    crop_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    crop = labels[crop_index]

    st.balloons()
    st.success(f"✅ **Recommended Crop:** {crop.capitalize()}")
    st.info(f"📈 **Model Confidence:** {confidence * 100:.2f}%")
