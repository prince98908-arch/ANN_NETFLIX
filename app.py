import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Page styling
st.set_page_config(page_title="Churn Predictor", layout="centered")

# Model aur Scaler load karne ka function
@st.cache_resource
def load_assets():
    model = load_model('churn_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("📊 Customer Churn Prediction")
st.write("Customer details enter karein prediction ke liye:")

# Form layout
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        watch_hours = st.number_input("Total Watch Hours", 0.0, 5000.0, 100.0)
        last_login = st.number_input("Last Login (Days)", 0, 365, 5)
        
    with col2:
        monthly_fee = st.number_input("Monthly Fee ($)", 0.0, 500.0, 50.0)
        profiles = st.number_input("Number of Profiles", 1, 5, 1)
        avg_watch = st.number_input("Avg Watch Time/Day", 0.0, 24.0, 2.0)

    # Categorical Inputs (Note: Iske liye training wala same encoding zaruri hai)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    
    predict_btn = st.form_submit_button("Predict Churn Risk")

if predict_btn:
    # 1. Input Data taiyar karein (Sequence wahi rakhein jo training mein tha)
    # Filhal hum numerical data par focus kar rahe hain:
    features = np.array([[age, watch_hours, last_login, monthly_fee, profiles, avg_watch]])
    
    # 2. Scale karein
    scaled_features = scaler.transform(features)
    
    # 3. Predict karein
    prediction = model.predict(scaled_features)
    risk_score = prediction[0][0]

    st.divider()
    if risk_score > 0.5:
        st.error(f"🔴 High Risk! Churn hone ka chance: {risk_score*100:.2f}%")
    else:
        st.success(f"🟢 Low Risk! Retention ka chance: {(1-risk_score)*100:.2f}%")
