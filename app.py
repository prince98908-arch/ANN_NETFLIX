import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# 1. Page Config
st.set_page_config(page_title="Netflix Churn Predictor", layout="wide")
st.title("📊 Netflix Customer Churn Prediction")

# 2. Assets Load Karein
@st.cache_resource
def load_assets():
    # GitHub par file ka naam 'churn_model.h5' hona chahiye
    model = load_model('churn_model.h5') 
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")

# 3. User Input Form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["North", "South", "East", "West"])
    with col2:
        sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        monthly_fee = st.number_input("Monthly Fee ($)", 10.0, 500.0, 50.0)
        profiles = st.number_input("Profiles", 1, 5, 2)
    with col3:
        watch_hours = st.number_input("Total Watch Hours", 0.0, 1000.0, 150.0)
        avg_watch = st.number_input("Avg Watch Time/Day", 0.0, 24.0, 3.0)
        last_login = st.number_input("Days Since Last Login", 0, 30, 5)
    
    genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Drama", "Sci-Fi"])
    submit = st.form_submit_button("Predict Churn Risk")

# 4. Prediction Logic (Handling 29 Features)
if submit:
    # A. Raw Data DataFrame
    data = {
        'age': age, 'gender': gender, 'subscription_type': sub_type,
        'watch_hours': watch_hours, 'last_login_days': last_login,
        'region': region, 'monthly_fee': monthly_fee, 
        'number_of_profiles': profiles, 'avg_watch_time_per_day': avg_watch,
        'favorite_genre': genre
    }
    df_input = pd.DataFrame([data])

    # B. Encoding (Jaise training mein pd.get_dummies kiya tha)
    df_encoded = pd.get_dummies(df_input)

    # C. Matching Columns (Jo 29 features training mein thay unhe yahan match karna)
    # Jo columns missing hain unhe 0 se fill karein
    for col in scaler.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Sequence sahi karein
    df_final = df_encoded[scaler.feature_names_in_]

    # D. Scale & Predict
    scaled_data = scaler.transform(df_final)
    prediction = model.predict(scaled_data)
    prob = float(prediction[0][0])

    st.divider()
    if prob > 0.5:
        st.error(f"⚠️ High Risk: {prob*100:.1f}% Churn Probability")
    else:
        st.success(f"✅ Safe: {(1-prob)*100:.1f}% Retention Probability")
