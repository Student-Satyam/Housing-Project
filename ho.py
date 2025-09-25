import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler8.pkl")
columns = joblib.load("columns8.pkl")

st.title("Sattu's 🏠 House Price Prediction App")

st.write("Enter the details below to predict the house price:")

# Create input fields for features
area = st.number_input("Area (in sq. ft)", min_value=500, max_value=10000, step=50)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
stories = st.number_input("Number of Stories", min_value=1, max_value=5, step=1)
parking = st.number_input("Number of Parking Spaces", min_value=0, max_value=5, step=1)
airconditioning = st.selectbox("Air Conditioning", ["No", "Yes"])
prefarea = st.selectbox("Preferred Area", ["No", "Yes"])

# Convert categorical inputs to numbers
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

# Make dataframe with same column order
input_data = pd.DataFrame([[prefarea, parking, area, bedrooms, bathrooms, airconditioning, stories]],
                          columns=columns)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"🏡 Estimated House Price: ₹ {prediction[0]:,.2f}")
