import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
MODEL_PATH = "model/insurance_model.pkl"

st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰"
)

st.title("Medical Insurance Cost Prediction")
st.write("Predict insurance charges using Machine Learning.")

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please add insurance_model.pkl inside model folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.divider()

# User inputs
age = st.number_input("Age", 18, 100, 30)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Prediction
if st.button("Predict Insurance Charges"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Insurance Charges: ${prediction:,.2f}"
    )

    st.caption(
        "This is an estimated prediction, not an actual insurance quotation."
    )