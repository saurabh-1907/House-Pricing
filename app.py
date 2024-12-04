import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("house_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Streamlit App
st.title("House Price Prediction")

# Input features
area = st.number_input("Area (in sq ft)", min_value=500, max_value=20000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)
stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=1)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=0)

# Categorical features
mainroad = st.selectbox("Near Main Road?", ["yes", "no"])
guestroom = st.selectbox("Guest Room?", ["yes", "no"])
basement = st.selectbox("Basement?", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating?", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning?", ["yes", "no"])
prefarea = st.selectbox("Preferred Area?", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Prepare input data
categorical_inputs = {
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}

# Encode categorical features
for feature, value in categorical_inputs.items():
    categorical_inputs[feature] = label_encoders[feature].transform([value])[0]

# Combine all features
input_features = [
    area, bedrooms, bathrooms, stories, parking,
    categorical_inputs["mainroad"],
    categorical_inputs["guestroom"],
    categorical_inputs["basement"],
    categorical_inputs["hotwaterheating"],
    categorical_inputs["airconditioning"],
    categorical_inputs["prefarea"],
    categorical_inputs["furnishingstatus"]
]

# Predict and display result
if st.button("Predict"):
    prediction = model.predict([input_features])
    st.success(f"Estimated House Price: ₹{prediction[0]:,.2f}")
