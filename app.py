import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("house_price_model_updated.pkl")

# App Title
st.title("House Price Prediction App")

# Input fields for user
st.header("Enter House Details")

area = st.number_input("Area (in sq ft)", min_value=0, value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=1)
stories = st.number_input("Number of Stories", min_value=1, value=1)
parking = st.number_input("Number of Parking Spaces", min_value=0, value=0)

mainroad = st.selectbox("Is there a main road?", ["No", "Yes"])
guestroom = st.selectbox("Is there a guestroom?", ["No", "Yes"])
basement = st.selectbox("Is there a basement?", ["No", "Yes"])
hotwaterheating = st.selectbox("Hot Water Heating?", ["No", "Yes"])
airconditioning = st.selectbox("Air Conditioning?", ["No", "Yes"])
prefarea = st.selectbox("Is there a preferred area?", ["No", "Yes"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

# Encode user inputs
data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad": 1 if mainroad == "Yes" else 0,
    "guestroom": 1 if guestroom == "Yes" else 0,
    "basement": 1 if basement == "Yes" else 0,
    "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
    "airconditioning": 1 if airconditioning == "Yes" else 0,
    "prefarea": 1 if prefarea == "Yes" else 0,
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "Semi-Furnished" else 0,
    "furnishingstatus_unfurnished": 1 if furnishingstatus == "Unfurnished" else 0,
}

# Convert to DataFrame
input_data = pd.DataFrame([data])

# Predict house price
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"Predicted House Price: ₹{round(prediction, 2)}")
