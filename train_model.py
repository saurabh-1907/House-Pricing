import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
file_path = "C:/Users/saura/Downloads/Housing.csv"
housing_data = pd.read_csv(file_path)

# Select features and target
categorical_features = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]
numerical_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]

# Encode categorical variables
label_encoders = {feature: LabelEncoder() for feature in categorical_features}
for feature in categorical_features:
    housing_data[feature] = label_encoders[feature].fit_transform(housing_data[feature])

# Define features (X) and target (y)
X = housing_data[numerical_features + categorical_features]
y = housing_data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "house_price_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and encoders saved!")
