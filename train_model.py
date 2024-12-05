import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
file_path = "C:/Users/saura/Downloads/Housing.csv"  # Replace with the correct dataset path
data = pd.read_csv(file_path)

# Encode binary categorical variables
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})

# One-hot encode the 'furnishingstatus' column
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)

# Separate features and target variable
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad', 'guestroom',
          'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']]
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the trained model
joblib.dump(model, "house_price_model_updated.pkl")
print("Model saved as house_price_model_updated.pkl")
