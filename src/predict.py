import joblib


# Load saved model and preprocessing pipeline
model = joblib.load("models/crop_yield_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model loaded successfully!")
print("Preprocessor loaded successfully!")

import pandas as pd


# Sample input
sample_data = pd.DataFrame([{
    "Crop": "Rice",
    "Crop_Year": 2020,
    "Season": "Kharif",
    "State": "Uttar Pradesh",
    "Area": 10000,
    "Annual_Rainfall": 1200,
    "Fertilizer": 1500000,
    "Pesticide": 5000
}])

# Preprocess the input
sample_processed = preprocessor.transform(sample_data)

# Make prediction
prediction = model.predict(sample_processed)

print("\n--- Test Prediction ---")
print("Predicted Crop Yield:", prediction[0])