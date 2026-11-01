import pandas as pd


# Load dataset
df = pd.read_csv("data/crop_yield.csv")

# Clean categorical columns
df["Crop"] = df["Crop"].str.strip()
df["Season"] = df["Season"].str.strip()
df["State"] = df["State"].str.strip()

# Remove Production because it causes target leakage
df = df.drop(columns=["Production"])

# Display basic information
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

# Separate features and target
X = df.drop(columns=["Yield"])
y = df["Yield"]

print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(y.name)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Identify categorical and numerical features
categorical_features = ["Crop", "Season", "State"]
numerical_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features)
    ]
)

# Fit preprocessing on training data and transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)

from sklearn.ensemble import RandomForestRegressor


# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train_processed, y_train)

print("\nModel training completed successfully!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Make predictions on test data
y_pred = model.predict(X_test_processed)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Make predictions on test data
y_pred = model.predict(X_test_processed)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": y_pred
})

print("\n--- Actual vs Predicted ---")
print(comparison.head(10))

# Calculate prediction errors
errors = y_test.values - y_pred

print("\n--- Error Analysis ---")
print("Mean Error:", errors.mean())
print("Minimum Error:", errors.min())
print("Maximum Error:", errors.max())

print("\nLargest Absolute Errors:")
error_df = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": y_pred,
    "Absolute Error": np.abs(errors)
})

print(
    error_df
    .sort_values("Absolute Error", ascending=False)
    .head(10)
)

import matplotlib.pyplot as plt


# Plot actual vs predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")

plt.tight_layout()
plt.show()

import joblib


# Save trained model
joblib.dump(model, "models/crop_yield_model.pkl")

# Save preprocessing pipeline
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nModel and preprocessor saved successfully!")


