import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load model and preprocessor
model = joblib.load("xgb_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
feature_info = joblib.load("feature_info.joblib")

# Sample input
input_pollutants = {
    "PM2.5": 46,
    "PM10": 98,
    "NO2": 12,
    "SO2": 2,
    "CO": 0.311,
    "O3": 0.012
}

# Input timestamp
timestamp_str = "2025-06-09 22:00:00"
timestamp = pd.to_datetime(timestamp_str)

# Extract time features from timestamp
time_features = {
    "Hour": timestamp.hour,
    "Day": timestamp.day,
    "Month": timestamp.month,
    "DayOfWeek": timestamp.dayofweek,
    "Quarter": timestamp.quarter,
    "DayOfYear": timestamp.dayofyear,
}

# Compute cyclical time features
time_features.update({
    "Hour_sin": np.sin(2 * np.pi * time_features["Hour"] / 24),
    "Hour_cos": np.cos(2 * np.pi * time_features["Hour"] / 24),
    "Month_sin": np.sin(2 * np.pi * time_features["Month"] / 12),
    "Month_cos": np.cos(2 * np.pi * time_features["Month"] / 12),
    "Day_sin": np.sin(2 * np.pi * time_features["Day"] / 31),
    "Day_cos": np.cos(2 * np.pi * time_features["Day"] / 31),
})

# Combine pollutant and time features
input_data = {**input_pollutants, **time_features}
input_df = pd.DataFrame([input_data])

# Match training feature columns
input_df = input_df[feature_info["all_columns"]]

# Preprocess and predict
processed_input = preprocessor.transform(input_df)
predicted_aqi = model.predict(processed_input)

print(f"✅ Timestamp: {timestamp_str}")
print(f"✅ Predicted AQI: {predicted_aqi[0]:.2f}")
