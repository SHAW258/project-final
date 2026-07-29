import sys
import sklearn.compose._column_transformer

# Monkey-patch _RemainderColsList for compatibility
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import joblib
import numpy as np
import pandas as pd

def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode('ascii'))

def fix_preprocessor_compat(prep):
    """Ensure older SimpleImputer instances have required attributes for scikit-learn 1.9+"""
    if hasattr(prep, 'transformers_'):
        for item in prep.transformers_:
            trans = item[1]
            if hasattr(trans, 'steps'):
                for step_name, step_obj in trans.steps:
                    if not hasattr(step_obj, '_fill_dtype'):
                        setattr(step_obj, '_fill_dtype', None)
                    if not hasattr(step_obj, '_parameter_constraints'):
                        setattr(step_obj, '_parameter_constraints', {})

try:
    model = joblib.load("xgb_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    feature_info = joblib.load("feature_info.joblib")
    
    fix_preprocessor_compat(preprocessor)

    safe_print("[SUCCESS] Loaded trained XGBoost model and preprocessor successfully!")

    input_pollutants = {
        "PM2.5": 36,
        "PM10": 64,
        "NO2": 8,
        "SO2": 2,
        "CO": 0.303,
        "O3": 0.016
    }
    timestamp = pd.to_datetime("2025-06-08 14:00:00")
    time_features = {
        "Hour": timestamp.hour,
        "Day": timestamp.day,
        "Month": timestamp.month,
        "DayOfWeek": timestamp.dayofweek,
        "Quarter": timestamp.quarter,
        "DayOfYear": timestamp.dayofyear,
        "Hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "Hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
        "Month_sin": np.sin(2 * np.pi * timestamp.month / 12),
        "Month_cos": np.cos(2 * np.pi * timestamp.month / 12),
        "Day_sin": np.sin(2 * np.pi * timestamp.day / 31),
        "Day_cos": np.cos(2 * np.pi * timestamp.day / 31),
    }

    input_data = {**input_pollutants, **time_features}
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_info["all_columns"]]

    processed_input = preprocessor.transform(input_df)
    predicted_aqi = model.predict(processed_input)

    safe_print(f"[SUCCESS] Native Trained XGBoost Model Predicted AQI: {predicted_aqi[0]:.2f}")

except Exception as e:
    safe_print(f"[FAIL] Error: {e}")
