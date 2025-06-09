import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

def load_model_and_preprocessor():
    try:
        model = joblib.load("xgb_model.joblib")
        preprocessor = joblib.load("preprocessor.joblib")
        feature_info = joblib.load("feature_info.joblib")
        print(" Model and preprocessor loaded successfully")
        return model, preprocessor, feature_info
    except FileNotFoundError as e:
        print(f" Error loading model files: {e}")
        return None, None, None

def create_time_features(timestamp):
    if not hasattr(timestamp, 'dayofweek'):
        timestamp = pd.to_datetime(timestamp)
    return {
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
        "Day_cos": np.cos(2 * np.pi * timestamp.day / 31)
    }

def prepare_prediction_data(pollutant_data, timestamp, feature_info):
    data_dict = {col: pollutant_data.get(col, 0.0) for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']}
    data_dict.update(create_time_features(timestamp))
    df = pd.DataFrame([data_dict])
    for col in feature_info['all_columns']:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_info['all_columns']]

def predict_single_timestamp(model, preprocessor, feature_info, pollutant_data, timestamp):
    df = prepare_prediction_data(pollutant_data, timestamp, feature_info)
    X = preprocessor.transform(df)
    return int(model.predict(X)[0])

def forecast_24_hours(model, preprocessor, feature_info, pollutant_data, start_timestamp):
    forecasts = []
    current_data = pollutant_data.copy()
    timestamp = start_timestamp

    for hour in range(24):
        prediction = predict_single_timestamp(model, preprocessor, feature_info, current_data, timestamp)
        forecasts.append({
            "timestamp": timestamp,
            "predicted_aqi": prediction,
            "hour_ahead": hour + 1
        })
        timestamp += timedelta(hours=1)
        for pollutant in current_data:
            variation = np.random.normal(0, 0.05)
            current_data[pollutant] *= (1 + variation)
            current_data[pollutant] = max(0, current_data[pollutant])
    return pd.DataFrame(forecasts)

def forecast_7_days(model, preprocessor, feature_info, pollutant_data, start_timestamp):
    forecasts = []
    current_data = pollutant_data.copy()
    timestamp = start_timestamp.replace(hour=12, minute=0, second=0, microsecond=0)

    for day in range(7):
        prediction = predict_single_timestamp(model, preprocessor, feature_info, current_data, timestamp)
        forecasts.append({
            "date": timestamp.date(),
            "predicted_aqi": prediction
        })
        timestamp += timedelta(days=1)
        for pollutant in current_data:
            variation = np.random.normal(0, 0.05)
            current_data[pollutant] *= (1 + variation)
            current_data[pollutant] = max(0, current_data[pollutant])
    return pd.DataFrame(forecasts)

def display_aqi_level(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "🔴 Unhealthy"
    elif aqi <= 300: return "🟣 Very Unhealthy"
    else: return "🟤 Hazardous"

def main():
    print("🔮 AQI Forecast System (24 Hours + 7 Days)")
    model, preprocessor, feature_info = load_model_and_preprocessor()
    if model is None:
        return

    print("\n Enter pollutant values:")
    pollutant_data = {}
    default_values = {'PM2.5': 35.0, 'PM10': 50.0, 'NO2': 40.0, 'SO2': 10.0, 'CO': 1.0, 'O3': 60.0}
    for col in default_values:
        val = input(f"{col} (default {default_values[col]}): ").strip()
        try:
            pollutant_data[col] = float(val) if val else default_values[col]
        except ValueError:
            print(f" Invalid input for {col}, using default: {default_values[col]}")
            pollutant_data[col] = default_values[col]

    current_time = datetime.now()

    # 24-Hour Forecast
    forecast_24h_df = forecast_24_hours(model, preprocessor, feature_info, pollutant_data, current_time)
    print("\n 24-Hour Forecast:")
    print("-" * 60)
    for _, row in forecast_24h_df.iterrows():
        ts = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        level = display_aqi_level(row['predicted_aqi'])
        print(f"{ts} | AQI: {row['predicted_aqi']} | {level}")

    print("\n 24-Hour Summary:")
    print(f"Average AQI: {forecast_24h_df['predicted_aqi'].mean():.0f}")
    print(f"Min AQI: {forecast_24h_df['predicted_aqi'].min()}")
    print(f"Max AQI: {forecast_24h_df['predicted_aqi'].max()}")
    print(f"Std Dev: {forecast_24h_df['predicted_aqi'].std():.0f}")

    # 7-Day Forecast
    forecast_7d_df = forecast_7_days(model, preprocessor, feature_info, pollutant_data, current_time)
    print("\n 7-Day Daily AQI Forecast:")
    print("-" * 60)
    for _, row in forecast_7d_df.iterrows():
        print(f"{row['date']} | AQI: {row['predicted_aqi']} | {display_aqi_level(row['predicted_aqi'])}")

    # Save results
    #timestamp_str = current_time.strftime('%Y%m%d_%H%M')
    #forecast_24h_df.to_csv(f"aqi_24h_forecast_{timestamp_str}.csv", index=False)
    #forecast_7d_df.to_csv(f"aqi_7d_forecast_{timestamp_str}.csv", index=False)
    print(f"\n All are Generated")

if __name__ == "__main__":
    main()