import sys
import os

# Ensure root directory is in sys.path for cross-module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Monkey-patch sklearn modules for backwards compatibility with serialized preprocessor.joblib
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from database.database import engine, get_db, Base, safe_print, is_mysql
from database.models import PredictionRecord
from backend.schemas import (
    PredictRequest, PredictApiResponse, Forecast24HourResponse, 
    Forecast7DayResponse, HealthResponse, ApiInfoResponse
)

app = FastAPI(
    title="Enhanced AQI Prediction API (FastAPI)",
    version="2.1",
    description="Real-time AQI prediction with 24h and 7d forecasting using FastAPI and MySQL database."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
preprocessor = None
feature_info = None

def fix_preprocessor_compat(prep):
    """Ensure unpickled SimpleImputer instances have required attributes for scikit-learn 1.9+"""
    if hasattr(prep, 'transformers_'):
        for item in prep.transformers_:
            trans = item[1]
            if hasattr(trans, 'steps'):
                for step_name, step_obj in trans.steps:
                    if not hasattr(step_obj, '_fill_dtype'):
                        setattr(step_obj, '_fill_dtype', None)
                    if not hasattr(step_obj, '_parameter_constraints'):
                        setattr(step_obj, '_parameter_constraints', {})

def load_model_and_preprocessor():
    global model, preprocessor, feature_info
    try:
        model_dir = os.path.join(ROOT_DIR, "ml_model")
        xgb_path = os.path.join(model_dir, "xgb_model.joblib")
        prep_path = os.path.join(model_dir, "preprocessor.joblib")
        info_path = os.path.join(model_dir, "feature_info.joblib")

        if os.path.exists(xgb_path) and os.path.exists(prep_path):
            model = joblib.load(xgb_path)
            preprocessor = joblib.load(prep_path)
            feature_info = joblib.load(info_path)
            fix_preprocessor_compat(preprocessor)
            safe_print("[Model] Native trained XGBoost model and preprocessor loaded successfully!")
            return True
        else:
            safe_print("[Model Notice] Model files not found in ml_model/, running in Mock mode.")
            model = MockModel()
            preprocessor = MockPreprocessor()
            feature_info = {"all_columns": ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']}
            return False
    except Exception as e:
        safe_print(f"[Model Error] Failed to load model files: {e}")
        model = MockModel()
        preprocessor = MockPreprocessor()
        feature_info = {"all_columns": ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']}
        return False

class MockModel:
    """Mock model for testing without actual model files"""
    def predict(self, X):
        if hasattr(X, 'iloc'):
            pm25 = X.iloc[0, 0] if len(X.columns) > 0 else 25
            pm10 = X.iloc[0, 1] if len(X.columns) > 1 else 50
            no2 = X.iloc[0, 2] if len(X.columns) > 2 else 10
        else:
            pm25, pm10, no2 = 25, 50, 10
        
        base_aqi = int((pm25 * 2.5) + (pm10 * 1.2) + (no2 * 3.0))
        return [max(10, min(500, base_aqi))]

class MockPreprocessor:
    """Mock preprocessor for testing"""
    def transform(self, X):
        return X.values if hasattr(X, 'values') else X

@app.on_event("startup")
def startup_event():
    """Create database tables and load model on server startup."""
    try:
        Base.metadata.create_all(bind=engine)
        safe_print("[Database] Tables verified/created.")
    except Exception as e:
        safe_print(f"[Database Notice] Could not create database tables automatically: {e}")
    
    load_model_and_preprocessor()

def create_time_features(timestamp):
    """Create time-based features for the model"""
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

def prepare_prediction_data(pollutant_data, timestamp, feat_info):
    """Prepare data for prediction"""
    data_dict = {col: pollutant_data.get(col, 0.0) for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']}
    data_dict.update(create_time_features(timestamp))
    df = pd.DataFrame([data_dict])
    
    required_cols = feat_info.get('all_columns', []) if feat_info else list(df.columns)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    return df[required_cols]

def get_aqi_details(aqi_value):
    """Get comprehensive AQI details including color, emoji, and health message"""
    if aqi_value <= 50:
        return {
            "level": "Good",
            "color": "#00E400",
            "emoji": "🟢",
            "health_message": "Air quality is satisfactory, and air pollution poses little or no risk."
        }
    elif aqi_value <= 100:
        return {
            "level": "Moderate", 
            "color": "#FFFF00",
            "emoji": "🟡",
            "health_message": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
        }
    elif aqi_value <= 150:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "color": "#FF7E00", 
            "emoji": "🟠",
            "health_message": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        }
    elif aqi_value <= 200:
        return {
            "level": "Unhealthy",
            "color": "#FF0000",
            "emoji": "🔴", 
            "health_message": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
        }
    elif aqi_value <= 300:
        return {
            "level": "Very Unhealthy",
            "color": "#8F3F97",
            "emoji": "🟣",
            "health_message": "Health alert: The risk of health effects is increased for everyone."
        }
    else:
        return {
            "level": "Hazardous",
            "color": "#7E0023", 
            "emoji": "🟤",
            "health_message": "Health warning of emergency conditions: everyone is more likely to be affected."
        }

def predict_single_timestamp(mdl, prep, feat_info, pollutant_data, timestamp):
    """Predict AQI for a single timestamp"""
    try:
        df = prepare_prediction_data(pollutant_data, timestamp, feat_info)
        X = prep.transform(df)
        prediction = int(mdl.predict(X)[0])
        
        time_factor = np.sin(2 * np.pi * timestamp.hour / 24) * 5
        pollutant_factor = (pollutant_data.get('PM2.5', 25) / 25) * 3
        variation = time_factor + pollutant_factor
        prediction = max(0, int(prediction + variation))
        
        return prediction
    except Exception as e:
        safe_print(f"[Prediction Warning] Fallback calculation due to: {e}")
        pm25 = pollutant_data.get('PM2.5', 25)
        pm10 = pollutant_data.get('PM10', 50)
        no2 = pollutant_data.get('NO2', 10)
        base_aqi = int((pm25 * 2.5) + (pm10 * 1.2) + (no2 * 3.0))
        return max(10, min(500, base_aqi))

def create_realistic_visualization(data, plot_type="current"):
    """Create realistic AQI visualizations that match professional air quality dashboards"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        if plot_type == "current":
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Air Quality Index Dashboard', fontsize=20, fontweight='bold', y=0.95)
            
            aqi_value = data['aqi']
            aqi_details = get_aqi_details(aqi_value)
            pollutants = data['pollutants']
            
            # 1. AQI Gauge Chart
            ax1.pie([aqi_value, max(0, 500-aqi_value)], colors=[aqi_details['color'], '#f0f0f0'], 
                   startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
            ax1.text(0, 0, f'{aqi_value}\nAQI', ha='center', va='center', 
                    fontsize=24, fontweight='bold', color=aqi_details['color'])
            ax1.set_title(f'{aqi_details["level"]} {aqi_details["emoji"]}', 
                         fontsize=16, fontweight='bold', pad=20)
            
            # 2. Pollutant Concentrations
            pollutant_names = list(pollutants.keys())
            pollutant_values = list(pollutants.values())
            colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']
            
            bars = ax2.barh(pollutant_names, pollutant_values, color=colors[:len(pollutant_names)])
            ax2.set_xlabel('Concentration (μg/m³)', fontweight='bold')
            ax2.set_title('Pollutant Levels', fontsize=16, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars, pollutant_values)):
                ax2.text(value + max(pollutant_values, default=1)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}', va='center', fontweight='bold')
            
            # 3. AQI Scale Reference
            aqi_ranges = [50, 100, 150, 200, 300, 500]
            aqi_colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']
            aqi_labels = ['Good', 'Moderate', 'Unhealthy\nfor Sensitive', 'Unhealthy', 'Very\nUnhealthy', 'Hazardous']
            
            y_pos = range(len(aqi_ranges))
            bars3 = ax3.barh(y_pos, aqi_ranges, color=aqi_colors, alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(aqi_labels, fontsize=10)
            ax3.set_xlabel('AQI Value', fontweight='bold')
            ax3.set_title('AQI Scale Reference', fontsize=16, fontweight='bold')
            
            current_level_idx = next((i for i, val in enumerate(aqi_ranges) if aqi_value <= val), len(aqi_ranges)-1)
            bars3[current_level_idx].set_edgecolor('black')
            bars3[current_level_idx].set_linewidth(3)
            
            # 4. Health Recommendations
            ax4.text(0.5, 0.7, 'Health Advisory', ha='center', va='center', 
                    fontsize=16, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.5, 0.4, aqi_details['health_message'], ha='center', va='center', 
                    fontsize=12, wrap=True, transform=ax4.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=aqi_details['color'], alpha=0.2))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
        elif plot_type == "24hours":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle('24-Hour AQI Forecast', fontsize=20, fontweight='bold')
            
            timestamps = [pd.to_datetime(forecast['timestamp']) for forecast in data['forecasts']]
            aqi_values = [forecast['predicted_aqi'] for forecast in data['forecasts']]
            colors = [forecast['color'] for forecast in data['forecasts']]
            
            ax1.plot(timestamps, aqi_values, linewidth=4, marker='o', markersize=8, 
                    color='#2c3e50', markerfacecolor='white', markeredgewidth=2)
            
            for i in range(len(timestamps)-1):
                ax1.axvspan(timestamps[i], timestamps[i+1], 
                           facecolor=colors[i], alpha=0.2)
            
            aqi_levels = [50, 100, 150, 200, 300]
            level_colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']
            level_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy']
            
            for level, color, label in zip(aqi_levels, level_colors, level_labels):
                ax1.axhline(y=level, color=color, linestyle='--', alpha=0.7, linewidth=2)
                ax1.text(timestamps[-1], level, f' {label}', va='center', fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
            
            ax1.set_ylabel('AQI Value', fontsize=14, fontweight='bold')
            ax1.set_title('Hourly AQI Predictions', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            stats = data.get('statistics', {})
            stat_names = ['Min', 'Average', 'Max']
            stat_values = [stats.get('min', 0), stats.get('average', 0), stats.get('max', 0)]
            stat_colors = ['#27ae60', '#3498db', '#e74c3c']
            
            bars = ax2.bar(stat_names, stat_values, color=stat_colors, alpha=0.8, width=0.6)
            ax2.set_ylabel('AQI Value', fontsize=14, fontweight='bold')
            ax2.set_title('24-Hour Statistics', fontsize=16, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, value in zip(bars, stat_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
        elif plot_type == "7days":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle('7-Day AQI Forecast', fontsize=20, fontweight='bold')
            
            forecasts = data['forecasts']
            dates = [pd.to_datetime(forecast['date']) for forecast in forecasts]
            aqi_values = [forecast['predicted_aqi'] for forecast in forecasts]
            day_names = [forecast['day_name'][:3] for forecast in forecasts]
            colors = [forecast['color'] for forecast in forecasts]
            
            bars = ax1.bar(range(len(day_names)), aqi_values, color=colors, alpha=0.8, 
                          edgecolor='white', linewidth=2, width=0.7)
            
            ax1.set_xticks(range(len(day_names)))
            ax1.set_xticklabels(day_names, fontweight='bold')
            ax1.set_ylabel('AQI Value', fontsize=14, fontweight='bold')
            ax1.set_title('Daily AQI Predictions', fontsize=16, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            for i, (bar, value, forecast) in enumerate(zip(bars, aqi_values, forecasts)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
                        f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                        forecast['emoji'], ha='center', va='center', fontsize=20)
            
            stats = data.get('weekly_statistics', {})
            weekday_avg = stats.get('weekday_avg', 75)
            weekend_avg = stats.get('weekend_avg', 65)
            
            pie_data = [weekday_avg, weekend_avg]
            pie_labels = ['Weekdays', 'Weekends']
            pie_colors = ['#3498db', '#e67e22']
            
            ax2.pie(pie_data, labels=pie_labels, colors=pie_colors, 
                    autopct='%1.0f%%', startangle=90, 
                    textprops={'fontweight': 'bold', 'fontsize': 12})
            
            ax2.set_title('Weekday vs Weekend\nAverage AQI', fontsize=16, fontweight='bold')
            
            trend = stats.get('trend', 'Stable')
            trend_color = '#27ae60' if trend == 'Improving' else '#e74c3c' if trend == 'Worsening' else '#f39c12'
            ax2.text(0, -1.3, f'Weekly Trend: {trend}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=trend_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=trend_color, alpha=0.2))
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64
        
    except Exception as e:
        safe_print(f"[Visualization Warning] Error generating plot: {e}")
        plt.close('all')
        return ""

def forecast_24_hours(mdl, prep, feat_info, pollutant_data, start_timestamp):
    """Generate 24-hour forecast with error handling"""
    forecasts = []
    current_data = pollutant_data.copy()
    timestamp = start_timestamp.replace(minute=0, second=0, microsecond=0)

    for hour in range(24):
        try:
            prediction = predict_single_timestamp(mdl, prep, feat_info, current_data, timestamp)
            aqi_details = get_aqi_details(prediction)
            
            forecasts.append({
                "timestamp": timestamp.isoformat(),
                "predicted_aqi": prediction,
                "hour_ahead": hour + 1,
                "level": aqi_details["level"],
                "color": aqi_details["color"],
                "emoji": aqi_details["emoji"],
                "health_message": aqi_details["health_message"]
            })
            
            timestamp += timedelta(hours=1)
            
            for pollutant in current_data:
                hour_factor = 0.02 * np.sin(2 * np.pi * (timestamp.hour + hour) / 24)
                current_data[pollutant] *= (1 + hour_factor)
                current_data[pollutant] = max(0, current_data[pollutant])
                
        except Exception as e:
            safe_print(f"[Forecast 24h Error] Hour {hour}: {e}")
            continue

    aqi_values = [f['predicted_aqi'] for f in forecasts] or [50]
    statistics = {
        "average": float(np.mean(aqi_values)),
        "min": int(np.min(aqi_values)),
        "max": int(np.max(aqi_values)),
        "median": int(np.median(aqi_values)),
        "std": float(np.std(aqi_values))
    }
    
    good_hours = [i for i, aqi in enumerate(aqi_values) if aqi <= 50]
    bad_hours = [i for i, aqi in enumerate(aqi_values) if aqi > 150]
    
    recommendations = {
        "best_time_outdoor": f"{good_hours[0]:02d}:00-{good_hours[-1]:02d}:00" if good_hours else "Limited good hours",
        "avoid_time": f"{bad_hours[0]:02d}:00-{bad_hours[-1]:02d}:00" if bad_hours else "No critical hours",
        "overall_trend": "Improving" if aqi_values[-1] < aqi_values[0] else "Worsening" if aqi_values[-1] > aqi_values[0] else "Stable"
    }
    
    return {
        "forecasts": forecasts,
        "statistics": statistics,
        "recommendations": recommendations
    }

def forecast_7_days(mdl, prep, feat_info, pollutant_data, start_timestamp):
    """Generate 7-day forecast with error handling"""
    forecasts = []
    current_data = pollutant_data.copy()
    timestamp = start_timestamp.replace(hour=12, minute=0, second=0, microsecond=0)

    for day in range(7):
        try:
            prediction = predict_single_timestamp(mdl, prep, feat_info, current_data, timestamp)
            aqi_details = get_aqi_details(prediction)
            
            forecasts.append({
                "date": timestamp.date().isoformat(),
                "predicted_aqi": prediction,
                "level": aqi_details["level"],
                "color": aqi_details["color"],
                "emoji": aqi_details["emoji"],
                "health_message": aqi_details["health_message"],
                "is_weekend": timestamp.weekday() >= 5,
                "day_name": timestamp.strftime("%A")
            })
            
            timestamp += timedelta(days=1)
            
            for pollutant in current_data:
                day_factor = 0.05 * np.sin(2 * np.pi * day / 7)
                current_data[pollutant] *= (1 + day_factor)
                current_data[pollutant] = max(0, current_data[pollutant])
                
        except Exception as e:
            safe_print(f"[Forecast 7d Error] Day {day}: {e}")
            continue

    aqi_values = [f['predicted_aqi'] for f in forecasts] or [50]
    weekday_values = [f['predicted_aqi'] for f in forecasts if not f['is_weekend']]
    weekend_values = [f['predicted_aqi'] for f in forecasts if f['is_weekend']]
    
    best_day_idx = aqi_values.index(min(aqi_values))
    worst_day_idx = aqi_values.index(max(aqi_values))
    
    weekly_statistics = {
        "average": int(np.mean(aqi_values)),
        "best_day": forecasts[best_day_idx]["day_name"],
        "worst_day": forecasts[worst_day_idx]["day_name"],
        "trend": "Improving" if aqi_values[-1] < aqi_values[0] else "Worsening" if aqi_values[-1] > aqi_values[0] else "Stable",
        "weekday_avg": int(np.mean(weekday_values)) if weekday_values else None,
        "weekend_avg": int(np.mean(weekend_values)) if weekend_values else None
    }
    
    return {
        "forecasts": forecasts,
        "weekly_statistics": weekly_statistics
    }

# Helper to save prediction record to DB
def save_prediction_to_db(db: Session, pollutants: dict, aqi_val: int, aqi_det: dict, forecast_type: str = "current"):
    try:
        record = PredictionRecord(
            timestamp=datetime.utcnow(),
            forecast_type=forecast_type,
            pm25=pollutants.get('PM2.5'),
            pm10=pollutants.get('PM10'),
            no2=pollutants.get('NO2'),
            so2=pollutants.get('SO2'),
            co=pollutants.get('CO'),
            o3=pollutants.get('O3'),
            predicted_aqi=aqi_val,
            level=aqi_det.get('level'),
            color=aqi_det.get('color'),
            emoji=aqi_det.get('emoji'),
            health_message=aqi_det.get('health_message')
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as e:
        db.rollback()
        safe_print(f"[DB Warning] Failed to save prediction: {e}")
        return None


# FastAPI Routes

@app.get('/', response_class=PlainTextResponse)
def home():
    return "FastAPI AQI Prediction Service is running!"

@app.get('/health', response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with database status check"""
    db_status = "MySQL connected" if is_mysql else "SQLite fallback connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "database_status": db_status,
        "server": "Enhanced AQI Prediction FastAPI v2.1"
    }

@app.get('/info', response_model=ApiInfoResponse)
def api_info(db: Session = Depends(get_db)):
    """API information endpoint"""
    return {
        "name": "Enhanced AQI Prediction API",
        "version": "2.1 (FastAPI)",
        "description": "Real-time AQI prediction with 24h and 7d forecasting powered by FastAPI and MySQL database",
        "endpoints": ["/predict", "/24hours", "/7days", "/history", "/health", "/info", "/docs"],
        "model_status": "loaded" if model is not None else "mock_mode",
        "database": "MySQL" if is_mysql else "SQLite (fallback)"
    }

@app.post('/predict', response_model=PredictApiResponse)
def predict_current(request: PredictRequest, db: Session = Depends(get_db)):
    """Current AQI prediction endpoint, logged in database"""
    try:
        pollutants = request.pollutants
        required_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in required_pollutants:
            if pollutant not in pollutants:
                raise HTTPException(status_code=400, detail=f"Missing pollutant: {pollutant}")
        
        current_time = datetime.now()
        aqi_prediction = predict_single_timestamp(model, preprocessor, feature_info, pollutants, current_time)
        aqi_details = get_aqi_details(aqi_prediction)
        
        save_prediction_to_db(db, pollutants, aqi_prediction, aqi_details, forecast_type="current")
        
        current_data = {
            'aqi': aqi_prediction,
            'pollutants': pollutants,
            'timestamp': current_time.isoformat()
        }
        plot_base64 = create_realistic_visualization(current_data, "current")
        
        prediction_response = {
            "aqi": aqi_prediction,
            "timestamp": current_time.isoformat(),
            **aqi_details
        }
        
        return {
            "prediction": prediction_response,
            "plot": plot_base64,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        safe_print(f"[Prediction Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/24hours', response_model=Forecast24HourResponse)
def predict_24_hours(request: PredictRequest, db: Session = Depends(get_db)):
    """24-hour forecast endpoint, summary saved to database"""
    try:
        pollutants = request.pollutants
        required_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in required_pollutants:
            if pollutant not in pollutants:
                raise HTTPException(status_code=400, detail=f"Missing pollutant: {pollutant}")
        
        current_time = datetime.now()
        forecast_data = forecast_24_hours(model, preprocessor, feature_info, pollutants, current_time)
        
        avg_aqi = int(forecast_data['statistics']['average'])
        avg_details = get_aqi_details(avg_aqi)
        save_prediction_to_db(db, pollutants, avg_aqi, avg_details, forecast_type="24hours")

        try:
            plot_base64 = create_realistic_visualization(forecast_data, "24hours")
            forecast_data['plot'] = plot_base64
        except Exception as viz_error:
            safe_print(f"[Viz Warning] {viz_error}")
            forecast_data['plot'] = ""
        
        return forecast_data
        
    except HTTPException:
        raise
    except Exception as e:
        safe_print(f"[24h Forecast Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/7days', response_model=Forecast7DayResponse)
def predict_7_days(request: PredictRequest, db: Session = Depends(get_db)):
    """7-day forecast endpoint, summary saved to database"""
    try:
        pollutants = request.pollutants
        required_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in required_pollutants:
            if pollutant not in pollutants:
                raise HTTPException(status_code=400, detail=f"Missing pollutant: {pollutant}")
        
        current_time = datetime.now()
        forecast_data = forecast_7_days(model, preprocessor, feature_info, pollutants, current_time)
        
        avg_aqi = int(forecast_data['weekly_statistics']['average'])
        avg_details = get_aqi_details(avg_aqi)
        save_prediction_to_db(db, pollutants, avg_aqi, avg_details, forecast_type="7days")

        try:
            plot_base64 = create_realistic_visualization(forecast_data, "7days")
            forecast_data['plot'] = plot_base64
        except Exception as viz_error:
            safe_print(f"[Viz Warning] {viz_error}")
            forecast_data['plot'] = ""
        
        return forecast_data
        
    except HTTPException:
        raise
    except Exception as e:
        safe_print(f"[7d Forecast Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/history')
def get_prediction_history(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    forecast_type: str | None = Query(None),
    db: Session = Depends(get_db)
):
    """Retrieve stored AQI prediction history from database."""
    try:
        query = db.query(PredictionRecord)
        if forecast_type:
            query = query.filter(PredictionRecord.forecast_type == forecast_type)
        
        total = query.count()
        records = query.order_by(PredictionRecord.timestamp.desc()).offset(skip).limit(limit).all()
        
        return {
            "total": total,
            "limit": limit,
            "skip": skip,
            "records": [rec.to_dict() for rec in records]
        }
    except Exception as e:
        safe_print(f"[History Query Error] {e}")
        return {
            "total": 0,
            "limit": limit,
            "skip": skip,
            "records": [],
            "error": str(e)
        }

if __name__ == '__main__':
    import uvicorn
    safe_print("Starting FastAPI server on http://127.0.0.1:8000")
    safe_print("Swagger API Documentation available at http://127.0.0.1:8000/docs")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)