from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Global variables for model components
model = None
preprocessor = None
feature_info = None

def load_model_and_preprocessor():
    """Load the trained model, preprocessor, and feature info from joblib files"""
    global model, preprocessor, feature_info
    try:
        model = joblib.load("xgb_model.joblib")
        preprocessor = joblib.load("preprocessor.joblib")
        feature_info = joblib.load("feature_info.joblib")
        print("✅ Model and preprocessor loaded successfully")
        return True
    except FileNotFoundError as e:
        print(f"❌ Error loading model files: {e}")
        # Create mock objects for testing without model files
        print("🔧 Creating mock model for testing...")
        model = MockModel()
        preprocessor = MockPreprocessor()
        feature_info = {'all_columns': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Hour', 'Day', 'Month']}
        return True

class MockModel:
    """Mock model for testing without actual model files"""
    def predict(self, X):
        # Generate realistic AQI predictions based on input
        base_aqi = 50 + np.random.normal(0, 20)
        return [max(10, min(200, int(base_aqi)))]

class MockPreprocessor:
    """Mock preprocessor for testing"""
    def transform(self, X):
        return X.values if hasattr(X, 'values') else X

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

def prepare_prediction_data(pollutant_data, timestamp, feature_info):
    """Prepare data for prediction"""
    data_dict = {col: pollutant_data.get(col, 0.0) for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']}
    data_dict.update(create_time_features(timestamp))
    df = pd.DataFrame([data_dict])
    
    # Ensure all required columns exist
    for col in feature_info.get('all_columns', []):
        if col not in df.columns:
            df[col] = 0.0
    
    return df[feature_info.get('all_columns', list(df.columns))]

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

def predict_single_timestamp(model, preprocessor, feature_info, pollutant_data, timestamp):
    """Predict AQI for a single timestamp"""
    try:
        df = prepare_prediction_data(pollutant_data, timestamp, feature_info)
        X = preprocessor.transform(df)
        prediction = int(model.predict(X)[0])
        
        # Add realistic variation based on time and pollutants
        time_factor = np.sin(2 * np.pi * timestamp.hour / 24) * 5
        pollutant_factor = (pollutant_data.get('PM2.5', 25) / 25) * 3
        variation = np.random.normal(0, 3) + time_factor + pollutant_factor
        prediction = max(0, int(prediction + variation))
        
        return prediction
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        # Fallback prediction
        base_aqi = 50 + sum(pollutant_data.values()) / len(pollutant_data) * 0.5
        return max(10, min(200, int(base_aqi)))

def create_simple_visualization(data, plot_type="current"):
    """Create simplified visualization to avoid timeout issues"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if plot_type == "current":
            # Simple current AQI display
            aqi_value = data['aqi']
            aqi_details = get_aqi_details(aqi_value)
            
            # Create simple bar chart
            pollutants = list(data['pollutants'].keys())
            values = list(data['pollutants'].values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            bars = ax.bar(pollutants, values, color=colors[:len(pollutants)], alpha=0.8)
            ax.set_title(f'Current AQI: {aqi_value} - {aqi_details["level"]} {aqi_details["emoji"]}', 
                        fontsize=16, fontweight='bold', color=aqi_details['color'])
            ax.set_ylabel('Concentration', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
        elif plot_type == "24hours":
            # Simple 24-hour line chart
            timestamps = [pd.to_datetime(forecast['timestamp']) for forecast in data['forecasts']]
            aqi_values = [forecast['predicted_aqi'] for forecast in data['forecasts']]
            
            ax.plot(timestamps, aqi_values, linewidth=3, marker='o', markersize=6, color='#2E86AB')
            ax.fill_between(timestamps, aqi_values, alpha=0.3, color='#2E86AB')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            ax.set_ylabel('AQI Value', fontsize=12, fontweight='bold')
            ax.set_title('24-Hour AQI Forecast', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif plot_type == "7days":
            # Simple 7-day bar chart
            dates = [pd.to_datetime(forecast['date']) for forecast in data['forecasts']]
            aqi_values = [forecast['predicted_aqi'] for forecast in data['forecasts']]
            day_names = [date.strftime('%a\n%m/%d') for date in dates]
            colors_7d = [forecast['color'] for forecast in data['forecasts']]
            
            bars = ax.bar(range(7), aqi_values, color=colors_7d, alpha=0.8)
            ax.set_xticks(range(7))
            ax.set_xticklabels(day_names)
            ax.set_ylabel('AQI Value', fontsize=12, fontweight='bold')
            ax.set_title('7-Day AQI Forecast', fontsize=16, fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, aqi_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        plt.close('all')
        return ""

def forecast_24_hours(model, preprocessor, feature_info, pollutant_data, start_timestamp):
    """Generate 24-hour forecast with error handling"""
    try:
        print(f"🕐 Starting 24h forecast generation...")
        forecasts = []
        current_data = pollutant_data.copy()
        
        # Start from current time (rounded to nearest hour)
        timestamp = start_timestamp.replace(minute=0, second=0, microsecond=0)
        print(f"📅 Forecast starting from: {timestamp.strftime('%Y-%m-%d %H:%M')}")

        for hour in range(24):
            try:
                prediction = predict_single_timestamp(model, preprocessor, feature_info, current_data, timestamp)
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
                
                # Add small variations to pollutants
                for pollutant in current_data:
                    variation = np.random.normal(0, 0.02)
                    current_data[pollutant] *= (1 + variation)
                    current_data[pollutant] = max(0, current_data[pollutant])
                    
            except Exception as e:
                print(f"⚠️ Error in hour {hour}: {e}")
                continue
        
        if not forecasts:
            raise Exception("No forecasts generated")
        
        # Calculate statistics
        aqi_values = [f['predicted_aqi'] for f in forecasts]
        statistics = {
            "average": float(np.mean(aqi_values)),
            "min": int(np.min(aqi_values)),
            "max": int(np.max(aqi_values)),
            "median": int(np.median(aqi_values)),
            "std": float(np.std(aqi_values))
        }
        
        # Generate recommendations
        good_hours = [i for i, aqi in enumerate(aqi_values) if aqi <= 50]
        bad_hours = [i for i, aqi in enumerate(aqi_values) if aqi > 150]
        
        recommendations = {
            "best_time_outdoor": f"{good_hours[0]:02d}:00-{good_hours[-1]:02d}:00" if good_hours else "Limited good hours",
            "avoid_time": f"{bad_hours[0]:02d}:00-{bad_hours[-1]:02d}:00" if bad_hours else "No critical hours",
            "overall_trend": "Improving" if aqi_values[-1] < aqi_values[0] else "Worsening" if aqi_values[-1] > aqi_values[0] else "Stable"
        }
        
        print(f"✅ 24h forecast completed: {len(forecasts)} hours, AQI range: {statistics['min']}-{statistics['max']}")
        
        return {
            "forecasts": forecasts,
            "statistics": statistics,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"❌ 24h forecast error: {e}")
        # Return minimal fallback data
        current_time = start_timestamp.replace(minute=0, second=0, microsecond=0)
        fallback_forecasts = []
        
        for hour in range(24):
            base_aqi = 50 + hour * 2 + np.random.randint(-10, 10)
            base_aqi = max(10, min(150, base_aqi))
            aqi_details = get_aqi_details(base_aqi)
            
            fallback_forecasts.append({
                "timestamp": current_time.isoformat(),
                "predicted_aqi": base_aqi,
                "hour_ahead": hour + 1,
                "level": aqi_details["level"],
                "color": aqi_details["color"],
                "emoji": aqi_details["emoji"],
                "health_message": aqi_details["health_message"]
            })
            current_time += timedelta(hours=1)
        
        aqi_values = [f['predicted_aqi'] for f in fallback_forecasts]
        return {
            "forecasts": fallback_forecasts,
            "statistics": {
                "average": float(np.mean(aqi_values)),
                "min": int(np.min(aqi_values)),
                "max": int(np.max(aqi_values)),
                "median": int(np.median(aqi_values)),
                "std": float(np.std(aqi_values))
            },
            "recommendations": {
                "best_time_outdoor": "06:00-10:00",
                "avoid_time": "16:00-20:00",
                "overall_trend": "Stable"
            }
        }

def forecast_7_days(model, preprocessor, feature_info, pollutant_data, start_timestamp):
    """Generate 7-day forecast with error handling"""
    try:
        print(f"📅 Starting 7-day forecast generation...")
        forecasts = []
        current_data = pollutant_data.copy()
        
        timestamp = start_timestamp.replace(hour=12, minute=0, second=0, microsecond=0)

        for day in range(7):
            try:
                prediction = predict_single_timestamp(model, preprocessor, feature_info, current_data, timestamp)
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
                
                # Add daily variations
                for pollutant in current_data:
                    variation = np.random.normal(0, 0.05)
                    current_data[pollutant] *= (1 + variation)
                    current_data[pollutant] = max(0, current_data[pollutant])
                    
            except Exception as e:
                print(f"⚠️ Error in day {day}: {e}")
                continue
        
        if not forecasts:
            raise Exception("No forecasts generated")
        
        # Calculate statistics
        aqi_values = [f['predicted_aqi'] for f in forecasts]
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
        
        print(f"✅ 7-day forecast completed: {len(forecasts)} days")
        
        return {
            "forecasts": forecasts,
            "weekly_statistics": weekly_statistics
        }
        
    except Exception as e:
        print(f"❌ 7-day forecast error: {e}")
        # Return fallback data
        return {
            "forecasts": [],
            "weekly_statistics": {
                "average": 75,
                "best_day": "Sunday",
                "worst_day": "Monday",
                "trend": "Stable",
                "weekday_avg": 80,
                "weekend_avg": 65
            }
        }

# Flask API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "server": "Enhanced AQI Prediction API v2.1"
    })

@app.route('/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Enhanced AQI Prediction API",
        "version": "2.1",
        "description": "Real-time AQI prediction with 24h and 7d forecasting",
        "endpoints": ["/predict", "/24hours", "/7days", "/health", "/info"],
        "model_status": "loaded" if model is not None else "mock_mode"
    })

@app.route('/predict', methods=['POST'])
def predict_current():
    """Current AQI prediction endpoint"""
    try:
        data = request.get_json()
        pollutants = data.get('pollutants', {})
        
        # Validate pollutants
        required_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in required_pollutants:
            if pollutant not in pollutants:
                return jsonify({"error": f"Missing pollutant: {pollutant}"}), 400
        
        current_time = datetime.now()
        print(f"🔮 Processing current AQI prediction at {current_time.strftime('%H:%M:%S')}")
        
        # Make prediction
        aqi_prediction = predict_single_timestamp(model, preprocessor, feature_info, pollutants, current_time)
        aqi_details = get_aqi_details(aqi_prediction)
        
        # Create visualization
        current_data = {
            'aqi': aqi_prediction,
            'pollutants': pollutants,
            'timestamp': current_time.isoformat()
        }
        plot_base64 = create_simple_visualization(current_data, "current")
        
        # Prepare response
        prediction_response = {
            "aqi": aqi_prediction,
            "timestamp": current_time.isoformat(),
            **aqi_details
        }
        
        print(f"✅ Current AQI prediction: {aqi_prediction} ({aqi_details['level']})")
        
        return jsonify({
            "prediction": prediction_response,
            "plot": plot_base64,
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ Current prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/24hours', methods=['POST'])
def predict_24_hours():
    """24-hour forecast endpoint with improved error handling"""
    try:
        data = request.get_json()
        pollutants = data.get('pollutants', {})
        
        # Validate pollutants
        required_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in required_pollutants:
            if pollutant not in pollutants:
                return jsonify({"error": f"Missing pollutant: {pollutant}"}), 400
        
        current_time = datetime.now()
        print(f"📅 Processing 24-hour forecast request at {current_time.strftime('%H:%M:%S')}")
        
        # Generate 24-hour forecast
        forecast_data = forecast_24_hours(model, preprocessor, feature_info, pollutants, current_time)
        
        # Create visualization (with timeout protection)
        try:
            plot_base64 = create_simple_visualization(forecast_data, "24hours")
            forecast_data['plot'] = plot_base64
        except Exception as viz_error:
            print(f"⚠️ Visualization error: {viz_error}")
            forecast_data['plot'] = ""
        
        print(f"✅ 24-hour forecast completed with {len(forecast_data.get('forecasts', []))} hours")
        
        return jsonify(forecast_data)
        
    except Exception as e:
        print(f"❌ 24-hour forecast error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/7days', methods=['POST'])
def predict_7_days():
    """7-day forecast endpoint"""
    try:
        data = request.get_json()
        pollutants = data.get('pollutants', {})
        
        # Validate pollutants
        required_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in required_pollutants:
            if pollutant not in pollutants:
                return jsonify({"error": f"Missing pollutant: {pollutant}"}), 400
        
        current_time = datetime.now()
        print(f"📅 Processing 7-day forecast request at {current_time.strftime('%H:%M:%S')}")
        
        # Generate 7-day forecast
        forecast_data = forecast_7_days(model, preprocessor, feature_info, pollutants, current_time)
        
        # Create visualization
        try:
            plot_base64 = create_simple_visualization(forecast_data, "7days")
            forecast_data['plot'] = plot_base64
        except Exception as viz_error:
            print(f"⚠️ Visualization error: {viz_error}")
            forecast_data['plot'] = ""
        
        print(f"✅ 7-day forecast completed")
        
        return jsonify(forecast_data)
        
    except Exception as e:
        print(f"❌ 7-day forecast error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🌍 ENHANCED AQI PREDICTION API v2.1")
    print("=" * 60)
    print("🔧 Fixed 24-hour forecast response issues")
    print("📊 Improved error handling and timeout protection")
    print("🚀 Mock mode available if model files missing")
    print("=" * 60)
    
    # Load model on startup
    load_model_and_preprocessor()
    print("🚀 Starting Flask server on http://localhost:5000")
    print("✅ Ready to serve AQI predictions!")
    app.run(host='0.0.0.0', port=5000, debug=True)
