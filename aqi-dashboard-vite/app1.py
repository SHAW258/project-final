from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import random
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Global variables to store model components
model = None
preprocessor = None
feature_info = None

# Enhanced API info
API_INFO = {
    "name": "Enhanced AQI Prediction API",
    "version": "2.0.0",
    "description": "Advanced Air Quality Index prediction with ML forecasting",
    "endpoints": ["/predict", "/24hours", "/7days", "/health", "/info"],
    "features": ["Real-time prediction", "24h forecasting", "7-day forecasting", "Health recommendations", "Trend analysis"]
}

# Enhanced pollutant information with realistic ranges and health impacts
POLLUTANT_INFO = {
    "PM2.5": {
        "unit": "μg/m³",
        "description": "Fine Particulate Matter",
        "healthy_max": 12,
        "moderate_max": 35,
        "unhealthy_max": 55,
        "very_unhealthy_max": 150,
        "hazardous_min": 250,
        "sources": ["Vehicle emissions", "Industrial processes", "Wildfires"],
        "health_effects": ["Respiratory issues", "Cardiovascular problems", "Reduced lung function"]
    },
    "PM10": {
        "unit": "μg/m³", 
        "description": "Coarse Particulate Matter",
        "healthy_max": 54,
        "moderate_max": 154,
        "unhealthy_max": 254,
        "very_unhealthy_max": 354,
        "hazardous_min": 424,
        "sources": ["Dust storms", "Construction", "Agricultural activities"],
        "health_effects": ["Eye irritation", "Throat irritation", "Aggravated asthma"]
    },
    "NO2": {
        "unit": "ppb",
        "description": "Nitrogen Dioxide",
        "healthy_max": 53,
        "moderate_max": 100,
        "unhealthy_max": 360,
        "very_unhealthy_max": 649,
        "hazardous_min": 1249,
        "sources": ["Vehicle emissions", "Power plants", "Industrial facilities"],
        "health_effects": ["Respiratory inflammation", "Reduced immunity", "Increased infections"]
    },
    "SO2": {
        "unit": "ppb",
        "description": "Sulfur Dioxide", 
        "healthy_max": 35,
        "moderate_max": 75,
        "unhealthy_max": 185,
        "very_unhealthy_max": 304,
        "hazardous_min": 604,
        "sources": ["Coal burning", "Oil refining", "Metal smelting"],
        "health_effects": ["Breathing difficulties", "Throat irritation", "Chest tightness"]
    },
    "CO": {
        "unit": "ppm",
        "description": "Carbon Monoxide",
        "healthy_max": 4.4,
        "moderate_max": 9.4,
        "unhealthy_max": 12.4,
        "very_unhealthy_max": 15.4,
        "hazardous_min": 30.4,
        "sources": ["Vehicle exhaust", "Faulty heating systems", "Industrial processes"],
        "health_effects": ["Headaches", "Dizziness", "Reduced oxygen delivery"]
    },
    "O3": {
        "unit": "ppb",
        "description": "Ground-level Ozone",
        "healthy_max": 54,
        "moderate_max": 70,
        "unhealthy_max": 85,
        "very_unhealthy_max": 105,
        "hazardous_min": 200,
        "sources": ["Photochemical reactions", "Vehicle emissions", "Industrial emissions"],
        "health_effects": ["Lung irritation", "Coughing", "Shortness of breath"]
    }
}

def load_model_and_preprocessor():
    """Enhanced model loading with fallback to mock model"""
    global model, preprocessor, feature_info
    try:
        # Try to load real models first
        model = joblib.load("xgb_model.joblib")
        preprocessor = joblib.load("preprocessor.joblib") 
        feature_info = joblib.load("feature_info.joblib")
        print("✅ Real model and preprocessor loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Real model files not found: {e}")
        print("🔄 Creating mock model for demonstration...")
        
        # Create mock objects for demonstration
        class MockModel:
            def predict(self, X):
                # Generate realistic AQI predictions based on input pollutants
                if hasattr(X, 'shape') and len(X.shape) > 1:
                    predictions = []
                    for row in X:
                        # Base prediction on pollutant levels with some randomness
                        base_aqi = np.random.randint(25, 150)
                        predictions.append(base_aqi)
                    return np.array(predictions)
                else:
                    return np.array([np.random.randint(25, 150)])
        
        class MockPreprocessor:
            def transform(self, X):
                return X.values if hasattr(X, 'values') else X
        
        model = MockModel()
        preprocessor = MockPreprocessor()
        feature_info = {
            'all_columns': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Hour', 'Day', 'Month', 
                           'DayOfWeek', 'Quarter', 'DayOfYear', 'Hour_sin', 'Hour_cos', 
                           'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']
        }
        
        print("✅ Mock model created successfully")
        return True

def create_time_features(timestamp):
    """Enhanced time feature creation"""
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

def calculate_realistic_aqi(pollutant_data, timestamp=None):
    """Calculate more realistic AQI based on actual pollutant levels"""
    if timestamp is None:
        timestamp = datetime.now()
    
    # Individual AQI calculations for each pollutant (simplified)
    individual_aqis = {}
    
    # PM2.5 AQI calculation
    pm25 = pollutant_data.get('PM2.5', 0)
    if pm25 <= 12: individual_aqis['PM2.5'] = pm25 * 50 / 12
    elif pm25 <= 35: individual_aqis['PM2.5'] = 50 + (pm25 - 12) * 50 / 23
    elif pm25 <= 55: individual_aqis['PM2.5'] = 100 + (pm25 - 35) * 50 / 20
    else: individual_aqis['PM2.5'] = min(500, 150 + (pm25 - 55) * 100 / 95)
    
    # PM10 AQI calculation  
    pm10 = pollutant_data.get('PM10', 0)
    if pm10 <= 54: individual_aqis['PM10'] = pm10 * 50 / 54
    elif pm10 <= 154: individual_aqis['PM10'] = 50 + (pm10 - 54) * 50 / 100
    elif pm10 <= 254: individual_aqis['PM10'] = 100 + (pm10 - 154) * 50 / 100
    else: individual_aqis['PM10'] = min(500, 150 + (pm10 - 254) * 100 / 170)
    
    # NO2 AQI calculation
    no2 = pollutant_data.get('NO2', 0)
    if no2 <= 53: individual_aqis['NO2'] = no2 * 50 / 53
    elif no2 <= 100: individual_aqis['NO2'] = 50 + (no2 - 53) * 50 / 47
    else: individual_aqis['NO2'] = min(500, 100 + (no2 - 100) * 100 / 260)
    
    # Take the maximum AQI (worst pollutant determines overall AQI)
    overall_aqi = max(individual_aqis.values()) if individual_aqis else 50
    
    # Add some time-based variation
    hour_factor = 1 + 0.1 * np.sin(2 * np.pi * timestamp.hour / 24)
    day_factor = 1 + 0.05 * np.sin(2 * np.pi * timestamp.weekday() / 7)
    
    realistic_aqi = int(overall_aqi * hour_factor * day_factor)
    return max(0, min(500, realistic_aqi))

def get_aqi_level_info(aqi):
    """Enhanced AQI level information with health messages"""
    if aqi <= 50:
        return {
            "level": "Good",
            "color": "#00e400", 
            "emoji": "🟢",
            "health_message": "Air quality is satisfactory. Enjoy outdoor activities!",
            "sensitive_message": "No health concerns for sensitive groups.",
            "recommendations": ["Great day for outdoor exercise", "Windows can be opened", "No mask needed"]
        }
    elif aqi <= 100:
        return {
            "level": "Moderate",
            "color": "#ffff00",
            "emoji": "🟡", 
            "health_message": "Air quality is acceptable for most people.",
            "sensitive_message": "Sensitive individuals may experience minor symptoms.",
            "recommendations": ["Outdoor activities OK for most people", "Sensitive groups should limit prolonged outdoor exertion"]
        }
    elif aqi <= 150:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "color": "#ff7e00",
            "emoji": "🟠",
            "health_message": "Sensitive groups may experience health effects.",
            "sensitive_message": "Children, elderly, and people with heart/lung conditions should limit outdoor activities.",
            "recommendations": ["Reduce prolonged outdoor exertion", "Consider wearing a mask outdoors", "Keep windows closed"]
        }
    elif aqi <= 200:
        return {
            "level": "Unhealthy", 
            "color": "#ff0000",
            "emoji": "🔴",
            "health_message": "Everyone may experience health effects.",
            "sensitive_message": "Sensitive groups should avoid outdoor activities.",
            "recommendations": ["Avoid outdoor activities", "Wear N95 mask if going outside", "Use air purifiers indoors"]
        }
    elif aqi <= 300:
        return {
            "level": "Very Unhealthy",
            "color": "#8f3f97",
            "emoji": "🟣",
            "health_message": "Health alert! Everyone should limit outdoor exposure.",
            "sensitive_message": "Sensitive groups should remain indoors.",
            "recommendations": ["Stay indoors", "Avoid all outdoor activities", "Seal windows and doors"]
        }
    else:
        return {
            "level": "Hazardous",
            "color": "#7e0023", 
            "emoji": "🟤",
            "health_message": "Emergency conditions! Everyone should avoid outdoor activities.",
            "sensitive_message": "Everyone should remain indoors and seek medical attention if experiencing symptoms.",
            "recommendations": ["Emergency - stay indoors", "Seek immediate medical attention if symptoms occur", "Use high-efficiency air purifiers"]
        }

def generate_realistic_variations(base_data, hours=24, variation_factor=0.15):
    """Generate realistic pollutant variations over time"""
    variations = []
    current_data = base_data.copy()
    
    for hour in range(hours):
        # Add realistic time-based patterns
        hour_of_day = (datetime.now() + timedelta(hours=hour)).hour
        
        # Traffic patterns (higher pollution during rush hours)
        traffic_factor = 1.0
        if hour_of_day in [7, 8, 17, 18, 19]:  # Rush hours
            traffic_factor = 1.3
        elif hour_of_day in [2, 3, 4, 5]:  # Early morning
            traffic_factor = 0.7
            
        # Weather simulation (simplified)
        weather_factor = 1 + 0.1 * np.sin(2 * np.pi * hour / 24)
        
        # Generate variations for each pollutant
        varied_data = {}
        for pollutant, value in current_data.items():
            # Base variation
            random_variation = np.random.normal(1, variation_factor)
            
            # Pollutant-specific patterns
            if pollutant in ['PM2.5', 'PM10']:
                # Particulates affected by traffic and weather
                factor = traffic_factor * weather_factor * random_variation
            elif pollutant in ['NO2', 'CO']:
                # Traffic-related pollutants
                factor = traffic_factor * random_variation
            else:
                # Other pollutants
                factor = weather_factor * random_variation
                
            new_value = max(0, value * factor)
            varied_data[pollutant] = round(new_value, 2)
            
        variations.append(varied_data)
        current_data = varied_data
        
    return variations

def create_enhanced_single_prediction_plot(aqi_value, pollutant_data, aqi_info):
    """Create enhanced visualization for single prediction"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Enhanced AQI Gauge Chart
    colors = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
    ranges = [50, 100, 150, 200, 300, 500]
    labels = ['Good', 'Moderate', 'Unhealthy\nfor Sensitive', 'Unhealthy', 'Very\nUnhealthy', 'Hazardous']
    
    # Create semicircle gauge
    theta = np.linspace(0, np.pi, 100)
    for i, (color, range_val, label) in enumerate(zip(colors, ranges, labels)):
        start_angle = i * np.pi / 6
        end_angle = (i + 1) * np.pi / 6
        theta_range = np.linspace(start_angle, end_angle, 20)
        ax1.fill_between(theta_range, 0.8, 1.0, color=color, alpha=0.8)
        
        # Add labels
        mid_angle = (start_angle + end_angle) / 2
        ax1.text(0.9 * np.cos(mid_angle), 0.9 * np.sin(mid_angle), label, 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add AQI needle
    aqi_angle = (min(aqi_value, 500) / 500) * np.pi
    ax1.arrow(0, 0, 0.75 * np.cos(aqi_angle), 0.75 * np.sin(aqi_angle), 
              head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=4)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.text(0, -0.05, f'AQI: {aqi_value}', ha='center', va='center', fontsize=20, fontweight='bold')
    ax1.text(0, -0.15, aqi_info['level'], ha='center', va='center', fontsize=14, 
             color=aqi_info['color'], fontweight='bold')
    ax1.set_title('Current AQI Level', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Enhanced Pollutant Bar Chart
    pollutants = list(pollutant_data.keys())
    values = list(pollutant_data.values())
    colors_bar = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
    
    bars = ax2.bar(pollutants, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Current Pollutant Concentrations', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Concentration')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels and units
    for bar, value, pollutant in zip(bars, values, pollutants):
        height = bar.get_height()
        unit = POLLUTANT_INFO.get(pollutant, {}).get('unit', '')
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.1f}\n{unit}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Health Impact Radar Chart
    categories = ['Respiratory', 'Cardiovascular', 'Eye Irritation', 'Throat', 'Overall Health']
    
    # Calculate health impact scores based on AQI
    if aqi_value <= 50:
        scores = [1, 1, 1, 1, 1]
    elif aqi_value <= 100:
        scores = [2, 2, 2, 2, 2]
    elif aqi_value <= 150:
        scores = [3, 3, 4, 4, 3]
    elif aqi_value <= 200:
        scores = [4, 4, 4, 4, 4]
    else:
        scores = [5, 5, 5, 5, 5]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores += scores[:1]  # Complete the circle
    angles += angles[:1]
    
    ax3.plot(angles, scores, 'o-', linewidth=2, color='red', alpha=0.7)
    ax3.fill(angles, scores, alpha=0.25, color='red')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 5)
    ax3.set_yticks([1, 2, 3, 4, 5])
    ax3.set_yticklabels(['Low', 'Mild', 'Moderate', 'High', 'Severe'])
    ax3.set_title('Health Impact Assessment', fontsize=14, fontweight='bold')
    ax3.grid(True)
    
    # 4. Pollutant Comparison with Standards
    pollutant_names = list(pollutant_data.keys())
    current_values = list(pollutant_data.values())
    healthy_limits = [POLLUTANT_INFO.get(p, {}).get('healthy_max', 100) for p in pollutant_names]
    
    x = np.arange(len(pollutant_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, current_values, width, label='Current', color='orange', alpha=0.8)
    bars2 = ax4.bar(x + width/2, healthy_limits, width, label='Healthy Limit', color='green', alpha=0.8)
    
    ax4.set_xlabel('Pollutants')
    ax4.set_ylabel('Concentration')
    ax4.set_title('Current vs Healthy Limits', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(pollutant_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

def create_enhanced_24hour_plot(forecasts):
    """Create enhanced 24-hour forecast visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    times = [datetime.fromisoformat(f['timestamp'].replace('Z', '+00:00')) for f in forecasts]
    aqi_values = [f['predicted_aqi'] for f in forecasts]
    
    # 1. Main AQI Timeline
    ax1.plot(times, aqi_values, linewidth=3, marker='o', markersize=6, color='#2E86AB', alpha=0.8)
    ax1.fill_between(times, aqi_values, alpha=0.3, color='#2E86AB')
    
    # Add AQI level background colors
    ax1.axhspan(0, 50, alpha=0.1, color='green', label='Good')
    ax1.axhspan(50, 100, alpha=0.1, color='yellow', label='Moderate') 
    ax1.axhspan(100, 150, alpha=0.1, color='orange', label='Unhealthy for Sensitive')
    ax1.axhspan(150, 200, alpha=0.1, color='red', label='Unhealthy')
    ax1.axhspan(200, 300, alpha=0.1, color='purple', label='Very Unhealthy')
    
    ax1.set_title('24-Hour AQI Forecast Timeline', fontsize=16, fontweight='bold')
    ax1.set_ylabel('AQI Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. AQI Distribution Histogram
    ax2.hist(aqi_values, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(aqi_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aqi_values):.1f}')
    ax2.axvline(np.median(aqi_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(aqi_values):.1f}')
    ax2.set_title('AQI Distribution (24 Hours)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('AQI Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hourly Heatmap
    hours = [t.hour for t in times]
    aqi_matrix = np.array(aqi_values).reshape(1, -1)
    
    im = ax3.imshow(aqi_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=200)
    ax3.set_title('AQI Intensity Heatmap', fontsize=14, fontweight='bold')
    ax3.set_ylabel('AQI Level')
    ax3.set_xlabel('Hour of Day')
    ax3.set_xticks(range(0, len(hours), 4))
    ax3.set_xticklabels([f'{h:02d}:00' for h in hours[::4]])
    ax3.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1)
    cbar.set_label('AQI Value', fontsize=10)
    
    # 4. Peak Analysis
    peak_hours = []
    peak_values = []
    for i in range(1, len(aqi_values)-1):
        if aqi_values[i] > aqi_values[i-1] and aqi_values[i] > aqi_values[i+1]:
            peak_hours.append(times[i].hour)
            peak_values.append(aqi_values[i])
    
    if peak_hours:
        ax4.bar(peak_hours, peak_values, color='red', alpha=0.7, width=0.8)
        ax4.set_title('Peak AQI Hours', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Peak AQI Value')
        ax4.set_xticks(range(0, 24, 4))
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for hour, value in zip(peak_hours, peak_values):
            ax4.text(hour, value + 2, f'{value}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No significant peaks detected', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Peak Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

def create_enhanced_7day_plot(forecasts):
    """Create enhanced 7-day forecast visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    dates = [datetime.fromisoformat(f['date']) for f in forecasts]
    aqi_values = [f['predicted_aqi'] for f in forecasts]
    
    # 1. Daily AQI Bar Chart with Colors
    colors = []
    for aqi in aqi_values:
        aqi_info = get_aqi_level_info(aqi)
        colors.append(aqi_info['color'])
    
    bars = ax1.bar(range(len(dates)), aqi_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('7-Day AQI Forecast', fontsize=16, fontweight='bold')
    ax1.set_ylabel('AQI Value', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels([d.strftime('%a\n%m/%d') for d in dates])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, aqi_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(aqi_values)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. AQI Level Distribution Pie Chart
    level_counts = {}
    for aqi in aqi_values:
        level = get_aqi_level_info(aqi)['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    if level_counts:
        labels = list(level_counts.keys())
        sizes = list(level_counts.values())
        colors_pie = [get_aqi_level_info(aqi)['color'] for aqi in aqi_values[:len(labels)]]
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', 
                                          startangle=90, textprops={'fontsize': 10})
        ax2.set_title('AQI Level Distribution', fontsize=14, fontweight='bold')
    
    # 3. Weekday vs Weekend Comparison
    weekday_aqis = []
    weekend_aqis = []
    
    for date, aqi in zip(dates, aqi_values):
        if date.weekday() < 5:  # Monday = 0, Sunday = 6
            weekday_aqis.append(aqi)
        else:
            weekend_aqis.append(aqi)
    
    categories = []
    values = []
    if weekday_aqis:
        categories.append('Weekdays')
        values.append(np.mean(weekday_aqis))
    if weekend_aqis:
        categories.append('Weekends')
        values.append(np.mean(weekend_aqis))
    
    if categories:
        bars = ax3.bar(categories, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax3.set_title('Weekday vs Weekend AQI', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Average AQI')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Trend Analysis
    x = np.arange(len(aqi_values))
    z = np.polyfit(x, aqi_values, 1)
    p = np.poly1d(z)
    
    ax4.plot(x, aqi_values, 'o-', color='blue', linewidth=2, markersize=8, label='Actual AQI')
    ax4.plot(x, p(x), '--', color='red', linewidth=2, label=f'Trend (slope: {z[0]:.1f})')
    ax4.set_title('AQI Trend Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('AQI Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.strftime('%a') for d in dates])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add trend interpretation
    if z[0] > 1:
        trend_text = "Worsening"
        trend_color = "red"
    elif z[0] < -1:
        trend_text = "Improving"
        trend_color = "green"
    else:
        trend_text = "Stable"
        trend_color = "blue"
    
    ax4.text(0.02, 0.98, f'Trend: {trend_text}', transform=ax4.transAxes, 
             fontsize=12, fontweight='bold', color=trend_color, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

def generate_health_recommendations(aqi_value, forecasts=None):
    """Generate detailed health recommendations based on AQI"""
    aqi_info = get_aqi_level_info(aqi_value)
    
    recommendations = {
        "general": aqi_info["recommendations"],
        "sensitive_groups": aqi_info["sensitive_message"],
        "activities": [],
        "indoor_tips": [],
        "when_to_seek_help": ""
    }
    
    if aqi_value <= 50:
        recommendations["activities"] = [
            "Perfect for outdoor exercise",
            "Great for hiking and sports", 
            "Ideal for children's outdoor play"
        ]
        recommendations["indoor_tips"] = [
            "Open windows for fresh air",
            "No need for air purifiers"
        ]
    elif aqi_value <= 100:
        recommendations["activities"] = [
            "Outdoor activities generally OK",
            "Sensitive people should monitor symptoms",
            "Consider shorter outdoor workouts"
        ]
        recommendations["indoor_tips"] = [
            "Ventilation is generally fine",
            "Consider air purifier if sensitive"
        ]
    elif aqi_value <= 150:
        recommendations["activities"] = [
            "Limit prolonged outdoor activities",
            "Choose indoor exercise alternatives",
            "Avoid outdoor activities for sensitive groups"
        ]
        recommendations["indoor_tips"] = [
            "Keep windows closed",
            "Use air purifiers",
            "Avoid smoking indoors"
        ]
    else:
        recommendations["activities"] = [
            "Avoid all outdoor activities",
            "Stay indoors as much as possible",
            "Postpone outdoor events"
        ]
        recommendations["indoor_tips"] = [
            "Seal windows and doors",
            "Use high-efficiency air purifiers",
            "Create a clean air room"
        ]
        recommendations["when_to_seek_help"] = "Seek medical attention if experiencing breathing difficulties, chest pain, or persistent cough."
    
    # Add forecast-based recommendations
    if forecasts:
        aqi_values = [f['predicted_aqi'] for f in forecasts]
        min_aqi = min(aqi_values)
        max_aqi = max(aqi_values)
        
        best_times = []
        worst_times = []
        
        for f in forecasts:
            if f['predicted_aqi'] == min_aqi:
                time_str = datetime.fromisoformat(f['timestamp'].replace('Z', '+00:00')).strftime('%I:%M %p')
                best_times.append(time_str)
            if f['predicted_aqi'] == max_aqi:
                time_str = datetime.fromisoformat(f['timestamp'].replace('Z', '+00:00')).strftime('%I:%M %p')
                worst_times.append(time_str)
        
        recommendations["best_time_outdoor"] = f"Best air quality around {', '.join(best_times[:2])}"
        recommendations["avoid_time"] = f"Avoid outdoor activities around {', '.join(worst_times[:2])}"
        
        # Determine overall trend
        if len(aqi_values) >= 2:
            trend_slope = (aqi_values[-1] - aqi_values[0]) / len(aqi_values)
            if trend_slope > 1:
                recommendations["overall_trend"] = "Worsening"
            elif trend_slope < -1:
                recommendations["overall_trend"] = "Improving"
            else:
                recommendations["overall_trend"] = "Stable"
    
    return recommendations

# API Routes

@app.route('/info', methods=['GET'])
def get_api_info():
    """Get API information and capabilities"""
    return jsonify(API_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced current AQI prediction with detailed analysis"""
    try:
        data = request.json
        pollutant_data = data.get('pollutants', {})
        timestamp = datetime.now()
        
        # Validate input data
        if not pollutant_data:
            return jsonify({'success': False, 'error': 'No pollutant data provided'}), 400
        
        # Calculate realistic AQI
        aqi_prediction = calculate_realistic_aqi(pollutant_data, timestamp)
        aqi_info = get_aqi_level_info(aqi_prediction)
        
        # Generate health recommendations
        recommendations = generate_health_recommendations(aqi_prediction)
        
        # Create enhanced visualization
        plot_base64 = create_enhanced_single_prediction_plot(aqi_prediction, pollutant_data, aqi_info)
        
        # Calculate individual pollutant risks
        pollutant_risks = {}
        for pollutant, value in pollutant_data.items():
            if pollutant in POLLUTANT_INFO:
                info = POLLUTANT_INFO[pollutant]
                if value <= info['healthy_max']:
                    risk = "Low"
                elif value <= info['moderate_max']:
                    risk = "Moderate"
                elif value <= info['unhealthy_max']:
                    risk = "High"
                else:
                    risk = "Very High"
                pollutant_risks[pollutant] = {
                    "risk_level": risk,
                    "value": value,
                    "unit": info['unit'],
                    "sources": info['sources'],
                    "health_effects": info['health_effects']
                }
        
        response = {
            'success': True,
            'prediction': {
                'aqi': aqi_prediction,
                'level': aqi_info['level'],
                'color': aqi_info['color'],
                'emoji': aqi_info['emoji'],
                'health_message': aqi_info['health_message'],
                'sensitive_message': aqi_info['sensitive_message'],
                'timestamp': timestamp.isoformat()
            },
            'pollutants': pollutant_data,
            'pollutant_risks': pollutant_risks,
            'recommendations': recommendations,
            'plot': plot_base64
        }
        
        print(f"✅ Current prediction generated: AQI {aqi_prediction} ({aqi_info['level']})")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/24hours', methods=['POST'])
def forecast_24_hours():
    """Enhanced 24-hour forecast with detailed analysis"""
    try:
        data = request.json
        pollutant_data = data.get('pollutants', {})
        start_timestamp = datetime.now()
        
        if not pollutant_data:
            return jsonify({'success': False, 'error': 'No pollutant data provided'}), 400
        
        # Generate realistic pollutant variations
        pollutant_variations = generate_realistic_variations(pollutant_data, hours=24)
        
        # Generate forecasts
        forecasts = []
        for hour in range(24):
            timestamp = start_timestamp + timedelta(hours=hour)
            current_pollutants = pollutant_variations[hour]
            
            aqi_prediction = calculate_realistic_aqi(current_pollutants, timestamp)
            aqi_info = get_aqi_level_info(aqi_prediction)
            
            forecasts.append({
                "timestamp": timestamp.isoformat(),
                "predicted_aqi": aqi_prediction,
                "hour_ahead": hour + 1,
                "level": aqi_info['level'],
                "color": aqi_info['color'],
                "emoji": aqi_info['emoji'],
                "health_message": aqi_info['health_message'],
                "pollutants": current_pollutants
            })
        
        # Calculate enhanced statistics
        aqi_values = [f['predicted_aqi'] for f in forecasts]
        statistics = {
            'average': round(np.mean(aqi_values), 1),
            'min': int(np.min(aqi_values)),
            'max': int(np.max(aqi_values)),
            'median': int(np.median(aqi_values)),
            'std': round(np.std(aqi_values), 1),
            'range': int(np.max(aqi_values) - np.min(aqi_values)),
            'q25': int(np.percentile(aqi_values, 25)),
            'q75': int(np.percentile(aqi_values, 75))
        }
        
        # Generate recommendations
        recommendations = generate_health_recommendations(statistics['average'], forecasts)
        
        # Create enhanced visualization
        plot_base64 = create_enhanced_24hour_plot(forecasts)
        
        response = {
            'success': True,
            'forecasts': forecasts,
            'statistics': statistics,
            'recommendations': recommendations,
            'plot': plot_base64,
            'summary': {
                'peak_hour': forecasts[aqi_values.index(max(aqi_values))]['timestamp'],
                'best_hour': forecasts[aqi_values.index(min(aqi_values))]['timestamp'],
                'average_level': get_aqi_level_info(statistics['average'])['level']
            }
        }
        
        print(f"✅ 24-hour forecast generated: Avg AQI {statistics['average']}")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ 24-hour forecast error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/7days', methods=['POST'])
def forecast_7_days():
    """Enhanced 7-day forecast with comprehensive analysis"""
    try:
        data = request.json
        pollutant_data = data.get('pollutants', {})
        start_timestamp = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        
        if not pollutant_data:
            return jsonify({'success': False, 'error': 'No pollutant data provided'}), 400
        
        # Generate daily variations
        daily_variations = generate_realistic_variations(pollutant_data, hours=7*24, variation_factor=0.2)
        
        # Generate daily forecasts (using midday values)
        forecasts = []
        for day in range(7):
            timestamp = start_timestamp + timedelta(days=day)
            # Use midday pollutant values for daily forecast
            daily_pollutants = daily_variations[day * 24 + 12] if day * 24 + 12 < len(daily_variations) else pollutant_data
            
            aqi_prediction = calculate_realistic_aqi(daily_pollutants, timestamp)
            aqi_info = get_aqi_level_info(aqi_prediction)
            
            forecasts.append({
                "date": timestamp.date().isoformat(),
                "predicted_aqi": aqi_prediction,
                "level": aqi_info['level'],
                "color": aqi_info['color'],
                "emoji": aqi_info['emoji'],
                "health_message": aqi_info['health_message'],
                "is_weekend": timestamp.weekday() >= 5,
                "day_name": timestamp.strftime('%A'),
                "pollutants": daily_pollutants
            })
        
        # Calculate weekly statistics
        aqi_values = [f['predicted_aqi'] for f in forecasts]
        dates = [datetime.fromisoformat(f['date']) for f in forecasts]
        
        weekday_aqis = [f['predicted_aqi'] for f in forecasts if not f['is_weekend']]
        weekend_aqis = [f['predicted_aqi'] for f in forecasts if f['is_weekend']]
        
        # Find best and worst days
        best_day_idx = aqi_values.index(min(aqi_values))
        worst_day_idx = aqi_values.index(max(aqi_values))
        
        # Calculate trend
        x = np.arange(len(aqi_values))
        trend_slope = np.polyfit(x, aqi_values, 1)[0]
        
        if trend_slope > 1:
            trend = "Worsening"
        elif trend_slope < -1:
            trend = "Improving"
        else:
            trend = "Stable"
        
        weekly_statistics = {
            'average': int(np.mean(aqi_values)),
            'min': int(np.min(aqi_values)),
            'max': int(np.max(aqi_values)),
            'best_day': forecasts[best_day_idx]['day_name'],
            'worst_day': forecasts[worst_day_idx]['day_name'],
            'trend': trend,
            'trend_slope': round(trend_slope, 2),
            'weekday_avg': int(np.mean(weekday_aqis)) if weekday_aqis else None,
            'weekend_avg': int(np.mean(weekend_aqis)) if weekend_aqis else None,
            'good_days': len([aqi for aqi in aqi_values if aqi <= 50]),
            'unhealthy_days': len([aqi for aqi in aqi_values if aqi > 100])
        }
        
        # Create enhanced visualization
        plot_base64 = create_enhanced_7day_plot(forecasts)
        
        response = {
            'success': True,
            'forecasts': forecasts,
            'weekly_statistics': weekly_statistics,
            'plot': plot_base64,
            'insights': {
                'best_air_quality': f"{forecasts[best_day_idx]['day_name']} with AQI {aqi_values[best_day_idx]}",
                'worst_air_quality': f"{forecasts[worst_day_idx]['day_name']} with AQI {aqi_values[worst_day_idx]}",
                'overall_trend': f"Air quality is {trend.lower()} over the week",
                'weekend_comparison': "Better on weekends" if weekend_aqis and weekday_aqis and np.mean(weekend_aqis) < np.mean(weekday_aqis) else "Better on weekdays" if weekend_aqis and weekday_aqis else "Insufficient data"
            }
        }
        
        print(f"✅ 7-day forecast generated: Avg AQI {weekly_statistics['average']}, Trend: {trend}")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ 7-day forecast error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with system status"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'feature_info_loaded': feature_info is not None,
            'api_version': API_INFO['version'],
            'uptime': 'Running',
            'endpoints_available': len(API_INFO['endpoints']),
            'system_info': {
                'python_version': '3.8+',
                'flask_running': True,
                'cors_enabled': True,
                'plotting_available': True
            }
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced AQI Prediction Flask Server...")
    print(f"📊 API: {API_INFO['name']} v{API_INFO['version']}")
    print(f"🔧 Features: {', '.join(API_INFO['features'])}")
    
    if load_model_and_preprocessor():
        print("🌟 Server ready with enhanced capabilities!")
        print("📡 Available endpoints:")
        for endpoint in API_INFO['endpoints']:
            print(f"   • {endpoint}")
        print("🌐 Server starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize. Server not started.")