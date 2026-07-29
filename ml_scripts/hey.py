"""
Script to convert XGBoost model to TensorFlow for TensorFlow Lite conversion
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
import warnings
warnings.filterwarnings("ignore")

def load_xgboost_data():
    """Load the same data used for XGBoost training"""
    try:
        df = pd.read_excel("Output_Bucket.xlsx")
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Apply same preprocessing as XGBoost script
        expected_columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
        available_columns = [col for col in expected_columns if col in df.columns]
        df = df[available_columns]
        
        # Convert timestamp and extract features
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        
        # Extract time features
        df["Hour"] = df["Timestamp"].dt.hour
        df["Day"] = df["Timestamp"].dt.day
        df["Month"] = df["Timestamp"].dt.month
        df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
        df["Quarter"] = df["Timestamp"].dt.quarter
        df["DayOfYear"] = df["Timestamp"].dt.dayofyear
        
        # Create cyclical features
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"]/24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"]/24)
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"]/12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"]/12)
        df["Day_sin"] = np.sin(2 * np.pi * df["Day"]/31)
        df["Day_cos"] = np.cos(2 * np.pi * df["Day"]/31)
        
        df.drop(columns=["Timestamp"], inplace=True)
        
        # Handle missing values and outliers (same as XGBoost script)
        df = df.dropna(subset=["AQI"])
        
        # Handle outliers
        pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
        for col in pollutant_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_tensorflow_model(input_shape):
    """Create a TensorFlow model that mimics XGBoost behavior"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='linear')  # Regression output
    ])
    
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

    
    return model

def train_tensorflow_model():
    """Train TensorFlow model for AQI prediction"""
    print("🔄 Loading and preprocessing data...")
    df = load_xgboost_data()
    
    if df is None:
        print("❌ Failed to load data")
        return None, None, None
    
    # Prepare features and target
    X = df.drop(columns=["AQI"])
    y = df["AQI"]
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Feature names: {X.columns.tolist()}")
    
    # Create and train model
    model = create_tensorflow_model(X_train.shape[1])
    
    print("🚀 Training TensorFlow model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test MAE: {test_mae:.4f}")
    
    # Save model and scaler
    model.save("aqi_tensorflow_model.h5")
    joblib.dump(scaler, "tensorflow_scaler.joblib")
    joblib.dump(X.columns.tolist(), "feature_names.joblib")
    
    print("💾 Model saved as 'aqi_tensorflow_model.h5'")
    print("💾 Scaler saved as 'tensorflow_scaler.joblib'")
    
    return model, scaler, X.columns.tolist()

def convert_to_tflite():
    """Convert TensorFlow model to TensorFlow Lite"""
    try:
        print("🔄 Converting to TensorFlow Lite...")
        
        # Load the saved model
        model = keras.models.load_model("aqi_tensorflow_model.h5")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Use float16 quantization for smaller model size
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open("aqi_model.tflite", "wb") as f:
            f.write(tflite_model)
        
        print("✅ TensorFlow Lite model saved as 'aqi_model.tflite'")
        print(f"📏 Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model
    
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None

def test_tflite_model():
    """Test the TensorFlow Lite model"""
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="aqi_model.tflite")
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("📋 TFLite Model Details:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output type: {output_details[0]['dtype']}")
        
        # Test with sample data
        scaler = joblib.load("tensorflow_scaler.joblib")
        feature_names = joblib.load("feature_names.joblib")
        
        # Create sample input (typical pollutant values)
        sample_data = {
            'PM2.5': 25.0, 'PM10': 50.0, 'NO2': 15.0, 
            'SO2': 5.0, 'CO': 1.0, 'O3': 80.0,
            'Hour': 12, 'Day': 15, 'Month': 6, 'DayOfWeek': 1,
            'Quarter': 2, 'DayOfYear': 166,
            'Hour_sin': 0.0, 'Hour_cos': 1.0,
            'Month_sin': 0.5, 'Month_cos': 0.866,
            'Day_sin': 0.484, 'Day_cos': 0.875
        }
        
        # Prepare input
        input_array = np.array([[sample_data[col] for col in feature_names]], dtype=np.float32)
        input_scaled = scaler.transform(input_array).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        
        # Get prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_aqi = int(prediction[0][0])
        
        print(f"🔮 Sample prediction: AQI = {predicted_aqi}")
        
        return True
    
    except Exception as e:
        print(f"❌ TFLite test error: {e}")
        return False

if __name__ == "__main__":
    print("🌍 AQI TensorFlow Model Converter")
    print("=" * 50)
    
    # Train TensorFlow model
    model, scaler, feature_names = train_tensorflow_model()
    
    if model is not None:
        # Convert to TensorFlow Lite
        tflite_model = convert_to_tflite()
        
        if tflite_model is not None:
            # Test the TFLite model
            test_tflite_model()
            print("\n✅ Conversion completed successfully!")
            print("📱 Ready for Android integration!")
        else:
            print("❌ TensorFlow Lite conversion failed")
    else:
        print("❌ TensorFlow model training failed")
