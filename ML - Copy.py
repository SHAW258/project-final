import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess
try:
    df = pd.read_excel("Output_Bucket.xlsx")
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Keep only the columns we see in the dataset
    expected_columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in expected_columns if col in df.columns]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following expected columns are missing: {missing_columns}")
    
    # Keep only relevant columns that exist in the dataframe
    df = df[available_columns]
    
    # Convert timestamp and extract features
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    
    # Drop rows with invalid timestamps
    invalid_timestamps = df["Timestamp"].isnull().sum()
    if invalid_timestamps > 0:
        print(f"Dropping {invalid_timestamps} rows with invalid timestamps")
        df = df.dropna(subset=["Timestamp"])
    
    # Extract more time features for better prediction
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Quarter"] = df["Timestamp"].dt.quarter
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    
    # Create cyclical features for time variables
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"]/24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"]/24)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"]/12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"]/12)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"]/31)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"]/31)
    
    # Now drop the timestamp column
    df.drop(columns=["Timestamp"], inplace=True)
    
    # Check for missing values in all columns
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Check for and handle missing target variable
    if "AQI" in df.columns and df["AQI"].isnull().sum() > 0:
        print(f"Warning: {df['AQI'].isnull().sum()} missing values in target variable. Dropping these rows.")
        df = df.dropna(subset=["AQI"])
    
    # Handle outliers in key pollutant variables using capping
    pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    for col in pollutant_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Capping {outliers} outliers in {col}")
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if "AQI" in numeric_cols:
        numeric_cols.remove("AQI")  # Remove target from features
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nUsing {len(numeric_cols)} numeric features: {numeric_cols}")
    print(f"Using {len(categorical_cols)} categorical features: {categorical_cols}")
    
    # Split data
    X = df.drop(columns=["AQI"])
    y = df["AQI"]
    
    # Save column information for future reference
    feature_info = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "all_columns": X.columns.tolist()
    }
    joblib.dump(feature_info, "feature_info.joblib")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols)
        ],
        remainder='drop'  # Drop any columns not explicitly included
    )
    
    # If we have categorical columns, add them to the pipeline
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor.transformers.append(('cat', categorical_transformer, categorical_cols))
    
    # Fit preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessor for future use
    joblib.dump(preprocessor, "preprocessor.joblib")
    
    # Train/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Optuna tuning
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=5)
        return -scores.mean()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)
    print("✅ Best XGBoost Params:", study.best_trial.params)
    
    # Train with early stopping
    best_params = study.best_trial.params
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
       
        verbose=True
    )
    
    # Save model
    joblib.dump(model, "xgb_model.joblib")
    
    # Evaluate with multiple metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Evaluation:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    # Feature importance analysis
    importances = model.feature_importances_
    
    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback for older scikit-learn versions
        numeric_features = numeric_cols
        if categorical_cols and len(categorical_cols) > 0:
            # Estimate the categorical features after one-hot encoding
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = []
            for i, col in enumerate(categorical_cols):
                cats = ohe.categories_[i]
                for cat in cats:
                    cat_features.append(f"{col}_{cat}")
            feature_names = numeric_features + cat_features
        else:
            feature_names = numeric_features
    
    # If the lengths don't match, use generic feature names
    if len(feature_names) != len(importances):
        print(f"Warning: Feature names mismatch. Using generic feature names.")
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n📊 Top 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # Save feature importance for future reference
    feature_importance.to_csv("feature_importance.csv", index=False)
    
except FileNotFoundError:
    print("Error: AQI.xlsx file not found. Please check file path.")
except Exception as e:
    print(f"An error occurred: {e}")