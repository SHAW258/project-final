import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

# Load your trained model and feature information
try:
    model = joblib.load("xgb_model.joblib")
    feature_info = joblib.load("feature_info.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    print("✅ Model and feature information loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model files: {e}")
    print("Please ensure you have run your ML.py script first to generate the model files.")
    exit()

# Load and preprocess data (same as your ML.py script)
try:
    df = pd.read_excel("Output_Bucket.xlsx")
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Keep only the columns we see in the dataset
    expected_columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    available_columns = [col for col in expected_columns if col in df.columns]
    df = df[available_columns]
    
    # Convert timestamp and extract features
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    
    # Extract time features (same as ML.py)
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
    df = df.dropna(subset=["AQI"])
    
    # Handle outliers (same as ML.py)
    pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    for col in pollutant_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Split data (same random state as ML.py)
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["AQI"])
    y = df["AQI"]
    
    X_processed = preprocessor.transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"✅ Data preprocessed successfully. Test set size: {len(X_test)}")
    
except Exception as e:
    print(f"❌ Error loading/preprocessing data: {e}")
    exit()

# Get feature names after preprocessing
def get_feature_names():
    """Get feature names after preprocessing"""
    try:
        # Try to get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            return preprocessor.get_feature_names_out()
        else:
            # Fallback method for older scikit-learn versions
            numeric_cols = feature_info['numeric_columns']
            categorical_cols = feature_info.get('categorical_columns', [])
            
            feature_names = numeric_cols.copy()
            
            # Add categorical feature names if they exist
            if categorical_cols:
                try:
                    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                    for i, col in enumerate(categorical_cols):
                        if hasattr(ohe, 'categories_') and i < len(ohe.categories_):
                            cats = ohe.categories_[i]
                            for cat in cats:
                                feature_names.append(f"{col}_{cat}")
                except:
                    pass
            
            return feature_names
    except Exception as e:
        print(f"Warning: Could not get feature names: {e}")
        # Return generic feature names
        n_features = X_processed.shape[1]
        return [f"feature_{i}" for i in range(n_features)]

feature_names = get_feature_names()

# Ensure feature names match the number of features
if len(feature_names) != X_processed.shape[1]:
    print(f"Warning: Feature names mismatch. Expected {X_processed.shape[1]}, got {len(feature_names)}")
    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

print(f"✅ Using {len(feature_names)} features")

# Get XGBoost built-in feature importance
xgb_importance = model.feature_importances_

# Calculate permutation importance (more reliable)
print("🔄 Calculating permutation importance (this may take a moment)...")
perm_importance = permutation_importance(model, X_test, y_test, 
                                       n_repeats=10, random_state=42, 
                                       scoring='neg_mean_absolute_error')

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'XGB_Importance': xgb_importance,
    'Perm_Importance_Mean': perm_importance.importances_mean,
    'Perm_Importance_Std': perm_importance.importances_std
})

# Sort by permutation importance
importance_df = importance_df.sort_values('Perm_Importance_Mean', ascending=True)

# Clean feature names for better display
def clean_feature_name(name):
    """Clean feature names for better readability"""
    # Remove 'num__' prefix if present
    if name.startswith('num__'):
        name = name[5:]
    
    # Replace underscores with spaces and capitalize
    name = name.replace('_', ' ').title()
    
    # Special cases for better readability
    replacements = {
        'Pm2.5': 'PM2.5',
        'Pm10': 'PM10',
        'No2': 'NO2',
        'So2': 'SO2',
        'Co': 'CO',
        'O3': 'O3',
        'Dayofweek': 'Day of Week',
        'Dayofyear': 'Day of Year',
        'Sin': '(sin)',
        'Cos': '(cos)'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name

importance_df['Feature_Clean'] = importance_df['Feature'].apply(clean_feature_name)

# Create comprehensive feature importance visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 1. Top 15 Permutation Importance (Horizontal Bar Plot)
ax1 = axes[0, 0]
top_15_perm = importance_df.tail(15)
bars1 = ax1.barh(range(len(top_15_perm)), top_15_perm['Perm_Importance_Mean'], 
                 xerr=top_15_perm['Perm_Importance_Std'],
                 color='skyblue', edgecolor='navy', alpha=0.7)
ax1.set_yticks(range(len(top_15_perm)))
ax1.set_yticklabels(top_15_perm['Feature_Clean'], fontsize=10)
ax1.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Features - Permutation Importance\n(with Standard Deviation)', 
              fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, val, std) in enumerate(zip(bars1, top_15_perm['Perm_Importance_Mean'], 
                                       top_15_perm['Perm_Importance_Std'])):
    ax1.text(bar.get_width() + std + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

# 2. Top 15 XGBoost Built-in Importance
ax2 = axes[0, 1]
importance_df_xgb = importance_df.sort_values('XGB_Importance', ascending=True)
top_15_xgb = importance_df_xgb.tail(15)
bars2 = ax2.barh(range(len(top_15_xgb)), top_15_xgb['XGB_Importance'], 
                 color='lightcoral', edgecolor='darkred', alpha=0.7)
ax2.set_yticks(range(len(top_15_xgb)))
ax2.set_yticklabels(top_15_xgb['Feature_Clean'], fontsize=10)
ax2.set_xlabel('XGBoost Feature Importance', fontsize=12, fontweight='bold')
ax2.set_title('Top 15 Features - XGBoost Built-in Importance', 
              fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, top_15_xgb['XGB_Importance'])):
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

# 3. Comparison of Both Importance Measures (Scatter Plot)
ax3 = axes[1, 0]
scatter = ax3.scatter(importance_df['XGB_Importance'], importance_df['Perm_Importance_Mean'], 
                     alpha=0.6, s=60, c='green', edgecolors='black', linewidth=0.5)
ax3.set_xlabel('XGBoost Feature Importance', fontsize=12, fontweight='bold')
ax3.set_ylabel('Permutation Importance', fontsize=12, fontweight='bold')
ax3.set_title('XGBoost vs Permutation Importance\nComparison', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add correlation coefficient
correlation = importance_df['XGB_Importance'].corr(importance_df['Perm_Importance_Mean'])
ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
         fontsize=11, fontweight='bold')

# Add trend line
z = np.polyfit(importance_df['XGB_Importance'], importance_df['Perm_Importance_Mean'], 1)
p = np.poly1d(z)
ax3.plot(importance_df['XGB_Importance'], p(importance_df['XGB_Importance']), 
         "r--", alpha=0.8, linewidth=2)

# 4. Feature Categories Analysis
ax4 = axes[1, 1]

# Categorize features
def categorize_feature(feature_name):
    """Categorize features into groups"""
    feature_lower = feature_name.lower()
    if any(pollutant in feature_lower for pollutant in ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3']):
        return 'Air Pollutants'
    elif any(time_feat in feature_lower for time_feat in ['hour', 'day', 'month', 'quarter']):
        return 'Time Features'
    elif any(cyclic in feature_lower for cyclic in ['sin', 'cos']):
        return 'Cyclical Features'
    else:
        return 'Other'

importance_df['Category'] = importance_df['Feature'].apply(categorize_feature)

# Calculate average importance by category
category_importance = importance_df.groupby('Category').agg({
    'Perm_Importance_Mean': ['mean', 'sum', 'count']
}).round(4)

category_importance.columns = ['Mean_Importance', 'Total_Importance', 'Feature_Count']
category_importance = category_importance.sort_values('Mean_Importance', ascending=True)

# Create category bar plot
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
bars4 = ax4.barh(range(len(category_importance)), category_importance['Mean_Importance'], 
                 color=colors[:len(category_importance)], alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(category_importance)))
ax4.set_yticklabels(category_importance.index, fontsize=11)
ax4.set_xlabel('Average Permutation Importance', fontsize=12, fontweight='bold')
ax4.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Add value and count labels
for i, (bar, val, count) in enumerate(zip(bars4, category_importance['Mean_Importance'], 
                                         category_importance['Feature_Count'])):
    ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f} (n={count})', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('aqi_feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed feature importance analysis
print("\n" + "="*80)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print(f"\n🔝 Top 10 Most Important Features (Permutation Importance):")
print("-" * 60)
top_10_features = importance_df.tail(10)
for i, (_, row) in enumerate(top_10_features.iterrows(), 1):
    print(f"{i:2d}. {row['Feature_Clean']:25} | {row['Perm_Importance_Mean']:8.4f} ± {row['Perm_Importance_Std']:6.4f}")

print(f"\n📈 Feature Category Analysis:")
print("-" * 60)
for category, data in category_importance.iterrows():
    print(f"{category:20} | Avg: {data['Mean_Importance']:7.4f} | Total: {data['Total_Importance']:7.4f} | Count: {data['Feature_Count']:2.0f}")

print(f"\n🔗 Correlation between XGBoost and Permutation Importance: {correlation:.4f}")

# Save detailed results
detailed_results = importance_df.sort_values('Perm_Importance_Mean', ascending=False)
detailed_results.to_csv('aqi_feature_importance_detailed.csv', index=False)

# Save category analysis
category_importance.to_csv('aqi_feature_importance_by_category.csv')

print(f"\n💾 Detailed results saved to:")
print(f"   - aqi_feature_importance_detailed.csv")
print(f"   - aqi_feature_importance_by_category.csv")

# Create a simple top features plot for presentations
plt.figure(figsize=(12, 8))
top_10 = importance_df.tail(10)
bars = plt.barh(range(len(top_10)), top_10['Perm_Importance_Mean'], 
                xerr=top_10['Perm_Importance_Std'],
                color='steelblue', alpha=0.8, edgecolor='navy')

plt.yticks(range(len(top_10)), top_10['Feature_Clean'], fontsize=12)
plt.xlabel('Permutation Importance', fontsize=14, fontweight='bold')
plt.title('Top 10 Most Important Features for AQI Prediction', 
          fontsize=16, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val, std in zip(bars, top_10['Perm_Importance_Mean'], top_10['Perm_Importance_Std']):
    plt.text(bar.get_width() + std + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('aqi_top_10_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Feature importance analysis complete!")
print("📊 Generated plots:")
print("   - aqi_feature_importance_comprehensive.png (4-panel detailed analysis)")
print("   - aqi_top_10_feature_importance.png (simple top 10 features)")