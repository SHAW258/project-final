import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import calendar

print("Loading air quality dataset...")

# Step 1: Load data from Excel file
try:
    df = pd.read_excel('Output_Bucket.xlsx')  # Replace with your file path
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print("Columns found:", df.columns.tolist())
    
except FileNotFoundError:
    print("ERROR: Excel file not found. Please update the file path.")
    print("Make sure your file path is correct in the line:")
    print("df = pd.read_excel('your_air_quality_file.xlsx')")
    exit()

# Step 2: Data preprocessing
print("\nPreprocessing data...")

# Convert TIMESTAMP to datetime
df['TIMESTAMP'] = pd.to_datetime(df['Timestamp'])

# Extract year and month
df['Year'] = df['TIMESTAMP'].dt.year
df['Month'] = df['TIMESTAMP'].dt.month

# Display basic info
print(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
print(f"Years covered: {sorted(df['Year'].unique())}")
print(f"Total data points: {len(df)}")

# Step 3: Calculate monthly and yearly statistics
print("\nCalculating statistics...")

# Yearly averages
yearly_stats = df.groupby('Year').agg({
    'PM2.5': 'mean',
    'PM10': 'mean', 
    'CO': 'mean',
    'NO2': 'mean',
    'SO2': 'mean',
    'O3': 'mean',
    'AQI': 'mean'
}).round(2).reset_index()

# Monthly averages (across all years)
monthly_stats = df.groupby('Month').agg({
    'PM2.5': 'mean',
    'PM10': 'mean', 
    'CO': 'mean',
    'NO2': 'mean',
    'SO2': 'mean',
    'O3': 'mean',
    'AQI': 'mean'
}).round(2).reset_index()

print("Data processing complete!")

# Step 4: Define pollutant configurations
pollutant_configs = {
    'PM2.5': {
        'title': 'PM2.5',
        'ylabel': 'PM2.5 (μg/m³)',
        'color': '#E74C3C',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    },
    'PM10': {
        'title': 'PM10', 
        'ylabel': 'PM10 (μg/m³)',
        'color': '#3498DB',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    },
    'CO': {
        'title': 'CO',
        'ylabel': 'CO (ppm)',
        'color': '#F39C12',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    },
    'NO2': {
        'title': 'NO2',
        'ylabel': 'NO2 (ppb)',
        'color': '#9B59B6',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    },
    'SO2': {
        'title': 'SO2',
        'ylabel': 'SO2 (ppb)', 
        'color': '#1ABC9C',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    },
    'O3': {
        'title': 'O3',
        'ylabel': 'O3 (ppm)',
        'color': '#E67E22',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    },
    'AQI': {
        'title': 'AQI',
        'ylabel': 'AQI (unitless)',
        'color': '#8E44AD',
        'monthly_ylim': (0, None),
        'yearly_ylim': (0, None)
    }
}

def create_individual_charts(pollutant):
    """Create monthly and yearly charts for a specific pollutant"""
    
    config = pollutant_configs[pollutant]
    
    # Create figure with 1x2 subplots (monthly and yearly)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{config["title"]} Analysis - Monthly and Yearly Trends', 
                 fontsize=16, fontweight='bold')
    
    # === MONTHLY CHART ===
    months = monthly_stats['Month']
    monthly_values = monthly_stats[pollutant]
    
    # Use gradient colors for months
    monthly_colors = plt.cm.viridis(np.linspace(0, 1, len(months)))
    
    bars1 = ax1.bar(months, monthly_values, 
                    color=monthly_colors, alpha=0.8, 
                    edgecolor='white', linewidth=1)
    
    ax1.set_title(f'Monthly Mean Values of {config["title"]} ({config["ylabel"].split("(")[1]}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month', fontsize=11)
    ax1.set_ylabel(config['ylabel'], fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Set month labels
    month_names = [calendar.month_abbr[i] for i in months]
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names)
    
    # Set y-axis limits
    if config['monthly_ylim'][1] is None:
        ax1.set_ylim(config['monthly_ylim'][0], max(monthly_values) * 1.15)
    else:
        ax1.set_ylim(config['monthly_ylim'])
    
    # Add value labels on bars
    for bar, value in zip(bars1, monthly_values):
        label = f'{value:.3f}' if pollutant in ['CO', 'O3'] and value < 1 else f'{value:.1f}'
        ax1.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + max(monthly_values) * 0.02,
                label, ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # === YEARLY CHART ===
    years = yearly_stats['Year']
    yearly_values = yearly_stats[pollutant]
    
    # Use different color for each year
    yearly_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(years)]
    
    bars2 = ax2.bar(years, yearly_values, 
                    color=yearly_colors, alpha=0.8, 
                    edgecolor='white', linewidth=1)
    
    ax2.set_title(f'Yearly Mean Values of {config["title"]} ({config["ylabel"].split("(")[1]}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel(config['ylabel'], fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Set y-axis limits
    if config['yearly_ylim'][1] is None:
        ax2.set_ylim(config['yearly_ylim'][0], max(yearly_values) * 1.15)
    else:
        ax2.set_ylim(config['yearly_ylim'])
    
    # Add value labels on bars
    for bar, value in zip(bars2, yearly_values):
        label = f'{value:.3f}' if pollutant in ['CO', 'O3'] and value < 1 else f'{value:.1f}'
        ax2.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + max(yearly_values) * 0.02,
                label, ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Set x-axis ticks for years
    ax2.set_xticks(years)
    ax2.set_xticklabels(years)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for this pollutant
    print(f"\n{config['title']} Statistics:")
    print("="*50)
    print("Monthly averages:")
    for month, value in zip(months, monthly_values):
        month_name = calendar.month_name[month]
        print(f"  {month_name:10}: {value:8.2f} {config['ylabel'].split('(')[1]}")
    
    print("\nYearly averages:")
    for year, value in zip(years, yearly_values):
        print(f"  {year}: {value:8.2f} {config['ylabel'].split('(')[1]}")
    
    print(f"\nOverall Statistics:")
    print(f"  Minimum: {df[pollutant].min():.2f} {config['ylabel'].split('(')[1]}")
    print(f"  Maximum: {df[pollutant].max():.2f} {config['ylabel'].split('(')[1]}")
    print(f"  Mean: {df[pollutant].mean():.2f} {config['ylabel'].split('(')[1]}")
    print(f"  Std Dev: {df[pollutant].std():.2f} {config['ylabel'].split('(')[1]}")

def create_all_individual_charts():
    """Create individual charts for all pollutants"""
    
    pollutants = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'AQI']
    
    print(f"\nGenerating individual charts for {len(pollutants)} pollutants...")
    print("This will create 14 charts total (2 per pollutant)")
    
    for i, pollutant in enumerate(pollutants, 1):
        print(f"\n[{i}/{len(pollutants)}] Creating charts for {pollutant}...")
        create_individual_charts(pollutant)
    
    print(f"\n✅ All {len(pollutants)} pollutant chart sets generated successfully!")

def create_summary_statistics():
    """Create and display summary statistics"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total records: {len(df):,}")
    print(f"  Date range: {df['TIMESTAMP'].min().strftime('%Y-%m-%d')} to {df['TIMESTAMP'].min().strftime('%Y-%m-%d')}")
    print(f"  Years covered: {len(df['Year'].unique())} years ({min(df['Year'])}-{max(df['Year'])})")
    
    print(f"\nMonthly Patterns (averaged across all years):")
    monthly_display = monthly_stats.round(2)
    monthly_display['Month_Name'] = monthly_display['Month'].apply(lambda x: calendar.month_name[x])
    print(monthly_display[['Month_Name', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'AQI']].to_string(index=False))
    
    print(f"\nYearly Trends:")
    print(yearly_stats.to_string(index=False))

# Step 5: Execute all functions
print("\nStarting comprehensive pollutant analysis...")

# Create individual charts for each pollutant
create_all_individual_charts()

# Display summary statistics
create_summary_statistics()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("Generated:")
print("✓ 7 Monthly trend charts")
print("✓ 7 Yearly trend charts") 
print("✓ Detailed statistics for each pollutant")
print("✓ Comprehensive summary statistics")
print(f"✓ Analysis of {len(df):,} data points from {min(df['Year'])}-{max(df['Year'])}")