import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# AQI breakpoints for all pollutants based on EPA guidelines

POLLUTANT_BREAKPOINTS = {
    'PM2.5': [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ],
    'PM10': [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ],
    'SO2': [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500)
    ],
    'NO2': [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500)
    ],
    'CO': [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500)
    ],
    'O3': [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
        (0.201, 0.300, 301, 400),
        (0.301, 0.400, 401, 500)
    ]
}


# Minimum detection limits for pollutants
MDL_VALUES = {
    'PM2.5': 10,    # Minimum detection limit for PM2.5
    'PM10': 20,     # Minimum detection limit for PM10
    'SO2': 4,       # Minimum detection limit for SO2
    'NO2': 5,       # Minimum detection limit for NO2
    'CO': 0.1,      # Minimum detection limit for CO (in mg/m³)
    'O3': 5         # Minimum detection limit for O3
}

def filter_valid_rows_cpcb(df, pollutant_columns):
    """
    Keep only rows that meet CPCB AQI calculation criteria:
    - At least 3 pollutant values are present (not null).
    - At least one of the pollutants is PM2.5 or PM10.
    """
    def is_valid(row):
        pollutants_available = sum(1 for col in pollutant_columns if col in row.index and pd.notna(row[col]))
        has_pm = ('PM2.5' in row.index and pd.notna(row['PM2.5'])) or ('PM10' in row.index and pd.notna(row['PM10']))
        return pollutants_available >= 3 and has_pm

    initial_count = len(df)
    df_filtered = df[df.apply(is_valid, axis=1)].copy()
    dropped_rows = initial_count - len(df_filtered)
    print(f"✅ CPCB filter applied: {dropped_rows} rows dropped, {len(df_filtered)} rows retained.")
    return df_filtered

def fill_missing_values(df, pollutant_columns):
    """
    Fill missing values according to CPCB guidelines:
    1. Forward and backward fill for time-series data
    2. Half of minimum detection limit for any remaining NaNs
    """
    # Make a copy to avoid modifying the original dataframe
    df_filled = df.copy()
    
    # First ensure all values are numeric
    for col in pollutant_columns:
        if col in df_filled.columns:
            df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')
    
    # Method 1: Forward fill and backward fill (for short gaps)
    for col in pollutant_columns:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].ffill().bfill()
    
    # Method 2: For any remaining NaN values, use minimum detection limits
    for col in pollutant_columns:
        if col in df_filled.columns and col in MDL_VALUES:
            # Fill remaining NaNs with half of MDL
            df_filled[col] = df_filled[col].fillna(MDL_VALUES[col] / 2)
    
    return df_filled

def calculate_sub_index(pollutant, concentration):
    """
    Calculate sub-index for a pollutant based on CPCB guidelines
    """
    if pd.isna(concentration) or concentration is None:
        return None
    
    # Convert concentration to float
    try:
        concentration = float(concentration)
    except (ValueError, TypeError):
        return None
    
    # Get breakpoints for the pollutant
    if pollutant not in POLLUTANT_BREAKPOINTS:
        return None
    
    # Calculate sub-index
    for bp_low, bp_high, index_low, index_high in POLLUTANT_BREAKPOINTS[pollutant]:
        if bp_low <= concentration <= bp_high:
            sub_index = ((index_high - index_low) / (bp_high - bp_low)) * (concentration - bp_low) + index_low
            return round(sub_index)
    
    # If concentration is higher than the highest breakpoint
    last_bp = POLLUTANT_BREAKPOINTS[pollutant][-1]
    if concentration > last_bp[1]:
        return last_bp[3]  # Return the highest AQI value
    
    return None

def calculate_aqi(row, pollutant_columns):
    """
    Calculate AQI based on sub-indices of all pollutants
    """
    sub_indices = {}
    max_aqi = None
    dominant_pollutant = None
    
    # Calculate sub-indices for each pollutant
    for pollutant in pollutant_columns:
        if pollutant in row.index and not pd.isna(row[pollutant]):
            try:
                # First convert to float to handle both numeric and string values
                value = float(row[pollutant])
                
                # Convert CO from μg/m³ to mg/m³ if needed
                if pollutant == 'CO' and value > 50:  # Likely in μg/m³ if > 50
                    value = value / 1000
                
                sub_index = calculate_sub_index(pollutant, value)
                sub_indices[pollutant] = sub_index
                
                if sub_index is not None:
                    if max_aqi is None or sub_index > max_aqi:
                        max_aqi = sub_index
                        dominant_pollutant = pollutant
            except (ValueError, TypeError) as e:
                print(f"Error processing {pollutant} value '{row[pollutant]}': {e}")
                sub_indices[pollutant] = None
    
    return max_aqi, dominant_pollutant, sub_indices

def get_aqi_category(aqi):
    """Get AQI category based on EPA AQI value"""
    if aqi is None:
        return "Insufficient Data"
    elif 0 <= aqi <= 50:
        return "Good"
    elif 51 <= aqi <= 100:
        return "Moderate"
    elif 101 <= aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= aqi <= 200:
        return "Unhealthy"
    elif 201 <= aqi <= 300:
        return "Very Unhealthy"
    elif 301 <= aqi <= 500:
        return "Hazardous"
    else:
        return "Out of Range"

def identify_pollutant_columns(df):
    """
    Identify pollutant columns in the dataframe using various methods
    """
    # Target pollutants according to CPCB
    target_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Method 1: Direct column match
    available_pollutants = [col for col in target_pollutants if col in df.columns]
    
    # Method 2: Case-insensitive matching
    if not available_pollutants:
        print("No standard pollutant columns found. Trying case-insensitive matching...")
        col_map = {}
        for target in target_pollutants:
            for col in df.columns:
                if col.upper() == target.upper() or col.replace(" ", "").upper() == target.upper():
                    col_map[col] = target
                    break
        
        if col_map:
            df = df.rename(columns=col_map)
            available_pollutants = [col for col in target_pollutants if col in df.columns]
            print(f"After case-insensitive matching, found: {', '.join(available_pollutants)}")
    
    # Method 3: Partial match (e.g., "PM2.5 (µg/m³)" matches "PM2.5")
    if not available_pollutants:
        print("Still searching for pollutant columns. Trying partial matching...")
        col_map = {}
        for target in target_pollutants:
            for col in df.columns:
                if target in col or target.replace(".", "") in col:
                    col_map[col] = target
                    break
        
        if col_map:
            df = df.rename(columns=col_map)
            available_pollutants = [col for col in target_pollutants if col in df.columns]
            print(f"After partial matching, found: {', '.join(available_pollutants)}")
    
    # Method 4: Manual column selection (interactive)
    if not available_pollutants and not AUTO_MODE:
        print("\nNo pollutant columns automatically identified. Please manually select:")
        print("Your dataset columns are:")
        for idx, col in enumerate(df.columns):
            print(f"  {idx+1}. {col}")
        
        pollutant_map = {}
        for pollutant in target_pollutants:
            try:
                response = input(f"Enter column number or name for {pollutant} (leave blank to skip): ")
                if not response:
                    continue
                    
                # Try as index
                if response.isdigit() and int(response) > 0 and int(response) <= len(df.columns):
                    idx = int(response) - 1
                    col_name = df.columns[idx]
                    pollutant_map[col_name] = pollutant
                # Try as column name
                elif response in df.columns:
                    pollutant_map[response] = pollutant
                else:
                    print(f"Column '{response}' not found, skipping {pollutant}.")
            except Exception as e:
                print(f"Error processing {pollutant}: {e}")
        
        # Rename columns based on manual mapping
        if pollutant_map:
            df = df.rename(columns=pollutant_map)
            available_pollutants = [col for col in target_pollutants if col in df.columns]
            print(f"After manual mapping, found: {', '.join(available_pollutants)}")
    
    return df, available_pollutants

def merge_datasets(file_paths):
    """
    Merge multiple Excel/CSV datasets into a single dataframe
    """
    all_dfs = []
    
    print(f"📂 Processing {len(file_paths)} files...")
    
    for file_path in file_paths:
        try:
            print(f"📄 Reading: {os.path.basename(file_path)}")
            
            # Determine file type and read accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                print(f"⚠️ Unsupported file format: {file_ext}. Skipping {file_path}")
                continue
                
            # Add source file column
            df['Source_File'] = os.path.basename(file_path)
            
            # Clean all column names - strip whitespace
            df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
            
            # Replace various forms of missing values with NaN
            df = df.replace(['NA', 'na', 'NaN', 'null', 'NULL', 'Null', 'None', 'none', 'N/A', 'n/a', ''], np.nan)
            
            all_dfs.append(df)
            print(f"✅ Added {len(df)} rows from {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    # Check if we have any dataframes to merge
    if not all_dfs:
        print("❌ No valid data files could be read.")
        return None
        
    # Determine merge strategy based on number of files
    if len(all_dfs) == 1:
        print("📊 Only one file processed, no merging needed.")
        return all_dfs[0]
    
    # First attempt: try to merge all datasets directly
    try:
        print("🔄 Attempting to merge all datasets...")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Successfully merged {len(all_dfs)} datasets with {len(merged_df)} total rows.")
        return merged_df
    except Exception as e:
        print(f"⚠️ Direct merge failed: {e}")
        print("🔄 Trying to align schemas before merging...")
    
    # Second attempt: try to find common columns
    common_columns = set.intersection(*[set(df.columns) for df in all_dfs])
    
    if not common_columns:
        print("❌ No common columns found between datasets.")
        
        # Last resort: just use the first file and warn the user
        print("⚠️ Using only the first file as a fallback. Other files will be ignored.")
        return all_dfs[0]
    
    # Use only common columns for merging
    print(f"🔄 Merging datasets using {len(common_columns)} common columns...")
    try:
        aligned_dfs = [df[list(common_columns)] for df in all_dfs]
        merged_df = pd.concat(aligned_dfs, ignore_index=True)
        print(f"✅ Successfully merged {len(all_dfs)} datasets with {len(merged_df)} total rows.")
        return merged_df
    except Exception as e:
        print(f"❌ Aligned merge failed: {e}")
        return all_dfs[0]  # Last resort fallback

def process_data(df, auto_mode=False):
    """
    Process the merged dataframe and calculate AQI
    """
    global AUTO_MODE
    AUTO_MODE = auto_mode
    
    try:
        # Identify pollutant columns
        df, available_pollutants = identify_pollutant_columns(df)
        
        if not available_pollutants:
            print("\n❌ ERROR: No pollutant columns could be identified for AQI calculation.")
            return None
        
        print(f"\n✅ Found pollutant columns: {', '.join(available_pollutants)}")
        
        # Apply CPCB row filtering criteria
        df = filter_valid_rows_cpcb(df, available_pollutants)
        
        # Fill missing values according to CPCB guidelines
        print("🔄 Filling missing values according to CPCB guidelines...")
        df_filled = fill_missing_values(df, available_pollutants)
        
        # Calculate AQI for each row
        print("📊 Calculating AQI...")
        aqi_results = []
        
        # Process row by row to better handle errors
        for idx, row in df_filled.iterrows():
            try:
                result = calculate_aqi(row, available_pollutants)
                aqi_results.append(result)
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                aqi_results.append((None, None, {p: None for p in available_pollutants}))
        
        # Extract results
        df_filled['AQI'] = [result[0] for result in aqi_results]
        df_filled['Dominant_Pollutant'] = [result[1] for result in aqi_results]
        
        # Add AQI category
        df_filled['AQI_Category'] = df_filled['AQI'].apply(get_aqi_category)
        
        print(f"\n✅ AQI calculation completed for {len(df_filled)} rows.")
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"Total records processed: {len(df_filled)}")
        print(f"Records with calculated AQI: {df_filled['AQI'].count()}")
        
        # Print AQI category distribution
        print("\n📊 AQI Category Distribution:")
        category_counts = df_filled['AQI_Category'].value_counts()
        for category, count in category_counts.items():
            print(f"{category}: {count}")
        
        # Print dominant pollutant distribution
        print("\n📊 Dominant Pollutant Distribution:")
        pollutant_counts = df_filled['Dominant_Pollutant'].value_counts()
        for pollutant, count in pollutant_counts.items():
            if pd.notna(pollutant):
                print(f"{pollutant}: {count}")
        
        return df_filled
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🌟 CPCB AQI Calculator with Multi-Dataset Merger 🌟")
    print("=================================================")
    
    print("\n📁 Data Source Selection:")
    print("1. Process a single file")
    print("2. Process multiple files (merged)")
    print("3. Process all Excel/CSV files in a directory")
    
    choice = input("\nEnter your choice (1-3): ")
    
    file_paths = []
    
    if choice == '1':
        # Single file processing
        file_path = input("\nEnter the path to your Excel/CSV file: ")
        if not os.path.exists(file_path):
            print(f"❌ Error: File '{file_path}' does not exist.")
            return
        file_paths = [file_path]
        
    elif choice == '2':
        # Multiple file processing
        print("\nEnter paths to your Excel/CSV files (one per line).")
        print("Enter an empty line when finished.")
        
        while True:
            file_path = input("File path (or empty to finish): ")
            if not file_path:
                break
                
            if not os.path.exists(file_path):
                print(f"❌ Warning: File '{file_path}' does not exist. Skipping.")
                continue
                
            file_paths.append(file_path)
            
        if not file_paths:
            print("❌ No valid files provided.")
            return
            
    elif choice == '3':
        # Process all files in a directory
        directory = input("\nEnter the directory path containing Excel/CSV files: ")
        
        if not os.path.isdir(directory):
            print(f"❌ Error: Directory '{directory}' does not exist.")
            return
            
        # Get all Excel and CSV files in the directory
        excel_files = glob.glob(os.path.join(directory, "*.xlsx")) + glob.glob(os.path.join(directory, "*.xls"))
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        file_paths = excel_files + csv_files
        
        if not file_paths:
            print(f"❌ No Excel or CSV files found in '{directory}'.")
            return
            
        print(f"📂 Found {len(file_paths)} Excel/CSV files in the directory.")
        
    else:
        print("❌ Invalid choice. Please run the program again.")
        return
    
    # Ask if automatic mode should be used
    auto_mode_input = input("\nRun in automatic mode? (y/n, default: n): ").lower()
    auto_mode = auto_mode_input == 'y'
    
    # Merge datasets
    merged_df = merge_datasets(file_paths)
    
    if merged_df is None:
        print("❌ Could not process data files.")
        return
        
    # Process the merged dataframe
    processed_df = process_data(merged_df, auto_mode)
    
    if processed_df is not None:
        # Create output file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"merged_aqi_results_{timestamp}.xlsx"
        
        # Ask user for output filename
        custom_filename = input(f"\nEnter output file name (default: {output_file}): ")
        if custom_filename:
            output_file = custom_filename
            # Add extension if not provided
            if not output_file.endswith(('.xlsx', '.xls')):
                output_file += '.xlsx'
        
        # Save updated DataFrame to Excel
        processed_df.to_excel(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()