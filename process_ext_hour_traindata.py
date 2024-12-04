import pandas as pd
import os

# 提取特徵
def process_hourly_radiation_data(file_path):
    radiation_data = {}
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    for _, row in df.iterrows():
        date = pd.to_datetime(row['日期'])
        month = date.month
        day = date.day
        hour = row['觀測時間(hour)']
        radiation_data[(month, day, hour)] = row['全天空日射量(MJ/㎡)']
    return radiation_data

# Process data for first location (L1-L14)
external_file_1 = "data/external_data/external_daily/C0Z100/combined_csv.csv"
radiation_data_1 = process_hourly_radiation_data(external_file_1)

# Process data for second location (L15-L17)
external_file_2 = "data/external_data/external_daily/466990/combined_csv.csv"
radiation_data_2 = process_hourly_radiation_data(external_file_2)

# Process training files
train_dir = "data/train_data_processed"

# 新增特徵到訓練資料
for i in range(1, 18):
    filename = f"L{i}_Train_combined_resampled_10T_processed.csv"
    file_path = os.path.join(train_dir, filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Select appropriate radiation data based on location
        radiation_data = radiation_data_1 if i <= 14 else radiation_data_2
        
        # Add hourly radiation data
        radiation_values_hourly = []
        for _, row in df.iterrows():
            rad_value = radiation_data.get((row['month'], row['day'], row['hour']), None)
            radiation_values_hourly.append(rad_value)
        
        df['GloblRad_hourly'] = radiation_values_hourly
        # Convert to numeric before interpolation
        df['GloblRad_hourly'] = pd.to_numeric(df['GloblRad_hourly'], errors='coerce')
        # Fill missing values using linear interpolation and round to 2 decimal places
        df['GloblRad_hourly'] = df['GloblRad_hourly'].interpolate(method='linear').round(2)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')

print('Radiation hourly data processed to traindata successfully.')