import pandas as pd
import os

# 提取特徵
def process_hourly_radiation_data(file_path):
    radiation_data = {}
    df = pd.read_csv(file_path)
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

# Process test file
test_file = "data/processed_test_data.csv"
df = pd.read_csv(test_file, encoding='utf-8-sig')

# Select appropriate radiation data based on location
radiation_values_hourly = []
for _, row in df.iterrows():
    location_code = row['LocationCode']
    radiation_data = radiation_data_1 if location_code <= 14 else radiation_data_2
    rad_value = radiation_data.get((row['month'], row['day'], row['hour']), None)
    radiation_values_hourly.append(rad_value)

df['GloblRad_hourly'] = radiation_values_hourly
df['GloblRad_hourly'] = pd.to_numeric(df['GloblRad_hourly'], errors='coerce')
df['GloblRad_hourly'] = df['GloblRad_hourly'].interpolate(method='linear').round(2) 

# Save updated file
df.to_csv(test_file, index=False, encoding='utf-8-sig')

print('Radiation hourly data processed to testdata successfully.')
