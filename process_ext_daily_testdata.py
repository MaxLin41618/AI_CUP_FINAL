import pandas as pd
import os

# 提取特徵
def process_radiation_data(external_dir, filename_prefix):
    radiation_data = {}
    for month in range(1, 12):
        filename = f"{filename_prefix}-2024-{month:02d}.csv"
        file_path = os.path.join(external_dir, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            for _, row in df.iterrows():
                day = int(row['ObsTime'])
                radiation_data[(month, day)] = row['GloblRad']
    return radiation_data

# Process data for first location (L1-L14)
external_dir_1 = "data/external_data/external_monthly/C0Z100/"
radiation_data_1 = process_radiation_data(external_dir_1, "C0Z100")

# Process data for second location (L15-L17)
external_dir_2 = "data/external_data/external_monthly/466990/"
radiation_data_2 = process_radiation_data(external_dir_2, "466990")

# Process test file
test_file = "data/processed_test_data.csv"
df = pd.read_csv(test_file, encoding='utf-8-sig')

# Select appropriate radiation data based on location
radiation_values = []
for _, row in df.iterrows():
    location_code = row['LocationCode'] 
    radiation_data = radiation_data_1 if location_code <= 14 else radiation_data_2
    rad_value = radiation_data.get((row['month'], row['day']), None)
    radiation_values.append(rad_value)

df['GloblRad_daily'] = radiation_values
df['GloblRad_daily'] = pd.to_numeric(df['GloblRad_daily'], errors='coerce')
df['GloblRad_daily'] = df['GloblRad_daily'].interpolate(method='linear').round(2)  # Fill missing values using linear interpolation and round to 2 decimal places

# Save updated file
df.to_csv(test_file, index=False, encoding='utf-8-sig')

print('Radiation daily data processed to testdata successfully.')