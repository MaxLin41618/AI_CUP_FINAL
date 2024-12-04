import pandas as pd
import glob
import os

def load_solar_angles():
    solar_angles = pd.read_csv(
        "data/external_data/external_angles/solar_angles_10min_merged.csv"
    )
    solar_angles['DateTime'] = pd.to_datetime(solar_angles['DateTime'])
    solar_angles.set_index('DateTime', inplace=True)
    return solar_angles[['仰角']]

def merge_solar_angles(target_file, solar_angles_df):
    # Read target file
    df = pd.read_csv(target_file)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Merge with solar angles
    df = df.merge(solar_angles_df, on='DateTime', how='left')
    
    # Fill missing values with 0
    df['仰角'] = df['仰角'].fillna(0)
    
    # Save back to the same file
    df.to_csv(target_file, index=False)

def main():
    # Load solar angles data
    solar_angles = load_solar_angles()
    
    # Process test data
    test_file = "data/processed_test_data.csv"
    merge_solar_angles(test_file, solar_angles)
    
    # Process training data files
    train_path = "data/train_data_processed"
    train_files = glob.glob(os.path.join(train_path, "*.csv"))
    
    for train_file in train_files:
        merge_solar_angles(train_file, solar_angles)

if __name__ == "__main__":
    main()
