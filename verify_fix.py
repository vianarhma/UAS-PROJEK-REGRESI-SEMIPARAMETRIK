
import pandas as pd
import numpy as np

# Mocking the DataLoader logic exactly as written in the file
def clean_and_load(path):
    print(f"Loading {path}...")
    try:
        df = pd.read_csv(path, sep=';')
        print("Initial Head:")
        print(df[['close', 'volume']].head())
        print("\nInitial Dtypes:")
        print(df.dtypes)
        
        df_clean = df.copy()
        
        # Logic from data_loader.py
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df_clean.columns:
                print(f"\nProcessing {col}...")
                # Convert to string first
                df_clean[col] = df_clean[col].astype(str)
                print(f"As string: {df_clean[col].iloc[0]}")
                
                # Remove thousand separator (.)
                df_clean[col] = df_clean[col].str.replace('.', '', regex=False)
                print(f"After remove dot: {df_clean[col].iloc[0]}")
                
                # Replace decimal comma with dot
                df_clean[col] = df_clean[col].str.replace(',', '.', regex=False)
                print(f"After replace comma: {df_clean[col].iloc[0]}")
                
                # Now convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"Final numeric: {df_clean[col].iloc[0]}")
        
        # Stats
        print("\n--- FINAL STATS ---")
        print(df_clean['close'].describe())
        
        return df_clean
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    clean_and_load('data/df_1h.csv')
