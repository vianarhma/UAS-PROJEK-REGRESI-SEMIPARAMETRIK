
import pandas as pd
import numpy as np

try:
    print("Loading CSV...")
    df = pd.read_csv('data/df_1h.csv', sep=';')
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(2))
    print("Tail:\n", df.tail(2))
    
    print("\n--- Data Types before processing ---")
    print(df.dtypes)
    
    # Simulate processing from data_loader.py
    print("\n--- Processing ---")
    df_clean = df.copy()
    
    # 1. Volume handling
    if 'volume' in df_clean.columns:
        print("Raw volume sample:", df_clean['volume'].iloc[0])
        df_clean['volume'] = df_clean['volume'].astype(str).str.replace('.', '', regex=False)
        print("Processed volume sample:", df_clean['volume'].iloc[0])

    # 2. Numeric conversion
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        print(f"Converting {col}...")
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        print(f"{col} - NaNs: {df_clean[col].isna().sum()}")
        print(f"{col} - Sample: {df_clean[col].iloc[:3].tolist()}")

    # 3. Check for flat values
    print("\n--- Stats ---")
    print(df_clean['close'].describe())
    
    print("\n--- Standard Deviation ---")
    std = df_clean['close'].std()
    print(f"Std Dev: {std}")
    if std == 0:
        print("WARNING: Standard deviation is 0, chart will be flat!")
    
except Exception as e:
    print(f"ERROR: {e}")
