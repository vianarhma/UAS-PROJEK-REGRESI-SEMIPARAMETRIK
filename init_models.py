import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from data.data_loader import DataLoader
from models.nadaraya_watson import NadarayaWatsonModel, prepare_data_for_nw
from models.lightgbm_model import LightGBMForecastModel, prepare_data_for_lgb
from utils import save_model

def init_models():
    print("⏳ Menghidupkan ulang model dengan sistem baru...")
    
    # 1. Load Data
    loader = DataLoader("data/df_1h.csv")
    df = loader.load_default_data()
    if df is None:
        print("❌ Gagal memuat data.")
        return
    
    df = loader.preprocess_data(df)
    
    # 2. Train Nadaraya-Watson
    print("🔵 Training Nadaraya-Watson (ini mungkin sebentar)...")
    X_train_nw, X_test_nw, y_train_nw, y_test_nw = prepare_data_for_nw(df, test_size=0.2, random_state=42)
    nw_model = NadarayaWatsonModel(kernel='gaussian')
    nw_model.fit(X_train_nw, y_train_nw, find_optimal_h=True, h_range=(5000, 20000))
    save_model(nw_model, 'nadaraya_watson')
    print("✅ Nadaraya-Watson selesai & disimpan as 'nadaraya_watson_latest.pkl'")
    
    # 3. Train LightGBM
    print("🟢 Training LightGBM...")
    X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb, feature_names = prepare_data_for_lgb(
        df, test_size=0.2, target_col='close', lag=1
    )
    lgb_model = LightGBMForecastModel(n_estimators=100, learning_rate=0.05, verbose=-1)
    lgb_model.fit(X_train_lgb, y_train_lgb)
    save_model(lgb_model, 'lightgbm')
    print("✅ LightGBM selesai & disimpan as 'lightgbm_latest.pkl'")
    
    print("\n✨ Selesai! Sekarang buka Streamlit, dashboard akan langsung cepat.")

if __name__ == "__main__":
    # Ensure saved_models exists
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    init_models()
