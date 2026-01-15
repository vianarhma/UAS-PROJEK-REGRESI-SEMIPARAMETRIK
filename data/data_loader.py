"""
Data Loader Module
Handles data loading with 2 options: default dataset or upload CSV
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

class DataLoader:
    """Class untuk handle loading data dengan berbagai opsi"""
    
    def __init__(self, default_path="data/df_1h.csv"):
        self.default_path = default_path
        
    def load_default_data(self):
        """Load data default dari file CSV"""
        try:
            df = pd.read_csv(self.default_path, sep=';')
            return df
        except FileNotFoundError:
            st.error(f"❌ File {self.default_path} tidak ditemukan!")
            return None
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None
    
    def validate_columns(self, df):
        """Validasi apakah kolom yang diperlukan ada"""
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            return False
        return True
    
    def preprocess_data(self, df):
        """Basic preprocessing untuk data"""
        try:
            # Copy dataframe
            df_clean = df.copy()
            
            # Convert datetime
            df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], dayfirst=True)
            
            # Handle All Numeric Columns (open, high, low, close, volume)
            num_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in num_cols:
                if col in df_clean.columns:
                    # Convert to string and clean up formatting
                    col_data = df_clean[col].astype(str)
                    
                    # Pattern check: if it looks like ID format (e.g. 33.123,50)
                    # For now, let's be robust: 
                    # 1. Remove all dots (thousands)
                    # 2. Replace comma with dot (decimal)
                    
                    # NOTE: We need to distinguish between 33.123 (33 thousand) 
                    # and 33.123 (33 point 123). 
                    # Logic: If more than one dot exists, it MUST be thousand separators.
                    # Logic 2: If price is > 1000 and has a dot, highly likely it's a thousand separator.
                    
                    def clean_numeric(val):
                        if pd.isna(val) or val == 'nan' or val == '':
                            return np.nan
                        s = str(val).strip()
                        if not s: return np.nan
                        
                        # Count dots and commas
                        dots = s.count('.')
                        commas = s.count(',')
                        
                        if dots > 1:
                            # Clearly thousand separators
                            s = s.replace('.', '')
                        
                        if commas > 0:
                            # Commas as decimals (ID format)
                            s = s.replace('.', '') # remove thousand dots
                            s = s.replace(',', '.') # comma to dot
                        
                        return s

                    df_clean[col] = col_data.apply(clean_numeric)
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            
            # Drop rows with NaN in critical columns
            df_clean = df_clean.dropna(subset=['datetime', 'close'])
            
            # Remove rows where close price is 0 or negative (invalid data)
            df_clean = df_clean[df_clean['close'] > 0]
            
            # Sort by datetime
            df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
            
            # Add time_num for Nadaraya-Watson
            df_clean['time_num'] = df_clean['datetime'].astype(np.int64) // 10**9
            
            return df_clean
            
        except Exception as e:
            st.error(f"❌ Error preprocessing data: {str(e)}")
            return None
    
    def get_dataset_info(self, df):
        """Get informasi dataset"""
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': f"{df['datetime'].min()} - {df['datetime'].max()}",
            'duration_days': (df['datetime'].max() - df['datetime'].min()).days,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        return info
    
    def load_data_with_options(self):
        """
        Load data dengan 2 opsi:
        1. Gunakan data default
        2. Upload CSV sendiri
        """
        st.subheader("📂 Pilih Sumber Data")
        
        # Radio button untuk pilih opsi
        option = st.radio(
            "Pilih cara load data:",
            ["📊 Gunakan Dataset Default", "📤 Upload CSV Sendiri"],
            horizontal=True
        )
        
        df = None
        
        if option == "📊 Gunakan Dataset Default":
            st.info("✅ Menggunakan dataset default: **df_1h.csv**")
            
            if st.button("🔄 Load Dataset Default"):
                with st.spinner("Loading data..."):
                    df = self.load_default_data()
                    
                    if df is not None:
                        if self.validate_columns(df):
                            df = self.preprocess_data(df)
                            if df is not None:
                                st.success(f"✅ Data berhasil dimuat! Total: {len(df)} baris")
                                st.session_state['df'] = df
                                st.session_state['data_loaded'] = True
        
        elif option == "📤 Upload CSV Sendiri":
            st.info("📁 Upload file CSV dengan format yang sesuai")
            
            uploaded_file = st.file_uploader(
                "Pilih file CSV",
                type=['csv'],
                help="File harus memiliki kolom: datetime, open, high, low, close, volume"
            )
            
            if uploaded_file is not None:
                with st.spinner("Loading data..."):
                    df = pd.read_csv(uploaded_file)
                    
                    if self.validate_columns(df):
                        df = self.preprocess_data(df)
                        if df is not None:
                            st.success(f"✅ Data berhasil dimuat! Total: {len(df)} baris")
                            st.session_state['df'] = df
                            st.session_state['data_loaded'] = True
        
        # Return data dari session state jika sudah ada
        if 'df' in st.session_state:
            return st.session_state['df']
        
        return df


def create_sample_data():
    """
    Fungsi untuk create sample data jika file tidak ada
    (Untuk development/testing)
    """
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    n = len(dates)
    
    # Generate synthetic price data
    np.random.seed(42)
    base_price = 100
    trend = np.linspace(0, 50, n)
    noise = np.random.normal(0, 5, n)
    close_price = base_price + trend + noise
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close_price + np.random.normal(0, 2, n),
        'high': close_price + np.abs(np.random.normal(2, 1, n)),
        'low': close_price - np.abs(np.random.normal(2, 1, n)),
        'close': close_price,
        'volume': np.random.uniform(1000000, 5000000, n)
    })
    
    return df


# Instance global untuk digunakan di seluruh aplikasi
data_loader = DataLoader()