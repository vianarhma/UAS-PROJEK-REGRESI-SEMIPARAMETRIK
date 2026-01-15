"""
LightGBM Model for Time Series Forecasting
Implementation of gradient boosting for 1-hour ahead price prediction
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

class LightGBMForecastModel:
    """
    Class untuk LightGBM Time Series Forecasting
    """
    
    def __init__(self, **params):
        """
        Initialize LightGBM model
        
        Parameters:
        -----------
        **params : dict
            Parameters untuk LGBMRegressor
        """
        # Default parameters
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': -1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        # Update dengan custom params
        default_params.update(params)
        
        # Initialize model
        self.model = LGBMRegressor(**default_params)
        self.params = default_params
        self.feature_names = None
        self.feature_importance = None
        
    def create_lag_features(self, df, lag_columns=['open', 'high', 'low', 'volume'], lag=1):
        """
        Create lag features untuk time series
        
        Parameters:
        -----------
        df : DataFrame
            Dataset original
        lag_columns : list
            Kolom yang akan di-lag
        lag : int
            Berapa periode lag (default: 1 untuk prediksi 1 jam)
        
        Returns:
        --------
        DataFrame : Dataset dengan lag features
        """
        df_lag = df.copy()
        
        # Create lag features
        for col in lag_columns:
            df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
        
        # Drop rows dengan NaN (hasil shifting)
        df_lag = df_lag.dropna()
        
        return df_lag
    
    def prepare_features(self, df, target_col='close', lag=1):
        """
        Prepare features dan target untuk training
        
        Parameters:
        -----------
        df : DataFrame
            Dataset
        target_col : str
            Nama kolom target
        lag : int
            Lag period
        
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        # Create lag features
        df_prepared = self.create_lag_features(df, lag=lag)
        
        # Define feature columns
        feature_cols = [col for col in df_prepared.columns if col.endswith(f'_lag{lag}')]
        
        # Extract X dan y
        X = df_prepared[feature_cols]
        y = df_prepared[target_col]
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    def fit(self, X, y):
        """
        Train model
        
        Parameters:
        -----------
        X : DataFrame or array
            Training features
        y : Series or array
            Training target
        """
        print("🔄 Training LightGBM model...")
        
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        print("✅ Model training completed!")
        
        return self
    
    def predict(self, X):
        """
        Predict target
        
        Parameters:
        -----------
        X : DataFrame or array
            Features untuk prediksi
        
        Returns:
        --------
        array : Predictions
        """
        return self.model.predict(X)
    
    def safe_predict(self, X):
        """
        Safe predict that bypasses scikit-learn validation
        Works around version compatibility issues
        
        Parameters:
        -----------
        X : DataFrame or array
            Features untuk prediksi
        
        Returns:
        --------
        array : Predictions
        """
        try:
            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            # Method 1: Use booster directly
            if hasattr(self.model, 'booster_'):
                return self.model.booster_.predict(X_array)
            
            # Method 2: Use predict with raw_score
            return self.model.predict(X_array, raw_score=False)
        except Exception as e:
            # Last resort: standard predict
            return self.model.predict(X)
    
    
    def get_feature_importance(self):
        """
        Get feature importance sebagai DataFrame
        
        Returns:
        --------
        DataFrame : Feature importance sorted
        """
        if self.feature_importance is None:
            raise ValueError("Model belum di-train!")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def get_model_info(self):
        """
        Get informasi model
        
        Returns:
        --------
        dict : Model information
        """
        info = {
            'model_name': 'LightGBM Regressor',
            'n_estimators': self.params['n_estimators'],
            'learning_rate': self.params['learning_rate'],
            'max_depth': self.params['max_depth'],
            'num_leaves': self.params['num_leaves'],
            'features': self.feature_names
        }
        
        return info


def prepare_data_for_lgb(df, test_size=0.2, target_col='close', lag=1):
    """
    Prepare data untuk LightGBM dengan temporal split
    
    Parameters:
    -----------
    df : DataFrame
        Dataset original
    test_size : float
        Proporsi data testing
    target_col : str
        Nama kolom target
    lag : int
        Lag period
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, feature_names)
    """
    # Create lag features
    df_lag = df.copy()
    lag_columns = ['open', 'high', 'low', 'volume']
    
    for col in lag_columns:
        df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
    
    # Drop NaN
    df_lag = df_lag.dropna()
    
    # Feature columns
    feature_cols = [f'{col}_lag{lag}' for col in lag_columns]
    
    # Extract X dan y
    X = df_lag[feature_cols]
    y = df_lag[target_col]
    
    # Temporal split (shuffle=False untuk menjaga urutan waktu)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


def get_lgb_explanation():
    """
    Return penjelasan lengkap tentang LightGBM
    
    Returns:
    --------
    dict : Dictionary berisi penjelasan
    """
    explanation = {
        'concept': """
        Regresi LightGBM adalah metode regresi berbasis ensemble machine learning yang menggunakan algoritma Gradient Boosting Decision Tree (GBDT) dengan pendekatan yang lebih efisien dan cepat dibandingkan GBM konvensional. Pada regresi LightGBM, model membangun banyak pohon keputusan (decision tree) secara bertahap, di mana setiap pohon baru berfungsi untuk memperbaiki kesalahan (residual) dari pohon sebelumnya.
        """,
        
        'formula': r"Obj = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)",
        
        'formula_explanation': """
        - L: Loss function (MSE, MAE, atau Huber loss untuk regresi)
        - Ω: Regularization term untuk prevent overfitting
        - f_k: Individual tree ke-k
        - K: Total jumlah trees (n_estimators)
        """,
        
        'key_techniques': {
            'Leaf-wise Growth': 'Tree tumbuh berdasarkan leaf dengan gain maksimal (bukan level-wise)',
            'Histogram-based': 'Diskritisasi fitur kontinyu untuk efisiensi komputasi',
            'GOSS': 'Gradient-based One-Side Sampling - sampling data berdasarkan gradien',
            'EFB': 'Exclusive Feature Bundling - bundling fitur yang mutually exclusive'
        },
        
        'steps': [
            "1. **Menentukan Variabel**: Variabel respon (Y kontinu) dan prediktor (X)",
            "2. **Persiapan dan Pra-pemrosesan Data**: Handle missing values, encoding kategorik, normalisasi opsional",
            "3. **Membagi Data**: Training set dan testing/validation set",
            "4. **Menentukan Fungsi Loss**: MSE, MAE, atau Huber loss untuk regresi",
            "5. **Inisialisasi Model**: Set initial prediction (bias)",
            "6. **Proses Boosting (Iteratif)**: Bangun tree baru untuk predict residual",
            "7. **Penyetelan Hyperparameter**: Tune n_estimators, learning_rate, num_leaves, dll",
            "8. **Evaluasi Model**: Gunakan RMSE, MAE, R², cross-validation error"
        ],
        
        'hyperparameters': {
            'n_estimators': 'Jumlah boosting iterations (trees)',
            'learning_rate': 'Shrinkage rate untuk prevent overfitting',
            'max_depth': 'Maksimal kedalaman tree (-1 = no limit)',
            'num_leaves': 'Maksimal jumlah leaves per tree',
            'min_child_samples': 'Minimal samples di leaf node',
            'subsample': 'Fraction of data untuk setiap tree',
            'colsample_bytree': 'Fraction of features untuk setiap tree'
        },
        
        'advantages': [
            "✅ Sangat cepat untuk data besar",
            "✅ Akurat untuk hubungan nonlinier",
            "✅ Mendukung data kategorik",
            "✅ Sedikit preprocessing",
            "✅ Built-in feature importance"
        ],
        
        'disadvantages': [
            "❌ Risiko overfitting jika parameter tidak tepat",
            "❌ Kurang transparan dibanding regresi klasik",
            "❌ Tidak cocok untuk data sangat kecil"
        ],
        
        'use_cases': [
            "📊 Time series forecasting",
            "💰 Financial prediction",
            "🏆 Kaggle competitions",
            "🎯 Ranking problems",
            "📈 Regression & classification tasks"
        ]
    }
    
    return explanation


def get_preprocessing_steps():
    """
    Return langkah-langkah preprocessing
    
    Returns:
    --------
    dict : Preprocessing steps
    """
    steps = {
        'steps': [
            {
                'step': '1. Load & Inspect Data',
                'description': 'Import dataset dan cek dimensi, tipe data, missing values',
                'code_example': """
df = pd.read_csv('df_1h.csv')
print(df.info())
print(df.describe())
                """
            },
            {
                'step': '2. Handling Missing Values',
                'description': 'Deteksi dan handle missing values',
                'code_example': """
# Check missing values
print(df.isnull().sum())

# Drop or impute
df = df.dropna()  # atau
df = df.fillna(method='ffill')
                """
            },
            {
                'step': '3. Data Type Conversion',
                'description': 'Konversi tipe data yang sesuai',
                'code_example': """
df['datetime'] = pd.to_datetime(df['datetime'])
df['close'] = pd.to_numeric(df['close'])
                """
            },
            {
                'step': '4. Feature Engineering',
                'description': 'Buat features baru (lag features untuk forecasting)',
                'code_example': """
# Untuk Nadaraya-Watson
df['time_num'] = df['datetime'].astype(np.int64) // 10**9

# Untuk LightGBM
df['open_lag1'] = df['open'].shift(1)
df['high_lag1'] = df['high'].shift(1)
df['low_lag1'] = df['low'].shift(1)
df['volume_lag1'] = df['volume'].shift(1)
df = df.dropna()
                """
            },
            {
                'step': '5. Outlier Detection',
                'description': 'Identifikasi outlier menggunakan IQR method',
                'code_example': """
Q1 = df['close'].quantile(0.25)
Q3 = df['close'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['close'] < Q1 - 1.5*IQR) | (df['close'] > Q3 + 1.5*IQR)
                """
            },
            {
                'step': '6. Train-Test Split',
                'description': 'Split data untuk training dan testing',
                'code_example': """
# Nadaraya-Watson (Random Split)
train_idx = np.random.choice(n, size=int(0.8*n), replace=False)

# LightGBM (Temporal Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
                """
            },
            {
                'step': '7. Data Validation',
                'description': 'Validasi data sebelum modeling',
                'code_example': """
assert df.isnull().sum().sum() == 0, "Still have NaN!"
assert len(X_train) > 0, "Train set empty!"
assert len(X_test) > 0, "Test set empty!"
                """
            }
        ]
    }
    
    return steps