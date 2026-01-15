"""
Nadaraya-Watson Kernel Regression Model
Implementation of nonparametric regression using kernel methods
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NadarayaWatsonModel:
    """
    Class untuk Nadaraya-Watson Kernel Regression
    """
    
    def __init__(self, kernel='gaussian', bandwidth=None):
        """
        Initialize model
        
        Parameters:
        -----------
        kernel : str
            Tipe kernel: 'gaussian', 'epanechnikov', 'uniform'
        bandwidth : float
            Bandwidth (h). Jika None, akan dicari optimal menggunakan GCV
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.h_optimal = None
        self.X_train = None
        self.y_train = None
        self.gcv_scores = None
        self.h_values = None
        
    def kernel_function(self, u):
        """
        Fungsi kernel
        
        Parameters:
        -----------
        u : array-like
            Input untuk kernel
        
        Returns:
        --------
        array : Output kernel
        """
        if self.kernel == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        
        elif self.kernel == 'epanechnikov':
            return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
        
        elif self.kernel == 'uniform':
            return 0.5 * (np.abs(u) <= 1)
        
        else:
            raise ValueError(f"Kernel '{self.kernel}' tidak dikenali!")
    
    def nadaraya_watson_estimator(self, x0, X, y, h):
        """
        Nadaraya-Watson estimator untuk satu titik (Legacy method for backward compatibility)
        """
        # Calculate kernel weights
        u = (x0 - X) / h
        weights = self.kernel_function(u)
        
        # Avoid division by zero
        if np.sum(weights) == 0:
            return np.mean(y)
        
        # Weighted average
        return np.sum(weights * y) / np.sum(weights)
    
    def calculate_gcv(self, X, y, h):
        """
        Calculate Generalized Cross-Validation (GCV) score
        Vectorized implementation for speed
        
        Parameters:
        -----------
        X : array
            Training features
        y : array
            Training target
        h : float
            Bandwidth to test
        
        Returns:
        --------
        float : GCV score
        """
        n = len(y)
        
        # Use vectorized operations
        # Compute difference matrix: (X[i] - X[j])
        # Reshape X to (n, 1) and (1, n) for broadcasting
        diff = (X[:, np.newaxis] - X[np.newaxis, :]) / h
        
        # Compute weight matrix K((Xi - Xj)/h)
        W = self.kernel_function(diff)
        
        # Compute sum of weights for each row
        # S_i = sum_j W_ij
        sum_weights = np.sum(W, axis=1)
        
        # Avoid division by zero
        # Create mask for zero sums
        zero_mask = sum_weights == 0
        sum_weights[zero_mask] = 1.0  # Set to 1 safe division
        
        # Compute y_hat = sum(W_ij * y_j) / sum(W_ij)
        # W * y broadcasts correctly if y is (n,) to (n,n)
        y_hat = np.sum(W * y, axis=1) / sum_weights
        
        # Handle zero weights fallback -> mean(y) or 0
        mean_y = np.mean(y)
        y_hat[zero_mask] = mean_y
        
        # Compute S_ii (diagonal of S matrix)
        # S_ij = W_ij / sum_k W_ik
        # S_ii = W_ii / sum_weights_i
        # W_ii is k(0) which is constant
        if self.kernel == 'gaussian':
            k0 = (1 / np.sqrt(2 * np.pi))
        elif self.kernel == 'epanechnikov':
            k0 = 0.75
        elif self.kernel == 'uniform':
            k0 = 0.5
        else:
            k0 = self.kernel_function(0.0)
            
        S_diag = k0 / sum_weights
        S_diag[zero_mask] = 1.0/n # Default fallback
        
        # Calculate MSE
        mse = np.mean((y - y_hat)**2)
        
        # Calculate GCV
        # GCV = MSE / (1 - mean(S_ii))^2
        denominator = (1 - np.mean(S_diag))**2
        
        # Avoid division by zero
        if denominator < 1e-10:
            return np.inf
        
        gcv = mse / denominator
        
        return gcv
    
    def find_optimal_bandwidth(self, X, y, h_range=None, n_candidates=40):
        """
        Cari bandwidth optimal menggunakan GCV
        
        Parameters:
        -----------
        X : array
            Training features
        y : array
            Training target
        h_range : tuple
            Range untuk bandwidth (min, max). Jika None, otomatis calculated
        n_candidates : int
            Jumlah kandidat bandwidth yang ditest
        
        Returns:
        --------
        float : Optimal bandwidth
        """
        if h_range is None:
            # Auto-calculate range based on data scale
            x_range = np.max(X) - np.min(X)
            h_min = x_range * 0.01  # 1% dari range
            h_max = x_range * 0.5   # 50% dari range
            h_range = (h_min, h_max)
        
        # Generate candidate bandwidths
        self.h_values = np.linspace(h_range[0], h_range[1], n_candidates)
        
        # Calculate GCV for each bandwidth
        self.gcv_scores = []
        
        # Add a check for data size. If too large, downsample for optimization part to save time
        n_samples = len(y)
        if n_samples > 5000:
            print(f"⚠️ Large dataset detected ({n_samples} samples). Using downsampling for bandwidth estimation.")
            # Deterministic sampling for consistency
            np.random.seed(42)
            indices = np.random.choice(n_samples, size=3000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
            
        for h in self.h_values:
            gcv = self.calculate_gcv(X_sample, y_sample, h)
            self.gcv_scores.append(gcv)
        
        # Find optimal
        self.h_optimal = self.h_values[np.argmin(self.gcv_scores)]
        
        return self.h_optimal
    
    def fit(self, X, y, find_optimal_h=True, h_range=None):
        """
        Fit model dengan data training
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training target
        find_optimal_h : bool
            Apakah mencari bandwidth optimal
        h_range : tuple
            Range untuk bandwidth search
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Find optimal bandwidth jika diminta
        if find_optimal_h or self.bandwidth is None:
            self.bandwidth = self.find_optimal_bandwidth(
                self.X_train, 
                self.y_train, 
                h_range=h_range
            )
            print(f"✅ Optimal bandwidth found: h = {self.bandwidth:.2f}")
        
        return self
    
    def predict(self, X):
        """
        Predict untuk data baru using vectorized operations
        
        Parameters:
        -----------
        X : array-like
            Data yang ingin diprediksi
        
        Returns:
        --------
        array : Prediksi
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model belum di-fit! Panggil .fit() terlebih dahulu.")
        
        X = np.array(X)
        
        # Vectorized prediction
        # X shape: (m,) -> (m, 1)
        # X_train shape: (n,) -> (1, n)
        # diff shape: (m, n)
        diff = (X[:, np.newaxis] - self.X_train[np.newaxis, :]) / self.bandwidth
        
        # weights shape: (m, n)
        weights = self.kernel_function(diff)
        
        # sum_weights shape: (m,)
        sum_weights = np.sum(weights, axis=1)
        
        # Avoid division by zero
        zero_mask = sum_weights == 0
        sum_weights[zero_mask] = 1.0
        
        # predictions shape: (m,)
        predictions = np.sum(weights * self.y_train, axis=1) / sum_weights
        
        # Fallback for zero weights
        predictions[zero_mask] = np.mean(self.y_train)
        
        return predictions
    
    def get_model_info(self):
        """
        Get informasi model
        
        Returns:
        --------
        dict : Informasi model
        """
        info = {
            'model_name': 'Nadaraya-Watson Kernel Regression',
            'kernel_type': self.kernel,
            'bandwidth': self.bandwidth,
            'optimal_bandwidth': self.h_optimal,
            'n_training_samples': len(self.X_train) if self.X_train is not None else 0
        }
        
        return info


def prepare_data_for_nw(df, test_size=0.2, random_state=42):
    """
    Prepare data untuk Nadaraya-Watson model
    
    Parameters:
    -----------
    df : DataFrame
        Dataset dengan kolom 'time_num' dan 'close'
    test_size : float
        Proporsi data testing
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    # Extract X dan y
    X = df['time_num'].values
    y = df['close'].values
    
    # Random split (tidak berdasarkan urutan waktu)
    n = len(y)
    np.random.seed(random_state)
    train_idx = np.random.choice(n, size=int((1-test_size)*n), replace=False)
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = np.delete(X, train_idx)
    y_test = np.delete(y, train_idx)
    
    return X_train, X_test, y_train, y_test


def get_nw_explanation():
    """
    Return penjelasan lengkap tentang Nadaraya-Watson
    
    Returns:
    --------
    dict : Dictionary berisi penjelasan
    """
    explanation = {
        'concept': """
        Regresi kernel adalah metode regresi nonparametrik yang digunakan untuk mengestimasi hubungan antara variabel respon Y dan variabel penjelas X tanpa mengasumsikan bentuk fungsi tertentu (misalnya linier atau kuadratik).
        
        Pendekatan ini bekerja dengan cara:
        - Mengestimasi nilai Y pada suatu titik x sebagai rata-rata berbobot dari data di sekitarnya
        - Bobot ditentukan oleh fungsi kernel dan bandwidth (parameter penghalus)
        
        Secara umum, estimator regresi kernel Nadaraya–Watson ditulis sebagai:
        """,
        
        'formula': r"\hat{m}(x) = \frac{\sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right) Y_i}{\sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right)}",
        
        'kernel_formula': r"K(\cdot) = \text{fungsi kernel}",
        
        'steps': [
            "1. **Menentukan Variabel**: Variabel responden (Y) dan variabel penjelas (X)",
            "2. **Memilih Fungsi Kernel**: Gaussian, Epanechnikov, atau uniform kernel",
            "3. **Menentukan Bandwidth (h)**: Parameter penghalus yang menentukan tingkat kehalusan kurva",
            "4. **Menghitung Bobot Kernel**: Untuk setiap titik x, hitung bobot berdasarkan jarak ke data training",
            "5. **Mengestimasi Nilai Regresi**: Hitung weighted average menggunakan bobot kernel",
            "6. **Evaluasi Model**: Gunakan MSE/RMSE, cross-validation error, dan visualisasi"
        ],
        
        'bandwidth_explanation': """
        Bandwidth menentukan tingkat kehalusan kurva:
        - h kecil → kurva tajam (risiko overfitting)
        - h besar → kurva terlalu halus (underfitting)
        """,
        
        'kernel_weight_calculation': """
        Untuk setiap titik x:
        - Hitung jarak x - X_i
        - Hitung bobot kernel K((x - X_i)/h)
        - Data yang lebih dekat → bobot lebih besar
        """,
        
        'estimation_formula': r"\hat{Y}(x) = \sum w_i(x) Y_i",
        
        'weight_formula': r"w_i(x) = \frac{K\left(\frac{x - X_i}{h}\right)}{\sum K\left(\frac{x - X_j}{h}\right)}",
        
        'evaluation_metrics': [
            "MSE / RMSE",
            "Cross Validation error", 
            "Visualisasi kurva regresi vs data asli"
        ],
        
        'advantages': [
            "✅ Tidak memerlukan asumsi bentuk fungsi",
            "✅ Fleksibel untuk data berpola nonlinier",
            "✅ Cocok untuk eksplorasi hubungan variabel"
        ],
        
        'disadvantages': [
            "❌ Sensitif terhadap bandwidth",
            "❌ Kurang efisien untuk data besar",
            "❌ Sulit diinterpretasi secara parametrik"
        ]
    }
    
    return explanation