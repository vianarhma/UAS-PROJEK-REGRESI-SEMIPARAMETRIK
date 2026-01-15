"""
Metrics Module
Fungsi-fungsi untuk menghitung evaluasi model
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate berbagai metrics untuk evaluasi model

    Parameters:
    -----------
    y_true : array-like
        Nilai aktual
    y_pred : array-like
        Nilai prediksi

    Returns:
    --------
    dict : Dictionary berisi semua metrics
    """
    # Pastikan array numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    # Accuracy dari MAPE
    accuracy = 100 - mape

    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    mean_error = np.mean(y_true - y_pred)
    std_error = np.std(y_true - y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Accuracy': accuracy,
        'Max_Error': max_error,
        'Mean_Error': mean_error,
        'Std_Error': std_error
    }

    return metrics


def calculate_percentage_error(y_true, y_pred):
    """
    Calculate percentage error untuk setiap prediksi

    Returns:
    --------
    array : Percentage error untuk setiap observasi
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return ((y_true - y_pred) / y_true) * 100


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (apakah prediksi arah benar)
    Berguna untuk time series

    Returns:
    --------
    float : Persentase prediksi arah yang benar
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) < 2:
        return 0.0

    # Calculate actual direction
    actual_direction = np.sign(np.diff(y_true))

    # Calculate predicted direction
    pred_direction = np.sign(np.diff(y_pred))

    # Calculate accuracy
    correct = np.sum(actual_direction == pred_direction)
    total = len(actual_direction)

    return (correct / total) * 100


def compare_models(metrics1, metrics2, model1_name="Model 1", model2_name="Model 2"):
    """
    Compare dua model dan return model terbaik

    Parameters:
    -----------
    metrics1 : dict
        Metrics dari model 1
    metrics2 : dict
        Metrics dari model 2
    model1_name : str
        Nama model 1
    model2_name : str
        Nama model 2

    Returns:
    --------
    dict : Hasil perbandingan
    """
    comparison = {
        'winner': {},
        'summary': {}
    }

    # Metrics yang lebih kecil = lebih baik
    lower_is_better = ['MSE', 'RMSE', 'MAE', 'MAPE', 'Max_Error']

    # Metrics yang lebih besar = lebih baik
    higher_is_better = ['R2', 'Accuracy']

    for metric in lower_is_better:
        if metric in metrics1 and metric in metrics2:
            if metrics1[metric] < metrics2[metric]:
                comparison['winner'][metric] = model1_name
                comparison['summary'][metric] = f"{model1_name} lebih baik ({metrics1[metric]:.4f} < {metrics2[metric]:.4f})"
            else:
                comparison['winner'][metric] = model2_name
                comparison['summary'][metric] = f"{model2_name} lebih baik ({metrics2[metric]:.4f} < {metrics1[metric]:.4f})"

    for metric in higher_is_better:
        if metric in metrics1 and metric in metrics2:
            if metrics1[metric] > metrics2[metric]:
                comparison['winner'][metric] = model1_name
                comparison['summary'][metric] = f"{model1_name} lebih baik ({metrics1[metric]:.4f} > {metrics2[metric]:.4f})"
            else:
                comparison['winner'][metric] = model2_name
                comparison['summary'][metric] = f"{model2_name} lebih baik ({metrics2[metric]:.4f} > {metrics1[metric]:.4f})"

    # Overall winner (berdasarkan jumlah metrics yang menang)
    winner_count = {}
    for winner in comparison['winner'].values():
        winner_count[winner] = winner_count.get(winner, 0) + 1

    overall_winner = max(winner_count, key=winner_count.get)
    comparison['overall_winner'] = overall_winner
    comparison['winner_count'] = winner_count

    return comparison


def format_metrics_table(metrics, model_name="Model"):
    """
    Format metrics menjadi dictionary untuk display di Streamlit

    Returns:
    --------
    dict : Formatted metrics
    """
    formatted = {
        'Model': model_name,
        'Accuracy (%)': f"{metrics['Accuracy']:.2f}%",
        'MAPE (%)': f"{metrics['MAPE']:.2f}%",
        'RMSE': f"{metrics['RMSE']:.4f}",
        'R² Score': f"{metrics['R2']:.4f}",
        'MAE': f"{metrics['MAE']:.4f}",
        'MSE': f"{metrics['MSE']:.4f}"
    }

    return formatted


def get_error_statistics(y_true, y_pred):
    """
    Get statistik error untuk analisis residual

    Returns:
    --------
    dict : Statistik error
    """
    errors = np.array(y_true) - np.array(y_pred)

    stats = {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'q25': np.percentile(errors, 25),
        'q75': np.percentile(errors, 75),
        'iqr': np.percentile(errors, 75) - np.percentile(errors, 25)
    }

    return stats


def calculate_forecast_error(y_true, y_pred, horizon=1):
    """
    Calculate forecast error untuk specific horizon
    Berguna untuk evaluasi prediksi 1 jam ke depan

    Parameters:
    -----------
    y_true : array-like
        Nilai aktual
    y_pred : array-like
        Nilai prediksi
    horizon : int
        Horizon prediksi (default: 1 jam)

    Returns:
    --------
    dict : Forecast error metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Forecast error
    fe = y_true - y_pred

    # Mean Forecast Error (bias)
    mfe = np.mean(fe)

    # Mean Absolute Forecast Error
    mafe = np.mean(np.abs(fe))

    # Forecast error metrics
    metrics = {
        'Horizon': f"{horizon} hour(s)",
        'Mean_Forecast_Error': mfe,
        'Mean_Absolute_Forecast_Error': mafe,
        'Forecast_Bias': 'Overprediction' if mfe < 0 else 'Underprediction',
        'Bias_Percentage': abs(mfe / np.mean(y_true)) * 100
    }

    return metrics