"""
Visualizations Module
Fungsi-fungsi untuk membuat visualisasi dengan Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Premium Color Palette
COLORS = {
    'primary': '#0ea5e9',    # Sky Blue
    'secondary': '#6366f1',  # Indigo
    'success': '#10b981',    # Emerald
    'danger': '#f43f5e',     # Rose
    'warning': '#f59e0b',    # Amber
    'text': '#1e293b',       # Slate 800
    'grid': '#e2e8f0',       # Slate 200
    'background': '#ffffff'
}

def _apply_premium_layout(fig, title="", x_title="", y_title="", height=500):
    """Helper untuk apply styling premium yang konsisten"""
    fig.update_layout(
        title=dict(
            text=title, 
            font=dict(size=18, family="Source Sans Pro", color=COLORS['text']),
            y=0.95
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=0.5,
            showline=True,
            linecolor=COLORS['grid']
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=0.5,
            showline=True,
            linecolor=COLORS['grid'],
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        hovermode='x unified',
        font=dict(family="Source Sans Pro", color=COLORS['text']),
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    return fig

def plot_time_series(df, y_column='close', title="Time Series Plot"):
    """
    Plot time series dengan indikator tren
    """
    # Ensure data is handled correctly
    df_plot = df.sort_values('datetime').copy()
    
    # Create figure
    fig = go.Figure()
    
    # Main Price Line - Make it MORE PROMINENT
    fig.add_trace(go.Scatter(
        x=df_plot['datetime'],
        y=df_plot[y_column],
        mode='lines',
        name='Harga Close (Aktual)',
        line=dict(color='#0ea5e9', width=2.5),  # Brighter blue, thicker line
        fill='tozeroy',
        fillcolor='rgba(14, 165, 233, 0.15)',
        opacity=1.0,  # Full opacity
        hovertemplate='<b>Waktu:</b> %{x}<br><b>Harga:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Add Moving Average (Simple Trend) - Make it less prominent
    try:
        ma_24 = df_plot[y_column].rolling(window=24).mean()
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=ma_24,
            mode='lines',
            name='Moving Avg (24H)',
            line=dict(color='#f59e0b', width=1.5, dash='dash'),
            opacity=0.6,  # Reduced opacity to not overpower actual data
            hovertemplate='<b>MA 24H:</b> $%{y:.2f}<extra></extra>'
        ))
    except:
        pass
    
    return _apply_premium_layout(fig, title, "Waktu", "Harga (USD)")

def plot_combined_analysis(y_true_1, y_pred_1, y_true_2, y_pred_2, 
                         model1_name="Nadaraya-Watson", model2_name="LightGBM"):
    """
    Plot perbandingan dua model yang lebih bersih
    """
    fig = go.Figure()

    # Actual Data (Shared)
    fig.add_trace(go.Scatter(
        x=list(range(len(y_true_1))), 
        y=y_true_1, 
        mode='lines', 
        name='Data Aktual',
        line=dict(color=COLORS['text'], width=2, dash='dot'),
        opacity=0.6
    ))

    # Model 1
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred_1))), 
        y=y_pred_1, 
        mode='lines', 
        name=f'Prediksi {model1_name}',
        line=dict(color=COLORS['primary'], width=2)
    ))

    # Model 2
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred_2))), 
        y=y_pred_2, 
        mode='lines', 
        name=f'Prediksi {model2_name}',
        line=dict(color=COLORS['success'], width=2)
    ))

    return _apply_premium_layout(fig, "Perbandingan Prediksi Model vs Aktual", "Index Data Test", "Harga (USD)")

def plot_comparison_two_models(y_true_1, y_pred_1, y_true_2, y_pred_2, 
                                model1_name="Model 1", model2_name="Model 2"):
    """
    Subplot comparison original (kept for compatibility but styled)
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"Analisis {model1_name}", f"Analisis {model2_name}"),
        vertical_spacing=0.15
    )
    
    x_idx = list(range(len(y_true_1)))
    
    # Model 1
    fig.add_trace(go.Scatter(x=x_idx, y=y_true_1, mode='lines', name='Aktual', 
                           line=dict(color=COLORS['grid'], width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_idx, y=y_pred_1, mode='lines', name=f'{model1_name}', 
                           line=dict(color=COLORS['primary'], width=2)), row=1, col=1)
    
    # Model 2
    fig.add_trace(go.Scatter(x=x_idx, y=y_true_2, mode='lines', name='Aktual', 
                           line=dict(color=COLORS['grid'], width=3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_idx, y=y_pred_2, mode='lines', name=f'{model2_name}', 
                           line=dict(color=COLORS['success'], width=2)), row=2, col=1)
    
    fig.update_layout(height=700, template='plotly_white', hovermode='x unified')
    return fig

def plot_error_distribution(y_true, y_pred, title="Distribusi Error"):
    """
    Histogram error dengan KDE style
    """
    errors = np.array(y_true) - np.array(y_pred)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Error',
        marker_color=COLORS['secondary'],
        opacity=0.75,
        histnorm='probability density'
    ))
    
    return _apply_premium_layout(fig, title, "Nilai Error (Selisih)", "Densitas", height=400)

def plot_scatter_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """
    Scatter plot dengan garis diagonal yang jelas
    """
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig = go.Figure()
    
    # Reference Line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(color=COLORS['grid'], width=2, dash='dash')
    ))
    
    # Data Scatter
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        name='Data Point',
        marker=dict(
            color=COLORS['primary'],
            size=8,
            opacity=0.5,
            line=dict(color='white', width=1)
        )
    ))
    
    return _apply_premium_layout(fig, title, "Harga Aktual (USD)", "Harga Prediksi (USD)", height=450)

def plot_residual_plot(y_true, y_pred, title="Residual Analysis"):
    """
    Residual plot yang lebih jelas
    """
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig = go.Figure()
    
    # Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['danger'], opacity=0.8)
    
    # Residuals
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        marker=dict(
            color=COLORS['secondary'],
            size=7,
            opacity=0.6,
            line=dict(color='white', width=0.5)
        ),
        name='Residual'
    ))
    
    return _apply_premium_layout(fig, title, "Harga Prediksi (ŷ)", "Residual (y - ŷ)", height=400)

def plot_feature_importance(feature_names, importance_values, title="Feature Importance"):
    """
    Horizontal Bar Chart yang cantik
    """
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    df_imp = df_imp.sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_imp['Feature'],
        x=df_imp['Importance'],
        orientation='h',
        marker=dict(
            color=df_imp['Importance'],
            colorscale='Viridis',
            showscale=False
        ),
        text=df_imp['Importance'].round(3),
        textposition='outside'
    ))
    
    return _apply_premium_layout(fig, title, "Importance Score", "", height=400)

def plot_correlation_heatmap(df, columns=None):
    """Correlation Heatmap"""
    if columns is None:
        columns = ['open', 'high', 'low', 'close', 'volume']
        
    corr = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Matriks Korelasi",
        height=500,
        font=dict(family="Source Sans Pro")
    )
    return fig

def plot_box_plot(df, title="Distribusi Outliers"):
    """Box Plot"""
    fig = go.Figure()
    
    for col in ['open', 'close', 'high', 'low']:
        fig.add_trace(go.Box(
            y=df[col], 
            name=col.capitalize(),
            marker_color=COLORS['primary'],
            boxpoints='outliers'
        ))
        
    return _apply_premium_layout(fig, title, "", "Harga (USD)")

def plot_gcv_curve(h_values, gcv_scores, h_optimal):
    """GCV Curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=h_values, y=gcv_scores,
        mode='lines', name='GCV Score',
        line=dict(color=COLORS['secondary'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[h_optimal], y=[min(gcv_scores)],
        mode='markers', name='Optimal H',
        marker=dict(color=COLORS['danger'], size=12, symbol='star', line=dict(color='white', width=2))
    ))
    
    return _apply_premium_layout(fig, "Optimal Bandwidth Selection (GCV)", "Bandwidth (h)", "GCV Score", height=400)
