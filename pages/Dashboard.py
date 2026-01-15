"""
Dashboard Page - Solana Price Prediction App
Visualisasi hasil prediksi dan perbandingan kedua model
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from models.nadaraya_watson import NadarayaWatsonModel, prepare_data_for_nw
from models.lightgbm_model import LightGBMForecastModel, prepare_data_for_lgb
from utils.styling import apply_custom_css, render_header, render_section_header, add_vertical_space, render_slide_navigator
from utils.metrics import calculate_metrics, compare_models, format_metrics_table
from utils import save_model, load_latest_model, get_model_hash
from utils.visualizations import (plot_comparison_two_models, plot_error_distribution, 
                                   plot_scatter_actual_vs_predicted, plot_feature_importance,
                                   plot_residual_plot, plot_gcv_curve)

# ==================== CACHING ====================

@st.cache_resource
def get_cached_models():
    """Load latest models from disk once and keep in memory"""
    nw = load_latest_model('nadaraya_watson')
    lgb = load_latest_model('lightgbm')
    return nw, lgb

@st.cache_data
def get_processed_data():
    """Load and preprocess data once"""
    data_loader = DataLoader()
    df_raw = data_loader.load_default_data()
    if df_raw is not None:
        return data_loader.preprocess_data(df_raw)
    return None

@st.cache_data
def get_predictions(_nw_model, _lgb_model, df):
    """Calculate predictions once for a given model and data"""
    # Prepare data for NW
    X_train_nw, X_test_nw, y_train_nw, y_test_nw = prepare_data_for_nw(df, test_size=0.2, random_state=42)
    y_pred_nw = _nw_model.predict(X_test_nw)
    metrics_nw = calculate_metrics(y_test_nw, y_pred_nw)

    # Prepare data for LGB
    X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb, feature_names = prepare_data_for_lgb(
        df, test_size=0.2, target_col='close', lag=1
    )
    y_pred_lgb = _lgb_model.predict(X_test_lgb)
    metrics_lgb = calculate_metrics(y_test_lgb, y_pred_lgb)
    
    return {
        'nw': {
            'X_train': X_train_nw, 'X_test': X_test_nw, 
            'y_train': y_train_nw, 'y_test': y_test_nw, 
            'y_pred': y_pred_nw, 'metrics': metrics_nw
        },
        'lgb': {
            'X_train': X_train_lgb, 'X_test': X_test_lgb, 
            'y_train': y_train_lgb, 'y_test': y_test_lgb, 
            'y_pred': y_pred_lgb, 'metrics': metrics_lgb,
            'feature_names': feature_names
        }
    }

# Apply custom CSS
apply_custom_css()

# ==================== HEADER ====================
render_header(
    title="📊 Dashboard Prediksi Harga Cryptocurrency",
    subtitle="Analisis & Visualisasi: Nadaraya-Watson vs LightGBM",
    icon="📈"
)

# ==================== CHECK DATA ====================
data_available = False

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("🔄 Loading default dataset...")
    df_default = get_processed_data()
    if df_default is not None:
        st.session_state['df'] = df_default
        st.session_state['data_loaded'] = True
        data_available = True
    else:
        st.error("❌ Failed to load data")
        st.stop()
else:
    data_available = True

df = st.session_state['df']

# ==================== TRAINING SECTION ====================
st.markdown("## 🔄 Model Training")

col1, col2 = st.columns(2)

with col1:
    st.info("**Konfigurasi Training:**\n- Train: 80% | Test: 20%\n- Random State: 42")

with col2:
    if st.button("🚀 Train Models", type="primary", use_container_width=True, key="train_button_config"):
        st.session_state['models_trained'] = False  # Reset

    if st.button("🔄 Force Retrain", help="Force retrain models even if saved models exist"):
        # Clear saved models and session state
        st.session_state['models_trained'] = False
        if 'saved_training_data' in st.session_state:
            del st.session_state['saved_training_data']
        if 'last_data_hash' in st.session_state:
            del st.session_state['last_data_hash']
        st.success("Cache cleared! Models will be retrained.")
        st.rerun()

# Check if models exist
nw_cached, lgb_cached = get_cached_models()
models_exist = nw_cached is not None and lgb_cached is not None

if models_exist:
    st.info("💾 **Saved models found!** App will use them for instant loading.")
else:
    st.warning("⚠️ **No saved models found.** Training will be performed from scratch.")

# ==================== TRAIN MODELS ====================
if st.button("🚀 Train Models", type="primary", use_container_width=True, key="train_button_main") or \
   'models_trained' in st.session_state and st.session_state['models_trained']:

    with st.spinner("⏳ Loading or training models... Mohon tunggu..."):
        # Hash to check if data changed
        data_hash = get_model_hash({
            'n_samples': len(df),
            'features': str(df.columns.tolist()),
            'target': 'close'
        })

        nw_model, lgb_model = get_cached_models()

        if nw_model and lgb_model and st.session_state.get('last_data_hash') == data_hash:
            # Use cached predictions where possible
            results = get_predictions(nw_model, lgb_model, df)
            
            X_train_nw, X_test_nw, y_train_nw, y_test_nw, y_pred_nw, metrics_nw = \
                results['nw']['X_train'], results['nw']['X_test'], results['nw']['y_train'], \
                results['nw']['y_test'], results['nw']['y_pred'], results['nw']['metrics']
                
            X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb, y_pred_lgb, metrics_lgb, feature_names = \
                results['lgb']['X_train'], results['lgb']['X_test'], results['lgb']['y_train'], \
                results['lgb']['y_test'], results['lgb']['y_pred'], results['lgb']['metrics'], \
                results['lgb']['feature_names']
            
            st.success("✅ Models and predictions loaded efficiently!")
        else:
            # Train new models (either first time or forced)
            st.info("🔄 Training new models...")
            progress_bar = st.progress(0)

            # NW
            X_train_nw, X_test_nw, y_train_nw, y_test_nw = prepare_data_for_nw(df, test_size=0.2, random_state=42)
            nw_model = NadarayaWatsonModel(kernel='gaussian')
            nw_model.fit(X_train_nw, y_train_nw, find_optimal_h=True, h_range=(5000, 20000))
            y_pred_nw = nw_model.predict(X_test_nw)
            metrics_nw = calculate_metrics(y_test_nw, y_pred_nw)
            save_model(nw_model, 'nadaraya_watson') # Always saves to _latest.pkl now
            progress_bar.progress(50)

            # LGB
            X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb, feature_names = prepare_data_for_lgb(
                df, test_size=0.2, target_col='close', lag=1
            )
            lgb_model = LightGBMForecastModel(n_estimators=100, learning_rate=0.05, verbose=-1)
            lgb_model.fit(X_train_lgb, y_train_lgb)
            y_pred_lgb = lgb_model.predict(X_test_lgb)
            metrics_lgb = calculate_metrics(y_test_lgb, y_pred_lgb)
            save_model(lgb_model, 'lightgbm')
            progress_bar.progress(100)
            
            # Clear resource cache to pick up new models
            st.cache_resource.clear()
            st.cache_data.clear()

        # Update session state
        st.session_state['models_trained'] = True
        st.session_state['nw_model'] = nw_model
        st.session_state['lgb_model'] = lgb_model
        st.session_state['y_test_nw'] = y_test_nw
        st.session_state['y_pred_nw'] = y_pred_nw
        st.session_state['metrics_nw'] = metrics_nw
        st.session_state['y_test_lgb'] = y_test_lgb
        st.session_state['y_pred_lgb'] = y_pred_lgb
        st.session_state['metrics_lgb'] = metrics_lgb
        st.session_state['y_train_nw'] = y_train_nw # Needed for KPI
        st.session_state['y_train_lgb'] = y_train_lgb # Needed for KPI
        st.session_state['last_data_hash'] = data_hash

# ==================== DASHBOARD VISUALIZATION ====================
if 'models_trained' in st.session_state and st.session_state['models_trained']:
    
    # Retrieve from session state
    nw_model = st.session_state['nw_model']
    lgb_model = st.session_state['lgb_model']
    y_test_nw = st.session_state['y_test_nw']
    y_pred_nw = st.session_state['y_pred_nw']
    y_test_lgb = st.session_state['y_test_lgb']
    y_pred_lgb = st.session_state['y_pred_lgb']
    metrics_nw = st.session_state['metrics_nw']
    metrics_lgb = st.session_state['metrics_lgb']
    
    st.divider()
    
    # ==================== KPI CARDS ====================
    st.markdown("## 📌 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔮 Prediksi NW (1 Jam)",
            f"${y_pred_nw[-1]:.2f}",
            f"{((y_pred_nw[-1] - y_test_nw[-1])/y_test_nw[-1]*100):.2f}%"
        )
    
    with col2:
        st.metric(
            "🔮 Prediksi LightGBM (1 Jam)",
            f"${y_pred_lgb[-1]:.2f}",
            f"{((y_pred_lgb[-1] - y_test_lgb.iloc[-1])/y_test_lgb.iloc[-1]*100):.2f}%"
        )
    
    with col3:
        st.metric("📊 Data Training NW", f"{len(st.session_state['y_train_nw']):,}")
        st.metric("📊 Data Testing NW", f"{len(y_test_nw):,}")
    
    with col4:
        st.metric("📊 Data Training LGB", f"{len(st.session_state['y_train_lgb']):,}")
        st.metric("📊 Data Testing LGB", f"{len(y_test_lgb):,}")
    
    st.divider()
    
    # ==================== PREDIKSI HARGA BERIKUTNYA ====================
    st.markdown("## 🔮 Prediksi Harga 1 Jam Kedepan")
    
    # Ambil data terakhir untuk prediksi
    last_data = df.iloc[-1:].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Nadaraya-Watson Prediction")
        
        # Prepare data untuk NW (menggunakan time_num terakhir)
        X_last_nw = df['time_num'].iloc[-1]
        pred_nw_next = nw_model.predict(np.array([X_last_nw]))[0]
        
        st.metric(
            label="Prediksi Harga Close 1 Jam Kedepan",
            value=f"${pred_nw_next:.2f}",
            delta=f"{((pred_nw_next - last_data['close'].values[0]) / last_data['close'].values[0] * 100):.2f}%",
            help="Prediksi berdasarkan data terakhir menggunakan Nadaraya-Watson Kernel Regression"
        )
        
        st.info(f"""
        **Data Input Terakhir:**
        - Time Num: {X_last_nw:.0f}
        - Close: ${last_data['close'].values[0]:.2f}
        """)
    
    with col2:
        st.markdown("### 🟢 LightGBM Prediction")
        
        # Prepare data untuk LGB (menggunakan lag features dari data terakhir)
        if len(df) >= 2:
            # Ambil data t-1 untuk lag features
            lag_data = df.iloc[-2:-1].copy()
            
            # Prepare input as pandas DataFrame to match training format
            X_last_lgb_df = pd.DataFrame({
                'open_lag1': [float(lag_data['open'].values[0])],
                'high_lag1': [float(lag_data['high'].values[0])],
                'low_lag1': [float(lag_data['low'].values[0])],
                'volume_lag1': [float(lag_data['volume'].values[0])]
            })
            
            try:
                # Use safe_predict method to bypass validation issues
                if hasattr(lgb_model, 'safe_predict'):
                    pred_array = lgb_model.safe_predict(X_last_lgb_df)
                    pred_lgb_next = float(pred_array[0])
                else:
                    # Fallback to regular predict
                    pred_lgb_next = float(lgb_model.predict(X_last_lgb_df.values)[0])
                
                st.metric(
                    label="Prediksi Harga Close 1 Jam Kedepan",
                    value=f"${pred_lgb_next:.2f}",
                    delta=f"{((pred_lgb_next - last_data['close'].values[0]) / last_data['close'].values[0] * 100):.2f}%",
                    help="Prediksi berdasarkan data terakhir menggunakan LightGBM"
                )
                
                st.info(f"""
                **Data Input Terakhir (Lag 1):**
                - Open (t-1): ${lag_data['open'].values[0]:.2f}
                - High (t-1): ${lag_data['high'].values[0]:.2f}
                - Low (t-1): ${lag_data['low'].values[0]:.2f}
                - Volume (t-1): {lag_data['volume'].values[0]:.0f}
                """)
            except Exception as e:
                st.error(f"Error predicting with LightGBM: {str(e)}")
                
                # Ultimate fallback: Use last prediction from test set
                try:
                    st.info("Using fallback: Getting prediction from existing test predictions...")
                    # Use the last prediction from test set as a proxy
                    pred_lgb_next = float(y_pred_lgb[-1])
                    
                    st.warning(f"⚠️ Using last test prediction as proxy: ${pred_lgb_next:.2f}")
                    st.metric(
                        label="Prediksi Harga Close 1 Jam Kedepan (Proxy)",
                        value=f"${pred_lgb_next:.2f}",
                        delta=f"{((pred_lgb_next - last_data['close'].values[0]) / last_data['close'].values[0] * 100):.2f}%",
                        help="Menggunakan prediksi terakhir dari test set sebagai proxy"
                    )
                except Exception as e2:
                    st.error(f"All prediction methods failed. Error: {str(e2)}")
                    pred_lgb_next = None
        else:
            st.warning("Data tidak cukup untuk prediksi LightGBM")
            pred_lgb_next = None
    
    # Perbandingan prediksi
    if len(df) >= 2 and pred_lgb_next is not None:
        st.markdown("### ⚖️ Perbandingan Hasil Prediksi")
        
        # New: Model Equation / Input Table
        st.markdown("#### 📝 Detail Input Model (Persamaan Akhir)")
        st.caption("Berikut adalah variabel input yang digunakan oleh model untuk menghasilkan prediksi di atas.")
        
        input_data = {
            'Variabel': ['Time (t)', 'Open (t-1)', 'High (t-1)', 'Low (t-1)', 'Volume (t-1)'],
            'Nilai Input': [
                f"{X_last_nw:.0f} (Timestamp)",
                f"${lag_data['open'].values[0]:.2f}",
                f"${lag_data['high'].values[0]:.2f}",
                f"${lag_data['low'].values[0]:.2f}", 
                f"{lag_data['volume'].values[0]:,.0f}"
            ],
            'Digunakan Oleh': ['Nadaraya-Watson', 'LightGBM', 'LightGBM', 'LightGBM', 'LightGBM']
        }
        st.table(pd.DataFrame(input_data))
        
        st.divider()
        
        # Comparison Table (Stable)
        st.markdown("#### 📊 Ringkasan Perbandingan Prediksi")
        pred_comparison = pd.DataFrame({
            'Model': ['Nadaraya-Watson', 'LightGBM'],
            'Prediksi Harga ($)': [f"{pred_nw_next:.2f}", f"{pred_lgb_next:.2f}"],
            'Perubahan (%)': [
                f"{((pred_nw_next - last_data['close'].values[0]) / last_data['close'].values[0] * 100):.2f}%",
                f"{((pred_lgb_next - last_data['close'].values[0]) / last_data['close'].values[0] * 100):.2f}%"
            ]
        })
        st.table(pred_comparison) # Stabilized using st.table instead of dataframe
        
        # Rekomendasi
        diff = pred_nw_next - pred_lgb_next
        if abs(diff) < 0.01:
            st.success("🤝 **Konsensus:** Kedua model memberikan prediksi yang sangat mirip (Converged).")
        elif diff > 0:
            st.info(f"📈 **Divergensi:** NW lebih optimis (+${diff:.2f}) dibanding LGB.")
        else:
            st.info(f"📈 **Divergensi:** LGB lebih optimis (+${abs(diff):.2f}) dibanding NW.")
        
        st.divider()
        
        # === NEW: TABEL PERBANDINGAN DETAIL PREDIKSI ===
        st.markdown("#### 🔍 Tabel Perbandingan Detail Kedua Model")
        st.caption("Tabel ini membandingkan hasil prediksi dari kedua metode untuk 10 data terakhir dalam test set.")
        
        # Get last 10 predictions from test set
        n_compare = min(10, len(y_test_nw))
        comparison_detail = pd.DataFrame({
            'Index': range(len(y_test_nw) - n_compare, len(y_test_nw)),
            'Harga Aktual ($)': [f"{val:.2f}" for val in y_test_nw[-n_compare:]],
            'Pred. NW ($)': [f"{val:.2f}" for val in y_pred_nw[-n_compare:]],
            'Error NW ($)': [f"{(pred - actual):.2f}" for actual, pred in zip(y_test_nw[-n_compare:], y_pred_nw[-n_compare:])],
            'Pred. LGB ($)': [f"{val:.2f}" for val in y_pred_lgb[-n_compare:]],
            'Error LGB ($)': [f"{(pred - actual):.2f}" for actual, pred in zip(y_test_lgb.values[-n_compare:], y_pred_lgb[-n_compare:])]
        })
        st.table(comparison_detail)  # Using st.table for stable display without jittering
        
        # Summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Rata-rata Error NW (10 data terakhir)",
                f"${np.mean([abs(pred - actual) for actual, pred in zip(y_test_nw[-n_compare:], y_pred_nw[-n_compare:])]):.2f}"
            )
        with col2:
            st.metric(
                "Rata-rata Error LGB (10 data terakhir)",
                f"${np.mean([abs(pred - actual) for actual, pred in zip(y_test_lgb.values[-n_compare:], y_pred_lgb[-n_compare:])]):.2f}"
            )
    
    
    st.divider()
    st.markdown("## 📊 Evaluasi Performa Model")
    
    # Create comparison table
    metrics_comparison = pd.DataFrame({
        'Metode': ['Nadaraya-Watson Kernel', 'LightGBM'],
        'Akurasi (%)': [f"{metrics_nw['Accuracy']:.2f}%", f"{metrics_lgb['Accuracy']:.2f}%"],
        'MAPE (%)': [f"{metrics_nw['MAPE']:.2f}%", f"{metrics_lgb['MAPE']:.2f}%"],
        'RMSE': [f"{metrics_nw['RMSE']:.4f}", f"{metrics_lgb['RMSE']:.4f}"],
        'R² Score': [f"{metrics_nw['R2']:.4f}", f"{metrics_lgb['R2']:.4f}"],
        'MAE': [f"{metrics_nw['MAE']:.4f}", f"{metrics_lgb['MAE']:.4f}"]
    })
    
    st.table(metrics_comparison)
    
    # Determine best model
    comparison = compare_models(metrics_nw, metrics_lgb, "Nadaraya-Watson", "LightGBM")
    
    if comparison['overall_winner'] == "Nadaraya-Watson":
        st.success(f"🏆 **Winner:** Nadaraya-Watson (Unggul di {comparison['winner_count']['Nadaraya-Watson']} metrik)")
    else:
        st.success(f"🏆 **Winner:** LightGBM (Unggul di {comparison['winner_count']['LightGBM']} metrik)")
    
    st.divider()
    
    # ==================== LINE CHART FORECASTING ====================
    # Using new plot_combined_analysis for cleaner look
    from utils.visualizations import plot_combined_analysis
    
    st.markdown("## 📈 Analisis Visual: Aktual vs Prediksi")
    
    # Plot comparison (Global View)
    fig_combined = plot_combined_analysis(
        y_test_nw, y_pred_nw,
        y_test_lgb.values, y_pred_lgb,
        model1_name="Nadaraya-Watson",
        model2_name="LightGBM"
    )
    st.plotly_chart(fig_combined, use_container_width=True)
    
    st.info("""
    💡 **Interpretasi Visual:**
    - **Garis Putus-putus (Hitam):** Data harga aktual.
    - **Garis Biru:** Prediksi model Nadaraya-Watson (Smoother).
    - **Garis Hijau:** Prediksi model LightGBM (More reactive).
    """)
    
    st.divider()
    

    
    # ==================== DETAILED ANALYSIS ====================
    st.markdown("## 🔍 Analisis Detail")
    
    analysis_tabs = st.tabs(["📉 Error Distribution", "🎯 Scatter Plot", "📊 Residual Plot", "📈 GCV Curve (NW)"])
    
    # Tab 1: Error Distribution
    with analysis_tabs[0]:
        st.markdown("### Distribusi Error Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Nadaraya-Watson")
            fig_err_nw = plot_error_distribution(
                y_test_nw, y_pred_nw,
                title="Error Distribution - Nadaraya-Watson"
            )
            st.plotly_chart(fig_err_nw, use_container_width=True)
            
            st.info(f"""
            **Statistik Error:**
            - Mean Error: ${metrics_nw['Mean_Error']:.4f}
            - Std Error: ${metrics_nw['Std_Error']:.4f}
            - Max Error: ${metrics_nw['Max_Error']:.4f}
            """)
        
        with col2:
            st.markdown("#### LightGBM")
            fig_err_lgb = plot_error_distribution(
                y_test_lgb, y_pred_lgb,
                title="Error Distribution - LightGBM"
            )
            st.plotly_chart(fig_err_lgb, use_container_width=True)
            
            st.info(f"""
            **Statistik Error:**
            - Mean Error: ${metrics_lgb['Mean_Error']:.4f}
            - Std Error: ${metrics_lgb['Std_Error']:.4f}
            - Max Error: ${metrics_lgb['Max_Error']:.4f}
            """)
        
        st.markdown("""
        💡 **Interpretasi:** 
        - Distribusi error yang mendekati normal (bell curve) menandakan model yang baik
        - Mean error mendekati 0 menunjukkan model tidak bias
        - Std error yang kecil menunjukkan prediksi yang konsisten
        """)
    
    # Tab 2: Scatter Plot
    with analysis_tabs[1]:
        st.markdown("### Scatter Plot: Actual vs Predicted")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Nadaraya-Watson")
            fig_scatter_nw = plot_scatter_actual_vs_predicted(
                y_test_nw, y_pred_nw,
                title="Actual vs Predicted - Nadaraya-Watson"
            )
            st.plotly_chart(fig_scatter_nw, use_container_width=True)
        
        with col2:
            st.markdown("#### LightGBM")
            fig_scatter_lgb = plot_scatter_actual_vs_predicted(
                y_test_lgb, y_pred_lgb,
                title="Actual vs Predicted - LightGBM"
            )
            st.plotly_chart(fig_scatter_lgb, use_container_width=True)
        
        st.markdown("""
        💡 **Interpretasi:** 
        - Titik yang dekat dengan garis merah (perfect prediction) menunjukkan prediksi yang akurat
        - Penyebaran yang tight menunjukkan konsistensi model
        - R² score yang tinggi terlihat dari kedekatan titik dengan garis diagonal
        """)
    
    # Tab 3: Residual Plot
    with analysis_tabs[2]:
        st.markdown("### Residual Plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Nadaraya-Watson")
            fig_res_nw = plot_residual_plot(
                y_test_nw, y_pred_nw,
                title="Residual Plot - Nadaraya-Watson"
            )
            st.plotly_chart(fig_res_nw, use_container_width=True)
        
        with col2:
            st.markdown("#### LightGBM")
            fig_res_lgb = plot_residual_plot(
                y_test_lgb, y_pred_lgb,
                title="Residual Plot - LightGBM"
            )
            st.plotly_chart(fig_res_lgb, use_container_width=True)
        
        st.markdown("""
        💡 **Interpretasi:** 
        - Residual yang tersebar random di sekitar garis nol menunjukkan model yang baik
        - Pattern tertentu dalam residual mengindikasikan model belum capture semua informasi
        - Heteroskedastisitas terlihat jika variance residual tidak konstan
        """)
    
    # Tab 4: GCV Curve
    with analysis_tabs[3]:
        st.markdown("### GCV Curve - Bandwidth Selection (Nadaraya-Watson)")
        
        if nw_model.gcv_scores is not None and nw_model.h_values is not None:
            fig_gcv = plot_gcv_curve(
                nw_model.h_values,
                nw_model.gcv_scores,
                nw_model.h_optimal
            )
            st.plotly_chart(fig_gcv, use_container_width=True)
            
            st.info(f"""
            **Bandwidth Optimal:** h = {nw_model.h_optimal:.2f}
            
            - GCV minimum: {min(nw_model.gcv_scores):.4f}
            - Bandwidth terkecil: {min(nw_model.h_values):.2f}
            - Bandwidth terbesar: {max(nw_model.h_values):.2f}
            """)
            
            st.markdown("""
            💡 **Interpretasi:** 
            - Bandwidth optimal dipilih di titik minimum GCV score
            - Bandwidth terlalu kecil → overfitting (GCV tinggi)
            - Bandwidth terlalu besar → underfitting (GCV tinggi)
            - Optimal bandwidth memberikan balance terbaik antara bias dan variance
            """)
    
    st.divider()
    
    # ==================== SUMMARY & RECOMMENDATIONS ====================
    st.markdown("## 📝 Summary & Rekomendasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Nadaraya-Watson")
        st.markdown(f"""
        **Performance:**
        - Accuracy: {metrics_nw['Accuracy']:.2f}%
        - RMSE: ${metrics_nw['RMSE']:.4f}
        - R² Score: {metrics_nw['R2']:.4f}
        
        **Karakteristik:**
        - ✅ Fleksibel dan nonparametrik
        - ✅ Tidak butuh asumsi distribusi
        - ⚠️ Komputasi lebih lambat
        - ⚠️ Sensitif terhadap bandwidth
        
        **Rekomendasi:**
        Cocok untuk exploratory analysis dan ketika hubungan antar variabel tidak jelas.
        """)
    
    with col2:
        st.markdown("### 🟢 LightGBM")
        st.markdown(f"""
        **Performance:**
        - Accuracy: {metrics_lgb['Accuracy']:.2f}%
        - RMSE: ${metrics_lgb['RMSE']:.4f}
        - R² Score: {metrics_lgb['R2']:.4f}
        
        **Karakteristik:**
        - ✅ Sangat cepat dan efisien
        - ✅ Handle kompleksitas tinggi
        - ✅ Built-in feature importance
        - ⚠️ Risk overfitting pada data kecil
        
        **Rekomendasi:**
        Cocok untuk production use dengan dataset besar dan memerlukan prediksi real-time.
        """)
    
    # Final recommendation
    winner = comparison['overall_winner']
    st.success(f"""
    ### 🎯 Kesimpulan
    
    Berdasarkan evaluasi komprehensif dengan berbagai metrik, **{winner}** memberikan 
    performa yang lebih baik untuk prediksi harga Solana 1 jam ke depan pada dataset ini.
    
    Namun, pemilihan model terbaik juga harus mempertimbangkan:
    - Kebutuhan komputasi dan kecepatan
    - Interpretability model
    - Kemudahan deployment
    - Maintenance cost
    """)

else:
    st.info("👆 Klik tombol **Train Models** untuk memulai training dan melihat dashboard visualisasi.")

# ==================== FOOTER ====================
add_vertical_space(2)

st.markdown("""
---
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>📈 <strong>Dashboard Prediksi</strong> | Powered by Streamlit & Plotly</p>
    <p style='font-size: 12px;'>Real-time Analysis & Visualization</p>
</div>
""", unsafe_allow_html=True)

# Slide Navigator
render_slide_navigator(3, 4)