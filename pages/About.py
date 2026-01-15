"""
About Page - Solana Price Prediction App
Menampilkan informasi tentang Solana, tujuan penelitian, dan dataset
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from assets.dataset_info import DATASET_INFO, COLUMN_DESCRIPTIONS, SOLANA_INFO, RESEARCH_OBJECTIVES
from utils.styling import apply_custom_css, render_header, render_section_header, render_info_box, add_vertical_space, render_slide_navigator
from utils.visualizations import plot_time_series, plot_correlation_heatmap, plot_box_plot

# Apply custom CSS
apply_custom_css()

# ==================== HEADER ====================
render_header(
    title="🪙 Cryptocurrency Price Prediction",
    subtitle="Advanced Machine Learning Analysis: Nadaraya-Watson vs LightGBM",
    icon="🤖"
)

# Professional ML Engineer Badge
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <div style='display: inline-block; background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%); 
                color: white; padding: 0.75rem 1.5rem; border-radius: 25px; 
                font-weight: 600; font-size: 0.9rem; box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3);'>
        🔬 MACHINE LEARNING ENGINEER PROJECT 🔬
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SECTION 1: TENTANG KUCOIN ====================
render_section_header("Tentang KuCoin", "🪙")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    **KuCoin** adalah platform mata uang kripto terkemuka dunia yang dipercaya oleh lebih dari 40 juta pengguna di lebih dari 200 negara dan wilayah. 
    Diluncurkan pada tahun **{SOLANA_INFO['launch_year']}** oleh **{SOLANA_INFO['founder']}**, 
    KuCoin menawarkan beragam pilihan perdagangan cryptocurrency.
    
    {SOLANA_INFO['description']}
    """)
    
    st.markdown("### 🌟 Fitur Utama KuCoin")
    for feature in SOLANA_INFO['key_features']:
        st.markdown(f"- {feature}")

with col2:
    # Display KuCoin logo
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%); 
                border-radius: 15px; color: white;'>
        <div style='font-size: 60px; margin-bottom: 10px; color: #fbbf24;'>🪙</div>
        <h2 style='margin: 0; color: white; font-size: 24px;'>KUCOIN</h2>
        <p style='margin: 5px 0; font-size: 16px; color: #e0e7ff;'>KCS</p>
        <div style='background: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px; margin-top: 12px;'>
            <p style='margin: 0; font-size: 12px; color: white;'>Cryptocurrency Exchange</p>
            <p style='margin: 0; font-size: 12px; color: #fbbf24;'>Founded 2017</p>
        </div>
        <div style='margin-top: 15px;'>
            <a href='https://www.kucoin.com/docs-new/introduction' target='_blank' 
               style='color: #fbbf24; text-decoration: none; font-size: 12px; font-weight: 600;'>
               📚 API Documentation →
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

add_vertical_space(1)

# Technology explanation
with st.expander("🔧 Teknologi KuCoin: Trading Engine & Security"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Trading Engine")
        st.markdown(SOLANA_INFO['technology']['Trading_Engine'])
    
    with col2:
        st.markdown("#### Security System")
        st.markdown(SOLANA_INFO['technology']['Security'])

# Use cases
st.markdown("### 💡 Use Cases KuCoin")
cols = st.columns(3)
use_cases = SOLANA_INFO['use_cases']
for i, col in enumerate(cols):
    with col:
        for use_case in use_cases[i::3]:
            st.markdown(f"""
            <div style='background-color: #f8fafc; padding: 15px; border-radius: 8px; 
                        margin-bottom: 10px; border-left: 4px solid #667eea;'>
                {use_case}
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ==================== SECTION 2: SUMBER DATA ====================
render_section_header("Sumber Data", "📊")

st.markdown("""
### 🔍 Data Sources Overview

Penelitian ini menggunakan data historis harga cryptocurrency yang diperoleh dari berbagai sumber terpercaya untuk memastikan akurasi dan reliabilitas analisis.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🪙 Solana (SOL) - Data dari KuCoin")
    st.markdown("""
    **Solana (SOL)** adalah platform blockchain high-performance yang revolusioner, dirancang khusus untuk mengatasi keterbatasan blockchain generasi sebelumnya dalam hal skalabilitas dan kecepatan.

    **🏗️ Arsitektur Solana:**
    - **Proof of Stake (PoS)**: Konsensus yang efisien dengan delegasi stake
    - **Proof of History (PoH)**: Mekanisme unik untuk mencatat urutan peristiwa dengan timestamp kriptografis
    - **Tower BFT**: Algoritma konsensus yang dioptimalkan untuk throughput tinggi
    - **Gulf Stream**: Protokol mempool-less untuk memproses transaksi secara preemptif

    **🚀 Performa Unggul:**
    - **Throughput**: 65,000+ TPS (Transaksi Per Detik)
    - **Latency**: <400ms waktu konfirmasi blok
    - **Biaya**: ~$0.00025 per transaksi
    - **Finality**: Konfirmasi instan dengan sub-second

    **💎 Ekosistem Solana:**
    - **DeFi**: Platform terbesar kedua setelah Ethereum dengan TVL >$2B
    - **NFT**: Marketplace seperti Magic Eden dengan volume trading harian jutaan
    - **Web3 Apps**: 350+ dApps aktif termasuk wallet, DEX, lending protocol
    - **Enterprise**: Adopsi oleh perusahaan besar untuk solusi blockchain

    **📊 Data Historis dari KuCoin:**
    - **Sumber**: KuCoin Exchange - platform terpercaya dengan 40M+ users
    - **Pair**: SOL/USDT (Stablecoin pegged to USD)
    - **Interval**: 1 jam (1h) candlestick data
    - **Kolom**: Timestamp, Open, High, Low, Close, Volume
    - **Periode**: Historical data lengkap dari listing hingga terkini
    - **Kualitas**: Data real-time dengan latensi minimal
    """)

with col2:
    st.markdown("### 📈 Sumber Data KuCoin Exchange")
    st.markdown("""
    **KuCoin sebagai Data Provider:**

    **🏆 Posisi KuCoin:**
    - **Top 3** Global Exchange berdasarkan volume trading
    - **40+ juta** pengguna terdaftar di 200+ negara
    - **500+** cryptocurrency tersedia untuk trading
    - **24/7** operasional dengan uptime 99.9%

    **🔗 API Integration:**
    - **REST API**: Untuk historical data retrieval
    - **WebSocket**: Real-time data streaming
    - **Rate Limits**: 10,000 requests/hour untuk developer
    - **Authentication**: Secure API key system

    **📋 Data Characteristics:**
    - **Format**: JSON response, convertible to CSV
    - **Granularity**: 1m, 5m, 15m, 1h, 1d intervals
    - **Depth**: Full OHLCV (Open, High, Low, Close, Volume)
    - **Quality**: Enterprise-grade data dengan validation
    - **Coverage**: Historical data sejak 2017

    **✨ Mengapa KuCoin?**
    - **Liquidity**: Volume trading SOL tertinggi
    - **Reliability**: 99.9% uptime dengan backup systems
    - **Transparency**: Public API documentation
    - **Support**: Developer community aktif
    """)

# Data collection methodology
st.markdown("### 🔧 Metodologi Pengumpulan Data dari KuCoin")
with st.expander("Detail Proses Pengumpulan Data"):
    st.markdown("""
    **1. KuCoin API Integration:**
    - Menggunakan **KuCoin Spot Market Data API** untuk historical data
    - Endpoint: `/api/v1/market/candles` untuk OHLCV data
    - Parameter: symbol=SOL-USDT, type=1hour, start/end timestamps
    - Authentication menggunakan API key untuk rate limit yang lebih tinggi

    **2. Data Retrieval Strategy:**
    - **Pagination**: Data diambil dalam batch untuk menghindari timeout
    - **Incremental Loading**: Data historis diambil dari periode terlama ke terkini
    - **Error Handling**: Retry mechanism untuk koneksi yang tidak stabil
    - **Rate Limiting**: Respect API limits dengan delay antar request

    **3. Data Cleaning & Validation:**
    - **Missing Values**: Handling gaps dalam data time series
    - **Outlier Detection**: Identifikasi anomali harga menggunakan statistical methods
    - **Data Consistency**: Cross-validation dengan multiple timeframes
    - **Format Standardization**: Konversi ke format pandas DataFrame

    **4. Data Storage & Caching:**
    - **Local Storage**: Data disimpan dalam format CSV untuk reproducibility
    - **Version Control**: Timestamp-based versioning untuk data updates
    - **Backup**: Multiple backup untuk data integrity
    - **Access Control**: Secure storage dengan encryption jika diperlukan

    **5. Quality Assurance:**
    - **Data Integrity Checks**: MD5 hash verification untuk data consistency
    - **Statistical Validation**: Distribution analysis untuk memastikan data quality
    - **Cross-Platform Verification**: Bandingkan dengan data dari sumber lain
    - **Documentation**: Comprehensive logging dari seluruh proses pengumpulan data
    """)

st.divider()

# ==================== SECTION 4: TUJUAN PENELITIAN ====================
render_section_header("Tujuan Penelitian", "🎯")

st.markdown(f"### Tujuan Utama")
st.info(RESEARCH_OBJECTIVES['main_objective'])

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Tujuan Spesifik")
    for i, obj in enumerate(RESEARCH_OBJECTIVES['specific_objectives'], 1):
        st.markdown(f"{i}. {obj}")

with col2:
    st.markdown("### ❓ Research Questions")
    for i, question in enumerate(RESEARCH_OBJECTIVES['research_questions'], 1):
        st.markdown(f"{i}. {question}")

st.markdown("### 🎁 Manfaat Penelitian")
cols = st.columns(4)
for i, col in enumerate(cols):
    with col:
        if i < len(RESEARCH_OBJECTIVES['benefits']):
            benefit = RESEARCH_OBJECTIVES['benefits'][i]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; 
                        color: white; height: 120px; display: flex; 
                        align-items: center; justify-content: center;'>
                <div>{benefit}</div>
            </div>
            """, unsafe_allow_html=True)

add_vertical_space(1)

with st.expander("💭 Hipotesis Penelitian"):
    st.markdown(RESEARCH_OBJECTIVES['hypothesis'])

st.divider()

# ==================== SECTION 5: LOAD DATA ====================
render_section_header("Dataset", "📊")

st.markdown(f"""
### 📁 {DATASET_INFO['name']}

**Sumber Data:** {DATASET_INFO['source']}  
**Interval:** {DATASET_INFO['interval']}  

{DATASET_INFO['description']}
""")

# Initialize data loader
data_loader = DataLoader()

# Auto-load default data if not already loaded (for presentation)
if 'df' not in st.session_state or st.session_state['df'] is None:
    st.info("🔄 Loading default dataset...")
    file_path = "data/df_1h.csv"
    
    loader = DataLoader(file_path)
    df_default = loader.load_default_data()
    
    if df_default is not None:
        # Preprocess
        df_default = loader.preprocess_data(df_default)
        if df_default is not None:
            # Save to session - USE 'df' key to be consistent everywhere
            st.session_state['df'] = df_default
            st.session_state['data_loaded'] = True
            st.success("✅ Dataset loaded successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to preprocess data")
    else:
        st.error("❌ Failed to load default data")

# Get data from session state
df = st.session_state.get('df')

# Load data dengan 2 opsi (user can still upload or reload)
df_options = data_loader.load_data_with_options()

# Use the data from options if user loaded new data, otherwise use session data
if df_options is not None:
    df = df_options
    # Update session state with new data
    st.session_state['df'] = df_options


# ==================== DATA RELOAD & PERSISTENCE ====================
with st.sidebar.expander("🛠️ Advanced Data Tools"):
    if st.button("🚨 NUCLEAR RESET (Clear All Data)", help="Clear all session state and force fresh reload"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session state cleared!")
        st.rerun()

# ==================== SECTION 4: DATASET INFO & PREVIEW ====================
if df is not None:
    # DEBUG: Help identify why line might be flat
    if df['close'].nunique() <= 1:
        st.error("⚠️ Data 'close' terdeteksi konstan atau tidak valid. Coba klik 'NUCLEAR RESET' di sidebar.")
    
    st.success("✅ Data berhasil dimuat!")

    # Display dataset info
    dataset_info = data_loader.get_dataset_info(df)


    st.markdown("### 📊 Informasi Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Total Data",
            value=f"{dataset_info['total_rows']:,}",
            help="Jumlah total observasi dalam dataset"
        )

    with col2:
        st.metric(
            label="📋 Jumlah Kolom",
            value=dataset_info['total_columns'],
            help="Jumlah variabel dalam dataset"
        )

    with col3:
        st.metric(
            label="📅 Durasi",
            value=f"{dataset_info['duration_days']} hari",
            help="Total durasi periode data"
        )

    with col4:
        st.metric(
            label="⚠️ Missing Values",
            value=dataset_info['missing_values'],
            help="Jumlah data yang hilang"
        )

    # Date range
    st.info(f"📅 **Periode Data:** {dataset_info['date_range']}")

    st.divider()

    # ==================== PREVIEW DATASET ====================
    st.markdown("### 👀 Preview Dataset")

    tab1, tab2, tab3 = st.tabs(["📋 Data Sample", "📊 Statistik Deskriptif", "ℹ️ Info Kolom"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔝 5 Data Pertama**")
            st.table(df.head())

        with col2:
            st.markdown("**🔚 5 Data Terakhir**")
            st.table(df.tail())

    with tab2:
        st.markdown("**📊 Statistik Deskriptif**")
        st.table(df.describe())

        # Additional statistics
        st.markdown("**📈 Statistik Tambahan**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rata-rata Close", f"${df['close'].mean():.2f}")
            st.metric("Median Close", f"${df['close'].median():.2f}")

        with col2:
            st.metric("Harga Tertinggi", f"${df['close'].max():.2f}")
            st.metric("Harga Terendah", f"${df['close'].min():.2f}")

        with col3:
            st.metric("Std Deviation", f"${df['close'].std():.2f}")
            st.metric("Range", f"${df['close'].max() - df['close'].min():.2f}")

    with tab3:
        st.markdown("**ℹ️ Deskripsi Kolom Dataset**")

        # Create table for column descriptions
        col_info = []
        for col_name, info in COLUMN_DESCRIPTIONS.items():
            col_info.append({
                'Kolom': info['name'],
                'Tipe Data': info['type'],
                'Deskripsi': info['description'],
                'Role': info['role']
            })

        col_df = pd.DataFrame(col_info)
        st.table(col_df)

        # Highlight target variable
        st.success("🎯 **Target Variable (Y):** close - Harga penutupan yang akan diprediksi 1 jam ke depan")

    st.divider()

    st.divider()

    # ==================== DATA VISUALIZATION ====================
    st.markdown("### 📈 Visualisasi Eksploratori")

    if df is not None:
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Time Series", "🔗 Correlation", "📦 Outliers"])

        with viz_tab1:
            st.markdown("#### 📈 Harga Close (Time Series)")
            st.caption("Pergerakan harga Solana (SOL) terhadap waktu dalam satuan jam.")
            
            # Ensure data is properly formatted before plotting
            if 'datetime' in df.columns and 'close' in df.columns:
                # Verify data integrity
                df_plot = df[['datetime', 'close']].dropna().sort_values('datetime')
                
                if len(df_plot) > 0:
                    fig_ts = plot_time_series(df_plot, y_column='close', title="Pergerakan Harga Solana (SOL)")
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    st.info("""
                    💡 **Analisis Tren:**
                    Grafik di atas menampilkan fluktuasi harga penutupan (Close Price).
                    - **Garis Biru:** Harga aktual setiap jam.
                    - **Garis Putus-putus:** Moving Average (24 jam) untuk melihat tren harian.
                    """)
                else:
                    st.error("❌ Data tidak valid untuk visualisasi")
            else:
                st.error("❌ Kolom 'datetime' atau 'close' tidak ditemukan dalam dataset")

        with viz_tab2:
            st.markdown("#### 🔗 Korelasi Antar Variabel")
            fig_corr = plot_correlation_heatmap(df)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("""
            💡 **Insight:** 
            Warna merah pekat menunjukkan korelasi positif yang kuat (mendekati 1).
            Hampir semua fitur harga (Open, High, Low, Close) memiliki korelasi sangat tinggi, yang wajar dalam data keuangan.
            """)

        with viz_tab3:
            st.markdown("#### 📦 Deteksi Outlier (Box Plot)")
            fig_box = plot_box_plot(df)
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.info("💡 Titik-titik di luar kotak menandakan **outlier** atau anomali harga yang ekstrim.")
    else:
        st.error("❌ Data tidak tersedia. Mohon cek Data Loading.")

    st.divider()

    # ==================== DATA QUALITY CHECK ====================
    st.markdown("### ✅ Data Quality Check")

    col1, col2, col3 = st.columns(3)

    with col1:
        if dataset_info['missing_values'] == 0:
            st.success("✅ Tidak ada missing values")
        else:
            st.warning(f"⚠️ Terdapat {dataset_info['missing_values']} missing values")

    with col2:
        if dataset_info['duplicate_rows'] == 0:
            st.success("✅ Tidak ada data duplikat")
        else:
            st.warning(f"⚠️ Terdapat {dataset_info['duplicate_rows']} baris duplikat")

    with col3:
        if df['close'].isnull().sum() == 0:
            st.success("✅ Target variable lengkap")
        else:
            st.error("❌ Target variable memiliki missing values")

    # Ready for modeling
    if dataset_info['missing_values'] == 0 and df['close'].isnull().sum() == 0:
        st.success("🎉 **Dataset siap untuk modeling!** Silakan lanjut ke menu Metodologi atau Dashboard.")
    else:
        st.warning("⚠️ **Perhatian:** Dataset memerlukan preprocessing lebih lanjut sebelum modeling.")

else:
    # Jika data belum dimuat
    st.warning("⚠️ Silakan pilih salah satu opsi untuk memuat data.")


    render_info_box(
        "Informasi",
        """
        Untuk memulai analisis, Anda dapat:
        1. **Gunakan Dataset Default** - Klik tombol "Load Dataset Default" di atas
        2. **Upload CSV Sendiri** - Pilih file CSV dengan format yang sesuai

        Format CSV yang diperlukan:
        - Kolom: datetime, open, high, low, close, volume
        - Datetime dalam format: YYYY-MM-DD HH:MM:SS
        - Semua nilai numerik untuk kolom harga dan volume
        """,
        type="info"
    )

# ==================== TECHNICAL SPECIFICATIONS ====================
render_section_header("🔧 Technical Specifications", "⚙️")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🤖 Machine Learning
    - **Nadaraya-Watson Kernel Regression**
    - **LightGBM Gradient Boosting**
    - **Cross-Validation & GCV**
    - **Hyperparameter Optimization**
    """)

with col2:
    st.markdown("""
    #### 📊 Data Science Stack
    - **Python 3.8+**
    - **Scikit-learn, NumPy, Pandas**
    - **Plotly, Streamlit**
    - **Statistical Analysis**
    """)

with col3:
    st.markdown("""
    #### 🎯 Performance Metrics
    - **RMSE, MAE, MAPE**
    - **R² Score, Accuracy**
    - **Directional Accuracy**
    - **Cross-validation Scores**
    """)

st.divider()

# ==================== FOOTER ====================
add_vertical_space(2)

st.markdown("""
---
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <div style='margin-bottom: 1rem;'>
        <span style='background: linear-gradient(135deg, #1e40af, #3b82f6); 
                     color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                     font-weight: 600; font-size: 0.9rem;'>
            🚀 Built with Professional ML Engineering Practices
        </span>
    </div>
    <p>🤖 <strong>Cryptocurrency Price Prediction Project</strong></p>
    <p>Advanced Time Series Analysis | Machine Learning Engineering</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        Powered by Python • Scikit-learn • LightGBM • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# Slide Navigator
render_slide_navigator(1, 4)