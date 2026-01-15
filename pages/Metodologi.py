"""
Metodologi Page - Solana Price Prediction App
Menjelaskan kedua metode: Nadaraya-Watson Kernel dan LightGBM
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.styling import apply_custom_css, render_header, render_section_header, render_info_box, add_vertical_space, render_slide_navigator
from models.nadaraya_watson import get_nw_explanation
from models.lightgbm_model import get_lgb_explanation, get_preprocessing_steps

# Apply custom CSS
apply_custom_css()

# ==================== HEADER ====================
render_header(
    title="📚 Metodologi Penelitian",
    subtitle="Penjelasan Lengkap: Nadaraya-Watson Kernel & LightGBM",
    icon="🔬"
)

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["🔵 Nadaraya-Watson Kernel", "🟢 LightGBM", "⚙️ ML Process & Preprocessing"])

# ==================== TAB 1: NADARAYA-WATSON ====================
with tab1:
    render_section_header("Metode Nadaraya-Watson Kernel Regression", "🔵")
    
    # Get explanation
    nw_exp = get_nw_explanation()
    
    # Concept
    st.markdown("### 📖 Penjelasan Konsep")
    st.markdown(nw_exp['concept'])
    
    add_vertical_space(1)
    
    # Mathematical Formula
    st.markdown("### 📐 Rumus Matematis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Formula Nadaraya-Watson Estimator:**")
        st.latex(nw_exp['formula'])
        
        st.markdown("""
        **di mana:**
        - $K(⋅)$ = fungsi kernel
        - $h$ = bandwidth
        - $X_i,Y_i$ = data pengamatan
        """)
    
    with col2:
        st.markdown("**Estimasi Nilai Regresi:**")
        st.latex(nw_exp['estimation_formula'])
        
        st.markdown("""
        **dengan bobot:**
        """)
        st.latex(nw_exp['weight_formula'])
    
    add_vertical_space(1)
    
    # How it works
    st.markdown("### 🔧 Cara Kerja")
    
    st.markdown("""
    Nadaraya-Watson menggunakan **weighted average** untuk mengestimasi nilai target. 
    Bobot ditentukan oleh jarak antara titik prediksi dengan titik observasi melalui fungsi kernel.
    """)
    
    # Visual explanation with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center; height: 200px;
                    display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 40px; margin-bottom: 10px;'>1️⃣</div>
            <h4 style='margin: 0; color: white;'>Calculate Distance</h4>
            <p style='margin-top: 10px; font-size: 14px;'>
                Hitung jarak antara titik prediksi dengan semua data training
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center; height: 200px;
                    display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 40px; margin-bottom: 10px;'>2️⃣</div>
            <h4 style='margin: 0; color: white;'>Apply Kernel</h4>
            <p style='margin-top: 10px; font-size: 14px;'>
                Transformasi jarak menjadi bobot menggunakan kernel function
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center; height: 200px;
                    display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 40px; margin-bottom: 10px;'>3️⃣</div>
            <h4 style='margin: 0; color: white;'>Weighted Average</h4>
            <p style='margin-top: 10px; font-size: 14px;'>
                Hitung weighted average dari nilai target
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    add_vertical_space(2)
    
    # Bandwidth Selection
    st.markdown("### 🎯 Pemilihan Bandwidth (h)")
    
    st.markdown(nw_exp['bandwidth_explanation'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.warning("""
        **h terlalu kecil**
        - Over-fitting
        - Kurva terlalu bergelombang
        - Sensitif terhadap noise
        """)
    
    with col2:
        st.success("""
        **h optimal**
        - Balance bias-variance
        - Smooth tapi tetap capture pattern
        - Dipilih menggunakan GCV
        """)
    
    with col3:
        st.warning("""
        **h terlalu besar**
        - Under-fitting
        - Kurva terlalu smooth
        - Kehilangan detail penting
        """)
    
    st.info("""
    💡 **Generalized Cross-Validation (GCV)** adalah metode untuk memilih bandwidth optimal 
    dengan meminimalkan prediction error tanpa melakukan k-fold cross-validation yang mahal.
    """)
    
    add_vertical_space(1)
    
    # Implementation Steps
    st.markdown("### 📋 Langkah-Langkah Implementasi")
    
    for i, step in enumerate(nw_exp['steps'], 1):
        with st.expander(f"**Step {i}:** {step.split(':')[0]}", expanded=(i==1)):
            st.markdown(step)
            
            # Add code example for specific steps
            if i == 1:  # Preprocessing
                st.code("""
# Convert datetime to numeric timestamp
df['datetime'] = pd.to_datetime(df['datetime'])
df['time_num'] = df['datetime'].astype(np.int64) // 10**9

# Define X and y
X = df['time_num'].values
y = df['close'].values
                """, language='python')
            
            elif i == 2:  # Split data
                st.code("""
# Random split (80-20)
n = len(y)
np.random.seed(42)
train_idx = np.random.choice(n, size=int(0.8*n), replace=False)

X_train = X[train_idx]
y_train = y[train_idx]
X_test = np.delete(X, train_idx)
y_test = np.delete(y, train_idx)
                """, language='python')
            
            elif i == 4:  # GCV
                st.code("""
# Test different bandwidth values
h_values = np.linspace(1000, 50000, 40)
gcv_scores = [calculate_gcv(X_train, y_train, h) for h in h_values]

# Select optimal bandwidth
h_optimal = h_values[np.argmin(gcv_scores)]
print(f"Optimal bandwidth: {h_optimal}")
                """, language='python')
            
            elif i == 5:  # Estimation
                st.code("""
def nadaraya_watson_estimator(x0, X, y, h):
    # Calculate kernel weights
    u = (x0 - X) / h
    weights = kernel_gaussian(u)
    
    # Weighted average
    return np.sum(weights * y) / np.sum(weights)
                """, language='python')
    
    add_vertical_space(1)
    
    # Bandwidth Selection
    st.markdown("### 🎯 Pemilihan Bandwidth (h)")
    
    st.markdown("""
    **Bandwidth** adalah parameter paling penting dalam Nadaraya-Watson. Nilai h yang optimal 
    menentukan seberapa "smooth" kurva regresi yang dihasilkan.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.warning("""
        **h terlalu kecil**
        - Over-fitting
        - Kurva terlalu bergelombang
        - Sensitif terhadap noise
        """)
    
    with col2:
        st.success("""
        **h optimal**
        - Balance bias-variance
        - Smooth tapi tetap capture pattern
        - Dipilih menggunakan GCV
        """)
    
    with col3:
        st.warning("""
        **h terlalu besar**
        - Under-fitting
        - Kurva terlalu smooth
        - Kehilangan detail penting
        """)
    
    st.info("""
    💡 **Generalized Cross-Validation (GCV)** adalah metode untuk memilih bandwidth optimal 
    dengan meminimalkan prediction error tanpa melakukan k-fold cross-validation yang mahal.
    """)
    
    add_vertical_space(1)
    
    # Advantages & Disadvantages
    st.markdown("### ⚖️ Kelebihan & Kekurangan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Kelebihan")
        for adv in nw_exp['advantages']:
            st.markdown(adv)
    
    with col2:
        st.markdown("#### ❌ Kekurangan")
        for dis in nw_exp['disadvantages']:
            st.markdown(dis)

# ==================== TAB 2: LIGHTGBM ====================
with tab2:
    render_section_header("Metode LightGBM (Light Gradient Boosting Machine)", "🟢")
    
    # Get explanation
    lgb_exp = get_lgb_explanation()
    
    # Concept
    st.markdown("### 📖 Penjelasan Konsep")
    st.markdown(lgb_exp['concept'])
    
    add_vertical_space(1)
    
    # Mathematical Formula
    st.markdown("### 📐 Objective Function")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Formula Objective:**")
        st.latex(lgb_exp['formula'])
        
        st.markdown(lgb_exp['formula_explanation'])
    
    with col2:
        st.markdown("**Gradient Boosting Concept:**")
        st.markdown("""
        Gradient Boosting bekerja dengan membangun model secara **sekuensial**:
        
        1. Model pertama: $F_1(x)$
        2. Model kedua memperbaiki: $F_2(x) = F_1(x) + h_2(x)$
        3. Model ketiga: $F_3(x) = F_2(x) + h_3(x)$
        4. Dan seterusnya...
        
        Setiap model baru fokus pada **residual error** dari model sebelumnya.
        """)
    
    add_vertical_space(1)
    
    # Key Techniques
    st.markdown("### 🔑 Teknik Utama LightGBM")
    
    for technique, description in lgb_exp['key_techniques'].items():
        with st.expander(f"**{technique}**"):
            st.markdown(description)
            
            if technique == "Leaf-wise Growth":
                st.image("https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png", 
                         caption="Leaf-wise vs Level-wise Tree Growth")
    
    add_vertical_space(1)
    
    # Implementation Steps
    st.markdown("### 📋 Langkah-Langkah Implementasi")
    
    for i, step in enumerate(lgb_exp['steps'], 1):
        with st.expander(f"**Step {i}:** {step.split(':')[0]}", expanded=(i==1)):
            st.markdown(step)
            
            # Add code examples
            if i == 1:  # Menentukan Variabel
                st.code("""
# Variabel respon (target)
y = df['close']  # Harga close (kontinu)

# Variabel prediktor
X = df[['open', 'high', 'low', 'volume']]  # Fitur input
                """, language='python')
            
            elif i == 2:  # Persiapan Data
                st.code("""
# Handle missing values
df = df.dropna()  # atau df.fillna(method='ffill')

# Encoding kategorik (jika ada)
# df['category'] = df['category'].astype('category')

# Normalisasi opsional (LightGBM tidak wajib)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
                """, language='python')
            
            elif i == 3:  # Split Data
                st.code("""
# Import library
from sklearn.model_selection import train_test_split

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
                """, language='python')
            
            elif i == 4:  # Fungsi Loss
                st.code("""
# Loss functions untuk regresi:
# 'l2' - Mean Squared Error (MSE)
# 'l1' - Mean Absolute Error (MAE) 
# 'huber' - Huber loss

# Contoh penggunaan
loss_function = 'l2'  # MSE untuk regresi
                """, language='python')
            
            elif i == 5:  # Inisialisasi Model
                st.code("""
# Inisialisasi dengan mean (bias)
initial_prediction = np.mean(y_train)
print(f"Initial prediction (bias): {initial_prediction}")
                """, language='python')
            
            elif i == 6:  # Proses Boosting
                st.code("""
# LightGBM otomatis handle proses boosting
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=100,      # Jumlah trees
    learning_rate=0.1,     # Learning rate
    objective='regression' # Untuk regresi
)

# Training
model.fit(X_train, y_train)
                """, language='python')
            
            elif i == 7:  # Hyperparameter Tuning
                st.code("""
# Hyperparameter tuning
best_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': -1,        # No limit
    'num_leaves': 31,       # Max leaves
    'min_child_samples': 20 # Min samples in leaf
}

model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)
                """, language='python')
            
            elif i == 8:  # Evaluasi
                st.code("""
# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
from sklearn.metrics import mean_squared_error, r2_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
                """, language='python')
    
    add_vertical_space(1)
    
    # Hyperparameters
    st.markdown("### ⚙️ Hyperparameters Penting")
    
    for param, description in lgb_exp['hyperparameters'].items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**`{param}`**")
        with col2:
            st.markdown(description)
    
    st.info("""
    💡 **Tips Hyperparameter Tuning:**
    - Start dengan default parameters
    - Increase `n_estimators` untuk model lebih complex
    - Decrease `learning_rate` untuk training lebih stabil
    - Tune `num_leaves` untuk control overfitting
    - Use `early_stopping` untuk prevent overfitting
    """)
    
    add_vertical_space(1)
    
    # Advantages & Disadvantages
    st.markdown("### ⚖️ Kelebihan & Kekurangan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Kelebihan")
        for adv in lgb_exp['advantages']:
            st.markdown(adv)
    
    with col2:
        st.markdown("#### ❌ Kekurangan")
        for dis in lgb_exp['disadvantages']:
            st.markdown(dis)
    
    add_vertical_space(1)
    
    # Use Cases
    st.markdown("### 💼 Use Cases LightGBM")
    
    cols = st.columns(len(lgb_exp['use_cases']))
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div style='background-color: #dcfce7; padding: 15px; border-radius: 8px; 
                        text-align: center; border-left: 4px solid #10b981;'>
                {lgb_exp['use_cases'][i]}
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 3: ML PROCESS & PREPROCESSING ====================
with tab3:
    render_section_header("ML Process & Preprocessing", "⚙️")
    
    st.markdown("""
    Preprocessing adalah tahap krusial sebelum modeling. Kualitas data menentukan kualitas model!
    """)
    
    # Get preprocessing steps
    prep_steps = get_preprocessing_steps()
    
    # Timeline visualization
    st.markdown("### 📍 Pipeline Preprocessing")
    
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 3px; border-radius: 10px; margin: 20px 0;'>
        <div style='background: white; padding: 20px; border-radius: 8px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='text-align: center; flex: 1;'>
                    <div style='width: 50px; height: 50px; border-radius: 50%; 
                                background: #667eea; color: white; display: flex; 
                                align-items: center; justify-content: center; 
                                margin: 0 auto; font-weight: bold;'>1</div>
                    <p style='margin-top: 10px; font-size: 12px;'>Load Data</p>
                </div>
                <div style='flex: 0.5; height: 3px; background: #667eea;'></div>
                <div style='text-align: center; flex: 1;'>
                    <div style='width: 50px; height: 50px; border-radius: 50%; 
                                background: #667eea; color: white; display: flex; 
                                align-items: center; justify-content: center; 
                                margin: 0 auto; font-weight: bold;'>2</div>
                    <p style='margin-top: 10px; font-size: 12px;'>Clean Data</p>
                </div>
                <div style='flex: 0.5; height: 3px; background: #667eea;'></div>
                <div style='text-align: center; flex: 1;'>
                    <div style='width: 50px; height: 50px; border-radius: 50%; 
                                background: #667eea; color: white; display: flex; 
                                align-items: center; justify-content: center; 
                                margin: 0 auto; font-weight: bold;'>3</div>
                    <p style='margin-top: 10px; font-size: 12px;'>Feature Engineering</p>
                </div>
                <div style='flex: 0.5; height: 3px; background: #667eea;'></div>
                <div style='text-align: center; flex: 1;'>
                    <div style='width: 50px; height: 50px; border-radius: 50%; 
                                background: #667eea; color: white; display: flex; 
                                align-items: center; justify-content: center; 
                                margin: 0 auto; font-weight: bold;'>4</div>
                    <p style='margin-top: 10px; font-size: 12px;'>Split Data</p>
                </div>
                <div style='flex: 0.5; height: 3px; background: #667eea;'></div>
                <div style='text-align: center; flex: 1;'>
                    <div style='width: 50px; height: 50px; border-radius: 50%; 
                                background: #10b981; color: white; display: flex; 
                                align-items: center; justify-content: center; 
                                margin: 0 auto; font-weight: bold;'>✓</div>
                    <p style='margin-top: 10px; font-size: 12px;'>Ready!</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    add_vertical_space(2)
    
    # Detailed steps
    st.markdown("### 📝 Detail Setiap Tahap")
    
    for step_info in prep_steps['steps']:
        with st.expander(f"**{step_info['step']}**", expanded=False):
            st.markdown(step_info['description'])
            
            if 'code_example' in step_info:
                st.code(step_info['code_example'], language='python')
    
    add_vertical_space(1)
    
    # Comparison: NW vs LGB preprocessing
    st.markdown("### 🔄 Perbedaan Preprocessing: NW vs LightGBM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔵 Nadaraya-Watson
        
        **Feature Engineering:**
        - Convert datetime → timestamp numerik
        - Single variable: `time_num`
        - Target: `close` price
        
        **Train-Test Split:**
        - Random sampling (80-20)
        - `shuffle=True`
        - Tidak menjaga urutan temporal
        
        **Rationale:**
        - NW tidak bergantung pada urutan waktu
        - Fokus pada jarak antar observasi
        - Random split menghindari bias temporal
        """)
    
    with col2:
        st.markdown("""
        #### 🟢 LightGBM
        
        **Feature Engineering:**
        - Create lag features (t-1)
        - Multiple features: open, high, low, volume lag
        - Target: `close` price
        
        **Train-Test Split:**
        - Temporal split (80-20)
        - `shuffle=False`
        - Menjaga urutan waktu
        
        **Rationale:**
        - Mensimulasikan prediksi real-time
        - Model hanya tahu data masa lalu
        - Temporal split lebih realistic untuk time series
        """)
    
    add_vertical_space(1)
    
    # Best Practices
    st.markdown("### 💡 Best Practices")
    
    best_practices = [
        "✅ Selalu cek missing values sebelum modeling",
        "✅ Validate data types untuk setiap kolom",
        "✅ Deteksi dan handle outliers dengan bijak (untuk crypto, outlier sering valid)",
        "✅ Visualize data sebelum preprocessing untuk understand pattern",
        "✅ Document setiap step preprocessing untuk reproducibility",
        "✅ Save cleaned dataset untuk future use",
        "✅ Check data leakage (jangan gunakan informasi dari future)",
        "✅ Validate bahwa tidak ada NaN setelah preprocessing"
    ]
    
    for practice in best_practices:
        st.markdown(practice)

# ==================== FOOTER ====================
add_vertical_space(2)

st.markdown("""
---
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>📚 <strong>Metodologi Penelitian</strong> | Regresi Semiparametrik</p>
    <p style='font-size: 12px;'>Nadaraya-Watson Kernel vs LightGBM</p>
</div>
""", unsafe_allow_html=True)

# Slide Navigator
render_slide_navigator(2, 4)