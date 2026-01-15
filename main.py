"""
Main Application File - Solana Price Prediction
Streamlit App untuk Prediksi Harga Solana menggunakan Nadaraya-Watson dan LightGBM

Author: Kelompok Regresi Semiparametrik
Course: Regresi Semiparametrik
Year: 2025/2026
"""

import streamlit as st
import sys
import os

# Set page config (HARUS DI PALING ATAS!)
st.set_page_config(
    page_title="Cryptocurrency Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': """
        # Cryptocurrency Price Prediction App
        
        Aplikasi prediksi harga cryptocurrency menggunakan:
        - Nadaraya-Watson Kernel Regression
        - LightGBM (Gradient Boosting)
        
        Developed by: Kelompok Regresi Semiparametrik
        """
    }
)

# Import utilities
from utils.styling import apply_custom_css

# Apply custom CSS
apply_custom_css()

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.title("🪙 Crypto Prediction")
st.sidebar.markdown("---")

# Navigation menu
st.sidebar.markdown("### 📍 Navigation")

# Initialize page in session state if not exists
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "🏠 About"

# Initialize slide navigation index if not exists
if 'slide_navigation_index' not in st.session_state:
    st.session_state['slide_navigation_index'] = 0

page = st.sidebar.radio(
    "Pilih Menu:",
    [
        "🏠 About & Data Sources",
        "📚 Metodologi Penelitian", 
        "📈 Dashboard & Model Results",
        "👥 Developer"
    ],
    index=st.session_state['slide_navigation_index'],  # Use session state for index
    label_visibility="collapsed",
    key="main_navigation"
)

# Update session state when page changes
# (Removed to avoid conflict with slide navigation)

st.sidebar.markdown("---")

# ==================== SIDEBAR INFO ====================
st.sidebar.markdown("### ℹ️ Quick Info")

if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    st.sidebar.success(f"✅ Data Loaded: {len(df)} rows")
    
    if 'models_trained' in st.session_state and st.session_state['models_trained']:
        st.sidebar.success("✅ Models Trained")
    else:
        st.sidebar.info("ℹ️ Models not trained yet")
else:
    st.sidebar.warning("⚠️ No data loaded")

st.sidebar.markdown("---")

# ==================== SIDEBAR PROJECT INFO ====================
st.sidebar.markdown("### 📚 Project Info")
st.sidebar.markdown("""
**Course:** Regresi Semiparametrik  
**Year:** 2024/2025  
**Topic:** Price Prediction  

**Methods:**
- 🔵 Nadaraya-Watson Kernel
- 🟢 LightGBM
""")

st.sidebar.markdown("---")

# ==================== SIDEBAR TIPS ====================
with st.sidebar.expander("💡 Tips"):
    st.markdown("""
    **Cara Menggunakan Aplikasi:**
    
    1. **About** - Upload dataset atau gunakan default
    2. **Metodologi** - Pelajari kedua metode
    3. **Dashboard** - Train model & lihat hasil
    4. **Developer** - Info tim
    
    **Keyboard Shortcuts:**
    - `Ctrl + R` - Refresh page
    - `Ctrl + K` - Clear cache
    """)

# ==================== SIDEBAR SETTINGS ====================
with st.sidebar.expander("⚙️ Settings"):
    st.markdown("**Display Options:**")
    
    # Theme toggle (informational, karena Streamlit theme dari settings)
    st.info("💡 Change theme dari Settings → Theme")
    
    # Cache control
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    # Reset session
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session reset! Please refresh page.")

st.sidebar.markdown("---")

# ==================== SIDEBAR FOOTER ====================
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px; color: #64748b;'>
    <p style='font-size: 12px; margin: 0;'>Made with ❤️</p>
    <p style='font-size: 12px; margin: 5px 0;'>Powered by Streamlit</p>
    <p style='font-size: 10px; margin: 5px 0;'>© 2024/2025</p>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================
# Routing ke page yang dipilih

if page == "🏠 About & Data Sources":
    # Import dan execute About page
    import importlib.util
    spec = importlib.util.spec_from_file_location("about", "pages/About.py")
    about_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(about_module)

elif page == "📚 Metodologi Penelitian":
    # Import dan execute Metodologi page
    import importlib.util
    spec = importlib.util.spec_from_file_location("metodologi", "pages/Metodologi.py")
    metodologi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metodologi_module)

elif page == "📈 Dashboard & Model Results":
    # Import dan execute Dashboard page
    import importlib.util
    spec = importlib.util.spec_from_file_location("dashboard", "pages/Dashboard.py")
    dashboard_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard_module)

elif page == "👥 Developer":
    # Import dan execute Developer page
    import importlib.util
    spec = importlib.util.spec_from_file_location("developer", "pages/Developer.py")
    developer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(developer_module)

# ==================== FLOATING ACTION BUTTON (Optional) ====================
# Add a floating "Back to Top" button using custom HTML/CSS
st.markdown("""
<style>
.floating-button {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 9999;
    transition: all 0.3s ease;
}

.floating-button:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}
</style>

<a href="#" class="floating-button" title="Back to Top">
    ↑
</a>
""", unsafe_allow_html=True)

# ==================== ERROR HANDLING ====================
# Global error handler
def handle_error(e):
    """Handle errors gracefully"""
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("💡 Try refreshing the page or clearing cache from sidebar settings.")
    
    with st.expander("🔍 Error Details"):
        st.code(str(e))

# ==================== SESSION STATE INITIALIZATION ====================
# Initialize session state variables if not exists
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False

# ==================== WELCOME MESSAGE (Only on first load) ====================
if 'first_visit' not in st.session_state:
    st.session_state['first_visit'] = True
    
    # Show welcome toast (only works in Streamlit 1.30+)
    try:
        st.toast("👋 Selamat datang di Solana Price Prediction App!", icon="🎉")
    except:
        pass  # Ignore if toast not supported