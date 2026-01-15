"""
Styling Module
Custom CSS dan styling functions untuk aplikasi Streamlit
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS untuk seluruh aplikasi"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container - Slide-like appearance */
    .main {
        padding: 1rem;
        background-color: #ffffff;
        min-height: 100vh;
    }
    
    /* Hide sidebar for slide presentation */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Full width layout */
    .block-container {
        max-width: none;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #334155;
        font-weight: 600;
        margin-top: 2rem;
        text-align: center;
    }
    
    h3 {
        color: #475569;
        font-weight: 600;
        text-align: center;
    }
    
    /* Metric Cards Custom */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(30, 64, 175, 0.3);
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-weight: 600;
        color: #1e293b;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1e40af;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1e40af, transparent);
    }
    
    /* Info, Success, Warning, Error boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card Style */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Gradient Background */
    .gradient-bg {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #1e40af;
        color: white;
    }
    
    /* Professional ML Engineer Enhancements */
    
    /* Smooth hover effects for all interactive elements */
    .stButton>button, .stTabs [data-baseweb="tab"], .custom-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(30, 64, 175, 0.3);
    }
    
    /* Enhanced card styling */
    .custom-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Professional color variables */
    :root {
        --primary-blue: #1e40af;
        --secondary-blue: #1d4ed8;
        --accent-blue: #3b82f6;
        --success-green: #059669;
        --warning-amber: #d97706;
        --error-red: #dc2626;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --bg-light: #f8fafc;
        --border-light: #e2e8f0;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-light);
    }
    
    /* Code block enhancements */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid var(--border-light);
        background: var(--bg-light);
    }
    
    /* Professional loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Professional progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-blue));
        border-radius: 10px;
    }
    
    /* Enhanced expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
        border: 1px solid var(--border-light);
        background: var(--bg-light);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .streamlit-expanderHeader:hover {
        background: #e2e8f0;
    }
    
    /* Professional selectbox and multiselect */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        border-radius: 8px;
        border: 1px solid var(--border-light);
    }
    
    /* Enhanced sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid var(--border-light);
    }
    
    /* Professional radio buttons */
    .stRadio > div {
        background: var(--bg-light);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid var(--border-light);
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(title, subtitle=None, icon=None):
    """Render header dengan gradient background"""
    icon_html = f"<span style='font-size: 3rem;'>{icon}</span>" if icon else ""
    subtitle_html = f"<p style='color: white; text-align: center; margin-top: 10px; font-size: 18px;'>{subtitle}</p>" if subtitle else ""
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%); 
                padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;'>
        {icon_html}
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(title, value, delta=None, icon=None):
    """Render metric card dengan styling custom"""
    icon_html = f"<div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>" if icon else ""
    delta_html = ""
    
    if delta:
        delta_color = "#10b981" if delta > 0 else "#ef4444"
        delta_symbol = "↑" if delta > 0 else "↓"
        delta_html = f"<div style='color: {delta_color}; font-size: 0.9rem; font-weight: 600;'>{delta_symbol} {abs(delta):.2f}%</div>"
    
    st.markdown(f"""
    <div class='custom-card' style='text-align: center;'>
        {icon_html}
        <div style='color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;'>{title}</div>
        <div style='color: #1e40af; font-size: 2rem; font-weight: 700;'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_info_box(title, content, type="info"):
    """Render info box dengan styling custom"""
    colors = {
        "info": {"bg": "#dbeafe", "border": "#3b82f6", "icon": "ℹ️"},
        "success": {"bg": "#dcfce7", "border": "#10b981", "icon": "✅"},
        "warning": {"bg": "#fef3c7", "border": "#f59e0b", "icon": "⚠️"},
        "error": {"bg": "#fee2e2", "border": "#ef4444", "icon": "❌"}
    }
    
    color = colors.get(type, colors["info"])
    
    st.markdown(f"""
    <div style='background-color: {color["bg"]}; 
                border-left: 4px solid {color["border"]}; 
                padding: 1rem; 
                border-radius: 8px; 
                margin: 1rem 0;'>
        <div style='font-weight: 600; margin-bottom: 0.5rem;'>
            {color["icon"]} {title}
        </div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_team_card(member):
    """Render team member card"""
    st.markdown(f"""
    <div class='custom-card' style='border-left: 4px solid #1e40af;'>
        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
            <div style='width: 60px; height: 60px; border-radius: 50%; 
                        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%); 
                        display: flex; align-items: center; justify-content: center; 
                        color: white; font-size: 1.5rem; font-weight: 700; margin-right: 1rem;'>
                {member['nama'][0]}
            </div>
            <div>
                <h3 style='margin: 0; color: #1e293b;'>{member['nama']}</h3>
                <p style='margin: 0; color: #64748b; font-size: 0.9rem;'>{member['nim']}</p>
            </div>
        </div>
        <div style='background-color: #f8fafc; padding: 0.75rem; border-radius: 6px;'>
            <strong style='color: #1e40af;'>Role:</strong> {member['role']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title, icon=None):
    """Render section header dengan styling"""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div style='margin: 2rem 0 1rem 0;'>
        <h2 style='color: #1e293b; border-bottom: 3px solid #1e40af; 
                   padding-bottom: 0.5rem; display: inline-block;'>
            {icon_html}{title}
        </h2>
    </div>
    """, unsafe_allow_html=True)


def render_badge(text, color="#1e40af"):
    """Render badge"""
    return f"""<span style='background-color: {color}; color: white; 
                padding: 0.25rem 0.75rem; border-radius: 20px; 
                font-size: 0.85rem; font-weight: 600; 
                display: inline-block; margin: 0.25rem;'>{text}</span>"""


def render_slide_navigator(current_slide, total_slides):
    """Render slide navigator dengan tombol previous dan next"""
    st.markdown("<br><br>", unsafe_allow_html=True)

    page_names = {
        1: "🏠 About & Data Sources",
        2: "📚 Metodologi Penelitian",
        3: "📈 Dashboard & Model Results",
        4: "👥 Developer"
    }

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_slide > 1:
            if st.button("⬅️ Previous", key=f"prev_slide_{current_slide}"):
                st.session_state['slide_navigation_index'] = current_slide - 2  # slide 2 -> index 0, slide 3 -> index 1, dst
                st.rerun()

    with col2:
        st.markdown(f"<div style='text-align: center; color: #1e40af; font-weight: 600;'>Slide {current_slide} of {total_slides}</div>", unsafe_allow_html=True)

    with col3:
        if current_slide < total_slides:
            if st.button("Next ➡️", key=f"next_slide_{current_slide}"):
                st.session_state['slide_navigation_index'] = current_slide  # index mulai dari 0, slide 1 -> index 0, next ke 1
                st.rerun()


def add_vertical_space(n=1):
    """Add vertical spacing"""
    for _ in range(n):
        st.markdown("<br>", unsafe_allow_html=True)