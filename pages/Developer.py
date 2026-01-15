"""
Developer Page - Solana Price Prediction App
Informasi tim developer dan project details
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assets.team_info import TEAM_MEMBERS, PROJECT_INFO, ACKNOWLEDGMENTS, TEAM_MOTTO, FUN_FACTS
from utils.styling import apply_custom_css, render_header, render_team_card, render_section_header, add_vertical_space, render_slide_navigator

# Apply custom CSS
apply_custom_css()

# ==================== HEADER ====================
render_header(
    title="👥 Tim Developer",
    subtitle="Kelompok Regresi Semiparametrik - Prediksi Harga Solana",
    icon="🎓"
)

# ==================== TEAM MOTTO ====================
st.markdown(f"""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 12px; margin-bottom: 30px; text-align: center;'>
    <div style='font-size: 48px; margin-bottom: 15px;'>💭</div>
    <p style='color: white; font-size: 18px; font-style: italic; margin: 0; line-height: 1.6;'>
        {TEAM_MOTTO}
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== TEAM MEMBERS ====================
render_section_header("Anggota Kelompok", "👨‍💻")

st.markdown("""
Tim kami terdiri dari 5 anggota dengan keahlian yang beragam, bekerja sama untuk 
mengembangkan sistem prediksi harga Solana menggunakan metode regresi semiparametrik 
dan machine learning.
""")

add_vertical_space(1)

# Display team members in rows of 2
for i in range(0, len(TEAM_MEMBERS), 2):
    cols = st.columns(2)
    
    for j, col in enumerate(cols):
        if i + j < len(TEAM_MEMBERS):
            member = TEAM_MEMBERS[i + j]
            with col:
                render_team_card(member)
                
                # Show responsibilities in expander
                with st.expander("📋 Responsibilities"):
                    for responsibility in member['responsibilities']:
                        st.markdown(f"- {responsibility}")

st.divider()

# ==================== PROJECT INFO ====================
render_section_header("Informasi Project", "📚")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    ### 📖 Mata Kuliah
    - **Nama:** {PROJECT_INFO['course_name']}
    - **Semester:** {PROJECT_INFO['semester']}
    - **Tahun Akademik:** {PROJECT_INFO['academic_year']}
    
    ### 👨‍🏫 Dosen Pengampu
    {PROJECT_INFO['lecturer']}
    """)

with col2:
    st.markdown(f"""
    ### 🏫 Institusi
    - **Universitas:** {PROJECT_INFO['university']}
    - **Fakultas:** {PROJECT_INFO['faculty']}
    - **Program Studi:** {PROJECT_INFO['department']}
    
    ### 📊 Judul Project
    {PROJECT_INFO['project_title']}
    """)

st.divider()

# ==================== TECHNOLOGY STACK ====================
render_section_header("Technology Stack", "🛠️")

st.markdown("### Tools & Libraries yang Digunakan")

# Create 3 columns for tools
col1, col2, col3 = st.columns(3)

tools_list = ACKNOWLEDGMENTS['tools_used']
tools_per_col = len(tools_list) // 3 + 1

with col1:
    for tool in tools_list[0:tools_per_col]:
        st.markdown(f"""
        <div style='background-color: #f8fafc; padding: 12px; border-radius: 8px; 
                    margin-bottom: 10px; border-left: 4px solid #667eea;'>
            {tool}
        </div>
        """, unsafe_allow_html=True)

with col2:
    for tool in tools_list[tools_per_col:tools_per_col*2]:
        st.markdown(f"""
        <div style='background-color: #f8fafc; padding: 12px; border-radius: 8px; 
                    margin-bottom: 10px; border-left: 4px solid #667eea;'>
            {tool}
        </div>
        """, unsafe_allow_html=True)

with col3:
    for tool in tools_list[tools_per_col*2:]:
        st.markdown(f"""
        <div style='background-color: #f8fafc; padding: 12px; border-radius: 8px; 
                    margin-bottom: 10px; border-left: 4px solid #667eea;'>
            {tool}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ==================== PROJECT TIMELINE ====================
render_section_header("Project Timeline", "📅")

st.markdown(f"""
**Periode Pengembangan:** {ACKNOWLEDGMENTS['development_period']}
""")

# Timeline visualization
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 3px; border-radius: 10px; margin: 20px 0;'>
    <div style='background: white; padding: 30px; border-radius: 8px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='text-align: center; flex: 1;'>
                <div style='font-size: 32px; margin-bottom: 10px;'>📋</div>
                <h4 style='margin: 5px 0; color: #667eea;'>Planning</h4>
                <p style='font-size: 14px; color: #64748b; margin: 5px 0;'>Week 1-2</p>
                <p style='font-size: 12px; color: #94a3b8;'>Research & Design</p>
            </div>
            <div style='flex: 0.2; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2);'></div>
            <div style='text-align: center; flex: 1;'>
                <div style='font-size: 32px; margin-bottom: 10px;'>💻</div>
                <h4 style='margin: 5px 0; color: #667eea;'>Development</h4>
                <p style='font-size: 14px; color: #64748b; margin: 5px 0;'>Week 3-6</p>
                <p style='font-size: 12px; color: #94a3b8;'>Coding & Testing</p>
            </div>
            <div style='flex: 0.2; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2);'></div>
            <div style='text-align: center; flex: 1;'>
                <div style='font-size: 32px; margin-bottom: 10px;'>🧪</div>
                <h4 style='margin: 5px 0; color: #667eea;'>Testing</h4>
                <p style='font-size: 14px; color: #64748b; margin: 5px 0;'>Week 7-8</p>
                <p style='font-size: 12px; color: #94a3b8;'>Validation & Tuning</p>
            </div>
            <div style='flex: 0.2; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2);'></div>
            <div style='text-align: center; flex: 1;'>
                <div style='font-size: 32px; margin-bottom: 10px;'>📝</div>
                <h4 style='margin: 5px 0; color: #667eea;'>Documentation</h4>
                <p style='font-size: 14px; color: #64748b; margin: 5px 0;'>Week 9-10</p>
                <p style='font-size: 12px; color: #94a3b8;'>Report & Presentation</p>
            </div>
            <div style='flex: 0.2; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2);'></div>
            <div style='text-align: center; flex: 1;'>
                <div style='font-size: 32px; margin-bottom: 10px;'>🎉</div>
                <h4 style='margin: 5px 0; color: #10b981;'>Completion</h4>
                <p style='font-size: 14px; color: #64748b; margin: 5px 0;'>Week 11</p>
                <p style='font-size: 12px; color: #94a3b8;'>Final Submission</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ==================== FUN FACTS ====================
render_section_header("Fun Facts", "🎉")

st.markdown("### Project Statistics")

# Display fun facts in columns
cols = st.columns(3)
for i, fact in enumerate(FUN_FACTS):
    with cols[i % 3]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; 
                    color: white; margin-bottom: 15px; height: 120px;
                    display: flex; align-items: center; justify-content: center;'>
            <p style='margin: 0; font-size: 16px; font-weight: 600;'>{fact}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ==================== ACKNOWLEDGMENTS ====================
render_section_header("Acknowledgments", "🙏")

st.markdown("### Terima Kasih Kepada:")

for thanks in ACKNOWLEDGMENTS['thanks_to']:
    st.markdown(f"""
    <div style='background-color: #f8fafc; padding: 15px; border-radius: 8px; 
                margin-bottom: 10px; border-left: 4px solid #10b981;'>
        ✨ {thanks}
    </div>
    """, unsafe_allow_html=True)

add_vertical_space(1)

st.markdown(f"""
### 📊 Data Source
{ACKNOWLEDGMENTS['data_source']}
""")

st.divider()

# ==================== FEEDBACK SECTION ====================
st.markdown("### 💬 Feedback & Suggestions")

st.markdown("""
Kami sangat menghargai feedback dan saran untuk pengembangan project ini. 
Jika Anda memiliki pertanyaan atau saran, silakan hubungi kami melalui email atau GitHub.
""")

with st.expander("📝 Leave a Feedback"):
    feedback_name = st.text_input("Nama Anda")
    feedback_email = st.text_input("Email Anda")
    feedback_message = st.text_area("Pesan Feedback", height=150)
    
    if st.button("📤 Submit Feedback"):
        if feedback_name and feedback_email and feedback_message:
            st.success("✅ Terima kasih atas feedback Anda! Kami akan segera meresponnya.")
        else:
            st.error("❌ Mohon lengkapi semua field!")

# ==================== CLOSING ====================
add_vertical_space(2)

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 40px; border-radius: 12px; text-align: center; color: white;'>
    <h2 style='margin: 0; color: white;'>🎓 Terima Kasih!</h2>
    <p style='margin-top: 15px; font-size: 16px;'>
        Project ini adalah hasil kerja keras dan kolaborasi tim.<br>
        Semoga bermanfaat untuk pengembangan ilmu pengetahuan!
    </p>
    <div style='margin-top: 20px; font-size: 24px;'>
        ⭐ ⭐ ⭐ ⭐ ⭐
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== FOOTER ====================
add_vertical_space(2)

st.markdown("""
---
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>👥 <strong>Developer Team</strong> | Regresi Semiparametrik 2024/2025</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        Made with ❤️ using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# Slide Navigator
render_slide_navigator(4, 4)