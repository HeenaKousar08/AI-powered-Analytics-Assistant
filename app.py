import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from eda import get_basic_stats, generate_visual, auto_clean_data
from agent import DataAgent
from fpdf import FPDF
import io
import time

# ------------------ 1. PAGE CONFIGURATION ------------------
st.set_page_config(
    page_title="AI-Powered Analytics Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ 2. SESSION STATE MANAGEMENT ------------------
if "history" not in st.session_state: st.session_state.history = []
if "df" not in st.session_state: st.session_state.df = None
if "latest_insight" not in st.session_state: st.session_state.latest_insight = ""

# ------------------ 3. PROFESSIONAL ENTERPRISE STYLING ------------------
accent_color = "#4F46E5" 

st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background-color: #F8FAFC;
    }}

    /* Sidebar base */
    [data-testid="stSidebar"] {{
        background-color: #0F172A !important;
        padding-top: 20px;
    }}

    /* Sidebar text */
    [data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}

    /* Card Styling */
    .card {{
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        color: #1E293B;
    }}

    /* File uploader container */
    [data-testid="stFileUploader"] {{
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px;
        border: 2px dashed #CBD5E1;
    }}

    /* Fix invisible text issue in uploader */
    [data-testid="stFileUploader"] * {{
        color: #1E293B !important;
    }}

    /* Buttons */
    .stButton>button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------ 4. SIDEBAR & NAVIGATION ------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    uploaded_file = st.file_uploader("Ingest Dataset", type=["csv", "xlsx"])
    
    st.divider()
    nav = st.radio("Navigation", ["Executive Dashboard", "Visual Forge", "AI Terminal", "Report Forge"])

    st.divider()
    st.markdown("### 🖥️ System Status")
    st.markdown(f'<div style="color:#10B981; font-weight:bold;">● Local Inference Active</div>', unsafe_allow_html=True)
    st.caption("Engine: Mistral-7B via Ollama")
    st.progress(100)

# ------------------ 5. MAIN LOGIC ------------------
if uploaded_file:
    # Load data if not already in session state
    if st.session_state.df is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")

    df = st.session_state.df

    if df is not None:
        st.title("🤖 AI-Powered Analytics Assistant")
        st.caption(f"Asset: {uploaded_file.name} | Security: Local-Only Inference")

        # KPI DASHBOARD
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Records", f"{len(df):,}")
        with c2: st.metric("Features", len(df.columns))
        with c3: st.metric("Missing Cells", df.isnull().sum().sum())
        with c4: st.metric("Duplicate Rows", df.duplicated().sum())

        st.divider()

        # ROUTING
        if nav == "Executive Dashboard":
            col_table, col_actions = st.columns([2, 1])
            with col_table:
                st.subheader("🔍 Dataset Inspection")
                st.dataframe(df.head(15), use_container_width=True)
            
            with col_actions:
                st.subheader("🛠 Rapid Sanitation")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                if st.button("✨ Auto-Repair Dataset", use_container_width=True):
                    st.session_state.df = auto_clean_data(df)
                    st.toast("Data Repaired Successfully!", icon="✅")
                    time.sleep(1)
                    st.rerun()
                
                st.write("---")
                
                if st.button("🧠 AI Quick Insight", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        bot = DataAgent(df)
                        st.session_state.latest_insight = bot.ask("Give 3 high-impact trends based on this data.")
                
                if st.session_state.latest_insight:
                    st.info(st.session_state.latest_insight)
                st.markdown('</div>', unsafe_allow_html=True)

        elif nav == "Visual Forge":
            st.subheader("📊 Multi-Engine Visual Forge")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            v1, v2, v3 = st.columns(3)
            with v1:
                chart_selection = st.selectbox("Chart Template", ["Histogram", "Box Plot", "Violin Plot", "Count Plot", "Heatmap (Correlation)", "Area Chart", "Pie Chart", "Interactive Scatter"])
            with v2: 
                x_axis = st.selectbox("X-Axis", df.columns)
            with v3: 
                y_axis = st.selectbox("Y-Axis (Optional)", [None] + list(df.columns))
            
            hue_val = st.selectbox("Grouping / Color", [None] + list(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🚀 Render High-Fidelity Visual"):
                with st.status("Engine computing...", expanded=False) as status:
                    result = generate_visual(df, chart_selection, x_axis, y_axis, hue_val)
                    if result is not None:
                        if hasattr(result, "to_json"): 
                            st.plotly_chart(result, use_container_width=True)
                        else: 
                            st.pyplot(result)
                        status.update(label="Render Complete", state="complete")
                    else:
                        st.error("Engine Error: Incompatible data types for selected chart.")

        elif nav == "AI Terminal":
            st.subheader("💬 Cognitive Command Line")
            user_query = st.chat_input("Ask about specific patterns or segments...")
            
            if user_query:
                with st.spinner("Agent computing..."):
                    bot = DataAgent(df)
                    response = bot.ask(user_query)
                    st.session_state.history.append((user_query, response))
            
            for q, r in reversed(st.session_state.history[-5:]):
                with st.chat_message("user"): st.write(q)
                with st.chat_message("assistant"): st.write(r)

        elif nav == "Report Forge":
            st.subheader("📄 Automated PDF Audit")
            st.markdown('<div class="card">Generate a comprehensive audit report powered by local LLM logic.</div>', unsafe_allow_html=True)
            
            if st.button("⚙️ Compile AI Data Audit"):
                with st.spinner("Mistral is drafting..."):
                    bot = DataAgent(df)
                    report_content = bot.ask("Generate a professional executive report analyzing this dataset.")
                    
                    # Sanitize content for FPDF latin-1 encoding
                    clean_report = report_content.encode('latin-1', 'ignore').decode('latin-1')
                    
                    pdf = FPDF()
                    pdf.add_page()
                    # Header
                    pdf.set_fill_color(30, 41, 59)
                    pdf.rect(0, 0, 210, 40, 'F') 
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font("Arial", 'B', 20)
                    pdf.cell(0, 20, txt="AI ANALYTICS EXECUTIVE AUDIT", ln=True, align='C')
                    
                    # Body
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Arial", size=11)
                    pdf.ln(25)
                    pdf.multi_cell(0, 8, txt=clean_report)
                    
                    st.download_button(
                        label="💾 Download Audit PDF", 
                        data=pdf.output(dest='S'), 
                        file_name="AI_Executive_Report.pdf", 
                        mime="application/pdf"
                    )

else:
    st.markdown(f"""
    <div style="text-align:center; padding-top:100px;">
        <h1 style="font-size:60px; color:{accent_color};">🤖</h1>
        <h1>AI-Powered Analytics Assistant</h1>
        <p style="opacity:0.6; font-size:1.2em;">Upload a CSV or Excel asset to initialize the local analytics core.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="card"><strong>🔒 Private</strong><br>All data stays on your machine. 100% Local Inference.</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="card"><strong>🧠 Agentic</strong><br>LLM-driven insights and self-correcting data sanitation.</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="card"><strong>📈 Enterprise</strong><br>Ready for production with high-fidelity visuals and PDF exports.</div>', unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center; padding:50px; color:gray; font-size:0.85em;'>AI-Powered Analytics Assistant v3.0 | Developer: Heena Kousar | 2026</div>", unsafe_allow_html=True)