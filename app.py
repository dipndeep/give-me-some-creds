"""
CreditRiskAI: Interactive Underwriting Simulator & Explainable AI Dashboard
=============================================================================
Day 4 Streamlit Production Application
Author: Ganendra Pradipa | https://linkedin.com/in/ganendrapratama
"""

import os
import glob
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager
import plotly.graph_objects as go
import shap

from src.scoring_engine import ScoringEngine, PRESET_PERSONAS

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditRiskAI | Underwriting Risk Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Design tokens: SATU keluarga font (Montserrat) untuk seluruh antarmuka.
# Hierarki dibangun lewat ukuran & ketebalan, bukan lewat pergantian font.
# -----------------------------------------------------------------------------
FONT_FAMILY = "Montserrat, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif"

PLOTLY_BASE = dict(
    paper_bgcolor="#111a33",
    plot_bgcolor="#111a33",
    font=dict(family=FONT_FAMILY, size=12, color="#e2e8f0"),
)
PLOTLY_AXIS = dict(
    gridcolor="#1e2d4d",
    zerolinecolor="#334155",
    linecolor="#334155",
    tickfont=dict(family=FONT_FAMILY, size=11, color="#94a3b8"),
    title_font=dict(family=FONT_FAMILY, size=12, color="#cbd5e1"),
)


def plotly_title(text: str) -> dict:
    """Judul grafik Plotly dengan gaya seragam."""
    return dict(text=text, x=0.02, xanchor="left", font=dict(family=FONT_FAMILY, size=15, color="#ffffff"))


# Matplotlib (SHAP waterfall) dirender di server sebagai gambar PNG, sehingga CSS tidak
# berpengaruh. Font Montserrat harus terdaftar secara lokal agar sama dengan UI.
# Letakkan berkas .ttf (mis. Montserrat-Regular.ttf, Montserrat-SemiBold.ttf) di assets/fonts/.
for _font_path in glob.glob(os.path.join("assets", "fonts", "*.ttf")):
    try:
        font_manager.fontManager.addfont(_font_path)
    except Exception:
        pass

MPL_DARK_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial"],
    "figure.facecolor": "#111a33",
    "axes.facecolor": "#111a33",
    "savefig.facecolor": "#111a33",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#e2e8f0",
}

# High-Contrast Corporate Fintech Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

    :root {
        --font-ui: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        --font-mono: ui-monospace, 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;

        --bg: #0b1329;
        --surface: #111a33;
        --surface-2: #131d36;
        --surface-3: #172447;
        --border: #1e2d4d;
        --border-strong: #334155;

        --text: #f1f5f9;
        --text-muted: #cbd5e1;
        --text-dim: #94a3b8;

        --accent: #38bdf8;
        --accent-strong: #0ea5e9;
        --ok: #34d399;
        --warn: #fbbf24;
        --bad: #f87171;
    }

    /* ===== 1. FONT: satu keluarga untuk semua elemen ===== */
    body, body * {
        font-family: var(--font-ui) !important;
    }

    /* Ikon Material Symbols jangan ikut diganti (kalau tidak, muncul teks seperti "keyboard_arrow_down") */
    body [data-testid="stIconMaterial"],
    body [class*="material-symbols"],
    body [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        font-feature-settings: 'liga' !important;
    }

    /* Kode tetap monospace */
    body code, body pre, body pre * {
        font-family: var(--font-mono) !important;
    }

    /* Angka rata (tabular) supaya KPI tidak "bergoyang" saat nilai berubah */
    .metric-val, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], td {
        font-variant-numeric: tabular-nums;
    }

    /* ===== 2. WARNA DASAR & SKALA TIPOGRAFI ===== */
    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text);
    }

    /* Warna teks: tanpa !important agar inline style (merah/hijau/biru) tetap bekerja */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--text-muted);
        line-height: 1.6;
    }
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {
        color: #ffffff;
        font-weight: 700;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    [data-testid="stMarkdownContainer"] h3 { font-size: 1.3rem; }
    [data-testid="stMarkdownContainer"] h4 { font-size: 1.1rem; }
    [data-testid="stMarkdownContainer"] h5 { font-size: 0.98rem; font-weight: 600; color: var(--text-muted); }

    /* Label widget */
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Caption */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, small {
        color: var(--text-dim);
        font-size: 0.82rem !important;
        line-height: 1.5;
    }

    /* ===== 3. SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #0d1733 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 1.05rem;
        margin-bottom: 4px;
    }

    /* ===== 4. EXPANDER ===== */
    [data-testid="stExpander"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stExpander"] details summary {
        background-color: #162244 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stExpander"] details summary p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    [data-testid="stExpander"] details > div {
        background-color: var(--surface) !important;
        padding: 14px !important;
    }

    /* ===== 5. INPUT ===== */
    input[type="number"], input[type="text"], [data-testid="stNumberInput"] input {
        background-color: var(--surface-3) !important;
        color: #ffffff !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    input[type="number"]:focus, input[type="text"]:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--text-dim) !important;
        font-size: 0.75rem !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--surface-2) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: 8px !important;
    }

    /* ===== 6. TAB ===== */
    [data-baseweb="tab-list"] {
        background-color: var(--bg) !important;
        border-bottom: 2px solid #1e293b !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }
    [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 10px 20px !important;
    }
    [data-baseweb="tab"] p {
        color: var(--text-dim) !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #131d36 !important;
        border-bottom: 2px solid var(--accent-strong) !important;
    }
    [data-baseweb="tab"][aria-selected="true"] p {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    /* ===== 7. TOMBOL (selector lama & baru Streamlit) ===== */
    [data-testid="baseButton-secondary"], [data-testid="stBaseButton-secondary"],
    [data-testid="stDownloadButton"] button {
        background-color: var(--surface-3) !important;
        color: #f8fafc !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }
    [data-testid="baseButton-secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stDownloadButton"] button:hover {
        background-color: #1e3a8a !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
    }
    [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
    }

    /* ===== 8. st.metric (What-If) ===== */
    [data-testid="stMetricLabel"] p {
        color: var(--text-dim) !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ===== 9. KOMPONEN KUSTOM ===== */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .header-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.015em;
        line-height: 1.25;
    }
    .header-box .header-sub {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 6px 0 0 0;
        max-width: 62ch;
    }
    .tag-badge {
        background: var(--surface);
        color: var(--accent);
        border: 1px solid #1e3a8a;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px 0 0 6px;
    }

    /* Kartu KPI: tinggi seragam, isi di tengah */
    .metric-card {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        min-height: 138px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-title {
        font-size: 0.78rem;
        color: var(--text-dim);
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .metric-desc {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 6px;
        line-height: 1.4;
    }

    /* Badge keputusan */
    .badge-approved, .badge-standard, .badge-review, .badge-declined {
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        line-height: 1.35;
    }
    .badge-approved { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }
    .badge-standard { background-color: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid #0ea5e9; }
    .badge-review   { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-declined { background-color: rgba(239, 68, 68, 0.15);  color: #f87171; border: 1px solid #ef4444; }

    .card-dark {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-dark h4 {
        font-size: 1.05rem;
        margin: 0 0 12px 0;
    }
    .adverse-item, .protective-item {
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        color: var(--text);
        font-size: 0.86rem;
        line-height: 1.5;
    }
    .adverse-item    { background: rgba(239, 68, 68, 0.08);  border-left: 3px solid #ef4444; }
    .protective-item { background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; }

    /* Tabel di dalam kartu (tab Tata Kelola) */
    .card-dark table { width: 100%; border-collapse: collapse; }
    .card-dark td {
        padding: 8px 0;
        border-bottom: 1px solid #1e293b;
        color: var(--text-muted);
        font-size: 0.85rem;
        vertical-align: top;
    }
    .card-dark tr:last-child td { border-bottom: none; }

    /* Alert */
    [data-testid="stAlert"] {
        background-color: var(--surface-2) !important;
        border: 1px solid #1e3a8a !important;
        border-radius: 8px !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e293b;
    }
    .footer-text p { color: #64748b; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Engine Initialization & Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model XGBoost & modul inferensi SHAP...")
def get_scoring_engine():
    """Initializes and caches the scoring engine instance across sessions."""
    return ScoringEngine()

try:
    engine = get_scoring_engine()
except Exception as e:
    st.error(f"Gagal memuat artefak model: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 3. Session State Management (Persona Autofill)
# -----------------------------------------------------------------------------
default_persona = PRESET_PERSONAS['prime']['data']

if 'in_age' not in st.session_state:
    st.session_state.in_age = int(default_persona['age'])
    st.session_state.in_income = float(default_persona['MonthlyIncome'])
    st.session_state.in_dti = float(default_persona['DebtRatio'] * 100.0)
    st.session_state.in_dep = int(default_persona['NumberOfDependents'])
    st.session_state.in_util = float(default_persona['RevolvingUtilizationOfUnsecuredLines'] * 100.0)
    st.session_state.in_lines = int(default_persona['NumberOfOpenCreditLinesAndLoans'])
    st.session_state.in_re = int(default_persona['NumberRealEstateLoansOrLines'])
    st.session_state.in_pd30 = int(default_persona['NumberOfTime30-59DaysPastDueNotWorse'])
    st.session_state.in_pd60 = int(default_persona['NumberOfTime60-89DaysPastDueNotWorse'])
    st.session_state.in_pd90 = int(default_persona['NumberOfTimes90DaysLate'])
    st.session_state.active_persona_name = PRESET_PERSONAS['prime']['name']


def apply_persona(persona_key: str):
    """Updates applicant form state to match the chosen persona."""
    persona = PRESET_PERSONAS[persona_key]
    data = persona['data']
    st.session_state.in_age = int(data['age'])
    st.session_state.in_income = float(data['MonthlyIncome'])
    st.session_state.in_dti = float(data['DebtRatio'] * 100.0)
    st.session_state.in_dep = int(data['NumberOfDependents'])
    st.session_state.in_util = float(data['RevolvingUtilizationOfUnsecuredLines'] * 100.0)
    st.session_state.in_lines = int(data['NumberOfOpenCreditLinesAndLoans'])
    st.session_state.in_re = int(data['NumberRealEstateLoansOrLines'])
    st.session_state.in_pd30 = int(data['NumberOfTime30-59DaysPastDueNotWorse'])
    st.session_state.in_pd60 = int(data['NumberOfTime60-89DaysPastDueNotWorse'])
    st.session_state.in_pd90 = int(data['NumberOfTimes90DaysLate'])
    st.session_state.active_persona_name = persona['name']


# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Applicant Form
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Preset Profil Risiko (1-Click)")
    st.caption("Pilih profil debitur representatif untuk pengujian cepat:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("Prime Executive", use_container_width=True):
            apply_persona('prime')
            st.rerun()
    with col_p2:
        if st.button("Stretched Youth", use_container_width=True):
            apply_persona('stretched')
            st.rerun()

    if st.button("Distressed Multi-Default", use_container_width=True):
        apply_persona('distressed')
        st.rerun()

    st.markdown(f"<span style='font-size:0.82rem; color:#94a3b8;'>Profil Aktif:</span> <strong style='color:#38bdf8;'>{st.session_state.active_persona_name}</strong>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Form Data Calon Debitur")

    # Section 1: Demografi & Finansial
    with st.expander("Profil Demografi & Finansial", expanded=True):
        age_input = st.slider("Usia Peminjam (Tahun)", 18, 95, key="in_age")
        income_input = st.number_input("Pendapatan Kotor Bulanan ($)", min_value=0.0, max_value=100000.0, step=500.0, key="in_income")
        debt_ratio_input = st.slider("Debt Ratio / DTI (%)", 0.0, 300.0, step=5.0, key="in_dti") / 100.0
        dependents_input = st.number_input("Jumlah Tanggungan Keluarga", min_value=0, max_value=12, step=1, key="in_dep")

    # Section 2: Portofolio & Utilisasi Kredit
    with st.expander("Portofolio & Utilisasi Kredit", expanded=True):
        utilization_input = st.slider("Utilisasi Limit Kartu Kredit (%)", 0.0, 200.0, step=5.0, key="in_util") / 100.0
        if utilization_input > 1.0:
            st.caption("<span style='color:#f87171;'>Peringatan: Utilisasi > 100% melampaui limit resmi (Over-limit).</span>", unsafe_allow_html=True)
        open_lines_input = st.number_input("Fasilitas Pinjaman Aktif", min_value=0, max_value=40, step=1, key="in_lines")
        real_estate_input = st.number_input("Jumlah KPR / Agunan Properti", min_value=0, max_value=15, step=1, key="in_re")

    # Section 3: Riwayat Keterlambatan
    with st.expander("Riwayat Keterlambatan (2 Tahun)", expanded=True):
        past_due_30_input = st.number_input("Tunggakan 30–59 Hari", min_value=0, max_value=10, step=1, key="in_pd30")
        past_due_60_input = st.number_input("Tunggakan 60–89 Hari", min_value=0, max_value=10, step=1, key="in_pd60")
        times_90_late_input = st.number_input("Tunggakan Fatal 90+ Hari", min_value=0, max_value=10, step=1, key="in_pd90")

    applicant_data = {
        'age': age_input,
        'MonthlyIncome': income_input,
        'DebtRatio': debt_ratio_input,
        'RevolvingUtilizationOfUnsecuredLines': utilization_input,
        'NumberOfOpenCreditLinesAndLoans': open_lines_input,
        'NumberRealEstateLoansOrLines': real_estate_input,
        'NumberOfTime30-59DaysPastDueNotWorse': past_due_30_input,
        'NumberOfTime60-89DaysPastDueNotWorse': past_due_60_input,
        'NumberOfTimes90DaysLate': times_90_late_input,
        'NumberOfDependents': dependents_input
    }


# -----------------------------------------------------------------------------
# 5. Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="header-title">CreditRiskAI: Interactive Underwriting Simulator</h1>
            <p class="header-sub">Sistem Penilaian Risiko Kredit Retail Berbasis XGBoost, Kalibrasi Skor FICO (300–850), & Explainable AI (SHAP)</p>
        </div>
        <div style="margin-top: 6px;">
            <span class="tag-badge">Model: XGBoost Champion</span>
            <span class="tag-badge" style="color: #34d399; border-color: #059669;">ROC-AUC: 0.8686</span>
            <span class="tag-badge" style="color: #fbbf24; border-color: #d97706;">KS-Statistic: 57.84% (Basel II)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Main Tabs: Single Applicant | Batch Scoring | Model Governance
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Underwriting Individual & XAI",
    "Evaluasi Portofolio Batch",
    "Tata Kelola & Audit Regulasi"
])


# =============================================================================
# TAB 1: SINGLE APPLICANT UNDERWRITING
# =============================================================================
with tab1:
    # Inference Calculation
    features_df = engine.prepare_features(applicant_data)
    prediction = engine.predict_risk(features_df)
    shap_results = engine.explain_applicant_shap(features_df)

    fico_score = prediction['fico_score']
    prob_pct = prediction['prob_default_pct']
    policy = prediction['policy']
    rating = prediction['fico_rating']

    # Top KPI Metric Cards Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Proyeksi Skor FICO</div>
            <div class="metric-val" style="color: {rating['color']};">{fico_score}</div>
            <div class="metric-desc"><strong style="color: {rating['color']};">{rating['badge']}</strong> ({rating['grade']})</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        diff_baseline = prob_pct - 6.68
        delta_color = "#f87171" if diff_baseline > 0 else "#34d399"
        delta_sign = "+" if diff_baseline > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Probabilitas Gagal Bayar (PD)</div>
            <div class="metric-val" style="color: {policy['color']};">{prob_pct:.1f}%</div>
            <div class="metric-desc">Baseline Portofolio: 6.68% (<span style="color: {delta_color}; font-weight: bold;">{delta_sign}{diff_baseline:.1f}%</span>)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        badge_class = {
            'approved': 'badge-approved',
            'approved_standard': 'badge-standard',
            'review': 'badge-review',
            'declined': 'badge-declined'
        }.get(policy['status'], 'badge-standard')

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Keputusan Underwriting</div>
            <div style="margin: 6px 0;">
                <span class="{badge_class}">{policy['decision']}</span>
            </div>
            <div class="metric-desc">{policy['tier']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rekomendasi Penawaran</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">{policy['interest_rate']}</div>
            <div class="metric-desc">Limit Plafon: {policy['credit_limit_multiplier']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Middle Section: Visual Gauge & Action Recommendation
    col_g1, col_g2 = st.columns([5, 6])

    with col_g1:
        st.markdown("#### Indikator Skor Risiko (FICO Scale)")
        
        # Plotly Speedometer
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fico_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'font': {'size': 44, 'color': '#ffffff', 'family': FONT_FAMILY}},
            gauge={
                'axis': {
                    'range': [300, 850],
                    'tickwidth': 1,
                    'tickcolor': "#94a3b8",
                    'tickvals': [300, 550, 650, 750, 850],
                    'tickfont': {'family': FONT_FAMILY, 'size': 11, 'color': '#94a3b8'}
                },
                'bar': {'color': rating['color'], 'thickness': 0.28},
                'bgcolor': "#1e293b",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [300, 550], 'color': "rgba(239, 68, 68, 0.25)"},
                    {'range': [550, 650], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [650, 750], 'color': "rgba(14, 165, 233, 0.25)"},
                    {'range': [750, 850], 'color': "rgba(16, 185, 129, 0.25)"}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 3},
                    'thickness': 0.75,
                    'value': 550
                }
            }
        ))
        fig_gauge.update_layout(
            **PLOTLY_BASE,
            height=280,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.markdown("#### Rekomendasi Kebijakan Underwriting")
        collateral_badge = "Diperlukan Agunan Fisik / Co-Signer" if policy['collateral_required'] else "Tanpa Agunan Tambahan (Clean Loan)"
        st.markdown(f"""
        <div class="card-dark" style="border-left: 4px solid {policy['color']}; height: 280px; overflow-y: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; font-size: 1.1rem; color: {policy['color']};">{policy['decision']}</span>
                <span style="font-size: 0.8rem; background: #1e293b; padding: 4px 10px; border-radius: 4px; color: #94a3b8;">Threshold Optimal: P = 0.56</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.55; margin-bottom: 12px;">
                {policy['action_recommendation']}
            </p>
            <div style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8;">
                <p style="margin: 4px 0; color: #cbd5e1;">• <strong>Ketentuan Agunan</strong>: {collateral_badge}</p>
                <p style="margin: 4px 0; color: #cbd5e1;">• <strong>Suku Bunga Penawaran</strong>: {policy['interest_rate']}</p>
                <p style="margin: 4px 0; color: #cbd5e1;">• <strong>Plafond Kredit</strong>: {policy['credit_limit_multiplier']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Explainable AI: SHAP Waterfall & Regulatory Compliance
    st.markdown("### Explainable AI (SHAP): Transparansi & Kepatuhan Regulasi")
    st.caption("Dekomposisi matematis faktor penentu keputusan secara transparan sesuai regulasi Fair Lending / FCRA / POJK.")

    col_x1, col_x2 = st.columns([6, 5])

    with col_x1:
        st.markdown("##### Dekomposisi Log-Odds (SHAP Waterfall)")
        
        # Waterfall SHAP bertema gelap. rc_context membuat teks, tick, dan latar ikut tema
        # (default SHAP menulis label hitam sehingga tak terbaca di latar navy) dan tidak
        # mengubah rcParams global.
        with plt.rc_context(MPL_DARK_RC):
            plt.subplots(figsize=(8, 4.6))
            shap.plots.waterfall(shap_results['shap_explanation'][0], max_display=7, show=False)
            fig_waterfall = plt.gcf()
            fig_waterfall.patch.set_facecolor('#111a33')
            plt.gca().set_facecolor('#111a33')
            plt.title("Kontribusi Fitur Terhadap Prediksi Log-Odds", fontsize=11, fontweight='bold', pad=10, loc='left')
            plt.tight_layout()
            st.pyplot(fig_waterfall, use_container_width=True)
            plt.close(fig_waterfall)

    with col_x2:
        st.markdown("##### Faktor Risiko & Kepatuhan Regulasi")

        # Adverse Action Reasons (Top 3 Risk Drivers)
        st.markdown("<p style='font-weight: 700; color: #f87171; margin-bottom: 6px;'>Faktor Penekan Risiko (Adverse Action Factors):</p>", unsafe_allow_html=True)
        if shap_results['top_adverse_drivers']:
            for adv in shap_results['top_adverse_drivers']:
                st.markdown(f"""
                <div class="adverse-item">
                    <strong style="color: #fca5a5;">{adv['label']}</strong>: {adv['narrative']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Tidak ditemukan faktor risiko signifikan yang menaikkan probabilitas gagal bayar.")

        # Protective Factors (Top 3 Positive Drivers)
        st.markdown("<p style='font-weight: 700; color: #34d399; margin-top: 14px; margin-bottom: 6px;'>Faktor Penguat Kredibilitas (Protective Factors):</p>", unsafe_allow_html=True)
        if shap_results['top_protective_drivers']:
            for prot in shap_results['top_protective_drivers']:
                st.markdown(f"""
                <div class="protective-item">
                    <strong style="color: #6ee7b7;">{prot['label']}</strong>: {prot['narrative']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Faktor pelindung minimal.")

    st.divider()

    # Interactive "What-If" Sensitivity Simulator
    st.markdown("### Simulasi Sensitivitas Kebijakan (What-If Analysis)")
    st.caption("Uji proyeksi dampak perbaikan profil terhadap skor kredit dan keputusan underwriting:")

    col_w1, col_w2, col_w3 = st.columns([4, 4, 3])

    with col_w1:
        sim_util = st.slider(
            "Simulasi: Penurunan Utilisasi Kartu Kredit (%)",
            0.0, 200.0,  # sama dengan batas slider di sidebar, agar nilai awal tak keluar rentang
            float(applicant_data['RevolvingUtilizationOfUnsecuredLines'] * 100.0),
            step=5.0,
            help="Simulasi pengurangan saldo tagihan kartu kredit debitur"
        ) / 100.0

    with col_w2:
        sim_re = st.number_input(
            "Simulasi: Penambahan Agunan KPR / Properti",
            min_value=0, max_value=15,  # sama dengan batas di sidebar
            value=int(applicant_data['NumberRealEstateLoansOrLines']),
            step=1,
            help="Simulasi penambahan agunan aset properti terverifikasi"
        )

    # Run What-If Prediction
    sim_data = applicant_data.copy()
    sim_data['RevolvingUtilizationOfUnsecuredLines'] = sim_util
    sim_data['NumberRealEstateLoansOrLines'] = sim_re
    sim_features_df = engine.prepare_features(sim_data)
    sim_prediction = engine.predict_risk(sim_features_df)

    with col_w3:
        st.markdown("##### Proyeksi Hasil Simulasi:")
        delta_score = sim_prediction['fico_score'] - fico_score
        delta_sign = "+" if delta_score >= 0 else ""

        st.metric(
            label="Skor FICO Simulasi",
            value=f"{sim_prediction['fico_score']}",
            delta=f"{delta_sign}{delta_score} Poin ({sim_prediction['policy']['decision']})",
            delta_color="normal"  # skor naik = hijau, turun = merah
        )
        st.caption(f"PD Simulasi: <strong style='color:#ffffff;'>{sim_prediction['prob_default_pct']:.1f}%</strong> | Status: <strong style='color:#38bdf8;'>{sim_prediction['policy']['tier']}</strong>", unsafe_allow_html=True)


# =============================================================================
# TAB 2: BATCH PORTFOLIO SCORING & STRESS TESTING
# =============================================================================
with tab2:
    st.markdown("### Evaluasi Portofolio Batch & Stress Testing")
    st.caption("Unggah berkas CSV calon debitur untuk melakukan penilaian portofolio massal dan audit risiko agregat:")

    col_b1, col_b2 = st.columns([4, 2])

    with col_b1:
        uploaded_file = st.file_uploader("Unggah Berkas CSV Calon Debitur", type=['csv'])

    with col_b2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("Muat 50 Data Sampel Benchmark", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Berkas berhasil dimuat: {len(batch_df)} aplikasi debitur.")
        except Exception as e:
            st.error(f"Gagal membaca berkas CSV: {e}")
    elif os.path.exists('data/sample_applicants.csv'):
        batch_df = pd.read_csv('data/sample_applicants.csv')
        st.info("Menggunakan 50 profil calon peminjam dari holdout validation set.")
    elif load_sample:
        st.warning("Berkas data/sample_applicants.csv tidak ditemukan. Unggah CSV sendiri untuk melanjutkan.")

    if batch_df is not None:
        with st.spinner("Memproses evaluasi risiko batch portofolio..."):
            scored_df = engine.score_batch(batch_df)

        # Portfolio Summary Metrics
        total_applicants = len(scored_df)
        approved_count = (scored_df['Underwriting_Decision'].isin(['INSTANT APPROVAL', 'STANDARD APPROVAL'])).sum()
        review_count = (scored_df['Underwriting_Decision'] == 'CONDITIONAL / MANUAL REVIEW').sum()
        declined_count = (scored_df['Underwriting_Decision'] == 'AUTOMATED DECLINE').sum()
        
        approval_rate = (approved_count / total_applicants) * 100.0
        avg_fico = int(scored_df['FICO_Score'].mean())
        avg_pd = scored_df['Prob_Default_Pct'].mean()

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric("Total Debitur Diskor", f"{total_applicants:,} Aplikasi")
        with col_p2:
            st.metric("Tingkat Kelulusan (Approval Rate)", f"{approval_rate:.1f}%")
        with col_p3:
            st.metric("Rata-rata Skor FICO Portofolio", f"{avg_fico}")
        with col_p4:
            st.metric("Rata-rata Probabilitas Gagal Bayar", f"{avg_pd:.2f}%")

        # Distribution Visuals
        col_c1, col_c2 = st.columns([6, 5])
        with col_c1:
            # FICO Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=scored_df['FICO_Score'],
                nbinsx=15,
                marker=dict(color='#0ea5e9', line=dict(color='#111a33', width=1)),
                opacity=0.9
            ))
            fig_hist.add_vline(
                x=550, line_dash="dash", line_color="#ef4444",
                annotation_text="Cutoff Boundary (550)",
                annotation_font=dict(family=FONT_FAMILY, size=11, color="#f87171")
            )
            fig_hist.update_layout(
                **PLOTLY_BASE,
                title=plotly_title("Distribusi Skor FICO Portofolio"),
                xaxis=dict(title=dict(text="Skor FICO"), **PLOTLY_AXIS),
                yaxis=dict(title=dict(text="Frekuensi Debitur"), **PLOTLY_AXIS),
                bargap=0.05,
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_c2:
            # Decision Pie Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Approved', 'Manual Review', 'Declined'],
                values=[approved_count, review_count, declined_count],
                hole=0.45,
                marker=dict(colors=['#10b981', '#f59e0b', '#ef4444'], line=dict(color='#111a33', width=2)),
                textfont=dict(family=FONT_FAMILY, size=12, color="#ffffff")
            )])
            fig_pie.update_layout(
                **PLOTLY_BASE,
                title=plotly_title("Proporsi Keputusan Portofolio"),
                legend=dict(font=dict(family=FONT_FAMILY, size=11, color="#cbd5e1")),
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Scored Table Display
        st.markdown("##### Tabel Hasil Keputusan Underwriting Batch")
        display_cols = [
            'FICO_Score', 'Prob_Default_Pct', 'Credit_Grade',
            'Underwriting_Decision', 'age', 'MonthlyIncome',
            'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'
        ]
        available_cols = [c for c in display_cols if c in scored_df.columns]
        st.dataframe(scored_df[available_cols], use_container_width=True, height=280)

        # Download CSV
        csv_data = scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Keputusan Scoring (CSV)",
            data=csv_data,
            file_name="processed_underwriting_results.csv",
            mime="text/csv"
        )


# =============================================================================
# TAB 3: MODEL GOVERNANCE & POLICY AUDIT
# =============================================================================
with tab3:
    st.markdown("### Tata Kelola Model, Kepatuhan Regulasi, & Audit Benchmark")
    st.caption("Dokumentasi teknis resmi untuk komite risiko perbankan, auditor model, dan regulator institusional:")

    col_gov1, col_gov2 = st.columns(2)

    with col_gov1:
        st.markdown("""
        <div class="card-dark">
            <h4 style="color: #38bdf8; margin-top: 0;">Spesifikasi Model: XGBoost Champion</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                <tr>
                    <td style="padding: 8px 0;"><strong>Algoritma</strong></td>
                    <td style="text-align: right; color: #ffffff;">Extreme Gradient Boosting (XGBoost)</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0;"><strong>Jumlah Estimator (Trees)</strong></td>
                    <td style="text-align: right; color: #ffffff;">120 Trees (max_depth = 5)</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0;"><strong>Bobot Penyeimbang Kelas</strong></td>
                    <td style="text-align: right; color: #ffffff;">scale_pos_weight = 13.96 (~14:1)</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0;"><strong>Metrik Diskriminasi</strong></td>
                    <td style="text-align: right; color: #34d399; font-weight: bold;">ROC-AUC: 0.8686 | PR-AUC: 0.4041</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0;"><strong>Kolmogorov-Smirnov (KS)</strong></td>
                    <td style="text-align: right; color: #34d399; font-weight: bold;">57.84% (Standar Basel II > 40%)</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0;"><strong>Gini Coefficient</strong></td>
                    <td style="text-align: right; color: #ffffff;">0.7372 (Daya Pisah Sangat Baik)</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0;"><strong>Ambang Batas Biaya Optimal</strong></td>
                    <td style="text-align: right; color: #fbbf24; font-weight: bold;">P = 0.56 (Asymmetric Loss Minimized)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_gov2:
        st.markdown("""
        <div class="card-dark">
            <h4 style="color: #fbbf24; margin-top: 0;">Matriks Kerugian Finansial Asimetris</h4>
            <p style="font-size: 0.86rem; color: #cbd5e1; line-height: 1.55;">
                Dalam underwriting retail, penolakan nasabah baik (False Positive) dan pelolosan nasabah macet (False Negative) memiliki dampak biaya yang sangat timpang:
            </p>
            <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; padding: 10px 14px; margin-bottom: 8px; font-size: 0.85rem; color: #f1f5f9;">
                <strong style="color: #f87171;">False Negative ($10,000 / Kasus):</strong> Bank menanggung hapus buku pinjaman pokok 100% ditambah biaya legal penagihan.
            </div>
            <div style="background: rgba(14, 165, 233, 0.1); border-left: 3px solid #0ea5e9; padding: 10px 14px; margin-bottom: 8px; font-size: 0.85rem; color: #f1f5f9;">
                <strong style="color: #38bdf8;">False Positive ($1,000 / Kasus):</strong> Kehilangan margin bunga pinjaman lancar (Opportunity Loss).
            </div>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 10px;">
                Berdasarkan simulasi 86 titik ambang batas pada 30.000 data validasi, batas potong <strong style="color: #ffffff;">P = 0.56</strong> menghasilkan kombinasi kerugian finansial terendah dan menjaga laju persetujuan portofolio di atas <strong style="color: #34d399;">80.3%</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Kepatuhan Regulasi & Standar Fair Lending")
    st.markdown("""
    Aplikasi ini dirancang dengan prinsip kepatuhan regulasi perbankan:
    1. **Equal Credit Opportunity Act (ECOA) & FCRA**: Kewajiban menerbitkan *Adverse Action Notice* dipenuhi melalui ekstraksi nilai SHAP lokal secara transparan tanpa diskriminasi.
    2. **Basel II / III Internal Ratings-Based (IRB) Compliance**: Metrik pemisahan risiko KS-Statistic (57.84%) dan kalibrasi skor log-odds memenuhi kualifikasi pemeringkatan risiko kredit internal bank.
    3. **Anti-Leakage Data Hygiene**: Pembersihan dan pra-pemrosesan data tidak melibatkan kebocoran target dari masa depan.
    """)

# -----------------------------------------------------------------------------
# 7. Footer & Portfolio Attributions
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-text">
    <p><strong>CreditRiskAI Portfolio Showcase</strong> | Developed by <strong>Ganendra Pradipa</strong></p>
    <p>Data Science & Risk Analytics Portfolio | Kaggle Give Me Some Credit Benchmark</p>
</div>
""", unsafe_allow_html=True)
