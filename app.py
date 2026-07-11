import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import datetime
from PIL import Image

# Import custom modules
from modules.core.state import init_session_state, StateHistory
from modules.core.quality import calculate_quality_score, get_improvement_suggestions
from modules.core.utils import load_dataset, export_dataset, optimize_memory
from modules.cleaning.missing_dups import impute_missing, drop_missing_rows, remove_duplicates, drop_empty_rows_cols
from modules.cleaning.text import clean_text_column
from modules.cleaning.types import auto_detect_types, convert_type
from modules.cleaning.outliers import detect_outliers_mask, handle_outliers
from modules.operations.columns import rename_columns, drop_columns, duplicate_column, merge_columns, split_column, reorder_columns
from modules.operations.rows import delete_rows, sort_rows, filter_rows_expression, random_sample
from modules.engineering.features import apply_math_op, create_conditional_col, extract_date_parts, calculate_age, calculate_date_difference, apply_log_transform, apply_binning, scale_features
from modules.engineering.encoding import encode_categorical
from modules.validation.checker import run_data_validation
from modules.eda.profiling import get_column_profile, get_dataset_summary_metrics
from modules.eda.visualizer import get_correlation_matrix_fig, get_distribution_fig, get_scatter_fig, get_line_fig, get_bar_fig, get_pie_fig, get_treemap_fig, get_violin_fig, get_qq_plot_fig
from modules.eda.reports import generate_ydata_report, generate_sweetviz_report, generate_custom_cleanmessy_report, YDATA_AVAILABLE, SWEETVIZ_AVAILABLE

# 1. Page Configuration
st.set_page_config(
    page_title="CleanMessy | Transform Messy Data into Clean Datasets",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

# 2. Inject CSS Theme
IS_DARK = st.session_state.theme == "dark"

# Define Theme Variables
if IS_DARK:
    bg = "#09090b"
    bg_subtle = "#0c0c0f"
    card = "#0c0c0f"
    card_hover = "#131316"
    border = "#1e1e24"
    border_subtle = "#16161a"
    border_hover = "#2a2a35"
    text = "#fafafa"
    text_muted = "#71717a"
    text_dim = "#52525b"
    accent = "#818cf8"
    accent_glow = "rgba(129, 140, 248, 0.12)"
    green = "#22c55e"
    green_muted = "rgba(34,197,94,0.12)"
    red = "#ef4444"
    red_muted = "rgba(239,68,68,0.12)"
    amber = "#f59e0b"
    amber_muted = "rgba(245,158,11,0.12)"
    shadow = "none"
else:
    bg = "#ffffff"
    bg_subtle = "#f9fafb"
    card = "#ffffff"
    card_hover = "#f4f4f5"
    border = "#e4e4e7"
    border_subtle = "#f0f0f2"
    border_hover = "#d4d4d8"
    text = "#09090b"
    text_muted = "#71717a"
    text_dim = "#a1a1aa"
    accent = "#2563eb"
    accent_glow = "rgba(37, 99, 235, 0.08)"
    green = "#16a34a"
    green_muted = "rgba(22,163,74,0.08)"
    red = "#dc2626"
    red_muted = "rgba(220,38,38,0.08)"
    amber = "#d97706"
    amber_muted = "rgba(217,119,6,0.08)"
    shadow = "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

# Read custom stylesheet
# Handle both local development and cloud deployment
css_template = ""
possible_paths = [
    os.path.join(os.path.dirname(__file__), "assets", "styles.css"),
    "assets/styles.css",
    "/app/cleanmessy/assets/styles.css",
]

for css_path in possible_paths:
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_template = f.read()
            break
    except FileNotFoundError:
        continue

if not css_template:
    st.warning("⚠️ CSS file not found. Using minimal styling.")

# Replace variables in CSS template
css_compiled = css_template.replace("var(--bg)", bg)\
                            .replace("var(--bg-subtle)", bg_subtle)\
                            .replace("var(--card)", card)\
                            .replace("var(--card-hover)", card_hover)\
                            .replace("var(--border)", border)\
                            .replace("var(--border-subtle)", border_subtle)\
                            .replace("var(--border-hover)", border_hover)\
                            .replace("var(--text)", text)\
                            .replace("var(--text-muted)", text_muted)\
                            .replace("var(--text-dim)", text_dim)\
                            .replace("var(--accent)", accent)\
                            .replace("var(--accent-glow)", accent_glow)\
                            .replace("var(--green)", green)\
                            .replace("var(--green-muted)", green_muted)\
                            .replace("var(--red)", red)\
                            .replace("var(--red-muted)", red_muted)\
                            .replace("var(--amber)", amber)\
                            .replace("var(--amber-muted)", amber_muted)\
                            .replace("var(--shadow)", shadow)

# Inject compiled CSS
st.markdown(f"<style>{css_compiled}</style>", unsafe_allow_html=True)
st.markdown(f"""
<style>
/* Responsive viewport */
@viewport {{ width: device-width; }}

header[data-testid="stHeader"] {{
    background: {bg} !important;
    border-bottom: 1px solid {border_subtle} !important;
}}
header[data-testid="stHeader"] > div {{
    padding-top: 0.2rem !important;
    padding-bottom: 0.2rem !important;
}}
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton {{
    display: none !important;
}}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"] {{
    position: fixed !important;
    top: 0.9rem !important;
    left: 0.9rem !important;
    z-index: 1001 !important;
    border-radius: 999px !important;
    border: 1px solid {border} !important;
    background: {card} !important;
    color: {text} !important;
    box-shadow: {shadow} !important;
}}
[data-testid="stSidebar"] {{
    min-width: 320px !important;
    max-width: 320px !important;
    border-right: 1px solid {border} !important;
    background: {bg} !important;
}}
[data-testid="stSidebar"][aria-expanded="false"] {{
    min-width: 320px !important;
    max-width: 320px !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: {bg} !important;
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container, section[data-testid="stMain"] {{
    background-color: {bg} !important;
    color: {text} !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
.block-container {{
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}}
/* Ensure images / iframes never overflow */
img, iframe, video {{ max-width: 100% !important; height: auto; }}

/* Streamlit's own column wrapper — allow flex-wrap for responsiveness */
[data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap !important;
    align-items: stretch !important;
}}

/* Plotly fills container width */
.js-plotly-plot, .plotly {{ width: 100% !important; }}
</style>
""", unsafe_allow_html=True)
# Also inject the viewport meta tag for proper mobile scaling
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
    unsafe_allow_html=True
)
st.markdown('<a class="skip-link" href="#main-content">Skip to content</a>', unsafe_allow_html=True)
st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

# 3. Custom UI Helper Components
def metric_card(label: str, value: str, delta: str = None, delta_type: str = "up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def push_df_state(df: pd.DataFrame, action: str):
    """Pushes a copy of df onto the state manager history."""
    st.session_state.dataset_history.push_state(df, action)
    st.session_state.current_df = st.session_state.dataset_history.get_current_df()

def require_dataset(page_title: str, button_key: str) -> pd.DataFrame:
    """Render an empty state for data-dependent pages until a dataset is loaded."""
    if st.session_state.current_df is not None:
        return st.session_state.current_df

    st.markdown(f"## {page_title}")
    st.info("Load a dataset first to use this section.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Go to Upload & Preview", key=f"{button_key}_upload", use_container_width=True):
            st.session_state.active_tab = "Upload & Preview"
            st.rerun()
    with c2:
        demo_available = os.path.exists("messy_dataset.csv")
        if st.button(
            "Load Demo Dataset",
            key=f"{button_key}_demo",
            use_container_width=True,
            disabled=not demo_available,
        ):
            try:
                df_demo = pd.read_csv("messy_dataset.csv")
                st.session_state.original_df = df_demo.copy()
                st.session_state.current_df = df_demo.copy()
                st.session_state.file_name = "messy_dataset.csv"
                st.session_state.upload_signature = None
                st.session_state.dataset_history = StateHistory(df_demo, "messy_dataset.csv")
                st.toast("Loaded messy_dataset.csv!", icon="✨")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load demo dataset: {str(e)}")

    if not os.path.exists("messy_dataset.csv"):
        st.caption("Demo dataset not found. Upload your own file to unlock this section.")

    st.stop()

# 4. Header Bar
def draw_header():
    head_left, head_mid, head_right = st.columns([7, 3, 2])
    with head_left:
        st.markdown("""
        <div class="brand">
            <span class="brand-icon">✨</span>
            <div>
                <span class="brand-name">CleanMessy</span>
                <div class="brand-tagline">Transform Messy Data into Clean, Analysis-Ready Datasets.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with head_mid:
        if st.session_state.dataset_history is not None:
            # Quick Undo / Redo controls in Header
            hist = st.session_state.dataset_history
            u_disabled = len(hist.history) <= 1
            r_disabled = len(hist.redo_history) == 0
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("↩️ Undo", disabled=u_disabled, key="header_undo", use_container_width=True):
                    undone = hist.undo()
                    if undone is not None:
                        st.session_state.current_df = undone
                        st.toast("Action Undone!", icon="↩️")
                        st.rerun()
            with c2:
                if st.button("↪️ Redo", disabled=r_disabled, key="header_redo", use_container_width=True):
                    redone = hist.redo()
                    if redone is not None:
                        st.session_state.current_df = redone
                        st.toast("Action Redone!", icon="↪️")
                        st.rerun()
            with c3:
                # Reset dataset to original
                if st.button("🔄 Reset", key="header_reset", use_container_width=True):
                    st.session_state.current_df = st.session_state.original_df.copy()
                    st.session_state.dataset_history = StateHistory(st.session_state.original_df, st.session_state.file_name)
                    st.toast("Dataset reset to original state!", icon="🔄")
                    st.rerun()
    with head_right:
        theme_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
        if st.button(theme_label, key="theme_toggle_btn", use_container_width=True):
            st.session_state.theme = "light" if IS_DARK else "dark"
            st.rerun()

draw_header()
st.markdown("---")

# 5. Sidebar Navigation
with st.sidebar:
    # Nav icons mapping
    nav_icons = {
        "Home": "🏠",
        "Upload & Preview": "📂",
        "Dashboard & Quality": "📊",
        "Column Profiling": "🔍",
        "Data Cleaning": "🧹",
        "Type Management": "🔄",
        "Column Operations": "📋",
        "Row Operations": "📝",
        "Feature Engineering": "⚙️",
        "Outlier Treatment": "🎯",
        "Data Validation": "✅",
        "EDA Charts": "📈",
        "Report Center": "📄",
        "Export / Download": "💾",
    }

    nav_options = list(nav_icons.keys())

    # Inject custom sidebar nav button CSS
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] .stMarkdown {{
        margin-bottom: 0 !important;
    }}
    .sidebar-brand {{
        padding: 0.25rem 0 1rem 0;
    }}
    .sidebar-brand-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        color: {text};
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }}
    .sidebar-brand-subtitle {{
        font-size: 0.78rem;
        color: {text_muted};
        line-height: 1.4;
    }}
    .sidebar-status {{
        margin: 0.85rem 0 1rem 0;
        padding: 0.85rem 0.9rem;
        border-radius: 12px;
        background: {bg_subtle};
        border: 1px solid {border};
    }}
    .sidebar-status-label {{
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {text_dim};
        margin-bottom: 0.25rem;
    }}
    .sidebar-status-value {{
        font-size: 0.82rem;
        color: {text};
        line-height: 1.4;
        font-weight: 600;
    }}
    .nav-group {{
        margin-bottom: 0.85rem;
        padding: 0.55rem;
        border-radius: 14px;
        border: 1px solid {border_subtle};
        background: {bg_subtle};
    }}
    .nav-section-label {{
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {text_muted};
        padding: 0.1rem 0.4rem 0.45rem;
    }}
    .nav-divider {{
        height: 1px;
        background: {border};
        margin: 0.85rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Section groupings
    nav_sections = [
        ("General", ["Home", "Upload & Preview"]),
        ("Analysis", ["Dashboard & Quality", "Column Profiling", "EDA Charts", "Report Center"]),
        ("Transform", ["Data Cleaning", "Type Management", "Outlier Treatment"]),
        ("Operations", ["Column Operations", "Row Operations"]),
        ("Engineering", ["Feature Engineering", "Data Validation"]),
        ("Output", ["Export / Download"]),
    ]

    active_tab = st.session_state.active_tab if st.session_state.active_tab in nav_options else "Home"
    st.session_state.active_tab = active_tab

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">CleanMessy Navigation</div>
            <div class="sidebar-brand-subtitle">Use the General section to upload data, then move through cleaning, analysis, and export.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.current_df is None:
        st.markdown(
            """
            <div class="sidebar-status">
                <div class="sidebar-status-label">Dataset Status</div>
                <div class="sidebar-status-value">No dataset loaded. Open <strong>Upload &amp; Preview</strong> from the General section to begin.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="sidebar-status">
                <div class="sidebar-status-label">Dataset Status</div>
                <div class="sidebar-status-value">{st.session_state.file_name or "Dataset loaded"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for section_label, section_items in nav_sections:
        st.markdown('<div class="nav-group">', unsafe_allow_html=True)
        st.markdown(f"##### {section_label}")
        for item in section_items:
            icon = nav_icons.get(item, "•")
            is_active = (item == active_tab)
            btn_label = f"{icon}  {item}"
            if st.button(
                btn_label,
                key=f"nav_{item}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_tab = item
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar Timeline if dataset is loaded
    if st.session_state.dataset_history is not None:
        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
        st.markdown("<div class='nav-section-label'>Operations History</div>", unsafe_allow_html=True)
        timeline = st.session_state.dataset_history.get_timeline()
        for item in reversed(timeline[-6:]):  # Show last 6 only
            dot_color = "#818cf8" if item["status"] == "active" else ("#71717a" if item["status"] == "past" else "#ef4444")
            st.markdown(
                f"""<div style='display:flex;align-items:flex-start;gap:0.5rem;padding:0.25rem 0.85rem;'>
                <span style='color:{dot_color};font-size:0.5rem;margin-top:5px;'>●</span>
                <div><div style='font-size:0.8rem;font-weight:600;color:#a1a1aa;'>{item['action'][:32]}</div>
                <div style='font-size:0.68rem;color:#52525b;'>{item['timestamp']}</div></div></div>""",
                unsafe_allow_html=True
            )

        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
        st.markdown("<div class='nav-section-label'>Settings</div>", unsafe_allow_html=True)
        st.session_state.auto_save = st.checkbox("Auto Save State", value=st.session_state.auto_save)
        st.session_state.default_format = st.selectbox("Download Format", ["CSV", "Excel", "JSON", "Parquet", "Pickle"], index=0)

        if st.button("⚡ Optimize RAM Usage", use_container_width=True, key="sidebar_opt_mem"):
            opt_df, stats = optimize_memory(st.session_state.current_df)
            push_df_state(opt_df, f"Memory Optimization Applied (Saved {stats['saved_mem_mb']} MB)")
            st.toast(f"Saved {stats['pct_saved']}% memory!", icon="⚡")
            st.rerun()

# 6. Page Render Logic

# ==========================================
# HOME PAGE
# ==========================================
if st.session_state.active_tab == "Home":
    st.markdown("""
    <div class="hero-card">
        <h1 class="hero-title">Transform Messy Data into Clean, Analysis-Ready Datasets.</h1>
        <p class="hero-subtitle">CleanMessy combines smart automation with granular manual control to accelerate your data wrangling. Upload, profile, fix data types, impute nulls, handle outliers, and construct features in a unified premium workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Data Profiling & Quality</div>
            <div class="feature-text">Analyze column statistics, check class frequencies, and calculate a Data Quality Score using modular evaluation heuristics.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧼</div>
            <div class="feature-title">Granular Wrangling</div>
            <div class="feature-text">Execute math column transforms, string casing, search-replace, missing value imputations, and outlier winsorization.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">One-Click Smart Clean</div>
            <div class="feature-text">Trigger a full pipeline sweep: normalize text casing, impute empty cells, trim spaces, drop constants, and optimize memory types.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3>Supported Input & Output File Formats</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **CSV / TSV**: Standard comma/tab separated files.
    - **Excel (.xlsx / .xls)**: Multi-sheet spreadsheet files.
    - **JSON**: Structured list of key-value objects.
    - **Parquet**: Standard column-oriented data format.
    """)
    
    if st.button("🚀 Upload Your Dataset to Get Started", use_container_width=True):
        st.session_state.active_tab = "Upload & Preview"
        st.rerun()

# ==========================================
# UPLOAD & PREVIEW
# ==========================================
elif st.session_state.active_tab == "Upload & Preview":
    st.markdown("## Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your dataset here", 
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"]
    )
    
    if os.path.exists("messy_dataset.csv") and st.session_state.current_df is None:
        st.markdown("---")
        st.markdown("#### 💡 Test with Mock Dataset")
        st.info("A demo dataset (`messy_dataset.csv`) was found in your directory. Click below to load it automatically.")
        if st.button("👉 Load Demo Dataset (messy_dataset.csv)", use_container_width=True):
            try:
                df_demo = pd.read_csv("messy_dataset.csv")
                st.session_state.original_df = df_demo.copy()
                st.session_state.current_df = df_demo.copy()
                st.session_state.file_name = "messy_dataset.csv"
                st.session_state.upload_signature = None
                st.session_state.dataset_history = StateHistory(df_demo, "messy_dataset.csv")
                st.toast("Loaded messy_dataset.csv!", icon="✨")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load demo dataset: {str(e)}")
                
    if uploaded_file is not None:
        upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        needs_reload = (
            st.session_state.upload_signature != upload_signature
            or st.session_state.original_df is None
        )

        if needs_reload:
            with st.spinner(f"Loading {uploaded_file.name}..."):
                df, err = load_dataset(uploaded_file, uploaded_file.name)
            if err:
                st.error(err)
            else:
                st.session_state.original_df = df.copy()
                st.session_state.current_df = df.copy()
                st.session_state.file_name = uploaded_file.name
                st.session_state.upload_signature = upload_signature
                st.session_state.dataset_history = StateHistory(df, uploaded_file.name)
                st.success(f"Successfully loaded {uploaded_file.name}!")
                st.rerun()
                
    if st.session_state.current_df is not None:
        df_active = st.session_state.current_df
        st.markdown(f"### File: `{st.session_state.file_name}`")
        
        # Display KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Rows Count", f"{df_active.shape[0]:,}")
        with k2:
            metric_card("Columns Count", f"{df_active.shape[1]:,}")
        with k3:
            total_cells = df_active.size
            missing = df_active.isnull().sum().sum()
            missing_pct = (missing / total_cells) * 100 if total_cells > 0 else 0
            metric_card("Missing Values", f"{missing:,} ({missing_pct:.1f}%)", delta_type="warn" if missing_pct > 10 else "up")
        with k4:
            memory_mb = df_active.memory_usage(deep=True).sum() / (1024 * 1024)
            metric_card("Memory Footprint", f"{memory_mb:.2f} MB")
            
        st.markdown("### Dataset Preview Options")
        preview_mode = st.radio("Select preview mode:", ["First 20 rows", "Last 20 rows", "Random Sample (20 rows)", "Full Dataset Statistics"], horizontal=True)
        
        if preview_mode == "First 20 rows":
            st.dataframe(df_active.head(20), use_container_width=True)
        elif preview_mode == "Last 20 rows":
            st.dataframe(df_active.tail(20), use_container_width=True)
        elif preview_mode == "Random Sample (20 rows)":
            st.dataframe(df_active.sample(min(20, len(df_active))), use_container_width=True)
        else:
            # Summary datatypes
            dtypes_df = pd.DataFrame({
                'Column Name': df_active.columns,
                'Inferred Datatype': [str(t) for t in df_active.dtypes],
                'Non-Null Count': df_active.count().values,
                'Null Count': df_active.isnull().sum().values
            })
            st.dataframe(dtypes_df, use_container_width=True)

# ==========================================
# DASHBOARD & QUALITY ASSESSMENT
# ==========================================
elif st.session_state.active_tab == "Dashboard & Quality":
    df_active = require_dataset("Data Quality Dashboard", "quality_dashboard")
    st.markdown("## Data Quality Dashboard")
    
    # Calculate Quality metrics
    metrics = get_dataset_summary_metrics(df_active)
    quality = calculate_quality_score(df_active)
    suggestions = get_improvement_suggestions(df_active)
    
    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset Score", f"{quality['score']}%")
    with c2:
        metric_card("Duplicate Rows", f"{metrics['duplicates']:,}")
    with c3:
        metric_card("Numeric Columns", f"{metrics['numeric']}")
    with c4:
        metric_card("Categorical Columns", f"{metrics['categorical']}")
        
    st.markdown("### Score Breakdown")
    sb1, sb2, sb3, sb4, sb5 = st.columns(5)
    with sb1:
        st.metric("Completeness (Nulls)", f"{quality['missing']}%")
    with sb2:
        st.metric("Uniqueness (Dups)", f"{quality['duplicates']}%")
    with sb3:
        st.metric("Regularity (Outliers)", f"{quality['outliers']}%")
    with sb4:
        st.metric("DataType Consistency", f"{quality['consistency']}%")
    with sb5:
        st.metric("Memory Efficiency", f"{quality['memory']}%")
        
    # Recommendations & Smart Clean Split
    s_col1, s_col2 = st.columns([2, 1])
    with s_col1:
        st.markdown("### Actionable Quality Recommendations")
        if len(suggestions) == 0:
            st.success("🎉 Outstanding! Your dataset contains 0 quality violations.")
        else:
            for sug in suggestions:
                # Severity badge colors
                b_color = "badge-red" if sug['severity'] == "danger" else ("badge-amber" if sug['severity'] == "warning" else "badge-blue")
                st.markdown(f"""
                <div class="card" style="padding: 1rem; border-left: 4px solid var(--accent); margin-bottom: 0.8rem;">
                    <span class="badge {b_color}">{sug['category']}</span>
                    <p style="margin: 0.3rem 0;">{sug['message']}</p>
                    <small style="color: var(--accent);"><strong>Fix:</strong> {sug['action']}</small>
                </div>
                """, unsafe_allow_html=True)
                
    with s_col2:
        st.markdown("### One-Click Smart Clean")
        st.info("Execute automated clean strategies: trim strings, normalize casing, impute numerical and text blanks, remove duplicate entries, drop static columns, and downcast memory sizes.")
        
        if st.button("⚡ Trigger Smart Clean", use_container_width=True):
            # Automated Pipeline Executions
            df_cleaned = df_active.copy()
            # 1. Trim Spaces
            text_cols = df_cleaned.select_dtypes(include=['object', 'str']).columns.tolist()
            if text_cols:
                df_cleaned = clean_text_column(df_cleaned, text_cols, 'trim')
                df_cleaned = clean_text_column(df_cleaned, text_cols, 'remove_extra_spaces')
            # 2. Impute nulls
            num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df_cleaned = impute_missing(df_cleaned, num_cols, 'median')
            cat_cols = df_cleaned.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
            if cat_cols:
                df_cleaned = impute_missing(df_cleaned, cat_cols, 'mode')
            # 3. Remove duplicates
            df_cleaned = remove_duplicates(df_cleaned, keep='first')
            # 4. Drop constant columns
            const_cols = [c for c in df_cleaned.columns if df_cleaned[c].nunique() <= 1]
            if const_cols:
                df_cleaned = drop_columns(df_cleaned, const_cols)
            # 5. Optimize memory
            df_cleaned, opt_stats = optimize_memory(df_cleaned)
            
            push_df_state(df_cleaned, "One-Click Smart Clean Executed")
            st.toast("Dataset Smart Clean complete!", icon="⚡")
            st.rerun()

# ==========================================
# COLUMN PROFILING
# ==========================================
elif st.session_state.active_tab == "Column Profiling":
    df_active = require_dataset("Column Profiling & Descriptive Statistics", "column_profiling")
    st.markdown("## Column Profiling & Descriptive Statistics")

    col_to_profile = st.selectbox("Select a column to examine:", df_active.columns)

    profile = get_column_profile(df_active, col_to_profile)

    if profile:
        # Always-visible header — no backticks so no hover/tooltip effect
        dtype_display = str(profile['dtype'])
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;'>"
            f"<h3 style='margin:0;'>Profile: {col_to_profile}</h3>"
            f"<span style='font-family:monospace;font-size:0.78rem;padding:3px 10px;"
            f"background:rgba(129,140,248,0.12);color:#818cf8;border-radius:6px;"
            f"border:1px solid rgba(129,140,248,0.2);'>{dtype_display}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            # Build stat rows — no markdown backticks in values
            p_rows = [
                ("Logical Datatype", dtype_display),
                ("Total Rows", f"{profile['count']:,}"),
                ("Non-Null Values", f"{profile['non_null']:,}"),
                ("Missing Values", f"{profile['missing']:,} ({profile['missing_pct']}%)"),
                ("Unique Values", f"{profile['unique']:,}"),
                ("Most Common (Mode)", str(profile['mode'])),
            ]

            if "mean" in profile:
                p_rows.extend([
                    ("Mean", f"{profile['mean']:,}"),
                    ("Median (Q2)", f"{profile['median']:,}"),
                    ("Min", f"{profile['min']:,}"),
                    ("Max", f"{profile['max']:,}"),
                    ("Std Dev", f"{profile['std']:,}"),
                    ("Variance", f"{profile['var']:,}"),
                    ("Skewness", str(profile['skew'])),
                    ("Kurtosis", str(profile['kurtosis'])),
                ])

            rows_html = "".join([
                f"<tr><td style='padding:0.6rem 0.9rem;color:#71717a;font-size:0.8rem;font-weight:600;'"
                f">{label}</td><td style='padding:0.6rem 0.9rem;text-align:right;font-weight:600;"
                f"font-size:0.85rem;'>{val}</td></tr>"
                for label, val in p_rows
            ])
            st.markdown(f"""
            <table style="width:100%;border-collapse:separate;border-spacing:0;
                          border:1px solid rgba(100,100,120,0.15);border-radius:10px;overflow:hidden;">
                <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

        with c2:
            st.markdown("**Column Distribution**")
            if pd.api.types.is_numeric_dtype(df_active[col_to_profile]):
                fig = get_distribution_fig(df_active, col_to_profile, IS_DARK)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                freq_vals = profile.get('frequent_values', [])
                if freq_vals:
                    freq_df = pd.DataFrame(freq_vals)
                    fig = px.bar(
                        freq_df,
                        x='value', y='count',
                        labels={'value': col_to_profile, 'count': 'Count'},
                        title=f"Top Values — {col_to_profile}",
                        color_discrete_sequence=['#818cf8' if IS_DARK else '#2563eb']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=40, b=10),
                        font=dict(family='DM Sans, sans-serif', size=11)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("High-cardinality column — no top values to chart.")

# ==========================================
# DATA CLEANING
# ==========================================
elif st.session_state.active_tab == "Data Cleaning":
    df_active = require_dataset("Data Cleaning Operations", "data_cleaning")
    st.markdown("## Data Cleaning Operations")
    
    clean_tab1, clean_tab2, clean_tab3 = st.tabs(["Handle Missing Values", "Handle Duplicate Rows", "String & Text Cleaning"])
    
    with clean_tab1:
        st.markdown("### Missing Value Imputations")
        target_cols = st.multiselect("Select columns to fill:", df_active.columns)
        impute_strategy = st.selectbox(
            "Select imputation strategy:",
            ["mean", "median", "mode", "ffill", "bfill", "constant"]
        )
        
        custom_val = None
        if impute_strategy == "constant":
            custom_val = st.text_input("Enter custom fill value:")
            
        if st.button("Apply Imputation", key="apply_impute_btn"):
            if not target_cols:
                st.warning("Please select at least one column.")
            else:
                # Preview operations
                df_temp = impute_missing(df_active, target_cols, impute_strategy, custom_val)
                push_df_state(df_temp, f"Imputed missing values in {', '.join(target_cols)} via {impute_strategy}")
                st.success("Imputation complete!")
                st.rerun()
                
        # Drop rows missing values option
        st.markdown("#### Drop Rows with Missing Values")
        drop_target_cols = st.multiselect("Select columns:", df_active.columns, key="drop_missing_cols")
        if st.button("Drop Rows", key="apply_drop_missing"):
            if not drop_target_cols:
                st.warning("Please select columns.")
            else:
                df_temp = drop_missing_rows(df_active, drop_target_cols)
                push_df_state(df_temp, f"Dropped rows with missing values in {', '.join(drop_target_cols)}")
                st.success(f"Dropped rows! New shape: {df_temp.shape}")
                st.rerun()

    with clean_tab2:
        st.markdown("### Duplicate Row Treatments")
        subset_cols = st.multiselect("Select subset columns to evaluate uniqueness (leave empty to evaluate full row):", df_active.columns)
        keep_strat = st.selectbox("Keep strategy:", ["first", "last", "none"])
        
        # Display current count
        dup_count = df_active.duplicated(subset=subset_cols if subset_cols else None).sum()
        st.info(f"Currently found: **{dup_count}** duplicate rows.")
        
        if st.button("Remove Duplicates", key="apply_rm_dups"):
            df_temp = remove_duplicates(df_active, subset=subset_cols if subset_cols else None, keep=keep_strat)
            push_df_state(df_temp, f"Removed duplicate rows (keeping: {keep_strat})")
            st.success("Duplicates removed successfully!")
            st.rerun()

    with clean_tab3:
        st.markdown("### String and Text Normalizations")
        str_cols = st.multiselect("Select text columns to clean:", df_active.select_dtypes(include=['object', 'str']).columns)
        
        text_op = st.selectbox(
            "Select text operation:",
            [
                ("Trim spaces (leading & trailing)", "trim"),
                ("Remove extra spaces (interior multiple spaces)", "remove_extra_spaces"),
                ("Remove punctuation", "remove_punct"),
                ("Remove numbers", "remove_nums"),
                ("Remove special characters", "remove_special"),
                ("Convert to UPPERCASE", "upper"),
                ("Convert to lowercase", "lower"),
                ("Convert to Title Case", "title"),
                ("Find & Replace", "find_replace"),
                ("Replace Missing/Empty Text with Placeholder", "replace_missing")
            ],
            format_func=lambda x: x[0]
        )
        
        find_val = None
        replace_val = None
        if text_op[1] == "find_replace":
            find_val = st.text_input("Find value:")
            replace_val = st.text_input("Replace with:")
        elif text_op[1] == "replace_missing":
            replace_val = st.text_input("Placeholder value:")
            
        if st.button("Clean Text Columns", key="apply_text_clean"):
            if not str_cols:
                st.warning("Please select at least one text column.")
            else:
                df_temp = clean_text_column(df_active, str_cols, text_op[1], find_val, replace_val)
                push_df_state(df_temp, f"Cleaned text in {', '.join(str_cols)}: {text_op[0]}")
                st.success("Text cleaning applied!")
                st.rerun()

# ==========================================
# DATA TYPE MANAGEMENT
# ==========================================
elif st.session_state.active_tab == "Type Management":
    df_active = require_dataset("Data Type Management", "type_management")
    st.markdown("## Data Type Management")
    
    st.markdown("### Current Inferred Columns & Types")
    detected = auto_detect_types(df_active)
    
    # Renders a table of columns, python types, and proposed types
    type_rows = []
    for col in df_active.columns:
        type_rows.append({
            "Column Name": col,
            "Pandas Type": str(df_active[col].dtype),
            "CleanMessy Inferred Type": detected[col]
        })
    st.dataframe(pd.DataFrame(type_rows), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Convert Column Datatype")
    col_to_conv = st.selectbox("Select column to convert:", df_active.columns)
    target_dt = st.selectbox("Target data type:", ["Integer", "Float", "Boolean", "Category", "Object", "Datetime"])
    
    dt_fmt = None
    if target_dt == "Datetime":
        dt_fmt = st.text_input("Optional Date format (e.g. %Y-%m-%d or %d/%m/%Y):", placeholder="Automatic detection")
        if dt_fmt == "":
            dt_fmt = None
            
    if st.button("Convert Datatype"):
        try:
            df_temp = convert_type(df_active, col_to_conv, target_dt, dt_fmt)
            push_df_state(df_temp, f"Converted column '{col_to_conv}' datatype to {target_dt}")
            st.success(f"Column '{col_to_conv}' converted successfully to {target_dt}!")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# ==========================================
# COLUMN OPERATIONS
# ==========================================
elif st.session_state.active_tab == "Column Operations":
    df_active = require_dataset("Column Operations", "column_operations")
    st.markdown("## Column-Level Operations")
    
    c_tab1, c_tab2, c_tab3, c_tab4, c_tab5 = st.tabs(["Rename Columns", "Drop Columns", "Merge Columns", "Split Columns", "Reorder Columns"])
    
    with c_tab1:
        st.markdown("### Rename Column")
        col_ren = st.selectbox("Select column to rename:", df_active.columns, key="rename_sel")
        new_name = st.text_input("Enter new name:")
        if st.button("Rename Column", key="apply_rename_btn"):
            if new_name == "":
                st.warning("New name cannot be empty.")
            else:
                df_temp = rename_columns(df_active, {col_ren: new_name})
                push_df_state(df_temp, f"Renamed column '{col_ren}' to '{new_name}'")
                st.success("Column renamed successfully!")
                st.rerun()

    with c_tab2:
        st.markdown("### Drop Column(s)")
        cols_drop = st.multiselect("Select columns to drop:", df_active.columns)
        if st.button("Drop Selected Columns", key="apply_drop_cols_btn"):
            if len(cols_drop) == 0:
                st.warning("Please select at least one column.")
            else:
                df_temp = drop_columns(df_active, cols_drop)
                push_df_state(df_temp, f"Dropped columns: {', '.join(cols_drop)}")
                st.success(f"Columns dropped! Remaining columns count: {df_temp.shape[1]}")
                st.rerun()

    with c_tab3:
        st.markdown("### Merge Columns")
        cols_merge = st.multiselect("Select two or more columns to merge:", df_active.columns)
        merge_name = st.text_input("Enter new merged column name:")
        merge_sep = st.text_input("Enter separator:", value=" ")
        
        if st.button("Merge Columns", key="apply_merge_cols_btn"):
            if len(cols_merge) < 2 or merge_name == "":
                st.warning("Please select at least 2 columns and provide a new column name.")
            else:
                df_temp = merge_columns(df_active, cols_merge, merge_name, merge_sep)
                push_df_state(df_temp, f"Merged columns {', '.join(cols_merge)} into '{merge_name}'")
                st.success(f"Merged columns successfully! Created column: {merge_name}")
                st.rerun()

    with c_tab4:
        st.markdown("### Split Columns")
        col_split = st.selectbox("Select column to split:", df_active.columns, key="split_sel")
        split_sep = st.text_input("Enter delimiter separator:", value="-")
        new_names_str = st.text_input("Enter new column names (separated by commas):", placeholder="Part1, Part2")
        
        if st.button("Split Column", key="apply_split_col_btn"):
            new_names = [n.strip() for n in new_names_str.split(",") if n.strip()]
            if not new_names:
                st.warning("Please provide names for the split columns.")
            else:
                df_temp = split_column(df_active, col_split, split_sep, new_names)
                push_df_state(df_temp, f"Split column '{col_split}' into {new_names}")
                st.success(f"Split column successfully! Created columns: {', '.join(new_names)}")
                st.rerun()

    with c_tab5:
        st.markdown("### Reorder / Shift Columns")
        # List of columns to reorder
        col_order_list = st.multiselect("Reorder columns (columns listed first will appear on left):", df_active.columns, default=df_active.columns.tolist())
        if st.button("Apply Column Order", key="apply_reorder_cols_btn"):
            df_temp = reorder_columns(df_active, col_order_list)
            push_df_state(df_temp, "Reordered columns")
            st.success("Column order updated!")
            st.rerun()

# ==========================================
# ROW OPERATIONS
# ==========================================
elif st.session_state.active_tab == "Row Operations":
    df_active = require_dataset("Row Operations", "row_operations")
    st.markdown("## Row-Level Operations")
    
    r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs(["Delete Rows", "Sort Rows", "Filter Rows", "Random Sampling"])
    
    with r_tab1:
        st.markdown("### Delete Rows by Index")
        row_del_input = st.text_input("Enter row index/indices to drop (comma-separated):")
        if st.button("Delete Rows"):
            indices_to_drop = []
            for i in row_del_input.split(","):
                i_str = i.strip()
                if i_str.isdigit():
                    indices_to_drop.append(int(i_str))
                elif i_str:
                    # In case of string indices
                    indices_to_drop.append(i_str)
                    
            if not indices_to_drop:
                st.warning("Please provide valid indices.")
            else:
                df_temp = delete_rows(df_active, indices_to_drop)
                push_df_state(df_temp, f"Deleted rows with indices: {indices_to_drop}")
                st.success(f"Rows deleted! New shape: {df_temp.shape}")
                st.rerun()

    with r_tab2:
        st.markdown("### Sort Rows by Column")
        col_sort = st.multiselect("Select columns to sort by:", df_active.columns)
        sort_order = st.radio("Order:", ["Ascending", "Descending"])
        
        if st.button("Sort Rows", key="apply_sort_btn"):
            if not col_sort:
                st.warning("Select columns.")
            else:
                df_temp = sort_rows(df_active, col_sort, ascending=(sort_order == "Ascending"))
                push_df_state(df_temp, f"Sorted dataset by {', '.join(col_sort)} ({sort_order})")
                st.success("Dataset sorted!")
                st.rerun()

    with r_tab3:
        st.markdown("### Filter Rows using Query Expressions")
        st.info("Write standard Python expressions e.g. `Age > 30 & Status == 'Active'`")
        filter_expr = st.text_input("Query expression:")
        if st.button("Apply Filter", key="apply_filter_btn"):
            if filter_expr == "":
                st.warning("Expression cannot be blank.")
            else:
                try:
                    df_temp = filter_rows_expression(df_active, filter_expr)
                    push_df_state(df_temp, f"Filtered dataset using: '{filter_expr}'")
                    st.success(f"Filtered dataset! Returned {len(df_temp)} rows.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with r_tab4:
        st.markdown("### Random Dataset Sampling")
        sample_mode = st.radio("Sampling method:", ["Absolute Rows Count", "Fraction of Dataset"])
        
        sample_val = st.number_input("Value:", min_value=0.0, step=1.0)
        
        if st.button("Generate Sample"):
            if sample_mode == "Absolute Rows Count":
                df_temp = random_sample(df_active, n=int(sample_val))
                push_df_state(df_temp, f"Sampled {int(sample_val)} random rows")
            else:
                df_temp = random_sample(df_active, frac=float(sample_val))
                push_df_state(df_temp, f"Sampled {float(sample_val)*100}% fraction of rows")
            st.success(f"Sample generated! New shape: {df_temp.shape}")
            st.rerun()

# ==========================================
# FEATURE ENGINEERING
# ==========================================
elif st.session_state.active_tab == "Feature Engineering":
    df_active = require_dataset("Feature Engineering", "feature_engineering")
    st.markdown("## Feature Engineering & Preprocessing")
    
    fe_tab1, fe_tab2, fe_tab3, fe_tab4, fe_tab5 = st.tabs(["Mathematical & Conditional", "Date & Time Engineering", "Feature Scaling", "Categorical Encoding", "Data Binning & Log Transforms"])
    
    with fe_tab1:
        st.markdown("### Custom Column Arithmetic")
        fe_col1 = st.selectbox("Select Column:", df_active.select_dtypes(include=[np.number]).columns, key="fe_math_c1")
        fe_op = st.selectbox("Arithmetic Operator:", ["add", "subtract", "multiply", "divide"])
        fe_operand = st.text_input("Operand (either constant numeric or another column name):")
        fe_new_name = st.text_input("New Column Name:", key="fe_math_new_name")
        
        if st.button("Apply Math Operation", key="apply_math_op_btn"):
            if fe_new_name == "" or fe_operand == "":
                st.warning("Please fill all inputs.")
            else:
                try:
                    df_temp = apply_math_op(df_active, fe_col1, fe_op, fe_operand, fe_new_name)
                    push_df_state(df_temp, f"Created column '{fe_new_name}' via: {fe_col1} {fe_op} {fe_operand}")
                    st.success("New column created!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    
        st.markdown("---")
        st.markdown("### Conditional Columns")
        cond_col = st.selectbox("Evaluate Condition on Column:", df_active.columns)
        cond_op = st.selectbox("Operator:", ["==", "!=", ">", "<", ">=", "<=", "contains"])
        cond_val = st.text_input("Value:")
        true_val = st.text_input("Value if True:")
        false_val = st.text_input("Value if False:")
        cond_new_name = st.text_input("New Column Name:", key="fe_cond_new_name")
        
        if st.button("Apply Condition", key="apply_cond_btn"):
            if cond_new_name == "":
                st.warning("New column name cannot be empty.")
            else:
                df_temp = create_conditional_col(df_active, cond_new_name, cond_col, cond_op, cond_val, true_val, false_val)
                push_df_state(df_temp, f"Created conditional column '{cond_new_name}' on '{cond_col}'")
                st.success("Conditional column added!")
                st.rerun()

    with fe_tab2:
        st.markdown("### Date Extraction")
        date_col = st.selectbox("Select Date Column:", df_active.columns, key="date_ext_sel")
        parts = st.multiselect("Select parts to extract:", ["Year", "Month", "Day", "DayOfWeek", "Quarter", "Hour"], default=["Year", "Month", "Day"])
        
        if st.button("Extract Date Parts"):
            if not parts:
                st.warning("Please select parts.")
            else:
                df_temp = extract_date_parts(df_active, date_col, parts)
                push_df_state(df_temp, f"Extracted {', '.join(parts)} from date '{date_col}'")
                st.success("Date parts extracted!")
                st.rerun()
                
        st.markdown("---")
        st.markdown("### Age Calculation")
        age_birth_col = st.selectbox("Select Birth Date Column:", df_active.columns, key="age_birth_sel")
        age_target_col = st.text_input("New Column Name for Age:", value="Age")
        
        if st.button("Calculate Age"):
            df_temp = calculate_age(df_active, age_birth_col, age_target_col)
            push_df_state(df_temp, f"Calculated age from birthdate '{age_birth_col}'")
            st.success("Age computed!")
            st.rerun()
            
        st.markdown("---")
        st.markdown("### Difference Between Dates")
        diff_col1 = st.selectbox("Start Date Column:", df_active.columns, key="diff_col1_sel")
        diff_col2 = st.selectbox("End Date Column:", df_active.columns, key="diff_col2_sel")
        diff_unit = st.selectbox("Difference Unit:", ["days", "weeks", "years", "hours"])
        diff_new_col = st.text_input("New Column Name:", key="diff_new_col")
        
        if st.button("Calculate Date Difference"):
            if diff_new_col == "":
                st.warning("New column name required.")
            else:
                df_temp = calculate_date_difference(df_active, diff_col1, diff_col2, diff_new_col, diff_unit)
                push_df_state(df_temp, f"Calculated date difference: {diff_col2} - {diff_col1} ({diff_unit})")
                st.success("Difference calculated!")
                st.rerun()

    with fe_tab3:
        st.markdown("### Feature Scaling")
        scale_cols = st.multiselect("Select columns to scale:", df_active.select_dtypes(include=[np.number]).columns)
        scaler_method = st.selectbox("Scaling Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "MaxAbsScaler"])
        
        if st.button("Scale Columns"):
            if not scale_cols:
                st.warning("Please select columns.")
            else:
                df_temp = scale_features(df_active, scale_cols, scaler_method)
                push_df_state(df_temp, f"Applied scaling ({scaler_method}) on columns: {', '.join(scale_cols)}")
                st.success("Scaling applied successfully!")
                st.rerun()

    with fe_tab4:
        st.markdown("### Categorical Encoding")
        enc_cols = st.multiselect("Select categorical columns to encode:", df_active.select_dtypes(include=['object', 'str', 'category']).columns)
        enc_method = st.selectbox(
            "Encoding Method:", 
            ["Label Encoding", "One Hot Encoding", "Ordinal Encoding", "Frequency Encoding", "Binary Encoding", "Hash Encoding", "Target Encoding"]
        )
        
        # Target column selection if target encoding
        t_col = None
        if enc_method == "Target Encoding":
            t_col = st.selectbox("Select Target Column:", df_active.select_dtypes(include=[np.number]).columns)
            
        enc_params = {}
        if enc_method == "Hash Encoding":
            enc_params['n_components'] = st.number_input("Number of components (hash bins):", min_value=2, value=8, step=1)
            
        if st.button("Encode Features"):
            if not enc_cols:
                st.warning("Please select columns.")
            else:
                try:
                    df_temp = encode_categorical(df_active, enc_cols, enc_method, target_column=t_col, params=enc_params)
                    push_df_state(df_temp, f"Encoded columns {', '.join(enc_cols)} using {enc_method}")
                    st.success("Encoding completed!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with fe_tab5:
        st.markdown("### Numerical Binning")
        bin_col = st.selectbox("Select numeric column to bin:", df_active.select_dtypes(include=[np.number]).columns)
        num_bins = st.number_input("Number of bins:", min_value=2, value=4, step=1)
        bin_strategy = st.selectbox("Binning Strategy:", ["uniform", "quantile"])
        
        labels_str = st.text_input("Optional comma-separated bin labels (leave blank for automatic numerical intervals):")
        bin_labels = [l.strip() for l in labels_str.split(",") if l.strip()] if labels_str else None
        
        bin_new_name = st.text_input("New Column Name:", value=f"{bin_col}_binned")
        
        if st.button("Bin Feature"):
            df_temp = apply_binning(df_active, bin_col, num_bins, bin_labels, bin_new_name, bin_strategy)
            push_df_state(df_temp, f"Binned '{bin_col}' into {num_bins} bins ({bin_strategy})")
            st.success("Binning complete!")
            st.rerun()
            
        st.markdown("---")
        st.markdown("### Log Transformations")
        log_col = st.selectbox("Select column for log(x + 1) transform:", df_active.select_dtypes(include=[np.number]).columns, key="log_col_sel")
        log_new_name = st.text_input("New Column Name:", value=f"{log_col}_log")
        
        if st.button("Apply Log Transform"):
            df_temp = apply_log_transform(df_active, log_col, log_new_name)
            push_df_state(df_temp, f"Log transformed '{log_col}' into '{log_new_name}'")
            st.success("Log transform applied!")
            st.rerun()

# ==========================================
# OUTLIER TREATMENT
# ==========================================
elif st.session_state.active_tab == "Outlier Treatment":
    df_active = require_dataset("Outlier Detection & Treatment", "outlier_treatment")
    st.markdown("## Outlier Detection & Treatment")
    
    out_col = st.selectbox("Select numeric column to evaluate:", df_active.select_dtypes(include=[np.number]).columns)
    out_method = st.selectbox("Outlier Detection Method:", ["IQR", "Z Score", "Modified Z Score", "Isolation Forest", "Local Outlier Factor", "DBSCAN"])
    
    # Render parameter controllers
    out_params = {}
    if out_method == "IQR":
        out_params['threshold'] = st.slider("IQR threshold factor:", 1.0, 3.0, 1.5, step=0.1)
    elif out_method in ["Z Score", "Modified Z Score"]:
        out_params['threshold'] = st.slider("Z-Score threshold count:", 1.5, 5.0, 3.0 if out_method == "Z Score" else 3.5, step=0.1)
    elif out_method in ["Isolation Forest", "Local Outlier Factor"]:
        out_params['contamination'] = st.slider("Contamination rate:", 0.01, 0.2, 0.05, step=0.01)
        if out_method == "Local Outlier Factor":
            out_params['n_neighbors'] = st.number_input("Neighbors count:", min_value=2, value=20)
    elif out_method == "DBSCAN":
        out_params['eps'] = st.number_input("Epsilon (neighborhood distance):", min_value=0.01, value=0.5, step=0.1)
        out_params['min_samples'] = st.number_input("Minimum samples per cluster:", min_value=1, value=5)

    # Detect outliers and render metrics
    mask, meta = detect_outliers_mask(df_active, out_col, out_method, out_params)
    outlier_count = mask.sum()
    outlier_pct = (outlier_count / len(df_active)) * 100
    
    # Render Outlier metrics
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        metric_card("Outlier Rows Count", f"{outlier_count:,}")
    with col_k2:
        metric_card("Outlier Percentage", f"{outlier_pct:.2f}%", delta_type="warn" if outlier_pct > 5 else "up")

    st.markdown("---")
    
    # Previews visual side-by-side
    st.markdown("### Outlier Visual Distribution")
    viz_type = st.radio("Preview plot:", ["Box Plot", "Distribution Plot (Histogram)"], horizontal=True)
    if viz_type == "Box Plot":
        fig = px.box(df_active, y=out_col, points="all", title=f"Box Plot containing outliers in {out_col}")
        # Custom color outliers in plot
        fig.update_traces(marker=dict(color='#ef4444' if IS_DARK else '#dc2626'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Highlight outliers in distribution
        df_viz = df_active.copy()
        df_viz['Is_Outlier'] = mask.map({True: 'Outlier', False: 'Normal'})
        fig = px.histogram(df_viz, x=out_col, color='Is_Outlier', color_discrete_map={'Normal': '#818cf8' if IS_DARK else '#2563eb', 'Outlier': '#ef4444'})
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("### Select Outlier Action")
    out_action = st.selectbox("Action:", ["Cap", "Replace", "Remove"])
    
    if st.button("Apply Outlier Treatment"):
        df_temp = handle_outliers(df_active, out_col, out_method, out_action, out_params)
        push_df_state(df_temp, f"Applied outlier treatment: {out_action} on '{out_col}' using {out_method}")
        st.success("Outliers treated successfully!")
        st.rerun()

# ==========================================
# DATA VALIDATION
# ==========================================
elif st.session_state.active_tab == "Data Validation":
    df_active = require_dataset("Data Validation Report", "data_validation")
    st.markdown("## Data Validation Report")
    
    st.info("Validation scans for schema anomalies, negative values, high cardinalities, formatting spaces, and duplicate features.")
    
    issues = run_data_validation(df_active)
    
    if len(issues) == 0:
        st.success("🎉 Data Validation completed with 0 errors/warnings! Your dataset is fully consistent.")
    else:
        # Display as styled items
        for issue in issues:
            b_color = "badge-red" if issue['severity'] == "danger" else ("badge-amber" if issue['severity'] == "warning" else "badge-blue")
            st.markdown(f"""
            <div class="card" style="padding: 1rem; margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="badge {b_color}">{issue['issue_type']}</span>
                    <small style="color:var(--text-muted);">Column: <code>{issue['column']}</code></small>
                </div>
                <p style="margin:0.4rem 0 0.2rem 0; font-size:0.9rem;">{issue['description']}</p>
                <small style="color:var(--text-muted);">Affected Rows count: <strong>{issue['affected_rows']}</strong></small>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# EDA CHARTS
# ==========================================
elif st.session_state.active_tab == "EDA Charts":
    df_active = require_dataset("Exploratory Data Analysis & Visual Chart Builder", "eda_charts")
    st.markdown("## Exploratory Data Analysis & Visual Chart Builder")
    
    c_builder1, c_builder2 = st.columns([1, 3])
    
    with c_builder1:
        st.markdown("### Chart Settings")
        chart_type = st.selectbox(
            "Chart Type:",
            ["Distribution Plot", "Scatter Plot", "Line Chart", "Bar Chart", "Pie Chart", "Treemap Plot", "Violin Plot", "Normal Q-Q Plot", "Correlation Matrix Heatmap"]
        )
        
        # Display settings depending on type
        x_c = None
        y_c = None
        color_c = None
        size_c = None
        agg_f = None
        
        if chart_type == "Correlation Matrix Heatmap":
            st.info("Heatmap calculates correlation across all numerical variables.")
        elif chart_type == "Distribution Plot":
            x_c = st.selectbox("Variable (X-axis):", df_active.select_dtypes(include=[np.number]).columns)
        elif chart_type == "Normal Q-Q Plot":
            x_c = st.selectbox("Variable (numeric):", df_active.select_dtypes(include=[np.number]).columns)
        elif chart_type in ["Scatter Plot", "Line Chart", "Violin Plot"]:
            x_c = st.selectbox("X-axis Column:", df_active.columns)
            y_c = st.selectbox("Y-axis Column (numeric):", df_active.select_dtypes(include=[np.number]).columns)
            color_c = st.selectbox("Color by (categorical) - Optional:", [None] + df_active.columns.tolist())
            if chart_type == "Scatter Plot":
                size_c = st.selectbox("Size by (numeric) - Optional:", [None] + df_active.select_dtypes(include=[np.number]).columns.tolist())
        elif chart_type == "Bar Chart":
            x_c = st.selectbox("X-axis Column:", df_active.columns)
            y_c = st.selectbox("Y-axis Column (numeric) - Optional:", [None] + df_active.select_dtypes(include=[np.number]).columns.tolist())
            color_c = st.selectbox("Color by (categorical) - Optional:", [None] + df_active.columns.tolist())
            if y_c:
                agg_f = st.selectbox("Aggregation Method:", ["sum", "mean", "median", "min", "max", "count"])
        elif chart_type == "Pie Chart":
            x_c = st.selectbox("Names Column (categories):", df_active.columns)
            y_c = st.selectbox("Values Column (numeric) - Optional:", [None] + df_active.select_dtypes(include=[np.number]).columns.tolist())
        elif chart_type == "Treemap Plot":
            path_c = st.multiselect("Path Hierarchy Columns:", df_active.columns)
            y_c = st.selectbox("Values Column (numeric) - Optional:", [None] + df_active.select_dtypes(include=[np.number]).columns.tolist(), key="treemap_val")

    with c_builder2:
        st.markdown("### Chart View")
        fig = None
        try:
            if chart_type == "Correlation Matrix Heatmap":
                fig = get_correlation_matrix_fig(df_active, IS_DARK)
            elif chart_type == "Distribution Plot":
                fig = get_distribution_fig(df_active, x_c, IS_DARK)
            elif chart_type == "Normal Q-Q Plot":
                fig = get_qq_plot_fig(df_active, x_c, IS_DARK)
            elif chart_type == "Scatter Plot":
                fig = get_scatter_fig(df_active, x_c, y_c, color_c, size_c, IS_DARK)
            elif chart_type == "Line Chart":
                fig = get_line_fig(df_active, x_c, y_c, color_c, IS_DARK)
            elif chart_type == "Violin Plot":
                fig = get_violin_fig(df_active, x_c, y_c, color_c, IS_DARK)
            elif chart_type == "Bar Chart":
                fig = get_bar_fig(df_active, x_c, y_c, color_col=color_c, agg_func=agg_f if agg_f else "count", is_dark=IS_DARK)
            elif chart_type == "Pie Chart":
                fig = get_pie_fig(df_active, x_c, y_c, IS_DARK)
            elif chart_type == "Treemap Plot":
                if not path_c:
                    st.info("Please select path hierarchy columns in Settings.")
                else:
                    fig = get_treemap_fig(df_active, path_c, y_c, IS_DARK)
                    
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart Export options
                st.markdown("#### Export Chart")
                ec1, ec2 = st.columns(2)
                with ec1:
                    # fig.write_html writes a str — use StringIO then encode to bytes
                    html_buf = io.StringIO()
                    fig.write_html(html_buf, full_html=True, include_plotlyjs='cdn')
                    st.download_button(
                        label="📥 Download Chart as HTML",
                        data=html_buf.getvalue().encode('utf-8'),
                        file_name=f"CleanMessy_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                with ec2:
                    # fig.to_json() returns a str directly — encode for download
                    json_str = fig.to_json()
                    st.download_button(
                        label="📥 Download Chart Config (JSON)",
                        data=json_str.encode('utf-8'),
                        file_name=f"CleanMessy_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error drawing chart: {str(e)}")

# ==========================================
# REPORT CENTER
# ==========================================
elif st.session_state.active_tab == "Report Center":
    df_active = require_dataset("Automated Data Profiling Reports", "report_center")
    st.markdown("## Automated Data Profiling Reports")
    
    st.markdown("### Generate Profiling Reports")
    st.info("Choose a reporting engine below. CleanMessy generates fully responsive, standalone HTML summary dashboards representing your dataset's metadata, correlations, validation errors, and column metrics.")
    
    col_rep1, col_rep2, col_rep3 = st.columns(3)
    
    with col_rep1:
        st.markdown("#### YData-Profiling Report")
        if not YDATA_AVAILABLE:
            st.warning("⚠️ YData-Profiling is not installed or supported in this Python environment.")
        else:
            if st.button("Generate YData Report", key="gen_ydata"):
                with st.spinner("Compiling profile report..."):
                    html_rep = generate_ydata_report(df_active)
                    if "Error" in html_rep:
                        st.error(html_rep)
                    else:
                        st.session_state.current_report_html = html_rep
                        st.session_state.report_name = "YData_Profiling_Report"
                        st.success("Report compiled successfully!")
                        
    with col_rep2:
        st.markdown("#### Sweetviz Comparison Report")
        if not SWEETVIZ_AVAILABLE:
            st.warning("⚠️ Sweetviz is not installed or supported in this Python environment.")
        else:
            if st.button("Generate Sweetviz Report", key="gen_sweetviz"):
                with st.spinner("Compiling Sweetviz dashboard..."):
                    html_rep = generate_sweetviz_report(df_active)
                    if "Error" in html_rep:
                        st.error(html_rep)
                    else:
                        st.session_state.current_report_html = html_rep
                        st.session_state.report_name = "Sweetviz_Profiling_Report"
                        st.success("Report compiled successfully!")
                        
    with col_rep3:
        st.markdown("#### CleanMessy Custom Report")
        st.success("⚡ Light-weight, high-performance HTML dashboard matching your active app theme. Works on all environments.")
        if st.button("Generate Custom Report", key="gen_custom"):
            with st.spinner("Compiling custom dashboard..."):
                html_rep = generate_custom_cleanmessy_report(df_active)
                st.session_state.current_report_html = html_rep
                st.session_state.report_name = "CleanMessy_Quality_Report"
                st.success("Report compiled successfully!")

    st.markdown("---")
    
    # Display and Download Report if generated
    if "current_report_html" in st.session_state and st.session_state.current_report_html is not None:
        rep_name = st.session_state.report_name
        st.markdown(f"### Active Report: `{rep_name}`")
        
        # Download button
        st.download_button(
            label="📥 Download HTML Report",
            data=st.session_state.current_report_html.encode('utf-8'),
            file_name=f"{rep_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
        
        # Option to embed view in Streamlit iframe
        st.markdown("#### Report Sandbox Preview")
        st.components.v1.html(st.session_state.current_report_html, height=800, scrolling=True)

# ==========================================
# EXPORT / DOWNLOAD
# ==========================================
elif st.session_state.active_tab == "Export / Download":
    df_active = require_dataset("Download Clean Dataset", "export_download")
    st.markdown("## Download Clean Dataset")
    
    st.markdown("### Summary Statistics")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"**Original Dataset File:** `{st.session_state.file_name}`")
        st.markdown(f"**Initial shape:** {st.session_state.dataset_history.history[0].shape}")
        st.markdown(f"**Cleaned dataset shape:** {df_active.shape}")
    with col_m2:
        # Operations count
        hist = st.session_state.dataset_history
        st.markdown(f"**Applied Operations Count:** {len(hist.history) - 1}")
        
    st.markdown("---")
    
    st.markdown("### Export Configurations")
    selected_format = st.selectbox("Export File Format:", ["CSV", "Excel", "JSON", "Parquet", "Pickle"], index=["CSV", "Excel", "JSON", "Parquet", "Pickle"].index(st.session_state.default_format))
    
    filename_input = st.text_input("Filename:", value=f"clean_{os.path.splitext(st.session_state.file_name)[0] if st.session_state.file_name else 'dataset'}")
    
    ext_map = {"CSV": ".csv", "Excel": ".xlsx", "JSON": ".json", "Parquet": ".parquet", "Pickle": ".pkl"}
    full_filename = f"{filename_input}{ext_map[selected_format]}"
    
    if st.button("Compile & Export Dataset", key="export_comp_btn", use_container_width=True):
        try:
            with st.spinner("Compiling binary export data..."):
                bytes_data, mime_type = export_dataset(df_active, selected_format)
                
                st.download_button(
                    label=f"📥 Download {full_filename}",
                    data=bytes_data,
                    file_name=full_filename,
                    mime=mime_type,
                    use_container_width=True
                )
                st.success("Dataset is compiled and ready for download!")
        except Exception as e:
            st.error(f"Failed to export: {str(e)}")
            
    # Quick processing logs preview
    st.markdown("---")
    st.markdown("### Operation Audit Logs")
    log_text = "\n".join(st.session_state.dataset_history.logs)
    st.text_area("Audit Trail:", value=log_text, height=200, disabled=True)
