import streamlit as st
from pathlib import Path
from pages import step_evolution, weights_dashboard, singular_values, cross_model, animations
from dashboard_utils import detect_experiments, get_data_path, is_HF_environment

# ---- HIDE DEFAULT MULTIPAGE MENU ----
hide_default_format = """
    <style>
    /* Hide "Pages" header */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none;
    }
    /* Hide the whole page-list container */
    section[data-testid="stSidebar"] ul {
        display: none;
    }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.title("Transformer Weight Analysis")

if not is_HF_environment():
    root = Path(get_data_path())
    experiments = detect_experiments(root)
    if experiments:
        selected = st.sidebar.selectbox("Experiment", experiments)
        st.session_state["experiment"] = selected
    else:
        st.sidebar.warning(f"No experiments found in `{root}`.")

pages = {
    "Weights Dashboard": weights_dashboard,
    "Singular Values": singular_values,
    "Cross-Model Comparison": cross_model,
    "Step Evolution": step_evolution,
    "Animations": animations,
}

page_name = st.sidebar.radio("Navigation", list(pages.keys()))

# Render selected page
pages[page_name].render()
