import streamlit as st
from pages import step_evolution, weights_dashboard, singular_values, cross_model, animations

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
import os
st.sidebar.write(f"SPACE_ID: {os.getenv('SPACE_ID', 'NOT SET')}")

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
