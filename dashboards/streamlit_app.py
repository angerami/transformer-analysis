import streamlit as st

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
from pages import step_evolution, weights_dashboard

st.sidebar.title("Transformer Weight Analysis")

pages = {
    "Weights Dashboard": weights_dashboard,
    "Step Evolution": step_evolution,
}

page_name = st.sidebar.radio("Navigation", list(pages.keys()))

# Render selected page
pages[page_name].render()