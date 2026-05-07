from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Clustering",
    page_icon="",
    layout="wide",
)

st.title("Clustering")
st.write(
    "This section will group articles whose embeddings are close together and "
    "help users inspect the themes that appear in each cluster."
)

st.info(
    "Planned next step: run lightweight clustering on the local embeddings and "
    "show cluster sizes, labels, and representative articles."
)
