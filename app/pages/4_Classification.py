from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Classification",
    page_icon="",
    layout="wide",
)

st.title("Classification")
st.write(
    "This section will show how embeddings can become features for predicting "
    "the labelled categories attached to each article."
)

st.info(
    "Planned next step: train a simple classifier on top of stored embeddings and "
    "compare results across the OpenAI and Hugging Face vectors."
)
