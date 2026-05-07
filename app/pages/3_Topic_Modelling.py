from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Topic Modelling",
    page_icon="",
    layout="wide",
)

st.title("Topic Modelling")
st.write(
    "This section will use embedding neighborhoods and article labels to create "
    "topic-oriented views of the newswire sample."
)

st.info(
    "Planned next step: combine projected embeddings, labels, and representative "
    "article snippets into an explorable topic map."
)
