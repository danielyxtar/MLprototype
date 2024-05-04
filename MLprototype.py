import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title

show_pages(
    [
        Page("MLprototype.py", "Home"),
        Page("pages/testJ.py", "Prediction"),
    ]
)

st.title('Welcome to Rhine Pharmeceuticals')

st.markdown(
    """
    <style>
    .img-container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use a div element to wrap the image and apply the CSS
st.markdown(
    f'<div class="img-container"><img src="logo.png" width="200"></div>', 
    unsafe_allow_html=True
)


