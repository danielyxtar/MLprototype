import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title
from PIL import Image

show_pages(
    [
        Page("MLprototype.py", "Home"),
        Page("pages/testJ.py", "Prediction"),
    ]
)

st.title('Welcome to Rhine Pharmeceuticals')

image = Image.open("logo.png")

# Add custom CSS to center any image content
st.markdown(
    """
    <style>
    .centered-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the image with Streamlit and apply CSS class for centering
st.image(image, width=180, use_column_width=False, output_format="PNG", classes="centered-img")

