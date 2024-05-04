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

col1, col2, col3 = st.columns([1,2,1]) 

# Use the middle column to display the image
with col2:
    st.image('logo.png', width=180)