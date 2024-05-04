import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title

show_pages(
    [
        Page("MLprototype.py", "Home"),
        Page("other_pages/testJ.py", "Prediction"),
    ]
)

st.set_page_config(
    page_title="Home"
)

st.title('Welcome to Rhine Pharmeceuticals')

st.image('logo.png', width=180)


