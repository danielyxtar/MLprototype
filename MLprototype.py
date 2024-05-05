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

#st.write("Start Analysing the severity of patients with HCV")

st.markdown("""
<p style='font-size: 20px; font-family: Helvetica;'>Start Analysing the severity of patients with HCV by going to the prediction tab</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style='font-size: 18px; font-family: Arial;'>Need more info on Hepatitis C virus? Head over to <a href='https://www.who.int/news-room/fact-sheets/detail/hepatitis-c' target='_blank'> for more info</a>.</p>
""", unsafe_allow_html=True)

