import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PrototypeJ"
)

st.title('Welcome to Rhine Pharmeceuticals')

st.write('CSV File Upload')

uploaded_file = st.file_uploader("Drag and Drop or Click to Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write(df)
else:
    st.write("Please upload a CSV file.")


