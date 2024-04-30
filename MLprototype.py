import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)
    base64_pdf = base64.b64encode(bytes_data.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'   
    st.markdown(pdf_display, unsafe_allow_html=True)

    #Faire une fonction avec le display
with open(bytes_data,"rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'   
    st.markdown(pdf_display, unsafe_allow_html=True)

    st.write('Statistics')
