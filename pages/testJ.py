import streamlit as st

col1, col2, col3 = st.columns([4, 4, 2])  # Adjust column widths as needed
with col2:
    st.image('logo.png', width=180)

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

tab1, tab2 = st.tabs(["📈 Analysis Chart", "🗃 Prediction"])

tab1.subheader("Analysis Chart")

# Prediction Tab
tab2.subheader("HCV Prediction")
with tab2:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns([3, 3, 3])  # Adjust column widths as needed
    with col1:
        txt_name = st.text_input("Name : ", placeholder='Enter Name')
        num_age = st.number_input("Age : ", placeholder='Enter Age', min_value=1, max_value=120, step=1)
        select_gender = st.selectbox("Gender : ", placeholder='Select Gender', options=["Male", "Female"])
        num_bmi = st.number_input("BMI : ", placeholder='Enter BMI', step=1, min_value=0)
        select_fever = st.selectbox("Fever : ", placeholder='Select Fever', options=["Absent","Present"])
        select_nausea = st.selectbox("Nausea/Vomting : ", placeholder='Select Nausea/Vomting', options=["Absent","Present"])
        select_headache = st.selectbox("Headache : ", placeholder='Select Headache', options=["Absent","Present"])
        select_diarrhea = st.selectbox("Diarrhea : ", placeholder='Select Diarrhea', options=["Absent","Present"])
        select_fatigue = st.selectbox("Fatigue & Generalized Bone Ache : ", placeholder='Select Fatigue & Generalized Bone Ache', options=["Absent","Present"])
        select_jaundice = st.selectbox("Jaundice : ", placeholder='Select Jaundice', options=["Absent","Present"])
   
   
    with col2:
        select_epigastric = st.selectbox("Epigastric Pain : ", placeholder='Select Epigastric Pain', options=["Absent","Present"])
        num_wbc = st.number_input("White Blood Cell : ", placeholder='Enter White Blood Cell', min_value=1, step=1)
        num_rbc = st.number_input("Red Blood Cell : ", placeholder='Enter Red Blood Cell', min_value=1, step=1)
        num_hgb = st.number_input("Hemoglobin : ", placeholder='Enter Hemoglobin', min_value=1, step=1)
        num_plat = st.number_input("Platelets : ", placeholder='Enter Platelets', min_value=1, step=1)
        num_ast1 = st.number_input("AST 1 : ", placeholder='Enter Aspartate Transaminase Ratio', min_value=1, step=1)
        num_alt1 = st.number_input("ALT 1 : ", placeholder='Enter Alanine Transaminase Ratio 1 Week', min_value=1, step=1)
        num_alt4 = st.number_input("ALT 4 : ", placeholder='Enter Alanine Transaminase Ratio 4 Week', min_value=1, step=1)
        num_alt12 = st.number_input("ALT 12 : ", placeholder='Enter Alanine Transaminase Ratio 12 Week', min_value=1, step=1)
        num_alt24 = st.number_input("ALT 24 : ", placeholder='Enter Alanine Transaminase Ratio 24 Week', min_value=1, step=1)
        col21, col22, col23 = st.columns([2, 4, 2])
        with col22:
            submit_btn = st.form_submit_button("Predict")
    

    with col3:
        num_alt36 = st.number_input("ALT 36 : ", placeholder='Enter Alanine Transaminase Ratio 36 Week', min_value=1, step=1)
        num_alt48 = st.number_input("ALT 48 : ", placeholder='Enter Alanine Transaminase Ratio 48 Week', min_value=1, step=1)
        num_altafter24 = st.number_input("ALT After 24 Weeks : ", placeholder='Enter ALT after 24 W	Alanine Transaminase Ratio 24 Weeks', min_value=1, step=1)
        num_rnabase = st.number_input("RNA Base : ", placeholder='Enter RNA Base', min_value=1, step=1)
        num_rna4 = st.number_input("RNA 4 : ", placeholder='Enter RNA 4', min_value=1, step=1)
        num_rna12 = st.number_input("RNA 12 : ", placeholder='Enter RNA 12', min_value=1, step=1)
        num_rnaeot = st.number_input("RNA EOT : ", placeholder='Enter RNA EOT', min_value=1, step=1)
        num_rnaef = st.number_input("RNA EF : ", placeholder='Enter RNA EF', min_value=1, step=1)
        num_bhg = st.number_input("Baseline Histological Grading : ", placeholder='Enter Baseline Histological Grading', min_value=1, step=1)
    

    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

@st.cache
def load_data():
    # Load your dataset here
    return pd.read_csv("HCV-Egy-Data.csv")

data = load_data()

# Split the dataset into features and target
X = data.drop(columns=["Baselinehistological staging"])  # Adjust column name if different
y = data["Baselinehistological staging"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit layout
st.image('logo.png', width=180)  # Adjust as needed

# Sidebar for contact preferences and shipping method
with st.sidebar:
    st.selectbox("How would you like to be contacted?", ["Email", "Home phone", "Mobile phone"])
    st.radio("Choose a shipping method", ["Standard (5-15 days)", "Express (2-5 days)"])

# Main tabs for Analysis and Prediction
tab1, tab2 = st.tabs(["📈 Analysis Chart", "🗃 Prediction"])

with tab1:
    tab1.subheader("Analysis Chart")
    # Include your analysis plot code here (e.g., line chart, histogram)

with tab2:
    tab2.subheader("HCV Prediction")
    with st.form("prediction_form"):
        # Dynamically create input fields based on data columns
        input_data = {}
        for column in X.columns:
            input_data[column] = st.number_input(f"{column} :", key=f"input_{column}")  # Unique key for each input

        submitted = st.form_submit_button("Predict")
        if submitted:
            # Prepare input data for prediction
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write("Predicted Baseline Histological Staging:", prediction[0])

