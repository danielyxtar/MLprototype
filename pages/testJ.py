import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Load data
data = pd.read_csv('HCV-Egy-Data.csv')
X = data.drop(columns=['Baselinehistological staging'])
y = data['Baselinehistological staging']

# Handle outliers
rna12_mean = data['RNA 12'].mean()
data.loc[:, 'RNA 12'] = data['RNA 12'].fillna(rna12_mean)

# Calculate IQR
Q1 = data['RNA 12'].quantile(0.25)
Q3 = data['RNA 12'].quantile(0.75)
IQR = Q3 - Q1
# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Identify outliers
outliers = (data['RNA 12'] < lower_bound) | (data['RNA 12'] > upper_bound)
# Replace outliers with mean
data.loc[outliers, 'RNA 12'] = rna12_mean

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

knn_improved = KNeighborsClassifier(n_neighbors=3)  
nb_improved = GaussianNB(var_smoothing=1.0)
mlp_improved = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.01, max_iter=2000, random_state=20)

base_models = [
    ('knn', knn_improved),
    ('nb', nb_improved),
    ('mlp', mlp_improved)
]

meta_model = LogisticRegression()

# Stacking classifier configuration
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)

# Training the model
stacking_model.fit(X_train, y_train)
dump(stacking_model, 'best_model.pkl')

# Load Model
model = load('best_model.pkl')

def preprocess_data(df):
    # Perform any necessary preprocessing steps here
    # For example, you might handle missing values or scale numerical features
    # Make sure the preprocessing steps are the same as what was done during training
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

#Input for new data
col1, col2, col3 = st.columns([4, 4, 2])  # Adjust column widths as needed
with col2:
    st.image('logo.png', width=180)

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
    

    # Make predictions
    new_data = {}
    gender_mapping = {'Male': 1, 'Female': 2}
    symp_mapping = {'Absent': 1, 'Present': 2}
    new_data = {
        'Age': num_age,
        'Gender': gender_mapping[select_gender],
        'BMI': num_bmi,
        'Fever': symp_mapping[select_fever],
        'Nausea/Vomting': symp_mapping[select_nausea],
        'Headache': symp_mapping[select_headache],
        'Diarrhea': symp_mapping[select_diarrhea],
        'Fatigue & generalized bone ache': symp_mapping[select_fatigue],
        'Jaundice': symp_mapping[select_jaundice],
        'Epigastric pain': symp_mapping[select_epigastric],
        'WBC': num_wbc,
        'RBC': num_rbc,
        'HGB': num_hgb,
        'Plat': num_plat,
        'AST 1': num_ast1,
        'ALT 1': num_alt1,
        'ALT4': num_alt4,
        'ALT 12': num_alt12,
        'ALT 24': num_alt24,
        'ALT 36': num_alt36,
        'ALT 48': num_alt48,
        'ALT after 24 w': num_altafter24,
        'RNA Base': num_rnabase,
        'RNA 4': num_rna4,
        'RNA 12': num_rna12,
        'RNA EOT': num_rnaeot,
        'RNA EF': num_rnaef,
        'Baseline histological Grading': num_bhg
    }

    if submit_btn:
        # Create a DataFrame with the new data and specify an index
        new_df = pd.DataFrame([new_data], index=[0])

        # Make predictions
        predictions = model.predict(new_df)

        # Display the prediction
        st.write("Predicted Baseline Histological Staging for " + txt_name + " is :", predictions)
    
