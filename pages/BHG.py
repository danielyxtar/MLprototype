import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
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

# Additional logic as needed
