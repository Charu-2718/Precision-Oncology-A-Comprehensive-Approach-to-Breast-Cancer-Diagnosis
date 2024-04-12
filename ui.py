
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load your dataset
df = pd.read_csv('data.csv')
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Label Encoding for the target variable 'diagnosis'
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Feature Scaling
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df.drop('diagnosis', axis=1)), columns=df.drop('diagnosis', axis=1).columns)

# Initialize Logistic Regression model
lr = LogisticRegression()
lr.fit(scaled_df, df['diagnosis'])

# Function to predict diagnosis based on input values
def predict_diagnosis(inputs):
    # Ensure input values are in correct format
    try:
        inputs = [float(val) for val in inputs]
    except ValueError:
        return "Please enter valid numeric values"

    # Make prediction
    prediction = lr.predict([inputs])
    if prediction == 1:
        st.write('<span style="color: red;">The diagnosis is Malignant</span>', unsafe_allow_html=True)
    else:
        st.write('<span style="color: green;">The diagnosis is Benign</span>', unsafe_allow_html=True)

# Streamlit app
st.title("Breast Cancer Diagnosis Prediction")

# Create input fields for independent variables
st.sidebar.header("Enter the values of the following details:")
inputs = []
for column in df.drop('diagnosis', axis=1).columns:
    input_val = st.sidebar.number_input(column)
    inputs.append(input_val)

# Predict button
if st.sidebar.button("Predict Diagnosis"):
    prediction_result = predict_diagnosis(inputs)

