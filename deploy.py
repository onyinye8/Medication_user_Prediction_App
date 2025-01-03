# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
random_forest_model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')  # Load the scaler used during the training of the dataset

# Title and Description
st.title('Medication User Satisfaction Prediction')
st.write('''
This application predicts medication users' satisfaction based on the users' ratings.
''')

# Input fields for user data
st.header('Enter Input Data')
excellent_rating = st.number_input('Excellent Rating', min_value=60.0, max_value=100.0, step=0.1)
average_rating = st.number_input('Average Rating', min_value=40.0, max_value=59.0, step=0.1)
poor_rating = st.number_input('Poor Rating', min_value=0.0, max_value=39.0, step=0.1)

# Prediction logic
if st.button('Predict'):
    # Organize input data into a DataFrame
    input_data = pd.DataFrame({
        'excellent_rating': [excellent_rating],
        'average_rating': [average_rating],
        'poor_rating': [poor_rating]
    })

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Make prediction using the Random Forest model
    prediction = random_forest_model.predict(scaled_data)[0]

    # Display the prediction
    st.success(f'The predicted medication user satisfaction is: {prediction:.2f}')
