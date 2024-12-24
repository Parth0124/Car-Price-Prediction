import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('model.h5')

# Load the scaled data and scalers
with open('scaled_data.pkl', 'rb') as file:
    scaled_data = pickle.load(file)

x_scaler = scaled_data['x_scaler']
y_scaler = scaled_data['y_scaler']

# Streamlit app interface
st.title("Car Price Prediction")

# Input form for new data
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_salary = st.number_input("Annual Salary ($)", min_value=10000, max_value=100000000000, value=50000)
credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0, max_value=1000000, value=5000)
net_worth = st.number_input("Net Worth ($)", min_value=10000, max_value=100000000000, value=150000)

# Convert gender to numeric (0 for female, 1 for male)
gender_value = 1 if gender == "Male" else 0

# Prepare the input data for prediction
input_data = np.array([[gender_value, age, annual_salary, credit_card_debt, net_worth]])

# Scale the input data using the same scaler that was used during training
input_data_scaled = x_scaler.transform(input_data)

# Make the prediction using the model
if st.button("Predict Car Price"):
    # Predict the car price on the scaled input data
    predicted_scaled_price = model.predict(input_data_scaled)

    # Inverse transform the predicted value to the original scale
    predicted_price = y_scaler.inverse_transform(predicted_scaled_price)

    # Show the predicted car price
    st.write(f"Predicted Car Price: ${predicted_price[0][0]:,.2f}")

