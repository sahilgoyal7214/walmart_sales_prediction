##Experiment 13a

import streamlit as st
from numpy import array
from sklearn.ensemble import RandomForestRegressor
import pickle as pk

def load_rfr_model(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pk.load(file)
        return model
    except Exception as e:
        st.write("Error loading the model:", str(e))
        return None

st.markdown("----")

st.title("Walmart Sales Prediction")

model_path = r"rfr_model.pkl"
rfr_model = load_rfr_model(model_path)

if rfr_model is not None:
    store = st.number_input("Enter store number: ")
    Fuel_Price = st.number_input("Enter fuel price: ")
    CPI = st.number_input("Enter CPI: ")
    Unemployment = st.number_input("Enter unemployment: ")
    Day = st.number_input("Enter day: ")
    Month = st.number_input("Enter month: ")
    Year = st.number_input("Enter year: ")
    holiday_flag = st.number_input("Is it a holiday?(0 for no, 1 for yes): ")

    if st.button("Predict"):
        input_data = array([[store, Fuel_Price, CPI, Unemployment, Day, Month, Year, holiday_flag]])
        prediction = rfr_model.predict(input_data)

        st.write(f"Predicted Sales for that day: {prediction[0]:.2f}")
else:
    st.write("Model failed to load. Please check the model file path.")