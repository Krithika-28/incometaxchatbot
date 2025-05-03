import streamlit as st
import pickle
import numpy as np

# Load the retrained model
with open("D:/Project/myenv/tax_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Income Tax Predictor", page_icon="📈")
st.title("📊 Predict Your Income Tax (Using Trained ML Model)")

salary = st.number_input("💰 Enter Your Annual Salary (in ₹)", min_value=0, step=1000)

if st.button("Predict Tax"):
    if salary == 0:
        st.warning("⚠ Please enter a valid salary.")
    else:
        input_data = np.array([[salary]])
        predicted_tax = model.predict(input_data)[0]
        st.success(f"💡 Estimated Total Tax Payable: ₹{predicted_tax:,.2f}")