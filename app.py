import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# 1. PAGE SETUP
st.set_page_config(page_title="Yes Bank Predictor", layout="centered")

st.title("📈 Yes Bank Stock Price Prediction")
st.write("Adjust the sliders below to predict the monthly closing price.")

# 2. LOAD ASSETS (Model & Scaler)
# This looks into your 'My_Project' folder on GitHub
folder = 'My_Projects'
model_path = os.path.join(folder, 'best_yesbank_model.joblib')
scaler_path = os.path.join(folder, 'scaler.joblib')

@st.cache_resource
def load_model_files():
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Could not find model files in '{folder}'. Error: {e}")
        return None, None

model, scaler = load_model_files()

# 3. USER INPUT SECTION (The Sliders)
if model is not None:
    with st.form("input_form"):
        st.subheader("Market Indicators")
        
        # Sliders: (Label, Min Value, Max Value, Default Value)
        open_p = st.slider("Opening Price (INR)", 0.0, 500.0, 150.0)
        high_p = st.slider("Monthly High (INR)", 0.0, 500.0, 160.0)
        low_p = st.slider("Monthly Low (INR)", 0.0, 500.0, 140.0)
        prev_close = st.slider("Previous Month Close (INR)", 0.0, 500.0, 145.0)
        
        predict_btn = st.form_submit_button("Generate Prediction")

    # 4. PREDICTION LOGIC
    if predict_btn:
        # Log transformation (to match training data)
        inputs_log = np.log1p([open_p, high_p, low_p, prev_close])
        
        # Feature Engineering (6 features for the scaler)
        ohlc_avg = np.mean(inputs_log)
        spread = inputs_log[1] - inputs_log[2]
        
        # Combine into the order expected by Scaler
        full_features = [[inputs_log[0], inputs_log[1], inputs_log[2], ohlc_avg, spread, inputs_log[3]]]
        
        # Scale and then select the 4 features the model uses
        scaled_data = scaler.transform(full_features)
        model_ready_data = scaled_data[:, [0, 1, 2, 5]]
        
        # Predict and convert back from Log scale
        prediction_log = model.predict(model_ready_data)
        final_price = np.expm1(prediction_log)
        
        # Show Results
        st.balloons()
        st.success(f"### Predicted Closing Price: ₹{final_price[0]:.2f}")
        
else:
    st.info("Waiting for model files to load from GitHub...")

# 5. FOOTER
st.markdown("---")
st.caption("Note: This is an ML project for educational purposes.")
