import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Yes Bank Price Predictor", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Yes Bank Stock Price Prediction")
st.write("This app uses a Tuned ElasticNet Regression model to predict monthly closing prices.")

# --- LOAD MODEL & SCALER ---
# Ensure the folder name matches your GitHub repository exactly
folder = 'My_Projects'
model_path = os.path.join(folder, 'best_yesbank_model.joblib')
scaler_path = os.path.join(folder, 'scaler.joblib')

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Check if 'My_Project' folder exists on GitHub and contains the .joblib files.")
        return None, None

model, scaler = load_assets()

# --- USER INPUT SECTION ---
if model and scaler:
    with st.form("prediction_form"):
        st.subheader("Enter Monthly Stock Data")
        
        col1, col2 = st.columns(2)
        with col1:
            open_p = st.number_input("Opening Price", min_value=0.0, value=180.5)
            low_p = st.number_input("Lowest Price of Month", min_value=0.0, value=175.0)
        
        with col2:
            high_p = st.number_input("Highest Price of Month", min_value=0.0, value=185.2)
            prev_close = st.number_input("Previous Month Close", min_value=0.0, value=178.0)
        
        # We need a dummy OHLC_Avg and Spread for the scaler
        # even though the model will only use 4 of the 6 features.
        submit_button = st.form_submit_button("Predict Closing Price")

    if submit_button:
        # 1. Log Transform the inputs (Matching Training)
        # Using log1p to stay consistent with np.log1p
        inputs_log = np.log1p([open_p, high_p, low_p, prev_close])
        
        # 2. Reconstruct the 6 features for the Scaler
        # Order: ['Open_log', 'High_log', 'Low_log', 'OHLC_Avg', 'Spread', 'Prev_Close']
        ohlc_avg = np.mean(inputs_log) # Simple average of logs
        spread = inputs_log[1] - inputs_log[2] # High_log - Low_log
        
        full_features = [[inputs_log[0], inputs_log[1], inputs_log[2], ohlc_avg, spread, inputs_log[3]]]
        
        # 3. Apply the Scaler
        scaled_data = scaler.transform(full_features)
        
        # 4. Slice to select the 4 features used by the ElasticNet Model
        # Indices: 0(Open), 1(High), 2(Low), 5(Prev_Close)
        model_input = scaled_data[:, [0, 1, 2, 5]]
        
        # 5. Predict and Inverse Log
        prediction_log = model.predict(model_input)
        final_price = np.expm1(prediction_log)
        
        # --- OUTPUT DISPLAY ---
        st.divider()
        st.balloons()
        st.subheader(f"Target Prediction: ₹{final_price[0]:.2f}")
        st.write("Note: This prediction is based on monthly historical trends.")

else:
    st.warning("Please resolve the file loading issue to proceed.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for Yes Bank Stock Analysis Project | ML Regression")
