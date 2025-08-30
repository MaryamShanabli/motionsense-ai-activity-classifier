# Streamlit app to predict human activity using the trained Gradient Boosting model

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# Load Scaler and Model
# ----------------------------
scaler = joblib.load("models/robust_scaler.joblib")
best_gb = joblib.load("models/best_gradient_boosting_model.joblib")

# Load feature names
feature_names = pd.read_csv("data/selected_features.csv", header=None).squeeze("columns").tolist()

# App class names
app_class_names = {0: "SEDENTARY", 1: "STANDING", 2: "ACTIVE"}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("MotionSense Activity Prediction")
st.write("Predict human activity from sensor data.")

# Upload CSV with features
uploaded_file = st.file_uploader("Upload a CSV file with features", type="csv")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    
    # Check that all required features exist
    missing_features = [f for f in feature_names if f not in df_input.columns]
    if missing_features:
        st.error(f"Missing features in uploaded CSV: {missing_features}")
    else:
        # Reorder columns to match training data
        df_input = df_input[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(df_input.values)
        
        # Predict
        y_pred = best_gb.predict(X_scaled)
        y_pred_class = [app_class_names[i] for i in y_pred]
        
        # Display predictions
        df_input["PredictedActivity"] = y_pred_class
        st.write("Predictions:")
        st.dataframe(df_input)