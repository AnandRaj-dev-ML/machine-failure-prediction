import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load model
model_path = hf_hub_download(
    repo_id="Rajanan/machine_failure_model",
    filename="best_machine_failure_model_v1.joblib",
    repo_type="model"
)
model = joblib.load(model_path)

st.title("Machine Failure Prediction App_w3")

# User input
Type = st.selectbox("Machine Type", ["H", "L", "M"])
air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 298.0)
process_temp = st.number_input("Process Temperature (K)", 250.0, 500.0, 324.0)
rot_speed = st.number_input("Rotational Speed (RPM)", 0, 3000, 1400)
torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
tool_wear = st.number_input("Tool Wear (min)", 0, 300, 10)

# ✅ MATCH TRAINING ENCODING
type_mapping = {"H": 0, "L": 1, "M": 2}

input_data = pd.DataFrame([{
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": rot_speed,
    "Torque": torque,
    "Tool wear": tool_wear,
    "Type": type_mapping[Type]
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    st.success(
        "Machine Failure" if prediction == 1 else "No Failure"
    )
