import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the diabetes prediction model
model = joblib.load('RF.pkl')

# Define the feature names for the new dataset
feature_names = [
    "Age", "BMI", "SBP", "DBP", "FPG", "Chol", "Tri", "HDL", "LDL",
    "ALT", "BUN", "CCR", "FFPG", "smoking", "drinking"
]

# Streamlit user interface
st.title("Diabetes Prediction")

# Input features
age = st.number_input("Age:", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=24.0)
sbp = st.number_input("SBP:", min_value=50, max_value=200, value=120)
dbp = st.number_input("DBP:", min_value=30, max_value=120, value=80)
fpg = st.number_input("FPG:", min_value=0.0, max_value=20.0, value=5.0)
chol = st.number_input("Chol:", min_value=0.0, max_value=10.0, value=4.5)
tri = st.number_input("Tri:", min_value=0.0, max_value=10.0, value=1.0)
hdl = st.number_input("HDL:", min_value=0.0, max_value=10.0, value=1.5)
ldl = st.number_input("LDL:", min_value=0.0, max_value=10.0, value=3.0)
alt = st.number_input("ALT:", min_value=0.0, max_value=100.0, value=30.0)
bun = st.number_input("BUN:", min_value=0.0, max_value=100.0, value=20.0)
ccr = st.number_input("CCR:", min_value=0.0, max_value=100.0, value=50.0)
ffpg = st.number_input("FFPG:", min_value=0.0, max_value=20.0, value=5.0)
smoking = st.selectbox("smoking:", options=[0, 1])
drinking = st.selectbox("drinking:", options=[0, 1])

# Collect input values into a list
feature_values = [age, bmi, sbp, dbp, fpg, chol, tri, hdl, ldl, alt, bun, ccr, ffpg, smoking, drinking]

# Convert the input feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Make prediction using the model
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Display the prediction results
    st.write(f"**Predicted Class (0=No Diabetes, 1=Diabetes):** {predicted_class}")
    st.write(f"**Predicted Probability:** {predicted_proba}")

    # Generate advice based on the prediction
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 0:
        advice = (
            f"Based on our model's prediction, you are unlikely to have diabetes."
            f" The probability of you not having diabetes is {probability:.1f}%. "
            "It is recommended to maintain a healthy lifestyle and continue reducing risk factors."
        )
    else:
        advice = (
            f"Based on our model's prediction, you may have diabetes."
            f" The probability of you having diabetes is {probability:.1f}%. "
            "It is recommended to consult a doctor for further diagnosis and treatment."
        )

    st.write(advice)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer.shap_values(features_df)

    # Extract SHAP values for the predicted class
    shap_values_for_instance = shap_values_Explanation[predicted_class][0]  # Select the first instance

    # Create SHAP explanation object for the predicted class
    shap_values_explanation = shap.Explanation(
        values=shap_values_for_instance,
        base_values=explainer.expected_value[predicted_class],
        data=features_df,
        feature_names=feature_names
    )

    # Directly show SHAP values
    st.write(f"SHAP values: {shap_values_for_instance}")

    # Display SHAP waterfall plot
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_explanation, show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")


