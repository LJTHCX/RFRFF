import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the diabetes prediction model
model = joblib.load('RF.pkl')

# Define the feature ranges for the new dataset
feature_ranges = {
    "Age": {"type": "numerical", "min": 18, "max": 100, "default": 30},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0},
    "SBP": {"type": "numerical", "min": 50, "max": 200, "default": 120},
    "DBP": {"type": "numerical", "min": 30, "max": 120, "default": 80},
    "FPG": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 5.0},
    "Chol": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 4.5},
    "Tri": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.0},
    "HDL": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.5},
    "LDL": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 3.0},
    "ALT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 30.0},
    "BUN": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 20.0},
    "CCR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 50.0},
    "FFPG": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 5.0},
    "smoking": {"type": "categorical", "options": [0, 1]},
    "drinking": {"type": "categorical", "options": [0, 1]},
}

# Streamlit user interface
st.title("Diabetes Prediction")

# Dynamic input generation
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# Convert the input feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

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

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer.shap_values(features_df)

    # Display SHAP plot for predicted class
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_Explanation[1][0], show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
