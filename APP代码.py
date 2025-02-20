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
    # Model prediction
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Extract the predicted class probability
    probability = predicted_proba[predicted_class] * 100

    # Display prediction result using Matplotlib (rendered text)
    text = f"Based on feature values, predicted possibility of diabetes is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    # Check the number of classes in the output and get the correct index
    class_index = predicted_class  # The predicted class index (0 or 1)

    # If only one class is predicted, we should use that class's SHAP values
    if isinstance(shap_values, list):  # For binary classification (2 classes)
        shap_fig = shap.force_plot(
            explainer.expected_value[class_index],
            shap_values[class_index],
            features_df,
            feature_names=feature_names,
            matplotlib=False  # Set to False for multiple samples
        )
    else:
        # For multi-class or other cases
        shap_fig = shap.force_plot(
            explainer.expected_value[0],  # Use the first class's expected value
            shap_values,
            features_df,
            feature_names=feature_names,
            matplotlib=False  # Set to False for multiple samples
        )

    # Save the SHAP force plot as an HTML file
    shap_fig_html = "shap_force_plot.html"
    shap.save_html(shap_fig_html, shap_fig)

    # Display the SHAP force plot in the Streamlit app
    st.markdown(f'<iframe src="{shap_fig_html}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)
