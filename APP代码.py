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
smoking = st.selectbox("Smoking:", options=[0, 1])
drinking = st.selectbox("Drinking:", options=[0, 1])

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
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

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
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

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
