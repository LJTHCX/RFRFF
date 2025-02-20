import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
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

# Streamlit 界面
st.title("Diabetes Prediction Model with SHAP Visualization")

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

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 获取预测类别的 SHAP 值
    shap_values_class = shap_values[predicted_class]  # 获取预测类别的 SHAP 值

    # 生成 SHAP 力图
    shap_fig = shap.force_plot(
        explainer.expected_value[predicted_class],
        shap_values_class[0],  # 获取第一个样本的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 力图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

    # 根据预测类别绘制 SHAP 水波图
    plt.figure(figsize=(10, 5), dpi=1200)
    if predicted_class == 0:
        shap.plots.waterfall(shap_values_class[0], show=False, max_display=15)  # 类别0的图
        plt.savefig("shap_plot_class_0.png", bbox_inches='tight', dpi=1200)
        st.image("shap_plot_class_0.png")
    elif predicted_class == 1:
        shap.plots.waterfall(shap_values_class[0], show=False, max_display=15)  # 类别1的图
        plt.savefig("shap_plot_class_1.png", bbox_inches='tight', dpi=1200)
        st.image("shap_plot_class_1.png")
