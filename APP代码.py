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

# 初始化按钮点击状态
if 'show_explanation' not in st.session_state:
    st.session_state.show_explanation = False

# 添加“解释变量”按钮
if st.button("Explain Variables"):
    # 切换解释显示状态
    st.session_state.show_explanation = not st.session_state.show_explanation

# 根据状态显示或隐藏解释内容
if st.session_state.show_explanation:
    st.info("""
        **Age**: Age of the person (18 to 100 years).
    """)
    st.info("""
        **BMI**: Body Mass Index, a measure of body fat based on height and weight.
    """)
    st.info("""
        **SBP**: Systolic Blood Pressure, the top number in a blood pressure reading.
    """)
    st.info("""
        **DBP**: Diastolic Blood Pressure, the bottom number in a blood pressure reading.
    """)
    st.info("""
        **FPG**: Fasting Plasma Glucose, a test for diabetes.
    """)
    st.info("""
        **Chol**: Total Cholesterol, a measure of all cholesterol in the blood.
    """)
    st.info("""
        **Tri**: Triglycerides, a type of fat found in blood.
    """)
    st.info("""
        **HDL**: High-Density Lipoprotein Cholesterol, "good" cholesterol.
    """)
    st.info("""
        **LDL**: Low-Density Lipoprotein Cholesterol, "bad" cholesterol.
    """)
    st.info("""
        **ALT**: Alanine Aminotransferase, an enzyme that helps metabolize proteins.
    """)
    st.info("""
        **BUN**: Blood Urea Nitrogen, a measure of kidney function.
    """)
    st.info("""
        **CCR**: Creatinine Clearance Rate, a test of kidney function.
    """)
    st.info("""
        **FFPG**: Fasting Free Plasma Glucose, another measure for diabetes diagnosis.
    """)
    st.info("""
        **Smoking**: Whether the person smokes (0 = No, 1 = Yes).
    """)
    st.info("""
        **Drinking**: Whether the person drinks alcohol (0 = No, 1 = Yes).
    """)

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

    # 保存预测结果到 session_state
    st.session_state.prediction = f"Based on feature values, predicted possibility of diabetes is {probability:.2f}%"

# 显示预测结果为文字
if 'prediction' in st.session_state:
    st.write(st.session_state.prediction)

# 新增：显示SHAP力图的按钮
if st.button("Show SHAP Force Plot"):
    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 获取当前预测类别的SHAP值
    class_index = model.predict(features)[0]  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
