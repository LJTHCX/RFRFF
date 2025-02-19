import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载糖尿病预测模型
model = joblib.load('RF.pkl') 

# 定义新数据集的特征名称
feature_names = [
    "年龄", "体重指数(BMI)", "收缩压(SBP)", "舒张压(DBP)", "空腹血糖(FPG)",
    "胆固醇(Chol)", "甘油三酯(Tri)", "高密度脂蛋白(HDL)", "低密度脂蛋白(LDL)",
    "丙氨酸氨基转移酶(ALT)", "尿素氮(BUN)", "肌酐清除率(CCR)", "空腹血糖(FFPG)",
    "吸烟(吸烟)", "饮酒(饮酒)"
]

# Streamlit 用户界面
st.title("糖尿病预测")

# 输入特征
age = st.number_input("年龄:", min_value=18, max_value=100, value=30)
bmi = st.number_input("体重指数(BMI):", min_value=10.0, max_value=50.0, value=24.0)
sbp = st.number_input("收缩压(SBP):", min_value=50, max_value=200, value=120)
dbp = st.number_input("舒张压(DBP):", min_value=30, max_value=120, value=80)
fpg = st.number_input("空腹血糖(FPG):", min_value=0.0, max_value=20.0, value=5.0)
chol = st.number_input("胆固醇(Chol):", min_value=0.0, max_value=10.0, value=4.5)
tri = st.number_input("甘油三酯(Tri):", min_value=0.0, max_value=10.0, value=1.0)
hdl = st.number_input("高密度脂蛋白(HDL):", min_value=0.0, max_value=10.0, value=1.5)
ldl = st.number_input("低密度脂蛋白(LDL):", min_value=0.0, max_value=10.0, value=3.0)
alt = st.number_input("丙氨酸氨基转移酶(ALT):", min_value=0.0, max_value=100.0, value=30.0)
bun = st.number_input("尿素氮(BUN):", min_value=0.0, max_value=100.0, value=20.0)
ccr = st.number_input("肌酐清除率(CCR):", min_value=0.0, max_value=100.0, value=50.0)
ffpg = st.number_input("空腹血糖(FFPG):", min_value=0.0, max_value=20.0, value=5.0)
smoking = st.selectbox("吸烟 (0=否, 1=是):", options=[0, 1])
drinking = st.selectbox("饮酒 (0=否, 1=是):", options=[0, 1])

# 收集输入值到一个列表
feature_values = [age, bmi, sbp, dbp, fpg, chol, tri, hdl, ldl, alt, bun, ccr, ffpg, smoking, drinking]

# 将输入的特征值转换为 DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("预测"):
    # 使用模型进行预测
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # 显示预测结果
    st.write(f"**预测类别 (0=无糖尿病, 1=糖尿病):** {predicted_class}")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 0:
        advice = (
            f"根据我们的模型预测，您可能没有糖尿病。"
            f"模型预测您没有糖尿病的概率为 {probability:.1f}%。"
            "建议保持健康的生活方式，继续减少风险。"
        )
    else:
        advice = (
            f"根据我们的模型预测，您可能患有糖尿病。"
            f"模型预测您患糖尿病的概率为 {probability:.1f}%。"
            "建议尽快咨询医生进行进一步的诊断和治疗。"
        )

    st.write(advice)

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer.shap_values(features_df)

    # 仅为预测类别显示 SHAP
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_Explanation[1][0], show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
