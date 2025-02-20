import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# 新增：显示SHAP瀑布图的按钮
if st.button("Show SHAP Waterfall Plot"):
    # 导入数据
    df = pd.read_excel('diabetes.xlsx')

    # 划分特征和目标变量
    X = df.drop(['Diabetes'], axis=1)
    y = df['Diabetes']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Diabetes'])

    # 使用最佳参数训练模型
    best_params = {'n_estimators': 356, 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 25, 'random_state': 42}

    # 创建并训练模型
    model_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=best_params['random_state']
    )

    model_rf.fit(X_train, y_train)

    # 计算shap值为Explanation格式
    explainer = shap.TreeExplainer(model_rf)
    shap_values_Explanation = explainer.shap_values(X_test)

    # 提取类别 0 的 SHAP 值
    shap_values_class_0 = shap_values_Explanation[0]

    # 绘制第3个样本0类别的 SHAP 瀑布图
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_class_0[2], show=False, max_display=13)
    plt.savefig("shap_waterfall_plot.pdf", format='pdf', bbox_inches='tight')
    plt.tight_layout()
    st.image("shap_waterfall_plot.pdf")
