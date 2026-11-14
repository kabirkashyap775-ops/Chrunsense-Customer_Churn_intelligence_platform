import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Model Performance | ChurnSense",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")
st.caption("Compare machine learning models used for customer churn prediction.")

# Load model comparison results
file_path = "models/model_comparison.csv"

if not os.path.exists(file_path):
    st.error("Model comparison file not found. Please check the models folder.")
    st.stop()

results = pd.read_csv(file_path)

# Convert metrics into percentages
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

for metric in metrics:
    results[metric] = results[metric] * 100

# Best model based on ROC-AUC
best_model = results.loc[results["ROC-AUC"].idxmax()]

st.success(
    f"🏆 Best model by ROC-AUC: {best_model['Model']} "
    f"({best_model['ROC-AUC']:.2f}%)"
)

# Model comparison chart
st.subheader("Model Metrics Comparison")

melted_data = results.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    melted_data,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    text_auto=".2f",
    title="Logistic Regression vs Random Forest",
    labels={"Score": "Score (%)"}
)

fig.update_layout(
    yaxis_range=[0, 100],
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Individual model metrics
st.subheader("Performance Metrics")

for _, row in results.iterrows():
    st.markdown(f"### {row['Model']}")

    columns = st.columns(5)

    for col, metric in zip(columns, metrics):
        col.metric(metric, f"{row[metric]:.2f}%")

# Detailed results
st.subheader("Detailed Comparison")

display_results = results.copy()

for metric in metrics:
    display_results[metric] = display_results[metric].map(
        lambda value: f"{value:.2f}%"
    )

st.dataframe(display_results, use_container_width=True, hide_index=True)

# Explanation
st.subheader("What do these metrics mean?")

with st.expander("Understand the evaluation metrics"):
    st.markdown("""
    - **Accuracy:** Percentage of all predictions that were correct.
    - **Precision:** Percentage of predicted churners who actually churned.
    - **Recall:** Percentage of actual churners correctly identified.
    - **F1 Score:** Balance between Precision and Recall.
    - **ROC-AUC:** Measures how well the model distinguishes between churners and non-churners.

    **Project observation:** Logistic Regression achieved the highest ROC-AUC
    and Recall, while Random Forest achieved higher Accuracy and Precision.
    """)