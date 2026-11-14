import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Business Insights | ChurnSense",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Business Insights & Recommendations")
st.caption("Discover customer churn patterns and identify opportunities to improve retention.")

# Load dataset
DATA_PATH = "data/cleaned_churn.csv"

if not os.path.exists(DATA_PATH):
    st.error("Cleaned dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Basic calculations
total_customers = len(df)
churned = df[df["Churn"] == "Yes"]
retained = df[df["Churn"] == "No"]

churn_rate = len(churned) / total_customers * 100

# KPI cards
st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{len(churned):,}")
col3.metric("Retained Customers", f"{len(retained):,}")
col4.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# Contract analysis
st.subheader("1. Churn by Contract Type")

contract_data = (
    df.groupby("Contract")
    .agg(
        Total_Customers=("Churn", "size"),
        Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
    )
    .reset_index()
)

contract_data["Churn_Rate"] = (
    contract_data["Churned_Customers"] /
    contract_data["Total_Customers"] * 100
)

fig1 = px.bar(
    contract_data,
    x="Contract",
    y="Churn_Rate",
    text_auto=".2f",
    title="Churn Rate by Contract Type",
    labels={"Churn_Rate": "Churn Rate (%)"}
)

st.plotly_chart(fig1, use_container_width=True)

highest_contract = contract_data.loc[
    contract_data["Churn_Rate"].idxmax()
]

st.info(
    f"💡 Highest churn rate: {highest_contract['Contract']} "
    f"({highest_contract['Churn_Rate']:.2f}%)."
)

# Tenure analysis
st.subheader("2. Customer Tenure Analysis")

tenure_data = (
    df.groupby("Churn")
    ["tenure"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    tenure_data,
    x="Churn",
    y="tenure",
    text_auto=".1f",
    title="Average Customer Tenure",
    labels={"tenure": "Average Tenure (Months)"}
)

st.plotly_chart(fig2, use_container_width=True)

# Monthly charges analysis
st.subheader("3. Monthly Charges Analysis")

charges_data = (
    df.groupby("Churn")
    ["MonthlyCharges"]
    .mean()
    .reset_index()
)

fig3 = px.bar(
    charges_data,
    x="Churn",
    y="MonthlyCharges",
    text_auto=".2f",
    title="Average Monthly Charges by Churn Status",
    labels={"MonthlyCharges": "Average Monthly Charges"}
)

st.plotly_chart(fig3, use_container_width=True)

# Payment method analysis
st.subheader("4. Payment Method Analysis")

payment_data = (
    df.groupby("PaymentMethod")
    .agg(
        Total_Customers=("Churn", "size"),
        Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
    )
    .reset_index()
)

payment_data["Churn_Rate"] = (
    payment_data["Churned_Customers"] /
    payment_data["Total_Customers"] * 100
)

fig4 = px.bar(
    payment_data,
    x="PaymentMethod",
    y="Churn_Rate",
    text_auto=".2f",
    title="Churn Rate by Payment Method",
    labels={"Churn_Rate": "Churn Rate (%)"}
)

st.plotly_chart(fig4, use_container_width=True)

# Business recommendations
st.divider()

st.subheader("5. Business Recommendations")

recommendations = [
    (
        "📄 Contract Strategy",
        "Offer discounts or incentives to customers on month-to-month contracts "
        "to encourage longer-term commitments."
    ),
    (
        "🤝 Customer Retention",
        "Contact customers who show signs of dissatisfaction and provide "
        "personalized retention offers."
    ),
    (
        "💰 Pricing Strategy",
        "Review pricing plans and offer suitable packages to customers "
        "with higher monthly charges."
    ),
    (
        "🛠️ Customer Support",
        "Improve technical support and provide faster issue resolution "
        "to strengthen customer satisfaction."
    ),
    (
        "💳 Payment Experience",
        "Make payment processes easier and communicate available payment "
        "options clearly."
    )
]

for title, description in recommendations:
    with st.expander(title):
        st.write(description)

# Final summary
st.divider()

st.subheader("Executive Summary")

st.markdown(f"""
- Total customers analyzed: **{total_customers:,}**
- Customers who churned: **{len(churned):,}**
- Overall churn rate: **{churn_rate:.2f}%**
- Contract type with the highest churn rate: **{highest_contract['Contract']}**
- Its churn rate: **{highest_contract['Churn_Rate']:.2f}%**

The business should prioritize customer retention, especially among
customer groups with higher observed churn rates.
""")

st.success("Business Insights analysis completed successfully!")