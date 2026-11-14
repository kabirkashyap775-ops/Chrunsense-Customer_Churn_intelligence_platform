import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Customer Analytics | ChurnSense",
    page_icon="📈",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_churn.csv")

df = load_data()

# Page heading
st.title("📈 Customer Analytics")
st.markdown(
    "Explore customer demographics, services, "
    "contracts and churn behavior."
)

st.markdown("---")

# Filters
st.sidebar.header("Analytics Filters")

contract_options = ["All"] + sorted(df["Contract"].dropna().unique().tolist())

selected_contract = st.sidebar.selectbox(
    "Select Contract",
    contract_options
)

filtered_df = df.copy()

if selected_contract != "All":
    filtered_df = filtered_df[
        filtered_df["Contract"] == selected_contract
    ]

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(filtered_df):,}")

churned = (filtered_df["Churn"] == "Yes").sum()

col2.metric("Churned Customers", f"{churned:,}")

churn_rate = (
    churned / len(filtered_df) * 100
    if len(filtered_df) > 0 else 0
)

col3.metric("Churn Rate", f"{churn_rate:.2f}%")

st.markdown("---")

# Contract Analysis
st.subheader("Customer Churn by Contract")

contract_churn = (
    filtered_df.groupby(["Contract", "Churn"])
    .size()
    .reset_index(name="Customers")
)

fig1 = px.bar(
    contract_churn,
    x="Contract",
    y="Customers",
    color="Churn",
    barmode="group",
    title="Churn Distribution Across Contracts",
    color_discrete_map={
        "Yes": "#EF553B",
        "No": "#00CC96"
    }
)

st.plotly_chart(fig1, use_container_width=True)

# Monthly Charges
st.subheader("Monthly Charges Analysis")

fig2 = px.box(
    filtered_df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges vs Churn",
    color_discrete_map={
        "Yes": "#EF553B",
        "No": "#00CC96"
    }
)

st.plotly_chart(fig2, use_container_width=True)

# Tenure Analysis
st.subheader("Customer Tenure Analysis")

fig3 = px.histogram(
    filtered_df,
    x="tenure",
    color="Churn",
    nbins=30,
    barmode="overlay",
    title="Customer Tenure Distribution",
    opacity=0.75
)

st.plotly_chart(fig3, use_container_width=True)

# Dataset
with st.expander("View Customer Records"):
    st.dataframe(filtered_df, use_container_width=True)

st.success("Customer Analytics loaded successfully!")