import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==========================================
# ChurnSense - Customer Churn Intelligence
# ==========================================

# Page Configuration
st.set_page_config(
    page_title="ChurnSense | Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    [data-testid="stMetric"] {
        background-color: #1c2333;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #303846;
    }

    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# Load Dataset
@st.cache_data
def load_data():
    path = "data/cleaned_churn.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


df = load_data()


# Sidebar
st.sidebar.title("📊 ChurnSense")
st.sidebar.caption("Customer Churn Intelligence Platform")

st.sidebar.markdown("---")

st.sidebar.info(
    "Analyze customer behavior, "
    "understand churn patterns and "
    "identify business opportunities."
)


# Main Dashboard
st.title("📊 ChurnSense")
st.subheader("Customer Churn Intelligence Dashboard")

st.markdown(
    "Understand customer behavior and discover "
    "patterns that drive customer churn."
)

st.markdown("---")


if df is None:

    st.error("Dataset not found!")

    st.info(
        "Please ensure data/cleaned_churn.csv exists."
    )

    st.stop()


# Calculate KPIs
total_customers = len(df)

churned_customers = (
    df["Churn"].astype(str).str.strip().eq("Yes").sum()
)

retained_customers = total_customers - churned_customers

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0 else 0
)


# KPI Cards
st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)

col3.metric(
    "Retained Customers",
    f"{retained_customers:,}"
)

col4.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)


st.markdown("---")


# Churn Distribution
st.subheader("Customer Churn Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    churn_data = pd.DataFrame({
        "Customer Status": ["Retained", "Churned"],
        "Customers": [
            retained_customers,
            churned_customers
        ]
    })

    fig = px.pie(
        churn_data,
        names="Customer Status",
        values="Customers",
        title="Customer Churn Distribution",
        hole=0.55,
        color_discrete_sequence=["#00CC96", "#EF553B"]
    )

    st.plotly_chart(fig, use_container_width=True)


with chart_col2:

    churn_counts = (
        df["Contract"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = ["Contract Type", "Customers"]

    fig = px.bar(
        churn_counts,
        x="Contract Type",
        y="Customers",
        title="Customers by Contract Type",
        color="Contract Type",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    st.plotly_chart(fig, use_container_width=True)


# Monthly Charges
st.markdown("---")

st.subheader("Monthly Charges Analysis")

if "MonthlyCharges" in df.columns:

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=30,
        title="Monthly Charges vs Customer Churn",
        barmode="overlay",
        opacity=0.75
    )

    st.plotly_chart(fig, use_container_width=True)


# Dataset Preview
st.markdown("---")

st.subheader("Customer Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# Footer
st.markdown("---")

st.caption(
    "ChurnSense | Machine Learning Powered Customer Intelligence"
)