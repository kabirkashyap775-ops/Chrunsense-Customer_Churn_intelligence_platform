# ==========================================
# ChurnSense - Exploratory Data Visualization
# ==========================================

import pandas as pd
import plotly.express as px
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_churn.csv"
OUTPUT_DIR = BASE_DIR / "assets"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("ChurnSense Visualization Module")
print("Dataset loaded:", df.shape)


# ==========================================
# 1. Customer Churn Distribution
# ==========================================

churn_counts = df["Churn"].value_counts().reset_index()
churn_counts.columns = ["Churn", "Customers"]

fig1 = px.pie(
    churn_counts,
    names="Churn",
    values="Customers",
    title="Customer Churn Distribution",
    hole=0.45
)

fig1.update_layout(
    template="plotly_dark",
    title_x=0.5
)

fig1.write_html(OUTPUT_DIR / "churn_distribution.html")

print("1. Churn distribution chart created!")


# ==========================================
# 2. Churn by Contract Type
# ==========================================

contract_data = (
    df.groupby(["Contract", "Churn"])
    .size()
    .reset_index(name="Customers")
)

fig2 = px.bar(
    contract_data,
    x="Contract",
    y="Customers",
    color="Churn",
    barmode="group",
    title="Customer Churn by Contract Type",
    text_auto=True
)

fig2.update_layout(
    template="plotly_dark",
    title_x=0.5
)

fig2.write_html(OUTPUT_DIR / "contract_churn.html")

print("2. Contract analysis chart created!")


# ==========================================
# 3. Monthly Charges vs Churn
# ==========================================

fig3 = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges vs Customer Churn"
)

fig3.update_layout(
    template="plotly_dark",
    title_x=0.5
)

fig3.write_html(OUTPUT_DIR / "monthly_charges.html")

print("3. Monthly charges chart created!")


# ==========================================
# 4. Customer Tenure vs Churn
# ==========================================

fig4 = px.histogram(
    df,
    x="tenure",
    color="Churn",
    barmode="overlay",
    nbins=30,
    title="Customer Tenure vs Churn",
    opacity=0.75
)

fig4.update_layout(
    template="plotly_dark",
    title_x=0.5
)

fig4.write_html(OUTPUT_DIR / "customer_tenure.html")

print("4. Customer tenure chart created!")


# ==========================================
# COMPLETED
# ==========================================

print("\nAll visualizations generated successfully!")
print("Charts saved inside the assets folder.")