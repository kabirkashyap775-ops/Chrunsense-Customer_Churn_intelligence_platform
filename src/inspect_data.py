# ==========================================
# ChurnSense - Exploratory Data Analysis
# ==========================================

import pandas as pd

# 1. Load dataset
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET INFORMATION ==========")

# 2. Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# 3. Display columns
print("\nColumn Names:")
print(df.columns.tolist())

# 4. Dataset information
print("\nDataset Information:")
df.info()

# 5. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 7. Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# 8. Customer churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# 9. Customer churn percentage
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n========== EDA COMPLETED ==========")