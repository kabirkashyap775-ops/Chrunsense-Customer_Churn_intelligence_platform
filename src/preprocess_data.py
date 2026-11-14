# ==========================================
# ChurnSense - Data Preprocessing
# ==========================================

import pandas as pd

# Load original dataset
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)

# 1. Clean whitespace from TotalCharges
df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()

# 2. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 3. Check missing values
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# 4. Handle missing values
# Customers with tenure 0 have no accumulated charges.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 5. Remove duplicate records if any
df = df.drop_duplicates()

# 6. Remove customerID
df = df.drop(columns=["customerID"])

# 7. Verify cleaned dataset
print("\nCleaned Dataset Shape:", df.shape)

print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

# 8. Save cleaned dataset
OUTPUT_PATH = "data/cleaned_churn.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")