# ChurnSense — Customer Churn Intelligence Platform

An interactive machine learning dashboard that analyzes customer
churn patterns and predicts customers who may leave a business.

## Overview

ChurnSense is a Customer Churn Intelligence Platform developed
using Python, Machine Learning, and Streamlit.

It helps businesses understand customer behavior, identify
potential churn risks, compare machine learning models, and
explore data-driven retention strategies.

## Features

- Interactive business dashboard
- Customer analytics and filtering
- Customer churn risk prediction
- Logistic Regression model
- Random Forest model
- Machine learning model comparison
- Business insights and recommendations
- Interactive Plotly visualizations

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Machine Learning Models

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest

### Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.81% | 75.59% |
| Precision | 50.43% | 52.96% |
| Recall | 78.34% | 71.66% |
| F1 Score | 61.36% | 60.91% |
| ROC-AUC | 84.16% | 83.22% |

Logistic Regression achieved the highest ROC-AUC and Recall.
Random Forest achieved higher Accuracy and Precision.

## Project Structure

```text
ChurnSense/
│
├── app.py
├── requirements.txt
├── Readme.md
├── .gitignore
│
├── assets/
│   ├── churn_distribution.html
│   ├── contract_churn.html
│   ├── customer_tenure.html
│   └── monthly_charges.html
│
├── data/
│   ├── raw/
│   └── cleaned_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── model_comparison.csv
│
├── pages/
│   ├── 1_Customer_Analytics.py
│   ├── 2_Customer_Risk_Prediction.py
│   ├── 3_Model_Performance.py
│   └── 4_Business_Insights.py
│
└── src/