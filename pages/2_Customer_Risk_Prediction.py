import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# ChurnSense - Customer Risk Prediction
# ==========================================

st.set_page_config(
    page_title="Customer Risk Prediction | ChurnSense",
    page_icon="🎯",
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
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model():

    model_path = "models/churn_model.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)

    return None


# Load Dataset
@st.cache_data
def load_data():

    data_path = "data/cleaned_churn.csv"

    if os.path.exists(data_path):
        return pd.read_csv(data_path)

    return None


model = load_model()
df = load_data()


# Sidebar
st.sidebar.title("📊 ChurnSense")
st.sidebar.caption("Customer Intelligence Platform")

st.sidebar.markdown("---")
st.sidebar.info(
    "Predict customer churn risk "
    "using Machine Learning."
)


# Page Header
st.title("🎯 Customer Risk Prediction")

st.subheader(
    "Predict which customers are likely to leave."
)

st.write(
    "Enter customer details below to calculate "
    "their probability of churning."
)

st.markdown("---")


# Verify Files
if model is None:

    st.error("Trained model not found!")

    st.info(
        "Please run: python src/train_model.py"
    )

    st.stop()


if df is None:

    st.error("Dataset not found!")

    st.stop()


# Feature Preparation
feature_columns = [
    col for col in df.columns
    if col not in ["Churn", "customerID"]
]

numerical_features = [
    col for col in feature_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

categorical_features = [
    col for col in feature_columns
    if col not in numerical_features
]


# Customer Input Form
st.subheader("👤 Customer Information")

input_data = {}

with st.form("customer_prediction_form"):

    st.markdown("### Numerical Information")

    num_cols = st.columns(3)

    for i, column in enumerate(numerical_features):

        with num_cols[i % 3]:

            minimum = float(df[column].min())
            maximum = float(df[column].max())
            default = float(df[column].median())

            if set(df[column].dropna().unique()).issubset({0, 1}):

                input_data[column] = st.selectbox(
                    column,
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )

            else:

                input_data[column] = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1.0 if pd.api.types.is_integer_dtype(df[column]) else 0.1
                )

    st.markdown("---")

    st.markdown("### Categorical Information")

    cat_cols = st.columns(3)

    for i, column in enumerate(categorical_features):

        with cat_cols[i % 3]:

            options = sorted(
                df[column].dropna().unique().tolist(),
                key=str
            )

            if options:

                input_data[column] = st.selectbox(
                    column,
                    options
                )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Predict Customer Risk",
        use_container_width=True
    )


# Prediction
if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    try:

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        churn_probability = probability * 100

        st.markdown("---")

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%"
            )

        with col2:

            status = (
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
            )

            st.metric(
                "Prediction",
                status
            )

        with col3:

            if churn_probability >= 70:
                risk = "High"

            elif churn_probability >= 40:
                risk = "Medium"

            else:
                risk = "Low"

            st.metric(
                "Risk Level",
                risk
            )

        # Progress Bar
        st.markdown("### Churn Risk Indicator")

        st.progress(
            min(max(int(churn_probability), 0), 100)
        )

        st.write(
            f"Estimated probability of churn: "
            f"{churn_probability:.2f}%"
        )

        # Recommendations
        st.markdown("---")

        st.subheader("💡 Retention Recommendations")

        recommendations = []

        if input_data.get("Contract") == "Month-to-month":
            recommendations.append(
                "Offer a discounted yearly contract."
            )

        if input_data.get("tenure", 100) <= 12:
            recommendations.append(
                "Provide special benefits to new customers."
            )

        if input_data.get("OnlineSecurity") == "No":
            recommendations.append(
                "Offer an online security service package."
            )

        if input_data.get("TechSupport") == "No":
            recommendations.append(
                "Provide technical support benefits."
            )

        if input_data.get("PaymentMethod") == "Electronic check":
            recommendations.append(
                "Encourage automatic payment methods."
            )

        if not recommendations:

            recommendations.append(
                "Provide loyalty rewards and personalized offers."
            )

        if risk == "High":

            st.error("⚠️ High-risk customer detected!")

        elif risk == "Medium":

            st.warning("⚠️ Customer requires attention.")

        else:

            st.success("✅ Customer has a relatively low churn risk.")

        for recommendation in recommendations:

            st.write("✔️", recommendation)

        # Customer Information
        with st.expander("View Customer Information"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# Footer
st.markdown("---")

st.caption(
    "ChurnSense | Machine Learning Powered Customer Intelligence"
)