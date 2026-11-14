# ==========================================
# ChurnSense - Machine Learning Model Training
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load Dataset
print("Loading cleaned dataset...")

df = pd.read_csv("data/cleaned_churn.csv")

# 2. Separate Features and Target
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
y = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Identify Numerical and Categorical Columns
numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

# 4. Preprocessing Pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Define Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

# 7. Train and Evaluate Models
results = []
best_model = None
best_score = -1
best_model_name = ""

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    if auc > best_score:
        best_score = auc
        best_model = pipeline
        best_model_name = name

# 8. Save Model Comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

# 9. Save Best Model
joblib.dump(best_model, "models/churn_model.pkl")

print("\nBEST MODEL:", best_model_name)
print("BEST ROC-AUC:", round(best_score, 4))

# 10. Final Evaluation
final_predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    final_predictions,
    target_names=["No Churn", "Churn"]
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, final_predictions))

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")