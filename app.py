
import streamlit as st
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

# ---------- Page setup ----------
st.set_page_config(
    page_title="Dry Bean Classification Lab",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Dry Bean Classification Lab")
st.write(
    "Compare five machine learning classifiers on the Dry Bean dataset. "
    "Upload labelled test data to evaluate a saved model."
)

# ---------- Model configuration ----------
MODEL_DIR = Path("models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

TARGET = "Class"

# ---------- Sidebar ----------
st.sidebar.header("Evaluation settings")

selected_model = st.sidebar.selectbox(
    "Choose a classifier",
    list(MODEL_FILES.keys()),
)

uploaded_file = st.sidebar.file_uploader(
    "Upload labelled test data (CSV)",
    type=["csv"],
)

# ---------- Dataset loading ----------
if uploaded_file is None:
    default_path = Path("test_data.csv")

    if default_path.exists():
        df = pd.read_csv(default_path)
        st.info("Using the local test_data.csv file.")
    else:
        st.warning("Upload a test CSV to begin.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

# ---------- Validate dataset ----------
if TARGET not in df.columns:
    st.error(f"The CSV must contain the target column '{TARGET}'.")
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[selected_model]

if not model_path.exists():
    st.error(f"Saved model not found: {model_path}")
    st.stop()

model = joblib.load(model_path)
expected_features = list(model.feature_names_in_) if hasattr(
    model, "feature_names_in_"
) else None

# Pipelines store feature names on the fitted classifier in some versions.
if expected_features is None and hasattr(model, "named_steps"):
    classifier = model.named_steps.get("classifier")
    if classifier is not None and hasattr(classifier, "feature_names_in_"):
        expected_features = list(classifier.feature_names_in_)

if expected_features is None:
    expected_features = [
        "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
        "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
        "Extent", "Solidity", "roundness", "Compactness",
        "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
    ]

missing_features = [
    feature for feature in expected_features if feature not in df.columns
]

if missing_features:
    st.error(f"Missing required features: {missing_features}")
    st.stop()

# Ignore unrelated columns; preserve the model's expected feature order.
X_test = df[expected_features].apply(pd.to_numeric, errors="coerce")
y_test = df[TARGET]

if X_test.isnull().any().any() or not np.isfinite(X_test.to_numpy()).all():
    st.error("Input features contain missing or non-finite values.")
    st.stop()

if y_test.isnull().any():
    st.error("The target column contains missing values.")
    st.stop()

# ---------- Prediction and evaluation ----------
try:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    model_classes = model.classes_

    if len(np.unique(y_test)) < 2:
        st.error("AUC requires at least two classes in the test data.")
        st.stop()

    y_test_binary = label_binarize(
        y_test,
        classes=model_classes,
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC (macro)": roc_auc_score(
            y_test_binary,
            y_proba,
            average="macro",
            labels=None,
        ),
        "Precision (macro)": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1 (macro)": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

# ---------- Display results ----------
st.subheader("Dataset overview")

col1, col2, col3 = st.columns(3)
col1.metric("Test rows", f"{len(df):,}")
col2.metric("Input features", len(expected_features))
col3.metric("Bean classes", y_test.nunique())

with st.expander("Preview uploaded test data"):
    st.dataframe(df.head(10), use_container_width=True)

st.subheader(f"Evaluation metrics — {selected_model}")

metric_columns = st.columns(3)

for index, (name, value) in enumerate(metrics.items()):
    metric_columns[index % 3].metric(name, f"{value:.4f}")

st.subheader("Confusion matrix")

labels = list(model_classes)
cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels],
)

st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification report")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
    output_dict=True,
)

st.dataframe(
    pd.DataFrame(report).transpose().round(4),
    use_container_width=True,
)

st.caption(
    "The application loads saved models and evaluates uploaded labelled "
    "test data. It does not retrain models on uploaded data."
)
