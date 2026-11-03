
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# 1. Load dataset
data_path = Path("./data/dry_bean.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["Class"])
import numpy as np
from sklearn.preprocessing import StandardScaler

print("\n========== NUMERICAL CHECK ==========")

print("Missing values:", X.isnull().sum().sum())
print("Infinite values:", np.isinf(X.to_numpy()).sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("All scaled values finite:",
      np.isfinite(X_scaled).all())

print("Largest absolute scaled value:",
      np.max(np.abs(X_scaled)))

print("=====================================")
y = df["Class"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Save the held-out test dataset for Streamlit evaluation
test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("\nTest dataset saved to test_data.csv")
print("Test dataset shape:", test_data.shape)

# 3. Define the five models
models = {
    "Logistic Regression": Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=5000,
        C=0.1,
        solver="lbfgs"
    )),
]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Naive Bayes": GaussianNB(),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        )),
    ]),
}

# 4. Create model output folder
model_dir = Path("./models")
model_dir.mkdir(parents=True, exist_ok=True)

results = []

# 5. Train, evaluate and save each model
classes = sorted(y.unique())
y_test_binary = label_binarize(y_test, classes=classes)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train using training data only
    model.fit(X_train, y_train)

    # Predict test labels and class probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate all six required metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(
            y_test_binary,
            y_proba,
            average="macro",
        ),
        "Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    results.append(metrics)

    # Save the complete fitted pipeline/model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, model_dir / filename)

    print(f"Saved: {model_dir / filename}")

# 6. Display and save comparison table
results_df = pd.DataFrame(results).set_index("Model")

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string())
print("\n======================================")

results_df.to_csv("model_comparison.csv")
print("\nMetrics saved to model_comparison.csv")
