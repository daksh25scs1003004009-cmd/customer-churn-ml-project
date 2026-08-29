
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=350, random_state=42, class_weight="balanced", max_depth=10
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

results = {}
fitted = {}

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4)
    }
    fitted[name] = (pipe, pred)

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_pipe, best_pred = fitted[best_name]
joblib.dump(best_pipe, MODEL_DIR / "churn_model.joblib")

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump({"best_model": best_name, "models": results}, f, indent=2)

cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(ax=ax)
ax.set_title(f"Confusion Matrix - {best_name}")
fig.tight_layout()
fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

print("Training complete.")
print("Best model:", best_name)
for name, metrics in results.items():
    print(name, metrics)
