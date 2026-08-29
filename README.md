# Customer Churn Prediction — Data Science & Machine Learning

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

## Project objective

Build a reproducible ML workflow from raw customer data to a usable prediction app:

- Data inspection and exploratory data analysis
- Feature preprocessing with `ColumnTransformer`
- Logistic Regression vs Random Forest
- Evaluation with Accuracy, Precision, Recall, F1-score and ROC-AUC
- Saved production pipeline using `joblib`
- Streamlit interface for individual predictions

## Project structure

```text
customer-churn-ml-project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── customer_churn.csv
│   └── generate_data.py
├── models/
│   └── churn_model.joblib
├── notebooks/
│   └── 01_eda_and_model.ipynb
├── reports/
│   ├── metrics.json
│   └── confusion_matrix.png
└── src/
    ├── __init__.py
    ├── predict.py
    └── train.py
```

## Dataset

The included dataset contains **2,500 synthetic telecom customer records**. It is designed to resemble a typical churn dataset while remaining self-contained and reproducible.

Because the data is synthetic, the reported model performance is for demonstration only and should **not** be presented as real-world business performance.

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python src/train.py
```

This creates:

- `models/churn_model.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

### 3. Launch the web app

```bash
streamlit run app.py
```

## Machine learning workflow

```text
Customer data
     ↓
Data validation / inspection
     ↓
Train-test split
     ↓
Numerical scaling + categorical one-hot encoding
     ↓
Logistic Regression + Random Forest
     ↓
Model comparison
     ↓
Best ROC-AUC model selected
     ↓
Saved preprocessing + model pipeline
     ↓
Streamlit prediction interface
```

## Evaluation metrics

The project reports:

| Metric | Purpose |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted churners actually churned |
| Recall | How many actual churners were detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ranking ability across classification thresholds |

For churn problems, **recall and ROC-AUC should be considered alongside accuracy**, because missing a likely churner can be more costly than flagging an additional low-risk customer.

## Technologies

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit.

## Important note

This project is an internship/portfolio demonstration. The dataset is synthetic, and the model should not be used for production decisions without real-world validation, monitoring, privacy review, and fairness analysis.
