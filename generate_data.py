
# Regenerates the same style of synthetic customer-churn dataset used by the project.
# The committed CSV is included so the project works immediately after cloning.
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
n = 2500

gender = rng.choice(["Female", "Male"], n)
senior = rng.choice([0, 1], n, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
tenure = np.clip(rng.gamma(2.2, 15, n).round().astype(int), 0, 72)
phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.22, 0.23])
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                     n, p=[0.34, 0.22, 0.23, 0.21])
support = rng.choice(["Yes", "No", "No internet service"], n, p=[0.20, 0.58, 0.22])
stream_tv = rng.choice(["Yes", "No", "No internet service"], n, p=[0.30, 0.48, 0.22])
stream_movies = rng.choice(["Yes", "No", "No internet service"], n, p=[0.31, 0.47, 0.22])
paperless = rng.choice(["Yes", "No"], n, p=[0.58, 0.42])

base_charge = np.where(internet == "Fiber optic", 72, np.where(internet == "DSL", 59, 25))
monthly = np.clip(base_charge + rng.normal(0, 8, n)
                  + np.where(stream_tv == "Yes", 8, 0)
                  + np.where(stream_movies == "Yes", 8, 0), 18, 125).round(2)
total = np.maximum(0, monthly * tenure * rng.normal(0.98, 0.07, n)).round(2)

logit = (-1.8 + 0.95*(contract == "Month-to-month") - 0.85*(contract == "Two year")
         - 0.035*tenure + 0.020*(monthly-60) + 0.65*(payment == "Electronic check")
         + 0.48*(internet == "Fiber optic") + 0.35*(senior == 1)
         + 0.32*(paperless == "Yes") - 0.28*(partner == "Yes")
         - 0.25*(dependents == "Yes") - 0.50*(support == "Yes") + rng.normal(0,0.65,n))
p = 1/(1+np.exp(-logit))
churn = rng.binomial(1,p)

pd.DataFrame({
    "customer_id":[f"CUST{i:05d}" for i in range(1,n+1)],
    "gender":gender, "senior_citizen":senior, "partner":partner, "dependents":dependents,
    "tenure_months":tenure, "phone_service":phone, "internet_service":internet,
    "contract":contract, "payment_method":payment, "tech_support":support,
    "streaming_tv":stream_tv, "streaming_movies":stream_movies,
    "paperless_billing":paperless, "monthly_charges":monthly, "total_charges":total,
    "churn":churn
}).to_csv(ROOT/"data"/"customer_churn.csv", index=False)

print("Dataset regenerated:", n, "rows")
