import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from config import ARTIFACTS
from features import to_matrix

REASON_TEXT = {
    "fico_score": "Credit score is low", "dti": "Debt-to-income ratio is high", "annual_inc": "Income is low",
    "revol_util": "Credit card utilisation is high", "inq_last_6mths": "Many recent credit enquiries",
    "delinq_2yrs": "Recent delinquencies on credit file", "pub_rec": "Derogatory public records",
    "loan_to_income": "Loan amount is large relative to income", "loan_amnt": "Loan amount requested is high",
    "credit_history_years": "Short credit history", "emp_length_years": "Short employment history",
    "open_acc": "Number of open accounts", "total_acc": "Number of total accounts",
    "revol_bal": "Revolving balance is high", "mort_acc": "Few mortgage accounts",
}

INPUT_FIELDS = ["loan_amnt", "annual_inc", "dti", "fico_score", "emp_length_years", "credit_history_years",
                "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
                "delinq_2yrs", "home_ownership", "verification_status", "purpose"]
CATEGORICAL_FIELDS = ["home_ownership", "verification_status", "purpose"]

def application_frame(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])[INPUT_FIELDS]
    for c in INPUT_FIELDS:
        if c not in CATEGORICAL_FIELDS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

class Scorer:
    def __init__(self):
        self.booster = xgb.Booster()
        self.booster.load_model(str(ARTIFACTS / "xgb_model.json"))
        self.calibrator = joblib.load(ARTIFACTS / "calibrator.joblib")
        self.meta = json.load(open(ARTIFACTS / "meta.json"))
        self.columns = self.meta["feature_columns"]

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(to_matrix(df), feature_names=self.columns)
        return self.calibrator.predict(self.booster.predict(dm))

    def reasons(self, df: pd.DataFrame, top_n: int = 3):
        dm = xgb.DMatrix(to_matrix(df), feature_names=self.columns)
        contrib = self.booster.predict(dm, pred_contribs=True)[:, :-1]
        out = []
        for row in contrib:
            order = np.argsort(-row)[:top_n]
            items = []
            for i in order:
                if row[i] <= 0:
                    continue
                name = self.columns[i]
                text = REASON_TEXT.get(name, name.replace("=", ": ").replace("_", " ").capitalize() + " raises risk")
                items.append({"feature": name, "text": text, "impact": round(float(row[i]), 3)})
            out.append(items)
        return out

    def decide(self, pd_value: float) -> str:
        return "approve" if pd_value <= self.meta["cutoff_pd"] else "decline"

_scorer = None
def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer
