import numpy as np
import pandas as pd

HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIF = ["Verified", "Source Verified", "Not Verified"]
PURPOSE = ["debt_consolidation", "credit_card", "home_improvement", "small_business", "car", "other"]

NUMERIC = ["loan_amnt", "annual_inc", "dti", "fico_score", "emp_length_years", "credit_history_years",
           "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
           "delinq_2yrs", "loan_to_income"]
CATEGORICAL = {"home_ownership": HOME, "verification_status": VERIF, "purpose": PURPOSE}

MONOTONE = {"fico_score": -1, "dti": 1, "annual_inc": -1, "revol_util": 1, "inq_last_6mths": 1,
            "delinq_2yrs": 1, "pub_rec": 1, "loan_to_income": 1}

def parse_emp_length(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    if s.startswith("<"):
        return 0.0
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else np.nan

def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y")
    first_cr = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["issue_date"] = issue
    out["loan_amnt"] = df["loan_amnt"].astype(float)
    out["annual_inc"] = df["annual_inc"].astype(float)
    out["dti"] = df["dti"].astype(float)
    out["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    out["emp_length_years"] = df["emp_length"].map(parse_emp_length)
    out["credit_history_years"] = (issue - first_cr).dt.days / 365.25
    for c in ["inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
              "mort_acc", "delinq_2yrs"]:
        out[c] = df[c].astype(float)
    out["home_ownership"] = df["home_ownership"]
    out["verification_status"] = df["verification_status"]
    out["purpose"] = df["purpose"]
    return out

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_ownership"] = out["home_ownership"].where(out["home_ownership"].isin(HOME), "OTHER")
    out["verification_status"] = out["verification_status"].where(out["verification_status"].isin(VERIF), "Not Verified")
    out["purpose"] = out["purpose"].where(out["purpose"].isin(PURPOSE), "other")
    out["dti"] = out["dti"].clip(lower=0, upper=60)
    out["revol_util"] = out["revol_util"].clip(lower=0, upper=150)
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].clip(lower=1)
    return out

def feature_columns():
    cols = list(NUMERIC)
    for name, cats in CATEGORICAL.items():
        cols += [f"{name}={c}" for c in cats]
    return cols

def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer(df)
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = df[c].astype(float)
    for name, cats in CATEGORICAL.items():
        for c in cats:
            X[f"{name}={c}"] = (df[name] == c).astype(float)
    return X[feature_columns()]

def monotone_tuple():
    return tuple(MONOTONE.get(c, 0) for c in feature_columns())
