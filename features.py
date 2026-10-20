import numpy as np
import pandas as pd

# Features are build here and imported by training scripts and by the API. One shared file.

HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIF = ["Verified", "Source Verified", "Not Verified"]
PURPOSE = ["debt_consolidation", "credit_card", "home_improvement", "small_business", "car", "other"]

NUMERIC = ["loan_amnt", "annual_inc", "dti", "fico_score", "emp_length_years", "credit_history_years", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revot_util", "total_acc", "mort_acc", "delinq_2yrs", "loan_to_income"]

CATEGORICAL = {"home_ownership": HOME, "verification_status": VERIF, "purpose": PURPOSE}

# Fixed lists of allowed category values. One-hot encoding must always produce the same columns in the same order, whether we score a million loans or one applicant.
# Anything unseen is mapped to a default bucket later
# CATEGORICAL is a dictionary: column name on left, allowed values on the right. "Purpuse" values that cover the most real loans.


# Business rules as constraints
MONOTONE = {"fico_score": -1, "dti": 1, "annual_inc": -1, "revol_util": 1, "inq_last_6mths": 1, "delinq_2yrs": 1, "pub_rec": 1, "loan_to_income": 1}

# +1 : The model must never predict lower risk when this feature goes up
# -1 : The opposite
# A higher FICO must never increase risk. Monotonic constraint. Without it, XG boost can learn old wiggles from noise.

# Converting employment length
def parse_emp_length(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    if s.startswith("<"):
        return 0.0
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else np.nan

# Converts text like "<1 year" to 0. and "10+ years" becomes 10. Missing stays NaN. XGBoost handles missing values natively.

# Raw Table to clean columns (training only)
def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    issue = pd.to_datetime(df["issue_d"], format = "%b-%Y")
    first_cr = pd.to_datetime(df["earliest_cr_line"], format = "%b-%Y", errors="coerce")
    out["issue_date"] = issue
    out["loan_amnt"] = df["loan_amnt"].astype(float)
    out["annual_inc"] = df["annual_inc"].astype(float)
    out["dti"] = df["dti"].astype(float)
    out["fico_score"] = (df["fico_range_low"] + df["fico_range_high"])/2
    out["emp_length_years"] = df["emp_length"].map(parse_emp_length)
    out["credit_history_years"] = (issue - first_cr).dt.days / 365.25
    for c in ["inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "delinq_2yrs"]:
        out[c] = df[c].astype(float)
        out["home_ownership"] = df["home_ownership"]
        out["verification_status"] = df["verification_status"]
        out["purpose"] = df["purpose"]
        return out
# Converts the raw table unto clean "canonical" column. 
# errors = "coerce" turns unparseable dates into missing instead of crashing.
# Fico comes as low-high range, so we take the midpoint. 
#Credit history is the gap between the issue date and the first credit line, measured in years (days divided by 365.25). 
# It is measured as "issue date", not todya, which keeps it point-in-time correct.




# Cleaning shared by training and serving
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_ownership"] = out["home_ownership"].where(out["home_ownership"].isin(HOME), "OTHER")
    


