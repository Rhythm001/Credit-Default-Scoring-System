import pandas as pd
from sqlalchemy import text
from sklearn.metrics import roc_auc_score
from config import GOOD_STATUSES, BAD_STATUSES, REPORTS
from db import get_engine

engine = get_engine()

print("\n== Loans by status ==")
with engine.connect() as conn:
    print(pd.read_sql(text("SELECT loan_status, count(*) AS n FROM raw_loans GROUP BY 1 ORDER BY 2 DESC"), conn))

print("\n== Bad rate by issue year (finished loans only) ==")
statuses = tuple(GOOD_STATUSES + BAD_STATUSES)
bad = tuple(BAD_STATUSES)
q = text("""
    SELECT right(issue_d, 4) AS year, count(*) AS n,
           avg(CASE WHEN loan_status IN :bad THEN 1.0 ELSE 0.0 END) AS bad_rate
    FROM raw_loans WHERE loan_status IN :st GROUP BY 1 ORDER BY 1
""")
with engine.connect() as conn:
    print(pd.read_sql(q, conn, params={"st": statuses, "bad": bad}).round(3))

sample_q = text("SELECT * FROM raw_loans WHERE loan_status IN :st ORDER BY random() LIMIT 100000")
with engine.connect() as conn:
    df = pd.read_sql(sample_q, conn, params={"st": statuses})
df["bad"] = df["loan_status"].isin(BAD_STATUSES).astype(int)

rows = []
for col in df.select_dtypes("number").columns:
    if col == "bad":
        continue
    s = df[col].fillna(df[col].median())
    if s.nunique() < 2:
        continue
    auc = roc_auc_score(df["bad"], s)
    rows.append({"column": col, "auc": round(auc, 3), "strength": round(abs(auc - 0.5) * 2, 3),
                 "missing_pct": round(df[col].isna().mean() * 100, 1)})
audit = pd.DataFrame(rows).sort_values("strength", ascending=False)

print("\n== Single-column strength (AUC distance from 0.5, scaled 0-1) ==")
print(audit.to_string(index=False))
audit.to_csv(REPORTS / "leakage_audit.csv", index=False)

ALLOWED = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high", "emp_length", "home_ownership",
           "verification_status", "purpose", "earliest_cr_line", "inq_last_6mths", "open_acc", "pub_rec",
           "revol_bal", "revol_util", "total_acc", "mort_acc", "delinq_2yrs"]
BANNED = ["total_pymnt", "total_rec_prncp", "recoveries", "last_fico_range_high", "out_prncp",
          "int_rate", "grade", "installment"]
print("\nAllowed:", ALLOWED)
print("Banned :", BANNED)
