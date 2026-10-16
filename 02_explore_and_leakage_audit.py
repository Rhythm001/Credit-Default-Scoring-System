"""
Explore historical loan outcomes and audit features for leakage.

Objectives:
1. Summarize loan statuses and bad rates by issue year.
2. Calculate single-feature AUC for numeric columns.
3. Flag suspiciously predictive features for investigation.
4. Export the audit report for review.

Core principle:
Every feature must be available at the intended prediction time.
High AUC is a warning signal, not proof of leakage.
See docs/data_exploration_and_leakage.md for the full theory.
"""

import pandas as pd
from sqlalchemy import text
from sklearn.metrics import roc_auc_score
from config import GOOD_STATUSES, BAD_STATUSES, REPORTS
from db import get_engine

engine = get_engine()

# how many loans per status
print("\n== Loans by status ==")
with engine.connect() as conn:
    print(pd.read_sql(text("SELECT loan_status, count(*) AS n FROM raw_loans GROUP BY 1 ORDER BY 2 DESC"), conn))
    # pd.read_sql runs the query and returns a table

# Bad Rate by Year (Vintage View)
print("\n== Bad rate by issue year (finished loans only) ==")
statuses = tuple(GOOD_STATUSES + BAD_STATUSES)
bad = tuple(BAD_STATUSES)
q = text("""
    SELECT right(issue_d, 4) AS year, count(*) AS n, avg(CASE WHEN loan_status IN :bad THEN 1.0 ELSE 0.0 END) AS bad_rate 
    FROM raw_loans WHERE loan_status IN :st GROUP BY 1 ORDER BY 1
""")
with engine.connect() as conn:
    print(pd.read_sql(q, conn, params={"st": statuses, "bad": bad}).round(3))
# issue_id is text such as Dec-2025, so right(issue_d, 4 extracts the year)
# :st and :bad are parameters: we pass the values separately instead of gluing them into the SQL text, which is the safe way to avoid SQL injection
# If Bad Rate drifts a lot, a model traind on old years may not fit new ones.


# Sample Finished Loans and Compute single-column AUC
sample_q = text("SELECT * FROM raw_loans WHERE loan_status IN :st ORDER BY random() Limit 100000")
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
                 "missing_pct": round(df[col].isna().mean()*100, 1)})

    
audit = pd.DataFrame(rows).sort_values("strength", ascending=False)
"""Pull a random 100,000 finished loans and create the 0/1 label 'bad'. 
For each numberic column, fill missing values with the column median (the audit only needs a rough ranking). Skip rows with a single value, and compulte the AUC of that column against bad.
AUC 0.5 means no power; 0.0 or 1.0 means a perfect (a column hat is perfectly reversed is just as informative).
So we report strength = |AUC - 0.5| x 2, which runs from 0(useless) to 1(perfect), whichever direction it points.
"""

print("\n ==Single-column strength (AUC distance from 0.5, scaled 0-1 ==)")
print(audit.to_string(index=False))
audit.to_csv(REPORTS/ "leakage_audit.csv", index=False)



ALLOWED = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high", "emp_length", "home_ownership", "verification_status", "purpose", "earliest_cr_line", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "delinq_2yrs"]
BANNED = ["total_pymnt", "total_rec_prncp", "recoveries", "last_fico_range_high", "out_prncp", "int_rate", "grade", "installment"]
print("\nAllowed:", ALLOWED)
print("Banned :", BANNED)
