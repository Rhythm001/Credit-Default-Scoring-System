import numpy as np
import pandas as pd
from config import DATA_RAW, RANDOM_SEED
rng = np.random.default_rng(RANDOM_SEED)
N = 60_000
months = pd.period_range("2012-01", "2016-12", freq="M")
issue = rng.choice(months.to_timestamp(), size=N)
fico = np.clip(rng.normal(700, 35, N), 660, 845).round(-0)
dti = np.clip(rng.gamma(4, 4.5, N), 0, 60)
income = np.exp(rng.normal(11.1, 0.55, N)).round(-2)
amount = (rng.choice(np.arange(1, 36), N) * 1000).astype(float)
inq = rng.poisson(0.8, N)
revol_util = np.clip(rng.normal(55, 24, N), 0, 120)
history = np.clip(rng.normal(16, 7, N), 1, 50)
pubrec = rng.poisson(0.15, N)
delinq = rng.poisson(0.3, N)
purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "other", "car"], N, p=[0.55, 0.2, 0.08, 0.04, 0.09, 0.04])
z = (-1.6 - 0.018 * (fico - 700) + 0.035 * (dti - 18) - 0.45 * (np.log(income) - 11.1)
+ 0.12 * inq + 0.008 * (revol_util - 55) + 0.2 * pubrec + 0.1 * delinq + 0.5 * (purpose == "small_business") + 0.00002 * (amount - 14000))
months_since_start = (issue - np.datetime64("2012-01-01")).astype("timedelta64[D]").astype(int) / 30
z = z + 0.004 * np.clip(months_since_start - 40, 0, None) # mild drift late in the data
p_default = 1 / (1 + np.exp(-z))
bad = rng.random(N) < p_default
issue_s = pd.to_datetime(issue).strftime("%b-%Y")
status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
status[pd.to_datetime(issue) >= "2016-01-01"] = "Current"
term = np.where(rng.random(N) < 0.25, " 60 months", " 36 months")
int_rate = np.clip(6 + 22 * p_default + rng.normal(0, 1.5, N), 5.3, 30.9).round(2)
grade = pd.cut(int_rate, [0, 8, 11, 14, 17, 21, 25, 40], labels=list("ABCDEFG")).astype(str)
total_pymnt = np.where(bad, amount * rng.uniform(0.2, 0.7, N), amount * (1 + 0.12 * rng.uniform(0.5, 1.3, N))).round(2)
recoveries = np.where(bad, amount * rng.uniform(0, 0.1, N), 0).round(2)
df = pd.DataFrame({ "id": np.arange(1_000_000, 1_000_000 + N).astype(str), "loan_amnt": amount, "term": term, "int_rate": int_rate, "installment": (amount / 36 * 1.15).round(2), "grade": grade, "emp_length": rng.choice(["< 1 year", "1 year", "3 years", "5 years", "10+ years", None], N), "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], N, p=[0.45, 0.45, 0.10]), "annual_inc": income, "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], N), "issue_d": issue_s, "loan_status": status, "purpose": purpose, "dti": dti.round(2), "delinq_2yrs": delinq,
"earliest_cr_line": [(pd.Timestamp(i) - pd.DateOffset(years=int(h))).strftime("%b-%Y") for i, h in zip(issue,
history)], "fico_range_low": fico, "fico_range_high": fico + 4, "inq_last_6mths": inq, "open_acc": rng.poisson(11, N), "pub_rec": pubrec, "revol_bal": np.exp(rng.normal(9.3, 1.0, N)).round(0), "revol_util": revol_util.round(1), "total_acc": rng.poisson(24, N), "mort_acc": rng.poisson(1.2, N), "total_pymnt": total_pymnt, "recoveries": recoveries, "last_fico_range_high": np.where(bad, fico - rng.uniform(20, 120, N), fico + rng.uniform(-10, 20, N)).round(0), "out_prncp": 0.0,
})
out = DATA_RAW / "sample_loans.csv.gz"
df.to_csv(out, index=False)
print("Wrote", out, df.shape)