import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, ARTIFACTS, REPORTS, PROFIT_RATE_GOOD, LGD
from scoring import get_scorer

sc = get_scorer()
valid = pd.read_parquet(DATA_PROCESSED / "valid.parquet")
test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
valid["pd"] = sc.predict_pd(valid)
test["pd"] = sc.predict_pd(test)

def profit_table(df, cutoffs):
    value = np.where(df["bad"] == 1, -LGD * df["loan_amnt"], PROFIT_RATE_GOOD * df["loan_amnt"])
    rows = []
    for c in cutoffs:
        ok = df["pd"] <= c
        rows.append({"cutoff": round(c, 3), "approval_rate": ok.mean(),
                     "bad_rate_approved": df.loc[ok, "bad"].mean() if ok.any() else np.nan,
                     "profit": value[ok.values].sum()})
    return pd.DataFrame(rows)

cutoffs = np.arange(0.03, 0.60, 0.01)
pv = profit_table(valid, cutoffs)
best = pv.loc[pv["profit"].idxmax(), "cutoff"]
print(f"Best cut-off on validation: PD <= {best:.2f}")

pt = profit_table(test, [best, 1.0])
print("\nTEST SET  (profit in the loan currency units)")
print(pt.rename(index={0: "model cut-off", 1: "approve everyone"}).round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pv["cutoff"], pv["profit"] / 1e6, label="validation")
ptest = profit_table(test, cutoffs)
ax.plot(ptest["cutoff"], ptest["profit"] / 1e6, label="test")
ax.axvline(best, ls="--", color="grey")
ax.set_xlabel("Cut-off (approve if PD <= x)"); ax.set_ylabel("Profit (millions)")
ax.set_title("Profit vs cut-off"); ax.legend()
fig.tight_layout(); fig.savefig(REPORTS / "profit_curve.png", dpi=120)

meta = json.load(open(ARTIFACTS / "meta.json"))
meta["cutoff_pd"] = float(best)
json.dump(meta, open(ARTIFACTS / "meta.json", "w"), indent=2)
print("Saved cutoff_pd =", best)
