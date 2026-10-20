import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from config import DATA_PROCESSED, ARTIFACTS, REPORTS
from features import engineer, NUMERIC, CATEGORICAL
from metrics import gini, ks

train, valid, test = [engineer(pd.read_parquet(DATA_PROCESSED / f"{n}.parquet")) for n in ("train", "valid", "test")]
FEATURES = NUMERIC + list(CATEGORICAL)
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

def fit_edges(s):
    return sorted(set(np.nanquantile(s, np.linspace(0, 1, 11)[1:-1]).round(4)))

def apply_bins(s, f, edges):
    if f in CATEGORICAL:
        return s.astype(object).where(s.notna(), "missing").astype(str)
    cut = pd.cut(s, [-np.inf] + edges + [np.inf])
    return cut.astype(object).where(cut.notna(), "missing").astype(str)

EDGES = {f: fit_edges(train[f]) for f in NUMERIC}

def woe_table(f):
    b = apply_bins(train[f], f, EDGES.get(f))
    t = pd.crosstab(b, train["bad"]).rename(columns={0: "good", 1: "bad"})
    t = t.reindex(columns=["good", "bad"], fill_value=0)
    dg = (t["good"] + 0.5) / t["good"].sum()
    db = (t["bad"] + 0.5) / t["bad"].sum()
    t["woe"] = np.log(dg / db)
    t["iv_part"] = (dg - db) * t["woe"]
    t["bad_rate"] = t["bad"] / (t["good"] + t["bad"])
    return t

TABLES = {f: woe_table(f) for f in FEATURES}
iv = pd.Series({f: t["iv_part"].sum() for f, t in TABLES.items()}).sort_values(ascending=False)
print("Information value per feature:\n", iv.round(3).to_string())
iv.to_csv(REPORTS / "iv_table.csv", header=["iv"])

SELECTED = [f for f in FEATURES if iv[f] >= 0.02]
print("\nSelected:", SELECTED)

def to_woe(df):
    out = pd.DataFrame(index=df.index)
    for f in SELECTED:
        b = apply_bins(df[f], f, EDGES.get(f))
        out[f] = b.map(TABLES[f]["woe"]).fillna(0.0)
    return out

Xtr = to_woe(train)
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(Xtr, 1 - train["bad"])
print("\nCoefficients (all should be positive):")
print(pd.Series(lr.coef_[0], index=SELECTED).round(3).to_string())

def score(df):
    logodds = lr.intercept_[0] + to_woe(df).values @ lr.coef_[0]
    return OFFSET + FACTOR * logodds

rows = []
n_feat = len(SELECTED)
for i, f in enumerate(SELECTED):
    for label, r in TABLES[f].iterrows():
        pts = FACTOR * lr.coef_[0][i] * r["woe"] + (OFFSET + FACTOR * lr.intercept_[0]) / n_feat
        rows.append({"feature": f, "bin": label, "woe": round(r["woe"], 3),
                     "bad_rate": round(r["bad_rate"], 3), "points": round(pts, 1)})
pd.DataFrame(rows).to_csv(REPORTS / "scorecard_table.csv", index=False)

print("\nPerformance (higher is better)")
for name, d in [("train", train), ("valid", valid), ("test", test)]:
    s = score(d)
    print(f"{name:5s} AUC={(gini(d['bad'], -s) + 1) / 2:.3f}  Gini={gini(d['bad'], -s):.3f}  KS={ks(d['bad'], -s):.3f}")

joblib.dump({"edges": EDGES, "woe": {f: TABLES[f]["woe"].to_dict() for f in SELECTED}, "selected": SELECTED,
             "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
             "factor": FACTOR, "offset": OFFSET}, ARTIFACTS / "scorecard.joblib")
st = score(test)
json.dump({"scorecard_test_auc": (gini(test["bad"], -st) + 1) / 2, "scorecard_test_ks": ks(test["bad"], -st)},
          open(REPORTS / "scorecard_metrics.json", "w"))
