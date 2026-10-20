import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from config import DATA_PROCESSED, REPORTS, RANDOM_SEED
from features import to_matrix
from scoring import get_scorer

sc = get_scorer()
test = pd.read_parquet(DATA_PROCESSED / "test.parquet").sample(3000, random_state=RANDOM_SEED)
X = to_matrix(test)

explainer = shap.TreeExplainer(sc.booster)
sv = explainer.shap_values(X)

contrib = sc.booster.predict(xgb.DMatrix(X, feature_names=sc.columns), pred_contribs=True)[:, :-1]
print("pred_contribs equals SHAP values:", np.allclose(sv, contrib, atol=1e-4))

imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
print("\nTop features by mean |SHAP|:\n", imp.head(10).round(3).to_string())
plt.figure(); shap.summary_plot(sv, X, show=False, max_display=12)
plt.tight_layout(); plt.savefig(REPORTS / "shap_beeswarm.png", dpi=120); plt.close()

test["pd"] = sc.predict_pd(test)
worst = test.sort_values("pd", ascending=False).head(3)
for (_, row), reasons in zip(worst.iterrows(), sc.reasons(worst)):
    print(f"\nPD = {row['pd']:.2f}  (actual bad = {row['bad']})")
    for r in reasons:
        print("   -", r["text"], f"(impact {r['impact']})")
