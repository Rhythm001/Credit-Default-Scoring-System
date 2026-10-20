import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from config import DATA_PROCESSED, ARTIFACTS, REPORTS, RANDOM_SEED, MODEL_VERSION
from calibration import PlattCalibrator
from features import to_matrix, feature_columns, monotone_tuple
from metrics import ks, calibration_table

train, valid, test = [pd.read_parquet(DATA_PROCESSED / f"{n}.parquet") for n in ("train", "valid", "test")]
Xtr, ytr = to_matrix(train), train["bad"]
Xva, yva = to_matrix(valid), valid["bad"]
Xte, yte = to_matrix(test), test["bad"]

def train_model(scale_pos_weight=1.0):
    m = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist",
        monotone_constraints=monotone_tuple(), scale_pos_weight=scale_pos_weight,
        eval_metric="auc", early_stopping_rounds=50, random_state=RANDOM_SEED, n_jobs=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m

plain = train_model(1.0)
neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
weighted = train_model(neg / pos)
print("Trees used: plain =", plain.best_iteration + 1, " weighted =", weighted.best_iteration + 1)
print(f"{'model':10s} {'AUC':>6s} {'Brier':>7s} {'mean predicted PD':>18s} (actual bad rate {yte.mean():.3f})")
for name, m in [("plain", plain), ("weighted", weighted)]:
    p = m.predict_proba(Xte)[:, 1]
    print(f"{name:10s} {roc_auc_score(yte, p):6.3f} {brier_score_loss(yte, p):7.4f} {p.mean():18.3f}")

model = plain
raw_va = model.predict_proba(Xva)[:, 1]
calibrator = PlattCalibrator().fit(raw_va, yva)

raw_te = model.predict_proba(Xte)[:, 1]
cal_te = calibrator.predict(raw_te)
print("\nTEST SET")
print(f"AUC {roc_auc_score(yte, cal_te):.3f} | PR-AUC {average_precision_score(yte, cal_te):.3f} | "
      f"KS {ks(yte, cal_te):.3f} | Brier raw {brier_score_loss(yte, raw_te):.4f} -> calibrated {brier_score_loss(yte, cal_te):.4f}")
print(calibration_table(yte, cal_te).to_string())
sc = json.load(open(REPORTS / "scorecard_metrics.json"))
print(f"\nScorecard test AUC {sc['scorecard_test_auc']:.3f}  vs  XGBoost test AUC {roc_auc_score(yte, cal_te):.3f}")

model.get_booster()[: model.best_iteration + 1].save_model(str(ARTIFACTS / "xgb_model.json"))
joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")
json.dump({"model_version": MODEL_VERSION, "feature_columns": feature_columns(), "cutoff_pd": 0.20,
           "test_auc": float(roc_auc_score(yte, cal_te)), "test_ks": float(ks(yte, cal_te))},
          open(ARTIFACTS / "meta.json", "w"), indent=2)
base = valid.sample(min(5000, len(valid)), random_state=RANDOM_SEED).copy()
base["pd"] = calibrator.predict(model.predict_proba(to_matrix(base))[:, 1])
base.to_csv(ARTIFACTS / "baseline.csv.gz", index=False)
print("\nSaved model, calibrator, meta and baseline to artifacts/")
