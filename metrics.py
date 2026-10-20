import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def gini(y, risk):
    return 2 * roc_auc_score(y, risk) - 1

def ks(y, risk):
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))

def calibration_table(y, pd_pred, bins=10):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_pred)})
    df["bucket"] = pd.qcut(df["p"], bins, duplicates="drop")
    t = df.groupby("bucket", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t.reset_index(drop=True).round(4)
