import numpy as np
import pandas as pd
from sqlalchemy import text
from config import ARTIFACTS
from db import get_engine
from features import engineer

MONITORED = ["pd", "fico_score", "dti", "annual_inc", "loan_amnt", "revol_util", "loan_to_income"]

def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().to_numpy(float)
    actual = pd.Series(actual).dropna().to_numpy(float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

def status(value):
    return "stable" if value < 0.10 else "watch" if value < 0.25 else "shifted"

def run_monitoring(window_days=7, min_rows=200, engine=None):
    engine = engine or get_engine()
    base = engineer(pd.read_csv(ARTIFACTS / "baseline.csv.gz"))
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT features, pd FROM score_log "
            "WHERE created_at >= now() - make_interval(days => :d)"), {"d": window_days}).fetchall()
        if len(rows) < min_rows:
            return {"status": "insufficient_data", "n_rows": len(rows), "max_psi": 0.0, "worst_feature": None}
        live = pd.DataFrame([r[0] for r in rows])
        live["pd"] = [r[1] for r in rows]
        live = engineer(live)
        results = [{"w": window_days, "n": len(live), "f": f, "p": psi(base[f], live[f]),
                    "s": status(psi(base[f], live[f]))} for f in MONITORED]
        conn.execute(text("INSERT INTO monitoring_psi (window_days, n_rows, feature, psi, status) "
                          "VALUES (:w, :n, :f, :p, :s)"), results)
    worst = max(results, key=lambda r: r["p"])
    return {"status": worst["s"], "n_rows": len(live), "max_psi": round(worst["p"], 4),
            "worst_feature": worst["f"], "detail": {r["f"]: round(r["p"], 4) for r in results}}
