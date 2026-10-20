import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from config import DATA_PROCESSED, RANDOM_SEED
from db import get_engine, init_tables
from scoring import get_scorer, INPUT_FIELDS

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=7)
ap.add_argument("--per-day", type=int, default=400)
args = ap.parse_args()

rng = np.random.default_rng(RANDOM_SEED)
sc, engine = get_scorer(), get_engine()
init_tables(engine)
pool = pd.read_parquet(DATA_PROCESSED / "test.parquet")
with engine.begin() as conn:
    conn.execute(text("DELETE FROM score_log WHERE request_id LIKE 'sim-%'"))

now = datetime.now(timezone.utc)
rows = []
for k in range(args.days):
    day = args.days - 1 - k
    drift = max(0, k - (args.days - 4))
    batch = pool.sample(args.per_day, replace=True, random_state=int(rng.integers(1e9))).copy()
    batch["fico_score"] = (batch["fico_score"] - 20 * drift).clip(lower=300)
    batch["dti"] = batch["dti"] + 4 * drift
    batch["pd"] = sc.predict_pd(batch)
    for i, (_, r) in enumerate(batch.iterrows()):
        feats = {f: (None if pd.isna(r[f]) else (r[f] if isinstance(r[f], str) else float(r[f]))) for f in INPUT_FIELDS}
        rows.append({"rid": f"sim-{day}-{i}", "ts": now - timedelta(days=day, minutes=int(rng.integers(0, 1400))),
                     "ver": sc.meta["model_version"], "pd": float(r["pd"]), "dec": sc.decide(float(r["pd"])),
                     "feat": json.dumps(feats)})

with engine.begin() as conn:
    conn.execute(text("""INSERT INTO score_log (request_id, created_at, model_version, pd, decision, features)
                         VALUES (:rid, :ts, :ver, :pd, :dec, CAST(:feat AS jsonb))"""), rows)
print(f"Inserted {len(rows):,} simulated scores over {args.days} days (drift in the last 3 days)")
