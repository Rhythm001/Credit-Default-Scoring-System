import pandas as pd
from sqlalchemy import text
from config import (GOOD_STATUSES, BAD_STATUSES, TERM_MONTHS, START_DATE, TRAIN_END, VALID_END,
                    TEST_END, DATA_PROCESSED)
from db import get_engine
from features import parse_raw

engine = get_engine()
q = text("""
    SELECT * FROM raw_loans
    WHERE loan_status IN :st AND trim(term) = :term
""")
with engine.connect() as conn:
    raw = pd.read_sql(q, conn, params={"st": tuple(GOOD_STATUSES + BAD_STATUSES), "term": f"{TERM_MONTHS} months"})
print("finished loans pulled:", len(raw))

feat = parse_raw(raw)
feat["bad"] = raw["loan_status"].isin(BAD_STATUSES).astype(int)
feat["loan_id"] = raw["id"].values

feat = feat[(feat["issue_date"] >= START_DATE) & (feat["issue_date"] <= TEST_END)]
train = feat[feat["issue_date"] <= TRAIN_END]
valid = feat[(feat["issue_date"] > TRAIN_END) & (feat["issue_date"] <= VALID_END)]
test = feat[(feat["issue_date"] > VALID_END) & (feat["issue_date"] <= TEST_END)]

for name, part in [("train", train), ("valid", valid), ("test", test)]:
    part.to_parquet(DATA_PROCESSED / f"{name}.parquet", index=False)
    print(f"{name:5s} rows={len(part):>8,}  bad_rate={part['bad'].mean():.3f}  "
          f"{part['issue_date'].min():%Y-%m} -> {part['issue_date'].max():%Y-%m}")
