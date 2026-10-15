import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "artifacts"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
for _p in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://credit:credit@localhost:5432/creditdb",
)


# What counts as good and bad
GOOD_STATUSES = ["Fully Paid"]
BAD_STATUSES = ["Charged Off", "Default"]


# Scope
TERM_MONTHS = 36
START_DATE = '2012-01-01'
# We only use 36-month loans issued from 2012. The data ends in late 2018. A 36-month load issued in 2015 has had its full life by them, so its outcome is known.


# The time split
TRAIN_END = "2014-12-31"
VALID_END = "2015-06-30"
TEST_END = "2015-12-31"


# Business Assumptions
PROFIT_RATE_GOOD = 0.15
LGD = 0.65
# used for the cut-off decision. These are *assumptions* fr learning. A good loan earns about 15% of the principal; a bad loan loses about 65% of principal (loss given default, LGD)


# Repeteability
RANDOM_SEED = 42
MODEL_VERSION = "xgb-v1"
# A fixed random seed makes your results repeatable.


