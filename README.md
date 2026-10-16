# Credit Default Scoring System

An end-to-end credit-risk project: predict the probability that a borrower defaults, turn it into an explainable scorecard, serve it through an API, and monitor it for drift.

Built as a hands-on learning project on the public Lending Club loan data.

## Status

Work in progress. Steps are committed as they are completed.

- [x] Step 0: Download the data
- [x] Step 1: Load 2.26M loans into PostgreSQL
- [x] Step 2: Explore the data and audit for leakage
- [ ] Step 3: Features and a time-based train/validation/test split
- [ ] Step 4: WoE scorecard baseline (logistic regression)
- [ ] Step 5: Monotone-constrained XGBoost and probability calibration
- [ ] Step 6: Profit-based approval cut-off
- [ ] Step 7: SHAP explanations and reason codes
- [ ] Step 8: FastAPI scoring service with an audit log
- [ ] Step 9: Docker and docker compose
- [ ] Step 10: Drift monitoring with PSI
- [ ] Step 11: Airflow DAG for daily monitoring

## What this project covers

- **Credit-risk modelling:** probability of default, WoE/IV scorecards, Gini and KS, calibration
- **Model validity:** time-based splits, leakage audit, handling loans that have not finished their term
- **Explainability:** SHAP and per-applicant reason codes
- **Engineering:** PostgreSQL, FastAPI, Docker, Airflow

## Tech stack

Python 3.12, pandas, scikit-learn, XGBoost, SHAP, PostgreSQL 16, SQLAlchemy, FastAPI, Docker, Apache Airflow.

## Data

Lending Club accepted loans (2007 to 2018 Q4), from Kaggle:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

The data is **not included in this repository**. The Kaggle page states no licence, so it is not redistributed here. `00_download_data.py` downloads it for you.

`make_sample_data.py` generates a small synthetic dataset with the same shape, for testing the code without the real file.

## Setup (Windows, PowerShell)

Requirements: Python 3.12, Git, Docker Desktop (with the engine running).

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt

docker volume create credit_pgdata
docker compose up -d db

python .\00_download_data.py
python .\01_load_to_postgres.py
```

The first download is about 1.3 GB. Kaggle may ask for an API token the first time.

Check the load:

```powershell
docker compose exec db psql -U credit -d creditdb -c "SELECT count(*) FROM raw_loans;"
```

## Project layout

```
00_download_data.py      download the Kaggle data into data/raw
01_load_to_postgres.py   load selected columns into PostgreSQL in chunks
config.py, db.py         paths and database helpers
make_sample_data.py      synthetic test data
app/                     FastAPI service (planned)
dags/                    Airflow DAG (planned)
Dockerfile, docker-compose*.yml
```

The local database password (`credit`) is a demo credential for a local container only.

## Known limitations (stated upfront)

- Trained only on **accepted** loans, so it cannot judge applicants the lender rejected (selection bias; no reject inference).
- Approval economics will use assumed gain and loss figures, not Lending Club's real cash flows.
- Only 36-month loans will be used, to avoid censored outcomes.
- Educational project. Not for real lending decisions.