# Credit Default Scoring System
![CI](https://github.com/Rhythm001/Credit-Default-Scoring-System/actions/workflows/ci.yml/badge.svg)


An end-to-end credit-risk project: predict the probability that a borrower defaults, turn it into an explainable scorecard, serve it through an API, and monitor it for drift.

Built as a hands-on learning project on the public Lending Club loan data.

## Status

All 11 steps are complete and run end to end.

- [x] Step 0: Download the data
- [x] Step 1: Load 2.26M loans into PostgreSQL
- [x] Step 2: Explore the data and audit for leakage
- [x] Step 3: Features and a time-based train/validation/test split
- [x] Step 4: WoE scorecard baseline (logistic regression)
- [x] Step 5: XGBoost and probability calibration
- [x] Step 6: Profit-based approval cut-off
- [x] Step 7: SHAP explanations and reason codes
- [x] Step 8: FastAPI scoring service with an audit log
- [x] Step 9: Docker and docker compose
- [x] Step 10: Drift monitoring with PSI
- [x] Step 11: Airflow DAG for scheduled monitoring

## Results (real Lending Club data)

Modelling set: about 589K finished 36-month loans issued 2012 to 2015, split by issue date so the test set is the future relative to training.

| Split | Period | Rows | Bad rate |
|---|---|---|---|
| Train | 2012-01 to 2014-12 | 306,462 | 13.2% |
| Validation | 2015-01 to 2015-06 | 120,790 | 15.1% |
| Test | 2015-07 to 2015-12 | 162,236 | 14.7% |

| Model (test set) | AUC | Gini | KS |
|---|---|---|---|
| WoE scorecard (logistic regression) | 0.654 | 0.31 | 0.22 |
| XGBoost, calibrated | 0.676 | 0.35 | 0.25 |

Lender-assigned `grade` and `int_rate` are deliberately excluded. They are the lender's own risk opinion and would make the model circular. That is why discrimination is modest, and it is typical for this data.

**Calibration.** Predicted default rates track actual rates closely across all ten risk deciles on the test set. A class-weighted XGBoost model with the same ranking power predicted a mean PD of 0.46 against an actual 0.147, which is why the final model is trained without class weights and then calibrated.

**Approval cut-off.** Chosen on the validation set (PD <= 0.18, close to the theoretical break-even PD of 0.1875) and reported on the untouched test set:

| Policy | Approval rate | Bad rate of approved | Profit (assumed economics) |
|---|---|---|---|
| Model cut-off 0.18 | 71.4% | 10.7% | 102.4M |
| Approve everyone | 100% | 14.7% | 74.3M |

Profit uses assumed figures (15% earned on a repaid loan, 65% lost on a default), not Lending Club's real cash flows. Treat it as an illustration of the method.

**Monitoring.** The drift demo uses simulated traffic with drift injected on purpose, to show the PSI alarm works. It is not a finding about real model drift.

## What this project covers

- **Credit-risk modelling:** probability of default, WoE/IV scorecards, Gini and KS, calibration
- **Model validity:** time-based splits, leakage audit, handling loans that have not finished their term
- **Explainability:** SHAP and per-applicant reason codes
- **Engineering:** PostgreSQL, FastAPI, Docker, Airflow, tests

## Tech stack

Python 3.12, pandas, scikit-learn, XGBoost, SHAP, PostgreSQL 16, SQLAlchemy, FastAPI, Docker, Apache Airflow, pytest.

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
```

## Run the pipeline

```powershell
python .\00_download_data.py            # about 1.3 GB; Kaggle may ask for an API token
python .\01_load_to_postgres.py         # raw loans into PostgreSQL
python .\02_explore_and_leakage_audit.py
python .\03_build_dataset.py            # features + time-based split (parquet)
python .\04_scorecard.py                # WoE/IV scorecard
python .\05_train_xgb.py                # XGBoost + calibration, saves artifacts/
python .\06_cutoff.py                   # profit-based cut-off
python .\07_shap.py                     # SHAP + reason codes
python -m pytest -q                     # tests
```

Check the load:

```powershell
docker compose exec db psql -U credit -d creditdb -c "SELECT count(*) FROM raw_loans;"
```

## Serve the model

Locally:

```powershell
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs and try `POST /score`. Every request is written to the `score_log` table as an audit trail.

In Docker (API + database):

```powershell
docker compose up --build
```

## Monitoring

```powershell
python .\08_simulate_traffic.py         # inserts simulated scores, with drift in the last days
python -c "import json; from monitoring import run_monitoring; print(json.dumps(run_monitoring(7), indent=1))"
```

PSI per feature is stored in `monitoring_psi` (below 0.10 stable, 0.10 to 0.25 watch, above 0.25 shifted). To run it on a schedule with Airflow:

```powershell
docker compose -f docker-compose.airflow.yml up
```

Airflow standalone with SQLite is for local development only. The admin password is generated on first start, so do not commit it anywhere.

## Project layout

```
00_download_data.py          download the Kaggle data into data/raw
01_load_to_postgres.py       load selected columns into PostgreSQL in chunks
02_explore_and_leakage_audit.py
03_build_dataset.py          features and time-based split
04_scorecard.py              WoE / IV / logistic scorecard
05_train_xgb.py              XGBoost, calibration, scoreboard
06_cutoff.py                 profit-based approval cut-off
07_shap.py                   SHAP and reason codes
08_simulate_traffic.py       simulated scoring traffic for monitoring
features.py, calibration.py, metrics.py, scoring.py, monitoring.py
config.py, db.py             settings and database helpers
make_sample_data.py          synthetic test data
app/                         FastAPI service
dags/                        Airflow DAG
tests/                       pytest suite
Dockerfile, docker-compose*.yml
```

The local database password (`credit`) is a demo credential for a local container only.

## Known limitations

- **Selection bias.** Trained only on accepted loans, so it cannot judge applicants the lender rejected. No reject inference was done.
- **Weak signal by design.** With the lender's grade and rate excluded, Gini is about 0.35. Real lenders add bureau and behavioural data.
- **Assumed economics.** The profit figures use assumed gain and loss rates.
- **36-month loans only**, to avoid censored outcomes.
- **Drift is detected on inputs only.** Real default outcomes arrive about 3 years later, so performance decay cannot be measured in time.
- **Not tested for fairness** across protected groups.
- **Reason codes** are generated from SHAP and are not reviewed for regulatory wording.
- Educational project. Not for real lending decisions.