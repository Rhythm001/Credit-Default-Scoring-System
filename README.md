# Credit Default Scoring System

![CI](https://github.com/Rhythm001/Credit-Default-Scoring-System/actions/workflows/ci.yml/badge.svg)

This project estimates the probability of default (PD) of a borrower, turns it into an explainable WoE scorecard, serves it through an API, and monitors it for drift with the Population Stability Index (PSI). I built it on the public Lending Club loan data to learn how credit-risk models are developed, validated and operated, rather than only trained.

## Overview

The pipeline has eleven steps. Each one is a script or module in this repository.

1. Download the Lending Club data and load 2.26 million loans into PostgreSQL.
2. Explore the data and run a leakage audit, which removes columns that are only known after a loan has already gone bad.
3. Build features and split the data by issue date into training, validation and test sets (a time-based split).
4. Fit a Weight-of-Evidence (WoE) scorecard, using Information Value (IV) to select features and logistic regression as the interpretable baseline.
5. Train an XGBoost model and calibrate its probabilities.
6. Choose an approval cut-off based on expected profit.
7. Explain individual decisions with SHAP and turn them into reason codes.
8. Serve the model through a FastAPI service that logs every request.
9. Package the service and database with Docker.
10. Monitor feature and score drift with PSI.
11. Schedule the monitoring job with an Airflow DAG.

## Key terms

I wrote these for anyone new to credit risk, myself included when I started.

| Term | Meaning in this project |
|---|---|
| **Default / "bad" loan** | A loan that ended as Charged Off. A "good" loan is one that was Fully Paid. |
| **Bad rate** | The share of loans that defaulted. Here it is about 13 to 15%. |
| **PD (probability of default)** | The model's estimate, between 0 and 1, that a given borrower will default. |
| **FICO score** | A US credit score from roughly 300 to 850. Higher means a safer borrower. |
| **DTI (debt-to-income)** | Monthly debt payments divided by monthly income. Higher means a more stretched borrower. |
| **Revolving utilisation** | The share of available credit-card limit that is in use. |
| **Grade and interest rate** | The lender's own risk label and the rate it charged. Excluded here, because they are the lender's opinion of the answer. |
| **Leakage** | Information that would not exist at the time of the decision, such as payments received after the loan started. It makes a model look better than it is. |
| **WoE (weight of evidence)** | For a group of borrowers, the log of (share of all good loans in the group) divided by (share of all bad loans in the group). Positive means safer than average. |
| **IV (information value)** | A single score for how well one feature separates good from bad. Below 0.02 is useless, 0.1 to 0.3 is medium, above 0.3 is strong. |
| **Scorecard** | A logistic regression on WoE values, converted into points, so that each feature adds or subtracts a readable number of points. |
| **AUC** | The chance that a randomly chosen bad borrower gets a higher PD than a randomly chosen good one. 0.5 is a coin flip. |
| **Gini** | 2 × AUC − 1. It rescales AUC so that 0 is useless and 1 is perfect. |
| **KS (Kolmogorov-Smirnov)** | The largest gap between the cumulative share of good and bad borrowers when sorted by score. Higher means better separation. |
| **Calibration** | Whether predicted probabilities match reality: of loans given PD 0.20, about 20% should actually default. This is a different property from ranking well. |
| **Cut-off** | The PD above which an application is declined. |
| **Break-even PD** | The PD at which expected gain equals expected loss. With gain g and loss L it is g ÷ (g + L), which is 0.15 ÷ 0.80 = 0.1875 here. |
| **Selection bias / reject inference** | The model only sees applicants the lender accepted. Reject inference tries to estimate how rejected applicants would have behaved. I did not attempt it. |
| **Censored outcome** | A loan that has not finished its term, so its final result is not yet known. Only 36-month loans old enough to have finished are used. |
| **SHAP and reason codes** | SHAP splits a single prediction into the contribution of each feature. Reason codes turn the largest contributions into a statement such as "high debt-to-income ratio". |
| **PSI (population stability index)** | A measure of how much a feature's distribution has moved since training. Below 0.10 is stable, 0.10 to 0.25 is worth watching, above 0.25 is a shift. |

## Results

The modelling set contains about 589,000 finished 36-month loans issued between 2012 and 2015. I split it by issue date, so the test set always lies in the future relative to the data the model was trained on.

| Split | Period | Rows | Bad rate |
|---|---|---|---|
| Train | 2012-01 to 2014-12 | 306,462 | 13.2% |
| Validation | 2015-01 to 2015-06 | 120,790 | 15.1% |
| Test | 2015-07 to 2015-12 | 162,236 | 14.7% |

| Model (test set) | AUC | Gini | KS |
|---|---|---|---|
| WoE scorecard (logistic regression) | 0.654 | 0.31 | 0.22 |
| XGBoost, calibrated | 0.676 | 0.35 | 0.25 |

I excluded the lender's own `grade` and `int_rate` fields. They already contain the lender's opinion of each borrower's risk, so using them would make the model partly circular. Without them, discrimination is modest, which is what I would expect from application data alone.

**Calibration.** On the test set, predicted default rates stay close to actual rates in each of the ten risk deciles. For comparison, a class-weighted XGBoost model ranked borrowers about as well but predicted an average PD of 0.46 against an actual rate of 0.147. This is why the final model is trained without class weights and then calibrated.

**Approval cut-off.** I chose the cut-off on the validation set, at a PD of 0.18, which is close to the break-even PD of 0.1875. I then measured it once on the untouched test set.

| Policy | Approval rate | Bad rate among approved | Profit (assumed economics) |
|---|---|---|---|
| Model cut-off at 0.18 | 71.4% | 10.7% | 102.4M |
| Approve everyone | 100% | 14.7% | 74.3M |

The profit figures rest on assumptions: 15% earned on a repaid loan and 65% lost on a default (the loss given default, LGD). They are not Lending Club's actual cash flows, so they illustrate the method rather than predict real earnings.

**Monitoring.** The drift demonstration uses simulated traffic in which drift was introduced on purpose. It shows that the PSI alerts work. It is not evidence that the model drifted on real data.

## Technology

Python 3.12, pandas, scikit-learn, XGBoost, SHAP, PostgreSQL 16, SQLAlchemy, FastAPI, Docker, Apache Airflow and pytest. The tests run automatically on every push through GitHub Actions.

## Data

The project uses the accepted-loans file (2007 to 2018 Q4) of the Lending Club dataset on Kaggle:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

The data is not included in this repository, because the Kaggle page does not state a licence that permits redistribution. The script `00_download_data.py` downloads it. For testing the code without the real file, `make_sample_data.py` generates a small synthetic dataset with the same structure.

## Getting started

These instructions are for Windows with PowerShell. You will need Python 3.12, Git and Docker Desktop, with the Docker engine running.

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt

docker volume create credit_pgdata
docker compose up -d db
```

## Running the pipeline

```powershell
python .\00_download_data.py            # about 1.3 GB; Kaggle may ask for an API token
python .\01_load_to_postgres.py         # loads the raw loans into PostgreSQL
python .\02_explore_and_leakage_audit.py
python .\03_build_dataset.py            # features and time-based split
python .\04_scorecard.py                # WoE / IV scorecard
python .\05_train_xgb.py                # XGBoost and calibration; saves artifacts/
python .\06_cutoff.py                   # profit-based cut-off
python .\07_shap.py                     # SHAP and reason codes
python -m pytest -q                     # tests
```

To confirm that the load worked:

```powershell
docker compose exec db psql -U credit -d creditdb -c "SELECT count(*) FROM raw_loans;"
```

## Serving the model

To run the API locally:

```powershell
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs and try `POST /score`. Every request is stored in the `score_log` table, which serves as an audit trail.

To run the API and the database together in Docker:

```powershell
docker compose up --build
```

The trained model files in `artifacts/` are committed to the repository, so the API and its tests work on a fresh clone without retraining.

## Monitoring

```powershell
python .\08_simulate_traffic.py         # inserts simulated scores, with drift in the last days
python -c "import json; from monitoring import run_monitoring; print(json.dumps(run_monitoring(7), indent=1))"
```

The PSI of each feature is stored in the `monitoring_psi` table. To run the check on a schedule with Airflow:

```powershell
docker compose -f docker-compose.airflow.yml up
```

Airflow runs here in standalone mode with SQLite, which is suitable for local development only. The admin password is generated when it first starts and should never be committed.

## Repository layout

```
00_download_data.py          downloads the Kaggle data into data/raw
01_load_to_postgres.py       loads selected columns into PostgreSQL in chunks
02_explore_and_leakage_audit.py
03_build_dataset.py          features and time-based split
04_scorecard.py              WoE, IV and the logistic scorecard
05_train_xgb.py              XGBoost, calibration and model comparison
06_cutoff.py                 profit-based approval cut-off
07_shap.py                   SHAP explanations and reason codes
08_simulate_traffic.py       simulated scoring traffic for monitoring
features.py, calibration.py, metrics.py, scoring.py, monitoring.py
config.py, db.py             settings and database helpers
make_sample_data.py          synthetic test data
app/                         FastAPI service
dags/                        Airflow DAG
tests/                       pytest suite
Dockerfile, docker-compose*.yml
```

The database password (`credit`) is a demonstration credential for a local container and is not used anywhere else.

## Limitations

- **Selection bias.** The model was trained only on loans that Lending Club accepted, so it cannot judge the kind of applicant who would have been rejected. I did not attempt reject inference.
- **Limited signal.** Without the lender's grade and interest rate, the Gini coefficient is about 0.35. Lenders in practice add credit-bureau and behavioural data.
- **Assumed economics.** The profit figures use assumed gains and losses rather than real cash flows.
- **Scope.** Only 36-month loans are used, so that every outcome is known and none is still in progress (no censoring).
- **Monitoring covers inputs only.** Default outcomes become known about three years after approval, so a fall in model performance cannot be detected in time using outcomes alone.
- **Fairness.** I have not tested the model for differences in outcomes across protected groups.
- **Reason codes.** They are generated from SHAP values and have not been reviewed for regulatory wording.

This is an educational project and should not be used for real lending decisions.