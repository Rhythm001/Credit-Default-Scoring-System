# Credit Default Scoring System

Scorecard + XGBoost + SHAP + FastAPI + Docker + Airflow drift monitoring, built on Lending Club loans.

Run order: `01_load_to_postgres.py` -> `02_explore_and_leakage_audit.py` -> `03_build_dataset.py` -> `04_scorecard.py` -> `05_train_xgb.py` -> `06_cutoff.py` -> `07_shap.py` -> `uvicorn app.main:app` -> `08_simulate_traffic.py` -> Airflow DAG.

No real data yet? Run `python make_sample_data.py` to generate a FAKE file to test the pipeline (numbers from it are meaningless).

Add your own results here once you have run it on the real data: test AUC/KS, scorecard vs XGBoost, chosen cut-off and profit vs approve-everyone, top SHAP features, PSI findings.
