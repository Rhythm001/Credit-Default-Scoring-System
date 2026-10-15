import os
import sys
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/airflow/project")
sys.path.insert(0, PROJECT_DIR)

@dag(
    dag_id="credit_monitoring",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["credit-risk", "monitoring"],
)
def credit_monitoring():
    @task
    def compute_drift() -> dict:
        from monitoring import run_monitoring
        return run_monitoring(window_days=3)

    @task
    def alert_if_drifted(summary: dict):
        print("Drift summary:", summary)
        if summary["status"] == "insufficient_data":
            print("Not enough recent traffic to judge drift.")
            return
        if summary["status"] == "shifted":
            raise AirflowFailException(
                f"Significant drift: {summary['worst_feature']} PSI={summary['max_psi']}. Review the model.")

    alert_if_drifted(compute_drift())

credit_monitoring()
