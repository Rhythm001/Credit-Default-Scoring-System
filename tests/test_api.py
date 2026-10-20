import copy
from fastapi.testclient import TestClient
from app.main import app

GOOD = dict(loan_amnt=8000, annual_inc=120000, dti=8, fico_score=790, emp_length_years=10,
            credit_history_years=22, inq_last_6mths=0, open_acc=12, pub_rec=0, revol_bal=5000,
            revol_util=15, total_acc=30, mort_acc=2, delinq_2yrs=0, home_ownership="MORTGAGE",
            verification_status="Verified", purpose="credit_card")
RISKY = dict(GOOD, loan_amnt=35000, annual_inc=28000, dti=38, fico_score=665, inq_last_6mths=5,
             revol_util=95, delinq_2yrs=2, home_ownership="RENT", purpose="small_business")

def client():
    return TestClient(app)

def test_health():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

def test_good_vs_risky():
    with client() as c:
        good = c.post("/score", json=GOOD).json()
        risky = c.post("/score", json=RISKY).json()
    assert good["pd"] < risky["pd"]
    assert good["decision"] == "approve"
    assert risky["decision"] == "decline"

def test_decline_has_reasons():
    with client() as c:
        risky = c.post("/score", json=RISKY).json()
    assert len(risky["reasons"]) >= 1
    assert all("text" in r for r in risky["reasons"])

def test_invalid_input_rejected():
    bad = copy.deepcopy(GOOD)
    bad["fico_score"] = 900
    with client() as c:
        assert c.post("/score", json=bad).status_code == 422

def test_fico_monotonic():
    with client() as c:
        pds = [c.post("/score", json=dict(RISKY, fico_score=f)).json()["pd"] for f in (640, 700, 760, 820)]
    assert pds == sorted(pds, reverse=True)

def test_missing_optional_fields_ok():
    p = dict(GOOD, emp_length_years=None, revol_util=None, mort_acc=None)
    with client() as c:
        assert c.post("/score", json=p).status_code == 200
