import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text
from db import get_engine, init_tables
from scoring import get_scorer, application_frame

log = logging.getLogger("credit-api")

class Application(BaseModel):
    loan_amnt: float = Field(gt=0, le=100_000, examples=[12000])
    annual_inc: float = Field(gt=0, examples=[65000])
    dti: float = Field(ge=0, le=100, examples=[18.5])
    fico_score: float = Field(ge=300, le=850, examples=[705])
    emp_length_years: Optional[float] = Field(default=None, ge=0, le=60)
    credit_history_years: float = Field(ge=0, le=80, examples=[14])
    inq_last_6mths: float = Field(ge=0, le=50, examples=[1])
    open_acc: float = Field(ge=0, le=100, examples=[10])
    pub_rec: float = Field(ge=0, le=50, examples=[0])
    revol_bal: float = Field(ge=0, examples=[9000])
    revol_util: Optional[float] = Field(default=None, ge=0, le=200)
    total_acc: float = Field(ge=0, le=200, examples=[22])
    mort_acc: Optional[float] = Field(default=None, ge=0, le=50)
    delinq_2yrs: float = Field(ge=0, le=50, examples=[0])
    home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    verification_status: Literal["Verified", "Source Verified", "Not Verified"]
    purpose: Literal["debt_consolidation", "credit_card", "home_improvement", "small_business", "car", "other"]

class Reason(BaseModel):
    feature: str
    text: str
    impact: float

class Decision(BaseModel):
    request_id: str
    model_version: str
    pd: float
    cutoff_pd: float
    decision: Literal["approve", "decline"]
    reasons: list[Reason]

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_scorer()
    try:
        init_tables()
    except Exception as e:
        log.warning("Database unavailable at startup, scores will not be logged: %s", e)
    yield

app = FastAPI(title="Credit Default Scoring API", version="1.0", lifespan=lifespan)

@app.get("/health")
def health():
    sc = get_scorer()
    return {"status": "ok", "model_version": sc.meta["model_version"], "cutoff_pd": sc.meta["cutoff_pd"]}

def log_score(request_id, version, pd_value, decision, features, reasons):
    try:
        with get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO score_log (request_id, model_version, pd, decision, features, reasons)
                VALUES (:rid, :ver, :pd, :dec, CAST(:feat AS jsonb), CAST(:rea AS jsonb))
            """), {"rid": request_id, "ver": version, "pd": pd_value, "dec": decision,
                   "feat": json.dumps(features), "rea": json.dumps(reasons)})
    except Exception as e:
        log.warning("Could not write score_log: %s", e)

@app.post("/score", response_model=Decision)
def score(app_in: Application):
    sc = get_scorer()
    features = app_in.model_dump()
    df = application_frame(features)
    pd_value = float(sc.predict_pd(df)[0])
    decision = sc.decide(pd_value)
    reasons = sc.reasons(df)[0]
    request_id = str(uuid.uuid4())
    log_score(request_id, sc.meta["model_version"], pd_value, decision, features, reasons)
    return Decision(request_id=request_id, model_version=sc.meta["model_version"], pd=round(pd_value, 4),
                    cutoff_pd=sc.meta["cutoff_pd"], decision=decision, reasons=reasons)
