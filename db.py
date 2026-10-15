from sqlalchemy import create_engine, text
from config import DATABASE_URL

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

DDL = [
    """
    CREATE TABLE IF NOT EXISTS score_log (
        id BIGSERIAL PRIMARY KEY,
        request_id TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        model_version TEXT NOT NULL,
        pd DOUBLE PRECISION NOT NULL,
        decision TEXT NOT NULL,
        features JSONB NOT NULL,
        reasons JSONB
    )
""",

"""
    CREATE TABLE IF NOT EXISTS monitoring_psi (
        id BIGSERIAL PRIMARY KEY,
        run_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        window_days INTEGER NOT NULL,
        n_rows INTEGER NOT NULL,
        feature TEXT NOT NULL,
        psi DOUBLE PRECISION NOT NULL,
        status TEXT NOT NULL
    )
"""
]

def init_tables(engine=None):
    engine = engine or get_engine()
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))

