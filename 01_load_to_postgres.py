import argparse # lets the script accept command-line options such as --max-rows
import pandas as pd # pandas reads the CSV
from sqlalchemy import text
from config import DATA_RAW
from db import get_engine, init_tables # come from db.py


# which columns to keep
KEEP = {
    "id", "loan_amnt", "term", "int_rate", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "total_pymnt", "total_rec_prncp", "recoveries", "last_fico_range_high", "out_prncp",
}

# locating file and starting up
def find_file(arg):
    if arg:
        return arg
    files = sorted(DATA_RAW.glob("*.csv*"))
    if not files:
        raise SystemExit("No CSV found in data/raw. Download the data first.")
    return str(files[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    path = find_file(args.file)
    print("Loading", path)

    engine = get_engine()
    init_tables(engine)

    # reading in chunks
    total = 0
    reader = pd.read_csv(path, usecols=lambda c: c in KEEP, chunksize=100_000, low_memory=False, nrows=args.max_rows) # 'usecols' witha fn keeps only columns in KEEP and silently ignores ones that do not exist. 'chunksize', 'read_csv' returns an iterator that hands us 100,000 rows at a time.
    for i, chunk in enumerate(reader): # 'enumerate' gives each chunk a number i starting at 0
        # Writing each chunk
        chunk = chunk[chunk["id"].astype(str).str.fullmatch(r"\d+")] # A real load id is all digits, so this line helps keep only rows whose id matches \d+ (one or more digits)
        chunk.to_sql("raw_loans", engine, if_exists="replace" if i==0 else "append", index=False, chunksize=5_000, method="multi")
        total += len(chunk) # writes the chunk to the table. the first chunk *replaces* the table (so re-running the script starts fresh) and the rest *append*. method="multi" sends many rows per INSERT statement, which is faster.
        print(f"    chunk {i}: {total:,} rows loaded")

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_raw_loans_issue ON raw_loans (issue_d)"))
        n = conn.execute(text("SELECT count(*) FROM raw_loans")).scalar()

    print("raw_loans now has", f"{n:,}", "rows")

if __name__ == "__main__":
    main()

