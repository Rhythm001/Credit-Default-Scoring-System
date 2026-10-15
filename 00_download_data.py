import shutil 
from pathlib import Path
import kagglehub

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok = True)

src = Path(kagglehub.dataset_download("wordsforthewise/lending-club"))

found = [p for p in src.rglob("*")
         if p.is_file() and "accepted" in p.name.lower() and p.suffix in {".csv", ".gz"}]

if not found:
    raise SystemExit(f"No accepted-loans file in {src}. Files there: {[p.name for p in src.rglob('*')]}")

for p in found:
    shutil.copy2(p, RAW/p.name)
    print("copied", p.name)