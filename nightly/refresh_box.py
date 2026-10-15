"""Refresh NBA player box scores (current season) from sportsdataverse's daily-updated release,
then rebuild data/processed/nba_player_box.csv.gz. Run before the live card in NBA season."""
import gzip, importlib.util, shutil, sys
from datetime import datetime
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "props_nba" / "player_box"
URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{y}.parquet"

def main():
    now = datetime.now()
    season = now.year + (1 if now.month >= 8 else 0)
    RAW.mkdir(parents=True, exist_ok=True)
    have_all = len(list(RAW.glob("player_box_*.parquet"))) > 5
    years = [season] if have_all else list(range(2012, season + 1))   # fresh clone: pull history once
    for y in years:
        r = requests.get(URL.format(y=y), timeout=120)
        if r.status_code == 200:
            (RAW / f"player_box_{y}.parquet").write_bytes(r.content)
            print(f"player_box_{y}: {len(r.content)/1e6:.1f} MB")
        else:
            print(f"player_box_{y}: HTTP {r.status_code} (season may not have started)")
    spec = importlib.util.spec_from_file_location("nbp", ROOT / "scripts" / "data" / "nba_props_build.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    box = m.build_box()
    out = ROOT / "data" / "processed" / "nba_player_box.csv.gz"
    box.to_csv(out, index=False, compression="gzip")
    plain = ROOT / "data" / "processed" / "nba_player_box.csv"
    if plain.exists():
        plain.unlink()
    print(f"wrote {out} ({len(box)} rows, through {box.date.max()})")

if __name__ == "__main__":
    main()
