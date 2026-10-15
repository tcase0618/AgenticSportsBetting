"""Pitcher-game strikeout dataset from Statcast via pybaseball (github.com/jldbc/pybaseball).
One row per pitcher per game: K, batters faced, pitches, whiffs, CSW, starter flag, opponent, home/away.
Pulled week by week, aggregated immediately (raw pitches are not kept). Resumable.

  python scripts/data/mlb_statcast_build.py 2024-03-20 2026-10-14
"""
import sys, warnings
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "mlb_statcast_pitcher_games.csv.gz"
PART = ROOT / "data" / "raw" / "statcast_parts"
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
CALLED = {"called_strike"}


def aggregate(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.game_type == "R"] if "game_type" in d else d
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d["is_pa_end"] = d.events.notna() & (d.events != "")
    d["k"] = d.events.isin(["strikeout", "strikeout_double_play"])
    d["whiff"] = d.description.isin(WHIFF)
    d["csw"] = d.description.isin(WHIFF | CALLED)
    d["pitch_team"] = d.home_team.where(d.inning_topbot == "Top", d.away_team)
    d["opp"] = d.away_team.where(d.inning_topbot == "Top", d.home_team)
    d["home"] = (d.inning_topbot == "Top").astype(int)
    first = d.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"]).groupby(["game_pk", "pitch_team"]).pitcher.first()
    g = d.groupby(["game_pk", "game_date", "pitcher", "player_name", "pitch_team", "opp", "home", "p_throws"]).agg(
        k=("k", "sum"), bf=("is_pa_end", "sum"), pitches=("pitch_type", "size"), whiffs=("whiff", "sum"),
        csw=("csw", "sum"), max_inning=("inning", "max")).reset_index()
    g["starter"] = [int(first.get((gp, t)) == p) for gp, t, p in zip(g.game_pk, g.pitch_team, g.pitcher)]
    return g


def main(start: str, end: str):
    from pybaseball import statcast
    PART.mkdir(parents=True, exist_ok=True)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    while s <= e:
        t = min(s + timedelta(days=6), e)
        f = PART / f"{s}_{t}.csv"
        if not f.exists():
            try:
                raw = statcast(start_dt=str(s), end_dt=str(t), verbose=False)
                agg = aggregate(raw)
                agg.to_csv(f, index=False)
                print(f"{s}..{t}: {len(raw)} pitches -> {len(agg)} pitcher-games", flush=True)
            except Exception as ex:
                print(f"{s}..{t}: failed {ex}", flush=True)
        s = t + timedelta(days=1)
    parts = [pd.read_csv(p) for p in sorted(PART.glob("*.csv")) if p.stat().st_size > 50]
    allg = pd.concat(parts).drop_duplicates(["game_pk", "pitcher"]).sort_values(["game_date", "game_pk"])
    allg.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT}: {len(allg)} pitcher-games, starters {allg.starter.sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
