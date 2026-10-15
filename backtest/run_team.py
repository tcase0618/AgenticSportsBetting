"""
Run the full strategy search for one team sport.

usage: python -m backtest.run_team nfl
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build, starter_features
from .strategies import market_pnl, side_strategies, total_strategies
from .evaluate import evaluate, describe
from .model import walk_forward, model_bets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"

# Elo tuning per sport (k, home-field in Elo pts, between-season regression)
ELO = {
    "nfl": dict(k=20, hfa=48, revert=0.33),
    "ncaaf": dict(k=25, hfa=65, revert=0.25),
    "nba": dict(k=20, hfa=70, revert=0.25),
    "ncaam": dict(k=25, hfa=80, revert=0.30),
    "mlb": dict(k=4, hfa=24, revert=0.33, mov=False),
    "nhl": dict(k=6, hfa=33, revert=0.33, mov=False),
}
MIN_TRAIN = {"nfl": 120, "ncaaf": 200, "nba": 200, "ncaam": 300, "mlb": 300, "nhl": 250}
MIN_HOLD = {"nfl": 60, "ncaaf": 100, "nba": 100, "ncaam": 150, "mlb": 150, "nhl": 120}


POWER_RE = r"^(SEC|Southeastern|Big Ten|Big 12|ACC|Atlantic Coast|Pac-12|Pac-10|Big East|FBS Independents)"
MODEL_EXTRA = {"nfl": ["wind", "temp", "div_game"]}


def dedupe(strats: dict) -> dict:
    """Drop strategies whose bet set is identical to a simpler one already kept."""
    seen, out = set(), {}
    for k in sorted(strats, key=lambda k: (k[1].count("+"), k[4] != "all", len(k[1]))):
        s = strats[k]
        h = (k[0], k[3] if k[0] == "tot" else "", hash(tuple(s.index)), round(float(s.sum()), 6))
        if h in seen:
            continue
        seen.add(h)
        out[k] = s
    return out


def load(sport: str) -> pd.DataFrame:
    if sport == "nfl":
        d = pd.read_csv(ROOT / "data" / "raw" / "nfl_games.csv")
        d = d.dropna(subset=["home_score"])
        out = pd.DataFrame({
            "date": d["gameday"], "season": d["season"], "home": d["home_team"], "away": d["away_team"],
            "home_score": d["home_score"], "away_score": d["away_score"],
            # nflverse spread_line is positive when HOME is favored -> flip to home perspective
            "spread_home": -d["spread_line"], "total": d["total_line"],
            "home_ml": d["home_moneyline"], "away_ml": d["away_moneyline"],
            "spread_home_odds": d["home_spread_odds"], "spread_away_odds": d["away_spread_odds"],
            "over_odds": d["over_odds"], "under_odds": d["under_odds"],
            "neutral": (d["location"] == "Neutral").astype(int),
            "playoff": (d["game_type"] != "REG").astype(int),
            "div_game": d["div_game"], "roof": d["roof"], "wind": d["wind"], "temp": d["temp"],
            "surface": d["surface"],
        })
        # unify relocated franchises
        out[["home", "away"]] = out[["home", "away"]].replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})
        return out
    d = pd.read_csv(DATA / f"{sport}.csv")
    if "playoff" not in d and "postseason" in d:
        d["playoff"] = d["postseason"].fillna(0).astype(int)
    if sport in ("mlb", "nhl"):
        # never assume -110 on a run/puck line: drop lines without real juice
        no_juice = d["spread_home_odds"].isna() | d["spread_away_odds"].isna()
        d.loc[no_juice, "spread_home"] = np.nan
        st = ("home_sp", "away_sp") if sport == "mlb" else ("home_goalie", "away_goalie")
        if st[0] in d:
            d = starter_features(d, *st, backup=(sport == "nhl"))
    if sport in ("ncaaf", "ncaam"):
        conf_cols = [c for c in ("home_conference", "away_conference") if c in d]
        if conf_cols:
            for side in ("home", "away"):
                c = d[f"{side}_conference"].astype(str)
                d[f"{side[0]}_power"] = c.str.contains(POWER_RE, case=False, regex=True).astype(int)
    return d


def split_for(seasons: pd.Series, frac=0.62) -> int:
    u = np.sort(seasons.unique())
    return int(u[int(len(u) * frac)])


def run(sport: str, split: int | None = None):
    raw = load(sport)
    g = build(raw, ELO.get(sport))
    g = g[g["season"] >= g["season"].min()]  # placeholder for sport-specific trims
    split = split or split_for(g["season"])
    S = float(g["ats_home"].std()) if g["spread_home"].notna().mean() > 0.5 else float(g["margin"].std())
    if sport in ("mlb", "nhl"):  # fixed +/-1.5 lines: scale thresholds on actual margins
        S = float(g["margin"].std())
    pnl = market_pnl(g)
    tq = g.loc[g.season < split, "total"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    strats = side_strategies(g, pnl, S, sport)
    strats.update(total_strategies(g, pnl, S, sport, tq))
    # walk-forward model: every prediction is out-of-sample (fit on prior seasons only)
    first_test = int(np.sort(g.season.unique())[min(5, g.season.nunique() - 1)])
    extra = [c for c in MODEL_EXTRA.get(sport, []) if c in g]
    preds = walk_forward(g, pnl, first_test, extra=extra)
    strats.update(model_bets(g, pnl, preds))
    strats = dedupe(strats)
    R = evaluate(strats, g, split, MIN_TRAIN.get(sport, 150), MIN_HOLD.get(sport, 60))
    OUT.mkdir(exist_ok=True)
    R.to_csv(OUT / f"{sport}_strategies.csv", index=False)
    meta = dict(sport=sport, seasons=[int(g.season.min()), int(g.season.max())], split=split,
                games=len(g), S=S, n_strategies=len(strats), n_evaluated=len(R),
                n_discovered=int(R.discovered.sum()), n_validated=int(R.validated.sum()),
                baseline={k: float(v.mean()) for k, v in pnl.items()})
    (OUT / f"{sport}_meta.json").write_text(json.dumps(meta, indent=2))
    return g, pnl, strats, R, meta


if __name__ == "__main__":
    sport = sys.argv[1]
    g, pnl, strats, R, meta = run(sport)
    print(json.dumps(meta, indent=2))
    cols = ["market", "condition", "role", "direction", "phase", "n_train", "roi_train", "p_train",
            "n_hold", "roi_hold", "z_hold", "q_hold", "roi_all", "pct_pos_seasons"]
    pd.set_option("display.width", 250)
    print(R[R.validated].head(30)[cols].to_string())
