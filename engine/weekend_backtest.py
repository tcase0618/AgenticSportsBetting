"""
Backtest of weekend mega-parlays on REAL closing prices (no simulation of outcomes).

  A. Edge-only weekend parlay: every validated strategy leg that fired that day, one parlay.
  B. NFL every-game Sunday parlay: one leg per game = the strategy leg if one fired, else the favorite ML.
  C. Probability-capped versions of B: the N most likely legs (N = 3, 5, 8, 10, 20).
Holdout = seasons the strategies never saw during discovery.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from backtest.core import american_to_decimal, decimal_to_american, fmt_american, no_vig_probs
from backtest.parlay import load_ledger

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def settle(dec: np.ndarray, out: np.ndarray) -> float:
    if (out < 0).any():
        return -1.0
    live = dec[out > 0]
    return float(np.prod(live) - 1) if len(live) else 0.0


def summarize(T: pd.DataFrame, label: str) -> dict:
    if T.empty:
        return dict(structure=label, tickets=0)
    hits = T[T.pnl > 0]
    return dict(structure=label, tickets=len(T), avg_legs=round(T.legs.mean(), 1), hit_rate=round((T.pnl > 0).mean(), 4),
                roi=round(T.pnl.mean(), 4), units=round(T.pnl.sum(), 1),
                avg_price=fmt_american(decimal_to_american(T.dec.median())),
                biggest_hit=fmt_american(decimal_to_american(hits.dec.max())) if len(hits) else "none",
                longest_drought=int(max((len(list(g)) for k, g in __import__("itertools").groupby(T.pnl < 0) if k), default=0)))


def edge_weekends(L: pd.DataFrame, tiers=("A", "B")) -> pd.DataFrame:
    x = L[L.tier.isin(tiers) & L.date.dt.dayofweek.isin([5, 6])]
    rows = []
    for d, day in x.groupby("date"):
        if len(day) < 2:
            continue
        rows.append(dict(date=d, legs=len(day), dec=float(np.prod(day.dec)), pnl=settle(day.dec.values, day.outcome.values),
                         holdout=bool(day.holdout.all())))
    return pd.DataFrame(rows)


def nfl_every_game(L: pd.DataFrame, caps=(None, 3, 5, 8, 10, 20)) -> dict[str, pd.DataFrame]:
    from backtest import run_team as rt
    g = rt.load("nfl").dropna(subset=["home_score", "home_ml", "away_ml"])
    g["date"] = pd.to_datetime(g.date)
    g = g[g.date.dt.dayofweek == 6]
    nfl_led = L[L.sport == "nfl"].copy()
    nfl_led["home"] = nfl_led.event.str.split(" @ ").str[1]
    out = {c: [] for c in caps}
    for d, day in g.groupby("date"):
        legs = []
        for _, r in day.iterrows():
            k = nfl_led[(nfl_led.date == d) & (nfl_led.home == r.home)]
            if len(k):
                k = k.iloc[0]
                legs.append((float(american_to_decimal(k.price)), int(k.outcome), 0.5 + 0.05, True))
            else:
                ph, pa, _ = no_vig_probs(r.home_ml, r.away_ml)
                fav_home = ph >= pa
                price = r.home_ml if fav_home else r.away_ml
                margin = (r.home_score - r.away_score) * (1 if fav_home else -1)
                legs.append((float(american_to_decimal(price)), int(np.sign(margin)), float(max(ph, pa)), False))
        legs.sort(key=lambda x: -x[2])  # most likely first
        season = int(day.season.iloc[0])
        for c in caps:
            sel = legs[:c] if c else legs
            if c and len(legs) < c:
                continue
            dec = np.array([x[0] for x in sel]); o = np.array([x[1] for x in sel])
            out[c].append(dict(date=d, season=season, legs=len(sel), dec=float(np.prod(dec)), pnl=settle(dec, o)))
    return {c: pd.DataFrame(v) for c, v in out.items()}


if __name__ == "__main__":
    L = load_ledger()
    rows = []
    for tiers in (("A",), ("A", "B"), ("A", "B", "C")):
        T = edge_weekends(L, tiers)
        rows.append(summarize(T, f"Edge-only weekend parlay, tiers {'+'.join(tiers)} (all seasons)"))
        rows.append(summarize(T[T.holdout], f"Edge-only weekend parlay, tiers {'+'.join(tiers)} (holdout)"))
    E = nfl_every_game(L)
    for c, T in E.items():
        rows.append(summarize(T, f"NFL Sunday every-game parlay, {'all games' if c is None else f'top {c} most-likely legs'}"))
    R = pd.DataFrame(rows)
    R.to_csv(RES / "weekend_parlay_backtest.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.to_string())
