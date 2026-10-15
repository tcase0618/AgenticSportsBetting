"""
Parlay backtester over the historical strategy ledger.

Rules that keep it honest:
  * Only legs from DIFFERENT events are combined (books price cross-game parlays as the
    product of decimal odds; same-game parlays use opaque correlation pricing we can't backtest).
  * A pushed leg drops out (parlay re-prices on remaining legs), exactly like a sportsbook.
  * Legs are only combined with other legs available the SAME calendar day.
  * "Holdout" = every leg comes from a season its strategy never saw during discovery.

usage: python -m backtest.parlay
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import american_to_decimal, perf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

TIERS = {
    # strategy names (from keys.py / ledger.EXTRA_LEGS) grouped by confidence
    "A": ["Youth Over Experience", "Wind Wall Unders", "Bounce-back dogs ATS"],
    "B": ["Rested Conference Dogs", "Blowout Bounce-Back", "Load-Management Overs"],
    "C": ["Blowout Believers", "Series Carryover Unders"],
}
TIER_OF = {s: t for t, ss in TIERS.items() for s in ss}
SPLITS = {}  # sport -> holdout start season, filled from results/<sport>_meta.json


def load_ledger() -> pd.DataFrame:
    L = pd.read_csv(OUT / "ledger.csv", parse_dates=["date"])
    L["dec"] = american_to_decimal(L["price"])
    L["tier"] = L["strategy"].map(TIER_OF)
    for sp in L.sport.unique():
        SPLITS[sp] = json.loads((OUT / f"{sp}_meta.json").read_text())["split"]
    L["holdout"] = L.apply(lambda r: r["season"] >= SPLITS[r["sport"]], axis=1)
    # one leg per event per day (if two strategies hit the same game, keep the higher tier)
    L = L.sort_values(["date", "event", "tier"]).drop_duplicates(["date", "event"], keep="first")
    return L.reset_index(drop=True)


def settle_parlay(legs: pd.DataFrame) -> tuple[float, float]:
    """Return (P&L on 1u stake, decimal price at placement)."""
    price = float(np.prod(legs["dec"]))
    if (legs["outcome"] < 0).any():
        return -1.0, price
    live = legs[legs["outcome"] > 0]
    if live.empty:  # all pushed
        return 0.0, price
    return float(np.prod(live["dec"]) - 1.0), price


def run_structures(L: pd.DataFrame, max_legs_day=6) -> dict[str, pd.DataFrame]:
    """Each structure -> DataFrame of tickets: date, n_legs, price, pnl, holdout."""
    out = {k: [] for k in ["singles", "rr2", "rr3", "full_day", "best2", "best3"]}
    for date, day in L.groupby("date"):
        day = day.sort_values(["tier", "price"])  # best tier first, shortest price first
        ho = bool(day["holdout"].all())
        for _, r in day.iterrows():
            out["singles"].append(dict(date=date, n_legs=1, price=r.dec, pnl=r.pnl, holdout=r.holdout))
        n = len(day)
        if n >= 2:
            for k, name in ((2, "rr2"), (3, "rr3")):
                if n >= k:
                    combos = list(itertools.combinations(range(n), k))
                    for c in combos:  # stake 1u per ticket
                        p, pr = settle_parlay(day.iloc[list(c)])
                        out[name].append(dict(date=date, n_legs=k, price=pr, pnl=p,
                                              holdout=bool(day.iloc[list(c)]["holdout"].all())))
            if n <= max_legs_day:
                p, pr = settle_parlay(day)
                out["full_day"].append(dict(date=date, n_legs=n, price=pr, pnl=p, holdout=ho))
            p, pr = settle_parlay(day.iloc[:2])
            out["best2"].append(dict(date=date, n_legs=2, price=pr, pnl=p, holdout=bool(day.iloc[:2]["holdout"].all())))
            if n >= 3:
                p, pr = settle_parlay(day.iloc[:3])
                out["best3"].append(dict(date=date, n_legs=3, price=pr, pnl=p, holdout=bool(day.iloc[:3]["holdout"].all())))
    return {k: pd.DataFrame(v) for k, v in out.items()}


def summarize(T: pd.DataFrame) -> dict:
    if T.empty:
        return dict(n=0)
    p = perf(T["pnl"])
    losing = (T["pnl"] < 0).astype(int).values
    streak = max((len(list(g)) for k, g in itertools.groupby(losing) if k == 1), default=0)
    return dict(tickets=p["n"], hit=float((T.pnl > 0).mean()), roi=p["roi"], units=p["units"], z=p["z"],
                max_dd=p["max_dd"], longest_losing=streak, avg_price=float(T.price.mean()),
                biggest_win=float(T.pnl.max()))


def report(L: pd.DataFrame, label: str) -> pd.DataFrame:
    S = run_structures(L)
    rows = []
    for name, T in S.items():
        for subset, TT in (("all", T), ("holdout", T[T.holdout] if len(T) else T)):
            rows.append(dict(pool=label, structure=name, sample=subset, **summarize(TT)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    L = load_ledger()
    pools = {
        "A only": L[L.tier == "A"],
        "A+B": L[L.tier.isin(["A", "B"])],
        "A+B+C": L,
    }
    R = pd.concat([report(df, name) for name, df in pools.items()], ignore_index=True)
    R.to_csv(OUT / "parlay_backtest.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())
