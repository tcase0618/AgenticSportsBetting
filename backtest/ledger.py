"""
Unified bet ledger: every historical bet placed by a strategy, with its exact price.

Columns: date, sport, strategy, event, pick, price (American), outcome (+1 win / 0 push / -1 loss), pnl
This is the raw material for parlay construction and bankroll simulation.

usage: python -m backtest.ledger            -> results/ledger.csv for all KEYS (+ EXTRA_LEGS)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .keys import KEYS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Validated runners-up that are also worth feeding into parlays (sport, name, key)
EXTRA_LEGS = [
    ("nfl", "Bounce-back dogs ATS", ("ats", "prev_big_ats_fail", "mid_dog", "on", "reg")),
]

MKT_COLS = {  # market -> (home/over pnl key, away/under pnl key, home/over price col, away/under price col, labels)
    "ats": ("ats_h", "ats_a", "spread_home_odds", "spread_away_odds"),
    "ml": ("ml_h", "ml_a", "home_ml", "away_ml"),
    "tot": ("tot_o", "tot_u", "over_odds", "under_odds"),
}


def _team_ledger(sport: str, name: str, key: tuple) -> pd.DataFrame:
    from . import run_team as rt
    from .features import build
    from .strategies import market_pnl, side_strategies, total_strategies
    g = build(rt.load(sport), rt.ELO.get(sport))
    meta = json.loads((OUT / f"{sport}_meta.json").read_text())
    pnl = market_pnl(g)
    mkt, cond, role = key[0], key[1], key[2]
    if mkt == "tot":
        tq = g.loc[g.season < meta["split"], "total"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        S = total_strategies(g, pnl, meta["S"], sport, tq, only=cond)
    else:
        S = side_strategies(g, pnl, meta["S"], sport, markets=(mkt,), only=(cond, role))
    s = S[key]
    hk, ak, hp, ap = MKT_COLS[mkt]
    rows = []
    for gid, v in s.items():
        r = g.loc[gid]
        # identify which side was bet by matching realized P&L (pushes: side irrelevant to price impact)
        if mkt == "tot":
            side_home = key[3] == "over"
        else:
            side_home = np.isclose(pnl[hk].loc[gid], v) and not np.isclose(pnl[ak].loc[gid], v) \
                if not np.isclose(v, 0) else None
            if side_home is None:  # push: infer from role definition is unnecessary; mark home
                side_home = True
        price = r[hp] if side_home else r[ap]
        if mkt == "tot":
            pick = f"{'OVER' if side_home else 'UNDER'} {r['total']}"
        elif mkt == "ats":
            line = r["spread_home"] if side_home else -r["spread_home"]
            pick = f"{r['home'] if side_home else r['away']} {line:+g}"
        else:
            pick = f"{r['home'] if side_home else r['away']} ML"
        rows.append(dict(date=r["date"], season=int(r["season"]), sport=sport, strategy=name,
                         event=f"{r['away']} @ {r['home']}", pick=pick, price=float(price),
                         outcome=int(np.sign(v)), pnl=float(v)))
    return pd.DataFrame(rows)


def _ufc_ledger(name: str, key: tuple) -> pd.DataFrame:
    from .run_ufc import prep, strategies
    f = prep(pd.read_csv(ROOT / "data" / "processed" / "ufc.csv"))
    f = f[f.a_ml.notna() & f.b_ml.notna()]
    S, pa, pb = strategies(f)
    s = S[key]
    rows = []
    for i, v in s.items():
        r = f.loc[i]
        side_a = np.isclose(pa.loc[i], v) and not np.isclose(pb.loc[i], v) if not np.isclose(v, 0) else True
        rows.append(dict(date=r["date"], season=int(r["season"]), sport="ufc", strategy=name,
                         event=f"{r['fighter_a']} vs {r['fighter_b']}",
                         pick=f"{r['fighter_a'] if side_a else r['fighter_b']} ML",
                         price=float(r["a_ml"] if side_a else r["b_ml"]), outcome=int(np.sign(v)), pnl=float(v)))
    return pd.DataFrame(rows)


def build_ledger() -> pd.DataFrame:
    parts = []
    specs = [(sp, d["name"], tuple(d["key"])) for sp, d in KEYS.items()] + EXTRA_LEGS
    for sport, name, key in specs:
        L = _ufc_ledger(name, key) if sport == "ufc" else _team_ledger(sport, name, key)
        print(f"{sport:6s} {name:28s} bets={len(L):5d} roi={L.pnl.mean():+.3f}")
        parts.append(L)
    L = pd.concat(parts, ignore_index=True)
    L["date"] = pd.to_datetime(L["date"]).dt.normalize()
    return L.sort_values(["date", "sport", "event"]).reset_index(drop=True)


if __name__ == "__main__":
    L = build_ledger()
    OUT.mkdir(exist_ok=True)
    L.to_csv(OUT / "ledger.csv", index=False)
    print(L.groupby("strategy").agg(bets=("pnl", "size"), roi=("pnl", "mean"), units=("pnl", "sum")).round(3))
