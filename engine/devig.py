"""
Vig removal for two-way markets, vectorized, with methods from penaltyblog
(github.com/martineastwood/penaltyblog), and an empirical test of which method's fair probabilities
best predict real outcomes on our historical closing moneylines.

Methods (two-way):
  multiplicative  p_i = pi_i / sum(pi)                        (what the engine used originally)
  additive        p_i = pi_i - (sum(pi) - 1) / 2              (== Shin's method for 2 outcomes)
  power           p_i = pi_i ** k,  sum = 1                   (shrinks longshots more)
  odds_ratio      p_i = pi_i / (c + pi_i - c * pi_i), sum = 1

  python -m engine.devig      -> results/devig_test.csv : log loss / Brier per method, discovery vs holdout
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.core import american_to_prob

ROOT = Path(__file__).resolve().parents[1]
BEST_FILE = ROOT / "results" / "devig_best.json"


def _bisect(f, lo, hi, n=60):
    lo, hi = np.full_like(f(lo), lo, dtype=float), np.full_like(f(lo), hi, dtype=float)
    for _ in range(n):
        mid = (lo + hi) / 2
        v = f(mid)
        lo, hi = np.where(v > 0, mid, lo), np.where(v > 0, hi, mid)
    return (lo + hi) / 2


def devig(odds_a, odds_b, method: str = "multiplicative"):
    """Fair probabilities (p_a, p_b) from American odds for a two-way market."""
    pa, pb = np.asarray(american_to_prob(odds_a), float), np.asarray(american_to_prob(odds_b), float)
    s = pa + pb
    if method == "multiplicative":
        a = pa / s
    elif method in ("additive", "shin"):
        a = np.clip(pa - (s - 1) / 2, 1e-6, 1 - 1e-6)
    elif method == "power":
        # find k >= 1 with pa^k + pb^k = 1 (increasing k lowers the sum)
        k = _bisect(lambda k: pa ** k + pb ** k - 1, 1.0, 5.0) if np.ndim(pa) else \
            _bisect(lambda k: np.atleast_1d(pa) ** k + np.atleast_1d(pb) ** k - 1, 1.0, 5.0)[0]
        a = pa ** k
    elif method == "odds_ratio":
        f = lambda c: pa / (c + pa - c * pa) + pb / (c + pb - c * pb) - 1
        c = _bisect(lambda c: np.atleast_1d(f(c)), 1.0, 5.0)
        c = c if np.ndim(pa) else c[0]
        a = pa / (c + pa - c * pa)
    else:
        raise ValueError(method)
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return a, 1 - a


def best_method() -> str:
    try:
        return json.loads(BEST_FILE.read_text())["method"]
    except Exception:
        return "multiplicative"


METHODS = ["multiplicative", "additive", "power", "odds_ratio"]
SPLITS = {"nfl": 2016, "ncaaf": 2018, "nba": 2019, "ncaam": 2020, "mlb": 2019, "nhl": 2019, "ufc": 2020}


def _games() -> pd.DataFrame:
    from backtest import run_team as rt
    rows = []
    for sp in ["nfl", "ncaaf", "nba", "ncaam", "mlb", "nhl"]:
        g = rt.load(sp).dropna(subset=["home_ml", "away_ml", "home_score", "away_score"])
        g = g[(g.home_ml.abs() >= 100) & (g.away_ml.abs() >= 100) & (g.home_score != g.away_score)]
        rows.append(pd.DataFrame({"sport": sp, "season": g.season, "a": g.home_ml, "b": g.away_ml,
                                  "y": (g.home_score > g.away_score).astype(int)}))
    u = pd.read_csv(ROOT / "data" / "processed" / "ufc.csv").dropna(subset=["a_ml", "b_ml"])
    u = u[u.winner.isin(["A", "B"])]
    rows.append(pd.DataFrame({"sport": "ufc", "season": pd.to_datetime(u.date).dt.year, "a": u.a_ml, "b": u.b_ml,
                              "y": (u.winner == "A").astype(int)}))
    G = pd.concat(rows, ignore_index=True)
    s = american_to_prob(G.a) + american_to_prob(G.b)
    return G[(s > 1.0) & (s < 1.15)].reset_index(drop=True)      # drop broken quotes


def evaluate() -> pd.DataFrame:
    G = _games()
    G["holdout"] = G.apply(lambda r: r.season >= SPLITS[r.sport], axis=1)
    out = []
    for m in METHODS:
        p, _ = devig(G.a.values, G.b.values, m)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -(G.y * np.log(p) + (1 - G.y) * np.log(1 - p))
        br = (p - G.y) ** 2
        # longshot calibration: games where one side is priced at +250 or longer
        dog = np.minimum(p, 1 - p)
        long_mask = dog < 0.28
        dog_won = np.where(p < 0.5, G.y, 1 - G.y)
        for sample, m_ in (("discovery", ~G.holdout), ("holdout", G.holdout)):
            for sp in ["ALL"] + sorted(G.sport.unique()):
                k = m_ & ((G.sport == sp) if sp != "ALL" else True)
                lk = k & long_mask
                out.append(dict(method=m, sample=sample, sport=sp, n=int(k.sum()), log_loss=float(ll[k].mean()),
                                brier=float(br[k].mean()), longshot_pred=float(dog[lk].mean()),
                                longshot_real=float(dog_won[lk].mean())))
    R = pd.DataFrame(out)
    return R


if __name__ == "__main__":
    R = evaluate()
    R.to_csv(ROOT / "results" / "devig_test.csv", index=False)
    A = R[R.sport == "ALL"].pivot_table(index="method", columns="sample", values=["log_loss", "brier", "longshot_pred", "longshot_real"])
    pd.set_option("display.width", 220)
    print(A.round(5).to_string())
    disc = R[(R.sport == "ALL") & (R["sample"] == "discovery")].sort_values("log_loss")
    best = disc.iloc[0].method
    hold = R[(R.sport == "ALL") & (R["sample"] == "holdout")].set_index("method")
    print(f"\nchosen on discovery: {best} | holdout log loss {hold.loc[best,'log_loss']:.5f} vs multiplicative "
          f"{hold.loc['multiplicative','log_loss']:.5f}")
    print(R[(R["sample"] == "holdout") & (R.sport != "ALL")].pivot_table(index="sport", columns="method", values="log_loss").round(5).to_string())
    BEST_FILE.write_text(json.dumps(dict(method=best, holdout_log_loss=float(hold.loc[best, "log_loss"]),
                                         multiplicative_holdout_log_loss=float(hold.loc["multiplicative", "log_loss"])), indent=1))
