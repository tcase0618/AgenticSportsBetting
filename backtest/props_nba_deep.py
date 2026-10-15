"""
Deeper NBA props analysis on the saved frame (run backtest.props_nba first).

  1. Stacked calibration: logistic regression on [logit p_mkt, logit p_mod, side/line terms] fit on the
     DISCOVERY season only, applied to holdout.
  2. Segment breakdown of the chosen strategy (book, side, line size, projection gap).
  3. Same-book cross-game parlays of the day's top legs (2-leg / 3-leg, plus round robins).
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .core import perf
from .props_nba import pick_bets, DISC, HOLD

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_features(p):
    X = pd.DataFrame({"lm": logit(p.p_mkt), "lp": logit(p.p_mod)})
    X["gap"] = (p.proj - p.mu_mkt) / p.sd_c          # model minus market, in SDs
    X["line"] = p.line
    X["lowline"] = (p.line < 10).astype(float)
    X["gap_low"] = X.gap * X.lowline
    X["rest1"] = (p.rest == 1).astype(float)
    X["bench"] = (p.starter_r5 < 0.5).astype(float)
    return X


def fit_stack(p):
    m = p[(p.season == DISC) & (p.alt_line == 0) & p.p_mod.notna() & p.p_mkt.notna()]
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(stack_features(m), m.y)
    coefs = dict(zip(stack_features(m).columns, clf.coef_[0].round(3)))
    ok = p.p_mod.notna() & p.p_mkt.notna()
    p = p.copy()
    p["p_stack"] = np.nan
    p.loc[ok, "p_stack"] = clf.predict_proba(stack_features(p[ok]))[:, 1]
    return p, coefs


def brier(p, col):
    m = p[(p.alt_line == 0) & p[col].notna()]
    return {s: round(float(np.mean((m[m.season == s][col] - m[m.season == s].y) ** 2)), 5) for s in (DISC, HOLD)}


def table(B, by):
    rows = []
    for k, x in B.groupby(by):
        for s in (DISC, HOLD):
            y = x[x.season == s]
            if len(y) >= 30:
                r = perf(y.pnl)
                rows.append({**({by: k} if isinstance(by, str) else dict(zip(by, k))), "season": s, "n": r["n"],
                             "roi": round(r["roi"], 4), "z": round(r["z"], 2)})
    return pd.DataFrame(rows)


def parlays(B: pd.DataFrame, legs=2, per_day=1, rr=False):
    """Per book per day: combine top-EV legs from DIFFERENT games into parlays at that book."""
    out = []
    for (d, book), x in B.groupby(["date", "book"]):
        x = x.sort_values("ev", ascending=False).drop_duplicates("game_id")
        if len(x) < legs:
            continue
        combos = itertools.combinations(range(min(len(x), 6)), legs) if rr else \
            [tuple(range(i * legs, (i + 1) * legs)) for i in range(per_day) if (i + 1) * legs <= len(x)]
        for c in combos:
            L = x.iloc[list(c)]
            dec = float(np.prod(L.dec))
            pnl = dec - 1 if L.win.all() else -1.0
            out.append(dict(date=d, season=int(L.season.iloc[0]), book=book, legs=legs, dec=dec, pnl=pnl,
                            ev_model=float(np.prod(1 + L.ev) - 1)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    p = pd.read_csv(OUT / "nba_props_frame.csv.gz", parse_dates=["date"], low_memory=False)
    p, coefs = fit_stack(p)
    print("stack coefs (fit on 2024-25 only):", coefs)
    for c in ("p_mkt", "p_mod", "p_stack"):
        print(f"  brier {c}: {brier(p, c)}")
    p["p_blend"] = 0.75 * p.p_mkt + 0.25 * p.p_mod
    rows = []
    for col in ("p_blend", "p_stack"):
        for thr in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
            B = pick_bets(p, col, thr)
            for s in (DISC, HOLD):
                x = B[B.season == s]
                r = perf(x.pnl)
                rows.append(dict(prob=col, thr=thr, season=s, n=r["n"], roi=round(r["roi"], 4), z=round(r["z"], 2)))
    R = pd.DataFrame(rows)
    print(R.pivot_table(index=["prob", "thr"], columns="season", values=["n", "roi", "z"]).to_string())
    R.to_csv(OUT / "nba_props_stack.csv", index=False)

    # choose operating point on DISCOVERY only: best discovery z among stack thresholds with n>=500
    import sys
    if len(sys.argv) > 2:
        col, thr = sys.argv[1], float(sys.argv[2])
    else:
        d = R[(R.season == DISC) & (R.n >= 500)].sort_values("z", ascending=False).iloc[0]
        col, thr = d.prob, d.thr
    print(f"\noperating point: {col} EV>={thr}")
    B = pick_bets(p, col, thr)
    B.to_csv(OUT / f"nba_props_bets_{col}_{thr}.csv", index=False)
    for by in ("side", "book"):
        print(table(B, by).pivot_table(index=by, columns="season", values=["n", "roi"]).to_string())
    B["line_bkt"] = pd.cut(B.line, [0, 9.5, 14.5, 19.5, 24.5, 60])
    print(table(B, "line_bkt").pivot_table(index="line_bkt", columns="season", values=["n", "roi"]).to_string())

    print("\nPARLAYS (same book, different games, top-EV legs of the day):")
    prow = []
    for legs, per_day, rr in ((2, 1, False), (2, 2, False), (3, 1, False), (2, 0, True), (3, 0, True), (4, 1, False)):
        T = parlays(B, legs, per_day, rr)
        for s in (DISC, HOLD):
            x = T[T.season == s]
            if len(x):
                r = perf(x.pnl)
                prow.append(dict(structure=f"{legs}-leg {'round-robin top6' if rr else f'top x{per_day}'}", season=s,
                                 tickets=r["n"], hit=round((x.pnl > 0).mean(), 3), roi=round(r["roi"], 4),
                                 units=round(r["units"], 1), z=round(r["z"], 2), avg_dec=round(x.dec.mean(), 2),
                                 max_dd=round(r["max_dd"], 1)))
    PR = pd.DataFrame(prow)
    print(PR.to_string())
    PR.to_csv(OUT / f"nba_props_parlays_{col}_{thr}.csv", index=False)
    s1 = B.groupby("season").pnl.agg(["size", "mean", "sum"])
    print("\nsingles:", s1.round(3).to_dict())
