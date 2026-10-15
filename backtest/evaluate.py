"""
Selection protocol (pre-registered, identical for every sport):

  1. DISCOVERY  (train seasons only): keep a strategy if
        n_train >= MIN_TRAIN, ROI_train > 0 and one-sided p_train < 0.05.
  2. VALIDATION (holdout seasons, never touched during discovery):
        report ROI / z / p on holdout; Benjamini-Hochberg q-value across all
        discovery survivors controls the false-discovery rate.
  3. RANKING: survivors with holdout ROI > 0 and n_holdout >= MIN_HOLD are ranked by
        holdout z, tie-broken by full-sample shrunk ROI and season consistency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import perf, shrunk_roi


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def evaluate(strats: dict, g: pd.DataFrame, split_season: int, min_train=150, min_hold=60,
             odds_lookup: dict | None = None) -> pd.DataFrame:
    season = g["season"]
    rows = []
    for key, s in strats.items():
        tr = s[season.loc[s.index].values < split_season]
        ho = s[season.loc[s.index].values >= split_season]
        if len(tr) < min_train:
            continue
        ptr = perf(tr)
        rec = dict(market=key[0], condition=key[1], role=key[2], direction=key[3], phase=key[4],
                   n_train=ptr["n"], roi_train=ptr["roi"], p_train=ptr["p"], win_train=ptr["win_pct"])
        pho = perf(ho) if len(ho) else dict(n=0)
        rec.update(n_hold=pho.get("n", 0), roi_hold=pho.get("roi", np.nan), z_hold=pho.get("z", np.nan),
                   p_hold=pho.get("p", np.nan), win_hold=pho.get("win_pct", np.nan),
                   units_hold=pho.get("units", np.nan))
        pall = perf(s, seasons=season.loc[s.index])
        rec.update(n_all=pall["n"], roi_all=pall["roi"], z_all=pall["z"], units_all=pall["units"],
                   pct_pos_seasons=pall.get("pct_pos_seasons"), max_dd=pall["max_dd"],
                   shrunk_all=shrunk_roi(pall["roi"], pall["n"], sd=float(s.std()) if len(s) > 1 else 1.0))
        rows.append(rec)
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    R["discovered"] = (R.roi_train > 0) & (R.p_train < 0.05)
    R["q_hold"] = np.nan
    d = R.discovered & (R.n_hold >= min_hold)
    R.loc[d, "q_hold"] = bh_qvalues(R.loc[d, "p_hold"].fillna(1).values)
    R["validated"] = d & (R.roi_hold > 0)
    R = R.sort_values(["validated", "z_hold", "shrunk_all"], ascending=[False, False, False])
    return R.reset_index(drop=True)


def describe(key, s: pd.Series, g: pd.DataFrame, split_season: int) -> dict:
    season = g["season"].loc[s.index]
    by_season = s.groupby(season.values).agg(["count", "sum", "mean"]).rename(
        columns={"count": "bets", "sum": "units", "mean": "roi"})
    return dict(key=key, train=perf(s[season.values < split_season]),
                holdout=perf(s[season.values >= split_season]),
                all=perf(s, seasons=season), by_season=by_season)
