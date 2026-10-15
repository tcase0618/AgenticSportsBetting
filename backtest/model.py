"""
Walk-forward ML "beat the closing line" models.

For each season s >= first_test, fit on ALL seasons < s, predict season s.
Targets (all relative to the market so the model must find what the line missed):
   ats : P(home covers)            -> bet side if P>thr (or <1-thr)
   tot : P(over)                   -> bet over/under
   ml  : P(home wins) blended vs. de-vigged market prob -> bet when edge > thr

Model: L2 logistic regression on standardized, median-imputed features
(deliberately low-variance; flexible models overfit betting noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .core import win_profit

BASE = ["rest", "su_streak", "ats_streak", "ou_streak", "gp", "ats_wpct", "su_wpct", "venue_run",
        "g_last7", "prev_margin", "prev_ats", "prev_ou",
        "ats_r3", "ats_r5", "ats_r10", "ou_r3", "ou_r5", "ou_r10", "mgn_r5", "mgn_r10", "wvm_r5", "wvm_r10"]


def feature_matrix(g: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=g.index)
    for b in BASE:
        X[f"h_{b}"], X[f"a_{b}"] = g[f"h_{b}"], g[f"a_{b}"]
    X["rest_diff"] = g["h_rest"] - g["a_rest"]
    X["spread_home"], X["total"] = g["spread_home"], g["total"]
    X["p_home_mkt"] = g["p_home_mkt"]
    X["elo_diff"] = g["elo_home"] - g["elo_away"]
    X["elo_vs_mkt"] = g["p_home_elo"] - g["p_home_mkt"]
    X["neutral"], X["playoff"] = g["neutral"], g["playoff"]
    for c in extra or []:
        X[c] = pd.to_numeric(g[c], errors="coerce")
    return X.astype(float)


def _fit_predict(Xtr, ytr, Xte, C=0.05):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=C, max_iter=2000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def walk_forward(g: pd.DataFrame, pnl: dict, first_test: int, extra=None, C=0.05) -> pd.DataFrame:
    """Out-of-sample predictions for every game in seasons >= first_test."""
    X = feature_matrix(g, extra)
    preds = pd.DataFrame(index=g.index, columns=["p_cover", "p_over", "p_win"], dtype=float)
    y_cov = np.sign(g["ats_home"])
    y_ou = np.sign(g["ou"])
    y_win = np.sign(g["margin"])
    for s in sorted(g.loc[g.season >= first_test, "season"].unique()):
        tr, te = g.season < s, g.season == s
        for col, y in (("p_cover", y_cov), ("p_over", y_ou), ("p_win", y_win)):
            m = tr & (y != 0) & y.notna() & X["p_home_mkt" if col == "p_win" else
                                                  ("spread_home" if col == "p_cover" else "total")].notna()
            if m.sum() < 500 or te.sum() == 0:
                continue
            Xtr = X[m]
            if col == "p_win":   # learn residual vs market: feature already includes p_home_mkt
                pass
            preds.loc[te, col] = _fit_predict(Xtr, (y[m] > 0).astype(int), X[te], C)
    return preds


def model_bets(g, pnl, preds, thresholds=(0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60),
               ml_edges=(0.02, 0.03, 0.04, 0.05, 0.07, 0.10)) -> dict:
    out = {}
    for thr in thresholds:
        p = preds["p_cover"]
        h, a = p >= thr, p <= 1 - thr
        out[("ats", f"model_p>={thr}", "model", "pick", "all")] = \
            pd.concat([pnl["ats_h"][h], pnl["ats_a"][a]]).dropna().sort_index()
        p = preds["p_over"]
        o, u = p >= thr, p <= 1 - thr
        out[("tot", f"model_p>={thr}", "model", "pick", "all")] = \
            pd.concat([pnl["tot_o"][o], pnl["tot_u"][u]]).dropna().sort_index()
    pw = preds["p_win"]
    for e in ml_edges:
        # edge vs the actual price we would pay (vigged), not the de-vigged prob
        bh = pw - 1 / (1 + win_profit(g["home_ml"])) >= e
        ba = (1 - pw) - 1 / (1 + win_profit(g["away_ml"])) >= e
        both = bh & ba
        out[("ml", f"model_edge>={e}", "model", "pick", "all")] = \
            pd.concat([pnl["ml_h"][bh & ~both], pnl["ml_a"][ba & ~both]]).dropna().sort_index()
    return out
