"""NHL winner + final-score predictor (walk-forward, see engine/predict/SPEC.md).

    python -m engine.predict.nhl          # full backtest -> results/predict/nhl_predict.md, nhl_preds.parquet, nhl_model.pkl

Live: `predict(slate_df)`; slate_df needs the NHL script columns for the slate (rows like engine.scripts.nhl output:
mu_home, mu_away, margin, total, p_home_script, p_home_sim, p_over_script, rls_home/away, mu_home_core/away_core,
goalie_home_eff/goalie_away_eff/goalie_*_n, home_g_backup/away_g_backup, h_opp_g_*/a_opp_g_*, home_b2b/away_b2b,
h_own_sf/h_opp_sa/h_own_ppg/h_opp_ppga (+ a_*), h_pace_known, shots_home/shots_away (scripted shots), playoff, season)
plus optional prices home_ml, away_ml, total_line, over_odds, under_odds, spread_home. Season = end year (2027 = 2026-27).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from engine.predict.common_count import OUT, ROOT, CountCfg, add_market, logit, predict_count, run

GROUPS = {
    "script": ["s_logit", "sim_logit", "margin", "total", "rls_home", "rls_away", "core_diff", "lo_over_script"],
    "goalies": ["goalie_home_eff", "goalie_away_eff", "goalie_home_ln", "goalie_away_ln", "home_g_backup", "away_g_backup",
                "h_opp_g_new", "a_opp_g_new"],
    "rest (back-to-back)": ["home_b2b", "away_b2b"],
    "pace/special teams": ["h_own_sf", "h_opp_sa", "h_own_ppg", "h_opp_ppga", "a_own_sf", "a_opp_sa", "a_own_ppg", "a_opp_ppga",
                           "h_pace_known", "shots_home", "shots_away"],
    "context": ["playoff", "season_frac"],
}


def build(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = add_market(d, p_bounds=(0.10, 0.90))
    d["sim_logit"] = logit(d.p_home_sim.astype(float))
    d["core_diff"] = d.mu_home_core.astype(float) - d.mu_away_core.astype(float)
    d["lo_over_script"] = logit(d.p_over_script.astype(float).fillna(0.5))
    d["goalie_home_ln"] = np.log1p(d.goalie_home_n.astype(float).fillna(0))
    d["goalie_away_ln"] = np.log1p(d.goalie_away_n.astype(float).fillna(0))
    start = d.groupby("season").date.transform("min")
    d["season_frac"] = (d.date - start).dt.days / 200.0
    for cols in GROUPS.values():
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).fillna(0.0)
    return d


CFG = CountCfg(sport="nhl", label="NHL", unit="goals", line_name="puck line", build=build, groups=GROUPS, maxg=12,
               margin_bands=(1, 2), rl_be_fav=0.37, rl_be_dog=0.677, hi_tier=0.65,
               notes=["Seasons labelled by end year (2021 = 2020-21 COVID season; 2027 = 2026-27 in progress). Final scores include "
                      "the OT/shootout winner, so there are no ties.",
                      "Closing moneylines are missing for most of 2022-23 and half of 2025-26; those games use the script as the "
                      "market fallback (has_mkt=0) and are excluded from model-vs-market comparisons.",
                      "Starting goalie (backup / new flags, Kalman goalie effect) is the actual starter, which is normally confirmed "
                      "before puck drop; treat goalie features as 'confirmed starter' information.",
                      "Puck line: data has the ±1.5 side but no puck-line prices, so break-even uses typical prices (-1.5 at +170, +1.5 at -210)."])


def predict(slate_df: pd.DataFrame) -> pd.DataFrame:
    with open(OUT / "nhl_model.pkl", "rb") as f:
        model = pickle.load(f)
    return predict_count(model, build(slate_df))


if __name__ == "__main__":
    raw = pd.read_parquet(ROOT / "results" / "scripts" / "nhl_games.parquet")
    run(CFG, raw)
