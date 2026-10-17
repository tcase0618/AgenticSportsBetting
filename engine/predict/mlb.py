"""MLB winner + final-score predictor (walk-forward, see engine/predict/SPEC.md).

    python -m engine.predict.mlb          # full backtest -> results/predict/mlb_predict.md, mlb_preds.parquet, mlb_model.pkl

Live: `predict(slate_df)`; slate_df needs the MLB script columns for the slate (output rows of engine.scripts.mlb:
mu_home, mu_away, margin, total, p_home_script, p_home_sim, p_over_script, rls_home/away, mu_home_core/away_core,
sp_home_eff/sp_away_eff/sp_*_n, park_home_eff/park_home_n, h_*/a_* starter-bullpen-fatigue columns, playoff) plus
optional prices home_ml, away_ml, total_line, over_odds, under_odds, spread_home.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from engine.predict.common_count import OUT, ROOT, CountCfg, add_market, logit, predict_count, run

SP = ["k", "csw", "depth", "sc", "new", "rest"]
GROUPS = {
    "script": ["s_logit", "sim_logit", "margin", "total", "rls_home", "rls_away", "core_diff", "lo_over_script"],
    "pitchers": ["sp_home_eff", "sp_away_eff", "sp_home_ln", "sp_away_ln"] + [f"h_opp_sp_{c}" for c in SP] + [f"a_opp_sp_{c}" for c in SP],
    "bullpen/fatigue/lineup": ["h_opp_pen3", "a_opp_pen3", "h_opp_yday", "h_own_yday", "a_opp_yday", "a_own_yday", "h_own_bat_k", "a_own_bat_k"],
    "park": ["park_home_eff", "park_ln"],
    "context": ["playoff", "season_frac"],
}


def build(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = add_market(d, p_bounds=(0.15, 0.85))
    d["sim_logit"] = logit(d.p_home_sim.astype(float))
    d["core_diff"] = d.mu_home_core.astype(float) - d.mu_away_core.astype(float)
    d["lo_over_script"] = logit(d.p_over_script.astype(float).fillna(0.5))
    d["sp_home_ln"] = np.log1p(d.sp_home_n.astype(float).fillna(0))
    d["sp_away_ln"] = np.log1p(d.sp_away_n.astype(float).fillna(0))
    d["park_ln"] = np.log1p(d.park_home_n.astype(float).fillna(0))
    start = d.groupby("season").date.transform("min")
    d["season_frac"] = (d.date - start).dt.days / 200.0
    for cols in GROUPS.values():
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).fillna(0.0)
    return d


CFG = CountCfg(sport="mlb", label="MLB", unit="runs", line_name="run line", build=build, groups=GROUPS, maxg=25,
               margin_bands=(1, 2), rl_be_fav=0.42, rl_be_dog=0.615, hi_tier=0.65,
               notes=["Seasons 2016-2026 (2026 results from the live feed; closing prices only for Sep 14-Oct 14 2026, 210 games).",
                      "Integrity note (results/KEYS.md): MLB 2021+ 'closing' prices failed a steam-move integrity check, so they may "
                      "be closer to openers / stale quotes than true closes. They are still a pre-game number, so they are used as a "
                      "feature after odds-ratio de-vig; clearly broken quotes are dropped (rule below). The model-vs-market comparison "
                      "for 2021+ is against these prices.",
                      "Run line: data has the ±1.5 side but no run-line prices, so break-even uses typical prices (-1.5 at +140, +1.5 at -160)."])


def predict(slate_df: pd.DataFrame) -> pd.DataFrame:
    with open(OUT / "mlb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return predict_count(model, build(slate_df))


if __name__ == "__main__":
    raw = pd.read_parquet(ROOT / "results" / "scripts" / "mlb_games.parquet")
    run(CFG, raw)
