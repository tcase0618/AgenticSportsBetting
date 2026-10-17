"""
Edge hunter "situational": 1-2 condition situational rules built on the rich pre-game features the script
engines provide (results/scripts/<sport>_games.parquet) and the walk-forward predictors
(results/predict/<sport>_preds.parquet).

Protocol (engine/edge_hunt/PROTOCOL.md):
  * discovery = seasons <= 2022 (data starts 2016; predictor columns start 2018), lockbox = 2023+.
  * every rule is a conjunction of 1-2 boolean "atoms" x market (side ML, side ATS, game total) x direction.
  * thresholds of continuous atoms = quantiles of the DISCOVERY rows only (stored in situational_thresholds.json
    so scan() reuses them; nothing is re-fit on the lockbox).
  * screening: one-sided z on discovery unit P&L, Benjamini-Hochberg q < 0.10 inside each family
    (A = pre-registered targeted hypotheses, B = broad 1-2 atom scan), min n per sport, ROI > 0.
  * lockbox once per surviving rule; verdict = lockbox ROI > 0 and (p < 0.05 / n_final or BH q < 0.10 across
    lockbox tests) and >= 60% of seasons positive in the full sample.

Prices: ML at the real closing American prices (broken quotes dropped). Spreads/totals: the data carry no juice for
NBA/NFL/NCAAF/NCAAM spreads & totals -> -110 ASSUMED (flagged). NHL totals have real over/under prices 2024+ and MLB
totals 2022+; earlier seasons -110 ASSUMED (flagged per bet). MLB run line / NHL puck line are NOT tested (no juice).
MLB 2021+ 'closing' prices failed an integrity check (results/KEYS.md): every MLB lockbox result is provisional.

Side rules are evaluated from the perspective of the team bet on ("back"). The atom set is symmetric (every t_*
atom has an o_* twin, home/away, script-above/below-market), so "fade X" = "back the opponent" = a back rule on the
mirrored atoms; the targeted family lists fades explicitly.

  python -m engine.edge_hunt.situational            # full research run -> results/edge_hunt/situational*.{md,csv}
  python -m engine.edge_hunt.situational --scan     # today's qualifying bets for the surviving rules
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
SCR, PRD, OUT = ROOT / "results" / "scripts", ROOT / "results" / "predict", ROOT / "results" / "edge_hunt"
THR_FILE = OUT / "situational_thresholds.json"
RULES_FILE = OUT / "situational_rules.json"

SPORTS = ["nba", "nfl", "nhl", "mlb", "ncaaf", "ncaam"]
DISC_END, LOCK_START, FIRST = 2022, 2023, 2016
MIN_N = dict(nba=200, nhl=200, mlb=300, ncaam=300, nfl=80, ncaaf=100)
MIN_N_LOCK_INFO = 30
ATS_OK = {"nba", "nfl", "ncaaf", "ncaam"}          # nhl puck line / mlb run line: no juice in the data -> not tested
P5 = {"SEC", "Big Ten", "ACC", "Big 12", "Pac-12"}
WIN110 = 100 / 110


# ------------------------------------------------------------------ helpers
def dec(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def imp(o):
    o = np.asarray(o, float)
    o = np.where(np.isfinite(o) & (np.abs(o) >= 100), o, -110.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 0, 100 / (o + 100), np.abs(o) / (np.abs(o) + 100))


def ml_ok(h, a):
    h, a = np.asarray(h, float), np.asarray(a, float)
    good = np.isfinite(h) & np.isfinite(a) & (np.abs(h) >= 100) & (np.abs(a) >= 100) & (np.abs(h) <= 5000) & (np.abs(a) <= 5000)
    s = np.where(good, imp(np.where(good, h, -110)) + imp(np.where(good, a, -110)), np.nan)
    return good & (s >= 1.0) & (s <= 1.15)


def zstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    m, sd = x.mean(), x.std(ddof=1)
    z = m / (sd / math.sqrt(n)) if sd > 0 else np.nan
    return n, m, z, float(norm.sf(z)) if np.isfinite(z) else np.nan


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    m = len(pv)
    r = pv[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(m)
    qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


# ------------------------------------------------------------------ data
def _merge_preds(sport, g):
    p = pd.read_parquet(PRD / f"{sport}_preds.parquet")
    p["game_id"] = p.game_id.astype(str)
    keep = {"p_home": "p_model_home", "pred_total": "model_total"}
    p = p[["game_id"] + list(keep)].rename(columns=keep)
    return g.merge(p, on="game_id", how="left")


def load_games(sport: str) -> pd.DataFrame:
    g = pd.read_parquet(SCR / f"{sport}_games.parquet")
    g["game_id"] = g.game_id.astype(str)
    g["date"] = pd.to_datetime(g.date)
    g = _merge_preds(sport, g)
    g = g[g.home_score.notna() & g.away_score.notna() & (g.season >= FIRST)].copy()
    if sport in ("nba", "ncaam"):
        g["p_script_home"] = norm.cdf(g["margin"] / g["sd_margin"])
    elif sport == "ncaaf":
        g["p_script_home"] = g["p_home_win"]
    else:
        g["p_script_home"] = g["p_home_script"]
    g["script_margin"] = g["margin"]
    g["script_total"] = g["total"]
    g["ml_ok"] = ml_ok(g.home_ml, g.away_ml)
    ih, ia = imp(g.home_ml.fillna(-110)), imp(g.away_ml.fillna(-110))
    g["p_mkt_home"] = np.where(g.ml_ok, ih / (ih + ia), np.nan)
    # ---- sport-specific derived columns
    if sport == "nfl":
        g = _nfl_qb_change(g)
        g["rest_home"], g["rest_away"] = g.home_rest, g.away_rest
    if sport == "nhl":
        g["b2b_home"], g["b2b_away"] = g.home_b2b, g.away_b2b
    if sport == "ncaaf":
        g["rest_home"], g["rest_away"] = g.rest_h.clip(upper=21), g.rest_a.clip(upper=21)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def _nfl_qb_change(g):
    """QB change = this game's starting QB differs from the team's previous game's starter (pre-game known)."""
    rows = pd.concat([g[["game_id", "date", "home", "home_qb_name"]].set_axis(["game_id", "date", "team", "qb"], axis=1),
                      g[["game_id", "date", "away", "away_qb_name"]].set_axis(["game_id", "date", "team", "qb"], axis=1)])
    rows = rows.sort_values(["team", "date"])
    rows["prev"] = rows.groupby("team").qb.shift()
    rows["qbchg"] = ((rows.qb != rows.prev) & rows.prev.notna() & rows.qb.notna()).astype(int)
    m = rows.set_index(["game_id", "team"]).qbchg
    g["qbchg_home"] = [m.get((i, t), 0) for i, t in zip(g.game_id, g.home)]
    g["qbchg_away"] = [m.get((i, t), 0) for i, t in zip(g.game_id, g.away)]
    return g


# (t_name, home column, away column)  -> t_<name> / o_<name> in the long frame
PAIRS = {
    "nba": [("rest", "rest_home", "rest_away"), ("b2b", "b2b_home", "b2b_away"), ("travel", "travel_home_kkm", "travel_away_kkm"),
            ("tz", "tz_home", "tz_away"), ("miss_min", "miss_min_home", "miss_min_away"), ("miss_st", "miss_st_home", "miss_st_away"),
            ("games", "games_home", "games_away")],
    "nfl": [("rest", "rest_home", "rest_away"), ("qbchg", "qbchg_home", "qbchg_away"), ("qbd", "home_qb_delta", "away_qb_delta")],
    "nhl": [("b2b", "b2b_home", "b2b_away"), ("gbk", "home_g_backup", "away_g_backup"), ("geff", "goalie_home_eff", "goalie_away_eff"),
            ("gnew", "a_opp_g_new", "h_opp_g_new")],       # h_opp_g_new = the goalie the HOME team faces is new = away goalie new
    "mlb": [("sp", "sp_home_eff", "sp_away_eff"), ("spnew", "a_opp_sp_new", "h_opp_sp_new"), ("pen3", "a_opp_pen3", "h_opp_pen3"),
            ("yday", "h_own_yday", "a_own_yday")],
    "ncaaf": [("rest", "rest_home", "rest_away")],
    "ncaam": [("rest", "rest_home", "rest_away"), ("gs", "gs_home", "gs_away")],
}


def long_frame(sport: str, g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (home rows 0..N-1, away rows N..2N-1): row i's opponent row is (i+N) % 2N."""
    parts = []
    for home in (True, False):
        s = 1 if home else -1
        d = pd.DataFrame({"game_id": g.game_id.values, "date": g.date.values, "season": g.season.values,
                          "team": (g.home if home else g.away).values, "opp": (g.away if home else g.home).values,
                          "is_home": int(home)})
        d["neutral"] = g["neutral"].fillna(False).astype(bool).values if "neutral" in g else False
        d["playoff"] = (g["playoff"].fillna(False).astype(bool).values if "playoff" in g else
                        (g.get("game_type", pd.Series("REG", index=g.index)) != "REG").values if sport == "nfl" else
                        g.get("postseason", pd.Series(False, index=g.index)).fillna(False).astype(bool).values)
        d["margin_t"] = (s * (g.home_score - g.away_score)).values
        d["ml_t"] = (g.home_ml if home else g.away_ml).values
        d["ml_ok"] = g.ml_ok.values
        d["spread_t"] = (g.spread_home if home else -g.spread_home).values
        pm = g.p_mkt_home.values
        d["p_mkt"] = pm if home else 1 - pm
        ps = g.p_script_home.values
        d["p_script"] = ps if home else 1 - ps
        pmod = g.p_model_home.values
        d["p_model"] = pmod if home else 1 - pmod
        d["sedge"] = d.p_script - d.p_mkt
        d["medge"] = d.p_model - d.p_mkt
        d["aedge"] = (s * g.script_margin).values + d.spread_t       # script margin minus market margin, team view
        for name, hc, ac in PAIRS.get(sport, []):
            d["t_" + name] = (g[hc] if home else g[ac]).values.astype(float)
            d["o_" + name] = (g[ac] if home else g[hc]).values.astype(float)
        if sport == "ncaaf":
            d["t_p5"] = (g.home_conference if home else g.away_conference).isin(P5).values.astype(float)
            d["o_p5"] = (g.away_conference if home else g.home_conference).isin(P5).values.astype(float)
            d["conf_game"] = (g.home_conference == g.away_conference).values.astype(float)
            d["t_epa"] = (s * g.epa_net).values
        if sport == "ncaam":
            d["conf_game"] = g.same_conf.values.astype(float)
        if sport == "nfl":
            d["div_game"] = g.div_game.fillna(0).values.astype(float)
        parts.append(d)
    L = pd.concat(parts, ignore_index=True)
    # P&L of backing the team
    o = L.ml_t.values
    L["pnl_ml"] = np.where(L.ml_ok, np.where(L.margin_t > 0, dec(np.where(L.ml_ok, o, -110)) - 1,
                                             np.where(L.margin_t < 0, -1.0, 0.0)), np.nan)
    c = L.margin_t + L.spread_t
    L["pnl_ats"] = np.where(L.spread_t.notna() & (sport in ATS_OK), np.where(c > 0, WIN110, np.where(c < 0, -1.0, 0.0)), np.nan)
    if "rest" in [p[0] for p in PAIRS.get(sport, [])]:
        L["rdiff"] = L.t_rest - L.o_rest
    return L


def total_frame(sport: str, g: pd.DataFrame) -> pd.DataFrame:
    T = g.copy()
    act = T.home_score + T.away_score
    line = T.total_line
    has = line.notna()
    oo = T["over_odds"] if "over_odds" in T else pd.Series(np.nan, index=T.index)
    uo = T["under_odds"] if "under_odds" in T else pd.Series(np.nan, index=T.index)
    real = oo.notna() & uo.notna() & (oo.abs() >= 100) & (uo.abs() >= 100)
    T["juice_assumed"] = (~real).astype(int)
    ov, un = np.where(real, oo, -110.0), np.where(real, uo, -110.0)
    T["pnl_over"] = np.where(has, np.where(act > line, dec(ov) - 1, np.where(act < line, -1.0, 0.0)), np.nan)
    T["pnl_under"] = np.where(has, np.where(act < line, dec(un) - 1, np.where(act > line, -1.0, 0.0)), np.nan)
    T["tedge"] = T.script_total - line
    T["tmedge"] = T.model_total - line
    T["abs_spread"] = T.spread_home.abs()
    T["playoff_"] = (T["playoff"].fillna(False).astype(bool) if "playoff" in T else
                     (T.game_type != "REG") if sport == "nfl" else T.get("postseason", pd.Series(False, index=T.index)).fillna(False).astype(bool))
    if sport == "nba":
        T["miss_sum"] = T.miss_min_home + T.miss_min_away
        T["miss_st_sum"] = T.miss_st_home + T.miss_st_away
        T["travel_sum"] = T.travel_home_kkm + T.travel_away_kkm
        T["any_b2b"] = ((T.b2b_home + T.b2b_away) > 0).astype(float)
        T["both_b2b"] = ((T.b2b_home + T.b2b_away) == 2).astype(float)
        T["any_tz2"] = ((T.tz_home.abs() >= 2) | (T.tz_away.abs() >= 2)).astype(float)
    if sport == "nfl":
        T["dome"] = T.roof.isin(["dome", "closed"]).astype(float)
        T["any_qbchg"] = ((T.qbchg_home + T.qbchg_away) > 0).astype(float)
        T["short_week"] = ((T.home_rest <= 5) | (T.away_rest <= 5)).astype(float)
        T["div_game_"] = T.div_game.fillna(0).astype(float)
    if sport == "nhl":
        T["any_gbk"] = ((T.home_g_backup + T.away_g_backup) > 0).astype(float)
        T["both_gbk"] = ((T.home_g_backup + T.away_g_backup) == 2).astype(float)
        T["geff_sum"] = T.goalie_home_eff + T.goalie_away_eff
        T["any_b2b"] = ((T.home_b2b + T.away_b2b) > 0).astype(float)
        T["both_b2b"] = ((T.home_b2b + T.away_b2b) == 2).astype(float)
        T["any_gnew"] = ((T.h_opp_g_new + T.a_opp_g_new) > 0).astype(float)
    if sport == "mlb":
        T["sp_sum"] = T.sp_home_eff + T.sp_away_eff
        T["pen_sum"] = T.h_opp_pen3 + T.a_opp_pen3
        T["any_spnew"] = ((T.h_opp_sp_new + T.a_opp_sp_new) > 0).astype(float)
        T["both_rested"] = ((T.h_own_yday == 0) & (T.a_own_yday == 0)).astype(float)
    if sport == "ncaaf":
        T["conf_game"] = (T.home_conference == T.away_conference).astype(float)
        T["fcs"] = T.fbs_vs_fcs.astype(float)
        T["p5g5"] = T.p5_vs_g5.astype(float)
    if sport == "ncaam":
        T["conf_game"] = T.same_conf.astype(float)
        T["tempo_sum"] = T.tempo_home + T.tempo_away
        T["early"] = (np.minimum(T.gs_home, T.gs_away) <= 5).astype(float)
    if "neutral" in T:
        T["neutral_"] = T.neutral.fillna(False).astype(float)
    return T


# ------------------------------------------------------------------ atoms
# atom spec: (name, column, kind, param); kind in ge/le (absolute), qhi/qlo (discovery quantile), eq
def side_atoms(sport):
    A = [("home", "is_home", "eq", 1), ("away", "is_home", "eq", 0),
         ("fav", "p_mkt", "ge", 0.5), ("dog", "p_mkt", "le", 0.5), ("bigfav", "p_mkt", "ge", 0.67), ("bigdog", "p_mkt", "le", 0.33),
         ("script>mkt top10%", "sedge", "qhi", 0.9), ("script>mkt top20%", "sedge", "qhi", 0.8),
         ("script<mkt bot10%", "sedge", "qlo", 0.1), ("script<mkt bot20%", "sedge", "qlo", 0.2),
         ("model>mkt top10%", "medge", "qhi", 0.9), ("model<mkt bot10%", "medge", "qlo", 0.1),
         ("playoff", "playoff", "eq", 1)]
    if sport in ATS_OK:
        A += [("script margin>mkt top10%", "aedge", "qhi", 0.9), ("script margin<mkt bot10%", "aedge", "qlo", 0.1)]
    if sport in ("nba", "ncaaf", "ncaam", "nfl"):
        A += [("rest adv>=2", "rdiff", "ge", 2), ("rest disadv<=-2", "rdiff", "le", -2)]
    if sport == "nba":
        A += [("t b2b", "t_b2b", "eq", 1), ("o b2b", "o_b2b", "eq", 1), ("t travel top20%", "t_travel", "qhi", 0.8),
              ("o travel top20%", "o_travel", "qhi", 0.8), ("t tz>=2 east", "t_tz", "ge", 2), ("t tz<=-2 west", "t_tz", "le", -2),
              ("o tz>=2 east", "o_tz", "ge", 2), ("o tz<=-2 west", "o_tz", "le", -2),
              ("t miss_min top10%", "t_miss_min", "qhi", 0.9), ("o miss_min top10%", "o_miss_min", "qhi", 0.9),
              ("t miss_st>=2", "t_miss_st", "ge", 2), ("o miss_st>=2", "o_miss_st", "ge", 2),
              ("t early (<10g)", "t_games", "le", 9)]
    if sport == "nfl":
        A += [("t qb change", "t_qbchg", "eq", 1), ("o qb change", "o_qbchg", "eq", 1),
              ("t qb_delta bot10%", "t_qbd", "qlo", 0.1), ("o qb_delta bot10%", "o_qbd", "qlo", 0.1),
              ("t short week", "t_rest", "le", 5), ("o short week", "o_rest", "le", 5),
              ("t off bye", "t_rest", "ge", 13), ("o off bye", "o_rest", "ge", 13), ("div game", "div_game", "eq", 1)]
    if sport == "nhl":
        A += [("t b2b", "t_b2b", "eq", 1), ("o b2b", "o_b2b", "eq", 1), ("t backup G", "t_gbk", "eq", 1), ("o backup G", "o_gbk", "eq", 1),
              ("t goalie top10%", "t_geff", "qhi", 0.9), ("t goalie bot10%", "t_geff", "qlo", 0.1),
              ("o goalie top10%", "o_geff", "qhi", 0.9), ("o goalie bot10%", "o_geff", "qlo", 0.1),
              ("t new G", "t_gnew", "eq", 1), ("o new G", "o_gnew", "eq", 1)]
    if sport == "mlb":
        A += [("t SP top10%", "t_sp", "qhi", 0.9), ("t SP bot10%", "t_sp", "qlo", 0.1), ("o SP top10%", "o_sp", "qhi", 0.9),
              ("o SP bot10%", "o_sp", "qlo", 0.1), ("t SP new", "t_spnew", "eq", 1), ("o SP new", "o_spnew", "eq", 1),
              ("t pen taxed top10%", "t_pen3", "qhi", 0.9), ("o pen taxed top10%", "o_pen3", "qhi", 0.9),
              ("t no game yday", "t_yday", "eq", 0), ("o no game yday", "o_yday", "eq", 0)]
    if sport == "ncaaf":
        A += [("t P5", "t_p5", "eq", 1), ("o P5", "o_p5", "eq", 1), ("conf game", "conf_game", "eq", 1),
              ("t epa_net top10%", "t_epa", "qhi", 0.9), ("t epa_net bot10%", "t_epa", "qlo", 0.1), ("neutral", "neutral", "eq", 1)]
    if sport == "ncaam":
        A += [("conf game", "conf_game", "eq", 1), ("neutral", "neutral", "eq", 1), ("t early (<=5g)", "t_gs", "le", 5)]
    return A


def total_atoms(sport):
    A = [("script total>line top10%", "tedge", "qhi", 0.9), ("script total>line top20%", "tedge", "qhi", 0.8),
         ("script total<line bot10%", "tedge", "qlo", 0.1), ("script total<line bot20%", "tedge", "qlo", 0.2),
         ("model total>line top10%", "tmedge", "qhi", 0.9), ("model total<line bot10%", "tmedge", "qlo", 0.1),
         ("line top10%", "total_line", "qhi", 0.9), ("line bot10%", "total_line", "qlo", 0.1),
         ("abs spread top10%", "abs_spread", "qhi", 0.9), ("playoff", "playoff_", "eq", 1)]
    if sport == "nba":
        A += [("miss_min sum top10%", "miss_sum", "qhi", 0.9), ("miss_st sum>=3", "miss_st_sum", "ge", 3),
              ("pace clash top10%", "pace_clash", "qhi", 0.9), ("pace clash bot10%", "pace_clash", "qlo", 0.1),
              ("travel sum top10%", "travel_sum", "qhi", 0.9), ("any b2b", "any_b2b", "eq", 1), ("both b2b", "both_b2b", "eq", 1),
              ("any |tz|>=2", "any_tz2", "eq", 1)]
    if sport == "nfl":
        A += [("wind>=12", "wind", "ge", 12), ("temp<=32", "temp", "le", 32), ("dome", "dome", "eq", 1), ("div game", "div_game_", "eq", 1),
              ("any qb change", "any_qbchg", "eq", 1), ("short week", "short_week", "eq", 1)]
    if sport == "nhl":
        A += [("any backup G", "any_gbk", "eq", 1), ("both backup G", "both_gbk", "eq", 1), ("goalies sum top10%", "geff_sum", "qhi", 0.9),
              ("goalies sum bot10%", "geff_sum", "qlo", 0.1), ("any b2b", "any_b2b", "eq", 1), ("both b2b", "both_b2b", "eq", 1),
              ("any new G", "any_gnew", "eq", 1)]
    if sport == "mlb":
        A += [("park top10%", "park_home_eff", "qhi", 0.9), ("park bot10%", "park_home_eff", "qlo", 0.1),
              ("SP sum top10%", "sp_sum", "qhi", 0.9), ("SP sum bot10%", "sp_sum", "qlo", 0.1), ("pens taxed top10%", "pen_sum", "qhi", 0.9),
              ("any SP new", "any_spnew", "eq", 1), ("both no game yday", "both_rested", "eq", 1)]
    if sport == "ncaaf":
        A += [("conf game", "conf_game", "eq", 1), ("FBS vs FCS", "fcs", "eq", 1), ("P5 vs G5", "p5g5", "eq", 1),
              ("epa_sum top10%", "epa_sum", "qhi", 0.9), ("epa_sum bot10%", "epa_sum", "qlo", 0.1),
              ("plays top10%", "plays_script", "qhi", 0.9), ("plays bot10%", "plays_script", "qlo", 0.1), ("neutral", "neutral_", "eq", 1)]
    if sport == "ncaam":
        A += [("conf game", "conf_game", "eq", 1), ("neutral", "neutral_", "eq", 1), ("tempo sum top10%", "tempo_sum", "qhi", 0.9),
              ("tempo sum bot10%", "tempo_sum", "qlo", 0.1), ("early season", "early", "eq", 1)]
    return A


def fit_thresholds(sport, L, T):
    thr = {}
    for frame, specs in ((L, side_atoms(sport)), (T, total_atoms(sport))):
        disc = frame[(frame.season <= DISC_END)]
        for name, col, kind, q in specs:
            if kind in ("qhi", "qlo"):
                thr[f"{col}|{q}"] = float(disc[col].quantile(q))
    return thr


def atom_mask(frame, spec, thr):
    name, col, kind, q = spec
    if col not in frame:
        return np.zeros(len(frame), bool)
    x = pd.to_numeric(frame[col], errors="coerce").values.astype(float)
    with np.errstate(invalid="ignore"):
        if kind == "eq":
            return x == q
        if kind == "ge":
            return x >= q
        if kind == "le":
            return x <= q
        if kind == "qhi":
            return x >= thr[f"{col}|{q}"]
        if kind == "qlo":
            return x <= thr[f"{col}|{q}"]
    raise ValueError(kind)


# ------------------------------------------------------------------ targeted family (pre-registered hypotheses)
ML_T = (0.05, 0.10, 0.15)
ATS_T = dict(nba=(2, 4, 6), nfl=(2, 4, 6), ncaaf=(3, 6, 9), ncaam=(2, 4, 6))
TOT_T = dict(nba=(3, 6, 9), nfl=(2, 4, 6), ncaaf=(3, 6, 9), ncaam=(3, 6, 9), nhl=(0.3, 0.5, 0.8), mlb=(0.5, 1.0, 1.5))


def targeted_rules(sport):
    """(rule name, frame 'side'|'tot', [atom specs], market, direction)."""
    R = []
    for a in ML_T:
        R.append((f"follow script ML edge >= {a:.2f}", "side", [("sedge>=", "sedge", "ge", a)], "ml", "back"))
        R.append((f"fade script ML edge >= {a:.2f}", "side", [("sedge>=", "sedge", "ge", a)], "ml", "fade"))
    if sport in ATS_OK:
        for b in ATS_T[sport]:
            R.append((f"follow script ATS edge >= {b} pts", "side", [("aedge>=", "aedge", "ge", b)], "ats", "back"))
            R.append((f"fade script ATS edge >= {b} pts", "side", [("aedge>=", "aedge", "ge", b)], "ats", "fade"))
    for c in TOT_T[sport]:
        R.append((f"follow script total >= line+{c}", "tot", [("tedge>=", "tedge", "ge", c)], "tot", "over"))
        R.append((f"fade script total >= line+{c}", "tot", [("tedge>=", "tedge", "ge", c)], "tot", "under"))
        R.append((f"follow script total <= line-{c}", "tot", [("tedge<=", "tedge", "le", -c)], "tot", "under"))
        R.append((f"fade script total <= line-{c}", "tot", [("tedge<=", "tedge", "le", -c)], "tot", "over"))
    sides = ["ml"] + (["ats"] if sport in ATS_OK else [])
    def both(name, atoms, mk):
        return [(f"back {name}", "side", atoms, mk, "back"), (f"fade {name}", "side", atoms, mk, "fade")]
    if sport == "nhl":
        R += both("team vs backup G (own starter)", [("o_gbk", "o_gbk", "eq", 1), ("t_gbk0", "t_gbk", "eq", 0)], "ml")
        R += both("team starting backup G", [("t_gbk", "t_gbk", "eq", 1)], "ml")
        R += both("backup G on b2b", [("t_gbk", "t_gbk", "eq", 1), ("t_b2b", "t_b2b", "eq", 1)], "ml")
        R += both("team on b2b vs rested", [("t_b2b", "t_b2b", "eq", 1), ("o_b2b0", "o_b2b", "eq", 0)], "ml")
        for d in ("over", "under"):
            R.append((f"{d}: any backup G", "tot", [("any_gbk", "any_gbk", "eq", 1)], "tot", d))
            R.append((f"{d}: both backup G", "tot", [("both_gbk", "both_gbk", "eq", 1)], "tot", d))
    if sport == "nfl":
        for mk in sides:
            R += both("team with QB change", [("t_qbchg", "t_qbchg", "eq", 1)], mk)
            R += both("QB change, opp no change", [("t_qbchg", "t_qbchg", "eq", 1), ("o_qbchg0", "o_qbchg", "eq", 0)], mk)
            R += both("QB much worse than base (qb_delta<=-0.08)", [("t_qbd", "t_qbd", "le", -0.08)], mk)
        for d in ("over", "under"):
            R.append((f"{d}: any QB change", "tot", [("any_qbchg", "any_qbchg", "eq", 1)], "tot", d))
    if sport == "nba":
        for mk in sides:
            R += both("team heavy missing minutes (top10%)", [("t_miss", "t_miss_min", "qhi", 0.9)], mk)
            R += both("team missing >=2 starters, opp <=0", [("t_mst", "t_miss_st", "ge", 2), ("o_mst0", "o_miss_st", "le", 0)], mk)
            R += both("b2b vs rested opp (rest>=2)", [("t_b2b", "t_b2b", "eq", 1), ("o_rest2", "o_rest", "ge", 2)], mk)
            R += both("b2b + travel top20%", [("t_b2b", "t_b2b", "eq", 1), ("t_trav", "t_travel", "qhi", 0.8)], mk)
            R += both("b2b + |tz|>=2 (abs)", [("t_b2b", "t_b2b", "eq", 1), ("t_tzabs", "t_tzabs", "ge", 2)], mk)
            R += both("rest disadvantage <=-2 + travel top20%", [("rd", "rdiff", "le", -2), ("t_trav", "t_travel", "qhi", 0.8)], mk)
        for d in ("over", "under"):
            R.append((f"{d}: missing minutes sum top10%", "tot", [("ms", "miss_sum", "qhi", 0.9)], "tot", d))
            R.append((f"{d}: both teams b2b", "tot", [("bb", "both_b2b", "eq", 1)], "tot", d))
    if sport == "mlb":
        R += both("team vs debut/new SP", [("o_spnew", "o_spnew", "eq", 1)], "ml")
        R += both("opp bullpen taxed (top10%) + own pen fresh", [("op", "o_pen3", "qhi", 0.9), ("tp", "t_pen3", "le", 0)], "ml")
    if sport == "ncaaf":
        for mk in sides:
            R += both("rest adv>=4 in conf game", [("rd", "rdiff", "ge", 4), ("cg", "conf_game", "eq", 1)], mk)
    if sport == "ncaam":
        for mk in sides:
            R += both("rest adv>=2 in conf game", [("rd", "rdiff", "ge", 2), ("cg", "conf_game", "eq", 1)], mk)
    return R


# ------------------------------------------------------------------ evaluation
def _pnl_vec(frame, market, direction):
    if market == "tot":
        return frame["pnl_over" if direction == "over" else "pnl_under"].values
    x = frame["pnl_" + market].values
    if direction == "fade":                       # bet the opponent row (i + N) % 2N
        N = len(x) // 2
        x = np.concatenate([x[N:], x[:N]])
    return x


def evaluate_rule(frame, mask, market, direction, seasons_mask):
    x = _pnl_vec(frame, market, direction)
    m = mask & np.isfinite(x) & seasons_mask
    return x[m], m


def broad_scan(sport, L, T, thr):
    """Vectorised 1-2 atom conjunctions. Returns a DataFrame of discovery stats per (rule, market, direction)."""
    rows = []
    disc_L = (L.season <= DISC_END).values
    disc_T = (T.season <= DISC_END).values
    for frame, specs, markets, dmask in ((L, side_atoms(sport), [("ml", "back")] + ([("ats", "back")] if sport in ATS_OK else []), disc_L),
                                         (T, total_atoms(sport), [("tot", "over"), ("tot", "under")], disc_T)):
        A = np.column_stack([atom_mask(frame, s, thr) for s in specs]).astype(np.float64)
        # drop degenerate atoms (e.g. predictor totals that equal the closing line -> 'edge' always 0)
        cov = A[dmask].mean(0)
        sd = np.array([np.nanstd(pd.to_numeric(frame.loc[dmask, s[1]], errors="coerce")) if s[1] in frame else 0 for s in specs])
        keep = (cov > 0.002) & (cov < 0.99) & (sd > 1e-6)
        specs = [s for s, k in zip(specs, keep) if k]
        A = A[:, keep]
        names = [s[0] for s in specs]
        k = len(specs)
        for mk, dr in markets:
            x = _pnl_vec(frame, mk, dr)
            v = np.isfinite(x) & dmask
            Av = A[v]
            xv = x[v]
            N = Av.T @ Av
            S1 = Av.T @ (Av * xv[:, None])
            S2 = Av.T @ (Av * (xv ** 2)[:, None])
            for i, j in itertools.combinations_with_replacement(range(k), 2):
                n = N[i, j]
                if n < 2 or (i != j and (n == N[i, i] or n == N[j, j])):   # empty, or pair == one of its singles
                    continue
                mean = S1[i, j] / n
                var = (S2[i, j] - n * mean ** 2) / (n - 1)
                z = mean / math.sqrt(var / n) if var > 0 else np.nan
                rows.append(dict(family="B_broad", sport=sport, market=mk, direction=dr,
                                 rule=names[i] if i == j else f"{names[i]} & {names[j]}",
                                 atoms=json.dumps([specs[i]] if i == j else [specs[i], specs[j]]),
                                 frame="side" if frame is L else "tot", disc_n=int(n), disc_roi=mean, disc_z=z))
    return pd.DataFrame(rows)


def targeted_scan(sport, L, T, thr):
    rows = []
    for name, fr, atoms, mk, dr in targeted_rules(sport):
        frame = L if fr == "side" else T
        m = np.ones(len(frame), bool)
        for a in atoms:
            m &= atom_mask(frame, a, thr)
        x, _ = evaluate_rule(frame, m, mk, dr, (frame.season <= DISC_END).values)
        n, mean, z, _ = zstat(x)
        rows.append(dict(family="A_targeted", sport=sport, market=mk, direction=dr, rule=name, atoms=json.dumps(atoms),
                         frame=fr, disc_n=n, disc_roi=mean, disc_z=z))
    return pd.DataFrame(rows)


def full_eval(frame, atoms, market, direction, thr, sport):
    m = np.ones(len(frame), bool)
    for a in atoms:
        m &= atom_mask(frame, a, thr)
    x = _pnl_vec(frame, market, direction)
    ok = m & np.isfinite(x)
    d = pd.DataFrame({"season": frame.season.values[ok], "date": frame.date.values[ok], "pnl": x[ok]})
    if market == "tot":
        d["juice_assumed"] = frame.juice_assumed.values[ok]
    else:
        d["juice_assumed"] = 1 if market == "ats" else 0
    return d


def perf_block(d):
    n, roi, z, p = zstat(d.pnl.values)
    cum = d.sort_values("date").pnl.cumsum()
    mdd = float((cum.cummax() - cum).max()) if len(cum) else np.nan
    return dict(n=n, roi=roi, z=z, p=p, units=float(d.pnl.sum()), max_dd=mdd)


# ------------------------------------------------------------------ research driver
def research():
    OUT.mkdir(parents=True, exist_ok=True)
    THR, frames, all_rows = {}, {}, []
    for sp in SPORTS:
        g = load_games(sp)
        L, T = long_frame(sp, g), total_frame(sp, g)
        if sp == "nba":
            L["t_tzabs"] = L.t_tz.abs()
        thr = fit_thresholds(sp, L, T)
        for name, fr, atoms, mk, dr in targeted_rules(sp):           # quantile atoms used only in targeted rules
            for a in atoms:
                if a[2] in ("qhi", "qlo"):
                    frame = L if fr == "side" else T
                    thr.setdefault(f"{a[1]}|{a[3]}", float(frame.loc[frame.season <= DISC_END, a[1]].quantile(a[3])))
        THR[sp] = thr
        frames[sp] = (L, T)
        all_rows += [targeted_scan(sp, L, T, thr), broad_scan(sp, L, T, thr)]
        print(sp, "games", len(g), file=sys.stderr)
    THR_FILE.write_text(json.dumps(THR, indent=1))
    C = pd.concat(all_rows, ignore_index=True)
    C["disc_p"] = norm.sf(C.disc_z)
    C["eligible"] = C.disc_n >= C.sport.map(MIN_N)
    C["disc_q"] = np.nan
    for fam in C.family.unique():
        m = (C.family == fam) & C.eligible
        C.loc[m, "disc_q"] = bh(C.loc[m, "disc_p"].values)
    C["screen_pass"] = C.eligible & (C.disc_roi > 0) & (C.disc_q < 0.10)
    # discovery-season stability (reported, also required: >= 60% of discovery seasons positive)
    C["disc_pos_seasons"] = ""
    C["lock_role"] = ""
    pass_idx = list(C.index[C.screen_pass])
    for i in pass_idx:
        r = C.loc[i]
        L, T = frames[r.sport]
        d = full_eval(L if r.frame == "side" else T, [tuple(a) for a in json.loads(r.atoms)], r.market, r.direction, THR[r.sport], r.sport)
        dd = d[d.season <= DISC_END].groupby("season").pnl.sum()
        C.loc[i, "disc_pos_seasons"] = f"{(dd > 0).sum()}/{len(dd)}"
        if (dd > 0).mean() >= 0.6:
            C.loc[i, "lock_role"] = "FINAL"
        else:
            C.loc[i, "lock_role"] = "dropped: discovery seasons <60% positive"
    # informational near-misses: best 5 per family by discovery z among eligible non-passing rules (cannot be TRUE)
    for fam in C.family.unique():
        m = (C.family == fam) & C.eligible & ~C.screen_pass & (C.disc_roi > 0)
        top = C[m].sort_values("disc_z", ascending=False).head(5).index
        C.loc[top, "lock_role"] = "INFO (failed screen; cannot be TRUE)"
    # ---- lockbox, once
    lock_cols = ["lock_n", "lock_roi", "lock_z", "lock_p", "lock_units", "lock_max_dd", "lock_juice_assumed",
                 "full_pos_seasons", "full_pct_pos", "lock_seasons"]
    for c in lock_cols:
        C[c] = np.nan if c not in ("full_pos_seasons", "lock_seasons") else ""
    for i in C.index[C.lock_role != ""]:
        if str(C.loc[i, "lock_role"]).startswith("dropped"):
            continue
        r = C.loc[i]
        L, T = frames[r.sport]
        d = full_eval(L if r.frame == "side" else T, [tuple(a) for a in json.loads(r.atoms)], r.market, r.direction, THR[r.sport], r.sport)
        lk = d[d.season >= LOCK_START]
        pb = perf_block(lk)
        by = d.groupby("season").pnl.sum()
        lby = lk.groupby("season").pnl.agg(["size", "sum"])
        C.loc[i, ["lock_n", "lock_roi", "lock_z", "lock_p", "lock_units", "lock_max_dd"]] = [pb["n"], pb["roi"], pb["z"], pb["p"], pb["units"], pb["max_dd"]]
        C.loc[i, "lock_juice_assumed"] = float(lk.juice_assumed.mean()) if len(lk) else np.nan
        C.loc[i, "full_pos_seasons"] = f"{(by > 0).sum()}/{len(by)}"
        C.loc[i, "full_pct_pos"] = float((by > 0).mean()) if len(by) else np.nan
        C.loc[i, "lock_seasons"] = "; ".join(f"{s}:{int(a)}b {b:+.1f}u" for s, (a, b) in lby.iterrows())
    fin = C.lock_role == "FINAL"
    n_final = int(fin.sum())
    C["lock_q"] = np.nan
    C.loc[fin, "lock_q"] = bh(C.loc[fin, "lock_p"].values)
    C["verdict"] = ""
    for i in C.index[C.lock_role != ""]:
        r = C.loc[i]
        if r.lock_role != "FINAL":
            C.loc[i, "verdict"] = "FALSE"
            continue
        sig = (r.lock_roi > 0) and ((r.lock_p < 0.05 / max(n_final, 1)) or (r.lock_q < 0.10))
        ok = bool(sig and r.full_pct_pos >= 0.6 and r.lock_n >= MIN_N_LOCK_INFO)
        v = "TRUE" if ok else "FALSE"
        if ok and r.sport == "mlb":
            v = "TRUE (provisional: MLB 2021+ price integrity)"
        C.loc[i, "verdict"] = v
    C.to_csv(OUT / "situational_candidates.csv", index=False)
    rules = C[C.verdict.str.startswith("TRUE")][["sport", "rule", "atoms", "frame", "market", "direction", "lock_roi"]]
    RULES_FILE.write_text(rules.to_json(orient="records", indent=1))
    write_md(C, frames, THR)
    return C


def _fmt(x, f="{:+.1%}"):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def write_md(C, frames, THR):
    n_tests = len(C)
    elig = C.eligible.sum()
    fam_counts = C.groupby("family").size().to_dict()
    sp_counts = C.groupby(["sport", "family"]).size().unstack(fill_value=0)
    fin = C[C.lock_role == "FINAL"]
    info = C[C.lock_role.str.startswith("INFO")]
    drop = C[C.lock_role.str.startswith("dropped")]
    lines = ["# Situational edge hunt (hunter: situational)", "",
             "Code: `engine/edge_hunt/situational.py` · all tests: `results/edge_hunt/situational_candidates.csv` · "
             "discovery thresholds: `results/edge_hunt/situational_thresholds.json`.", "",
             "## Design",
             f"* Data: `results/scripts/<sport>_games.parquet` (Kalman script margin/total/win prob, rest, b2b, travel, timezone, missing minutes, "
             "QB name/QB delta, SP & park effects, goalie effect & backup flag, pens, EPA, weather, conference) joined to "
             "`results/predict/<sport>_preds.parquet` (walk-forward model; available 2018+). Seasons 2016-2026.",
             "* Discovery = seasons <= 2022, lockbox = 2023+. Continuous thresholds = discovery-only quantiles.",
             "* Rules = 1 or 2 atoms (conjunction) x market (side ML, side ATS, game total) x direction. Side rules are written from the "
             "perspective of the team bet on; the atom set is mirror-symmetric so every fade is the 'back' of the mirrored atoms "
             "(family A lists fades explicitly).",
             "* Family A (targeted, pre-registered): script-vs-close disagreement follow/fade at fixed thresholds per sport and market; "
             "NHL backup-goalie spots; NFL QB-change spots; NBA heavy-missing-minutes spots; NBA/NHL rest-travel-timezone stacks; MLB new SP / bullpen; NCAA rest-in-conference.",
             "* Family B (broad): every single atom and every pair of atoms per sport.",
             f"* Screening: n >= {MIN_N} (discovery), ROI > 0, Benjamini-Hochberg q < 0.10 within family on one-sided discovery p, "
             "and >= 60% of discovery seasons positive. Survivors get the lockbox exactly once.",
             "* Prices: ML = real closing prices (quotes with |odds|>5000 or overround outside 1.00-1.15 dropped). "
             "**Spreads and totals for NBA/NFL/NCAAF/NCAAM carry no juice in the data -> -110 ASSUMED.** NHL totals: real over/under "
             "prices 2024+, -110 assumed before. MLB totals: real prices 2022+, -110 assumed before. MLB run line and NHL puck line "
             "not tested (no juice in data). **MLB 2021+ closing prices failed an integrity check (results/KEYS.md) — any MLB "
             "lockbox number is provisional.** NCAAM has no lines in 2024-25, so its lockbox is 2023 + 2026.",
             "", "## How many rules were screened", "",
             f"**n_tests = {n_tests}** rules logged ({', '.join(f'{k}: {v}' for k, v in fam_counts.items())}); "
             f"{elig} met the minimum-n requirement and entered BH.", "",
             sp_counts.to_markdown(), "",
             f"Screen survivors (BH q<0.10, ROI>0): {int(C.screen_pass.sum())}; dropped for discovery-season instability: {len(drop)}; "
             f"final lockbox candidates: {len(fin)}.", ""]
    hdr = "| family | sport | market | rule (bet) | disc n | disc ROI | disc z | disc q | lock n | lock ROI | lock z | lock p | lock BH q | units | max DD | lock -110 assumed | seasons + (full) | verdict |"
    sep = "|" + "---|" * 18
    def row(r):
        bet = r.direction if r.market == "tot" else f"{r.direction} {r.market.upper()}"
        return (f"| {r.family} | {r.sport} | {bet} | {r.rule} | {int(r.disc_n)} | {_fmt(r.disc_roi)} | {r.disc_z:.2f} | {_fmt(r.disc_q, '{:.3f}')} | "
                f"{'' if pd.isna(r.lock_n) else int(r.lock_n)} | {_fmt(r.lock_roi)} | {_fmt(r.lock_z, '{:.2f}')} | {_fmt(r.lock_p, '{:.4f}')} | "
                f"{_fmt(r.lock_q, '{:.3f}')} | {_fmt(r.lock_units, '{:+.1f}')} | {_fmt(r.lock_max_dd, '{:.1f}')} | "
                f"{_fmt(r.lock_juice_assumed, '{:.0%}')} | {r.full_pos_seasons} | {r.verdict} |")
    lines += ["## Verdicts — final candidates (survived discovery screen)", ""]
    if len(fin):
        lines += [hdr, sep] + [row(r) for _, r in fin.sort_values("lock_z", ascending=False).iterrows()]
        lines += ["", "Lockbox by season:", ""] + [f"* {r.sport} {r.rule} [{r.direction} {r.market}]: {r.lock_seasons}" for _, r in fin.iterrows()]
    else:
        lines += ["None: no rule survived the discovery screen."]
    if len(drop):
        lines += ["", "Passed BH but dropped (fewer than 60% of discovery seasons positive; lockbox not opened):", ""]
        lines += [f"* {r.sport} {r.rule} [{r.direction} {r.market}] disc n={int(r.disc_n)} ROI {_fmt(r.disc_roi)} z={r.disc_z:.2f} seasons {r.disc_pos_seasons}" for _, r in drop.iterrows()]
    lines += ["", "## Informational near-misses (top-5 discovery z per family that FAILED the screen; lockbox shown for context only — cannot be TRUE)", "",
              hdr, sep] + [row(r) for _, r in info.sort_values(["family", "disc_z"], ascending=[True, False]).iterrows()]
    # targeted family summary (all rules)
    A = C[C.family == "A_targeted"].copy()
    lines += ["", "## Family A (targeted) — discovery results for every pre-registered rule", "",
              "| sport | market | bet | rule | disc n | disc ROI | disc z | BH q |", "|---|---|---|---|---|---|---|---|"]
    for _, r in A.sort_values(["sport", "disc_z"], ascending=[True, False]).iterrows():
        lines.append(f"| {r.sport} | {r.market} | {r.direction} | {r.rule} | {int(r.disc_n) if pd.notna(r.disc_n) else 0} | "
                     f"{_fmt(r.disc_roi)} | {_fmt(r.disc_z, '{:.2f}')} | {_fmt(r.disc_q, '{:.3f}')} |")
    B = C[(C.family == "B_broad") & C.eligible]
    lines += ["", "## Family B (broad) — top 15 by discovery z", "",
              "| sport | market | bet | rule | disc n | disc ROI | disc z | BH q |", "|---|---|---|---|---|---|---|---|"]
    for _, r in B.sort_values("disc_z", ascending=False).head(15).iterrows():
        lines.append(f"| {r.sport} | {r.market} | {r.direction} | {r.rule} | {int(r.disc_n)} | {_fmt(r.disc_roi)} | {r.disc_z:.2f} | {_fmt(r.disc_q, '{:.3f}')} |")
    # p-value calibration check: are there more small p than chance?
    lines += ["", "## Multiple-testing sanity check", "",
              "Share of eligible tests with one-sided discovery p below each cut (pure noise would give ~ the cut itself; "
              "side rules in family B are correlated, so this is only indicative):", ""]
    for fam in ("A_targeted", "B_broad"):
        e = C[(C.family == fam) & C.eligible]
        lines.append(f"* {fam}: n={len(e)}; p<0.05: {np.mean(e.disc_p < 0.05):.1%}; p<0.01: {np.mean(e.disc_p < 0.01):.1%}; "
                     f"p<0.001: {np.mean(e.disc_p < 0.001):.2%}; min p = {e.disc_p.min():.2e}")
    lines += ["", "## Post-hoc look (disclosed; lockbox burned for this rule)", "", POSTHOC_NOTE]
    lines += ["", "## Mechanism notes", "", MECH_PLACEHOLDER, "", "## Live use", "",
              "`python -m engine.edge_hunt.situational --scan [--sport nba]` pulls today's slate (`nightly.live.today_slate`, ESPN + odds board), "
              "rebuilds the schedule features (rest, b2b) from the game history, de-vigs the ML, and checks the TRUE rules in "
              "`results/edge_hunt/situational_rules.json` (discovery thresholds from `situational_thresholds.json`). Atoms that need a "
              "script-engine column (script edge, goalie/SP effect, missing minutes, travel) are filled from today's row of "
              "`results/scripts/<sport>_games.parquet` when the nightly script run has already appended it; otherwise the game is reported "
              "as `needs_script_refresh` instead of being silently skipped."]
    (OUT / "situational.md").write_text("\n".join(lines) + "\n")


POSTHOC_NOTE = (
    "MLB 'follow script ML edge >= 0.15' shows +61.5% on 77 discovery bets (z 3.25) but is below the pre-set MLB minimum n "
    "(300), so it never entered screening. After the run I inspected it on all seasons (2016-25: 136 bets, avg price +275, "
    "max +1500, market p as low as 0.06 where the script says ~0.5): it is driven by stale/broken MLB quotes (the known "
    "integrity problem), not a situation. Because I looked at its 2023+ seasons, its lockbox is burned and it is NOT a candidate.")
MECH_PLACEHOLDER = (
    "No rule earned a mechanism claim. The likely reason no situational edge survives: every feature tested here is public "
    "before the close (rest/b2b/travel from the schedule, confirmed goalies/SPs/QBs, injury-driven missing minutes, "
    "weather), and the closing line already prices it — the script reports show the learned market weight sits at "
    "0.9-1.0 and corr(script - close, actual - close) is ~0 (NBA 0.010 margin / -0.019 total). Script-vs-close "
    "disagreement therefore measures script error, not market error, in every sport (follow AND fade both lose roughly the vig). "
    "Only 1.2-1.8% of eligible tests reached one-sided p<0.05 in discovery (noise alone would give ~5%; the vig pushes the "
    "whole distribution below zero), so the search found less signal than chance, not more. The best NFL total rule "
    "(wind>=12 unders) is the existing 'Wind Wall Unders' key re-discovered, not a new edge.")


# ------------------------------------------------------------------ live scan
def _live_rest(sport, slate):
    """Days since each team's last game, from history + today's slate (nightly.live.history_with_today)."""
    from nightly.live import history_with_today
    H = history_with_today(sport, slate)
    H["date"] = pd.to_datetime(H.date)
    rows = pd.concat([H[["date", "home"]].rename(columns={"home": "team"}), H[["date", "away"]].rename(columns={"away": "team"})])
    rows = rows.drop_duplicates().sort_values(["team", "date"])
    rows["rest"] = rows.groupby("team").date.diff().dt.days
    return rows.set_index(["team", "date"]).rest


def scan(sport: str | None = None, day=None, rules: list[dict] | None = None, include_info: bool = False) -> pd.DataFrame:
    """Today's instances of the TRUE situational rules (or `rules`, e.g. near-misses for paper tracking with
    include_info=True). Returns one row per qualifying bet, or a needs_script_refresh row."""
    from nightly.live import today_slate
    if rules is None:
        rules = json.loads(RULES_FILE.read_text()) if RULES_FILE.exists() else []
        if include_info and (OUT / "situational_candidates.csv").exists():
            C = pd.read_csv(OUT / "situational_candidates.csv")
            rules += C[C.lock_role.fillna("").str.startswith("INFO")][["sport", "rule", "atoms", "frame", "market", "direction", "lock_roi"]].to_dict("records")
    THR = json.loads(THR_FILE.read_text()) if THR_FILE.exists() else {}
    out = []
    sports = [sport] if sport else sorted({r["sport"] for r in rules})
    for sp in sports:
        R = [r for r in rules if r["sport"] == sp]
        if not R:
            continue
        s = today_slate(sp, day)
        if s is None or s.empty:
            continue
        s = s.copy()
        s["date"] = pd.to_datetime(s.get("date", pd.Timestamp.today().normalize())).dt.normalize()
        # board prices, falling back to ESPN's consensus line; slate 'total' is the market line
        for c, e in (("home_ml", "espn_home_ml"), ("away_ml", "espn_away_ml"), ("spread_home", "espn_spread_home"), ("total", "espn_total")):
            s[c] = pd.to_numeric(s.get(c), errors="coerce")
            if e in s:
                s[c] = s[c].fillna(pd.to_numeric(s[e], errors="coerce"))
        s = s.rename(columns={"total": "total_line"})
        # schedule features
        try:
            rest = _live_rest(sp, s)
            s["rest_home"] = [rest.get((t, d), np.nan) for t, d in zip(s.home, s.date)]
            s["rest_away"] = [rest.get((t, d), np.nan) for t, d in zip(s.away, s.date)]
            s["b2b_home"], s["b2b_away"] = (s.rest_home == 1).astype(float), (s.rest_away == 1).astype(float)
        except Exception:
            pass
        # script columns: today's rows of the script parquet if the nightly run appended them
        G = pd.read_parquet(SCR / f"{sp}_games.parquet")
        G["date"] = pd.to_datetime(G.date).dt.normalize()
        G = _merge_preds(sp, G.assign(game_id=G.game_id.astype(str)))
        keyed = G.set_index(["date", "home", "away"])
        have = [(d, h, a) in keyed.index for d, h, a in zip(s.date, s.home, s.away)]
        s["home_score"], s["away_score"] = 0.0, 0.0
        for c in keyed.columns:
            if c not in s or c in ("over_odds", "under_odds") and s[c].isna().all():
                s[c] = [keyed.loc[(d, h, a), c] if ok else np.nan for d, h, a, ok in zip(s.date, s.home, s.away, have)]
        s["game_id"] = s.get("game_id", pd.Series(range(len(s)))).astype(str)
        s["season"] = s.season.fillna(LOCK_START) if "season" in s else LOCK_START
        g = s.copy()
        if sp in ("nba", "ncaam"):
            g["p_script_home"] = norm.cdf(g["margin"] / g["sd_margin"]) if "sd_margin" in g else np.nan
        elif sp == "ncaaf":
            g["p_script_home"] = g.get("p_home_win", np.nan)
        else:
            g["p_script_home"] = g.get("p_home_script", np.nan)
        g["script_margin"], g["script_total"] = g.get("margin", np.nan), g.get("total", np.nan)
        g["ml_ok"] = ml_ok(g.home_ml, g.away_ml)
        ih, ia = imp(g.home_ml.fillna(-110)), imp(g.away_ml.fillna(-110))
        g["p_mkt_home"] = np.where(g.ml_ok, ih / (ih + ia), np.nan)
        if sp == "nfl":
            g = _nfl_qb_change(g) if g.get("home_qb_name", pd.Series(dtype=object)).notna().any() else g.assign(qbchg_home=np.nan, qbchg_away=np.nan)
        # live schedule features win over (possibly missing) script-parquet values
        if "b2b_home" in s:
            g["home_b2b"], g["away_b2b"] = s.b2b_home.values, s.b2b_away.values
            g["b2b_home"], g["b2b_away"] = s.b2b_home.values, s.b2b_away.values
            g["home_rest"], g["away_rest"] = s.rest_home.values, s.rest_away.values
            g["rest_home"], g["rest_away"] = s.rest_home.values, s.rest_away.values
        for c in [p[1] for p in PAIRS.get(sp, [])] + [p[2] for p in PAIRS.get(sp, [])]:
            if c not in g:
                g[c] = np.nan
        L, T = long_frame(sp, g), total_frame(sp, g)
        if sp == "nba":
            L["t_tzabs"] = L.t_tz.abs()
        thr = THR.get(sp, {})
        for r in R:
            atoms = [tuple(a) for a in json.loads(r["atoms"])]
            frame = L if r["frame"] == "side" else T
            m = np.ones(len(frame), bool)
            missing = []
            for a in atoms:
                col = a[1]
                if col not in frame or frame[col].isna().all():
                    missing.append(col)
                m &= atom_mask(frame, a, thr)
            if r["direction"] == "fade":
                N = len(m) // 2
                m = np.concatenate([m[N:], m[:N]])
            for i in np.where(m)[0]:
                row = frame.iloc[i]
                if r["frame"] == "side":
                    pick = f"{row.team} {'ML' if r['market'] == 'ml' else f'{row.spread_t:+g}'}"
                    gi = i % (len(frame) // 2)
                    juice = s["spread_home_odds" if row.is_home else "spread_away_odds"].iloc[gi] if "spread_home_odds" in s else np.nan
                    price = row.ml_t if r["market"] == "ml" else (juice if pd.notna(juice) else -110)
                    ev = f"{row.opp if row.is_home == 0 else row.team} vs {row.team if row.is_home == 0 else row.opp}"
                else:
                    pick = f"{r['direction'].upper()} {row.total_line:g}"
                    price = row.get("over_odds" if r["direction"] == "over" else "under_odds", -110)
                    ev = f"{row.away} @ {row.home}"
                out.append(dict(sport=sp, rule=r["rule"], event=ev, pick=pick, price=price, lock_roi=r["lock_roi"]))
            if missing:
                for i in range(len(s)):
                    out.append(dict(sport=sp, rule=r["rule"], event=f"{s.away.iloc[i]} @ {s.home.iloc[i]}",
                                    pick="needs_script_refresh: " + ",".join(sorted(set(missing))), price=np.nan, lock_roi=r["lock_roi"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--sport")
    ap.add_argument("--info", action="store_true", help="also scan the informational near-miss rules (paper only)")
    a = ap.parse_args()
    if a.scan:
        r = scan(a.sport, include_info=a.info)
        print(r.to_string() if len(r) else "no qualifying situational bets today (no TRUE rules / no matches)")
    else:
        C = research()
        print(C[C.lock_role != ""][["family", "sport", "market", "direction", "rule", "disc_n", "disc_roi", "disc_z", "disc_q",
                                     "lock_role", "lock_n", "lock_roi", "lock_z", "lock_p", "lock_q", "full_pos_seasons", "verdict"]].to_string())
        print("n_tests", len(C))
