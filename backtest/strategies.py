"""
Strategy generator for head-to-head team sports.

A strategy = (market, team condition, role filter, direction, phase).
  market     : 'ats' (spread / run line / puck line), 'ml' (moneyline), 'tot' (over/under)
  condition  : a pre-game fact about team T (streak, rest, recent ATS form, Elo vs market...)
  role       : what T is in this game (fav/dog/home/away/price bucket)
  direction  : 'on' = bet T, 'vs' = bet T's opponent
  phase      : regular-season slice or playoffs

Each strategy yields a Series of unit P&L indexed by game id (max one bet per game).
Games where BOTH teams satisfy the condition are skipped (no clean side).
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .core import settle

TEAM_COLS = None  # populated lazily


# --------------------------------------------------------------- per-game P&L
def market_pnl(g: pd.DataFrame) -> dict:
    """Unit P&L for each side of each market, NaN where no line."""
    out = {}
    ok = g["spread_home"].notna()
    out["ats_h"] = pd.Series(np.where(ok, settle(np.sign(g["ats_home"]), g["spread_home_odds"]), np.nan), g.index)
    out["ats_a"] = pd.Series(np.where(ok, settle(np.sign(-g["ats_home"]), g["spread_away_odds"]), np.nan), g.index)
    ok = g["home_ml"].notna() & g["away_ml"].notna()
    out["ml_h"] = pd.Series(np.where(ok, settle(np.sign(g["margin"]), g["home_ml"].fillna(-110)), np.nan), g.index)
    out["ml_a"] = pd.Series(np.where(ok, settle(np.sign(-g["margin"]), g["away_ml"].fillna(-110)), np.nan), g.index)
    ok = g["total"].notna()
    out["tot_o"] = pd.Series(np.where(ok, settle(np.sign(g["ou"]), g["over_odds"]), np.nan), g.index)
    out["tot_u"] = pd.Series(np.where(ok, settle(np.sign(-g["ou"]), g["under_odds"]), np.nan), g.index)
    return out


# --------------------------------------------------------------- team views
def team_view(g: pd.DataFrame, side: str) -> pd.DataFrame:
    """Columns describing team T (side 'h' or 'a') and its opponent O, unprefixed."""
    me, op = ("h_", "a_") if side == "h" else ("a_", "h_")
    v = pd.DataFrame(index=g.index)
    for c in [c for c in g.columns if c.startswith("h_")]:
        base = c[2:]
        v[base] = g[me + base]
        v["o_" + base] = g[op + base]
    sgn = 1 if side == "h" else -1
    v["spread"] = sgn * g["spread_home"]
    v["ml"] = g["home_ml"] if side == "h" else g["away_ml"]
    v["p_mkt"] = g["p_home_mkt"] if side == "h" else g["p_away_mkt"]
    v["p_elo"] = g["p_home_elo"] if side == "h" else 1 - g["p_home_elo"]
    v["home"] = (side == "h") & (g["neutral"] == 0)
    v["fav"] = np.where(v["p_mkt"].notna(), v["p_mkt"] > 0.5, v["spread"] < 0)
    # line movement open -> close from T's perspective (>0 = market moved TOWARD T)
    if "spread_open_home" in g:
        v["spread_move"] = sgn * (pd.to_numeric(g["spread_open_home"], errors="coerce") - g["spread_home"])
    else:
        v["spread_move"] = np.nan
    # moneyline movement: change in no-vig win prob from open to close (>0 = toward T)
    if "home_ml_open" in g:
        from .core import no_vig_probs
        po_h, po_a, _ = no_vig_probs(pd.to_numeric(g["home_ml_open"], errors="coerce"),
                                     pd.to_numeric(g["away_ml_open"], errors="coerce"))
        v["ml_move"] = v["p_mkt"] - (po_h if side == "h" else po_a)
    else:
        v["ml_move"] = np.nan
    for c in ("st_starts", "st_wvm10", "st_ra5", "st_backup", "power"):
        if c not in v:
            v[c], v["o_" + c] = np.nan, np.nan
    return v


def team_conditions(S: float, sport: str) -> dict:
    """name -> f(view)->bool. S = points scale (sd of ATS margin) for thresholds."""
    c = {
        "any": lambda v: pd.Series(True, v.index),
        # --- previous game / overreaction
        "prev_blowout_loss": lambda v: v["prev_margin"] <= -1.2 * S,
        "prev_blowout_win": lambda v: v["prev_margin"] >= 1.2 * S,
        "prev_big_ats_fail": lambda v: v["prev_ats"] <= -1.0 * S,
        "prev_big_ats_cover": lambda v: v["prev_ats"] >= 1.0 * S,
        "prev_upset_loss_as_fav": lambda v: (v["prev_margin"] < 0) & (v["prev_spread"] <= -0.4 * S),
        "prev_upset_win_as_dog": lambda v: (v["prev_margin"] > 0) & (v["prev_spread"] >= 0.4 * S),
        # --- streaks
        "su_won3": lambda v: v["su_streak"] >= 3,
        "su_won5": lambda v: v["su_streak"] >= 5,
        "su_lost3": lambda v: v["su_streak"] <= -3,
        "su_lost5": lambda v: v["su_streak"] <= -5,
        "ats_won3": lambda v: v["ats_streak"] >= 3,
        "ats_lost3": lambda v: v["ats_streak"] <= -3,
        "ats_won5": lambda v: v["ats_streak"] >= 5,
        "ats_lost5": lambda v: v["ats_streak"] <= -5,
        # --- rolling form vs the market
        "hot_ats_r5": lambda v: v["ats_r5"] >= 0.5 * S,
        "cold_ats_r5": lambda v: v["ats_r5"] <= -0.5 * S,
        "hot_ats_r10": lambda v: v["ats_r10"] >= 0.35 * S,
        "cold_ats_r10": lambda v: v["ats_r10"] <= -0.35 * S,
        "ats_pct_hi": lambda v: (v["ats_wpct"] >= 0.65) & (v["gp"] >= 8),
        "ats_pct_lo": lambda v: (v["ats_wpct"] <= 0.35) & (v["gp"] >= 8),
        "mkt_underrates_r10": lambda v: v["wvm_r10"] >= 0.15,
        "mkt_overrates_r10": lambda v: v["wvm_r10"] <= -0.15,
        # --- model vs market
        "elo_edge_5": lambda v: (v["p_elo"] - v["p_mkt"]) >= 0.05,
        "elo_edge_10": lambda v: (v["p_elo"] - v["p_mkt"]) >= 0.10,
        "elo_edge_15": lambda v: (v["p_elo"] - v["p_mkt"]) >= 0.15,
        # --- schedule
        "rest_adv2": lambda v: (v["rest"] - v["o_rest"]) >= 2,
        "rest_disadv2": lambda v: (v["o_rest"] - v["rest"]) >= 2,
        "long_road_trip": lambda v: (~v["home"]) & (v["venue_run"] >= 3),
        "long_homestand": lambda v: v["home"] & (v["venue_run"] >= 4),
        "season_opener": lambda v: v["first_of_season"] == 1,
        "early_season": lambda v: (v["gp"] < 4) & (v["first_of_season"] == 0),
        # --- line movement (actionable: bet at the CLOSE knowing the open)
        "steam_toward": lambda v: v["spread_move"] >= 0.1 * S,
        "steam_toward_big": lambda v: v["spread_move"] >= 0.2 * S,
        "steam_against": lambda v: v["spread_move"] <= -0.1 * S,
        "steam_against_big": lambda v: v["spread_move"] <= -0.2 * S,
    }
    c.update({
        "ml_steam_toward": lambda v: v["ml_move"] >= 0.03,
        "ml_steam_against": lambda v: v["ml_move"] <= -0.03,
    })
    if sport in ("mlb", "nhl"):
        c.update({
            "starter_underrated": lambda v: v["st_wvm10"] >= 0.12,
            "starter_overrated": lambda v: v["st_wvm10"] <= -0.12,
            "starter_new": lambda v: v["st_starts"] < 3,
            "opp_starter_new": lambda v: (v["o_st_starts"] < 3) & (v["st_starts"] >= 10),
            "starter_shelled": lambda v: v["st_ra5"] >= (6 if sport == "mlb" else 4),
            "starter_dealing": lambda v: v["st_ra5"] <= (2.5 if sport == "mlb" else 1.8),
        })
    if sport == "nhl":
        c.update({
            "backup_goalie": lambda v: v["st_backup"] == 1,
            "opp_backup_goalie": lambda v: (v["o_st_backup"] == 1) & (v["st_backup"] == 0),
        })
    if sport in ("ncaaf", "ncaam"):
        c.update({
            "power_vs_mid": lambda v: (v["power"] == 1) & (v["o_power"] == 0),
            "mid_vs_power": lambda v: (v["power"] == 0) & (v["o_power"] == 1),
        })
    if sport in ("nba", "nhl", "ncaam"):
        c.update({
            "b2b": lambda v: v["rest"] == 1,
            "b2b_vs_rested": lambda v: (v["rest"] == 1) & (v["o_rest"] >= 2),
            "rested_vs_b2b": lambda v: (v["rest"] >= 2) & (v["o_rest"] == 1),
            "3in4": lambda v: v["g_last4"] >= 2,
            "4in7": lambda v: v["g_last7"] >= 3,
        })
    if sport in ("nfl", "ncaaf"):
        c.update({
            "off_bye": lambda v: v["rest"] >= 13,
            "off_bye_vs_nonbye": lambda v: (v["rest"] >= 13) & (v["o_rest"] < 10),
            "short_week": lambda v: v["rest"] <= 5,
            "off_mnf_vs_rested": lambda v: (v["rest"] <= 6) & (v["o_rest"] >= 7),
        })
    if sport == "mlb":
        c.update({
            "first_game_after_off": lambda v: v["rest"] >= 2,
            "4in7": lambda v: v["g_last7"] >= 6,
        })
    return c


def roles(S: float) -> dict:
    """What team T is in the game being bet."""
    return {
        "any": lambda v: pd.Series(True, v.index),
        "fav": lambda v: v["fav"],
        "dog": lambda v: ~v["fav"],
        "home": lambda v: v["home"],
        "away": lambda v: ~v["home"],
        "home_dog": lambda v: v["home"] & ~v["fav"],
        "road_fav": lambda v: ~v["home"] & v["fav"],
        "big_dog": lambda v: v["p_mkt"] <= 0.30,
        "mid_dog": lambda v: (v["p_mkt"] > 0.30) & (v["p_mkt"] < 0.45),
        "small_dog": lambda v: (v["p_mkt"] >= 0.45) & (v["p_mkt"] <= 0.5),
        "small_fav": lambda v: (v["p_mkt"] > 0.5) & (v["p_mkt"] < 0.60),
        "mid_fav": lambda v: (v["p_mkt"] >= 0.60) & (v["p_mkt"] < 0.75),
        "big_fav": lambda v: v["p_mkt"] >= 0.75,
    }


# late-season game-number thresholds (= 75% of the median season length incl. playoffs, 2008-2025)
LATE_GP = {"nba": 78, "nhl": 78, "mlb": 133, "ncaam": 29, "nfl": 14, "ncaaf": 9}

PHASES = {
    "all": lambda g: pd.Series(True, g.index),
    "reg": lambda g: g["playoff"] == 0,
    "playoff": lambda g: g["playoff"] == 1,
    "conf": lambda g: g["conference_game"].fillna(False).astype(bool) if "conference_game" in g
                      else pd.Series(False, g.index),
    "nonconf": lambda g: ~g["conference_game"].fillna(True).astype(bool) if "conference_game" in g
                         else pd.Series(False, g.index),
}


# --------------------------------------------------------------- generation
def side_strategies(g, pnl, S, sport, markets=("ats", "ml"), only=None):
    """only=(condition, role) restricts generation to one cell (fast path for ledgers)."""
    vh, va = team_view(g, "h"), team_view(g, "a")
    conds, rls = team_conditions(S, sport), roles(S)
    ch = {k: f(vh).fillna(False).astype(bool) for k, f in conds.items()}
    ca = {k: f(va).fillna(False).astype(bool) for k, f in conds.items()}
    rh = {k: f(vh).fillna(False).astype(bool) for k, f in rls.items()}
    ra = {k: f(va).fillna(False).astype(bool) for k, f in rls.items()}
    out = {}
    for cname in conds:
        if only and cname != only[0]:
            continue
        exclusive_h = ch[cname] & (~ca[cname] if cname != "any" else True)
        exclusive_a = ca[cname] & (~ch[cname] if cname != "any" else True)
        for rname in rls:
            if only and rname != only[1]:
                continue
            if cname == "any" and rname == "any":
                continue
            mh = exclusive_h & rh[rname]
            ma = exclusive_a & ra[rname]
            if cname == "any":       # pure role strategy: both sides may qualify; take
                ma = ma & ~mh        # home when both (only symmetric roles like fav/dog collide)
            for mkt in markets:
                for direction in ("on", "vs"):
                    if direction == "on":
                        s = pd.concat([pnl[f"{mkt}_h"][mh], pnl[f"{mkt}_a"][ma]])
                    else:
                        s = pd.concat([pnl[f"{mkt}_a"][mh], pnl[f"{mkt}_h"][ma]])
                    s = s.dropna().sort_index()
                    if len(s):
                        for ph, pf in PHASES.items():
                            keep = pf(g).loc[s.index]
                            ss = s[keep.values]
                            if len(ss):
                                out[(mkt, cname, rname, direction, ph)] = ss
    return out


def total_filters(g: pd.DataFrame, S: float, sport: str, total_q: dict) -> dict:
    """Game-level filters for over/under bets. total_q: train-set total quantiles."""
    q = total_q
    f = {
        "all": pd.Series(True, g.index),
        "total_lowest20": g["total"] <= q[0.2],
        "total_highest20": g["total"] >= q[0.8],
        "total_mid": (g["total"] > q[0.4]) & (g["total"] < q[0.6]),
        "both_over_r5": (g["h_ou_r5"] > 0.25 * S) & (g["a_ou_r5"] > 0.25 * S),
        "both_under_r5": (g["h_ou_r5"] < -0.25 * S) & (g["a_ou_r5"] < -0.25 * S),
        "both_over_streak3": (g["h_ou_streak"] >= 3) & (g["a_ou_streak"] >= 3),
        "both_under_streak3": (g["h_ou_streak"] <= -3) & (g["a_ou_streak"] <= -3),
        "one_over_streak4": (g["h_ou_streak"] >= 4) | (g["a_ou_streak"] >= 4),
        "one_under_streak4": (g["h_ou_streak"] <= -4) | (g["a_ou_streak"] <= -4),
        "big_spread": g["spread_home"].abs() >= 1.0 * S,
        "pickem": g["spread_home"].abs() <= 0.15 * S,
        "early_season": (g["h_gp"] < 4) & (g["a_gp"] < 4),
        # fixed, known-in-advance threshold (team's Nth game of the season incl. playoffs)
        "late_season": g["h_gp"] >= LATE_GP.get(sport, 0.75 * g.groupby("season")["h_gp"].transform("max")),
        "prev_both_over_big": (g["h_prev_ou"] >= 0.8 * S) & (g["a_prev_ou"] >= 0.8 * S),
        "prev_both_under_big": (g["h_prev_ou"] <= -0.8 * S) & (g["a_prev_ou"] <= -0.8 * S),
        "playoff": g["playoff"] == 1,
    }
    if "total_open" in g:
        mv = g["total"] - pd.to_numeric(g["total_open"], errors="coerce")
        f["total_up"] = mv >= 0.1 * S
        f["total_up_big"] = mv >= 0.2 * S
        f["total_down"] = mv <= -0.1 * S
        f["total_down_big"] = mv <= -0.2 * S
    if "conference_game" in g:
        f["conf_game"] = g["conference_game"].fillna(False).astype(bool)
    if sport in ("nba", "nhl", "ncaam"):
        f["both_b2b"] = (g["h_rest"] == 1) & (g["a_rest"] == 1)
        f["away_b2b"] = (g["a_rest"] == 1) & (g["h_rest"] >= 2)
        f["home_b2b"] = (g["h_rest"] == 1) & (g["a_rest"] >= 2)
        f["both_rested3"] = (g["h_rest"] >= 3) & (g["a_rest"] >= 3)
    if sport in ("nfl", "ncaaf"):
        if "wind" in g:
            w = pd.to_numeric(g["wind"], errors="coerce")
            f["wind15"] = w >= 15
            f["wind10"] = w >= 10
        if "temp" in g:
            t = pd.to_numeric(g["temp"], errors="coerce")
            f["cold32"] = t <= 32
        if "roof" in g:
            f["dome"] = g["roof"].isin(["dome", "closed"])
            f["outdoor"] = g["roof"].isin(["outdoors", "open"])
        if "div_game" in g:
            f["div_game"] = g["div_game"] == 1
        f["off_bye_either"] = (g["h_rest"] >= 13) | (g["a_rest"] >= 13)
        f["short_week_both"] = (g["h_rest"] <= 5) & (g["a_rest"] <= 5)
    return {k: v.fillna(False).astype(bool) for k, v in f.items()}


def total_strategies(g, pnl, S, sport, total_q, only=None):
    filt = total_filters(g, S, sport, total_q)
    names = list(filt)
    out = {}
    combos = [(a,) for a in names] + [
        (a, b) for a, b in itertools.combinations(names, 2) if a != "all" and b != "all"]
    if only:
        combos = [tuple(only.split("+"))]
    for combo in combos:
        m = filt[combo[0]].copy()
        for c in combo[1:]:
            m &= filt[c]
        if m.sum() < 30:
            continue
        for side in ("o", "u"):
            s = pnl[f"tot_{side}"][m].dropna()
            if len(s):
                out[("tot", "+".join(combo), "game", "over" if side == "o" else "under", "all")] = s
    return out


# --------------------------------------------------------------- live qualification
def side_masks(g, S, sport, cond, role, direction="on", phase="all"):
    """Which games qualify for a side strategy, and which side to bet.
    Returns (bet_home, bet_away) boolean Series. Uses only pre-game columns, so it
    works for unplayed games (no result needed)."""
    vh, va = team_view(g, "h"), team_view(g, "a")
    cf, rf = team_conditions(S, sport)[cond], roles(S)[role]
    ch, ca = cf(vh).fillna(False).astype(bool), cf(va).fillna(False).astype(bool)
    if cond != "any":
        ch, ca = ch & ~ca, ca & ~ch
    mh = ch & rf(vh).fillna(False).astype(bool)
    ma = ca & rf(va).fillna(False).astype(bool)
    if cond == "any":
        ma = ma & ~mh
    ph = PHASES[phase](g).fillna(False).astype(bool)
    mh, ma = mh & ph, ma & ph
    return (mh, ma) if direction == "on" else (ma, mh)


def total_mask(g, S, sport, total_q, combo):
    f = total_filters(g, S, sport, total_q)
    m = pd.Series(True, g.index)
    for c in combo.split("+"):
        m &= f[c]
    return m & g["total"].notna()
