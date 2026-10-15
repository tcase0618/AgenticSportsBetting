"""
Leak-free pre-game feature engineering for head-to-head team sports.

Input: a standardized game table with columns
    date, season, home, away, home_score, away_score,
    spread_home (home perspective, negative = home favored), total,
    home_ml, away_ml, [spread_home_odds, spread_away_odds, over_odds, under_odds,
    neutral, playoff, ...]

Every rolling feature is computed on a team-game "long" table, sorted by date,
and SHIFTED by one game so a row only ever sees that team's prior games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import no_vig_probs, DEFAULT_JUICE


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["spread_home_odds", "spread_away_odds", "over_odds", "under_odds"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(DEFAULT_JUICE)
    for c in ["home_ml", "away_ml", "spread_home", "total", "home_score", "away_score"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["neutral", "playoff"]:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # sanity: drop absurd moneylines (data-entry errors) rather than bet them
    for c in ["home_ml", "away_ml"]:
        bad = df[c].abs() < 100
        df.loc[bad, c] = np.nan
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values(["date", "home"]).reset_index(drop=True)
    df["gid"] = np.arange(len(df))

    df["margin"] = df["home_score"] - df["away_score"]
    df["pts"] = df["home_score"] + df["away_score"]
    df["ats_home"] = df["margin"] + df["spread_home"]          # >0 home covers
    df["ou"] = df["pts"] - df["total"]                          # >0 over
    ph, pa, vig = no_vig_probs(df["home_ml"], df["away_ml"])
    df["p_home_mkt"], df["p_away_mkt"], df["overround"] = ph, pa, vig
    return df


def long_table(g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one per team, team-perspective metrics."""
    cols = dict(gid="gid", date="date", season="season", playoff="playoff", neutral="neutral",
                total="total", ou="ou")
    h = g[list(cols)].copy()
    h["team"], h["opp"], h["is_home"] = g["home"], g["away"], 1
    h["pf"], h["pa"] = g["home_score"], g["away_score"]
    h["spread"], h["ml"], h["p_mkt"] = g["spread_home"], g["home_ml"], g["p_home_mkt"]
    a = g[list(cols)].copy()
    a["team"], a["opp"], a["is_home"] = g["away"], g["home"], 0
    a["pf"], a["pa"] = g["away_score"], g["home_score"]
    a["spread"], a["ml"], a["p_mkt"] = -g["spread_home"], g["away_ml"], g["p_away_mkt"]
    L = pd.concat([h, a], ignore_index=True)
    L["margin"] = L["pf"] - L["pa"]
    L["ats"] = L["margin"] + L["spread"]
    L["win"] = (L["margin"] > 0).astype(float)
    L["cover"] = np.sign(L["ats"])
    # how much better than market expectation (win prob) the team did
    L["win_vs_mkt"] = L["win"] - L["p_mkt"]
    return L.sort_values(["team", "date", "gid"]).reset_index(drop=True)


def _streak(s: pd.Series) -> pd.Series:
    """Signed running streak of +1/-1 values (0 breaks), computed per group."""
    out, cur = [], 0
    for v in s:
        if v > 0:
            cur = cur + 1 if cur > 0 else 1
        elif v < 0:
            cur = cur - 1 if cur < 0 else -1
        else:
            cur = 0
        out.append(cur)
    return pd.Series(out, index=s.index)


def team_features(L: pd.DataFrame) -> pd.DataFrame:
    """Pre-game rolling features. Each value only uses games strictly before."""
    L = L.copy()
    grp = L.groupby("team", sort=False)
    L["prev_date"] = grp["date"].shift(1)
    L["rest"] = (L["date"] - L["prev_date"]).dt.days
    L["prev_season"] = grp["season"].shift(1)
    L["first_of_season"] = (L["prev_season"] != L["season"]).astype(int)

    # streaks / previous game, reset each season
    gs = L.groupby(["team", "season"], sort=False)
    L["prev_margin"] = gs["margin"].shift(1)
    L["prev_ats"] = gs["ats"].shift(1)
    L["prev_home"] = gs["is_home"].shift(1)
    L["prev_ou"] = gs["ou"].shift(1)
    L["prev_spread"] = gs["spread"].shift(1)
    L["su_streak"] = gs["margin"].transform(lambda s: _streak(np.sign(s)).shift(1))
    L["ats_streak"] = gs["ats"].transform(lambda s: _streak(np.sign(s)).shift(1))
    L["ou_streak"] = gs["ou"].transform(lambda s: _streak(np.sign(s)).shift(1))
    L["gp"] = gs.cumcount()
    L["ats_wpct"] = gs["cover"].transform(lambda s: (s > 0).astype(float).expanding().mean().shift(1))
    L["su_wpct"] = gs["win"].transform(lambda s: s.expanding().mean().shift(1))
    for n in (3, 5, 10):
        L[f"ats_r{n}"] = gs["ats"].transform(lambda s: s.rolling(n, min_periods=n).mean().shift(1))
        L[f"ou_r{n}"] = gs["ou"].transform(lambda s: s.rolling(n, min_periods=n).mean().shift(1))
        L[f"mgn_r{n}"] = gs["margin"].transform(lambda s: s.rolling(n, min_periods=n).mean().shift(1))
        L[f"wvm_r{n}"] = gs["win_vs_mkt"].transform(lambda s: s.rolling(n, min_periods=n).mean().shift(1))
    # road trip / home stand length entering this game
    L["venue_run"] = gs["is_home"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount())
    # games in last 4 / 7 days (fatigue) – counts prior games only
    L["g_last4"] = _games_in_window(L, 4)
    L["g_last7"] = _games_in_window(L, 7)
    return L


def _games_in_window(L, days):
    out = np.zeros(len(L))
    for team, idx in L.groupby("team", sort=False).indices.items():
        d = L["date"].values[idx].astype("datetime64[D]").astype(np.int64)
        lo = np.searchsorted(d, d - days, side="left")
        out[idx] = np.arange(len(idx)) - lo
    return out


def elo(g: pd.DataFrame, k=20.0, hfa=55.0, revert=0.25, mov=True) -> pd.DataFrame:
    """Pre-game Elo for each game (margin-of-victory multiplier, season regression)."""
    r, last_season = {}, {}
    eh, ea = np.zeros(len(g)), np.zeros(len(g))
    for i, row in enumerate(g.itertuples(index=False)):
        for t in (row.home, row.away):
            if t not in r:
                r[t] = 1500.0
            elif last_season.get(t) != row.season:
                r[t] = 1500 + (1 - revert) * (r[t] - 1500)
            last_season[t] = row.season
        rh, ra = r[row.home], r[row.away]
        eh[i], ea[i] = rh, ra
        adv = 0 if row.neutral else hfa
        exp_h = 1 / (1 + 10 ** (-(rh + adv - ra) / 400))
        res = 1.0 if row.margin > 0 else (0.0 if row.margin < 0 else 0.5)
        mult = 1.0
        if mov:
            d = abs(row.margin)
            mult = np.log(d + 1) * 2.2 / ((rh + adv - ra if res == 1 else ra - rh - adv) * 0.001 + 2.2)
        delta = k * mult * (res - exp_h)
        r[row.home], r[row.away] = rh + delta, ra - delta
    g = g.copy()
    g["elo_home"], g["elo_away"] = eh, ea
    adv = np.where(g["neutral"] == 1, 0, hfa)
    g["p_home_elo"] = 1 / (1 + 10 ** (-(g["elo_home"] + adv - g["elo_away"]) / 400))
    return g


def build(df: pd.DataFrame, elo_params: dict | None = None) -> pd.DataFrame:
    """Full pipeline -> one row per game with home_* / away_* pre-game features."""
    g = standardize(df)
    g = elo(g, **(elo_params or {}))
    L = team_features(long_table(g))
    feat_cols = ["rest", "first_of_season", "prev_margin", "prev_ats", "prev_home", "prev_ou",
                 "prev_spread", "su_streak", "ats_streak", "ou_streak", "gp", "ats_wpct", "su_wpct",
                 "venue_run", "g_last4", "g_last7"] + \
                [f"{p}_r{n}" for p in ("ats", "ou", "mgn", "wvm") for n in (3, 5, 10)]
    H = L[L.is_home == 1].set_index("gid")[feat_cols].add_prefix("h_")
    A = L[L.is_home == 0].set_index("gid")[feat_cols].add_prefix("a_")
    g = g.set_index("gid").join(H).join(A).reset_index()
    g["rest_diff"] = g["h_rest"] - g["a_rest"]
    return g


def starter_features(df: pd.DataFrame, home_col: str, away_col: str, backup=False) -> pd.DataFrame:
    """Leak-free features for the listed starter (MLB pitcher / NHL goalie).
    Adds h_st_* / a_st_* columns, each computed from the starter's PRIOR starts only:
      st_starts : prior starts in the data
      st_wvm10  : mean(team won - market win prob) over his last 10 starts (market misprices him?)
      st_ra5    : mean runs/goals allowed by his team in his last 5 starts
      st_backup : (backup=True) starter made < 30% of his team's previous 10 starts
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    ph, pa, _ = no_vig_probs(pd.to_numeric(d["home_ml"], errors="coerce"), pd.to_numeric(d["away_ml"], errors="coerce"))
    rows = []
    for side, col, me, op, p in (("h", home_col, "home", "away", ph), ("a", away_col, "away", "home", pa)):
        t = pd.DataFrame({"i": d.index, "date": d["date"], "team": d[me], "st": d[col].astype(str).str.strip().str.lower(),
                          "win": (d[f"{me}_score"] > d[f"{op}_score"]).astype(float),
                          "ra": d[f"{op}_score"], "p": p, "side": side})
        rows.append(t)
    T = pd.concat(rows).sort_values(["date", "i"]).reset_index(drop=True)
    T["wvm"] = T["win"] - T["p"]
    gs = T.groupby("st", sort=False)
    T["st_starts"] = gs.cumcount()
    T["st_wvm10"] = gs["wvm"].transform(lambda s: s.rolling(10, min_periods=5).mean().shift(1))
    T["st_ra5"] = gs["ra"].transform(lambda s: s.rolling(5, min_periods=3).mean().shift(1))
    if backup:
        T = T.sort_values(["team", "date", "i"])
        share = np.full(len(T), np.nan)
        for team, idx in T.groupby("team", sort=False).indices.items():
            sts = T["st"].values[idx]
            for j in range(len(idx)):
                prev = sts[max(0, j - 10):j]
                if len(prev) >= 5:
                    share[idx[j]] = (prev == sts[j]).mean()
        T["st_share10"] = share
        T["st_backup"] = (T["st_share10"] < 0.3).astype(float).where(T["st_share10"].notna())
    feats = [c for c in T.columns if c.startswith("st_")]
    for side in ("h", "a"):
        sub = T[T.side == side].set_index("i")[feats].add_prefix(f"{side}_")
        d = d.join(sub)
    return d
