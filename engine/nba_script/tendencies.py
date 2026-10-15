"""
Leak-free NBA team tendencies (pre-game) from player box scores.

Pipeline
  team_games(): aggregate player box -> one row per (game_id, team) with both sides' raw counts.
  tendencies(): per-team rolling means (last 5 / last 10 / season-to-date / cross-season EWM),
                every one SHIFTED by one game so the row for a game only sees that team's earlier games.
  league_ref(): league-average of a per-game stat over the 60 days strictly before each date
                (used to turn team ratings into multipliers without peeking at the same day).

Per team-game stats (o_ = what the team did, d_ = what its opponents did against it):
  poss      game possessions = mean of both teams' (FGA + 0.44*FTA - OREB + TOV)
  pace      poss per 48 minutes (OT-adjusted with team minutes / 5)
  ortg/drtg points scored / allowed per 100 possessions
  fg3a_rate 3PA / FGA, ft_rate FTA / FGA, oreb_pct OREB / (OREB + opp DREB), tov_pct TOV / poss
  d_pos_G/F/C  opponent points scored by guards / forwards / centers per 100 possessions
  starter_share  starters' share of the team's points (bench share = 1 - this)
  rest, b2b  days since the team's previous game, back-to-back flag (known pre-game)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.props_nba import proc_path

POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "GF": "F", "C": "C"}
RATE_COLS = ["poss", "pace", "ortg", "drtg", "net", "o_fg3a_rate", "o_ft_rate", "o_oreb_pct", "o_tov_pct",
             "d_fg3a_rate", "d_ft_rate", "d_oreb_pct", "d_tov_pct", "d_pos_G", "d_pos_F", "d_pos_C",
             "o_pos_G", "o_pos_F", "o_pos_C", "starter_share", "pts", "opp_pts"]
WINDOWS = (5, 10)
EWM_HL = 12          # half-life (games) of the cross-season exponentially weighted tendency


def load_box(cols=None) -> pd.DataFrame:
    use = ["date", "season", "season_type", "game_id", "team", "team_name", "opp", "home", "player_id", "player",
           "position", "starter", "did_not_play", "minutes", "pts", "oreb", "dreb", "fg3a", "fga", "fta", "tov",
           "team_score", "opp_score"]
    b = pd.read_csv(proc_path("nba_player_box"), low_memory=False, usecols=cols or use)
    b = b[b.season_type.isin([2, 3])]                       # drop all-star / exhibition rows
    b["date"] = pd.to_datetime(b["date"])
    return b


def team_games(b: pd.DataFrame | None = None) -> pd.DataFrame:
    b = load_box() if b is None else b
    b = b.copy()
    b["pg"] = b.position.map(POS_GROUP).fillna("F")
    for g in "GFC":
        b[f"pts_{g}"] = np.where(b.pg == g, b.pts, 0.0)
    b["pts_starter"] = np.where(b.starter == 1, b.pts, 0.0)
    agg = dict(date=("date", "first"), season=("season", "first"), season_type=("season_type", "first"),
               opp=("opp", "first"), home=("home", "first"), team_name=("team_name", "first"),
               pts=("team_score", "first"), opp_pts=("opp_score", "first"),
               minutes=("minutes", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"),
               dreb=("dreb", "sum"), tov=("tov", "sum"), fg3a=("fg3a", "sum"),
               pts_G=("pts_G", "sum"), pts_F=("pts_F", "sum"), pts_C=("pts_C", "sum"),
               pts_starter=("pts_starter", "sum"), box_pts=("pts", "sum"))
    t = b.groupby(["game_id", "team"], sort=False).agg(**agg).reset_index()
    # incomplete box (missing player rows) -> rate stats unusable; keep final score
    bad = (t.minutes < 235) | (t.fga <= 0) | ((t.box_pts - t.pts).abs() > 2)
    raw = ["fga", "fta", "oreb", "dreb", "tov", "fg3a", "pts_G", "pts_F", "pts_C", "pts_starter"]
    t.loc[bad, raw] = np.nan
    t["team_poss"] = t.fga + 0.44 * t.fta - t.oreb + t.tov
    o = t[["game_id", "team"] + raw + ["team_poss", "minutes"]].rename(
        columns={c: f"opp_{c}" for c in raw + ["team_poss", "minutes"]}).rename(columns={"team": "opp"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["poss"] = (t.team_poss + t.opp_team_poss) / 2
    t["game_min"] = (t.minutes / 5).clip(lower=48).round()
    t["pace"] = t.poss * 48 / t.game_min
    t["ortg"] = 100 * t.pts / t.poss
    t["drtg"] = 100 * t.opp_pts / t.poss
    t["net"] = t.ortg - t.drtg
    t["o_fg3a_rate"] = t.fg3a / t.fga
    t["o_ft_rate"] = t.fta / t.fga
    t["o_oreb_pct"] = t.oreb / (t.oreb + t.opp_dreb)
    t["o_tov_pct"] = t.tov / t.poss
    t["d_fg3a_rate"] = t.opp_fg3a / t.opp_fga
    t["d_ft_rate"] = t.opp_fta / t.opp_fga
    t["d_oreb_pct"] = t.opp_oreb / (t.opp_oreb + t.dreb)
    t["d_tov_pct"] = t.opp_tov / t.poss
    for g in "GFC":
        t[f"o_pos_{g}"] = 100 * t[f"pts_{g}"] / t.poss
        t[f"d_pos_{g}"] = 100 * t[f"opp_pts_{g}"] / t.poss
    t["starter_share"] = t.pts_starter / t.pts
    t["margin"] = t.pts - t.opp_pts
    return t.sort_values(["date", "game_id", "team"]).reset_index(drop=True)


def _shifted(g, col, kind, n=None):
    if kind == "roll":
        return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    if kind == "ewm":
        return g[col].transform(lambda s: s.shift(1).ewm(halflife=EWM_HL, min_periods=1, ignore_na=True).mean())
    if kind == "std":                                      # season-to-date
        return g[col].transform(lambda s: s.shift(1).expanding().mean())
    raise ValueError(kind)


def league_ref(t: pd.DataFrame, col: str, days: int = 60) -> pd.Series:
    """League mean of `col` over team-games in [D-days, D) for each row's date D (strictly prior)."""
    d = t.groupby("date")[col].agg(["sum", "count"]).sort_index()
    full = d.reindex(pd.date_range(d.index.min(), d.index.max()), fill_value=0)
    cs = full.cumsum()
    prev = cs.shift(1)                                     # through D-1
    lag = cs.shift(days + 1)                               # through D-days-1
    s = (prev["sum"] - lag["sum"].fillna(0)) / (prev["count"] - lag["count"].fillna(0))
    # very first dates of the sample: fall back to expanding mean
    s = s.fillna(prev["sum"] / prev["count"])
    return t["date"].map(s)


def tendencies(t: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game_id, team) with pre-game tendencies only (plus same-game outcomes for targets)."""
    t = team_games() if t is None else t
    t = t.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    gt = t.groupby("team", sort=False)
    gs = t.groupby(["team", "season"], sort=False)
    new = {}
    for c in RATE_COLS:
        for n in WINDOWS:
            new[f"{c}_r{n}"] = _shifted(gt, c, "roll", n)
        new[f"{c}_std"] = _shifted(gs, c, "std")
        new[f"{c}_ewm"] = _shifted(gt, c, "ewm")
    t = pd.concat([t, pd.DataFrame(new, index=t.index)], axis=1)
    t["gp_season"] = gs.cumcount()
    prev = gt["date"].shift(1)
    t["rest"] = (t.date - prev).dt.days.clip(upper=7).fillna(7)
    t["b2b"] = (t.rest == 1).astype(int)
    # games in the previous 6 days (schedule density)
    t["g_last7"] = gt["date"].transform(
        lambda s: pd.Series(np.searchsorted(s.values, s.values, side="left")
                            - np.searchsorted(s.values, (s - pd.Timedelta(days=6)).values, side="left"),
                            index=s.index))
    # league reference levels (strictly before date)
    for c in ["pace", "ortg", "d_pos_G", "d_pos_F", "d_pos_C", "starter_share"]:
        t[f"lg_{c}"] = league_ref(t, c)
    for g in "GFC":                                        # opponent-position defence multiplier (>1 = leaky)
        t[f"posdef_{g}"] = t[f"d_pos_{g}_ewm"] / t[f"lg_d_pos_{g}"]
    return t.sort_values(["date", "game_id", "team"]).reset_index(drop=True)


TEND_FEATS = [f"{c}_{k}" for c in ["pace", "ortg", "drtg", "net"] for k in ("r5", "r10", "std", "ewm")] + \
    [f"{c}_ewm" for c in ["o_fg3a_rate", "o_ft_rate", "o_oreb_pct", "o_tov_pct", "d_fg3a_rate", "d_ft_rate",
                          "d_oreb_pct", "d_tov_pct", "starter_share"]] + ["rest", "b2b", "g_last7", "gp_season"]


if __name__ == "__main__":
    import time
    t0 = time.time()
    T = tendencies()
    print(T.shape, f"{time.time() - t0:.0f}s")
    print(T[["date", "team", "opp", "pace", "pace_r10", "pace_ewm", "ortg_ewm", "drtg_ewm", "posdef_G", "posdef_C",
             "starter_share_ewm", "rest", "b2b", "g_last7"]].tail(8).to_string())
    print(T.groupby("season")[["pace", "ortg"]].mean().round(1).T.to_string())
