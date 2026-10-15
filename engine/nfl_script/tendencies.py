"""
Leak-free NFL team-week tendencies from nflverse play-by-play.

For every (team, game) row the features use ONLY games played strictly before
that game: rolling last-4 / last-8 games (crossing season boundaries) and
season-to-date, each shrunk toward a prior = league average of the PREVIOUS
season blended with the team's previous-season value (so week 1 is all prior).

usage: python -m engine.nfl_script.tendencies      (builds + caches the tables)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PBP = ROOT / "data" / "raw" / "nfl_pbp"
CACHE = PBP / "_cache"
FIRST, LAST = 2001, 2026
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

COLS = ["game_id", "play_id", "season", "posteam", "defteam", "play_type", "pass", "rush",
        "pass_attempt", "rush_attempt", "sack", "interception", "fumble_lost", "epa", "success",
        "yards_gained", "xpass", "wp", "down", "half_seconds_remaining", "game_seconds_remaining",
        "yardline_100", "fixed_drive", "fixed_drive_result", "two_point_attempt", "qtr"]

# (stat, numerator, denominator, shrink weight K in denominator units)
STATS = [
    ("epa_pass", "epa_pass", "n_pass", 120),
    ("epa_rush", "epa_rush", "n_rush", 100),
    ("sr", "succ", "n_play", 180),
    ("sr_pass", "succ_pass", "n_pass", 120),
    ("sr_rush", "succ_rush", "n_rush", 100),
    ("expl", "expl", "n_play", 180),
    ("neu_pass", "neu_pass", "neu_n", 80),
    ("proe", "proe", "proe_n", 80),
    ("spp", "secs", "secs_n", 80),          # neutral seconds per play
    ("plays", "n_play", "g", 3),
    ("pass_att", "pass_att", "g", 3),
    ("rush_att", "rush_att", "g", 3),
    ("rz_td", "rz_td", "rz_trips", 10),
    ("sack_rate", "sacks", "n_pass", 120),
    ("to_rate", "to", "n_play", 180),
    ("pts", "pts", "g", 3),
]
# defense-only vulnerability stats from the player-week box scores
DEF_EXTRA = [
    ("ypt_WR", "rec_yds_WR", "tgt_WR", 50),
    ("ypt_TE", "rec_yds_TE", "tgt_TE", 30),
    ("ypt_RB", "rec_yds_RB", "tgt_RB", 30),
    ("cr_WR", "rec_WR", "tgt_WR", 50),
    ("cr_TE", "rec_TE", "tgt_TE", 30),
    ("cr_RB", "rec_RB", "tgt_RB", 30),
    ("ryd_WR", "rec_yds_WR", "g", 3),
    ("ryd_TE", "rec_yds_TE", "g", 3),
    ("ryd_RB", "rec_yds_RB", "g", 3),
    ("ypc", "rush_yds", "carries", 80),
    ("ypa", "pass_yds", "att", 100),
    ("cmp", "cmpl", "att", 100),
]


def load_games() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data" / "raw" / "nfl_games.csv")
    g = g[(g.season >= FIRST) & (g.season <= LAST)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_MAP)
    g["date"] = pd.to_datetime(g["gameday"])
    return g


def team_game_pbp(season: int) -> pd.DataFrame:
    """Offensive and defensive per-game sums for one season of pbp."""
    f = PBP / f"play_by_play_{season}.parquet"
    d = pq.read_table(f, columns=COLS).to_pandas()
    d = d.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    # seconds until the next snap (pace), only within same drive & same offense
    nxt = d.groupby("game_id")[["game_seconds_remaining", "posteam", "fixed_drive"]].shift(-1)
    el = d["game_seconds_remaining"] - nxt["game_seconds_remaining"]
    same = (nxt["posteam"] == d["posteam"]) & (nxt["fixed_drive"] == d["fixed_drive"])
    p = d[(d["pass"].eq(1) | d["rush"].eq(1)) & d["play_type"].isin(["pass", "run"])
          & d["epa"].notna() & d["two_point_attempt"].ne(1) & d["posteam"].notna()].copy()
    p["el"] = el.loc[p.index].where(same.loc[p.index])
    ip, ir = p["pass"].eq(1), p["pass"].ne(1) & p["rush"].eq(1)
    yg = p["yards_gained"].fillna(0)
    neu = p["wp"].between(0.2, 0.8) & (p["half_seconds_remaining"] > 120) & p["qtr"].le(4)
    neu12 = neu & p["down"].isin([1, 2])
    out = pd.DataFrame({
        "game_id": p["game_id"], "off": p["posteam"], "dfn": p["defteam"],
        "n_play": 1.0, "n_pass": ip * 1.0, "n_rush": ir * 1.0,
        "epa_pass": p["epa"].where(ip, 0.0), "epa_rush": p["epa"].where(ir, 0.0),
        "succ": p["success"].fillna(0), "succ_pass": p["success"].fillna(0).where(ip, 0.0),
        "succ_rush": p["success"].fillna(0).where(ir, 0.0),
        "expl": ((ip & (yg >= 20)) | (ir & (yg >= 10))) * 1.0,
        "neu_pass": (neu12 & ip) * 1.0, "neu_n": neu12 * 1.0,
        "proe": (ip * 1.0 - p["xpass"]).where(neu & p["xpass"].notna(), 0.0),
        "proe_n": (neu & p["xpass"].notna()) * 1.0,
        "secs": p["el"].clip(0, 60).where(neu & p["el"].between(1, 60), 0.0),
        "secs_n": (neu & p["el"].between(1, 60)) * 1.0,
        "pass_att": (p["pass_attempt"].eq(1) & p["sack"].ne(1)) * 1.0,
        "rush_att": p["rush_attempt"].eq(1) * 1.0,
        "sacks": p["sack"].fillna(0), "to": (p["interception"].fillna(0) + p["fumble_lost"].fillna(0)).clip(0, 1),
    })
    s = out.groupby(["game_id", "off", "dfn"], as_index=False).sum()
    # red-zone trips / TDs per drive
    dr = d[d["posteam"].notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(
        rz=("yardline_100", lambda x: (x <= 20).any()), res=("fixed_drive_result", "first")).reset_index()
    dr["rz_td"] = (dr["rz"] & dr["res"].eq("Touchdown")) * 1.0
    rz = dr.groupby(["game_id", "posteam"]).agg(rz_trips=("rz", "sum"), rz_td=("rz_td", "sum")).reset_index()
    s = s.merge(rz.rename(columns={"posteam": "off"}), on=["game_id", "off"], how="left")
    s["season"] = season
    return s


def box_defense() -> pd.DataFrame:
    """Per defense-game yards/targets allowed by receiver position (player-week box scores)."""
    w = pd.read_csv(ROOT / "data" / "processed" / "nfl_player_week.csv.gz",
                    usecols=["game_id", "team", "opp", "position", "targets", "receptions", "receiving_yards",
                             "carries", "rushing_yards", "attempts", "completions", "passing_yards"])
    w["pos"] = w["position"].where(w["position"].isin(["WR", "TE", "RB"]), "OTH")
    w.loc[w["position"].eq("FB"), "pos"] = "RB"
    rec = w.pivot_table(index=["game_id", "opp"], columns="pos",
                        values=["targets", "receptions", "receiving_yards"], aggfunc="sum", fill_value=0)
    rec.columns = [{"targets": "tgt", "receptions": "rec", "receiving_yards": "rec_yds"}[a] + "_" + b
                   for a, b in rec.columns]
    tot = w.groupby(["game_id", "opp"])[["carries", "rushing_yards", "attempts", "completions", "passing_yards"]].sum()
    tot.columns = ["carries", "rush_yds", "att", "cmpl", "pass_yds"]
    out = rec.join(tot).reset_index().rename(columns={"opp": "team"})
    return out


def team_game_table(force: bool = False) -> pd.DataFrame:
    """Long table: one row per (team, game) incl. unplayed games, with off_/def_ sums."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = CACHE / "team_game.parquet"
    if fn.exists() and not force:
        return pd.read_parquet(fn)
    parts = [team_game_pbp(s) for s in range(FIRST, LAST + 1) if (PBP / f"play_by_play_{s}.parquet").exists()]
    s = pd.concat(parts, ignore_index=True)
    vals = [c for c in s.columns if c not in ("game_id", "off", "dfn", "season")]
    off = s.rename(columns={"off": "team"}).drop(columns=["dfn", "season"])
    off.columns = ["game_id", "team"] + ["off_" + c for c in vals]
    dfn = s.rename(columns={"dfn": "team"}).drop(columns=["off", "season"])
    dfn.columns = ["game_id", "team"] + ["def_" + c for c in vals]
    g = load_games()
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = g[["game_id", "season", "week", "game_type", "date", f"{side}_team", f"{opp}_team",
               f"{side}_score", f"{opp}_score"]].copy()
        r.columns = ["game_id", "season", "week", "game_type", "date", "team", "opp", "pts_for", "pts_against"]
        r["is_home"] = int(side == "home")
        rows.append(r)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(off, on=["game_id", "team"], how="left").merge(dfn, on=["game_id", "team"], how="left")
    bx = box_defense()
    bx.columns = ["game_id", "team"] + ["def_" + c for c in bx.columns[2:]]
    t = t.merge(bx, on=["game_id", "team"], how="left")
    t["played"] = t["pts_for"].notna() & t["off_n_play"].notna()
    t["off_g"] = t["played"] * 1.0
    t["def_g"] = t["played"] * 1.0
    t["off_pts"] = t["pts_for"]
    t["def_pts"] = t["pts_against"]
    t = t.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    t.to_parquet(fn)
    return t


def _ratio(num, den, prior, k):
    return (num + k * prior) / (den + k)


def tendencies(force: bool = False) -> pd.DataFrame:
    """Pre-game features per (team, game). Every feature uses only earlier games."""
    fn = CACHE / "tendencies.parquet"
    if fn.exists() and not force:
        return pd.read_parquet(fn)
    t = team_game_table()
    specs = [("off", *s) for s in STATS] + [("def", *s) for s in STATS] + [("def", *s) for s in DEF_EXTRA]
    raw_cols = sorted({f"{sd}_{c}" for sd, _, n, d, _ in specs for c in (n, d)})
    X = t[raw_cols].astype(float).fillna(0.0).mul(t["played"].astype(float).values, axis=0)
    X = X.astype(float)
    grp = t["team"]
    # season totals (team & league) for priors -> taken from the PREVIOUS season only
    seas = pd.concat([t[["team", "season"]], X], axis=1).groupby(["team", "season"]).sum()
    lg = seas.groupby(level="season").sum()
    feats = {}
    cum_all = X.groupby(grp).cumsum() - X                                     # strictly before
    key = [t["team"], t["season"]]
    cum_std = X.groupby(key).cumsum() - X
    roll = {}
    for n in (4, 8):
        roll[n] = X.groupby(grp).transform(lambda s: s.rolling(n, min_periods=1).sum().shift(1)).fillna(0.0)
    # played-game positional rolling would skip byes automatically since byes have no row
    for sd, name, num, den, k in specs:
        nc, dc = f"{sd}_{num}", f"{sd}_{den}"
        lg_r = (lg[nc] / lg[dc])
        lg_prev = t["season"].map(lg_r.shift(1)).fillna(t["season"].map(lg_r))  # first season: own (warm-up only)
        tm_r = (seas[nc] / seas[dc].replace(0, np.nan))
        tm_prev = pd.Series(tm_r.groupby(level="team").shift(1).reindex(
            pd.MultiIndex.from_arrays([t["team"], t["season"]])).values, index=t.index)
        prior = lg_prev + 0.5 * (tm_prev.fillna(lg_prev) - lg_prev)
        f = f"{sd}_{name}"
        feats[f + "_std"] = _ratio(cum_std[nc], cum_std[dc], prior, k)
        feats[f + "_l4"] = _ratio(roll[4][nc], roll[4][dc], prior, k)
        feats[f + "_l8"] = _ratio(roll[8][nc], roll[8][dc], prior, k)
        feats[f + "_lg"] = lg_prev
    F = pd.DataFrame(feats)
    F["games_std"] = cum_std["off_g"].values
    F["games_all"] = cum_all["off_g"].values
    keep = ["game_id", "team", "opp", "season", "week", "game_type", "date", "is_home", "played",
            "pts_for", "pts_against", "off_n_play", "off_pass_att", "off_rush_att", "off_n_pass", "off_n_rush"]
    out = pd.concat([t[keep].reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    out.to_parquet(fn)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    team_game_table(force=True)
    T = tendencies(force=True)
    print(T.shape, f"{time.time() - t0:.0f}s")
    pd.set_option("display.width", 250)
    print(T[T.season == 2025].groupby("team")[["off_epa_pass_std", "def_epa_pass_std", "off_proe_std",
                                                "off_spp_std", "off_plays_std", "def_ypt_WR_std"]].last().round(3).head(8))
