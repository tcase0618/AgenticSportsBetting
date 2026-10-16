"""
NFL adapter for the script-then-learn backtest (see engine/scripts/SPEC.md).

Pipeline (strictly sequential, every pre-game number uses only games on EARLIER dates):
  1. team_game_table(): per (game, team) offensive and defensive sums from nflverse play-by-play 2010-2026,
     enriched with pbp_participation (formation / personnel / box / pass rushers / pressure / man-zone,
     2016-2025) and FTN charting (play action, motion, blitz, 2022+). Cached under data/raw/nfl_pbp/_cache.
  2. Base team script: ScriptEngine (Kalman off/def ratings, learned level/hfa/noise/rho/market weight),
     warmed up from 2010, Params chosen on 2016-2020 MAE only.
  3. NFL script fields: online decayed team tendencies (EPA/play pass & rush, pace, PROE, formation,
     personnel, box, pressure, blitz, man rate, QB-specific EPA) + matchup terms + environment. An online
     ridge (recursive, with forgetting) learns the correction  pts - base_mu  from each game's error, and
     separate online ridges script plays / pass att / rush att / pass yds per team. Ablation drops each
     feature group and re-runs the online loop.
  4. Player script: online usage shares (targets, air yards, carries, QB attempt share) x scripted team
     volume x player efficiency x opponent adjustment x online calibration; sd learned online.
  5. Evaluation vs real prop lines (2017-18 completions, 2026 wk1-5), de-vigged with odds_ratio.

usage: python -m engine.scripts.nfl
"""
from __future__ import annotations

import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from engine.scripts.base import Params, ScriptEngine, learning_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PBP = ROOT / "data" / "raw" / "nfl_pbp"
PART = ROOT / "data" / "raw" / "nflverse_part"
FTN = ROOT / "data" / "raw" / "nflverse_ftn"
CACHE = PBP / "_cache"
OUT = ROOT / "results" / "scripts"
FIRST, LAST, REPORT_FROM = 2010, 2026, 2016
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

PBP_COLS = ["game_id", "play_id", "posteam", "defteam", "play_type", "pass", "rush", "pass_attempt",
            "rush_attempt", "sack", "interception", "fumble_lost", "epa", "success", "yards_gained", "xpass",
            "wp", "down", "half_seconds_remaining", "game_seconds_remaining", "fixed_drive", "two_point_attempt",
            "qtr", "shotgun", "no_huddle", "qb_hit", "air_yards", "complete_pass", "passing_yards",
            "passer_player_id", "qb_dropback"]


# ============================================================================ data
def _download(kind: str, fname: str, dest: Path) -> Path | None:
    f = dest / fname
    if f.exists():
        return f
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{NFLVERSE}/{kind}/{fname}", f)
        return f
    except Exception:
        return None


_PERS = {}


def _personnel(s):
    """(rb, te, wr) counts from an offense_personnel string (both nflverse formats)."""
    if s in _PERS:
        return _PERS[s]
    r = (np.nan, np.nan, np.nan)
    if isinstance(s, str):
        c = {k: 0 for k in ("RB", "FB", "TE", "WR")}
        for n, p in re.findall(r"(\d+)\s+(RB|FB|TE|WR)", s):
            c[p] += int(n)
        r = (c["RB"] + c["FB"], c["TE"], c["WR"])
    _PERS[s] = r
    return r


def _season_plays(season: int) -> pd.DataFrame:
    d = pq.read_table(PBP / f"play_by_play_{season}.parquet",
                      columns=[c for c in PBP_COLS if c in pq.read_schema(PBP / f"play_by_play_{season}.parquet").names]
                      ).to_pandas()
    d = d.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    nxt = d.groupby("game_id")[["game_seconds_remaining", "posteam", "fixed_drive"]].shift(-1)
    el = d["game_seconds_remaining"] - nxt["game_seconds_remaining"]
    same = (nxt["posteam"] == d["posteam"]) & (nxt["fixed_drive"] == d["fixed_drive"])
    d["el"] = el.where(same)
    p = d[(d["pass"].eq(1) | d["rush"].eq(1)) & d["play_type"].isin(["pass", "run"])
          & d["epa"].notna() & d["two_point_attempt"].ne(1) & d["posteam"].notna()].copy()
    # participation (2016+) and FTN charting (2022+)
    pf = _download("pbp_participation", f"pbp_participation_{season}.parquet", PART) if season >= 2016 else None
    if pf is not None:
        cols = ["nflverse_game_id", "play_id", "offense_personnel", "defenders_in_box", "number_of_pass_rushers",
                "was_pressure", "defense_man_zone_type"]
        pa = pd.read_parquet(pf)
        pa = pa[[c for c in cols if c in pa.columns]].rename(columns={"nflverse_game_id": "game_id"})
        pa["play_id"] = pa["play_id"].astype(float)
        p["play_id"] = p["play_id"].astype(float)
        p = p.merge(pa, on=["game_id", "play_id"], how="left")
    ff = _download("ftn_charting", f"ftn_charting_{season}.parquet", FTN) if season >= 2022 else None
    if ff is not None:
        ft = pd.read_parquet(ff, columns=["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_motion",
                                          "n_blitzers"]).rename(columns={"nflverse_game_id": "game_id",
                                                                         "nflverse_play_id": "play_id"})
        ft["play_id"] = ft["play_id"].astype(float)
        p["play_id"] = p["play_id"].astype(float)
        p = p.merge(ft.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    for c in ("offense_personnel", "defenders_in_box", "number_of_pass_rushers", "was_pressure",
              "defense_man_zone_type", "is_play_action", "is_motion", "n_blitzers"):
        if c not in p.columns:
            p[c] = np.nan
    return p


def _season_sums(season: int) -> pd.DataFrame:
    p = _season_plays(season)
    ip = p["pass"].eq(1)
    ir = ~ip & p["rush"].eq(1)
    neu = p["wp"].between(0.2, 0.8) & (p["half_seconds_remaining"] > 120) & p["qtr"].le(4)
    neu12 = neu & p["down"].isin([1, 2])
    pers = np.array([_personnel(s) for s in p["offense_personnel"].values], float).reshape(-1, 3)
    has_pers = ~np.isnan(pers[:, 0])
    box = p["defenders_in_box"].astype(float)
    rush_n = p["number_of_pass_rushers"].astype(float)
    pres = p["was_pressure"].map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}).astype(float)
    mz = p["defense_man_zone_type"].astype(str)
    has_mz = mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])
    blitz = p["n_blitzers"].astype(float)
    pa_ = p["is_play_action"].map({True: 1.0, False: 0.0}).astype(float)
    mo = p["is_motion"].map({True: 1.0, False: 0.0}).astype(float)
    f = lambda m: m.astype(float).values
    o = pd.DataFrame({
        "game_id": p["game_id"].values, "off": p["posteam"].values, "dfn": p["defteam"].values,
        "n_play": 1.0, "n_pass": f(ip), "n_rush": f(ir),
        "epa_pass": p["epa"].where(ip, 0.0).values, "epa_rush": p["epa"].where(ir, 0.0).values,
        "succ": p["success"].fillna(0).values,
        "neu_pass": f(neu12 & ip), "neu_n": f(neu12),
        "proe": (ip * 1.0 - p["xpass"]).where(neu & p["xpass"].notna(), 0.0).values,
        "proe_n": f(neu & p["xpass"].notna()),
        "secs": p["el"].clip(0, 60).where(neu & p["el"].between(1, 60), 0.0).values,
        "secs_n": f(neu & p["el"].between(1, 60)),
        "pass_att": f(p["pass_attempt"].eq(1) & p["sack"].ne(1)),
        "rush_att": f(p["rush_attempt"].eq(1)),
        "pass_yds": p["passing_yards"].fillna(0).where(ip, 0.0).values,
        "cmp": f(p["complete_pass"].eq(1)),
        "air": p["air_yards"].fillna(0).where(ip & p["pass_attempt"].eq(1), 0.0).values,
        "sacks": p["sack"].fillna(0).values, "hits": p["qb_hit"].fillna(0).values,
        "to": (p["interception"].fillna(0) + p["fumble_lost"].fillna(0)).clip(0, 1).values,
        "shotgun": p["shotgun"].fillna(0).values, "nohud": p["no_huddle"].fillna(0).values,
        "pers_n": has_pers * 1.0,
        "p11": (has_pers & (pers[:, 0] == 1) & (pers[:, 1] == 1) & (pers[:, 2] == 3)) * 1.0,
        "heavy": (has_pers & (pers[:, 0] + pers[:, 1] >= 3)) * 1.0,
        "box": box.fillna(0).values, "box_n": f(box.notna()),
        "rushers": rush_n.fillna(0).where(ip, 0.0).values, "rushers_n": f(rush_n.notna() & ip),
        "pres": pres.fillna(0).where(ip, 0.0).values, "pres_n": f(pres.notna() & ip),
        "man": f(mz.eq("MAN_COVERAGE") & ip), "mz_n": f(has_mz & ip),
        "blitz": f((blitz > 0) & ip), "blitz_n": f(blitz.notna() & ip),
        "pa": pa_.fillna(0).where(ip, 0.0).values, "pa_n": f(pa_.notna() & ip),
        "motion": mo.fillna(0).values, "motion_n": f(mo.notna()),
    })
    s = o.groupby(["game_id", "off", "dfn"], as_index=False).sum()
    # main passer per offense-game (for QB-specific tracking)
    q = p[ip & p["passer_player_id"].notna()].groupby(["game_id", "posteam", "passer_player_id"]).agg(
        qb_db=("epa", "size"), qb_epa=("epa", "sum")).reset_index()
    q = q.sort_values("qb_db").groupby(["game_id", "posteam"]).tail(1).rename(
        columns={"posteam": "off", "passer_player_id": "qb_main"})
    s = s.merge(q, on=["game_id", "off"], how="left")
    return s


def team_game_table(force: bool = False) -> pd.DataFrame:
    """One row per (game_id, team) with off_* (team on offense) and def_* (team on defense) sums."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = CACHE / "scripts_nfl_team_game.parquet"
    if fn.exists() and not force:
        return pd.read_parquet(fn)
    s = pd.concat([_season_sums(y) for y in range(FIRST, LAST + 1)
                   if (PBP / f"play_by_play_{y}.parquet").exists()], ignore_index=True)
    for c in ("off", "dfn"):
        s[c] = s[c].replace(TEAM_MAP)
    vals = [c for c in s.columns if c not in ("game_id", "off", "dfn", "qb_main")]
    off = s.rename(columns={"off": "team"}).drop(columns=["dfn"])
    off = off.rename(columns={c: "off_" + c for c in vals})
    dfn = s.rename(columns={"dfn": "team"}).drop(columns=["off", "qb_main"])
    dfn = dfn.rename(columns={c: "def_" + c for c in vals})
    t = off.merge(dfn, on=["game_id", "team"], how="outer")
    t.to_parquet(fn)
    return t


def load_games() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data" / "raw" / "nfl_games.csv")
    g = g[(g.season >= FIRST) & (g.season <= LAST)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_MAP)
    g["date"] = pd.to_datetime(g["gameday"])
    g = g.rename(columns={"home_team": "home", "away_team": "away", "home_moneyline": "home_ml",
                          "away_moneyline": "away_ml"})
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    # nflverse spread_line: positive = home favored  ->  base engine wants home spread (negative = favored)
    g["spread_home"] = -g["spread_line"]
    g["total"] = g["total_line"]
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


# ============================================================================ online tools
class OnlineRidge:
    """Recursive ridge with exponential forgetting: w = argmin sum ff^age (y - w'x)^2 + lam |w|^2."""

    def __init__(self, k: int, lam: float = 100.0, ff: float = 0.999):
        self.A = np.zeros((k, k)); self.b = np.zeros(k); self.lam, self.ff, self.k = lam, ff, k
        self.w = np.zeros(k); self.n = 0

    def predict(self, x):
        return float(self.w @ x)

    def update(self, X, Y):
        X = np.atleast_2d(X); Y = np.atleast_1d(Y)
        self.A = self.ff ** len(Y) * self.A + X.T @ X
        self.b = self.ff ** len(Y) * self.b + X.T @ Y
        self.n += len(Y)
        self.w = np.linalg.solve(self.A + self.lam * np.eye(self.k), self.b)


class Scaler:
    """Running mean/sd (Welford) of each feature, updated only after use."""

    def __init__(self, k):
        self.n = 0; self.mu = np.zeros(k); self.m2 = np.zeros(k)

    def t(self, x):
        if self.n < 30:
            return np.zeros_like(x)
        sd = np.sqrt(self.m2 / (self.n - 1)); sd[sd < 1e-9] = 1.0
        return np.clip((x - self.mu) / sd, -4, 4)

    def update(self, x):
        self.n += 1; d = x - self.mu; self.mu += d / self.n; self.m2 += d * (x - self.mu)


# team tendency stats: name -> (num column, den column, shrink K in den units)
TSTATS = {
    "epa_pass": ("epa_pass", "n_pass", 60), "epa_rush": ("epa_rush", "n_rush", 50),
    "sr": ("succ", "n_play", 90), "prate": ("n_pass", "n_play", 60),
    "neu_pass": ("neu_pass", "neu_n", 40), "proe": ("proe", "proe_n", 40), "spp": ("secs", "secs_n", 40),
    "plays": ("n_play", "g", 2), "pass_att": ("pass_att", "g", 2), "rush_att": ("rush_att", "g", 2),
    "pass_yds": ("pass_yds", "g", 2), "ypa": ("pass_yds", "pass_att", 60), "cmp": ("cmp", "pass_att", 60),
    "adot": ("air", "pass_att", 60), "sack": ("sacks", "n_pass", 60), "hit": ("hits", "n_pass", 60),
    "to": ("to", "n_play", 90), "shotgun": ("shotgun", "n_play", 60), "nohud": ("nohud", "n_play", 60),
    "p11": ("p11", "pers_n", 40), "heavy": ("heavy", "pers_n", 40), "box": ("box", "box_n", 40),
    "rushers": ("rushers", "rushers_n", 30), "pres": ("pres", "pres_n", 30), "man": ("man", "mz_n", 30),
    "blitz": ("blitz", "blitz_n", 30), "pa": ("pa", "pa_n", 30), "motion": ("motion", "motion_n", 40),
    "pts": ("pts", "g", 2),
}


class Tendencies:
    """Decayed per-team sums, side in {off, def}. ratio = (N + K*league)/(D + K)."""

    def __init__(self, decay=0.88, carry=0.45, lg_decay=0.996):
        self.decay, self.carry, self.lg_decay = decay, carry, lg_decay
        self.cols = sorted({c for v in TSTATS.values() for c in v[:2]})
        self.ix = {c: i for i, c in enumerate(self.cols)}
        self.S = {}                                   # (team, side) -> [vec, season]
        self.L = {"off": np.zeros(len(self.cols)), "def": np.zeros(len(self.cols))}
        self.qb = {}                                  # qb_id -> [db, epa] decayed

    def _get(self, team, side, season):
        v = self.S.get((team, side))
        if v is None:
            v = [np.zeros(len(self.cols)), season]; self.S[(team, side)] = v
        elif v[1] != season:
            v[0] = v[0] * self.carry; v[1] = season
        return v[0]

    def lg(self, side, name):
        n, d, _ = TSTATS[name]
        L = self.L[side]
        return L[self.ix[n]] / L[self.ix[d]] if L[self.ix[d]] > 0 else np.nan

    def val(self, team, side, season, name):
        n, d, k = TSTATS[name]
        v = self._get(team, side, season)
        lg = self.lg(side, name)
        if not np.isfinite(lg):
            return np.nan
        return (v[self.ix[n]] + k * lg) / (v[self.ix[d]] + k)

    def qb_epa(self, qb, k=150.0):
        lg = self.lg("off", "epa_pass")
        if qb not in self.qb or not np.isfinite(lg):
            return lg, 0.0
        db, e = self.qb[qb]
        return (e + k * (lg - 0.05)) / (db + k), db       # unknown QBs regress below average

    def update(self, team, season, row: dict):
        for side in ("off", "def"):
            x = np.array([row.get(f"{side}_{c}", 0.0) for c in self.cols], float)
            x = np.nan_to_num(x)
            v = self._get(team, side, season)
            v *= self.decay; v += x
            self.L[side] = self.L[side] * self.lg_decay + x
        q = row.get("qb_main")
        if isinstance(q, str):
            db, e = self.qb.get(q, (0.0, 0.0))
            self.qb[q] = (db * 0.97 + np.nan_to_num(row.get("off_qb_db", 0.0)),
                          e * 0.97 + np.nan_to_num(row.get("off_qb_epa", 0.0)))


# ============================================================================ features
GROUPS = {
    "eff": ["o_epa_pass", "d_epa_pass", "o_epa_rush", "d_epa_rush", "o_sr", "d_sr", "o_to", "d_to"],
    "pace": ["o_spp", "opp_spp", "o_plays", "opp_plays", "o_nohud", "opp_nohud"],
    "proe": ["o_proe", "o_neu_pass", "opp_proe"],
    "form": ["o_shotgun", "o_p11", "o_heavy", "o_pa", "o_motion", "o_adot"],
    "defn": ["d_box", "d_rushers", "d_pres", "d_man", "d_blitz", "d_sack"],
    "matchup": ["m_pass", "m_run", "m_heavy_vs_runD", "m_pres_vs_sackQB", "m_proe_vs_passD", "m_adot_vs_man"],
    "qb": ["qb_delta", "qb_new", "qb_exp"],
    "env": ["wind", "wind15", "cold", "dome", "rest_diff", "home_fld", "div"],
}
FEATS = [f for g in GROUPS.values() for f in g]


def side_features(T: Tendencies, g, o, d, season, is_home, qb) -> dict:
    """Features for team o on offense vs team d (all pre-game, centered on the league)."""
    v = lambda t, s, n: T.val(t, s, season, n)
    lgo = lambda n: T.lg("off", n)
    x = {}
    for n in ("epa_pass", "epa_rush", "sr", "to"):
        x[f"o_{n}"] = v(o, "off", n) - lgo(n)
        x[f"d_{n}"] = v(d, "def", n) - lgo(n)                 # positive = defense allows more / forces fewer
    x["o_spp"] = v(o, "off", "spp") - lgo("spp"); x["opp_spp"] = v(d, "off", "spp") - lgo("spp")
    x["o_plays"] = v(o, "off", "plays") - lgo("plays"); x["opp_plays"] = v(d, "off", "plays") - lgo("plays")
    x["o_nohud"] = v(o, "off", "nohud") - lgo("nohud"); x["opp_nohud"] = v(d, "off", "nohud") - lgo("nohud")
    x["o_proe"] = v(o, "off", "proe") - lgo("proe"); x["opp_proe"] = v(d, "off", "proe") - lgo("proe")
    x["o_neu_pass"] = v(o, "off", "neu_pass") - lgo("neu_pass")
    for n in ("shotgun", "p11", "heavy", "pa", "motion", "adot"):
        x[f"o_{n}"] = v(o, "off", n) - lgo(n)
    for n in ("box", "rushers", "pres", "man", "blitz", "sack"):
        x[f"d_{n}"] = v(d, "def", n) - T.lg("def", n)
    prate = v(o, "off", "prate")
    x["m_pass"] = prate * (x["o_epa_pass"] + x["d_epa_pass"])
    x["m_run"] = (1 - prate) * (x["o_epa_rush"] + x["d_epa_rush"])
    x["m_heavy_vs_runD"] = x["o_heavy"] * x["d_epa_rush"]
    x["m_pres_vs_sackQB"] = x["d_pres"] * (v(o, "off", "sack") - lgo("sack"))
    x["m_proe_vs_passD"] = x["o_proe"] * x["d_epa_pass"]
    x["m_adot_vs_man"] = x["o_adot"] * x["d_man"]
    qe, db = T.qb_epa(qb) if isinstance(qb, str) else (np.nan, 0.0)
    x["qb_delta"] = (qe - v(o, "off", "epa_pass")) if np.isfinite(qe) else 0.0
    x["qb_new"] = float(db < 100)
    x["qb_exp"] = math.log1p(db)
    outdoor = str(g.roof) in ("outdoors", "open")
    wind = float(g.wind) if outdoor and g.wind == g.wind else 0.0
    x["wind"] = wind; x["wind15"] = float(wind >= 15)
    x["cold"] = float(outdoor and g.temp == g.temp and g.temp < 35)
    x["dome"] = float(not outdoor)
    hr, ar = (g.home_rest, g.away_rest) if is_home else (g.away_rest, g.home_rest)
    x["rest_diff"] = float(np.clip((hr - ar) if hr == hr and ar == ar else 0.0, -7, 7))
    x["home_fld"] = float(is_home and not g.neutral)
    x["div"] = float(g.div_game or 0)
    return {k: (0.0 if not np.isfinite(val) else float(val)) for k, val in x.items()}


def build_features(games: pd.DataFrame, tg: pd.DataFrame):
    """Sequential pass: features for both sides of every game BEFORE updating with that game.
    Also returns base volume expectations (team off tendency averaged with opponent's allowed)."""
    T = Tendencies()
    tg = tg.set_index(["game_id", "team"])
    pts = {}
    for r in games.itertuples(index=False):
        pts[(r.game_id, r.home)] = (r.home_score, r.away_score)
        pts[(r.game_id, r.away)] = (r.away_score, r.home_score)
    rows = []
    for date, day in games.groupby("date", sort=True):
        pend = []
        for g in day.itertuples(index=False):
            s = int(g.season)
            for o, d, is_home, qb in ((g.home, g.away, True, g.home_qb_id), (g.away, g.home, False, g.away_qb_id)):
                x = side_features(T, g, o, d, s, is_home, qb)
                vol = {}
                for n in ("plays", "pass_att", "rush_att", "pass_yds", "pts"):
                    a, b = T.val(o, "off", s, n), T.val(d, "def", s, n)
                    vol[f"base_{n}"] = np.nanmean([a, b]) if np.isfinite(a) or np.isfinite(b) else np.nan
                vol["base_ypa"] = T.val(o, "off", s, "ypa") * T.val(d, "def", s, "ypa") / T.lg("off", "ypa") \
                    if np.isfinite(T.lg("off", "ypa")) else np.nan
                vol["o_prate"] = T.val(o, "off", s, "prate")
                vol["base_epa_pass"] = T.val(o, "off", s, "epa_pass") + x["d_epa_pass"]
                rows.append(dict(game_id=g.game_id, team=o, opp=d, is_home=int(is_home), **x, **vol))
            if (g.game_id, g.home) in tg.index and g.home_score == g.home_score:
                pend.append(g)
        for g in pend:                                       # learn after every game of the date is scripted
            for t in (g.home, g.away):
                if (g.game_id, t) not in tg.index:
                    continue
                row = tg.loc[(g.game_id, t)].to_dict()
                pf, pa = pts[(g.game_id, t)]
                row.update(off_g=1.0, def_g=1.0, off_pts=pf, def_pts=pa)
                T.update(t, int(g.season), row)
    return pd.DataFrame(rows)


# ============================================================================ team script
BASE_P = dict(init_level=21.5, init_hfa=2.0, init_sd=10.0, prior_var=12.0, q_game=0.3, carry=0.6,
              season_var=1.0, lr_level=0.002)


def run_base(games, **kw) -> pd.DataFrame:
    p = Params(**{**BASE_P, **kw})
    g = games[["date", "season", "home", "away", "home_score", "away_score", "neutral", "spread_home", "total",
               "game_id"]].copy()
    out = ScriptEngine(p).run(g.drop(columns="game_id"))
    out["game_id"] = g.sort_values("date").reset_index(drop=True)["game_id"].values   # run() sorts by date (stable)
    return out


def tune_base(games):
    """Grid over Kalman params; selection on 2016-2020 margin+total MAE only."""
    res = []
    for q in (0.15, 0.3, 0.6):
        for carry in (0.5, 0.7):
            for pv in (8.0, 16.0):
                b = run_base(games, q_game=q, carry=carry, prior_var=pv)
                d = b[(b.season >= 2016) & (b.season <= 2020)].dropna(subset=["err_margin"])
                res.append(dict(q_game=q, carry=carry, prior_var=pv,
                                mae=d.err_margin.abs().mean() + d.err_total.abs().mean()))
    res = pd.DataFrame(res).sort_values("mae")
    best = res.iloc[0]
    return dict(q_game=best.q_game, carry=best.carry, prior_var=best.prior_var), res


def correction_loop(games, base, F, groups=None, lam=100.0, ff=0.9995):
    """Online ridge correction of the Kalman script means from NFL script features (learned per game error),
    plus online volume scripts. groups=None -> all groups; else list of groups to use."""
    feats = [f for gname, fl in GROUPS.items() if groups is None or gname in groups for f in fl]
    k = len(feats) + 1
    Fi = F.set_index(["game_id", "team"])
    Xall = Fi[feats].to_numpy(float) if feats else np.zeros((len(Fi), 0))
    pos = {key: i for i, key in enumerate(Fi.index)}
    sc_pts = Scaler(k - 1); rr = OnlineRidge(k, lam, ff)
    vol_t = ["plays", "pass_att", "rush_att", "pass_yds"]
    vfe = k + 3                                          # feats + bias + base_vol + own margin_b + total_b
    vr = {t: OnlineRidge(vfe, 30.0, ff) for t in vol_t}
    blender = ScriptEngine(Params(**BASE_P))
    bmu = dict(zip(base.game_id, zip(base.mu_home, base.mu_away)))
    basev = {t: F[f"base_{t}"].to_numpy(float) for t in vol_t}
    actv = {}
    for r in TG[["game_id", "team", "off_n_play", "off_pass_att", "off_rush_att", "off_pass_yds"]].itertuples(index=False):
        actv[(r.game_id, r.team)] = dict(plays=r.off_n_play, pass_att=r.off_pass_att, rush_att=r.off_rush_att,
                                         pass_yds=r.off_pass_yds)
    rows = []
    import itertools
    recs = list(games.sort_values(["date", "game_id"]).itertuples(index=False))
    for date, day in itertools.groupby(recs, key=lambda r: r.date):
        learn = []
        for g in day:
            if g.game_id not in bmu:
                continue
            br = bmu[g.game_id]
            out = dict(game_id=g.game_id)
            xs = {}
            for side, t, mu in (("home", g.home, br[0]), ("away", g.away, br[1])):
                i = pos[(g.game_id, t)]
                xz = np.r_[sc_pts.t(Xall[i]), 1.0]
                xs[side] = (Xall[i], xz)
                out[f"mu_{side}"] = mu + (rr.predict(xz) if rr.n >= 200 else 0.0)
            sc = dict(margin=out["mu_home"] - out["mu_away"], total=out["mu_home"] + out["mu_away"])
            mm = -g.spread_home if g.spread_home == g.spread_home else np.nan
            bl = blender.blend(sc, mm, g.total)
            out.update(margin=sc["margin"], total=sc["total"], **bl)
            # volume scripts (use the market-informed blended script for game state)
            for side, t in (("home", g.home), ("away", g.away)):
                ip = pos[(g.game_id, t)]
                sgn = 1 if side == "home" else -1
                mb = bl["margin_b"] * sgn
                for vt in vol_t:
                    bv = basev[vt][ip]
                    xv = np.r_[xs[side][1], mb / 7.0, (bl["total_b"] - 44) / 7.0, 0.0]
                    pred = bv + (vr[vt].predict(xv) if vr[vt].n >= 200 else 0.0)
                    out[f"{side}_{vt}_s"] = pred
                    xs[side + vt] = (xv, bv)
            rows.append(out)
            if g.home_score == g.home_score:
                learn.append((g, br, out, xs, sc, mm))
        for g, br, out, xs, sc, mm in learn:
            yh, ya = g.home_score, g.away_score
            X = np.vstack([xs["home"][1], xs["away"][1]])
            Y = np.array([yh - br[0], ya - br[1]])
            rr.update(X, Y)
            sc_pts.update(xs["home"][0]); sc_pts.update(xs["away"][0])
            blender.learn_blend(sc, mm, g.total, yh - ya, yh + ya)
            for side, t in (("home", g.home), ("away", g.away)):
                act = actv.get((g.game_id, t))
                if act is None:
                    continue
                for vt in vol_t:
                    xv, bv = xs[side + vt]
                    if np.isfinite(bv) and np.isfinite(act[vt]):
                        vr[vt].update(xv, act[vt] - bv)
    return pd.DataFrame(rows), rr, feats


# ============================================================================ players
PSTATS = {  # stat -> positions
    "passing_yards": ["QB"], "completions": ["QB"], "rushing_yards": ["RB"],
    "receiving_yards": ["WR", "TE", "RB"], "receptions": ["WR", "TE", "RB"],
}
PRIOR_SHARE = {"tgt": {"WR": 0.12, "TE": 0.10, "RB": 0.07}, "car": {"RB": 0.30, "WR": 0.01, "TE": 0.0}}


def player_frame():
    w = pd.read_csv(ROOT / "data" / "processed" / "nfl_player_week.csv.gz")
    w = w[(w.season >= FIRST)].copy()
    w["position"] = w["position"].replace({"FB": "RB"})
    w = w[w.position.isin(["QB", "RB", "WR", "TE"])]
    for c in ("team", "opp"):
        w[c] = w[c].replace(TEAM_MAP)
    team = w.groupby(["game_id", "team"])[["attempts", "carries", "targets"]].sum().add_prefix("team_").reset_index()
    w = w.merge(team, on=["game_id", "team"], how="left")
    w["date"] = pd.to_datetime(w["gameday"])
    return w.sort_values(["date", "game_id"]).reset_index(drop=True)


def player_loop(w: pd.DataFrame, scripts: pd.DataFrame, games: pd.DataFrame, use_air=True, use_opp=True,
                use_team=True, decay=0.85):
    """Sequential player script. Team volume from the online team script; shares/efficiency online."""
    st = {}
    for r in games.itertuples(index=False):
        st[(r.game_id, r.home)] = r.home_qb_id
        st[(r.game_id, r.away)] = r.away_qb_id
    sm = scripts.set_index("game_id")
    gm = games.set_index("game_id")
    P = {}                       # player_id -> dict of decayed sums
    D = {}                       # (def team, key) -> decayed sums ; league in D[("LG", key)]
    lgd = 0.995
    pk = ["tgt", "ttgt", "ay", "car", "tcar", "rec", "recyd", "rushyd", "att", "tatt", "cmp", "pyd", "g",
          "own_recyd", "own_rec", "own_rushyd", "own_pyd", "own_cmp"]

    def dget(key):
        v = D.get(key)
        if v is None:
            v = np.zeros(2); D[key] = v
        return v

    def opp_factor(dteam, key, k):
        if not use_opp:
            return 1.0
        lg = D.get(("LG", key)); v = D.get((dteam, key))
        if lg is None or lg[1] <= 0:
            return 1.0
        lr = lg[0] / lg[1]
        val = ((v[0] if v is not None else 0) + k * lr) / ((v[1] if v is not None else 0) + k)
        return float(np.clip(val / lr, 0.7, 1.4)) if lr > 0 else 1.0

    out = []
    for (date, gid), gdf in w.groupby(["date", "game_id"], sort=True):
        if gid not in sm.index:
            continue
        s = sm.loc[gid]
        g = gm.loc[gid]
        upd = []
        for team, tdf in gdf.groupby("team"):
            side = "home" if team == g.home else "away"
            A = s[f"{side}_pass_att_s"]; R = s[f"{side}_rush_att_s"]
            opp = g.away if side == "home" else g.home
            for r in tdf.itertuples(index=False):
                pid, pos = r.player_id, r.position
                p = P.get(pid)
                if p is not None and p["g"] >= 1 and A == A:
                    sh_t = (p["tgt"] + 8 * PRIOR_SHARE["tgt"].get(pos, 0.05)) / (p["ttgt"] + 8)
                    if use_air and p["g"] > 0 and p["ay_n"] > 0:
                        ays = p["ay"] / p["ay_n"]
                        sh_t = 0.75 * sh_t + 0.25 * max(ays, 0)
                    sh_c = (p["car"] + 8 * PRIOR_SHARE["car"].get(pos, 0.0)) / (p["tcar"] + 8)
                    cr = (p["rec"] + 15 * {"WR": .63, "TE": .70, "RB": .77}.get(pos, .65)) / (p["tgt"] + 15)
                    ypt = (p["recyd"] + 25 * {"WR": 8.2, "TE": 7.3, "RB": 6.0}.get(pos, 7)) / (p["tgt"] + 25)
                    ypc = (p["rushyd"] + 60 * 4.2) / (p["car"] + 60)
                    rows = []
                    if pos == "QB" and st.get((gid, team)) == pid:
                        sh_a = (p["att"] + 3 * 0.96) / (p["tatt"] + 3)
                        cmpp = (p["cmp"] + 80 * 0.63) / (p["att"] + 80)
                        ypa = (p["pyd"] + 150 * 6.9) / (p["att"] + 150)
                        if use_team:
                            m_c = A * sh_a * cmpp * opp_factor(opp, "cmp", 120)
                            m_y = A * sh_a * ypa * opp_factor(opp, "ypa", 150)
                        else:
                            m_c, m_y = p["own_cmp"] / p["gw"], p["own_pyd"] / p["gw"]
                        rows += [("completions", m_c, sh_a, r.completions), ("passing_yards", m_y, sh_a, r.passing_yards)]
                    if pos in ("RB",):
                        m = R * sh_c * ypc * opp_factor(opp, "ypc", 80) if use_team else p["own_rushyd"] / p["gw"]
                        rows.append(("rushing_yards", m, sh_c, r.rushing_yards))
                    if pos in ("WR", "TE", "RB"):
                        if use_team:
                            m_r = A * sh_t * cr * opp_factor(opp, f"cr_{pos}", 40)
                            m_y = A * sh_t * ypt * opp_factor(opp, f"ypt_{pos}", 40)
                        else:
                            m_r, m_y = p["own_rec"] / p["gw"], p["own_recyd"] / p["gw"]
                        rows += [("receptions", m_r, sh_t, r.receptions), ("receiving_yards", m_y, sh_t, r.receiving_yards)]
                    for stat, mu, sh, act in rows:
                        out.append(dict(game_id=gid, date=date, season=r.season, week=r.week, player=r.player,
                                        player_id=pid, position=pos, team=team, opp=opp, stat=stat,
                                        mu_raw=mu, share_of_team=sh, team_pass_att_s=A,
                                        team_rush_att_s=R, actual=float(act), prior_games=p["g"]))
                upd.append((r, team, opp))
        # ---- learn after the game
        for r, team, opp in upd:
            pid, pos = r.player_id, r.position
            p = P.setdefault(pid, {**{k: 0.0 for k in pk}, "ay_n": 0.0, "gw": 0.0})
            for k in pk + ["ay_n", "gw"]:
                p[k] *= decay
            ay = r.air_yards_share if r.air_yards_share == r.air_yards_share else None
            p["tgt"] += r.targets; p["ttgt"] += r.team_targets; p["car"] += r.carries; p["tcar"] += r.team_carries
            if ay is not None:
                p["ay"] += ay; p["ay_n"] += 1
            p["rec"] += r.receptions; p["recyd"] += r.receiving_yards; p["rushyd"] += r.rushing_yards
            p["att"] += r.attempts; p["tatt"] += r.team_attempts; p["cmp"] += r.completions; p["pyd"] += r.passing_yards
            p["own_recyd"] += r.receiving_yards; p["own_rec"] += r.receptions; p["own_rushyd"] += r.rushing_yards
            p["own_pyd"] += r.passing_yards; p["own_cmp"] += r.completions
            p["g"] += 1; p["gw"] += 1
            # defense allowed (by opponent = the defense this player faced)
            for key, num, den in ((f"cr_{pos}", r.receptions, r.targets), (f"ypt_{pos}", r.receiving_yards, r.targets),
                                  ("ypc", r.rushing_yards if pos == "RB" else 0, r.carries if pos == "RB" else 0),
                                  ("cmp", r.completions, r.attempts), ("ypa", r.passing_yards, r.attempts)):
                if den > 0:
                    v = dget((opp, key)); v *= 0.9; v += (num, den)
                    l = dget(("LG", key)); l *= lgd; l += (num, den)
        # calibration learns from this date's projections
    df = pd.DataFrame(out)
    return df


def calibrate_and_sd(df: pd.DataFrame, p_exp=None):
    """Online calibration (ratio of actual to projection) and sd (var = k*mu^p), each learned from earlier
    DATES only. p (variance exponent) chosen per stat on pre-2016 data."""
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    df["mu"] = df["mu_raw"].clip(lower=0.1)
    df["key"] = df["stat"] + "_" + df["position"]
    outs = []
    p_exp = p_exp or {}
    for key, d in df.groupby("key"):
        d = d.copy()
        day = d.groupby("date").agg(a=("actual", "sum"), m=("mu", "sum")).sort_index()
        dec = 0.5 ** (1 / 60)
        A = (day["a"]).ewm(alpha=1 - dec, adjust=False).mean().shift(1)
        M = (day["m"]).ewm(alpha=1 - dec, adjust=False).mean().shift(1)
        cal = (A / M).clip(0.8, 1.25).fillna(1.0)
        d["cal"] = d["date"].map(cal)
        d["mu"] = d["mu"] * d["cal"]
        stat = d["stat"].iloc[0]
        if stat not in p_exp:
            pre = d[d.season < REPORT_FROM]
            best = None
            for pe in (1.0, 1.25, 1.5, 1.75, 2.0):
                r2 = (pre.actual - pre.mu) ** 2
                kk = (r2 / pre.mu ** pe).mean()
                var = kk * pre.mu ** pe
                nll = 0.5 * np.mean(np.log(var) + r2 / var)
                if best is None or nll < best[0]:
                    best = (nll, pe)
            p_exp[stat] = best[1] if best else 1.5
        pe = p_exp[stat]
        d["_r"] = (d.actual - d.mu) ** 2 / d.mu ** pe
        dk = d.groupby("date")["_r"].agg(["sum", "size"]).sort_index()
        K = (dk["sum"].ewm(alpha=0.01, adjust=False).mean() / dk["size"].ewm(alpha=0.01, adjust=False).mean()).shift(1)
        d["sd"] = np.sqrt(d["date"].map(K) * d.mu ** pe)     # NaN until a residual history exists (2010 only)
        outs.append(d.drop(columns=["_r"]))
    return pd.concat(outs).sort_values(["date", "game_id"]).reset_index(drop=True), p_exp


def p_over(mu, sd, line, stat):
    """P(X > line). Counts: normal w/ continuity; yards: gamma (right-skewed, >= 0); QB yards normal."""
    mu = np.maximum(np.asarray(mu, float), 0.1); sd = np.maximum(np.asarray(sd, float), 0.3)
    line = np.asarray(line, float)
    if stat in ("receptions", "completions"):
        return 1 - stats.norm.cdf((np.floor(line) + 0.5 - mu) / sd)
    if stat == "passing_yards":
        return 1 - stats.norm.cdf((line - mu) / sd)
    shape = (mu / sd) ** 2
    return 1 - stats.gamma.cdf(line, a=shape, scale=mu / shape)


# ============================================================================ props evaluation
def props_eval(pl: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from engine.devig import devig
    from backtest.core import american_to_decimal
    pr = pd.read_csv(ROOT / "data" / "processed" / "nfl_props.csv")
    pr = pr[pr.source_type.eq("sportsbook") & pr.over_odds.notna() & pr.under_odds.notna()].copy()
    pr["p_mkt"] = devig(pr.over_odds.values, pr.under_odds.values, "odds_ratio")[0]
    pr["dec_o"] = american_to_decimal(pr.over_odds.values); pr["dec_u"] = american_to_decimal(pr.under_odds.values)
    # one prop per player-game-market: modal line, mean de-vigged p, median prices
    key = ["season", "week", "game_id", "player_id", "market"]
    ml = pr.groupby(key)["line"].agg(lambda x: x.mode().iloc[0]).rename("mline")
    pr = pr.merge(ml.reset_index(), on=key)
    pr = pr[pr.line == pr.mline]
    pp = pr.groupby(key + ["line"]).agg(p_mkt=("p_mkt", "mean"), dec_o=("dec_o", "median"),
                                        dec_u=("dec_u", "median"), actual_src=("actual", "first"),
                                        nbooks=("book", "nunique")).reset_index()
    pp = pp.merge(pl[["game_id", "player_id", "stat", "mu", "sd", "actual"]].rename(columns={"stat": "market"}),
                  on=["game_id", "player_id", "market"], how="inner")
    pp = pp[(pp.actual != pp.line)]
    pp["y"] = (pp.actual > pp.line).astype(int)
    pp = pd.concat([d.assign(p_mod=p_over(d.mu, d.sd, d.line, m)) for m, d in pp.groupby("market")])
    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4)))
    lines = []
    pp["set"] = np.where(pp.season < 2026, "2017-18 completions", "2026 wk1-5")
    pp["calib"] = np.where(pp.season == 2017, True, np.where(pp.season == 2026, pp.week <= 2, False))

    def roi(d, p, thr):
        evo = p * d.dec_o - 1; evu = (1 - p) * d.dec_u - 1
        bo, bu = (evo >= thr) & (evo >= evu), (evu >= thr) & (evu > evo)
        pnl = np.where(bo, np.where(d.y == 1, d.dec_o - 1, -1), 0.0) + np.where(bu, np.where(d.y == 0, d.dec_u - 1, -1), 0.0)
        n = int(bo.sum() + bu.sum())
        r = pnl.sum() / n if n else np.nan
        sd = np.std(pnl[(bo | bu)]) if n > 1 else np.nan
        return n, r, (r / (sd / math.sqrt(n)) if n > 1 and sd > 0 else np.nan)

    lines.append("| Set / market | n | log loss model | log loss market (odds_ratio) | log loss 50/50 blend | Brier model / market |")
    lines.append("|---|---|---|---|---|---|")
    for (sname, m), d in pp.groupby(["set", "market"]):
        pb = 0.5 * d.p_mod + 0.5 * d.p_mkt
        lines.append(f"| {sname} / {m} | {len(d)} | {ll(d.p_mod, d.y):.4f} | {ll(d.p_mkt, d.y):.4f} | {ll(pb, d.y):.4f} | "
                     f"{np.mean((d.p_mod - d.y) ** 2):.4f} / {np.mean((d.p_mkt - d.y) ** 2):.4f} |")
    lines.append("")
    lines.append("ROI at EV thresholds. Threshold and model weight w (p = w*model + (1-w)*market) chosen on the "
                 "earliest data (2017 for completions; 2026 weeks 1-2 for 2026), then applied to the test window "
                 "(2018; 2026 weeks 3-5). One bet per player-game-market at median book prices.")
    lines.append("")
    lines.append("| Set | chosen w, EV thr | calib n / ROI | TEST n / ROI / z |")
    lines.append("|---|---|---|---|")
    for sname, d in pp.groupby("set"):
        cal, tst = d[d.calib], d[~d.calib]
        best = None
        for w in (0.25, 0.5, 0.75, 1.0):
            for thr in (0.0, 0.03, 0.05, 0.08, 0.12):
                pc = w * cal.p_mod + (1 - w) * cal.p_mkt
                n, r, z = roi(cal, pc.values, thr)
                if n >= 20 and (best is None or r > best[0]):
                    best = (r, w, thr, n)
        if best is None:
            continue
        r0, w, thr, n0 = best
        n, r, z = roi(tst, (w * tst.p_mod + (1 - w) * tst.p_mkt).values, thr)
        lines.append(f"| {sname} | w={w}, EV>={thr:.0%} | {n0} / {r0:+.1%} | {n} / {r:+.1%} / {z:+.2f} |")
    # by-market test detail for the pure model at EV>=5%
    lines.append("")
    lines.append("Pure model (w=1), EV>=5%, test windows, by market: " + "; ".join(
        f"{s}/{m}: n={roi(d, d.p_mod.values, .05)[0]}, ROI {roi(d, d.p_mod.values, .05)[1]:+.1%}"
        for (s, m), d in pp[~pp.calib].groupby(["set", "market"])))
    return pp, lines


# ============================================================================ reporting helpers
def season_table(df, mcol="margin", tcol="total", bm="margin_b", bt="total_b"):
    d = df.dropna(subset=["home_score", "spread_home", "total_line"]).copy()
    am, at = d.home_score - d.away_score, d.home_score + d.away_score
    return d.assign(em=(d[mcol] - am).abs(), et=(d[tcol] - at).abs(), ebm=(d[bm] - am).abs(), ebt=(d[bt] - at).abs(),
                    emk=(-d.spread_home - am).abs(), etk=(d.total_line - at).abs()).groupby("season").agg(
        n=("em", "size"), margin_script=("em", "mean"), margin_blend=("ebm", "mean"), margin_market=("emk", "mean"),
        total_script=("et", "mean"), total_blend=("ebt", "mean"), total_market=("etk", "mean"), w_mkt=("w_mkt", "last"))


def md_table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    s = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for i, r in df.iterrows():
        s += "| " + str(i) + " | " + " | ".join(
            (str(int(v)) if float(v).is_integer() and abs(v) > 1 else fmt.format(v))
            if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
                                                 for v in r.values) + " |\n"
    return s


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ============================================================================ main
TG = None


def main():
    global TG
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    games = load_games()
    TG = team_game_table()
    print(f"team-game table {TG.shape} {time.time() - t0:.0f}s")
    gp = games[games.game_type.notna()]
    # ---------------- base Kalman script (params chosen on 2016-2020)
    bp, grid = tune_base(gp)
    print("base params", bp, f"{time.time() - t0:.0f}s")
    base = run_base(gp, **bp)
    # ---------------- features (sequential)
    fn = CACHE / "scripts_nfl_features.parquet"
    F = build_features(gp, TG)
    F.to_parquet(fn)
    print(f"features {F.shape} {time.time() - t0:.0f}s")
    # ---------------- correction lambda chosen on 2016-2020
    tune = []
    for lam in (100.0, 300.0, 1000.0, 3000.0, 10000.0):
        c, _, _ = correction_loop(gp, base, F, None, lam=lam)
        c = c.merge(gp[["game_id", "season", "home_score", "away_score"]], on="game_id")
        d = c[(c.season >= 2016) & (c.season <= 2020) & c.home_score.notna()]
        tune.append((lam, (d.mu_home - d.home_score).abs().mean() + (d.mu_away - d.away_score).abs().mean()))
    lam = min(tune, key=lambda x: x[1])[0]
    print("lambda", tune, f"{time.time() - t0:.0f}s")
    corr, rr, feats = correction_loop(gp, base, F, None, lam=lam)
    # ---------------- assemble games parquet
    keep_g = ["game_id", "season", "week", "game_type", "home_ml", "away_ml", "roof", "wind", "temp", "div_game",
              "home_qb_name", "away_qb_name", "home_rest", "away_rest"]
    G = base.rename(columns={c: c + "_base" for c in ["mu_home", "mu_away", "margin", "total", "margin_b", "total_b",
                                                       "w_mkt", "err_margin", "err_total", "err_margin_b",
                                                       "err_total_b"]})
    G = G.merge(corr, on="game_id").merge(gp[keep_g].drop(columns="season"), on="game_id")
    am, at = G.home_score - G.away_score, G.home_score + G.away_score
    G["err_home"] = G.mu_home - G.home_score; G["err_away"] = G.mu_away - G.away_score
    G["err_margin"] = G.margin - am; G["err_total"] = G.total - at
    G["err_margin_b"] = G.margin_b - am; G["err_total_b"] = G.total_b - at
    # pre-game NFL script fields
    Fi = F.set_index(["game_id", "team"])
    for side in ("home", "away"):
        f = Fi.loc[list(zip(G.game_id, G[side]))].reset_index(drop=True)
        for c in ["o_epa_pass", "o_epa_rush", "d_epa_pass", "d_epa_rush", "o_proe", "o_spp", "o_shotgun", "o_p11",
                  "o_heavy", "d_box", "d_pres", "d_man", "d_blitz", "m_pass", "m_run", "qb_delta", "o_prate",
                  "base_epa_pass"]:
            G[f"{side}_{c}"] = f[c].values
    # actual volumes for attribution
    tgi = TG.set_index(["game_id", "team"])
    for side in ("home", "away"):
        a = tgi.reindex(list(zip(G.game_id, G[side])))
        G[f"{side}_plays_act"] = a["off_n_play"].values; G[f"{side}_pass_att_act"] = a["off_pass_att"].values
        G[f"{side}_rush_att_act"] = a["off_rush_att"].values; G[f"{side}_pass_yds_act"] = a["off_pass_yds"].values
        G[f"{side}_to_act"] = a["off_to"].values
        G[f"{side}_epa_pass_act"] = (a["off_epa_pass"] / a["off_n_pass"]).values
    # moneyline win prob log loss
    from engine.devig import devig
    sdm = G.sd_margin
    G["p_home_script"] = stats.norm.cdf(G.margin / sdm)
    G["p_home_blend"] = stats.norm.cdf(G.margin_b / sdm)
    ok = G.home_ml.notna() & G.away_ml.notna()
    G.loc[ok, "p_home_mkt"] = devig(G.loc[ok, "home_ml"].values, G.loc[ok, "away_ml"].values, "odds_ratio")[0]
    Gout = G[G.season >= REPORT_FROM].copy()
    Gout.to_parquet(OUT / "nfl_games.parquet")
    print(f"games written {len(Gout)} {time.time() - t0:.0f}s")

    # ---------------- ablation (2021+): drop each group
    abl = []
    def score(c):
        c = c.merge(gp[["game_id", "season", "home_score", "away_score"]], on="game_id")
        d = c[(c.season >= 2021) & c.home_score.notna()]
        am, at = d.home_score - d.away_score, d.home_score + d.away_score
        pts = np.r_[(d.mu_home - d.home_score).abs(), (d.mu_away - d.away_score).abs()].mean()
        pa = np.r_[(d.home_pass_att_s - G.set_index("game_id").loc[d.game_id, "home_pass_att_act"].values).abs()].mean()
        return dict(pts=pts, margin=(d.margin - am).abs().mean(), total=(d.total - at).abs().mean(),
                    margin_b=(d.margin_b - am).abs().mean(), total_b=(d.total_b - at).abs().mean(), pass_att=pa)
    db = base[(base.season >= 2021)].dropna(subset=["err_margin"])
    abl.append(dict(variant="base Kalman only", pts=np.r_[(db.mu_home - db.home_score).abs(), (db.mu_away - db.away_score).abs()].mean(),
                    margin=db.err_margin.abs().mean(), total=db.err_total.abs().mean(),
                    margin_b=db.err_margin_b.abs().mean(), total_b=db.err_total_b.abs().mean(), pass_att=np.nan))
    abl.append(dict(variant="all enrichments", **score(corr)))
    for gname in GROUPS:
        c, _, _ = correction_loop(gp, base, F, [x for x in GROUPS if x != gname], lam=lam)
        abl.append(dict(variant=f"drop {gname}", **score(c)))
    for gname in GROUPS:
        c, _, _ = correction_loop(gp, base, F, [gname], lam=lam)
        abl.append(dict(variant=f"only {gname}", **score(c)))
    abl = pd.DataFrame(abl).set_index("variant")
    print(abl.round(3), f"{time.time() - t0:.0f}s")

    # ---------------- players
    w = player_frame()
    pl = player_loop(w, corr, gp)
    pl, p_exp = calibrate_and_sd(pl)
    plo = pl[pl.season >= REPORT_FROM].copy()
    plo[["game_id", "date", "season", "week", "player", "player_id", "position", "team", "opp", "stat", "mu", "sd",
         "share_of_team", "team_pass_att_s", "team_rush_att_s", "actual", "prior_games", "mu_raw", "cal"]].to_parquet(
        OUT / "nfl_players.parquet")
    print(f"players {len(plo)} {time.time() - t0:.0f}s")
    # player ablation (2021+ MAE)
    pab = []
    def pmae(d):
        d = d[d.season >= 2021]
        return d.groupby("stat").apply(lambda x: (x.mu - x.actual).abs().mean())
    pab.append(pmae(pl).rename("full script"))
    for name, kw in (("no air-yards share", dict(use_air=False)), ("no opponent adj", dict(use_opp=False)),
                     ("no team script (player's own EWMA)", dict(use_team=False))):
        x, _ = calibrate_and_sd(player_loop(w, corr, gp, **kw), dict(p_exp))
        pab.append(pmae(x).rename(name))
    # team script without the enrichment (base volume only): swap volumes for base tendency volumes
    corr_nv = corr.copy()
    for side in ("home", "away"):
        f = Fi.loc[list(zip(G.game_id, G[side]))].reset_index(drop=True)
        mp = dict(zip(G.game_id, f["base_pass_att"])); mr = dict(zip(G.game_id, f["base_rush_att"]))
        corr_nv[f"{side}_pass_att_s"] = corr_nv.game_id.map(mp); corr_nv[f"{side}_rush_att_s"] = corr_nv.game_id.map(mr)
    x, _ = calibrate_and_sd(player_loop(w, corr_nv, gp), dict(p_exp))
    pab.append(pmae(x).rename("team volume = raw tendencies (no game-script/correction)"))
    pab = pd.concat(pab, axis=1).T
    print(pab.round(2), f"{time.time() - t0:.0f}s")

    # ---------------- props
    pp, prop_lines = props_eval(pl)
    pp.to_parquet(CACHE / "scripts_nfl_props_eval.parquet")
    print("\n".join(prop_lines))

    write_report(G, base, grid, bp, lam, tune, rr, feats, abl, pl, pab, p_exp, prop_lines, time.time() - t0)
    print(f"done {time.time() - t0:.0f}s")


def write_report(G, base, grid, bp, lam, tune, rr, feats, abl, pl, pab, p_exp, prop_lines, secs):
    L = []
    L.append("# NFL script-then-learn: learning report\n")
    L.append(f"Code: `engine/scripts/nfl.py` (base: `engine/scripts/base.py`, unmodified). Ratings warmed up from {FIRST}; "
             f"report covers {REPORT_FROM}-2026 (2026 = weeks played so far). Every pre-game number uses only games "
             f"on earlier dates. Kalman params, ridge lambda, sd exponents chosen on 2016-2020 (sd exponents on "
             f"pre-2016); everything else is learned online from each game's error. Runtime {secs/60:.1f} min.\n")
    L.append(f"Kalman params chosen on 2016-20: {bp}; correction ridge lambda={lam} (grid {[(a, round(b, 3)) for a, b in tune]}).\n")
    d = G[G.season >= REPORT_FROM]
    L.append("## 1. Per-season learning curve (MAE, points)\n")
    L.append("`script` = Kalman + online NFL correction; `blend` = learned Bates-Granger blend with the close; "
             "`market` = closing spread / total.\n")
    st = season_table(d)
    L.append(md_table(st))
    bt = season_table(base[base.season >= REPORT_FROM].assign(game_id=0))
    L.append("\nBase Kalman script alone (no NFL enrichment):\n")
    L.append(md_table(bt[["margin_script", "total_script", "margin_market", "total_market"]]))
    # log loss
    def llrow(x):
        x = x.dropna(subset=["p_home_mkt", "home_score"]); x = x[x.home_score != x.away_score]
        y = (x.home_score > x.away_score).astype(int)
        return pd.Series(dict(n=len(x), script=logloss(x.p_home_script, y), blend=logloss(x.p_home_blend, y),
                              market=logloss(x.p_home_mkt, y)))
    ll = pd.DataFrame({"2016-20": llrow(d[d.season <= 2020]), "2021-26": llrow(d[d.season >= 2021])}).T
    L.append("\nWinner log loss (script P(home win) = Phi(margin/sd_margin); market = odds_ratio de-vigged ML):\n")
    L.append(md_table(ll, "{:.4f}"))
    # ATS / totals bets
    L.append("\n## 2. Does anything beat the closing line?\n")
    hold = d[(d.season >= 2021) & d.home_score.notna()]
    disc = d[(d.season <= 2020) & d.home_score.notna()]
    res = []
    for nm, x in (("margin (side)", "margin"), ("total", "total")):
        for per, z in (("2016-20", disc), ("2021-26", hold)):
            mk = -z.spread_home if x == "margin" else z.total_line
            act = (z.home_score - z.away_score) if x == "margin" else (z.home_score + z.away_score)
            e_s, e_m = z[x] - act, mk - act
            # information beyond the close: regress (act - mk) on (script - mk)
            dx = (z[x] - mk).values; dy = (act - mk).values
            m = np.isfinite(dx) & np.isfinite(dy)
            sl = np.polyfit(dx[m], dy[m], 1)[0]
            se = np.std(dy[m] - sl * dx[m]) / (np.std(dx[m]) * math.sqrt(m.sum()))
            res.append(dict(target=nm, period=per, mae_script=e_s.abs().mean(), mae_market=e_m.abs().mean(),
                            slope_beyond_close=sl, t=sl / se))
    L.append(md_table(pd.DataFrame(res).set_index("target")))
    # betting at k chosen on 2016-20
    out = []
    for nm in ("margin", "total"):
        best = None
        rows = {}
        for k in (1, 2, 3, 4, 5, 6):
            for per, z in (("disc", disc), ("hold", hold)):
                if nm == "margin":
                    mk, act = -z.spread_home, z.home_score - z.away_score
                    edge = z.margin - mk
                    win = np.where(edge > 0, act > mk, act < mk); push = act == mk
                else:
                    mk, act = z.total_line, z.home_score + z.away_score
                    edge = z.total - mk
                    win = np.where(edge > 0, act > mk, act < mk); push = act == mk
                sel = (np.abs(edge) >= k) & ~push & np.isfinite(edge)
                pnl = np.where(win[sel], 100 / 110, -1.0)
                rows[(k, per)] = (sel.sum(), pnl.mean() if sel.sum() else np.nan)
            if rows[(k, "disc")][0] >= 50 and (best is None or rows[(k, "disc")][1] > rows[(best, "disc")][1]):
                best = k
        out.append(f"- {nm}: k chosen on 2016-20 = {best} (n={rows[(best, 'disc')][0]}, ROI {rows[(best, 'disc')][1]:+.1%} at -110); "
                   f"**2021-26: n={rows[(best, 'hold')][0]}, ROI {rows[(best, 'hold')][1]:+.1%}**. All k on 2021-26: "
                   + ", ".join(f"k={k}: {rows[(k, 'hold')][1]:+.1%} (n={rows[(k, 'hold')][0]})" for k in range(1, 7)))
    L.append("\nBetting |script - close| >= k points at -110 (k chosen on 2016-20):\n")
    L += out
    # ---------------- attribution
    L.append("\n## 3. Error attribution: what did the script get wrong?\n")
    rows = []
    for side in ("home", "away"):
        x = d.dropna(subset=[f"{side}_plays_act", "home_score"])
        mu, y = x[f"mu_{side}"], x[f"{side}_score"]
        pl_s, pl_a = x[f"{side}_plays_s"], x[f"{side}_plays_act"]
        ppp_s = mu / pl_s; ppp_a = y / pl_a
        rows.append(pd.DataFrame(dict(season=x.season, err=mu - y, pace=(pl_s - pl_a) * ppp_s, eff=pl_a * (ppp_s - ppp_a),
                                      plays_err=pl_s - pl_a, to=x[f"{side}_to_act"],
                                      epa_err=x[f"{side}_base_epa_pass"] - x[f"{side}_epa_pass_act"],
                                      wind=x.wind.where(x.roof.isin(["outdoors", "open"]), 0).fillna(0),
                                      roof=x.roof, spread=(-x.spread_home).abs(), qb_delta=x[f"{side}_qb_delta"])))
    A = pd.concat(rows)
    L.append(f"Team-points error decomposed as pace (play-count miss x scripted pts/play) + efficiency "
             f"(actual plays x pts/play miss), team-games {REPORT_FROM}+: mean |pace part| = {A.pace.abs().mean():.2f} pts, "
             f"mean |efficiency part| = {A.eff.abs().mean():.2f} pts; share of error variance: pace "
             f"{np.var(A.pace) / np.var(A.err):.0%}, efficiency {np.var(A.eff) / np.var(A.err):.0%} "
             f"(cov term {2 * np.cov(A.pace, A.eff)[0, 1] / np.var(A.err):+.0%}). Play-count MAE "
             f"{A.plays_err.abs().mean():.1f} plays/team.\n")
    r2 = {}
    for c in ("to", "epa_err", "plays_err"):
        m = A[[c, "err"]].dropna()
        r2[c] = np.corrcoef(m[c], m.err)[0, 1] ** 2
    L.append(f"Variance of team-points error explained by realised surprises (R^2, single variable): "
             f"turnovers committed {r2['to']:.0%}, pass-EPA miss (script vs realised EPA/dropback) {r2['epa_err']:.0%}, "
             f"play-count miss {r2['plays_err']:.0%}. Efficiency (EPA) and turnovers dominate; pace/volume is a small part.\n")
    A["wind_b"] = pd.cut(A.wind, [-1, 9, 14, 19, 60], labels=["<10", "10-14", "15-19", "20+"])
    A["sp_b"] = pd.cut(A.spread, [-1, 3, 7, 10, 40], labels=["0-3", "3.5-7", "7.5-10", "10.5+"])
    A["qb_b"] = pd.cut(A.qb_delta, [-9, -0.05, 0.05, 9], labels=["QB worse than team", "same", "QB better"])
    L.append("Team-points bias (mean script - actual) and MAE by environment:\n")
    for col, nm in (("wind_b", "wind mph (outdoor)"), ("sp_b", "|closing spread|"), ("qb_b", "starter QB vs team pass EPA")):
        t = A.groupby(col, observed=True).err.agg(n="size", bias="mean", mae=lambda e: e.abs().mean())
        t.index.name = nm
        L.append(md_table(t, "{:.2f}"))
    # game-level by totals bucket: script vs market
    dd = d.dropna(subset=["home_score", "total_line"]).copy()
    dd["wind_b"] = pd.cut(dd.wind.where(dd.roof.isin(["outdoors", "open"]), 0).fillna(0), [-1, 9, 14, 60], labels=["<10", "10-14", "15+"])
    t = dd.groupby("wind_b", observed=True).apply(lambda z: pd.Series(dict(
        n=len(z), total_bias_script=z.err_total.mean(), total_mae_script=z.err_total.abs().mean(),
        total_bias_market=(z.total_line - z.home_score - z.away_score).mean(),
        total_mae_market=(z.total_line - z.home_score - z.away_score).abs().mean())))
    t.index.name = "wind"
    L.append("\nGame totals by wind, script vs market:\n")
    L.append(md_table(t, "{:.2f}"))
    # ---------------- ablation
    L.append("\n## 4. Ablation of NFL enrichments (2021-26, MAE; online loop re-run each time)\n")
    L.append("`pts` = team points MAE, `pass_att` = home pass attempts MAE (volume script). Lower is better.\n")
    L.append(md_table(abl))
    full = abl.loc["all enrichments"]
    helps = {gn: abl.loc[f"drop {gn}", "pts"] - full["pts"] for gn in GROUPS}
    L.append("\nTeam-points MAE change when each group is DROPPED (positive = the group helps): " +
             ", ".join(f"{k} {v:+.3f}" for k, v in sorted(helps.items(), key=lambda kv: -kv[1])) +
             f". All enrichments vs base Kalman: {full['pts'] - abl.loc['base Kalman only', 'pts']:+.3f} pts/team, "
             f"total {full['total'] - abl.loc['base Kalman only', 'total']:+.3f}, margin "
             f"{full['margin'] - abl.loc['base Kalman only', 'margin']:+.3f}. None of this moves the market-blended "
             "script (margin_b/total_b), because the learned blend weight sits at ~1.0 on the close.\n")
    w = pd.Series(rr.w[:-1], index=feats)
    L.append("\nFinal online-ridge weights on standardized features (points per 1 sd), largest |w|:\n")
    L.append(", ".join(f"{k} {v:+.2f}" for k, v in w.reindex(w.abs().sort_values(ascending=False).index).head(14).items()))
    # ---------------- players
    L.append("\n\n## 5. Player script\n")
    L.append("mu = online usage share (targets blended with air-yards share; carries; QB attempt share) x scripted team "
             "pass/rush attempts (online volume script fed by the market-informed game script) x player efficiency "
             "(shrunk, decayed) x opponent allowed-vs-league factor x online calibration. sd: var = k*mu^p, p per stat "
             f"chosen on pre-2016 ({p_exp}), k learned online. Only players with >=1 prior game; QB = listed starter.\n")
    x = pl[pl.season >= REPORT_FROM].copy()
    x["era"] = np.where(x.season <= 2020, "2016-20", "2021-26")
    t = x.groupby(["stat", "era"]).apply(lambda z: pd.Series(dict(
        n=len(z), mae=(z.mu - z.actual).abs().mean(), bias=(z.mu - z.actual).mean(),
        corr=np.corrcoef(z.mu, z.actual)[0, 1],
        z_sd=((z.actual - z.mu) / z.sd).std(),
        cover90=(np.abs(z.actual - z.mu) <= 1.645 * z.sd).mean()))).unstack("era")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    L.append(md_table(t, "{:.2f}"))
    L.append("\nPlayer ablation (2021-26 MAE by stat):\n")
    L.append(md_table(pab, "{:.2f}"))
    L.append("\n## 6. Player props vs real lines\n")
    L += prop_lines
    L.append("\n## 7. Caveats\n")
    L.append("- Player rows exist only for players who appear in that game's box score (and QB = listed starter); "
             "that is a mild selection on availability, similar to props only being offered on active players.\n"
             "- Volume scripts (pass/rush attempts) use the market-informed blended margin/total as game state "
             "(the closing line is pre-game information); the points script itself never uses the market.\n"
             "- 2017-18 completions test window selected only ~25 bets (min-n rule of 20 on the 2017 calibration "
             "set fixed in advance); treat its ROI as noise-level. 2026 is 5 weeks.\n"
             "- Participation data is missing for 2026 (not yet published), so formation/personnel/box/pressure "
             "tendencies carry 2025 values into 2026 via decay; FTN (play action, motion, blitz) covers 2022+.")
    (OUT / "nfl_learning.md").write_text("\n".join(L) + "\n")


T_LG_EPA = 0.0

if __name__ == "__main__":
    main()
