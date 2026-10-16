"""
NBA adapter for the script-then-learn engine (see SPEC.md).

    python -m engine.scripts.nba            -> results/scripts/nba_games.parquet, nba_players.parquet,
                                               nba_learning.md (+ nba_*.csv tables)

Layers (all strictly sequential; a game's script only uses games on EARLIER dates):
  1. CORE   engine.scripts.base.ScriptEngine Kalman team off/def ratings.  We drive it date-by-date: every game
            of date D is scripted before any game of date D is learned (stricter than base.run).
  2. ENRICH possession-based team tendencies from the player box (shifted EWMAs -> only earlier games):
            pace, ORtg/DRtg, 3PA rate (off/allowed), opp 3P% allowed, FT rate, OREB%, TOV%; schedule (rest, b2b,
            3-in-4, travel km, time-zone shift, road-trip length); injuries proxy (usual rotation players / starters
            who did NOT play the team's PREVIOUS game, from that game's box/DNP list); matchup products
            (pace clash, 3PA-heavy offence vs 3PA-allowing / poor-perimeter D, OREB vs OREB-allowed, TOV vs forced
            TOV, FT rate vs fouling D).
  3. LEARN  online RLS (ridge prior, exponential forgetting) on the core's per-team scoring error:
            pts_T - mu_core_T = x_T . beta, beta updated after every date by that date's errors.
  4. MARKET learned Bates-Granger blend of the enriched script with the closing line (separate for margin and
            total), and a second online RLS that tries to predict the CLOSING-LINE error from
            (script - market) + enrichment features ("does anything beat the close?").
  5. PLAYERS pts/reb/ast/3PM = online usage share x scripted team stat (points from the team script,
            reb/ast/3PM from scripted possessions x per-possession rates) x minutes factor (blowout risk from the
            script's margin distribution, ratios learned online by role tier), renormalised for teammates missing
            from the previous game, then an online linear calibration and variance model per stat.
Tuning (Kalman params, RLS forgetting/ridge, renormalisation exponent) uses 2016-2020 only; 2021+ is reported.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine.scripts.base import Params, ScriptEngine
from backtest.props_nba import proc_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "scripts"
WARM, START, TUNE_END, REPORT = 2010, 2016, 2020, 2021
HL = 10                                   # EWMA half-life (team games) of team tendencies
TEAM_FIX = {"NJ": "BKN"}

ARENA = {  # lat, lon, utc offset (standard time)
    "ATL": (33.757, -84.396, -5), "BOS": (42.366, -71.062, -5), "BKN": (40.683, -73.976, -5),
    "CHA": (35.225, -80.839, -5), "CHI": (41.881, -87.674, -6), "CLE": (41.496, -81.688, -5),
    "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.008, -7), "DET": (42.341, -83.055, -5),
    "GS": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "IND": (39.764, -86.155, -5),
    "LAC": (34.043, -118.267, -8), "LAL": (34.043, -118.267, -8), "MEM": (35.138, -90.051, -6),
    "MIA": (25.781, -80.187, -5), "MIL": (43.045, -87.917, -6), "MIN": (44.980, -93.276, -6),
    "NO": (29.949, -90.082, -6), "NY": (40.751, -73.993, -5), "OKC": (35.463, -97.515, -6),
    "ORL": (28.539, -81.384, -5), "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7),
    "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8), "SA": (29.427, -98.437, -6),
    "TOR": (43.643, -79.379, -5), "UTAH": (40.768, -111.901, -7), "WSH": (38.898, -77.021, -5)}

CORE_PARAMS = dict(init_level=100.0, init_hfa=3.0, init_sd=11.5, prior_var=16.0, q_game=0.08, carry=0.75,
                   season_var=0.0, lr_level=0.004)


# =============================================================== fast core (identical maths)
class FastEngine(ScriptEngine):
    """Same model as ScriptEngine; learn() keeps running window sums for the sd / rho update (identical maths)."""

    def __init__(self, p):
        super().__init__(p)
        self._buf = np.zeros((2000, 2)); self._n = 0
        self._s = np.zeros(5)                       # sum r0, r1, r0^2, r1^2, r0*r1 over the last <=2000

    def learn(self, h, a, season, y_h, y_a, neutral=False):
        H, A = self.teams[h], self.teams[a]
        x = np.r_[H.x, A.x]
        P = np.zeros((4, 4)); P[:2, :2] = H.P; P[2:, 2:] = A.P
        Hm = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], float)
        adv = 0.0 if neutral else self.hfa / 2
        pred = np.array([self.m + adv, self.m - adv]) + Hm @ x
        R = np.array([[1, self.rho], [self.rho, 1]]) * self.sd ** 2
        S = Hm @ P @ Hm.T + R
        K = P @ Hm.T @ np.linalg.inv(S)
        innov = np.array([y_h, y_a]) - pred
        x = x + K @ innov
        P = (np.eye(4) - K @ Hm) @ P
        H.x, A.x = x[:2], x[2:]
        H.P, A.P = P[:2, :2], P[2:, 2:]
        H.games += 1; A.games += 1
        self.m += self.p.lr_level * innov.mean()
        if not neutral:
            self.hfa += self.p.lr_level * (innov[0] - innov[1])
        i = self._n % 2000
        if self._n >= 2000:
            o = self._buf[i]
            self._s -= np.array([o[0], o[1], o[0] ** 2, o[1] ** 2, o[0] * o[1]])
        self._buf[i] = innov
        self._s += np.array([innov[0], innov[1], innov[0] ** 2, innov[1] ** 2, innov[0] * innov[1]])
        self._n += 1
        if self._n >= 200:
            k = min(self._n, 2000)
            s0, s1, s00, s11, s01 = self._s / k
            exc = max((s00 + s11) / 2 - np.mean(np.diag(Hm @ P @ Hm.T)) / 2, self.p.min_sd ** 2)
            self.sd = 0.98 * self.sd + 0.02 * math.sqrt(exc)
            v0, v1 = s00 - s0 ** 2, s11 - s1 ** 2
            if v0 > 0 and v1 > 0:
                c = (s01 - s0 * s1) / math.sqrt(v0 * v1)
                self.rho = 0.98 * self.rho + 0.02 * float(np.clip(c, -0.5, 0.8))
        return innov



# =============================================================== online ridge / RLS
class RLS:
    """Recursive least squares with ridge prior (precision `ridge`) and forgetting factor `lam` per update."""

    def __init__(self, k: int, lam: float = 0.9995, ridge: float = 10.0):
        self.w = np.zeros(k)
        self.P = np.eye(k) / ridge
        self.lam = lam

    def predict(self, X):
        return np.asarray(X) @ self.w

    def update(self, x, y):
        Px = self.P @ x
        g = Px / (self.lam + x @ Px)
        self.w = self.w + g * (y - x @ self.w)
        self.P = (self.P - np.outer(g, Px)) / self.lam


# =============================================================== data
def load_box() -> pd.DataFrame:
    use = ["date", "season", "season_type", "game_id", "team", "opp", "home", "player_id", "player", "starter",
           "did_not_play", "minutes", "pts", "reb", "oreb", "dreb", "ast", "fg3m", "fg3a", "fga", "fta", "tov",
           "team_score", "opp_score"]
    b = pd.read_csv(proc_path("nba_player_box"), low_memory=False, usecols=use)
    b = b[b.season >= WARM - 1]
    b["date"] = pd.to_datetime(b["date"])
    b["team"] = b.team.replace(TEAM_FIX)
    b["opp"] = b.opp.replace(TEAM_FIX)
    return b


def load_games(box: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data" / "processed" / "nba.csv", low_memory=False)
    g = g[g.season >= WARM].rename(columns={"espn_game_id": "game_id", "home": "home_name", "away": "away_name"})
    g["date"] = pd.to_datetime(g["date"])
    ab = box[["game_id", "team", "home"]].drop_duplicates(["game_id", "team"])
    hm = ab[ab.home == 1].drop_duplicates("game_id").set_index("game_id").team
    aw = ab[ab.home == 0].drop_duplicates("game_id").set_index("game_id").team
    g["home"], g["away"] = g.game_id.map(hm), g.game_id.map(aw)
    g = g.dropna(subset=["home", "away"])
    g = g[g.home.isin(ARENA) & g.away.isin(ARENA)]
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def team_games(box: pd.DataFrame, gids) -> pd.DataFrame:
    b = box[box.game_id.isin(gids)]
    agg = {c: (c, "sum") for c in ["minutes", "fga", "fta", "oreb", "dreb", "tov", "fg3a", "fg3m", "ast", "reb", "pts"]}
    t = b.groupby(["game_id", "team"]).agg(date=("date", "first"), season=("season", "first"), opp=("opp", "first"),
                                           home=("home", "first"), score=("team_score", "first"),
                                           opp_score=("opp_score", "first"), **agg).reset_index()
    raw = ["fga", "fta", "oreb", "dreb", "tov", "fg3a", "fg3m", "ast", "reb"]
    bad = (t.minutes < 235) | (t.fga <= 0) | ((t.pts - t.score).abs() > 2)
    t.loc[bad, raw] = np.nan
    t["tposs"] = t.fga + 0.44 * t.fta - t.oreb + t.tov
    o = t[["game_id", "team", "tposs"] + raw].rename(columns={c: f"o_{c}" for c in raw + ["tposs"]}).rename(
        columns={"team": "opp"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["poss"] = (t.tposs + t.o_tposs) / 2
    t["gmin"] = (t.minutes / 5).clip(lower=48).round()
    t["pace"] = t.poss * 48 / t.gmin
    t["ortg"] = 100 * t.score / t.poss
    t["drtg"] = 100 * t.opp_score / t.poss
    t["f3r"] = t.fg3a / t.fga
    t["f3r_d"] = t.o_fg3a / t.o_fga
    t["p3_d"] = t.o_fg3m / t.o_fg3a
    t["ftr"] = t.fta / t.fga
    t["ftr_d"] = t.o_fta / t.o_fga
    t["orb"] = t.oreb / (t.oreb + t.o_dreb)
    t["orb_d"] = t.o_oreb / (t.o_oreb + t.dreb)
    t["tov_r"] = t.tov / t.poss
    t["tov_d"] = t.o_tov / t.poss
    for s in ("reb", "ast", "fg3m"):
        t[f"{s}_pp"] = t[s] / t.poss                     # team stat per possession
        t[f"{s}_pp_d"] = t[f"o_{s}"] / t.poss            # opponents' stat per possession (allowed)
    return t.sort_values(["date", "game_id", "team"]).reset_index(drop=True)


TEND = ["pace", "ortg", "drtg", "f3r", "f3r_d", "p3_d", "ftr", "ftr_d", "orb", "orb_d", "tov_r", "tov_d",
        "reb_pp", "reb_pp_d", "ast_pp", "ast_pp_d", "fg3m_pp", "fg3m_pp_d"]


def gewm(df: pd.DataFrame, key: str, col: str, hl: float) -> pd.Series:
    """EWMA of `col` over each group's EARLIER rows only (shift(1) then EWM). df must be sorted by key, time."""
    sh = df.groupby(key, sort=False)[col].shift(1)
    e = sh.groupby(df[key], sort=False).ewm(halflife=hl, min_periods=1, ignore_na=True).mean()
    return e.droplevel(0).reindex(df.index)


def league_ref(t: pd.DataFrame, col: str, days: int = 60) -> pd.Series:
    """League mean of `col` over team-games in [D-days, D) (strictly before each row's date)."""
    d = t.groupby("date")[col].agg(["sum", "count"]).sort_index()
    full = d.reindex(pd.date_range(d.index.min(), d.index.max()), fill_value=0)
    cs = full.cumsum()
    prev, lag = cs.shift(1), cs.shift(days + 1)
    s = (prev["sum"] - lag["sum"].fillna(0)) / (prev["count"] - lag["count"].fillna(0)).replace(0, np.nan)
    s = s.fillna(prev["sum"] / prev["count"])
    return t["date"].map(s)


def tendencies(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    new = {f"{c}_e": gewm(t, "team", c, HL) for c in TEND}
    t = pd.concat([t, pd.DataFrame(new, index=t.index)], axis=1)
    for c in TEND:
        t[f"lg_{c}"] = league_ref(t, c)
        t[f"{c}_e"] = t[f"{c}_e"].fillna(t[f"lg_{c}"])
    return t


def schedule(g: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team): rest, b2b, 3-in-4, travel km & tz shift from the previous game venue, road-trip length."""
    rows = []
    bubble = (g.season == 2020) & (g.date >= "2020-07-01")
    for side, oth in (("home", "away"), ("away", "home")):
        v = g.home.where(~bubble, "ORL")
        rows.append(pd.DataFrame({"game_id": g.game_id, "team": g[side], "date": g.date, "venue": v,
                                  "is_home": int(side == "home")}))
    s = pd.concat(rows).sort_values(["team", "date"]).reset_index(drop=True)
    gt = s.groupby("team", sort=False)
    prev_date, prev_venue = gt.date.shift(1), gt.venue.shift(1)
    s["rest"] = (s.date - prev_date).dt.days.clip(upper=5).fillna(5)
    s["b2b"] = (s.rest == 1).astype(float)
    d3 = gt.date.shift(2)
    s["g3in4"] = ((s.date - d3).dt.days <= 3).astype(float)
    lat = s.venue.map(lambda x: ARENA[x][0]); lon = s.venue.map(lambda x: ARENA[x][1]); tz = s.venue.map(lambda x: ARENA[x][2])
    plat = prev_venue.map(lambda x: ARENA[x][0] if isinstance(x, str) else np.nan)
    plon = prev_venue.map(lambda x: ARENA[x][1] if isinstance(x, str) else np.nan)
    ptz = prev_venue.map(lambda x: ARENA[x][2] if isinstance(x, str) else np.nan)
    p1, p2 = np.radians(plat), np.radians(lat)
    dl = np.radians(lon - plon)
    km = 6371 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))
    new_season = (s.date - prev_date).dt.days > 60
    s["travel"] = np.where(new_season | prev_date.isna(), 0, km.fillna(0)) / 1000
    s["tz"] = np.where(new_season | prev_date.isna(), 0, (tz - ptz).fillna(0))    # + = travelled east
    # consecutive road games so far including this one (0 at home)
    road = 1 - s.is_home
    blk = (road != road.groupby(s.team).shift(1)).astype(int).groupby(s.team).cumsum()
    s["road_len"] = road * (road.groupby([s.team, blk]).cumcount() + 1)
    return s[["game_id", "team", "rest", "b2b", "g3in4", "travel", "tz", "road_len"]]


def injury_proxy(box: pd.DataFrame) -> pd.DataFrame:
    """For each (game_id, team): usual rotation players / starters (from the 5 games before the team's previous
    game) who did NOT play the previous game.  Uses only the previous game's box/DNP list -> pre-game info."""
    played = box[(box.did_not_play != 1) & (box.minutes.fillna(0) > 0)]
    per = defaultdict(dict)
    for gid, tm, pid, mn, st, pts, ts in zip(played.game_id.values, played.team.values, played.player_id.values,
                                              played.minutes.values, played.starter.values, played.pts.values,
                                              played.team_score.values):
        per[(gid, tm)][pid] = (mn, st, pts / max(ts, 1))
    tg = box[["game_id", "team", "date"]].drop_duplicates(["game_id", "team"]).sort_values(["team", "date", "game_id"])
    out = []
    for tm, x in tg.groupby("team", sort=False):
        seq = x.game_id.tolist()
        hist = [per.get((gid, tm), {}) for gid in seq]
        for k, gid in enumerate(seq):
            mm = ms = mp = 0.0
            if k >= 3:
                prev = hist[k - 1]
                win = hist[max(0, k - 6):k - 1]
                cnt, mins, sts, shr = defaultdict(int), defaultdict(float), defaultdict(int), defaultdict(float)
                for d in win:
                    for pid, (mn, st, sh) in d.items():
                        cnt[pid] += 1; mins[pid] += mn; sts[pid] += st; shr[pid] += sh
                for pid, c in cnt.items():
                    if c >= 3 and pid not in prev:
                        am = mins[pid] / c
                        if am >= 18:
                            mm += am; mp += shr[pid] / c
                        if sts[pid] >= 3:
                            ms += 1
            out.append((gid, tm, mm / 48, ms, mp))
    return pd.DataFrame(out, columns=["game_id", "team", "miss_min", "miss_st", "miss_pts"])


# =============================================================== features
GROUPS = {
    "base": ["bias", "is_home", "playoff"],
    "poss": ["d_poss"],
    "rest": ["rest_T", "rest_O", "b2b_T", "b2b_O", "g3in4_T", "g3in4_O"],
    "travel": ["travel_T", "travel_O", "tz_T", "tz_O", "road_T", "road_O"],
    "inj": ["miss_min_T", "miss_min_O", "miss_st_T", "miss_st_O", "miss_pts_T", "miss_pts_O"],
    "matchup": ["pace_clash", "m_3pa", "m_3pct", "m_orb", "m_tov", "m_ft"],
}
ALL_FEATS = [f for v in GROUPS.values() for f in v]


def side_frame(g: pd.DataFrame, T: pd.DataFrame, side: str) -> pd.DataFrame:
    """Rows from the perspective of team T (= g[side]) vs O."""
    oth = "away" if side == "home" else "home"
    a = T.add_suffix("_T").rename(columns={"game_id_T": "game_id", "team_T": side})
    b = T.add_suffix("_O").rename(columns={"game_id_O": "game_id", "team_O": oth})
    d = g[["game_id", "date", "season", "home", "away", "playoff"]].merge(a, on=["game_id", side], how="left") \
        .merge(b, on=["game_id", oth], how="left")
    d["side"] = side
    lg = lambda c: d[f"lg_{c}_T"]
    d["poss_pred"] = lg("pace") + (d.pace_e_T - lg("pace")) + (d.pace_e_O - lg("pace"))
    d["eff_pred"] = lg("ortg") + (d.ortg_e_T - lg("ortg")) + (d.drtg_e_O - lg("ortg"))
    d["pts_poss"] = d.poss_pred * d.eff_pred / 100
    dev = lambda c, s="T": d[f"{c}_e_{s}"] - lg(c)
    d["pace_clash"] = dev("pace") * dev("pace", "O")
    d["m_3pa"] = dev("f3r") * dev("f3r_d", "O")
    d["m_3pct"] = dev("f3r") * dev("p3_d", "O")
    d["m_orb"] = dev("orb") * dev("orb_d", "O")
    d["m_tov"] = dev("tov_r") * dev("tov_d", "O")
    d["m_ft"] = dev("ftr") * dev("ftr_d", "O")
    for c in ("rest", "b2b", "g3in4", "travel", "tz", "miss_min", "miss_st", "miss_pts"):
        d[f"{c}_T"], d[f"{c}_O"] = d[f"{c}_T"], d[f"{c}_O"]
    d["road_T"], d["road_O"] = d.road_len_T, d.road_len_O
    d["is_home"] = float(side == "home")
    d["bias"] = 1.0
    return d


def build_features(box: pd.DataFrame, g: pd.DataFrame):
    t0 = time.time()
    TG = team_games(box, g.game_id)
    T = tendencies(TG)
    S = schedule(g)
    I = injury_proxy(box[box.game_id.isin(g.game_id)])
    keep = ["game_id", "team"] + [f"{c}_e" for c in TEND] + [f"lg_{c}" for c in TEND]
    T2 = T[keep].merge(S, on=["game_id", "team"], how="right").merge(I, on=["game_id", "team"], how="left")
    H, A = side_frame(g, T2, "home"), side_frame(g, T2, "away")
    print(f"  features built ({time.time() - t0:.0f}s)")
    return TG, T, H, A


# =============================================================== core (date-batched)
def run_core(g: pd.DataFrame, **kw) -> pd.DataFrame:
    p = dict(CORE_PARAMS); p.update(kw)
    eng = FastEngine(Params(**p))
    out = []
    for _, blk in g.groupby("date", sort=True):
        scripts = []
        for r in blk.itertuples(index=False):
            neutral = bool(r.neutral)
            sc = eng.script(r.home, r.away, int(r.season), neutral)
            mm = -r.spread_home if r.spread_home == r.spread_home else np.nan
            b = eng.blend(sc, mm, r.total)
            scripts.append((r, sc, mm, neutral))
            out.append(dict(game_id=r.game_id, **sc, margin_b_core=b["margin_b"], total_b_core=b["total_b"],
                            w_mkt_core=b["w_mkt"], hfa=eng.hfa, level=eng.m, noise_sd=eng.sd, score_rho=eng.rho,
                            games_home=eng.teams[r.home].games, games_away=eng.teams[r.away].games))
        for r, sc, mm, neutral in scripts:
            am, at = r.home_score - r.away_score, r.home_score + r.away_score
            eng.learn_blend(sc, mm, r.total, am, at)
            eng.learn(r.home, r.away, int(r.season), float(r.home_score), float(r.away_score), neutral)
    c = pd.DataFrame(out).rename(columns={"margin": "margin_core", "total": "total_core", "mu_home": "mu_home_core",
                                           "mu_away": "mu_away_core"})
    return c


_G = None


def _core_job(kw):
    return run_core(_G, **kw)


def parallel_core(g, grid, workers=4):
    """Run the core for several parameter sets in parallel (fork shares the game frame)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    global _G
    _G = g
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_core_job, grid))


# =============================================================== correction + market layers
def standardizer(H, A, g, feats):
    ref = pd.concat([H, A])
    ref = ref[ref.season <= 2012]
    mu = ref[feats].mean(); sd = ref[feats].std().replace(0, 1).fillna(1)
    for f in ("bias", "is_home", "playoff"):
        if f in feats:
            mu[f], sd[f] = 0.0, 1.0
    return mu, sd


def run_layers(g, C, H, A, groups=tuple(GROUPS), lam=0.9995, ridge=10.0, bg_window=1500, mkt_lam=0.9995,
               mkt_ridge=50.0, return_coefs=False):
    """Online correction on the core + learned market blend + market-residual learner. Returns per-game frame."""
    feats = [f for k in groups for f in GROUPS[k]]
    d = g[["game_id", "date", "season", "home_score", "away_score", "spread_home", "total"]].merge(C, on="game_id")
    Hx, Ax = H.set_index("game_id").loc[d.game_id], A.set_index("game_id").loc[d.game_id]
    hfa2 = d.hfa.values / 2
    Hx = Hx.assign(d_poss=Hx.pts_poss.values - (d.mu_home_core.values - hfa2))
    Ax = Ax.assign(d_poss=Ax.pts_poss.values - (d.mu_away_core.values + hfa2))
    mu, sd = standardizer(Hx.reset_index(), Ax.reset_index(), g, feats) if feats else (None, None)
    XH = ((Hx[feats] - mu) / sd).fillna(0).values if feats else np.zeros((len(d), 0))
    XA = ((Ax[feats] - mu) / sd).fillna(0).values if feats else np.zeros((len(d), 0))
    yH = d.home_score.values - d.mu_home_core.values
    yA = d.away_score.values - d.mu_away_core.values
    mkt_m = -d.spread_home.values
    mkt_t = d.total.values
    am = (d.home_score - d.away_score).values
    at = (d.home_score + d.away_score).values
    # market-residual feature sets (margin: differences; total: sums)
    mfe = ["rest", "b2b", "travel", "miss_min", "miss_st"]
    DM = np.c_[[(Hx[f + "_T"].values - Ax[f + "_T"].values) for f in mfe]].T
    DT = np.c_[[(Hx[f + "_T"].values + Ax[f + "_T"].values) for f in mfe] + [Hx.pace_clash.values]].T
    DM = np.nan_to_num((DM - np.nanmean(DM[:2000], 0)) / (np.nanstd(DM[:2000], 0) + 1e-9))
    DT = np.nan_to_num((DT - np.nanmean(DT[:2000], 0)) / (np.nanstd(DT[:2000], 0) + 1e-9))
    k = len(feats)
    rls = RLS(k, lam, ridge) if k else None
    rm = RLS(2 + DM.shape[1], mkt_lam, mkt_ridge)
    rt = RLS(2 + DT.shape[1], mkt_lam, mkt_ridge)
    n = len(d)
    corr_h, corr_a = np.zeros(n), np.zeros(n)
    w_m, w_t = np.full(n, np.nan), np.full(n, np.nan)
    pm, pt = np.full(n, np.nan), np.full(n, np.nan)
    eo_m, ek_m, eo_t, ek_t = [], [], [], []
    wm_cur = wt_cur = 0.5
    dates = d.date.values
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, n]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        idx = np.arange(s0, s1)
        if k:
            corr_h[idx] = XH[idx] @ rls.w
            corr_a[idx] = XA[idx] @ rls.w
        mh, ma = d.mu_home_core.values[idx] + corr_h[idx], d.mu_away_core.values[idx] + corr_a[idx]
        sm, st = mh - ma, mh + ma
        w_m[idx], w_t[idx] = wm_cur, wt_cur
        zm = np.c_[np.ones(len(idx)), (sm - mkt_m[idx]) / 5, DM[idx]]
        zt = np.c_[np.ones(len(idx)), (st - mkt_t[idx]) / 8, DT[idx]]
        pm[idx], pt[idx] = mkt_m[idx] + zm @ rm.w, mkt_t[idx] + zt @ rt.w
        # ---- reveal & learn
        for j, i in enumerate(idx):
            if k:
                rls.update(XH[i], yH[i]); rls.update(XA[i], yA[i])
            if np.isfinite(mkt_m[i]) and np.isfinite(mkt_t[i]):
                eo_m.append(sm[j] - am[i]); ek_m.append(mkt_m[i] - am[i])
                eo_t.append(st[j] - at[i]); ek_t.append(mkt_t[i] - at[i])
                rm.update(zm[j], am[i] - mkt_m[i]); rt.update(zt[j], at[i] - mkt_t[i])
        if len(eo_m) >= 100:
            for eo, ek, nm in ((eo_m, ek_m, "m"), (eo_t, ek_t, "t")):
                if len(eo) > bg_window:
                    del eo[:-bg_window]; del ek[:-bg_window]
                a, b = np.array(eo), np.array(ek)
                den = np.mean((a - b) ** 2)
                w = float(np.clip(np.mean((a - b) * a) / den, 0, 1)) if den > 1e-9 else 0.5
                if nm == "m":
                    wm_cur = w
                else:
                    wt_cur = w
    o = d[["game_id"]].copy()
    o["corr_home"], o["corr_away"] = corr_h, corr_a
    o["mu_home"] = d.mu_home_core.values + corr_h
    o["mu_away"] = d.mu_away_core.values + corr_a
    o["margin"], o["total"] = o.mu_home - o.mu_away, o.mu_home + o.mu_away
    o["w_mkt_margin"], o["w_mkt_total"] = w_m, w_t
    o["w_mkt"] = (w_m + w_t) / 2
    o["margin_b"] = np.where(np.isfinite(mkt_m), w_m * mkt_m + (1 - w_m) * o.margin, o.margin)
    o["total_b"] = np.where(np.isfinite(mkt_t), w_t * mkt_t + (1 - w_t) * o.total, o.total)
    o["margin_mx"], o["total_mx"] = pm, pt          # market + learned market-residual correction
    if return_coefs:
        return o, dict(zip(feats, rls.w)) if k else {}
    return o


def mae(x):
    return float(np.nanmean(np.abs(x)))


def score(df, g, seasons):
    m = g.set_index("game_id").loc[df.game_id]
    sel = m.season.isin(seasons).values & np.isfinite(m.spread_home.values)
    am = (m.home_score - m.away_score).values[sel]
    at = (m.home_score + m.away_score).values[sel]
    return dict(mae_margin=mae(df.margin.values[sel] - am), mae_total=mae(df.total.values[sel] - at),
                mae_margin_b=mae(df.margin_b.values[sel] - am), mae_total_b=mae(df.total_b.values[sel] - at),
                mae_margin_mx=mae(df.margin_mx.values[sel] - am), mae_total_mx=mae(df.total_mx.values[sel] - at),
                mae_margin_mkt=mae(-m.spread_home.values[sel] - am), mae_total_mkt=mae(m.total.values[sel] - at),
                n=int(sel.sum()))


# =============================================================== players
STATS = {"pts": "score", "reb": "reb", "ast": "ast", "fg3m": "fg3m"}


def player_frame(box, g, games_out, T, H, A):
    """Pre-game player features coupled to the team script (expensive part, computed once)."""
    t0 = time.time()
    pl = box[(box.did_not_play != 1) & (box.minutes.fillna(0) > 0) & box.game_id.isin(g.game_id)].copy()
    tt = T[["game_id", "team", "score", "reb", "ast", "fg3m"]].rename(
        columns={"reb": "t_reb", "ast": "t_ast", "fg3m": "t_fg3m", "score": "t_pts"})
    pl = pl.merge(tt, on=["game_id", "team"], how="left")
    pl = pl.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    a = 1 - 0.5 ** (1 / 8)
    for s in STATS:
        pl[f"sh_{s}"] = pl[s] / pl[f"t_{s}"].replace(0, np.nan)
        pl[f"sh_{s}_e"] = gewm(pl, "player_id", f"sh_{s}", 8)
        pl[f"sh_{s}_es"] = gewm(pl, "player_id", f"sh_{s}", 3)      # short memory
        pl[f"sh_{s}_el"] = gewm(pl, "player_id", f"sh_{s}", 25)     # long memory
        # share EWMA right AFTER this game (what a later game's renormalisation may use)
        pl[f"post_{s}"] = ((1 - a) * pl[f"sh_{s}_e"] + a * pl[f"sh_{s}"]).fillna(pl[f"sh_{s}"])
    pl["min_e"] = gewm(pl, "player_id", "minutes", 8)
    pl["st_r5"] = pl.groupby("player_id", sort=False).starter.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    pl["gp_n"] = pl.groupby("player_id", sort=False).cumcount()
    # renormalisation: summed post-game shares of the players who played the team's PREVIOUS game
    tgo = pl[["game_id", "team", "date"]].drop_duplicates(["game_id", "team"]).sort_values(["team", "date", "game_id"])
    tgo["prev_gid"] = tgo.groupby("team").game_id.shift(1)
    ssum = pl.groupby(["game_id", "team"])[[f"post_{s}" for s in STATS]].sum().reset_index() \
        .rename(columns={"game_id": "prev_gid"}).rename(columns={f"post_{s}": f"avail_{s}" for s in STATS})
    tgo = tgo.merge(ssum, on=["prev_gid", "team"], how="left")
    pl = pl.merge(tgo[["game_id", "team"] + [f"avail_{s}" for s in STATS]], on=["game_id", "team"], how="left")
    pl = pl[pl.gp_n >= 3].copy()
    # team script per side
    gm = games_out.set_index("game_id")
    side = []
    for sd in ("home", "away"):
        X = H if sd == "home" else A
        x = X[["game_id", sd, "poss_pred"] + [f"{s}_pp_e_T" for s in ("reb", "ast", "fg3m")] +
              [f"{s}_pp_d_e_O" for s in ("reb", "ast", "fg3m")] + [f"lg_{s}_pp_T" for s in ("reb", "ast", "fg3m")]] \
            .rename(columns={sd: "team"})
        x = x[x.game_id.isin(gm.index)]
        m = gm.loc[x.game_id]
        sgn = 1 if sd == "home" else -1
        x["tpts_b"] = ((m.total_b + sgn * m.margin_b) / 2).values
        x["tpts_s"] = (m.mu_home if sd == "home" else m.mu_away).values
        mb, ms_, sdm = sgn * m.margin_b.values, sgn * m.margin.values, m.sd_margin.values
        x["blow_p"] = norm.cdf((-15 - mb) / sdm) + 1 - norm.cdf((15 - mb) / sdm)
        x["blow_p_s"] = norm.cdf((-15 - ms_) / sdm) + 1 - norm.cdf((15 - ms_) / sdm)
        for s in ("reb", "ast", "fg3m"):
            x[f"tstat_{s}"] = x.poss_pred * (x[f"{s}_pp_e_T"] + x[f"{s}_pp_d_e_O"] - x[f"lg_{s}_pp_T"])
        side.append(x[["game_id", "team", "tpts_b", "tpts_s", "blow_p", "blow_p_s"] + [f"tstat_{s}" for s in ("reb", "ast", "fg3m")]])
    pl = pl.merge(pd.concat(side), on=["game_id", "team"], how="left")
    # ---- minutes model: tier blowout ratios learned online (strictly earlier dates)
    pl["tier"] = np.where(pl.st_r5 >= 0.5, "S", "B") + pd.cut(pl.min_e, [-1, 18, 28, 34, 60],
                                                              labels=["<18", "18-28", "28-34", "34+"]).astype(str)
    pl["blow"] = ((pl.team_score - pl.opp_score).abs() >= 15).astype(float)
    pl["mratio"] = (pl.minutes / pl.min_e).clip(0, 3)
    agg = pl.assign(rb=pl.mratio * pl.blow, rc=pl.mratio * (1 - pl.blow), nb=pl.blow, nc=1 - pl.blow) \
        .groupby(["tier", "date"])[["rb", "rc", "nb", "nc"]].sum().sort_index()
    cum = (agg.groupby(level=0).cumsum() - agg).reset_index()
    pl = pl.merge(cum, on=["tier", "date"], how="left")
    pr = 20
    pl["r_blow"] = (pl.rb + pr * 0.9) / (pl.nb + pr)
    pl["r_close"] = (pl.rc + pr * 1.0) / (pl.nc + pr)
    pl["base_bp"] = (pl.nb + 4) / (pl.nb + pl.nc + 20)
    for v, bpc in (("", "blow_p"), ("_s", "blow_p_s")):
        bp = pl[bpc].fillna(pl.base_bp)
        pl[f"min_adj{v}"] = (bp * pl.r_blow + (1 - bp) * pl.r_close) / (pl.base_bp * pl.r_blow + (1 - pl.base_bp) * pl.r_close)
    pl["min_exp"] = pl.min_e * pl.min_adj
    print(f"  player frame {len(pl)} rows ({time.time() - t0:.0f}s)")
    return pl


def player_project(pl, alpha=0.5, decay=0.997):
    """mu = share x renorm x scripted team stat x minutes factor, then online calibration + variance per stat."""
    tstat = {"pts": pl.tpts_b, "reb": pl.tstat_reb, "ast": pl.tstat_ast, "fg3m": pl.tstat_fg3m}
    rows = []
    for s in STATS:
        renorm = (1 / pl[f"avail_{s}"].clip(0.6, 1.4)).fillna(1) ** alpha
        team_s = tstat[s]
        share = pl[f"sh_{s}_e"].fillna(0) * renorm
        raw = share * team_s * pl.min_adj
        raw_nm = share * (pl.tpts_s if s == "pts" else team_s) * pl.min_adj_s
        k_s = (pl[f"sh_{s}_es"].fillna(pl[f"sh_{s}_e"]) * renorm * pl.min_adj).fillna(0)
        k_l = (pl[f"sh_{s}_el"].fillna(pl[f"sh_{s}_e"]) * renorm * pl.min_adj).fillna(0)
        rows.append(pd.DataFrame({"game_id": pl.game_id, "date": pl.date, "season": pl.season, "player_id": pl.player_id,
                                  "player": pl.player, "team": pl.team, "opp": pl.opp, "stat": s, "mu_raw": raw,
                                  "mu_raw_nm": raw_nm, "share_of_team": share, "k_s": k_s, "k_l": k_l,
                                  "team_s": team_s, "team_s_nm": (pl.tpts_s if s == "pts" else team_s) * pl.min_adj_s / pl.min_adj, "team_stat_script": team_s,
                                  "min_e": pl.min_e, "min_exp": pl.min_exp, "minutes": pl.minutes,
                                  "blowout_p": pl.blow_p, "tier": pl.tier, "actual": pl[s],
                                  "team_actual": pl[f"t_{s}"]}))
    P = pd.concat(rows, ignore_index=True)
    P = P[np.isfinite(P.mu_raw) & np.isfinite(P.mu_raw_nm) & P.actual.notna()].sort_values(["stat", "date"]).reset_index(drop=True)
    out = []
    for s, x in P.groupby("stat", sort=False):
        x = x.copy()
        y = x.actual.values
        dts = x.date.values
        st = np.r_[0, np.flatnonzero(dts[1:] != dts[:-1]) + 1, len(x)]
        for v in ("", "_nm"):
            tsv = x[f"team_s{v}"].values
            # online calibration on [1, share8, share3, share25] x scripted team stat x minutes factor
            X = np.nan_to_num(np.c_[np.ones(len(x)), x[f"mu_raw{v}"].values, x.k_s.values * tsv, x.k_l.values * tsv])
            A_ = np.eye(4); b_ = np.array([0.0, 1.0, 0.0, 0.0])
            Av = np.eye(2); bv = np.array([1.0, 1.0])
            mu = np.empty(len(x)); var = np.empty(len(x))
            for a0, a1 in zip(st[:-1], st[1:]):
                Xi = X[a0:a1]
                w = np.linalg.solve(A_, b_); wv = np.linalg.solve(Av, bv)
                mu[a0:a1] = np.maximum(Xi @ w, 0.05)
                Zi = np.c_[np.ones(a1 - a0), mu[a0:a1]]
                var[a0:a1] = np.maximum(Zi @ wv, 0.05)
                r2 = (y[a0:a1] - mu[a0:a1]) ** 2
                A_ = decay * A_ + Xi.T @ Xi; b_ = decay * b_ + Xi.T @ y[a0:a1]
                Av = decay * Av + Zi.T @ Zi; bv = decay * bv + Zi.T @ r2
            x[f"mu{v}"], x[f"sd{v}"] = mu, np.sqrt(var)
        out.append(x)
    return pd.concat(out, ignore_index=True)


# =============================================================== props evaluation
def props_eval(P: pd.DataFrame):
    from engine.devig import devig
    from backtest.props_nba import pick_bets
    from backtest.core import perf
    p = pd.read_csv(ROOT / "results" / "nba_props_frame.csv.gz", low_memory=False)
    p["date"] = pd.to_datetime(p["date"])
    pts = P[P.stat == "pts"][["date", "player_id", "mu", "sd", "mu_nm", "sd_nm", "blowout_p", "min_exp", "min_e"]] \
        .rename(columns={"player_id": "pid"}).drop_duplicates(["date", "pid"])
    pts["pid"] = pts.pid.astype("int64"); p["pid"] = p.pid.astype("int64")
    p = p.merge(pts, on=["date", "pid"], how="left")
    cov = p.mu.notna().mean()
    p["p_dv"] = devig(p.over_odds.values, p.under_odds.values, "odds_ratio")[0]
    p["p_script"] = pd.Series(1 - norm.cdf((p.line - p.mu) / p.sd), index=p.index).fillna(p.p_mod)
    p["p_script_nm"] = pd.Series(1 - norm.cdf((p.line - p.mu_nm) / p.sd_nm), index=p.index).fillna(p.p_mod)
    p["p_blend"] = 0.75 * p.p_mkt + 0.25 * p.p_mod
    # choose blend weight on 2025 main-line log loss
    d = p[(p.season == 2025) & (p.alt_line == 0)]
    ll = lambda q, y: float(-np.mean(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4))))
    grid = []
    for a in np.arange(0, 0.55, 0.05):
        for b in np.arange(0, 0.55 - a, 0.05):
            q = (1 - a - b) * d.p_mkt + a * d.p_mod + b * d.p_script
            grid.append((a, b, ll(q, d.y)))
    a, b, _ = min(grid, key=lambda r: r[2])
    p["p_mkt_script"] = 0.75 * p.p_mkt + 0.25 * p.p_script
    p["p_sel"] = (1 - a - b) * p.p_mkt + a * p.p_mod + b * p.p_script
    probs = ["p_dv", "p_mkt", "p_mod", "p_script", "p_script_nm", "p_blend", "p_mkt_script", "p_sel"]
    LL = []
    for (s, main), x in p.groupby([p.season, p.alt_line == 0]):
        for c in probs:
            LL.append(dict(season=s, main_only=main, prob=c, n=len(x), logloss=ll(x[c], x.y),
                           brier=float(np.mean((x[c] - x.y) ** 2))))
    LL = pd.DataFrame(LL)
    # ROI: thresholds chosen on 2025 (n>=300), reported on 2026
    R = []
    for c in probs[1:]:
        for thr in (0.02, 0.04, 0.06, 0.08, 0.10):
            B = pick_bets(p, c, thr)
            for s in (2025, 2026):
                x = B[B.season == s]
                r = perf(x.pnl) if len(x) else {}
                R.append(dict(prob=c, thr=thr, season=s, n=len(x), roi=r.get("roi", np.nan), z=r.get("z", np.nan),
                              roi_over=x[x.side == "over"].pnl.mean(), roi_under=x[x.side == "under"].pnl.mean()))
    R = pd.DataFrame(R)
    ch = []
    for c in probs[1:]:
        r25 = R[(R.prob == c) & (R.season == 2025) & (R.n >= 300)]
        if len(r25):
            t = r25.sort_values("roi", ascending=False).iloc[0].thr
            r26 = R[(R.prob == c) & (R.season == 2026) & (R.thr == t)].iloc[0]
            ch.append(dict(prob=c, thr=t, roi_2025=r25[r25.thr == t].roi.iloc[0], n_2025=int(r25[r25.thr == t].n.iloc[0]),
                           roi_2026=r26.roi, n_2026=int(r26.n), z_2026=r26.z, under_2026=r26.roi_under,
                           over_2026=r26.roi_over))
    # MAE of point projections vs existing model
    q = p[p.alt_line == 0].drop_duplicates(["date", "pid"])
    m = dict(mae_script=mae(q.mu - q.actual), mae_script_nm=mae(q.mu_nm - q.actual), mae_props_model=mae(q.proj - q.actual),
             mae_mkt=mae(q.mu_mkt - q.actual), n=int(q.mu.notna().sum()))
    return dict(LL=LL, ROI=R, chosen=pd.DataFrame(ch), blend_w=(a, b), coverage=cov, mae=m)


# =============================================================== reporting helpers
def ml_logloss(df):
    from engine.devig import devig
    x = df.dropna(subset=["home_ml", "away_ml"])
    x = x[(x.home_ml.abs() >= 100) & (x.away_ml.abs() >= 100) & (x.home_score != x.away_score)]
    y = (x.home_score > x.away_score).astype(float).values
    pm = devig(x.home_ml.values, x.away_ml.values, "odds_ratio")[0]
    ps = norm.cdf(x.margin / x.sd_margin)
    pb = norm.cdf(x.margin_b / x.sd_margin)
    f = lambda q: -np.mean(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4)))
    return x.assign(_y=y, _pm=pm, _ps=ps, _pb=pb), f


def md_table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                         else str(v) for v in r) + " |")
    return "\n".join(lines)


def ats_roi(pred_margin, mkt_margin, act_margin, thr):
    """Bet the side the prediction favours vs the closing spread at -110 when |edge| >= thr."""
    e = pred_margin - mkt_margin
    sel = np.abs(e) >= thr
    res = np.sign(e[sel]) * (act_margin[sel] - mkt_margin[sel])
    win, push = res > 0, res == 0
    pnl = np.where(push, 0, np.where(win, 100 / 110, -1.0))
    return len(pnl), float(pnl.mean()) if len(pnl) else np.nan


# =============================================================== main
def main():
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    box = load_box()
    g = load_games(box)
    print(f"games {len(g)}  box {len(box)} ({time.time() - T0:.0f}s)")
    TG, T, H, A = build_features(box, g)
    tune_seasons = range(START, TUNE_END + 1)
    rep_seasons = range(REPORT, 2027)

    # ---- tune core Kalman on 2016-2020 (margin + total MAE of the core script)
    best, tune_rows = None, []
    grid = [dict(q_game=qg, prior_var=pv, carry=ca) for qg in (0.03, 0.08, 0.2) for pv in (10.0, 25.0) for ca in (0.6, 0.8)]
    for kw, C in zip(grid, parallel_core(g, grid)):
        m = g[["game_id", "season", "home_score", "away_score"]].merge(C, on="game_id")
        m = m[m.season.isin(tune_seasons)]
        sc = mae(m.margin_core - (m.home_score - m.away_score)) + mae(m.total_core - (m.home_score + m.away_score))
        tune_rows.append(dict(**kw, mae_sum_2016_20=sc))
        if best is None or sc < best[0]:
            best = (sc, kw, C)
    core_kw, C = best[1], best[2]
    print(f"core tuned {core_kw} ({time.time() - T0:.0f}s)")

    # ---- tune RLS forgetting / ridge on 2016-2020
    rls_rows, bestr = [], None
    for lam in (0.999, 0.9997, 0.99995):
        for ridge in (3.0, 30.0, 300.0):
            o = run_layers(g, C, H, A, lam=lam, ridge=ridge)
            sc = score(o, g, tune_seasons)
            v = sc["mae_margin"] + sc["mae_total"]
            rls_rows.append(dict(lam=lam, ridge=ridge, mae_sum_2016_20=v))
            if bestr is None or v < bestr[0]:
                bestr = (v, lam, ridge)
    _, LAM, RIDGE = bestr
    print(f"rls tuned lam={LAM} ridge={RIDGE} ({time.time() - T0:.0f}s)")
    O, coefs = run_layers(g, C, H, A, lam=LAM, ridge=RIDGE, return_coefs=True)

    # ---- ablation (report 2021+, also 2016-20)
    abl = []
    configs = [("core only (no enrichment)", ())] + [("full", tuple(GROUPS))] + \
              [(f"drop {k}", tuple(x for x in GROUPS if x != k)) for k in GROUPS if k != "base"] + \
              [(f"base + {k} only", ("base", k)) for k in GROUPS if k != "base"]
    for name, grp in configs:
        o = run_layers(g, C, H, A, groups=grp, lam=LAM, ridge=RIDGE)
        if not grp:
            o["margin"], o["total"] = C.set_index("game_id").loc[o.game_id, "margin_core"].values, \
                C.set_index("game_id").loc[o.game_id, "total_core"].values
        for per, ss in (("2016-20", tune_seasons), ("2021+", rep_seasons)):
            sc = score(o, g, ss)
            abl.append(dict(config=name, period=per, mae_margin=sc["mae_margin"], mae_total=sc["mae_total"],
                            mae_margin_mkt=sc["mae_margin_mkt"], mae_total_mkt=sc["mae_total_mkt"], n=sc["n"]))
    ABL = pd.DataFrame(abl)
    print(ABL.round(3).to_string())

    # ---- assemble game output
    gm = g.rename(columns={"total": "total_line"}).merge(C, on="game_id").merge(O, on="game_id")
    Hs = H.set_index("game_id").loc[gm.game_id]; As = A.set_index("game_id").loc[gm.game_id]
    for c, src in (("poss_pred", "poss_pred"), ("pace_home_e", "pace_e_T"), ("ortg_home_e", "ortg_e_T"),
                   ("drtg_home_e", "drtg_e_T"), ("f3r_home_e", "f3r_e_T"), ("ftr_home_e", "ftr_e_T"),
                   ("orb_home_e", "orb_e_T"), ("tov_home_e", "tov_r_e_T"), ("pts_poss_home", "pts_poss"),
                   ("rest_home", "rest_T"), ("b2b_home", "b2b_T"), ("travel_home_kkm", "travel_T"),
                   ("tz_home", "tz_T"), ("miss_min_home", "miss_min_T"), ("miss_st_home", "miss_st_T"),
                   ("pace_clash", "pace_clash"), ("m_3pa_home", "m_3pa"), ("m_3pct_home", "m_3pct")):
        gm[c] = Hs[src].values
    for c, src in (("pace_away_e", "pace_e_T"), ("ortg_away_e", "ortg_e_T"), ("drtg_away_e", "drtg_e_T"),
                   ("f3r_away_e", "f3r_e_T"), ("pts_poss_away", "pts_poss"), ("rest_away", "rest_T"),
                   ("b2b_away", "b2b_T"), ("travel_away_kkm", "travel_T"), ("tz_away", "tz_T"),
                   ("miss_min_away", "miss_min_T"), ("miss_st_away", "miss_st_T"), ("m_3pa_away", "m_3pa"),
                   ("m_3pct_away", "m_3pct")):
        gm[c] = As[src].values
    poss_act = TG.groupby("game_id").poss.first()
    gm["poss_actual"] = gm.game_id.map(poss_act)
    am, at = gm.home_score - gm.away_score, gm.home_score + gm.away_score
    gm["err_margin"], gm["err_total"] = gm.margin - am, gm.total - at
    gm["err_margin_b"], gm["err_total_b"] = gm.margin_b - am, gm.total_b - at
    gm["err_margin_mkt"], gm["err_total_mkt"] = -gm.spread_home - am, gm.total_line - at
    gm["err_margin_core"], gm["err_total_core"] = gm.margin_core - am, gm.total_core - at
    gm["err_margin_mx"], gm["err_total_mx"] = gm.margin_mx - am, gm.total_mx - at
    gm["err_home"], gm["err_away"] = gm.mu_home - gm.home_score, gm.mu_away - gm.away_score
    gm["err_poss"] = gm.poss_pred - gm.poss_actual
    keep = ["game_id", "date", "season", "home", "away", "neutral", "playoff", "mu_home", "mu_away", "sd_home",
            "sd_away", "rho", "sd_margin", "sd_total", "margin", "total", "margin_b", "total_b", "w_mkt",
            "w_mkt_margin", "w_mkt_total", "margin_mx", "total_mx", "mu_home_core", "mu_away_core", "margin_core",
            "total_core", "corr_home", "corr_away", "spread_home", "total_line", "spread_open_home", "total_open",
            "home_ml", "away_ml", "home_score", "away_score", "hfa", "level", "games_home", "games_away"] + \
        [c for c in gm.columns if c.startswith(("pace_", "ortg_", "drtg_", "f3r_", "ftr_", "orb_", "tov_", "pts_poss",
                                                "rest_", "b2b_", "travel_", "tz_", "miss_", "m_3p"))] + \
        ["poss_pred", "poss_actual"] + [c for c in gm.columns if c.startswith("err_")]
    keep = list(dict.fromkeys(keep))
    G = gm[gm.season >= START][keep].reset_index(drop=True)
    G.to_parquet(OUT / "nba_games.parquet", index=False)
    print(f"games written {len(G)} ({time.time() - T0:.0f}s)")

    # ---- players (alpha tuned on 2016-2020 pts MAE)
    gall = gm[keep]
    bestp = None
    PL = player_frame(box, g, gall, TG, H, A)
    for alpha in (0.0, 0.5, 1.0):
        P = player_project(PL, alpha=alpha)
        x = P[(P.stat == "pts") & P.season.between(START, TUNE_END)]
        v = mae(x.mu - x.actual)
        print(f"  alpha={alpha}: pts MAE 2016-20 {v:.4f}")
        if bestp is None or v < bestp[0]:
            bestp = (v, alpha, P)
    ALPHA, P = bestp[1], bestp[2]
    P = P[P.season >= START]
    pcols = ["game_id", "date", "season", "player_id", "player", "team", "opp", "stat", "mu", "sd", "mu_nm", "sd_nm",
             "share_of_team", "team_stat_script", "min_e", "min_exp", "blowout_p", "tier", "actual", "minutes",
             "team_actual", "mu_raw"]
    P[pcols].to_parquet(OUT / "nba_players.parquet", index=False)
    print(f"players written {len(P)} ({time.time() - T0:.0f}s)")
    PE = props_eval(P)

    write_report(G, ABL, pd.DataFrame(tune_rows), pd.DataFrame(rls_rows), core_kw, LAM, RIDGE, coefs, P, PE, ALPHA)
    print(f"done ({time.time() - T0:.0f}s)")


def write_report(G, ABL, TUNE, RLSG, core_kw, LAM, RIDGE, coefs, P, PE, ALPHA):
    L = []
    d = G.dropna(subset=["err_margin"])
    dm = d.dropna(subset=["err_margin_mkt", "err_total_mkt"])
    # learning curve
    lc = dm.groupby("season").agg(games=("err_margin", "size"),
                                  margin_core=("err_margin_core", lambda x: x.abs().mean()),
                                  margin_script=("err_margin", lambda x: x.abs().mean()),
                                  margin_blend=("err_margin_b", lambda x: x.abs().mean()),
                                  margin_mkt_plus=("err_margin_mx", lambda x: x.abs().mean()),
                                  margin_market=("err_margin_mkt", lambda x: x.abs().mean()),
                                  total_core=("err_total_core", lambda x: x.abs().mean()),
                                  total_script=("err_total", lambda x: x.abs().mean()),
                                  total_blend=("err_total_b", lambda x: x.abs().mean()),
                                  total_mkt_plus=("err_total_mx", lambda x: x.abs().mean()),
                                  total_market=("err_total_mkt", lambda x: x.abs().mean()),
                                  w_mkt_margin=("w_mkt_margin", "last"), w_mkt_total=("w_mkt_total", "last")).reset_index()
    X, f = ml_logloss(d)
    llr = X.groupby("season").apply(lambda x: pd.Series({"ml_ll_script": f(x._ps.values), "ml_ll_blend": f(x._pb.values),
                                                         "ml_ll_market": f(x._pm)})).reset_index()
    lc = lc.merge(llr, on="season", how="left")
    r21 = dm[dm.season >= REPORT]
    L.append("# NBA script-then-learn: learning report\n")
    L.append("Code: `engine/scripts/nba.py` (core = `engine/scripts/base.py`, unmodified). Games 2010-2026 are run "
             "sequentially (2010-15 warm-up). Each date's games are all scripted before any of them is learned. Tuning "
             "used 2016-2020 only, and 2021+ is the reported period. Market = closing spread/total; moneyline de-vigged "
             "with `devig(..., 'odds_ratio')`.\n")
    L.append(f"Tuned core Kalman: {core_kw}. Online RLS correction: forgetting λ={LAM}, ridge={RIDGE}. "
             f"Player renormalisation exponent α={ALPHA}.\n")
    L.append("## Bottom line\n")
    s = lambda c: r21[c].abs().mean()
    L.append(f"- **2021+ margin MAE:** core {s('err_margin_core'):.3f}, enriched script {s('err_margin'):.3f}, "
             f"learned blend {s('err_margin_b'):.3f}, market + learned residual {s('err_margin_mx'):.3f}, "
             f"**closing line {s('err_margin_mkt'):.3f}** (n={len(r21)}).")
    L.append(f"- **2021+ total MAE:** core {s('err_total_core'):.3f}, enriched {s('err_total'):.3f}, blend "
             f"{s('err_total_b'):.3f}, market + residual {s('err_total_mx'):.3f}, **closing line {s('err_total_mkt'):.3f}**.")
    L.append(f"- **Learned market weight** at the end of 2026: margin {lc.w_mkt_margin.iloc[-1]:.2f}, total "
             f"{lc.w_mkt_total.iloc[-1]:.2f}; mean over 2021+ was {r21.w_mkt_margin.mean():.2f} / {r21.w_mkt_total.mean():.2f}.")
    # residual information
    cm = np.corrcoef(r21.margin - (-r21.spread_home), -(r21.err_margin_mkt))[0, 1]
    ct = np.corrcoef(r21.total - r21.total_line, -(r21.err_total_mkt))[0, 1]
    L.append(f"- **Information beyond the close (2021+):** corr(script − close, actual − close) is {cm:.3f} for "
             f"margin and {ct:.3f} for total.")
    op = r21.dropna(subset=["spread_open_home", "total_open"])
    if len(op) > 200:
        cmo = np.corrcoef(op.margin + op.spread_open_home, op.spread_open_home - op.spread_home)[0, 1]
        cto = np.corrcoef(op.total - op.total_open, op.total_line - op.total_open)[0, 1]
        L.append(f"- **Line movement (CLV) on games with openers ({len(op)}):** corr(script − open, close − open) is "
                 f"{cmo:.3f} for spreads and {cto:.3f} for totals. If positive, the script anticipates where the line "
                 "moves, even though it does not beat the close.")
    # ATS / OU ROI with thresholds chosen on 2016-20
    tr = dm[dm.season.between(START, TUNE_END)]
    rows = []
    for nm, pc, mk, ac in (("spread: script", "margin", "mkt_m", "act_m"), ("spread: mkt+resid", "margin_mx", "mkt_m", "act_m"),
                           ("total: script", "total", "total_line", "act_t"), ("total: mkt+resid", "total_mx", "total_line", "act_t")):
        def arr(x):
            mkm = -x.spread_home.values if mk == "mkt_m" else x.total_line.values
            act = (x.home_score - x.away_score).values if ac == "act_m" else (x.home_score + x.away_score).values
            return x[pc].values, mkm, act
        bt, best_roi = None, -9
        for thr in (0.5, 1, 2, 3, 4, 6):
            n_, r_ = ats_roi(*arr(tr), thr)
            if n_ >= 300 and r_ > best_roi:
                bt, best_roi = thr, r_
        n2, r2 = ats_roi(*arr(r21), bt) if bt else (0, np.nan)
        rows.append(dict(bet=nm, thr=bt, roi_2016_20=best_roi, n_2021p=n2, roi_2021p=r2))
    L.append("\n**Betting the closing number at −110** (edge threshold picked on 2016-20 with n≥300, reported on 2021+):\n")
    L.append(md_table(pd.DataFrame(rows)))
    L.append("\n## Learning curve (per season, MAE in points; ML log loss)\n")
    L.append(md_table(lc))
    # within-season learning
    d2 = dm.copy()
    lg_ = pd.concat([d2[["game_id", "season", "date", "home"]].rename(columns={"home": "t"}),
                     d2[["game_id", "season", "date", "away"]].rename(columns={"away": "t"})]).sort_values("date")
    lg_["k"] = lg_.groupby(["season", "t"]).cumcount()
    d2["gnum"] = pd.cut(d2.game_id.map(lg_.groupby("game_id").k.min()), [-1, 10, 25, 50, 82, 200],
                        labels=["0-10", "11-25", "26-50", "51-82", "playoffs/83+"])
    ws = d2[d2.season >= START].groupby("gnum", observed=True).agg(n=("err_margin", "size"),
                                                                 script=("err_margin", lambda x: x.abs().mean()),
                                                                 core=("err_margin_core", lambda x: x.abs().mean()),
                                                                 market=("err_margin_mkt", lambda x: x.abs().mean())).reset_index()
    L.append("\n**Within-season learning:** margin MAE by the fewer of the two teams' games played that season (2016+):\n")
    L.append(md_table(ws.astype({"gnum": str})))
    # error attribution
    L.append("\n## Error attribution\n")
    e = d[d.poss_actual.notna() & (d.season >= REPORT)].copy()
    ppp_pred = e.total / e.poss_pred
    ppp_act = (e.home_score + e.away_score) / e.poss_actual
    e["pace_part"] = (e.poss_pred - e.poss_actual) * ppp_pred
    e["eff_part"] = e.poss_actual * (ppp_pred - ppp_act)
    vt = e.err_total.var()
    L.append(f"**Totals (2021+), split into pace and efficiency.** Total error = (poss_pred − poss_act)·ppp_pred + "
             f"poss_act·(ppp_pred − ppp_act). Pace part: sd {e.pace_part.std():.2f} pts. Efficiency part: sd "
             f"{e.eff_part.std():.2f} pts. The total-error sd is {np.sqrt(vt):.2f}, and the covariance share is "
             f"{2 * np.cov(e.pace_part, e.eff_part)[0, 1] / vt:.2f}. Possession MAE is {mae(e.err_poss):.2f}, and the "
             f"mean possessions error (bias) is {e.err_poss.mean():+.2f}. Shooting-efficiency noise dominates. Overtime "
             f"games are {(e.poss_actual > e.poss_pred + 8).mean():.1%} of games with +8 or more possessions.\n")
    # by game type
    e = d[d.season >= REPORT].copy()
    e["b2b_any"] = (e.b2b_home + e.b2b_away) > 0
    e["inj_any"] = (e.miss_st_home + e.miss_st_away) > 0
    e["blowout"] = (e.home_score - e.away_score).abs() >= 20
    e["clash"] = pd.qcut(e.pace_clash, 3, labels=["slow-fast clash", "neutral", "same-tempo"])
    e["three"] = pd.qcut(e.m_3pa_home + e.m_3pa_away, 3, labels=["3pt matchup low", "mid", "high"])
    rows = []
    for nm, col in (("back-to-back involved", "b2b_any"), ("starter missed prev game", "inj_any"),
                    ("final margin ≥20", "blowout"), ("playoff", "playoff")):
        for v in (True, False):
            x = e[e[col] == v]
            rows.append(dict(group=f"{nm}={v}", n=len(x), mae_margin_script=mae(x.err_margin), mae_margin_mkt=mae(x.err_margin_mkt),
                             mae_total_script=mae(x.err_total), mae_total_mkt=mae(x.err_total_mkt),
                             bias_total_script=x.err_total.mean(), bias_total_mkt=x.err_total_mkt.mean()))
    for col in ("clash", "three"):
        for v, x in e.groupby(col, observed=True):
            rows.append(dict(group=f"{v}", n=len(x), mae_margin_script=mae(x.err_margin), mae_margin_mkt=mae(x.err_margin_mkt),
                             mae_total_script=mae(x.err_total), mae_total_mkt=mae(x.err_total_mkt),
                             bias_total_script=x.err_total.mean(), bias_total_mkt=x.err_total_mkt.mean()))
    L.append("**By game type (2021+).** Where the script falls furthest behind the market:\n")
    L.append(md_table(pd.DataFrame(rows)))
    L.append("\n**Online RLS coefficients at the end of 2026** (points per 1 sd of feature, for team T's score; "
             "`d_poss` = possession-model points − core points):\n")
    L.append(md_table(pd.DataFrame({"feature": list(coefs), "coef": list(coefs.values())})))
    # ablation
    L.append("\n## Ablation (online correction re-run with groups dropped; MAE)\n")
    L.append(md_table(ABL.pivot_table(index="config", columns="period", values=["mae_margin", "mae_total"]).round(3).reset_index()
                      .set_axis(["config", "margin 2016-20", "margin 2021+", "total 2016-20", "total 2021+"], axis=1)))
    mk = ABL[ABL.period == "2021+"].iloc[0]
    L.append(f"\nClosing line on the same 2021+ games: margin {mk.mae_margin_mkt:.3f}, total {mk.mae_total_mkt:.3f}.\n")
    L.append("Core tuning grid (2016-20):\n\n" + md_table(TUNE) + "\n\nRLS grid (2016-20):\n\n" + md_table(RLSG))
    # players
    L.append("\n## Player script\n")
    pr = P[P.season >= REPORT]
    rows = []
    for s_, x in pr.groupby("stat"):
        # oracle attribution: replace scripted team stat by actual / expected minutes by actual
        team_or = x.mu * (x.team_actual / x.team_stat_script)
        min_or = x.mu * (x.minutes / x.min_exp)
        naive = x.groupby("player_id").actual.transform(lambda z: z.shift(1).rolling(10, 1).mean())
        rows.append(dict(stat=s_, n=len(x), mae_script=mae(x.mu - x.actual), mae_no_market=mae(x.mu_nm - x.actual),
                         mae_last10_avg=mae(naive - x.actual), mae_if_team_stat_known=mae(team_or - x.actual),
                         mae_if_minutes_known=mae(min_or - x.actual),
                         coverage_1sd=float(((x.actual - x.mu).abs() <= x.sd).mean())))
    L.append("MAE on 2021+, by stat. The oracle columns swap in one true quantity, which shows how much error each "
             "part of the chain carries: team-script error, then minutes error, and the rest is usage/efficiency.\n")
    L.append(md_table(pd.DataFrame(rows)))
    bl = pr[pr.stat == "pts"].copy()
    bl["bp_bin"] = pd.qcut(bl.blowout_p, 4)
    bt = bl.groupby("bp_bin", observed=True).apply(lambda x: pd.Series({
        "n": len(x), "pred_min_ratio": (x.min_exp / x.min_e).mean(), "actual_min_ratio": (x.minutes / x.min_e).mean()})).reset_index()
    L.append("\n**Blowout minutes model, 34+ min starters and others pooled** (pts rows, 2021+): predicted vs actual "
             "minutes relative to the EWMA, by quartile of scripted blowout probability:\n")
    L.append(md_table(bt.astype({"bp_bin": str})))
    # props
    L.append("\n## Points props vs real lines (2024-25 = choose, 2025-26 = report)\n")
    L.append(f"Projection coverage of prop quotes: {PE['coverage']:.1%}. Main-line point MAE on unique player-days: "
             f"{PE['mae']}.\n")
    LLp = PE["LL"].pivot_table(index="prob", columns=["season", "main_only"], values="logloss")
    LLp.columns = [f"LL {s} {'main' if m else 'all'}" for s, m in LLp.columns]
    L.append("Log loss. `p_dv` is the quote's own odds-ratio de-vig. `p_mkt` is the leave-one-book-out consensus. "
             "`p_mod` is the existing HGB model. `p_blend` = .75 p_mkt + .25 p_mod. `p_script` / `p_script_nm` are "
             "this script, with and without the market-blended team script. `p_mkt_script` = .75 p_mkt + .25 p_script. "
             f"`p_sel` uses weights (mod, script) = {tuple(round(v, 2) for v in PE['blend_w'])} chosen on 2025 main-line log loss.\n")
    L.append(md_table(LLp.reset_index(), "{:.5f}"))
    L.append("\n**ROI at the EV threshold chosen on 2024-25** (n≥300), reported on 2025-26 (`pick_bets`, best quote per "
             "player-day):\n")
    L.append(md_table(PE["chosen"]))
    R = PE["ROI"]
    L.append("\nFull ROI grid:\n")
    L.append(md_table(R.pivot_table(index=["prob", "thr"], columns="season", values=["n", "roi"]).reset_index()
                      .set_axis(["prob", "thr", "n 2025", "n 2026", "roi 2025", "roi 2026"], axis=1)))
    (OUT / "nba_learning.md").write_text("\n".join(L) + "\n" + CONCLUSION_PLACEHOLDER + "\n")
    PE["LL"].to_csv(OUT / "nba_props_logloss.csv", index=False)
    PE["ROI"].to_csv(OUT / "nba_props_roi.csv", index=False)


CONCLUSION_PLACEHOLDER = "<!-- conclusions -->"

if __name__ == "__main__":
    main()
