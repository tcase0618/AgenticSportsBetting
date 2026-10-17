"""
Script-fit same-game parlays (SGP): legs that follow the pre-game script, three joint probabilities, fair odds.

Used two ways:
  * nightly engine:  build_sgps(script_row, player_rows, lines, sport=...) -> ranked SGP list
  * backtest:        engine/scripts/sgp_backtest.py imports the leg / simulation / copula helpers below

Legs "fit the script": every leg is taken in the direction the script leans versus the line
(script P(pick) > market-fair P(pick)): side (spread, or moneyline for MLB/NHL and as an alternative),
game total, player props (NBA pts/reb/ast/3pm, NFL pass/rush/rec yds & receptions, NHL SOG/points,
MLB starter Ks). UFC: winner + goes-distance / inside-distance.

Joint probability of an SGP, three ways
  script     Monte Carlo from the script: simulate_scores() for the two team scores; each player's stat is
             coupled to his team's simulated score through a latent normal  z_p = b*z_team + sqrt(1-b^2)*e
             (b = share * sd_team / sd_player for share-of-scoring stats (NBA pts, NHL points), otherwise b
             learned from PRIOR seasons' script residuals), mapped through the player's own marginal
             (normal / negative binomial / Poisson).
  market     de-vigged market marginals + Gaussian copula with pairwise correlations learned online from
             settled games only (RhoTable). This approximates the fair SGP price before the book's hold.
  indep      product of the market marginals (how a naive parlay is priced).
The builder's ranked output uses a calibrated script probability (online logistic, learned in the
backtest) when results/scripts/sgp_state.json exists.

Prop "lines" when no real line exists are a MARKET PROXY built only from prior data: the player's
trailing 10-game average (min 5 games, prior games only) rounded to x.5, with a fair over-probability
equal to the historical over-rate of such proxy lines (prior dates only). That is NOT a real book.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from backtest.core import american_to_decimal, decimal_to_american, fmt_american
from engine.devig import devig
from engine.scripts.base import simulate_scores

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / "results" / "scripts" / "sgp_state.json"

COUNT_SPORTS = {"nhl", "mlb"}
ML_SPORTS = {"nhl", "mlb"}                     # side leg = moneyline (fixed +/-1.5 run/puck lines have no prices)
TIE_HOME = {"mlb": 0.52, "nhl": 0.50}          # simulated ties -> extra innings / OT share to home
N_SIM = 5000
DEC_110 = float(american_to_decimal(-110))
SHRINK_N0 = 300                                # rho shrinkage pseudo-pairs
PROXY_N0 = 40                                  # Beta(20,20) prior on proxy-line over rates


# ------------------------------------------------------------------ small math
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fair2(a, b):
    """odds_ratio de-vig of a two-way market; NaN if either side missing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) >= 100) & (np.abs(b) >= 100)
    if np.any(ok):
        pa, _ = devig(np.atleast_1d(np.broadcast_to(a, out.shape)[ok]), np.atleast_1d(np.broadcast_to(b, out.shape)[ok]), "odds_ratio")
        out[ok] = pa
    return out


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def bvn_cdf(h, k, r):
    """P(X<h, Y<k) for standard bivariate normal with corr r (vectorized; Gauss-Legendre on Plackett's integral)."""
    h, k, r = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(r, float))
    r = np.clip(r, -0.995, 0.995)
    t = (r[..., None] / 2) * (_GL_X + 1)                    # nodes on [0, r]
    w = (r[..., None] / 2) * _GL_W
    hh, kk = h[..., None], k[..., None]
    f = np.exp(-(hh ** 2 - 2 * t * hh * kk + kk ** 2) / (2 * (1 - t ** 2))) / np.sqrt(1 - t ** 2)
    return norm.cdf(h) * norm.cdf(k) + (w * f).sum(-1) / (2 * np.pi)


def tetrachoric_counts(n11, n10, n01, n00):
    """Latent correlation from 2x2 counts (vectorized bisection)."""
    n = n11 + n10 + n01 + n00
    pa = (n11 + n10 + 0.5) / (n + 1)
    pb = (n11 + n01 + 0.5) / (n + 1)
    pab = (n11 + 0.25) / (n + 1)
    za, zb = norm.ppf(pa), norm.ppf(pb)
    lo, hi = np.full(n.shape, -0.95), np.full(n.shape, 0.95)
    for _ in range(30):
        mid = (lo + hi) / 2
        v = bvn_cdf(za, zb, mid)
        lo, hi = np.where(v < pab, mid, lo), np.where(v < pab, hi, mid)
    return (lo + hi) / 2


def mvn_orthant(p: np.ndarray, R: np.ndarray, base: np.ndarray) -> np.ndarray:
    """P(all legs win) under a Gaussian copula. p: (C,k) marginals, R: (C,k,k), base: (M,4) shared N(0,1).
    k=2 analytic; k>2 Monte Carlo with shared draws (same draws for every candidate -> low relative noise)."""
    C, k = p.shape
    z = norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    if k == 1:
        return p[:, 0]
    if k == 2:
        return bvn_cdf(z[:, 0], z[:, 1], R[:, 0, 1])
    out = np.empty(C)
    for i in range(C):
        Ri = R[i]
        w, V = np.linalg.eigh(Ri)
        Ri = V @ np.diag(np.clip(w, 1e-4, None)) @ V.T
        d = np.sqrt(np.diag(Ri)); Ri = Ri / np.outer(d, d)
        L = np.linalg.cholesky(Ri)
        Z = base[:, :k] @ L.T
        out[i] = np.mean(np.all(Z < z[i], axis=1))       # leg wins <=> Z < Phi^-1(p)
    return out


# ------------------------------------------------------------------ online correlation table
class RhoTable:
    """Pairwise latent correlations between canonical leg events, learned from settled games only.
    Canonical events: 'fav' (favourite covers / wins), 'over' (game over), 'team' (player's team covers/wins),
    player 'over'. Keys like 'nba|side-total|3.5-7', 'nba|side-prop|pts', 'nba|total-prop|reb',
    'nba|pp-tm|ast|pts' (teammates), 'nba|pp-op|pts|pts' (opponents)."""

    def __init__(self, n0=SHRINK_N0):
        self.c: dict[str, np.ndarray] = {}
        self.rho: dict[str, float] = {}
        self.n0 = n0

    def observe(self, key: str, a: bool, b: bool):
        v = self.c.get(key)
        if v is None:
            v = self.c[key] = np.zeros(4)
        v[0 if (a and b) else 1 if a else 2 if b else 3] += 1

    def refresh(self):
        if not self.c:
            return
        keys = list(self.c)
        M = np.array([self.c[k] for k in keys])
        r = tetrachoric_counts(M[:, 0], M[:, 1], M[:, 2], M[:, 3])
        n = M.sum(1)
        r = r * n / (n + self.n0)
        self.rho = dict(zip(keys, map(float, r)))

    def get(self, key: str) -> float:
        return self.rho.get(key, 0.0)

    def to_json(self):
        return {k: dict(rho=round(self.rho.get(k, 0.0), 4), n=int(v.sum())) for k, v in self.c.items()}

    @classmethod
    def from_json(cls, d):
        t = cls()
        t.rho = {k: v["rho"] for k, v in d.items()}
        return t


class OnlineLogit:
    """Online logistic regression (AdaGrad), updated one settled batch at a time."""

    def __init__(self, w0, lr=0.05):
        self.w = np.asarray(w0, float).copy()
        self.G = np.full_like(self.w, 1e-8)
        self.lr = lr

    def predict(self, X):
        return 1 / (1 + np.exp(-np.clip(X @ self.w, -30, 30)))

    def update(self, X, y):
        if len(y) == 0:
            return
        g = X.T @ (y - self.predict(X)) / len(y)
        self.G += g ** 2
        self.w += self.lr * g / np.sqrt(self.G)


def cal_features(p_script, n_legs):
    return np.c_[np.ones(len(p_script)), logit(p_script), np.asarray(n_legs, float) - 2]


def comb_features(p_script, p_mkt, n_legs):
    return np.c_[np.ones(len(p_script)), logit(p_script), logit(p_mkt), np.asarray(n_legs, float) - 2]


# ------------------------------------------------------------------ buckets / pair keys
def side_bucket(sport, spread_home, p_home_mkt):
    if sport in ML_SPORTS or not np.isfinite(spread_home):
        pf = max(p_home_mkt, 1 - p_home_mkt) if np.isfinite(p_home_mkt) else 0.5
        return "p<.6" if pf < 0.6 else "p.6-.7" if pf < 0.7 else "p>.7"
    s = abs(spread_home)
    return "<=3" if s <= 3 else "3.5-7" if s <= 7.5 else "7.5-14" if s <= 14.5 else "14.5+"


def pair_key(sport, A: dict, B: dict):
    """(key, sign): latent corr between the two legs AS PICKED = sign * rho[key]."""
    ka, kb = A["kind"], B["kind"]
    order = {"side": 0, "ml": 0, "total": 1, "prop": 2}
    if order[ka] > order[kb]:
        A, B, ka, kb = B, A, kb, ka
    o = A["o"] * B["o"]
    if ka in ("side", "ml") and kb == "total":
        return f"{sport}|{ka}-total|{A['bucket']}", o
    if ka in ("side", "ml") and kb == "prop":
        t = 1 if B["team_is_home"] == A["fav_home"] else -1        # player's team is the favourite?
        return f"{sport}|{ka}-prop|{B['stat']}", o * t
    if ka == "total" and kb == "prop":
        return f"{sport}|total-prop|{B['stat']}", o
    if ka == "prop" and kb == "prop":
        rel = "tm" if A["team_is_home"] == B["team_is_home"] else "op"
        s1, s2 = sorted([A["stat"], B["stat"]])
        return f"{sport}|pp-{rel}|{s1}|{s2}", o
    return None, 0


def market_R(sport, legs: list[dict], rho: RhoTable):
    k = len(legs)
    R = np.eye(k)
    for i, j in itertools.combinations(range(k), 2):
        key, s = pair_key(sport, legs[i], legs[j])
        R[i, j] = R[j, i] = s * rho.get(key) if key else 0.0
    return R


# ------------------------------------------------------------------ legs
def game_legs(sport: str, g) -> list[dict]:
    """Side / moneyline / total legs from a script row with lines. Each leg carries the pick in the script's
    direction, its market-fair probability and (where a real price exists) the decimal price."""
    legs = []
    sp, tl = float(g.get("spread_home", np.nan)), float(g.get("total_line", np.nan))
    hm, am = g.get("home_ml", np.nan), g.get("away_ml", np.nan)
    p_home_ml = float(fair2(hm, am)) if np.isfinite(hm) and np.isfinite(am) else np.nan
    fav_home = (sp < 0) if (sport not in ML_SPORTS and np.isfinite(sp) and sp != 0) else \
        (p_home_ml > 0.5 if np.isfinite(p_home_ml) else True)
    bucket = side_bucket(sport, sp, p_home_ml)
    base = dict(fav_home=bool(fav_home), bucket=bucket)
    if sport not in ML_SPORTS and np.isfinite(sp):
        legs.append(dict(kind="side", line=sp, p_mkt_home=0.5, dec_home=DEC_110, dec_away=DEC_110, **base))
    if np.isfinite(p_home_ml):
        legs.append(dict(kind="ml", line=0.0, p_mkt_home=p_home_ml, dec_home=float(american_to_decimal(hm)),
                         dec_away=float(american_to_decimal(am)), **base))
    if np.isfinite(tl):
        oo, uo = g.get("over_odds", np.nan), g.get("under_odds", np.nan)
        po = float(fair2(oo, uo)) if np.isfinite(oo) and np.isfinite(uo) else 0.5
        legs.append(dict(kind="total", line=tl, p_mkt_home=po,
                         dec_home=float(american_to_decimal(oo)) if np.isfinite(oo) else DEC_110,
                         dec_away=float(american_to_decimal(uo)) if np.isfinite(uo) else DEC_110, **base))
    return legs


def player_threshold(dist: str, mu, sd, r, line):
    """Latent-normal threshold t such that stat > line  <=>  z > t; returns (t, P(over))."""
    mu, sd, r, line = (np.asarray(x, float) for x in (mu, sd, r, line))
    if dist == "normal":
        t = (line - mu) / np.maximum(sd, 1e-6)
        return t, 1 - norm.cdf(t)
    k = np.floor(line)
    if dist == "poisson":
        F = poisson.cdf(k, np.maximum(mu, 1e-6))
    else:
        rr = np.maximum(r, 0.05)
        F = nbinom.cdf(k, rr, rr / (rr + np.maximum(mu, 1e-6)))
    F = np.clip(F, 1e-6, 1 - 1e-6)
    return norm.ppf(F), 1 - F


def simulate_game(sport: str, g, game_lgs: list[dict], prop_lgs: list[dict], n=N_SIM, rng=None):
    """Script Monte Carlo. Returns (win, push) boolean matrices (n, L) for legs IN THE PICKED DIRECTION,
    order = game legs then props. Requires each leg's 'pick_home' (True = home/over side) to be set."""
    rng = rng or np.random.default_rng(0)
    cm = sport in COUNT_SPORTS
    S = simulate_scores(g, n=n, count_model=cm, rng=rng).astype(float)
    h, a = S[:, 0], S[:, 1]
    L = len(game_lgs) + len(prop_lgs)
    win = np.zeros((n, L), bool)
    push = np.zeros((n, L), bool)
    tie_u = rng.random(n) < TIE_HOME.get(sport, 0.5)
    for j, lg in enumerate(game_lgs):
        if lg["kind"] == "side":
            m = h - a + lg["line"]
            w_home, w_away = m > 0, m < 0
        elif lg["kind"] == "ml":
            w_home = (h > a) | ((h == a) & tie_u)
            w_away = ~w_home
        else:
            t = h + a
            w_home, w_away = t > lg["line"], t < lg["line"]
        win[:, j] = w_home if lg["pick_home"] else w_away
        push[:, j] = ~(w_home | w_away)
    if prop_lgs:
        zt = {True: (h - h.mean()) / max(h.std(), 1e-9), False: (a - a.mean()) / max(a.std(), 1e-9)}
        for jj, lg in enumerate(prop_lgs):
            b = float(np.clip(lg["beta"], -0.9, 0.9))
            zc = zt[bool(lg["couple_home"])]
            z = b * zc + math.sqrt(1 - b * b) * rng.standard_normal(n)
            win[:, len(game_lgs) + jj] = (z > lg["thr"]) if lg["pick_home"] else (z <= lg["thr"])
    return win, push


def script_joint(win, push, idx):
    valid = ~push[:, idx].any(1)
    if valid.sum() < 50:
        return np.nan
    return float(win[valid][:, idx].all(1).mean())


def candidate_sets(game_lgs: list[dict], prop_lgs: list[dict]) -> list[tuple]:
    """2-4 leg SGPs per game (indices into game_lgs + prop_lgs)."""
    kinds = {lg["kind"]: i for i, lg in enumerate(game_lgs)}
    side = kinds.get("side", kinds.get("ml"))
    ml = kinds.get("ml") if "side" in kinds else None
    tot = kinds.get("total")
    P = [len(game_lgs) + i for i in range(len(prop_lgs))]
    c = []
    if side is not None and tot is not None:
        c.append((side, tot))
        if ml is not None:
            c.append((ml, tot))
    if P:
        if side is not None and tot is not None:
            c.append((side, tot, P[0]))
            if len(P) > 1:
                c.append((side, tot, P[0], P[1]))
        if side is not None:
            c.append((side, P[0]))
        if tot is not None:
            c.append((tot, P[0]))
        if len(P) > 1:
            c.append((P[0], P[1]))
    return c


def leg_label(lg, g) -> str:
    if lg["kind"] == "side":
        team, line = (g["home"], lg["line"]) if lg["pick_home"] else (g["away"], -lg["line"])
        return f"{team} {line:+g}"
    if lg["kind"] == "ml":
        return f"{g['home'] if lg['pick_home'] else g['away']} ML"
    if lg["kind"] == "total":
        return f"{'Over' if lg['pick_home'] else 'Under'} {lg['line']:g}"
    return f"{lg['player']} {'over' if lg['pick_home'] else 'under'} {lg['line']:g} {lg['stat']}" + \
        ("" if lg.get("real") else " (proxy line)")


def finalize_game_legs(game_lgs, p_script_home):
    """Pick each game leg in the script's direction and attach o / p_script / p_mkt / dec for the pick."""
    for lg, ps in zip(game_lgs, p_script_home):
        lg["p_script_home"] = ps
        lg["pick_home"] = bool(ps >= lg["p_mkt_home"])
        lg["p_script"] = ps if lg["pick_home"] else 1 - ps
        lg["p_mkt"] = lg["p_mkt_home"] if lg["pick_home"] else 1 - lg["p_mkt_home"]
        lg["dec"] = lg["dec_home"] if lg["pick_home"] else lg["dec_away"]
        if lg["kind"] in ("side", "ml"):
            lg["o"] = 1 if lg["pick_home"] == lg["fav_home"] else -1
        else:
            lg["o"] = 1 if lg["pick_home"] else -1
        lg["real"] = True
    return game_lgs


def script_marginals_game(sport, g, game_lgs, n=N_SIM, rng=None):
    """P(home side / over) from the script for each game leg (pick-independent)."""
    for lg in game_lgs:
        lg["pick_home"] = True
    win, push = simulate_game(sport, g, game_lgs, [], n=n, rng=rng)
    out = []
    for j in range(len(game_lgs)):
        v = ~push[:, j]
        out.append(float(win[v, j].mean()) if v.any() else 0.5)
    return out


# ------------------------------------------------------------------ nightly entry point
def _load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def prop_legs_from_rows(sport: str, players: pd.DataFrame, lines: dict, state: dict) -> list[dict]:
    """players: one row per (player, stat) with mu, sd, [nb_r], team_is_home, [beta], [trail_avg].
    lines['props'][(player, stat)] = {'line', 'over_odds', 'under_odds'} (real) -> else proxy line."""
    props = (lines or {}).get("props", {})
    betas = state.get("beta", {}).get(sport, {})
    proxy = state.get("proxy_rate", {}).get(sport, {})
    dist = {"nhl": {"sog": "nb", "points": "poisson"}, "mlb": {"strikeouts": "nb"}}.get(sport, {})
    out = []
    for r in players.itertuples(index=False):
        d = r._asdict()
        q = props.get((d["player"], d["stat"]))
        if q is not None and np.isfinite(q.get("over_odds", np.nan)) and np.isfinite(q.get("under_odds", np.nan)):
            line, p_over, real = float(q["line"]), float(fair2(q["over_odds"], q["under_odds"])), True
            dec_o, dec_u = float(american_to_decimal(q["over_odds"])), float(american_to_decimal(q["under_odds"]))
        elif np.isfinite(d.get("trail_avg", np.nan)):
            line = math.floor(d["trail_avg"]) + 0.5
            p_over = proxy.get(f"{d['stat']}|{line if line < 10 else 'hi'}", proxy.get(d["stat"], 0.5))
            real, dec_o, dec_u = False, np.nan, np.nan
        else:
            continue
        t, p_s = player_threshold(dist.get(d["stat"], "normal"), d["mu"], d.get("sd", np.nan), d.get("nb_r", np.nan), line)
        beta = d.get("beta", np.nan)
        if not np.isfinite(beta):
            beta = betas.get(d["stat"], 0.2)
        pick_over = bool(p_s >= p_over)
        out.append(dict(kind="prop", player=d["player"], stat=d["stat"], line=line, thr=float(t), beta=beta,
                        team_is_home=bool(d["team_is_home"]),
                        couple_home=bool(d["team_is_home"]) if sport != "mlb" else not bool(d["team_is_home"]),
                        pick_home=pick_over, o=1 if pick_over else -1, real=real,
                        p_script=float(p_s if pick_over else 1 - p_s), p_mkt=float(p_over if pick_over else 1 - p_over),
                        dec=dec_o if pick_over else dec_u))
    out.sort(key=lambda x: -x["p_script"] / max(x["p_mkt"], 1e-6))
    seen, best = set(), []
    for lg in out:                                            # one leg per player, best script edge first
        if lg["player"] not in seen and lg["p_script"] > lg["p_mkt"]:
            seen.add(lg["player"]); best.append(lg)
    return best


def build_sgps(script_row, player_rows: pd.DataFrame | None = None, lines: dict | None = None, sport: str = "nba",
               edge: float = 0.05, n_sim: int = 20000, top_players: int = 3, seed: int = 0) -> list[dict]:
    """Ranked script-fit SGPs for one game.

    script_row : a row of results/scripts/<sport>_games.parquet style (mu_home, mu_away, sd_home, sd_away, rho,
                 home, away); lines (spread_home, total_line, home_ml, away_ml, over_odds, under_odds) may be on
                 the row or in `lines`.
    player_rows: DataFrame (player, stat, mu, sd, [nb_r], team_is_home, [beta], [trail_avg]).
    lines      : {'spread_home', 'total_line', 'home_ml', 'away_ml', 'over_odds', 'under_odds',
                  'props': {(player, stat): {'line', 'over_odds', 'under_odds'}}}
    Returns list of dicts: legs, p_script (raw MC), p_script_cal (learned calibration, if state exists),
    p_market_fair, p_independent, fair_american, bet_if_fanduel_at_least (fair x (1+edge)), ratio.
    """
    state = _load_state()
    g = dict(script_row)
    for k, v in (lines or {}).items():
        if k != "props":
            g[k] = v
    rng = np.random.default_rng(seed)
    glegs = game_legs(sport, g)
    finalize_game_legs(glegs, script_marginals_game(sport, g, glegs, n=n_sim, rng=rng))
    plegs = prop_legs_from_rows(sport, player_rows, lines, state)[:top_players] if player_rows is not None and len(player_rows) else []
    for lg in glegs:
        lg["label"] = leg_label(lg, g)
    for lg in plegs:
        lg["label"] = leg_label(lg, g)
    win, push = simulate_game(sport, g, glegs, plegs, n=n_sim, rng=rng)
    legs = glegs + plegs
    rho = RhoTable.from_json(state.get("rho", {}))
    cal = state.get("cal", {}).get(sport)
    base = np.random.default_rng(1).standard_normal((40000, 4))
    out = []
    for idx in candidate_sets(glegs, plegs):
        L = [legs[i] for i in idx]
        ps = script_joint(win, push, list(idx))
        pm_marg = np.array([[lg["p_mkt"] for lg in L]])
        pm = float(mvn_orthant(pm_marg, market_R(sport, L, rho)[None], base)[0])
        pi = float(np.prod(pm_marg))
        pc = float(OnlineLogit(cal).predict(cal_features([ps], [len(L)]))[0]) if cal else ps
        p_dec = pc
        fair_dec = 1 / max(p_dec, 1e-6)
        thr_dec = fair_dec * (1 + edge)
        out.append(dict(legs=[lg["label"] for lg in L], n_legs=len(L), p_script=ps, p_script_cal=pc,
                        p_market_fair=pm, p_independent=pi, ratio=p_dec / max(pm, 1e-9),
                        fair_american=fmt_american(decimal_to_american(fair_dec)),
                        market_fair_american=fmt_american(decimal_to_american(1 / max(pm, 1e-6))),
                        bet_if_fanduel_at_least=fmt_american(decimal_to_american(thr_dec)),
                        all_real_lines=all(lg.get("real", True) for lg in L)))
    out.sort(key=lambda d: -d["ratio"])
    return out


if __name__ == "__main__":      # smoke test on the last NBA game in the script file
    gm = pd.read_parquet(ROOT / "results/scripts/nba_games.parquet").iloc[-1]
    allp = pd.read_parquet(ROOT / "results/scripts/nba_players.parquet",
                           columns=["game_id", "date", "player_id", "player", "team", "stat", "mu", "sd", "share_of_team", "min_exp", "actual"])
    pl = allp[allp.game_id == gm["game_id"]]
    pl = pl[pl.min_exp >= 24].assign(team_is_home=lambda d: d.team == gm["home"])
    pl["beta"] = np.where(pl.stat == "pts", np.clip(pl.share_of_team * gm["sd_home"] / pl.sd, 0, 0.9), np.nan)
    prior = allp[(allp.date < gm["date"]) & allp.player_id.isin(pl.player_id)].sort_values("date")
    trail = prior.groupby(["player_id", "stat"]).actual.apply(lambda s: s.tail(10).mean() if len(s) >= 5 else np.nan)
    pl = pl.merge(trail.rename("trail_avg").reset_index(), on=["player_id", "stat"], how="left")   # market proxy, prior games only
    lines = dict(spread_home=gm["spread_home"], total_line=gm["total_line"], home_ml=gm["home_ml"], away_ml=gm["away_ml"])
    print(gm["away"], "@", gm["home"], gm["date"].date())
    for s in build_sgps(gm, pl, lines, sport="nba")[:5]:
        print(s["legs"], f"script {s['p_script']:.3f} cal {s['p_script_cal']:.3f} mkt {s['p_market_fair']:.3f}",
              s["fair_american"], "bet if FD >=", s["bet_if_fanduel_at_least"])
