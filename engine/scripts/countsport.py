"""Shared machinery for the low-scoring (count-model) adapters: MLB runs and NHL goals.

Wraps the shared core (`base.ScriptEngine`, count model) with:
  * keyed 1-D Kalman components (starting pitcher, goalie, park) that shift one side's expected score and
    learn from that side's innovation after each game,
  * an online recursive-least-squares (RLS, with forgetting) correction on pre-game features,
  * online logistic layers (extended-Kalman logistic) for (a) calibrating the simulated script win
    probability and (b) blending script and de-vigged market probability.
Everything is strictly sequential: each game's script uses state built only from earlier games; the reveal
then updates every component. Win/total probabilities come from `base.simulate_scores(..., count_model=True)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine.scripts.base import Params, ScriptEngine, simulate_scores


def american_to_prob(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def devig_pair(a, b):
    pa, pb = american_to_prob(a), american_to_prob(b)
    return pa / (pa + pb)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------------------------------- components
@dataclass
class KalmanComponent:
    """Per-key scalar effect (in score units on the side it affects)."""
    prior_mean: float = 0.0
    prior_var: float = 0.1
    q: float = 0.0
    carry: float = 0.8
    gain_scale: float = 1.0
    state: dict = field(default_factory=dict)        # key -> [mean, var, season, n]

    def get(self, key, season):
        if key is None or (isinstance(key, float) and key != key):
            return 0.0, 0.0, 0
        s = self.state.get(key)
        if s is None:
            s = [self.prior_mean, self.prior_var, season, 0]
            self.state[key] = s
        elif s[2] != season:
            s[0] = self.prior_mean + (s[0] - self.prior_mean) * self.carry
            s[1] = s[1] * self.carry ** 2 + self.prior_var * (1 - self.carry ** 2)
            s[2] = season
        s[1] += self.q
        return s[0], s[1], s[3]

    def update(self, key, innov, obs_var, weight=1.0):
        if key is None or key not in self.state:
            return
        s = self.state[key]
        h = weight
        k = self.gain_scale * s[1] * h / (h * h * s[1] + obs_var)
        s[0] += k * innov
        s[1] = (1 - k * h) * s[1]
        s[3] += 1


class RLS:
    """Recursive least squares with exponential forgetting: residual ~ beta . x."""

    def __init__(self, dim, lam=0.9998, p0=0.05, noise=1.0):
        self.b = np.zeros(dim)
        self.P = np.eye(dim) * p0
        self.lam = lam
        self.noise = noise

    def predict(self, x):
        return float(self.b @ x)

    def update(self, x, y):
        Px = self.P @ x
        g = Px / (self.lam * self.noise + x @ Px)
        self.b = self.b + g * (y - self.b @ x)
        self.P = (self.P - np.outer(g, Px)) / self.lam


class OnlineLogit:
    """Extended-Kalman logistic regression (online Newton), with forgetting."""

    def __init__(self, w0, p0=1.0, lam=0.9995):
        self.w = np.asarray(w0, float)
        self.P = np.eye(len(self.w)) * p0
        self.lam = lam

    def predict(self, x):
        return float(sigmoid(self.w @ x))

    def update(self, x, y):
        self.P = self.P / self.lam
        p = sigmoid(self.w @ x)
        v = p * (1 - p)
        Px = self.P @ x
        self.P = self.P - np.outer(Px, Px) * v / (1 + v * (x @ Px))
        self.w = self.w + self.P @ x * (y - p)


# ------------------------------------------------------------------------------------------- run loop
@dataclass
class AdapterConfig:
    params: Params
    components: dict                      # name -> KalmanComponent
    feature_names: list                   # per-side feature names (columns f"h_{n}" / f"a_{n}" in games)
    use_rls: bool = True
    rls_lam: float = 0.9998
    rls_p0: float = 0.02
    tie_home: float = 0.5                 # share of simulated ties credited to home (extra innings / OT)
    n_sim: int = 6000
    margin_sd: float = 4.3                # used to translate the de-vigged moneyline into a market margin
    min_mu: float = 0.3


def run_adapter(g: pd.DataFrame, cfg: AdapterConfig, verbose: bool = False) -> pd.DataFrame:
    """g needs: game_id, date, season, home, away, home_score, away_score, p_mkt (de-vigged home win prob or
    NaN), total_line, plus per-side feature columns h_<f>/a_<f> and component key columns
    `<comp>_h_key` (component acting on HOME score, e.g. the AWAY starting pitcher) / `<comp>_a_key`."""
    eng = ScriptEngine(cfg.params)
    comps = cfg.components
    nf = len(cfg.feature_names)
    rls = RLS(nf, lam=cfg.rls_lam, p0=cfg.rls_p0, noise=cfg.params.init_sd ** 2)
    calib = OnlineLogit([0.0, 1.0], p0=0.5)                 # p_cal = s(a + b*logit(p_sim))
    blend = OnlineLogit([0.0, 0.5, 0.5], p0=0.5)            # p_blend = s(c + u*logit(p_cal) + v*logit(p_mkt))
    rng = np.random.default_rng(7)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    Xh = g[[f"h_{n}" for n in cfg.feature_names]].to_numpy(float) if nf else np.zeros((len(g), 0))
    Xa = g[[f"a_{n}" for n in cfg.feature_names]].to_numpy(float) if nf else np.zeros((len(g), 0))
    Xh, Xa = np.nan_to_num(Xh), np.nan_to_num(Xa)
    keys = {c: (g[f"{c}_h_key"].to_numpy(object), g[f"{c}_a_key"].to_numpy(object)) for c in comps}
    out = []
    for i, r in enumerate(g.itertuples(index=False)):
        season = int(r.season)
        neutral = bool(getattr(r, "neutral", 0) or 0)
        sc = eng.script(r.home, r.away, season, neutral)
        ext_h = ext_a = 0.0
        var_ext_h = var_ext_a = 0.0
        comp_vals = {}
        for c, comp in comps.items():
            kh, ka = keys[c][0][i], keys[c][1][i]
            mh, vh, nh = comp.get(kh, season)
            ma, va, na = comp.get(ka, season)
            comp_vals[c] = (mh, ma, nh, na)
            ext_h += mh; ext_a += ma; var_ext_h += vh; var_ext_a += va
        mu_h0 = sc["mu_home"] + ext_h
        mu_a0 = sc["mu_away"] + ext_a
        rh = rls.predict(Xh[i]) if (cfg.use_rls and nf) else 0.0
        ra = rls.predict(Xa[i]) if (cfg.use_rls and nf) else 0.0
        mu_h = max(mu_h0 + rh, cfg.min_mu)
        mu_a = max(mu_a0 + ra, cfg.min_mu)
        sd_h = math.sqrt(sc["sd_home"] ** 2 + var_ext_h)
        sd_a = math.sqrt(sc["sd_away"] ** 2 + var_ext_a)
        row_s = dict(mu_home=mu_h, mu_away=mu_a, sd_home=sd_h, sd_away=sd_a, rho=sc["rho"])
        sims = simulate_scores(row_s, n=cfg.n_sim, count_model=True, rng=rng)
        dh = sims[:, 0] - sims[:, 1]
        p_sim = float(np.mean(dh > 0) + cfg.tie_home * np.mean(dh == 0))
        tl = r.total_line
        p_over_sim = float(np.mean(sims.sum(1) > tl)) if tl == tl else np.nan
        p_push = float(np.mean(sims.sum(1) == tl)) if tl == tl else np.nan
        xc = np.array([1.0, float(logit(p_sim))])
        p_cal = calib.predict(xc)
        p_mkt = r.p_mkt
        has_mkt = p_mkt == p_mkt
        if has_mkt:
            xb = np.array([1.0, float(logit(p_cal)), float(logit(p_mkt))])
            p_bl = blend.predict(xb)
            mkt_margin = cfg.margin_sd * float(norm.ppf(np.clip(p_mkt, 0.02, 0.98)))
        else:
            p_bl, mkt_margin = p_cal, np.nan
        sc_full = dict(sc, mu_home=mu_h, mu_away=mu_a, margin=mu_h - mu_a, total=mu_h + mu_a)
        b = eng.blend(sc_full, mkt_margin, tl if tl == tl else np.nan)
        row = dict(game_id=r.game_id, date=r.date, season=season, home=r.home, away=r.away,
                   mu_home=mu_h, mu_away=mu_a, sd_home=sd_h, sd_away=sd_a, rho=sc["rho"],
                   margin=mu_h - mu_a, total=mu_h + mu_a, mu_home_core=sc["mu_home"], mu_away_core=sc["mu_away"],
                   rls_home=rh, rls_away=ra, **b,
                   p_home_sim=p_sim, p_home_script=p_cal, p_home_mkt=p_mkt, p_home_blend=p_bl,
                   p_over_script=p_over_sim, p_push_script=p_push, mkt_margin=mkt_margin,
                   total_line=tl, home_score=r.home_score, away_score=r.away_score,
                   level=eng.m, hfa=eng.hfa, noise_sd=eng.sd, score_rho=eng.rho)
        for c, (mh, ma, nh, na) in comp_vals.items():
            row[f"{c}_home_eff"] = mh; row[f"{c}_away_eff"] = ma
            row[f"{c}_home_n"] = nh; row[f"{c}_away_n"] = na
        yh, ya = r.home_score, r.away_score
        if yh == yh and ya == ya:
            yh, ya = float(yh), float(ya)
            act_m, act_t = yh - ya, yh + ya
            win = 1.0 if yh > ya else (0.0 if yh < ya else np.nan)
            row.update(err_margin=row["margin"] - act_m, err_total=row["total"] - act_t,
                       err_margin_b=b["margin_b"] - act_m, err_total_b=b["total_b"] - act_t,
                       err_margin_mkt=mkt_margin - act_m if has_mkt else np.nan,
                       err_total_mkt=tl - act_t if tl == tl else np.nan,
                       err_home=mu_h - yh, err_away=mu_a - ya, home_win=win)
            # --- learn
            eng.learn_blend(sc_full, mkt_margin, tl if tl == tl else np.nan, act_m, act_t)
            if win == win:
                calib.update(xc, win)
                if has_mkt:
                    blend.update(xb, win)
            innov_h, innov_a = yh - mu_h, ya - mu_a
            obs = eng.sd ** 2
            if cfg.use_rls and nf:
                rls.update(Xh[i], yh - mu_h0)
                rls.update(Xa[i], ya - mu_a0)
            for c, comp in comps.items():
                kh, ka = keys[c][0][i], keys[c][1][i]
                if kh is not None and kh == kh and kh == ka:          # same key on both sides (park)
                    comp.update(kh, (innov_h + innov_a) / 2, obs * (1 + eng.rho) / 2)
                else:
                    comp.update(kh, innov_h, obs + sc["sd_home"] ** 2 - eng.sd ** 2)
                    comp.update(ka, innov_a, obs + sc["sd_away"] ** 2 - eng.sd ** 2)
            eng.learn(r.home, r.away, season, yh - (mu_h - sc["mu_home"]), ya - (mu_a - sc["mu_away"]), neutral)
        out.append(row)
        if verbose and i % 5000 == 0:
            print(i, r.date, flush=True)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- reporting
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def season_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["err_margin"]).copy()
    m = d.dropna(subset=["home_win", "p_home_mkt"])
    t = d.dropna(subset=["err_total_mkt"])
    rows = []
    for s, x in m.groupby("season"):
        tt = t[t.season == s]
        rows.append(dict(season=s, games=len(d[d.season == s]), games_mkt=len(x),
                         ll_script=ll(x.p_home_script, x.home_win).mean(), ll_market=ll(x.p_home_mkt, x.home_win).mean(),
                         ll_blend=ll(x.p_home_blend, x.home_win).mean(),
                         brier_script=((x.p_home_script - x.home_win) ** 2).mean(),
                         brier_market=((x.p_home_mkt - x.home_win) ** 2).mean(),
                         mae_total_script=tt.err_total.abs().mean(), mae_total_market=tt.err_total_mkt.abs().mean(),
                         mae_total_blend=tt.err_total_b.abs().mean(),
                         mae_margin_script=x.err_margin.abs().mean(), mae_margin_market=x.err_margin_mkt.abs().mean(),
                         w_mkt_end=x.w_mkt.iloc[-1]))
    return pd.DataFrame(rows).set_index("season")


def summary(df: pd.DataFrame, since: int) -> dict:
    d = df[(df.season >= since)].dropna(subset=["home_win", "p_home_mkt", "err_total_mkt"])
    return dict(n=len(d), ll_script=ll(d.p_home_script, d.home_win).mean(), ll_market=ll(d.p_home_mkt, d.home_win).mean(),
                ll_blend=ll(d.p_home_blend, d.home_win).mean(),
                mae_total_script=d.err_total.abs().mean(), mae_total_market=d.err_total_mkt.abs().mean(),
                mae_total_blend=d.err_total_b.abs().mean(),
                rmse_total_script=np.sqrt((d.err_total ** 2).mean()), rmse_total_market=np.sqrt((d.err_total_mkt ** 2).mean()),
                mae_home_runs=(d.err_home.abs().mean()))


def md_table(df: pd.DataFrame, floatfmt=3) -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    idx = df.index.name or ""
    lines = ["| " + " | ".join([idx] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        vals = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                vals.append("" if v != v else f"{v:.{floatfmt}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join([str(i)] + vals) + " |")
    return "\n".join(lines)


def bet_sim(d: pd.DataFrame, p_col: str, thr: float = 0.03) -> dict:
    """Flat 1u moneyline bets where model prob exceeds de-vigged market by thr, at the posted closing price."""
    x = d.dropna(subset=["home_win", "home_ml", "away_ml", "p_home_mkt"])
    pnl, n = [], 0
    for r in x.itertuples(index=False):
        p = getattr(r, p_col)
        for side, pm, odds, won in (("h", p, r.home_ml, r.home_win == 1), ("a", 1 - p, r.away_ml, r.home_win == 0)):
            pmk = r.p_home_mkt if side == "h" else 1 - r.p_home_mkt
            if pm - pmk > thr:
                dec = 1 + (odds / 100 if odds > 0 else 100 / -odds)
                pnl.append(dec - 1 if won else -1.0)
    pnl = np.array(pnl)
    if len(pnl) == 0:
        return dict(bets=0, roi=np.nan, z=np.nan)
    return dict(bets=len(pnl), roi=pnl.mean(), z=pnl.mean() / (pnl.std(ddof=1) / math.sqrt(len(pnl))) if len(pnl) > 1 else np.nan)
