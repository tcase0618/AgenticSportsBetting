"""Shared walk-forward game predictor for the count sports (MLB runs, NHL goals).

Pipeline (see engine/predict/SPEC.md):
  1. features   = script columns (Kalman ratings, pitcher/goalie, park, rest, pace) + de-vigged market
                  (engine.devig odds_ratio; clearly broken quotes dropped -> filled with the script, flag=0).
  2. win model  = candidates  market | script | lr_small | lr_full | lgbm | stack, refit every month
                  (expanding window, strictly date < first game of the month). Picked on 2018-2020 log loss.
  3. score model= Poisson GLM (or LightGBM-Poisson) for E[home], E[away]; negative-binomial marginals with a
                  dispersion r estimated on the training set; dependence (negative correlation, no ties,
                  walk-offs / OT winners / empty-netters) captured by Dixon-Coles style multiplicative weights on
                  the margin and on the total, fitted by iterative proportional fitting to the training set.
                  The joint grid is then tilted so P(home wins) equals the win model's probability.
  4. outputs    = most likely score consistent with the pick, P(cover +-1.5), P(over line), Platt-calibrated
                  walk-forward on earlier out-of-sample predictions.
Also: ablation, confidence tiers, daily top-N parlays, "does more history help?" study, markdown report.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from engine.devig import american_to_prob, devig

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "predict"
TIER_EDGES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0001]
TIER_LABELS = ["50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85+"]
WIN_MODELS = ["market", "script", "lr_small", "lr_full", "lgbm", "stack"]
LGB_CLS = dict(n_estimators=350, learning_rate=0.03, num_leaves=15, min_child_samples=150, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=1)
LGB_REG = dict(LGB_CLS, objective="poisson")


# ----------------------------------------------------------------------------------------------- utils
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def tier_of(p):
    return pd.cut(np.asarray(p, float), TIER_EDGES, labels=TIER_LABELS, right=False)


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def safe_devig(a, b, max_over=1.12, min_over=0.995):
    """odds_ratio de-vig of a two-way American pair; NaN where the quote is missing or clearly broken."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) >= 100) & (np.abs(b) >= 100)
    out = np.full(len(a), np.nan)
    if ok.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            over = american_to_prob(a[ok]) + american_to_prob(b[ok])
        good = (over >= min_over) & (over <= max_over)
        idx = np.where(ok)[0][good]
        if len(idx):
            p, _ = devig(a[idx], b[idx], "odds_ratio")
            out[idx] = np.asarray(p, float)
    return out


def implied(o):
    o = np.asarray(o, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(o) & (np.abs(o) >= 100), american_to_prob(np.where(np.isfinite(o), o, 100)), np.nan)


def add_market(d: pd.DataFrame, script_col="p_home_script", script_total="total", max_gap=0.30, p_bounds=(0.03, 0.97)) -> pd.DataFrame:
    """Common market / script features. Needs home_ml, away_ml, over_odds, under_odds, total_line, spread_home."""
    d = d.copy()
    for c in ["home_ml", "away_ml", "over_odds", "under_odds", "total_line", "spread_home"]:
        if c not in d:
            d[c] = np.nan
    p = safe_devig(d.home_ml.values, d.away_ml.values)
    ps = d[script_col].astype(float).values
    broken = np.isfinite(p) & ((np.abs(p - ps) > max_gap) | (p < p_bounds[0]) | (p > p_bounds[1]))   # swapped / stale / absurd quotes
    p[broken] = np.nan
    d["mkt_broken"] = broken.astype(int)
    d["p_mkt"] = p
    d["has_mkt"] = np.isfinite(p).astype(float)
    d["s_logit"] = logit(ps)
    d["mkt_logit"] = np.where(np.isfinite(p), logit(np.nan_to_num(p, nan=0.5)), d.s_logit)
    d["mkt_gap"] = d.mkt_logit - d.s_logit          # 0 when no usable price (no 0/1 flags in linear models: they extrapolate)
    tl = d.total_line.astype(float).values
    d["has_tl"] = np.isfinite(tl).astype(float)
    d["tl"] = np.where(np.isfinite(tl), tl, d[script_total].astype(float).values)
    d["l_tl"] = np.log(np.clip(d.tl, 0.5, None))
    po = safe_devig(d.over_odds.values, d.under_odds.values, max_over=1.15)
    d["has_ou"] = (np.isfinite(po) & np.isfinite(tl)).astype(float)
    d["lo_over_mkt"] = np.where(d.has_ou > 0, logit(np.nan_to_num(po, nan=0.5)), 0.0)
    d["l_mu_home"] = np.log(np.clip(d.mu_home.astype(float), 0.2, None))
    d["l_mu_away"] = np.log(np.clip(d.mu_away.astype(float), 0.2, None))
    # vigged implied prob of each ML side (break-even)
    d["be_home"] = implied(d.home_ml.values)
    d["be_away"] = implied(d.away_ml.values)
    d["be_over"] = implied(d.over_odds.values)
    d["be_under"] = implied(d.under_odds.values)
    return d


# ------------------------------------------------------------------------------------------ config
@dataclass
class CountCfg:
    sport: str
    label: str                       # "MLB"
    unit: str                        # "runs" / "goals"
    line_name: str                   # "run line" / "puck line"
    build: Callable                  # df -> df with features (calls add_market)
    groups: dict                     # feature-group -> list of feature columns (win + score models)
    maxg: int = 25
    first_season: int = 2018
    select_seasons: tuple = (2018, 2019, 2020)
    report_from: int = 2021
    hist_from: int = 2023
    margin_bands: tuple = (1, 2)
    rl_be_fav: float = 0.42          # typical break-even of the -1.5 side
    rl_be_dog: float = 0.615         # typical break-even of the +1.5 side
    hi_tier: float = 0.65            # "high-confidence" threshold for the history study
    notes: list = field(default_factory=list)

    @property
    def market_feats(self):
        return ["mkt_logit", "mkt_gap", "lo_over_mkt", "l_tl"]

    def all_feats(self, drop=()):
        f = []
        for g, cols in self.groups.items():
            if g not in drop:
                f += cols
        if "market" not in drop:
            f = self.market_feats + f
        return list(dict.fromkeys(f))

    def score_feats(self, drop=()):
        base = ["l_mu_home", "l_mu_away"] if "script" not in drop else []
        return list(dict.fromkeys(base + self.all_feats(drop)))


# ------------------------------------------------------------------------------------------ model fits
def clipper():
    """Clip standardised features to +-5 SD: a feature whose spread changes (e.g. over/under juice that only exists
    from 2022) must not be extrapolated 400 SD by a linear model."""
    return FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0})


def _lr(C):
    return make_pipeline(StandardScaler(), clipper(), LogisticRegression(C=C, max_iter=3000))


def small_cols(feats):
    c = [x for x in ("mkt_logit", "s_logit") if x in feats]
    return c or ["s_logit"]


def fit_win(tr, feats, kind, w=None):
    y = tr.home_win.values.astype(int)
    if kind == "lr_small":
        m = _lr(1.0)
        m.fit(tr[small_cols(feats)].values, y, logisticregression__sample_weight=w)
        return m
    if kind == "lr_full":
        m = _lr(0.05)
        m.fit(tr[feats].values, y, logisticregression__sample_weight=w)
        return m
    if kind == "lgbm":
        m = lgb.LGBMClassifier(**LGB_CLS)
        m.fit(tr[feats].values, y, sample_weight=w)
        return m
    raise ValueError(kind)


def pred_win(m, d, feats, kind):
    X = d[small_cols(feats)].values if kind == "lr_small" else d[feats].values
    return m.predict_proba(X)[:, 1]


def fit_score(tr, sfeats, kind, w=None):
    ms = []
    for tgt in ["home_score", "away_score"]:
        y = tr[tgt].values.astype(float)
        if kind == "glm":
            m = make_pipeline(StandardScaler(), clipper(), PoissonRegressor(alpha=1e-3, max_iter=300, solver="newton-cholesky"))
            m.fit(tr[sfeats].values, y, poissonregressor__sample_weight=w)
        else:
            m = lgb.LGBMRegressor(**LGB_REG)
            m.fit(tr[sfeats].values, y, sample_weight=w)
        ms.append(m)
    return ms


def pred_score(ms, d, sfeats):
    X = d[sfeats].values
    return np.clip(ms[0].predict(X), 0.2, None), np.clip(ms[1].predict(X), 0.2, None)


def nb_r(mu, y):
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    den = np.sum((y - mu) ** 2 - mu)
    return float(np.sum(mu ** 2) / den) if den > 0 else np.inf


def marg_pmf(mu, r, K):
    k = np.arange(K + 1)
    if not np.isfinite(r) or r > 500:
        P = poisson.pmf(k[None, :], mu[:, None])
    else:
        P = nbinom.pmf(k[None, :], r, (r / (r + mu))[:, None])
    return P / P.sum(1, keepdims=True)


def base_grid(mu_h, mu_a, r, wm, wt, K):
    G = marg_pmf(mu_h, r, K)[:, :, None] * marg_pmf(mu_a, r, K)[:, None, :]
    k = np.arange(K + 1)
    M = k[:, None] - k[None, :] + K
    T = k[:, None] + k[None, :]
    G = G * (wm[M] * wt[T])[None]
    return G / G.sum((1, 2), keepdims=True)


def fit_dependence(mu_h, mu_a, r, yh, ya, K, iters=6, nmax=8000):
    """IPF: margin & total weights so the average predicted margin/total distribution matches the data."""
    if len(mu_h) > nmax:
        mu_h, mu_a, yh, ya = mu_h[-nmax:], mu_a[-nmax:], yh[-nmax:], ya[-nmax:]
    yh = np.clip(yh.astype(int), 0, K)
    ya = np.clip(ya.astype(int), 0, K)
    n = len(yh)
    emp_m = np.bincount(yh - ya + K, minlength=2 * K + 1).astype(float)
    emp_t = np.bincount(yh + ya, minlength=2 * K + 1).astype(float)
    wm = np.ones(2 * K + 1)
    wt = np.ones(2 * K + 1)
    k = np.arange(K + 1)
    M = (k[:, None] - k[None, :] + K).ravel()
    T = (k[:, None] + k[None, :]).ravel()
    for _ in range(iters):
        G = base_grid(mu_h, mu_a, r, wm, wt, K).sum(0).ravel()
        pm = np.bincount(M, G, minlength=2 * K + 1)
        wm *= (emp_m + 0.5) / (pm + 0.5)
        G = base_grid(mu_h, mu_a, r, wm, wt, K).sum(0).ravel()
        pt = np.bincount(T, G, minlength=2 * K + 1)
        wt *= (emp_t + 0.5) / (pt + 0.5)
    return wm, wt


def grid_stats(mu_h, mu_a, r, wm, wt, K, p_home, line_home, tl, yh=None, ya=None, chunk=4000):
    """Tilt the joint grid to p_home and read off everything we report."""
    n = len(mu_h)
    k = np.arange(K + 1)
    M = k[:, None] - k[None, :]
    T = k[:, None] + k[None, :]
    out = {c: np.zeros(n) for c in ["exp_h", "exp_a", "mode_h", "mode_a", "any_h", "any_a", "p_mode",
                                       "p_cover_home", "p_over", "p_push", "p_actual", "p_home_grid"]}
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        G = base_grid(mu_h[sl], mu_a[sl], r, wm, wt, K)
        hw = (M > 0)[None]
        aw = (M < 0)[None]
        Ph = (G * hw).sum((1, 2))
        Pa = (G * aw).sum((1, 2))
        out["p_home_grid"][sl] = Ph / (Ph + Pa)
        p = p_home[sl]
        G = G * hw * (p / np.maximum(Ph, 1e-12))[:, None, None] + G * aw * ((1 - p) / np.maximum(Pa, 1e-12))[:, None, None]
        G = G / G.sum((1, 2), keepdims=True)
        out["exp_h"][sl] = (G.sum(2) * k[None]).sum(1)
        out["exp_a"][sl] = (G.sum(1) * k[None]).sum(1)
        flat = G.reshape(len(G), -1)
        am = flat.argmax(1)
        out["any_h"][sl], out["any_a"][sl] = am // (K + 1), am % (K + 1)
        pick_home = p >= 0.5
        Gc = np.where(pick_home[:, None, None], G * hw, G * aw).reshape(len(G), -1)
        cm = Gc.argmax(1)
        out["mode_h"][sl], out["mode_a"][sl] = cm // (K + 1), cm % (K + 1)
        out["p_mode"][sl] = Gc[np.arange(len(G)), cm]
        lh = line_home[sl]
        out["p_cover_home"][sl] = (G * ((M[None] + lh[:, None, None]) > 0)).sum((1, 2))
        t = tl[sl]
        out["p_over"][sl] = (G * (T[None] > t[:, None, None])).sum((1, 2))
        out["p_push"][sl] = (G * (T[None] == t[:, None, None])).sum((1, 2))
        if yh is not None:
            hh = np.clip(np.nan_to_num(yh[sl]).astype(int), 0, K)
            aa = np.clip(np.nan_to_num(ya[sl]).astype(int), 0, K)
            out["p_actual"][sl] = G[np.arange(len(G)), hh, aa]
    return out


# ------------------------------------------------------------------------------------------ walk-forward
def make_periods(d, first_season, freq="M", last_season=None):
    pr = d[d.season >= first_season]
    if last_season is not None:
        pr = pr[pr.season <= last_season]
    key = pr.date.dt.to_period("M").astype(str) if freq == "M" else pr.season.astype(str)
    per = []
    for k, idx in pr.groupby(key, sort=True).groups.items():
        per.append((k, d.loc[idx, "date"].min(), np.asarray(idx)))
    per.sort(key=lambda x: x[1])
    return per


def train_weights(tr, season, window=None, half_life=None):
    keep = np.ones(len(tr), bool)
    if window is not None:
        keep = tr.season.values >= season - window
    w = np.ones(len(tr))
    if half_life is not None:
        w = 0.5 ** (np.clip(season - 1 - tr.season.values, 0, None) / half_life)
    return keep, w


def walk(d, cfg: CountCfg, kinds=("lr_small", "lr_full", "lgbm"), score_kinds=("glm", "lgbm"), drop=(),
         freq="M", window=None, half_life=None, first_season=None, verbose=True):
    """Pass 1: walk-forward win probabilities + score means (+ NB r and dependence weights per period)."""
    feats = cfg.all_feats(drop)
    sfeats = cfg.score_feats(drop)
    first_season = first_season or cfg.first_season
    per = make_periods(d, first_season, freq)
    res = []
    dep = {}
    stack_cols = ["lr_full", "lgbm"]
    t0 = time.time()
    for i, (k, start, idx) in enumerate(per):
        te = d.loc[idx]
        season = int(te.season.iloc[0])
        tr = d[(d.date < start) & d.home_win.notna() & d.home_score.notna()]
        keep, w = train_weights(tr, season, window, half_life)
        tr, w = tr[keep], w[keep]
        o = pd.DataFrame(index=idx)
        o["period"] = k
        for kind in kinds:
            m = fit_win(tr, feats, kind, w)
            o[kind] = pred_win(m, te, feats, kind)
        for sk in score_kinds:
            ms = fit_score(tr, sfeats, sk, w)
            mh, ma = pred_score(ms, te, sfeats)
            o[f"mu_h_{sk}"], o[f"mu_a_{sk}"] = mh, ma
            th, ta = pred_score(ms, tr, sfeats)
            r = nb_r(np.r_[th, ta], np.r_[tr.home_score.values, tr.away_score.values])
            wm, wt = fit_dependence(th, ta, r, tr.home_score.values, tr.away_score.values, cfg.maxg)
            dep[(k, sk)] = (r, wm, wt)
        # stack: logistic on earlier out-of-sample predictions
        if all(c in kinds for c in stack_cols):
            prev = pd.concat(res) if res else pd.DataFrame()
            if len(prev) >= 1500:
                pj = prev.join(d[["home_win", "mkt_logit", "has_mkt"]])
                pj = pj[pj.home_win.notna()]
                Xs = np.c_[logit(pj.lr_full), logit(pj.lgbm), pj.mkt_logit, pj.has_mkt]
                st = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, pj.home_win.astype(int))
                o["stack"] = st.predict_proba(np.c_[logit(o.lr_full), logit(o.lgbm), te.mkt_logit, te.has_mkt])[:, 1]
            else:
                o["stack"] = o["lr_full"]
        res.append(o)
        if verbose and (i % 12 == 0):
            print(f"  [{cfg.sport}] period {k} train={len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(res)
    out["market"] = np.where(d.loc[out.index, "has_mkt"] > 0, d.loc[out.index, "p_mkt"], d.loc[out.index, "p_home_script"])
    out["script"] = d.loc[out.index, "p_home_script"].values
    return out, dep


def default_line(d, p_home):
    sh = d.spread_home.astype(float).values
    has = np.isfinite(sh) & (np.abs(sh) == 1.5)
    return np.where(has, sh, np.where(p_home >= 0.5, -1.5, 1.5)), has


def pass2(d, w1, dep, cfg, win_kind, score_kind):
    """Grid outputs for the chosen win/score models + walk-forward Platt calibration of cover/over."""
    o = w1[["period"]].copy()
    o["p_home"] = w1[win_kind].values
    g_parts = []
    for k, sub in o.groupby("period", sort=False):
        r, wm, wt = dep[(k, score_kind)]
        dd = d.loc[sub.index]
        line, has = default_line(dd, sub.p_home.values)
        gs = grid_stats(w1.loc[sub.index, f"mu_h_{score_kind}"].values, w1.loc[sub.index, f"mu_a_{score_kind}"].values,
                        r, wm, wt, cfg.maxg, sub.p_home.values, line, dd.tl.values,
                        dd.home_score.values, dd.away_score.values)
        gp = pd.DataFrame(gs, index=sub.index)
        gp["line_home"], gp["has_rl"], gp["nb_r"] = line, has.astype(int), r
        g_parts.append(gp)
    o = o.join(pd.concat(g_parts))
    o["mu_h"] = w1[f"mu_h_{score_kind}"]
    o["mu_a"] = w1[f"mu_a_{score_kind}"]
    # calibration (walk-forward over periods)
    dd = d.loc[o.index]
    o["p_over_np_raw"] = o.p_over / np.clip(1 - o.p_push, 1e-9, None)
    o["p_cover_raw"] = o.p_cover_home
    margin = dd.home_score - dd.away_score
    tot = dd.home_score + dd.away_score
    o["y_cover"] = ((margin + o.line_home) > 0).astype(float)
    o["y_over"] = np.where(tot > dd.tl, 1.0, np.where(tot < dd.tl, 0.0, np.nan))
    o["p_cover_cal"] = o.p_cover_raw
    o["p_over_cal"] = o.p_over_np_raw
    periods = list(dict.fromkeys(o.period))
    for k in periods:
        cur = o.period == k
        start = dd.loc[cur, "date"].min()
        prev = o[(dd.date < start) & dd.home_score.notna()]
        if len(prev) >= 1500:
            Xc = np.c_[logit(prev.p_cover_raw), prev.line_home]
            cc = LogisticRegression(max_iter=2000).fit(Xc, prev.y_cover.astype(int))
            o.loc[cur, "p_cover_cal"] = cc.predict_proba(np.c_[logit(o.loc[cur, "p_cover_raw"]), o.loc[cur, "line_home"]])[:, 1]
            pv = prev[prev.y_over.notna() & (dd.loc[prev.index, "has_tl"] > 0)]
            if len(pv) >= 1500:
                Xo = np.c_[logit(pv.p_over_np_raw), dd.loc[pv.index, "lo_over_mkt"], dd.loc[pv.index, "has_ou"]]
                co = LogisticRegression(max_iter=2000).fit(Xo, pv.y_over.astype(int))
                cdd = dd.loc[cur]
                o.loc[cur, "p_over_cal"] = co.predict_proba(np.c_[logit(o.loc[cur, "p_over_np_raw"]), cdd.lo_over_mkt, cdd.has_ou])[:, 1]
    return o


# ------------------------------------------------------------------------------------------ reporting
def md(df, floatfmt=".3f", index=True):
    return df.to_markdown(floatfmt=floatfmt, index=index)


def tier_table(p_pick, y, season, be, from_season, extra_seasons=True):
    t = pd.DataFrame({"tier": tier_of(p_pick), "y": y, "season": season, "be": be, "p": p_pick})
    t = t[t.y.notna()]
    rows = []
    for lab in TIER_LABELS:
        s = t[t.tier == lab]
        r = {"tier": lab}
        tun = s[s.season < from_season]
        rep = s[s.season >= from_season]
        r["n_2018_20"] = len(tun)
        r["win%_2018_20"] = 100 * tun.y.mean() if len(tun) else np.nan
        if extra_seasons:
            for ss in sorted(t.season.unique()):
                if ss >= from_season:
                    r[f"n_{ss}"] = int((rep.season == ss).sum())
        r["n_2021+"] = len(rep)
        r["avg_p_2021+"] = 100 * rep.p.mean() if len(rep) else np.nan
        r["win%_2021+"] = 100 * rep.y.mean() if len(rep) else np.nan
        r["breakeven%"] = 100 * rep.be.mean() if len(rep) and rep.be.notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("tier")


def best_tier(tt, nmin=100):
    ok = tt[tt["n_2021+"] >= nmin]
    if ok.empty:
        return None
    lab = ok["win%_2021+"].idxmax()
    return lab, int(ok.loc[lab, "n_2021+"]), float(ok.loc[lab, "win%_2021+"])


def parlay_table(df, prob_col, hit_col, from_season, ns=(1, 2, 3, 4, 5), group="date"):
    t = df[df.season >= from_season].dropna(subset=[hit_col])
    rows = []
    for n in ns:
        hits, probs, days = [], [], 0
        for _, g in t.groupby(group):
            if len(g) < n:
                continue
            top = g.nlargest(n, prob_col)
            hits.append(bool(top[hit_col].all()))
            probs.append(float(np.prod(top[prob_col])))
            days += 1
        rows.append({"legs": n, "days": days, "hit%": 100 * np.mean(hits) if hits else np.nan,
                     "model_expected%": 100 * np.mean(probs) if probs else np.nan})
    return pd.DataFrame(rows).set_index("legs")


def win_metrics(p, y):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return dict(n=int(ok.sum()), acc=100 * float(((p[ok] >= 0.5) == (y[ok] == 1)).mean()), ll=logloss(p[ok], y[ok]),
                brier=float(np.mean((p[ok] - y[ok]) ** 2)))


# ------------------------------------------------------------------------------------------ main runner
def run(cfg: CountCfg, raw: pd.DataFrame):
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = cfg.build(raw).sort_values(["date", "game_id"]).reset_index(drop=True)
    d.loc[(d.home_score == d.away_score), "home_win"] = np.nan          # rare ties / suspended: no SU result
    feats = cfg.all_feats()
    sfeats = cfg.score_feats()
    print(f"[{cfg.sport}] {len(d)} games, {len(feats)} win features, {len(sfeats)} score features", flush=True)

    # ---------------- pass 1: walk-forward, monthly refits
    w1, dep = walk(d, cfg)
    ev = d.loc[w1.index]
    sel = ev.season.isin(cfg.select_seasons) & ev.home_win.notna()
    sel_tab = []
    for m in WIN_MODELS:
        r = win_metrics(w1.loc[sel, m], ev.loc[sel, "home_win"])
        r2 = win_metrics(w1.loc[~sel & (ev.season >= cfg.report_from), m], ev.loc[~sel & (ev.season >= cfg.report_from), "home_win"])
        sel_tab.append({"model": m, "ll_2018_20": r["ll"], "acc%_2018_20": r["acc"], "brier_2018_20": r["brier"],
                        "ll_2021+": r2["ll"], "acc%_2021+": r2["acc"]})
    sel_tab = pd.DataFrame(sel_tab).set_index("model")
    win_kind = sel_tab.ll_2018_20.idxmin()
    # score model choice: Poisson deviance 2018-2020
    sc_tab = []
    for sk in ["glm", "lgbm"]:
        dev = 0.0
        for side, tgt in [("h", "home_score"), ("a", "away_score")]:
            mu = w1.loc[sel, f"mu_{side}_{sk}"].values
            y = ev.loc[sel, tgt].values
            dev += 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0) - (y - mu))
        rep = ev.season >= cfg.report_from
        mae_t = np.mean(np.abs(w1.loc[rep, f"mu_h_{sk}"] + w1.loc[rep, f"mu_a_{sk}"] - ev.loc[rep, "home_score"] - ev.loc[rep, "away_score"]))
        sc_tab.append({"score_model": sk, "poisson_dev_2018_20": dev / 2, "total_MAE_2021+": mae_t})
    sc_tab = pd.DataFrame(sc_tab).set_index("score_model")
    score_kind = sc_tab.poisson_dev_2018_20.idxmin()
    print(f"[{cfg.sport}] chosen win model={win_kind}, score model={score_kind} ({time.time()-T0:.0f}s)", flush=True)

    # ---------------- pass 2: grids + calibrated cover/over
    g = pass2(d, w1, dep, cfg, win_kind, score_kind)
    P = d.loc[g.index, ["game_id", "date", "season", "home", "away", "home_score", "away_score", "home_win",
                        "p_mkt", "has_mkt", "p_home_script", "tl", "has_tl", "has_ou", "spread_home",
                        "home_ml", "away_ml", "over_odds", "under_odds", "be_home", "be_away", "be_over", "be_under",
                        "mkt_broken"]].copy()
    for m in WIN_MODELS:
        P[f"p_{m}"] = w1[m]
    P = P.join(g.drop(columns=["period"]))
    P["pick_home"] = (P.p_home >= 0.5).astype(int)
    P["pick"] = np.where(P.pick_home == 1, P.home, P.away)
    P["pick_prob"] = np.maximum(P.p_home, 1 - P.p_home)
    P["tier"] = tier_of(P.pick_prob).astype(str)
    P["pick_hit"] = np.where(P.home_win.isna(), np.nan, (P.pick_home == P.home_win).astype(float))
    P["mkt_fav_home"] = np.where(P.has_mkt > 0, (P.p_mkt >= 0.5).astype(float), np.nan)
    P["mkt_hit"] = np.where(P.home_win.isna() | P.mkt_fav_home.isna(), np.nan, (P.mkt_fav_home == P.home_win).astype(float))
    P["script_hit"] = np.where(P.home_win.isna(), np.nan, ((P.p_home_script >= 0.5) == (P.home_win == 1)).astype(float))
    P["mkt_pick_prob"] = np.where(P.has_mkt > 0, np.maximum(P.p_mkt, 1 - P.p_mkt), np.nan)
    P["be_pick"] = np.where(P.pick_home == 1, P.be_home, P.be_away)
    P["pred_home_score"] = P.mode_h.astype(int)
    P["pred_away_score"] = P.mode_a.astype(int)
    P["pred_margin"] = P.exp_h - P.exp_a
    P["pred_total"] = P.exp_h + P.exp_a
    P["exact_hit"] = ((P.pred_home_score == P.home_score) & (P.pred_away_score == P.away_score)).astype(float)
    P["exact_any_hit"] = ((P.any_h == P.home_score) & (P.any_a == P.away_score)).astype(float)
    am = P.home_score - P.away_score
    pm = P.pred_home_score - P.pred_away_score
    P["spread_pick_home"] = (P.p_cover_cal >= 0.5).astype(int)
    P["spread_prob"] = np.maximum(P.p_cover_cal, 1 - P.p_cover_cal)
    P["spread_hit"] = np.where(P.spread_pick_home == 1, P.y_cover, 1 - P.y_cover)
    pick_side_line = np.where(P.spread_pick_home == 1, P.line_home, -P.line_home)
    P["spread_pick"] = [f"{h if c == 1 else a} {l:+.1f}" for h, a, c, l in zip(P.home, P.away, P.spread_pick_home, pick_side_line)]
    P["be_spread"] = np.where(pick_side_line < 0, cfg.rl_be_fav, cfg.rl_be_dog)
    P["total_pick"] = np.where(P.p_over_cal >= 0.5, "OVER", "UNDER")
    P["total_prob"] = np.maximum(P.p_over_cal, 1 - P.p_over_cal)
    P["total_hit"] = np.where(P.y_over.isna(), np.nan, np.where(P.total_pick == "OVER", P.y_over, 1 - P.y_over))
    P["be_total"] = np.where(P.total_pick == "OVER", P.be_over, P.be_under)
    P.loc[P.be_total.isna(), "be_total"] = 110 / 210

    rep = P[P.season >= cfg.report_from]
    tun = P[P.season.isin(cfg.select_seasons)]

    # ---------------- per-season table
    rows = []
    for s, gg in P.groupby("season"):
        gm = gg[gg.has_mkt > 0]
        rows.append({"season": s, "games": int(gg.pick_hit.notna().sum()), "model%": 100 * gg.pick_hit.mean(),
                     "script%": 100 * gg.script_hit.mean(), "games_w_mkt": int(gm.mkt_hit.notna().sum()),
                     "model%_mkt_games": 100 * gm.pick_hit.mean(), "mkt_fav%": 100 * gm.mkt_hit.mean(),
                     "ll_model": logloss(gg.p_home[gg.home_win.notna()], gg.home_win.dropna()),
                     "ll_mkt": logloss(gm.p_mkt[gm.home_win.notna()], gm.home_win.dropna()) if len(gm) else np.nan,
                     "exact%": 100 * gg.exact_hit.mean(), "MAE_total": np.mean(np.abs(gg.pred_total - gg.home_score - gg.away_score))})
    season_tab = pd.DataFrame(rows).set_index("season")

    def agg_row(gg, name):
        gm = gg[gg.has_mkt > 0]
        return {"sample": name, "games": int(gg.pick_hit.notna().sum()), "model%": 100 * gg.pick_hit.mean(),
                "script%": 100 * gg.script_hit.mean(), "games_w_mkt": int(gm.mkt_hit.notna().sum()),
                "model%_mkt_games": 100 * gm.pick_hit.mean(), "mkt_fav%": 100 * gm.mkt_hit.mean(),
                "agree%": 100 * ((gm.pick_home == gm.mkt_fav_home).mean()),
                "model%_when_disagree": 100 * gm[gm.pick_home != gm.mkt_fav_home].pick_hit.mean()}
    su_sum = pd.DataFrame([agg_row(tun, "2018-2020 (selection)"), agg_row(rep, f"{cfg.report_from}+ (report)")]).set_index("sample")

    # ---------------- score accuracy
    def score_row(gg, name):
        tot = gg.home_score + gg.away_score
        mar = gg.home_score - gg.away_score
        pmg = gg.pred_home_score - gg.pred_away_score
        win_ok = gg.pick_hit == 1
        r = {"sample": name, "n": len(gg),
             "MAE_home(mean)": np.mean(np.abs(gg.exp_h - gg.home_score)), "MAE_away(mean)": np.mean(np.abs(gg.exp_a - gg.away_score)),
             "MAE_margin(mean)": np.mean(np.abs(gg.pred_margin - mar)), "MAE_total(mean)": np.mean(np.abs(gg.pred_total - tot)),
             "MAE_total(mkt line)": np.mean(np.abs(gg.tl - tot)[gg.has_tl > 0]),
             "MAE_home(score pick)": np.mean(np.abs(gg.pred_home_score - gg.home_score)),
             "MAE_total(score pick)": np.mean(np.abs(gg.pred_home_score + gg.pred_away_score - tot)),
             "exact_score%": 100 * gg.exact_hit.mean(), "exact_score%(unconstrained mode)": 100 * gg.exact_any_hit.mean(),
             "avg_p(pred score)%": 100 * gg.p_mode.mean(), "avg_p(actual score)%": 100 * gg.p_actual.mean(),
             "winner&exact_margin%": 100 * (win_ok & (pmg == mar)).mean()}
        for b in cfg.margin_bands:
            r[f"winner&margin±{b}%"] = 100 * (win_ok & ((pmg - mar).abs() <= b)).mean()
        return r
    score_tab = pd.DataFrame([score_row(tun, "2018-2020"), score_row(rep, f"{cfg.report_from}+")]).set_index("sample").T
    # baseline: always the single most common score of all earlier seasons
    mode_rows = []
    for s in sorted(rep.season.unique()):
        prior = d[(d.season < s) & d.home_score.notna()]
        top = prior.groupby(["home_score", "away_score"]).size().idxmax()
        cur = rep[rep.season == s]
        mode_rows.append(((cur.home_score == top[0]) & (cur.away_score == top[1])).sum())
    baseline_exact = 100 * sum(mode_rows) / len(rep)
    top_scores = rep.groupby(["pred_home_score", "pred_away_score"]).agg(n=("exact_hit", "size"), hit=("exact_hit", "mean")).sort_values("n", ascending=False).head(8)
    top_scores["hit"] *= 100
    top_scores.index = [f"{int(a)}-{int(b)}" for a, b in top_scores.index]
    # dependence diagnostics: correlation implied vs actual, tie share
    corr_act = float(np.corrcoef(rep.home_score, rep.away_score)[0, 1])

    # ---------------- tiers
    t_ml = tier_table(P.pick_prob.values, P.pick_hit.values, P.season.values, P.be_pick.values, cfg.report_from)
    rl = P[P.has_rl > 0]
    t_rl = tier_table(rl.spread_prob.values, rl.spread_hit.values, rl.season.values, rl.be_spread.values, cfg.report_from)
    tt = P[P.has_tl > 0]
    t_tot = tier_table(tt.total_prob.values, tt.total_hit.values, tt.season.values, tt.be_total.values, cfg.report_from)
    # market-favourite tiers for comparison
    pm_ = P[P.has_mkt > 0]
    t_mkt = tier_table(pm_.mkt_pick_prob.values, pm_.mkt_hit.values, pm_.season.values,
                       np.where(pm_.mkt_fav_home == 1, pm_.be_home, pm_.be_away), cfg.report_from, extra_seasons=False)
    bt_ml, bt_rl, bt_tot = best_tier(t_ml), best_tier(t_rl), best_tier(t_tot)
    rl_side = rl[rl.season >= cfg.report_from].assign(side=lambda x: np.where(x.be_spread < 0.5, "-1.5 side", "+1.5 side")) \
        .groupby("side").agg(n=("spread_hit", "size"), win=("spread_hit", "mean"), avg_p=("spread_prob", "mean"))
    rl_side[["win", "avg_p"]] *= 100
    ou_side = tt[(tt.season >= cfg.report_from) & tt.total_hit.notna()].groupby("total_pick").agg(n=("total_hit", "size"), win=("total_hit", "mean"))
    ou_side["win"] *= 100
    raw_ou = tt[(tt.season >= cfg.report_from) & tt.y_over.notna()]
    raw_ou_acc = 100 * np.mean((raw_ou.p_over_np_raw >= 0.5) == (raw_ou.y_over == 1))
    raw_rl = rl[rl.season >= cfg.report_from]
    raw_rl_acc = 100 * np.mean((raw_rl.p_cover_raw >= 0.5) == (raw_rl.y_cover == 1))

    # ---------------- calibration
    cal = rep[rep.home_win.notna()].assign(b=pd.cut(rep.p_home, [0, .3, .4, .45, .5, .55, .6, .7, 1])).groupby("b", observed=True) \
        .agg(n=("home_win", "size"), p_model=("p_home", "mean"), home_won=("home_win", "mean"))

    # ---------------- parlays
    par = parlay_table(P.assign(hit=P.pick_hit), "pick_prob", "hit", cfg.report_from)
    parm = parlay_table(P[P.has_mkt > 0].assign(hit=P.mkt_hit), "mkt_pick_prob", "hit", cfg.report_from)
    par = par.join(parm.rename(columns={"days": "days_mkt", "hit%": "mkt_fav_hit%", "model_expected%": "mkt_expected%"}))

    # ---------------- final fit on everything (for live predict + formula)
    full = d[d.home_win.notna() & d.home_score.notna()]
    final = {"cfg": {k: getattr(cfg, k) for k in ["sport", "label", "maxg", "rl_be_fav", "rl_be_dog"]},
             "feats": feats, "sfeats": sfeats, "win_kind": win_kind, "score_kind": score_kind, "trained_through": str(d.date.max().date())}
    if win_kind in ("lr_small", "lr_full", "lgbm"):
        final["win_model"] = fit_win(full, feats, win_kind)
    elif win_kind == "stack":
        final["lr_full"] = fit_win(full, feats, "lr_full")
        final["lgbm"] = fit_win(full, feats, "lgbm")
        pj = P[P.home_win.notna()].join(d[["mkt_logit"]])
        final["stacker"] = LogisticRegression(max_iter=2000).fit(
            np.c_[logit(pj.p_lr_full), logit(pj.p_lgbm), pj.mkt_logit, pj.has_mkt], pj.home_win.astype(int))
    if win_kind in ("lr_full", "lr_small"):
        lr_for_formula = final["win_model"]
    elif win_kind == "stack":
        lr_for_formula = final["lr_full"]
    else:
        lr_for_formula = fit_win(full, feats, "lr_full")
    ms = fit_score(full, sfeats, score_kind)
    final["score_models"] = ms
    th, ta = pred_score(ms, full, sfeats)
    r_all = nb_r(np.r_[th, ta], np.r_[full.home_score.values, full.away_score.values])
    wm, wt = fit_dependence(th, ta, r_all, full.home_score.values, full.away_score.values, cfg.maxg)
    final.update(nb_r=r_all, wm=wm, wt=wt)
    cvp = P[P.season >= cfg.first_season]
    final["cover_cal"] = LogisticRegression(max_iter=2000).fit(np.c_[logit(cvp.p_cover_raw), cvp.line_home], cvp.y_cover.astype(int))
    ov = cvp[cvp.y_over.notna() & (cvp.has_tl > 0)].join(d[["lo_over_mkt"]])
    final["over_cal"] = LogisticRegression(max_iter=2000).fit(np.c_[logit(ov.p_over_np_raw), ov.lo_over_mkt, ov.has_ou], ov.y_over.astype(int))
    glm_formula = ms if score_kind == "glm" else fit_score(full, sfeats, "glm")
    with open(OUT / f"{cfg.sport}_model.pkl", "wb") as f:
        pickle.dump(final, f)

    # ---------------- formula
    lr = lr_for_formula
    sc = lr.named_steps["standardscaler"]
    co = lr.named_steps["logisticregression"].coef_[0]
    lr_intercept = float(lr.named_steps["logisticregression"].intercept_[0] - np.sum(co * sc.mean_ / sc.scale_))
    ffeats = small_cols(feats) if win_kind == "lr_small" else feats
    fo = pd.DataFrame({"feature": ffeats, "coef_per_SD": co, "coef_raw_unit": co / sc.scale_, "SD": sc.scale_}) \
        .assign(a=lambda x: x.coef_per_SD.abs()).sort_values("a", ascending=False).drop(columns="a").head(14).set_index("feature")
    gsc = glm_formula[0].named_steps["standardscaler"]
    gco = glm_formula[0].named_steps["poissonregressor"].coef_
    gfo = pd.DataFrame({"feature": sfeats, "home_coef_per_SD": gco, "away_coef_per_SD": glm_formula[1].named_steps["poissonregressor"].coef_,
                        "home_coef_raw": gco / gsc.scale_}).assign(a=lambda x: x.home_coef_per_SD.abs() + x.away_coef_per_SD.abs()) \
        .sort_values("a", ascending=False).drop(columns="a").head(12).set_index("feature")
    imp = None
    if win_kind in ("lgbm", "stack"):
        lg = final.get("win_model") if win_kind == "lgbm" else final["lgbm"]
        imp = pd.Series(lg.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False).head(12)
        imp = (100 * imp / imp.sum()).rename("gain_share%").to_frame()
    stack_coef = None
    if win_kind == "stack":
        stack_coef = dict(zip(["logit(lr_full)", "logit(lgbm)", "market logit", "has_market"], final["stacker"].coef_[0]))
    K = cfg.maxg
    dep_tab = pd.DataFrame({"margin": np.arange(-4, 5), "weight": wm[np.arange(-4, 5) + K] / wm[K + 1]}).set_index("margin").T

    # ---------------- ablation (season refits)
    print(f"[{cfg.sport}] ablation ({time.time()-T0:.0f}s)", flush=True)
    abl_kinds = tuple(dict.fromkeys(([win_kind] if win_kind in ("lr_small", "lr_full", "lgbm") else []) + ["lr_full", "lgbm"]))
    abl = []
    variants = [("all features", ())] + [(f"drop {gname}", (gname,)) for gname in ["market"] + list(cfg.groups)]
    for name, drop in variants:
        if name.startswith("drop market") and False:
            pass
        wa, depa = walk(d, cfg, kinds=abl_kinds, score_kinds=("glm",), drop=drop, freq="S", verbose=False)
        pk = win_kind if win_kind in wa.columns else ("stack" if "stack" in wa.columns else "lr_full")
        ea = d.loc[wa.index]
        m = ea.season >= cfg.report_from
        wmx = win_metrics(wa.loc[m, pk], ea.loc[m, "home_win"])
        mt = np.mean(np.abs(wa.loc[m, "mu_h_glm"] + wa.loc[m, "mu_a_glm"] - ea.loc[m, "home_score"] - ea.loc[m, "away_score"]))
        mm = np.mean(np.abs(wa.loc[m, "mu_h_glm"] - wa.loc[m, "mu_a_glm"] - ea.loc[m, "home_score"] + ea.loc[m, "away_score"]))
        wf = win_metrics(wa.loc[m, "lr_full"], ea.loc[m, "home_win"])
        abl.append({"variant": name, "features": len(cfg.all_feats(drop)), "SU%_2021+": wmx["acc"], "logloss_2021+": wmx["ll"],
                    "MAE_total_2021+": mt, "MAE_margin_2021+": mm, "lr_full_SU%": wf["acc"], "lr_full_logloss": wf["ll"]})
    abl = pd.DataFrame(abl).set_index("variant")
    abl["dSU_vs_all"] = abl["SU%_2021+"] - abl.loc["all features", "SU%_2021+"]
    abl["dLL_vs_all(+ = group helps)"] = abl["logloss_2021+"] - abl.loc["all features", "logloss_2021+"]
    abl["lr_full_dLL(+ = helps)"] = abl["lr_full_logloss"] - abl.loc["all features", "lr_full_logloss"]
    # script-only and market-only reference rows
    m = rep.home_win.notna()
    ref_m = win_metrics(rep.p_market[m], rep.home_win[m])
    ref_s = win_metrics(rep.p_script[m], rep.home_win[m])
    abl.loc["market only (no model)"] = pd.Series({"SU%_2021+": ref_m["acc"], "logloss_2021+": ref_m["ll"]})
    abl.loc["script only (no model)"] = pd.Series({"SU%_2021+": ref_s["acc"], "logloss_2021+": ref_s["ll"],
                                                   "MAE_total_2021+": np.mean(np.abs(d.loc[rep.index, "total"] - rep.home_score - rep.away_score))})

    # ---------------- does more history help? (eval 2023+, season refits)
    print(f"[{cfg.sport}] history study ({time.time()-T0:.0f}s)", flush=True)
    hist = history_study(d, cfg, win_kind, score_kind)

    # ---------------- learning curve within the walk-forward (per season log loss already) -> write report
    P_out = P.drop(columns=["any_h", "any_a"])
    P_out.to_parquet(OUT / f"{cfg.sport}_preds.parquet", index=False)

    rep_md = render_report(cfg, d, P, sel_tab, win_kind, sc_tab, score_kind, season_tab, su_sum, score_tab, baseline_exact,
                           top_scores, corr_act, t_ml, t_rl, t_tot, t_mkt, bt_ml, bt_rl, bt_tot, rl_side, ou_side, raw_ou_acc, raw_rl_acc,
                           cal, par, fo, gfo, imp, stack_coef, dep_tab, r_all, abl, hist, T0, lr_intercept)
    (OUT / f"{cfg.sport}_predict.md").write_text(rep_md)
    summary = {"sport": cfg.label, "SU_model_2021+": su_sum.iloc[1]["model%_mkt_games"], "SU_mkt_2021+": su_sum.iloc[1]["mkt_fav%"],
               "SU_model_all_2021+": su_sum.iloc[1]["model%"], "exact%": score_tab.loc["exact_score%"].iloc[1],
               "best_tier": bt_ml, "parlay3": par.loc[3, "hit%"], "rl_best": bt_rl, "ou_best": bt_tot,
               "MAE_total": score_tab.loc["MAE_total(mean)"].iloc[1], "win_kind": win_kind, "score_kind": score_kind}
    print(summary, f"{time.time()-T0:.0f}s", flush=True)
    return summary


def history_study(d, cfg, win_kind, score_kind):
    kinds = ("lr_full", "lgbm") if win_kind == "stack" else tuple(dict.fromkeys(((win_kind,) if win_kind in ("lr_small", "lr_full", "lgbm") else ()) + ("lr_full",)))
    pk_default = win_kind if win_kind in ("lr_small", "lr_full", "lgbm", "stack") else "lr_full"
    rows = []
    variants = [(f"last {w} season(s)", dict(window=w)) for w in (1, 2, 4, 6)] + [("all prior seasons (from 2016)", {})] + \
               [(f"all, recency half-life {h} season(s)", dict(half_life=h)) for h in (1, 2, 4)]
    for name, kw in variants:
        # stack needs earlier OOS predictions -> start the walk 2 seasons before the evaluation window
        wa, _ = walk(d, cfg, kinds=kinds, score_kinds=(score_kind,), freq="S", first_season=cfg.hist_from - 3, verbose=False, **kw)
        pk = pk_default if pk_default in wa.columns else "lr_full"
        ea = d.loc[wa.index]
        m = (ea.season >= cfg.hist_from) & ea.home_win.notna()
        p = wa.loc[m, pk].values
        y = ea.loc[m, "home_win"].values
        pp = np.maximum(p, 1 - p)
        hit = ((p >= 0.5) == (y == 1))
        hi = pp >= cfg.hi_tier
        mh, ma = wa.loc[m, f"mu_h_{score_kind}"].values, wa.loc[m, f"mu_a_{score_kind}"].values
        hs, as_ = ea.loc[m, "home_score"].values, ea.loc[m, "away_score"].values
        rows.append({"training data": name, "n_eval": int(m.sum()), "SU%": 100 * hit.mean(), "logloss": logloss(p, y),
                     f"MAE_{cfg.unit}_per_side": np.mean(np.r_[np.abs(mh - hs), np.abs(ma - as_)]),
                     "MAE_total": np.mean(np.abs(mh + ma - hs - as_)),
                     f"n_pick≥{int(cfg.hi_tier*100)}%": int(hi.sum()), f"win%_pick≥{int(cfg.hi_tier*100)}%": 100 * hit[hi].mean() if hi.any() else np.nan,
                     "lr_full_SU%": 100 * np.mean((wa.loc[m, "lr_full"].values >= 0.5) == (y == 1)),
                     "lr_full_logloss": logloss(wa.loc[m, "lr_full"].values, y)})
    return pd.DataFrame(rows).set_index("training data")


def history_verdict(hist, unit_col):
    w = hist.iloc[:5]
    best = w.logloss.idxmin()
    bf = w.lr_full_logloss.idxmin()
    rec = hist.iloc[5:]
    s = (f"Best plain window by log loss: **{best}** (SU {w.loc[best,'SU%']:.1f}%). "
         f"Last-1-season vs all: SU {w.iloc[0]['SU%']:.1f}% -> {w.iloc[4]['SU%']:.1f}%, log loss {w.iloc[0]['logloss']:.4f} -> {w.iloc[4]['logloss']:.4f}, "
         f"{unit_col} {w.iloc[0][unit_col]:.3f} -> {w.iloc[4][unit_col]:.3f}. ")
    gain_4_all = w.iloc[2].logloss - w.iloc[4].logloss
    s += ("More history keeps improving log loss beyond 4 seasons. " if gain_4_all > 0.0005 else
          "Gains plateau after ~2-4 seasons (4 seasons -> all changes log loss by "
          f"{-gain_4_all:+.4f}); differences in SU% of a few tenths are inside noise (SE ≈ {100*np.sqrt(0.24/w.n_eval.iloc[0]):.1f} pts). ")
    s += (f"Full logistic: best window **{bf}**; last 1 -> all: SU {w.iloc[0]['lr_full_SU%']:.1f}% -> {w.iloc[4]['lr_full_SU%']:.1f}%, "
          f"log loss {w.iloc[0]['lr_full_logloss']:.4f} -> {w.iloc[4]['lr_full_logloss']:.4f}. ")
    rb = rec.logloss.idxmin()
    s += (f"Recency weighting: best is **{rb}** (log loss {rec.loc[rb,'logloss']:.4f} vs unweighted {w.iloc[4]['logloss']:.4f}) - "
          + ("it helps slightly." if rec.loc[rb, "logloss"] < w.iloc[4]["logloss"] - 0.0003 else "no real gain over equal weights."))
    return s


def render_report(cfg, d, P, sel_tab, win_kind, sc_tab, score_kind, season_tab, su_sum, score_tab, baseline_exact, top_scores,
                  corr_act, t_ml, t_rl, t_tot, t_mkt, bt_ml, bt_rl, bt_tot, rl_side, ou_side, raw_ou_acc, raw_rl_acc, cal, par,
                  fo, gfo, imp, stack_coef, dep_tab, r_all, abl, hist, T0, lr_intercept=None):
    L = []
    rf = cfg.report_from
    rep = P[P.season >= rf]
    L.append(f"# {cfg.label} game predictor: winner + final score\n")
    L.append(f"Generated by `python -m engine.predict.{cfg.sport}` ({(time.time()-T0)/60:.1f} min). Strict walk-forward: "
             f"every month from {cfg.first_season} the models are refit on all games dated before that month (2016 onward) "
             f"and predict the month. Model picked on {cfg.select_seasons[0]}-{cfg.select_seasons[-1]} log loss; numbers reported "
             f"for {rf}+ ({len(rep):,} games) were never used to pick anything.\n")
    for n in cfg.notes:
        L.append(f"* {n}")
    nb = int(P.mkt_broken.sum())
    L.append(f"* Market: closing moneyline de-vigged with `engine.devig.devig(a, b, 'odds_ratio')`. Quotes dropped as broken: "
             f"missing, |odds|<100, overround outside 0.995-1.12, de-vigged price outside the sport's plausible range or >30 pts "
             f"away from the script ({nb} dropped by the last two rules in the predicted seasons). Games without a usable price fall back to the script (flag `has_mkt`=0).\n")

    L.append("## Headline (2021+)\n")
    s = su_sum.iloc[1]
    L.append(f"| metric | value |\n|---|---|\n| SU winner, model (all games) | {s['model%']:.2f}% of {int(s['games']):,} |\n"
             f"| SU winner, model vs market favourite (games with a usable price) | {s['model%_mkt_games']:.2f}% vs {s['mkt_fav%']:.2f}% (n={int(s['games_w_mkt']):,}) |\n"
             f"| SU winner, script alone | {s['script%']:.2f}% |\n"
             f"| Exact final score (score consistent with pick) | {score_tab.loc['exact_score%'].iloc[1]:.2f}% (always-most-common-score baseline {baseline_exact:.2f}%) |\n"
             f"| MAE total {cfg.unit} (model mean / market line) | {score_tab.loc['MAE_total(mean)'].iloc[1]:.3f} / {score_tab.loc['MAE_total(mkt line)'].iloc[1]:.3f} |\n"
             + (f"| Best ML tier (≥100 games) | {bt_ml[0]}: {bt_ml[2]:.1f}% (n={bt_ml[1]}) |\n" if bt_ml else "")
             + f"| Daily top-1 / top-2 / top-3 parlay | {par.loc[1,'hit%']:.1f}% / {par.loc[2,'hit%']:.1f}% / {par.loc[3,'hit%']:.1f}% |\n"
             + (f"| Best {cfg.line_name} tier | {bt_rl[0]}: {bt_rl[2]:.1f}% (n={bt_rl[1]}) |\n" if bt_rl else "")
             + (f"| Best O/U tier | {bt_tot[0]}: {bt_tot[2]:.1f}% (n={bt_tot[1]}) |\n" if bt_tot else ""))

    L.append("## Model selection (win probability)\n")
    L.append(f"Chosen: **{win_kind}** (lowest 2018-2020 log loss). `market` = de-vigged closing ML (script where missing); "
             "`script` = Kalman script probability; `lr_small` = logistic(market logit, script logit); `lr_full` = L2 logistic on all "
             "features; `lgbm` = LightGBM; `stack` = logistic on [lr_full, lgbm, market] trained on earlier out-of-sample predictions.\n")
    L.append(md(sel_tab, ".4f"))
    L.append(f"\nScore-mean model: **{score_kind}** (Poisson deviance per game, 2018-2020):\n")
    L.append(md(sc_tab, ".4f"))

    L.append("\n## 1. Straight-up winner accuracy per season\n")
    L.append("`model%` all games; `model%_mkt_games` / `mkt_fav%` on games with a usable closing price; `script%` = script alone.\n")
    L.append(md(season_tab, ".3f"))
    L.append("\n" + md(su_sum, ".2f"))
    L.append(f"\nCalibration of the model's P(home win), {rf}+:\n")
    L.append(md(cal, ".3f"))

    L.append("\n## 2. Score accuracy\n")
    L.append(f"Joint score model: E[home], E[away] from the {score_kind.upper()} Poisson-mean model; negative-binomial marginals "
             f"(dispersion r = {r_all:.1f} on the full fit; {'≈ Poisson' if not np.isfinite(r_all) or r_all > 100 else 'over-dispersed'}); "
             f"dependence = learned multiplicative weights on margin and total (IPF), which removes ties, captures the actual "
             f"home/away correlation ({corr_act:+.3f} in {rf}+) and the extra-innings / OT / empty-net margin shape; grid tilted to the "
             f"model's P(home win). `pred score` = most likely cell among cells where the predicted winner wins.\n")
    L.append("Margin weights relative to a 1-goal/run home win (full fit):\n")
    L.append(md(dep_tab, ".3f"))
    L.append("\n" + md(score_tab, ".3f"))
    L.append(f"\nMost frequent predicted scores {rf}+ (n, hit%):\n")
    L.append(md(top_scores, ".1f"))

    L.append(f"\n## 3. Win rate by confidence tier\n")
    L.append(f"Tier = model probability of the side picked. `breakeven%` = vigged implied probability of the picked side at the "
             f"closing price (ML, totals); for the {cfg.line_name} (no prices in the data) a typical price is assumed: "
             f"-1.5 side {cfg.rl_be_fav:.0%}, +1.5 side {cfg.rl_be_dog:.0%}.\n")
    L.append("### (a) Moneyline winner\n")
    L.append(md(t_ml, ".1f"))
    L.append("\nMarket favourite by its own de-vigged probability (comparison):\n")
    L.append(md(t_mkt, ".1f"))
    L.append(f"\n### (b) {cfg.line_name} (±1.5) pick from the score distribution (calibrated walk-forward)\n")
    L.append(md(t_rl, ".1f"))
    L.append(f"\nBy side ({rf}+):\n\n" + md(rl_side, ".1f") + f"\n\nUncalibrated grid pick accuracy {rf}+: {raw_rl_acc:.1f}%.\n")
    L.append("### (c) Total (over/under the closing total; pushes excluded)\n")
    L.append(md(t_tot, ".1f"))
    L.append(f"\nBy side ({rf}+):\n\n" + md(ou_side, ".1f") + f"\n\nUncalibrated grid O/U accuracy {rf}+: {raw_ou_acc:.1f}%.\n")
    if bt_ml:
        L.append(f"\n**Highest-win% tier with ≥100 games {rf}+: ML {bt_ml[0]} → {bt_ml[2]:.1f}% (n={bt_ml[1]})"
                 + (f"; {cfg.line_name} {bt_rl[0]} → {bt_rl[2]:.1f}% (n={bt_rl[1]})" if bt_rl else "")
                 + (f"; totals {bt_tot[0]} → {bt_tot[2]:.1f}% (n={bt_tot[1]})" if bt_tot else "") + ".**\n")

    L.append(f"\n## 4. Daily top-N confidence parlays ({rf}+)\n")
    L.append("Each day, the N most confident ML picks parlayed (days with ≥N games). Compared with the N biggest market favourites.\n")
    L.append(md(par, ".1f"))

    L.append("\n## 5. The formula (plain English)\n")
    L.append(f"Win model **{win_kind}**. " + ("Logistic part (full-data fit)" if win_kind != "lr_small" else "Its logistic fit on all games")
             + (f", intercept {lr_intercept:+.3f}" if lr_intercept is not None else "") + "; terms per one standard deviation and per raw unit:\n")
    L.append(md(fo, ".4f"))
    top = fo.head(5)
    L.append("\nIn words: logit P(home wins) ≈ " + " + ".join(f"{v:+.3f}×{k}" for k, v in top.coef_raw_unit.items())
             + " + (smaller terms)" + (f" {lr_intercept:+.3f}" if lr_intercept is not None else "") + ". `mkt_logit` = log-odds of the de-vigged closing moneyline, `s_logit` = script log-odds.\n")
    if stack_coef:
        L.append("Stack weights: " + ", ".join(f"{k} {v:+.3f}" for k, v in stack_coef.items()) + "\n")
    if imp is not None:
        L.append("LightGBM gain share (top features):\n\n" + md(imp, ".1f") + "\n")
    L.append(f"Score means (Poisson GLM, log link): log E[home {cfg.unit}] and log E[away {cfg.unit}] = intercept + Σ coef×feature, largest terms:\n")
    L.append(md(gfo, ".4f"))
    L.append(f"\nSo predicted {cfg.unit} ≈ the script's expected {cfg.unit} pulled toward the closing total (`l_tl`) and the "
             "moneyline-implied strength, then the NB grid turns the two means into a full score distribution.\n")

    L.append(f"\n## 6. Ablation ({rf}+, season refits, drop one feature group)\n")
    L.append(f"Chosen model **{win_kind}** (SU%/logloss columns) and, because a small model ignores most groups, the full logistic "
             "(`lr_full_*` columns) re-run without each group. Positive dLL = removing the group made log loss worse (the group helps). "
             "Score MAE columns come from the Poisson GLM re-fit without the group.\n")
    L.append(md(abl, ".4f"))
    L.append("\nFeature groups: " + "; ".join(f"**{g}**: {', '.join(c)}" for g, c in cfg.groups.items())
             + f"; **market**: {', '.join(cfg.market_feats)}.\n")

    L.append(f"\n## Does more history help?\n")
    L.append(f"Evaluation fixed at {cfg.hist_from}+ seasons (season refits; for season S the model sees only seasons < S, restricted "
             f"to the last k seasons, or all seasons with exponential recency weights 0.5^(age/half-life)). Data start 2016, so "
             f"'last 6' and 'all' differ only by the 2016 season for {cfg.hist_from}. High-confidence tier = pick probability ≥{cfg.hi_tier:.0%}.\n")
    L.append(f"Columns without a prefix = the chosen model (**{win_kind}**, score model {score_kind}); `lr_full_*` = the all-feature "
             "logistic, which has many more parameters and is the better test of whether more data helps.\n")
    L.append(md(hist, ".4f"))
    L.append("\n" + history_verdict(hist, f"MAE_{cfg.unit}_per_side") + "\n")

    L.append(f"\n## Learning curve per season (model log loss vs market)\n")
    L.append("The per-season table in section 1 is the walk-forward learning curve (each season predicted by a model that has "
             "seen every earlier game; monthly refits keep adding the current season).\n")

    L.append(f"\n## Live use\n")
    L.append(f"`engine.predict.{cfg.sport}.predict(slate_df)` loads `results/predict/{cfg.sport}_model.pkl` (fit on all games "
             f"through {d.date.max().date()}). Slate needs the script columns (run `engine.scripts.{cfg.sport}` for the slate) "
             "plus home_ml, away_ml, total_line, over_odds, under_odds, spread_home (prices optional; missing prices fall back to the script).\n")
    L.append(f"Files: `results/predict/{cfg.sport}_preds.parquet` (every predicted game, walk-forward), `{cfg.sport}_model.pkl`.\n")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------ live predict
def predict_count(model: dict, feat_df: pd.DataFrame) -> pd.DataFrame:
    """feat_df = cfg.build(slate) output."""
    d = feat_df
    feats, sfeats, wk = model["feats"], model["sfeats"], model["win_kind"]
    if wk == "market":
        p = np.where(d.has_mkt > 0, d.p_mkt, d.p_home_script)
    elif wk == "script":
        p = d.p_home_script.values
    elif wk == "stack":
        a = model["lr_full"].predict_proba(d[feats].values)[:, 1]
        b = model["lgbm"].predict_proba(d[feats].values)[:, 1]
        p = model["stacker"].predict_proba(np.c_[logit(a), logit(b), d.mkt_logit, d.has_mkt])[:, 1]
    else:
        p = pred_win(model["win_model"], d, feats, wk)
    mh, ma = pred_score(model["score_models"], d, sfeats)
    K = model["cfg"]["maxg"]
    line, has = default_line(d, p)
    gs = grid_stats(mh, ma, model["nb_r"], model["wm"], model["wt"], K, p, line, d.tl.values)
    pc = model["cover_cal"].predict_proba(np.c_[logit(gs["p_cover_home"]), line])[:, 1]
    pnp = gs["p_over"] / np.clip(1 - gs["p_push"], 1e-9, None)
    po = model["over_cal"].predict_proba(np.c_[logit(pnp), d.lo_over_mkt, d.has_ou])[:, 1]
    pick_home = p >= 0.5
    side_line = np.where(pc >= 0.5, line, -line)
    out = pd.DataFrame({
        "home": d.home.values, "away": d.away.values, "p_home": p,
        "pick": np.where(pick_home, d.home, d.away), "pick_prob": np.maximum(p, 1 - p),
        "pred_home_score": gs["mode_h"].astype(int), "pred_away_score": gs["mode_a"].astype(int),
        "pred_margin": gs["exp_h"] - gs["exp_a"], "pred_total": gs["exp_h"] + gs["exp_a"],
        "spread_pick": [f"{h if c >= 0.5 else a} {l:+.1f}" for h, a, c, l in zip(d.home, d.away, pc, side_line)],
        "spread_prob": np.maximum(pc, 1 - pc),
        "total_pick": np.where(po >= 0.5, "OVER", "UNDER"), "total_prob": np.maximum(po, 1 - po),
    }, index=d.index)
    out["total_line"] = d.tl.values
    out["p_exact_score"] = gs["p_mode"]
    out["tier"] = tier_of(out.pick_prob).astype(str)
    return out
