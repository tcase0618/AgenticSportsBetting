"""
NCAAM adapter for the script-then-learn engine (see SPEC.md).

    python -m engine.scripts.ncaam      -> results/scripts/ncaam_games.parquet, ncaam_learning.md

No player box exists, so all enrichments come from the scores themselves (plus schedule / conference / neutral):
  CORE    base.ScriptEngine Kalman off/def (driven date-by-date: all games of a date scripted before any is learned).
          `FastEngine` subclasses it with an identical-maths but O(1) noise/correlation update (ring-buffer sums
          instead of re-building a 2000-row array every game) so the 70k-game loop is fast.
  TEMPO   a multiplicative tempo/efficiency Kalman on log points:
             log pts_h = l + tempo_h + tempo_a + off_h + def_a (+/- hfa)
          tempo moves BOTH scores, off/def only one, so the shared part of the score error teaches tempo and the
          split part teaches efficiency.  -> mult-model points, tempo clash.
  FORM    EWMA (shifted) points for / against per team -> EWMA-implied points vs the core.
  CONF    conference effects learned online from NON-conference games: running mean of the core's margin and total
          errors by conference (decays 50% at each new season), same-conference flag, conference x early-season.
  SCHED   rest days, neutral site, postseason flag.
  LEARN   online RLS correction of each team's scoring error from those features; learned market blend
          (Bates-Granger, margin & total separately) and an online market-residual learner.
Tuning on 2016-2020; 2021+ reported (2024 and 2025 have no lines in the data -> market comparisons skip them).
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

from engine.scripts.base import Params, ScriptEngine
from engine.scripts.nba import RLS, FastEngine, mae, md_table, ats_roi, ml_logloss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "scripts"
WARM, START, TUNE_END, REPORT = 2013, 2016, 2020, 2021
CORE_PARAMS = dict(init_level=70.0, init_hfa=3.5, init_sd=10.0, prior_var=40.0, q_game=0.1, carry=0.7,
                   season_var=0.0, lr_level=0.002)


def load_games() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data" / "processed" / "ncaam.csv", low_memory=False)
    g = g[(g.season >= WARM)].dropna(subset=["home_score", "away_score"]).copy()
    g["date"] = pd.to_datetime(g["date"])
    g["game_id"] = g.espn_game_id
    g = g.drop_duplicates("game_id")
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def run_core(g, cls=FastEngine, **kw):
    p = dict(CORE_PARAMS); p.update(kw)
    eng = cls(Params(**p))
    out = []
    for _, blk in g.groupby("date", sort=True):
        sc_list = []
        for r in blk.itertuples(index=False):
            neutral = bool(r.neutral)
            sc = eng.script(r.home, r.away, int(r.season), neutral)
            mm = -r.spread_home if r.spread_home == r.spread_home else np.nan
            b = eng.blend(sc, mm, r.total)
            sc_list.append((r, sc, mm, neutral))
            out.append(dict(game_id=r.game_id, **sc, margin_b_core=b["margin_b"], total_b_core=b["total_b"],
                            w_mkt_core=b["w_mkt"], hfa=eng.hfa, level=eng.m, noise_sd=eng.sd, score_rho=eng.rho,
                            games_home=eng.teams[r.home].games, games_away=eng.teams[r.away].games,
                            x_home=eng.teams[r.home].x.copy(), x_away=eng.teams[r.away].x.copy()))
        for r, sc, mm, neutral in sc_list:
            eng.learn_blend(sc, mm, r.total, r.home_score - r.away_score, r.home_score + r.away_score)
            eng.learn(r.home, r.away, int(r.season), float(r.home_score), float(r.away_score), neutral)
    c = pd.DataFrame(out).rename(columns={"margin": "margin_core", "total": "total_core", "mu_home": "mu_home_core",
                                           "mu_away": "mu_away_core"})
    return c


_G = None


def _core_job(kw):
    return run_core(_G, **kw)


def parallel_core(g, grid, workers=4):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    global _G
    _G = g
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_core_job, grid))


# ------------------------------------------------------------------ score-based enrichments (one sequential pass)
def enrich(g: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Pre-game tempo/efficiency (multiplicative Kalman), EWMA form, conference residuals, rest. Date-batched."""
    t0 = time.time()
    core = C.set_index("game_id")
    tstate = {}                                                  # team -> [x(3), P(3x3), season]
    l, hl = math.log(70.0), 0.025                                # log level, log home adv
    obs_var, q = 0.012, 2e-4
    prior = np.diag([0.004, 0.01, 0.01])
    ew = {}                                                      # team -> [pf, pa, season]
    a_ew = 1 - 0.5 ** (1 / 6)
    conf_m, conf_t, conf_season = defaultdict(float), defaultdict(float), {}
    last_date = {}
    rows = []
    Hm = np.array([[1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0]], float)  # state [t_h,o_h,d_h,t_a,o_a,d_a]
    # y_h = l + t_h + t_a + o_h + d_a ; y_a = l + t_h + t_a + o_a + d_h
    Hm = np.array([[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0]], float)
    cur_season = None
    for _, blk in g.groupby("date", sort=True):
        pend = []
        for r in blk.itertuples(index=False):
            s = int(r.season)
            if s != cur_season:
                cur_season = s
                for c in list(conf_m):
                    conf_m[c] *= 0.5; conf_t[c] *= 0.5
            st = []
            for tm in (r.home, r.away):
                v = tstate.get(tm)
                if v is None:
                    v = [np.zeros(3), prior.copy(), s]; tstate[tm] = v
                elif v[2] != s:
                    v[0] = v[0] * 0.7; v[1] = v[1] * 0.49 + prior * 0.51; v[2] = s
                v[1] = v[1] + np.eye(3) * q
                st.append(v)
            x = np.r_[st[0][0], st[1][0]]
            P = np.zeros((6, 6)); P[:3, :3] = st[0][1]; P[3:, 3:] = st[1][1]
            adv = 0.0 if r.neutral else hl
            eta = Hm @ x + l + np.array([adv, -adv])
            vh = Hm @ P @ Hm.T
            mu_mult = np.exp(eta + 0.5 * (np.diag(vh) + obs_var))
            e = []
            for tm in (r.home, r.away):
                v = ew.get(tm)
                e.append((np.nan, np.nan) if v is None else (v[0], v[1]))
            ewm_h = (e[0][0] + e[1][1]) / 2                       # home pts for & away pts allowed
            ewm_a = (e[1][0] + e[0][1]) / 2
            ch, ca = r.home_conference, r.away_conference
            same = float(isinstance(ch, str) and ch == ca)
            rest = [min((r.date - last_date[tm]).days, 7) if tm in last_date else 7 for tm in (r.home, r.away)]
            rows.append(dict(game_id=r.game_id, mult_home=mu_mult[0], mult_away=mu_mult[1],
                             tempo_home=st[0][0][0], tempo_away=st[1][0][0], ewm_home=ewm_h, ewm_away=ewm_a,
                             conf_m_home=conf_m.get(ch, 0.0) if isinstance(ch, str) else 0.0,
                             conf_m_away=conf_m.get(ca, 0.0) if isinstance(ca, str) else 0.0,
                             conf_t_home=conf_t.get(ch, 0.0) if isinstance(ch, str) else 0.0,
                             conf_t_away=conf_t.get(ca, 0.0) if isinstance(ca, str) else 0.0,
                             same_conf=same, rest_home=rest[0], rest_away=rest[1]))
            pend.append((r, st, x, P, eta, same))
        for r, st, x, P, eta, same in pend:
            y = np.log(np.maximum([r.home_score, r.away_score], 20.0))
            Rm = np.eye(2) * obs_var
            S = Hm @ P @ Hm.T + Rm
            K = P @ Hm.T @ np.linalg.inv(S)
            inn = y - eta
            x = x + K @ inn
            P = (np.eye(6) - K @ Hm) @ P
            st[0][0], st[1][0] = x[:3], x[3:]
            st[0][1], st[1][1] = P[:3, :3], P[3:, 3:]
            l += 0.002 * inn.mean()
            if not r.neutral:
                hl += 0.001 * (inn[0] - inn[1])
            for tm, pf, pa in ((r.home, r.home_score, r.away_score), (r.away, r.away_score, r.home_score)):
                v = ew.get(tm)
                if v is None:
                    ew[tm] = [pf, pa, r.season]
                else:
                    v[0] = (1 - a_ew) * v[0] + a_ew * pf; v[1] = (1 - a_ew) * v[1] + a_ew * pa
                last_date[tm] = r.date
            ch, ca = r.home_conference, r.away_conference
            if isinstance(ch, str) and isinstance(ca, str) and not same:
                c = core.loc[r.game_id]
                em = (r.home_score - r.away_score) - c.margin_core
                et = (r.home_score + r.away_score) - c.total_core
                conf_m[ch] += 0.01 * (em / 2 - conf_m[ch]); conf_m[ca] += 0.01 * (-em / 2 - conf_m[ca])
                conf_t[ch] += 0.01 * (et / 2 - conf_t[ch]); conf_t[ca] += 0.01 * (et / 2 - conf_t[ca])
    E = pd.DataFrame(rows)
    print(f"  enrichments ({time.time() - t0:.0f}s)")
    return E


GROUPS = {
    "base": ["bias", "is_home", "neutral", "playoff"],
    "tempo": ["d_mult", "tempo_clash", "tempo_sum"],
    "form": ["d_ewm"],
    "conf": ["conf_m_T", "conf_m_O", "conf_t_T", "conf_t_O", "same_conf", "conf_early"],
    "rest": ["rest_T", "rest_O"],
}


def side_X(d: pd.DataFrame, side: str) -> pd.DataFrame:
    o = "away" if side == "home" else "home"
    X = pd.DataFrame(index=d.index)
    X["bias"] = 1.0
    X["is_home"] = float(side == "home") * (1 - d.neutral)
    X["neutral"] = d.neutral.astype(float)
    X["playoff"] = d.playoff.astype(float)
    X["d_mult"] = d[f"mult_{side}"] - d[f"mu_{side}_core"]
    X["tempo_clash"] = d.tempo_home * d.tempo_away
    X["tempo_sum"] = d.tempo_home + d.tempo_away
    X["d_ewm"] = d[f"ewm_{side}"] - d[f"mu_{side}_core"]
    X["conf_m_T"], X["conf_m_O"] = d[f"conf_m_{side}"], d[f"conf_m_{o}"]
    X["conf_t_T"], X["conf_t_O"] = d[f"conf_t_{side}"], d[f"conf_t_{o}"]
    X["same_conf"] = d.same_conf
    early = (np.minimum(d.gs_home, d.gs_away) < 8).astype(float)
    X["conf_early"] = early * (d[f"conf_m_{side}"] - d[f"conf_m_{o}"])
    X["rest_T"], X["rest_O"] = d[f"rest_{side}"], d[f"rest_{o}"]
    return X


def run_layers(d, groups=tuple(GROUPS), lam=0.9997, ridge=30.0, bg_window=1500, mkt_lam=0.9997, mkt_ridge=50.0,
               return_coefs=False):
    feats = [f for k in groups for f in GROUPS[k]]
    XH, XA = side_X(d, "home")[feats], side_X(d, "away")[feats]
    ref = pd.concat([XH, XA])[np.r_[d.season.values, d.season.values] <= 2015]
    mu, sd = ref.mean(), ref.std().replace(0, 1).fillna(1)
    for f in ("bias", "is_home", "neutral", "playoff", "same_conf"):
        if f in feats:
            mu[f], sd[f] = 0.0, 1.0
    XH = ((XH - mu) / sd).fillna(0).values; XA = ((XA - mu) / sd).fillna(0).values
    yH = (d.home_score - d.mu_home_core).values; yA = (d.away_score - d.mu_away_core).values
    mkt_m, mkt_t = -d.spread_home.values, d.total.values
    am, at = (d.home_score - d.away_score).values, (d.home_score + d.away_score).values
    k = len(feats)
    rls = RLS(k, lam, ridge) if k else None
    rm, rt = RLS(4, mkt_lam, mkt_ridge), RLS(4, mkt_lam, mkt_ridge)
    n = len(d)
    ch, ca = np.zeros(n), np.zeros(n)
    w_m, w_t, pm, pt = (np.full(n, np.nan) for _ in range(4))
    eo_m, ek_m, eo_t, ek_t = [], [], [], []
    wm = wt = 0.5
    dates = d.date.values
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, n]
    neu = d.neutral.values.astype(float)
    cfd = (d.conf_m_home - d.conf_m_away).values
    for s0, s1 in zip(starts[:-1], starts[1:]):
        idx = np.arange(s0, s1)
        if k:
            ch[idx], ca[idx] = XH[idx] @ rls.w, XA[idx] @ rls.w
        mh, ma = d.mu_home_core.values[idx] + ch[idx], d.mu_away_core.values[idx] + ca[idx]
        sm, st = mh - ma, mh + ma
        w_m[idx], w_t[idx] = wm, wt
        zm = np.c_[np.ones(len(idx)), (sm - mkt_m[idx]) / 5, neu[idx], cfd[idx]]
        zt = np.c_[np.ones(len(idx)), (st - mkt_t[idx]) / 8, neu[idx], (d.tempo_home.values[idx] + d.tempo_away.values[idx]) * 10]
        pm[idx], pt[idx] = mkt_m[idx] + zm @ rm.w, mkt_t[idx] + zt @ rt.w
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
                a_, b_ = np.array(eo), np.array(ek)
                den = np.mean((a_ - b_) ** 2)
                w = float(np.clip(np.mean((a_ - b_) * a_) / den, 0, 1)) if den > 1e-9 else 0.5
                if nm == "m":
                    wm = w
                else:
                    wt = w
    o = pd.DataFrame({"game_id": d.game_id.values, "corr_home": ch, "corr_away": ca})
    o["mu_home"], o["mu_away"] = d.mu_home_core.values + ch, d.mu_away_core.values + ca
    o["margin"], o["total"] = o.mu_home - o.mu_away, o.mu_home + o.mu_away
    o["w_mkt_margin"], o["w_mkt_total"] = w_m, w_t
    o["w_mkt"] = (w_m + w_t) / 2
    o["margin_b"] = np.where(np.isfinite(mkt_m), w_m * mkt_m + (1 - w_m) * o.margin, o.margin)
    o["total_b"] = np.where(np.isfinite(mkt_t), w_t * mkt_t + (1 - w_t) * o.total, o.total)
    o["margin_mx"], o["total_mx"] = pm, pt
    if return_coefs:
        return o, (dict(zip(feats, rls.w)) if k else {})
    return o


def score(o, d, seasons, market_only=True):
    sel = d.season.isin(seasons).to_numpy().copy()
    if market_only:
        sel = sel & np.isfinite(d.spread_home.values) & np.isfinite(d.total.values)
    am = (d.home_score - d.away_score).values[sel]; at = (d.home_score + d.away_score).values[sel]
    return dict(mae_margin=mae(o.margin.values[sel] - am), mae_total=mae(o.total.values[sel] - at),
                mae_margin_mkt=mae(-d.spread_home.values[sel] - am), mae_total_mkt=mae(d.total.values[sel] - at),
                n=int(sel.sum()))


def main():
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_games()
    print(f"games {len(g)}")
    tune = range(START, TUNE_END + 1); rep = range(REPORT, 2027)
    grid = [dict(q_game=qg, carry=ca) for qg, ca in ((0.05, 0.6), (0.15, 0.8), (0.3, 0.9), (0.6, 0.9),
                                                     (1.0, 0.9), (0.3, 0.97), (0.6, 0.97), (1.0, 0.97))]
    best, TUNE = None, []
    import os, pickle
    cache = os.environ.get("NCAAM_CORE_CACHE")
    if cache and os.path.exists(cache):
        res = pickle.load(open(cache, "rb"))
    else:
        res = parallel_core(g, grid)
        if cache:
            pickle.dump(res, open(cache, "wb"))
    for kw, C in zip(grid, res):
        m = g[["game_id", "season", "home_score", "away_score"]].merge(C, on="game_id")
        m = m[m.season.isin(tune)]
        v = mae(m.margin_core - (m.home_score - m.away_score)) + mae(m.total_core - (m.home_score + m.away_score))
        TUNE.append(dict(**kw, mae_sum_2016_20=v))
        if best is None or v < best[0]:
            best = (v, kw, C)
    core_kw, C = best[1], best[2]
    print(f"core tuned {core_kw} ({time.time() - T0:.0f}s)")
    E = enrich(g, C)
    d = g.merge(C.drop(columns=["x_home", "x_away"]), on="game_id").merge(E, on="game_id")
    # games played this season (pre-game) for early-season interactions
    lg = pd.concat([d[["game_id", "season", "date", "home"]].rename(columns={"home": "t"}),
                    d[["game_id", "season", "date", "away"]].rename(columns={"away": "t"})]).sort_values(["date", "game_id"])
    lg["k"] = lg.groupby(["season", "t"]).cumcount()
    kk = lg.groupby(["game_id", "t"]).k.first()
    d["gs_home"] = [kk[(a, b)] for a, b in zip(d.game_id, d.home)]
    d["gs_away"] = [kk[(a, b)] for a, b in zip(d.game_id, d.away)]
    RG, bestr = [], None
    for lam in (0.9995, 0.9999):
        for ridge in (10.0, 100.0):
            o = run_layers(d, lam=lam, ridge=ridge)
            sc = score(o, d, tune, market_only=False)
            v = sc["mae_margin"] + sc["mae_total"]
            RG.append(dict(lam=lam, ridge=ridge, mae_sum_2016_20=v))
            if bestr is None or v < bestr[0]:
                bestr = (v, lam, ridge)
    _, LAM, RIDGE = bestr
    print(f"rls tuned {LAM} {RIDGE} ({time.time() - T0:.0f}s)")
    O, coefs = run_layers(d, lam=LAM, ridge=RIDGE, return_coefs=True)
    ABL = []
    configs = [("core only", ()), ("full", tuple(GROUPS))] + \
              [(f"drop {k}", tuple(x for x in GROUPS if x != k)) for k in GROUPS if k != "base"] + \
              [(f"base + {k} only", ("base", k)) for k in GROUPS if k != "base"]
    for name, grp in configs:
        o = run_layers(d, groups=grp, lam=LAM, ridge=RIDGE)
        if not grp:
            o["margin"], o["total"] = d.margin_core.values, d.total_core.values
        for per, ss in (("2016-20", tune), ("2021+", rep)):
            for mo in (False, True):
                sc = score(o, d, ss, market_only=mo)
                ABL.append(dict(config=name, period=per, games="with lines" if mo else "all", **sc))
    ABL = pd.DataFrame(ABL)
    print(ABL.round(3).to_string())
    G = d.merge(O, on="game_id")
    G = G.rename(columns={"total_x": "total_line", "total_y": "total"})
    am, at = G.home_score - G.away_score, G.home_score + G.away_score
    for nm, mcol, tcol in (("", "margin", "total"), ("_b", "margin_b", "total_b"), ("_core", "margin_core", "total_core"),
                           ("_mx", "margin_mx", "total_mx")):
        G[f"err_margin{nm}"], G[f"err_total{nm}"] = G[mcol] - am, G[tcol] - at
    G["err_margin_mkt"], G["err_total_mkt"] = -G.spread_home - am, G.total_line - at
    G["err_home"], G["err_away"] = G.mu_home - G.home_score, G.mu_away - G.away_score
    keep = ["game_id", "date", "season", "home", "away", "home_conference", "away_conference", "neutral", "playoff",
            "mu_home", "mu_away", "sd_home", "sd_away", "rho", "sd_margin", "sd_total", "margin", "total", "margin_b",
            "total_b", "w_mkt", "w_mkt_margin", "w_mkt_total", "margin_mx", "total_mx", "mu_home_core", "mu_away_core",
            "margin_core", "total_core", "corr_home", "corr_away", "mult_home", "mult_away", "tempo_home", "tempo_away",
            "ewm_home", "ewm_away", "conf_m_home", "conf_m_away", "conf_t_home", "conf_t_away", "same_conf",
            "rest_home", "rest_away", "gs_home", "gs_away", "spread_home", "total_line", "spread_open_home",
            "total_open", "home_ml", "away_ml", "home_score", "away_score", "hfa", "level", "games_home",
            "games_away"] + [c for c in G.columns if c.startswith("err_")]
    G = G[G.season >= START][keep].reset_index(drop=True)
    G.to_parquet(OUT / "ncaam_games.parquet", index=False)
    write_report(G, ABL, pd.DataFrame(TUNE), pd.DataFrame(RG), core_kw, LAM, RIDGE, coefs)
    print(f"done ({time.time() - T0:.0f}s)")


def write_report(G, ABL, TUNE, RG, core_kw, LAM, RIDGE, coefs):
    L = ["# NCAAM script-then-learn: learning report\n",
         "Code: `engine/scripts/ncaam.py`. The core is `base.ScriptEngine` maths through `FastEngine`, which has an "
         "identical update with O(1) window sums. Games run sequentially from 2013, which is warm-up; there is no "
         "2012 file. Each date is scripted fully before it is learned. Tuning used 2016-20, and 2021+ is reported. "
         "**2024 and 2025 have no betting lines in `ncaam.csv`**, so market comparisons use 2021-23 + 2026. Lines "
         "cover about 68% of games in 2016-18 and 92% from 2019 on, mostly the D1-vs-D1 games.\n",
         f"Tuned core: {core_kw}. RLS: λ={LAM}, ridge={RIDGE}.\n", "## Bottom line\n"]
    d = G.dropna(subset=["err_margin"])
    dm = d.dropna(subset=["err_margin_mkt", "err_total_mkt"])
    r = dm[dm.season >= REPORT]
    s = lambda c: r[c].abs().mean()
    L.append(f"- **2021+ games with lines (n={len(r)}), margin MAE:** core {s('err_margin_core'):.3f}, enriched "
             f"{s('err_margin'):.3f}, blend {s('err_margin_b'):.3f}, market + learned residual {s('err_margin_mx'):.3f}, "
             f"**closing line {s('err_margin_mkt'):.3f}**.")
    L.append(f"- **Total MAE:** core {s('err_total_core'):.3f}, enriched {s('err_total'):.3f}, blend {s('err_total_b'):.3f}, "
             f"market + residual {s('err_total_mx'):.3f}, **closing line {s('err_total_mkt'):.3f}**.")
    L.append(f"- **Learned market weight, mean over 2021+:** margin {r.w_mkt_margin.mean():.2f}, total {r.w_mkt_total.mean():.2f}.")
    cm = np.corrcoef(r.margin + r.spread_home, -r.err_margin_mkt)[0, 1]
    ct = np.corrcoef(r.total - r.total_line, -r.err_total_mkt)[0, 1]
    L.append(f"- **Information beyond the close:** corr(script − close, actual − close) is {cm:.3f} for margin and "
             f"{ct:.3f} for total.")
    op = r.dropna(subset=["spread_open_home", "total_open"])
    op = op[(op.spread_open_home != op.spread_home) | (op.total_open != op.total_line) | True]
    if len(op) > 200:
        L.append(f"- **Line movement on games with openers ({len(op)}):** corr(script − open, close − open) is "
                 f"{np.corrcoef(op.margin + op.spread_open_home, op.spread_open_home - op.spread_home)[0, 1]:.3f} for "
                 f"spreads and {np.corrcoef(op.total - op.total_open, op.total_line - op.total_open)[0, 1]:.3f} for totals.")
    tr = dm[dm.season.between(START, TUNE_END)]
    rows = []
    for nm, pc, kind in (("spread: script", "margin", "m"), ("spread: mkt+resid", "margin_mx", "m"),
                         ("total: script", "total", "t"), ("total: mkt+resid", "total_mx", "t")):
        arr = lambda x: (x[pc].values, -x.spread_home.values if kind == "m" else x.total_line.values,
                         (x.home_score - x.away_score).values if kind == "m" else (x.home_score + x.away_score).values)
        bt, br = None, -9
        for thr in (1, 2, 3, 4, 6, 8):
            n_, r_ = ats_roi(*arr(tr), thr)
            if n_ >= 300 and r_ > br:
                bt, br = thr, r_
        n2, r2 = ats_roi(*arr(r), bt) if bt else (0, np.nan)
        rows.append(dict(bet=nm, thr=bt, roi_2016_20=br, n_2021p=n2, roi_2021p=r2))
    L.append("\n**Betting the close at −110** (threshold chosen on 2016-20 with n≥300):\n")
    L.append(md_table(pd.DataFrame(rows)))
    lc = d.groupby("season").agg(games=("err_margin", "size"),
                                 margin_core=("err_margin_core", lambda x: x.abs().mean()),
                                 margin_script=("err_margin", lambda x: x.abs().mean()),
                                 total_core=("err_total_core", lambda x: x.abs().mean()),
                                 total_script=("err_total", lambda x: x.abs().mean())).reset_index()
    lm = dm.groupby("season").agg(games_w_line=("err_margin", "size"),
                                  margin_script_l=("err_margin", lambda x: x.abs().mean()),
                                  margin_blend_l=("err_margin_b", lambda x: x.abs().mean()),
                                  margin_market=("err_margin_mkt", lambda x: x.abs().mean()),
                                  total_script_l=("err_total", lambda x: x.abs().mean()),
                                  total_blend_l=("err_total_b", lambda x: x.abs().mean()),
                                  total_market=("err_total_mkt", lambda x: x.abs().mean()),
                                  w_mkt_m=("w_mkt_margin", "last"), w_mkt_t=("w_mkt_total", "last")).reset_index()
    X, f = ml_logloss(d)
    llr = X.groupby("season").apply(lambda x: pd.Series({"ml_ll_script": f(x._ps.values, x._y.values), "ml_ll_blend": f(x._pb.values, x._y.values),
                                                         "ml_ll_market": f(x._pm.values, x._y.values)})).reset_index()
    L.append("\n## Learning curve\n\nAll games (script only):\n")
    L.append(md_table(lc))
    L.append("\nGames with closing lines (`_l` = script on the same games):\n")
    L.append(md_table(lm.merge(llr, on="season", how="left")))
    d2 = dm[dm.season >= START].copy()
    d2["gnum"] = pd.cut(np.minimum(d2.gs_home, d2.gs_away), [-1, 3, 8, 15, 25, 60],
                        labels=["0-3", "4-8", "9-15", "16-25", "26+"])
    ws = d2.groupby("gnum", observed=True).agg(n=("err_margin", "size"), core=("err_margin_core", lambda x: x.abs().mean()),
                                               script=("err_margin", lambda x: x.abs().mean()),
                                               market=("err_margin_mkt", lambda x: x.abs().mean())).reset_index()
    L.append("\n**Within-season learning:** margin MAE by the fewer of the two teams' games played that season "
             "(2016+, games with lines):\n")
    L.append(md_table(ws.astype({"gnum": str})))
    L.append("\n## Error attribution\n")
    e = d[d.season >= REPORT]
    shared = (e.err_home + e.err_away) / 2; split = (e.err_home - e.err_away) / 2
    L.append(f"Per-team errors split into a **shared** part ((e_h + e_a)/2; game environment, tempo, whistle, shooting "
             f"variance common to both) with sd {shared.std():.2f}, and a **split** part ((e_h − e_a)/2; relative "
             f"strength / efficiency) with sd {split.std():.2f}. Their correlation is {np.corrcoef(shared, split)[0, 1]:.2f}. "
             f"The learned score correlation rho is {e.rho.mean():.2f}.\n")
    e = dm[dm.season >= REPORT].copy()
    e["mismatch"] = pd.cut(e.spread_home.abs(), [-1, 4, 9, 15, 60], labels=["|spread|≤4", "4-9", "9-15", ">15"])
    e["early"] = np.where(np.minimum(e.gs_home, e.gs_away) < 8, "first 8 games", "later")
    e["kind"] = np.where(e.playoff == 1, "postseason", np.where(e.same_conf == 1, "conference", "non-conference"))
    e["site"] = np.where(e.neutral == 1, "neutral", "home/away")
    rows = []
    for col in ("mismatch", "early", "kind", "site"):
        for v, x in e.groupby(col, observed=True):
            rows.append(dict(group=str(v), n=len(x), mae_margin_script=mae(x.err_margin), mae_margin_mkt=mae(x.err_margin_mkt),
                             gap_margin=mae(x.err_margin) - mae(x.err_margin_mkt), mae_total_script=mae(x.err_total),
                             mae_total_mkt=mae(x.err_total_mkt), bias_margin_script=x.err_margin.mean(),
                             bias_total_script=x.err_total.mean()))
    L.append("**By game type (2021+, games with lines).** `gap` is script MAE − market MAE:\n")
    L.append(md_table(pd.DataFrame(rows)))
    cf = e.groupby("home_conference").apply(lambda x: pd.Series({"n": len(x), "gap_margin": mae(x.err_margin) - mae(x.err_margin_mkt),
                                                                 "bias_margin": x.err_margin.mean()})).reset_index()
    cf = cf[cf.n >= 150].sort_values("gap_margin", ascending=False)
    L.append("\n**Home-team conference, worst and best relative to market** (2021+, n≥150):\n")
    L.append(md_table(pd.concat([cf.head(5), cf.tail(5)])))
    L.append("\n**Final RLS coefficients** (points per 1 sd feature, for team T's score):\n")
    L.append(md_table(pd.DataFrame({"feature": list(coefs), "coef": list(coefs.values())})))
    L.append("\n## Ablation (MAE; online correction re-run without each group)\n")
    pv = ABL.pivot_table(index="config", columns=["games", "period"], values=["mae_margin", "mae_total"]).round(3)
    pv.columns = [f"{a.replace('mae_', '')} {b} {c}" for a, b, c in pv.columns]
    L.append(md_table(pv.reset_index()))
    mk = ABL[(ABL.period == "2021+") & (ABL.games == "with lines")].iloc[0]
    L.append(f"\nClosing line on the 2021+ games with lines: margin {mk.mae_margin_mkt:.3f}, total {mk.mae_total_mkt:.3f}.\n")
    L.append("Core grid (2016-20):\n\n" + md_table(TUNE) + "\n\nRLS grid:\n\n" + md_table(RG))
    (OUT / "ncaam_learning.md").write_text("\n".join(L) + "\n<!-- conclusions -->\n")


if __name__ == "__main__":
    main()
