"""
Shared walk-forward game predictor for the market team sports (NFL, NCAAF, NBA, NCAAM).  See engine/predict/SPEC.md.

Inputs: results/scripts/<sport>_games.parquet (pre-game script columns + closing lines).
Pipeline (strictly walk-forward, no leakage):
  * features = every pre-game numeric script column (post-game err_*, *_act, scores, poss_actual dropped)
    + market features (odds_ratio de-vigged ML -> logit, -spread, total, line moves, script-minus-market).
  * two variants: MKT (games with a closing spread; boosts from the market) and NOMKT (script + context only;
    used for games without a line, e.g. NCAAM 2024-25).
  * base models per target: logistic (CV'd L2) / LightGBM boosted FROM the market logit (init_score) for
    P(home win); ridge / LightGBM on the residual from the market for margin and total.
  * stack: logistic (resp. linear) meta-model over [lr, gbm, market] fitted on the out-of-sample predictions
    of earlier seasons only -> this is also the calibration step.
  * refit each season (fit on season < S) or monthly (fit on date < first of month); candidate and refit mode
    picked on 2018-2020, everything reported on 2021+.
  * P(cover) / P(over) from a normal score distribution around the predicted margin / total with the
    out-of-sample residual sd of earlier seasons.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pickle
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import (LinearRegression, LogisticRegression,  # noqa: E402
                                  LogisticRegressionCV, RidgeCV)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from backtest.core import american_to_prob  # noqa: E402
from engine.devig import devig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "results" / "scripts"
OUT = ROOT / "results" / "predict"
FIRST_WF, SEL0, SEL1, REPORT = 2017, 2018, 2020, 2021
TIER_EDGES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0001]
TIER_LABELS = ["50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85+"]

CFG = {  # band = margin-error windows for "right winner AND margin within x"
    "nfl": dict(band=(3, 7), min_margin=1, name="NFL"),
    "ncaaf": dict(band=(3, 7), min_margin=1, name="NCAAF"),
    "nba": dict(band=(3, 6), min_margin=1, name="NBA"),
    "ncaam": dict(band=(3, 6), min_margin=1, name="NCAAM"),
}

LEAK = re.compile(r"^err_|_act$|_score$|^poss_actual$|^game_id$|^season$|^date$|^y_")
SKIP = {"level", "hfa", "noise_sd", "score_rho", "w_mkt", "w_mkt_base", "w_mkt_margin", "w_mkt_total",
        "spread_home", "home_ml", "away_ml", "p_home_mkt", "spread_open_home", "total_open"}
MARKET = ["mkt_logit", "mkt_margin", "total_line", "spread_move", "total_move", "script_vs_mkt",
          "total_vs_line", "has_ml"]
BLEND = re.compile(r"(_b|_b_base|_mx)$|^p_home_blend$|^p_home_win_b$")
CONTEXT = re.compile(r"neutral|playoff|postseason|is_post|rest|b2b|travel|tz_|miss_|wind|temp|div_game|roof|"
                     r"^week$|games_|^gs_|same_conf|conf_|p5_vs|fbs_vs|qb_delta")


# ------------------------------------------------------------------ data
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def spread_map(d):
    """slope b with logit(p_ml) ~ b * (-spread) on early-season rows that have both (market-only fit)."""
    m = d.p_ml.notna() & d.spread_home.notna()
    early = m & (d.season <= d.loc[m, "season"].min() + 2)
    x, y = -d.loc[early, "spread_home"].to_numpy(float), logit(d.loc[early, "p_ml"])
    return float((x * y).sum() / (x * x).sum())


def add_features(d: pd.DataFrame, b: float | None = None):
    d = d.copy()
    for c in ["spread_home", "total_line", "home_ml", "away_ml", "spread_open_home", "total_open"]:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ok = d.home_ml.notna() & d.away_ml.notna() & (d.home_ml.abs() >= 100) & (d.away_ml.abs() >= 100)
    d["p_ml"] = np.nan
    if ok.any():
        d.loc[ok, "p_ml"] = devig(d.loc[ok, "home_ml"].to_numpy(float), d.loc[ok, "away_ml"].to_numpy(float),
                                  "odds_ratio")[0]
    if b is None:
        b = spread_map(d)
    d["has_mkt"] = d.spread_home.notna()
    d["has_ml"] = d.p_ml.notna().astype(float)
    d["mkt_p"] = d.p_ml.fillna(pd.Series(sigmoid(-b * d.spread_home.to_numpy(float)), index=d.index))
    d["mkt_logit"] = logit(d.mkt_p)
    d["mkt_margin"] = -d.spread_home
    d["spread_move"] = d.spread_open_home - d.spread_home        # >0: line moved toward home
    d["total_move"] = d.total_line - d.total_open
    d["p_script"] = norm.cdf(d["margin"] / d["sd_margin"])
    d["script_logit"] = logit(d.p_script)
    d["script_vs_mkt"] = d["margin"] - d.mkt_margin
    d.loc[d.has_mkt & d.total_line.isna(), "total_line"] = d["total"]
    d["total_vs_line"] = d["total"] - d.total_line
    if "roof" in d:
        d["roof_dome"] = d.roof.isin(["dome", "closed"]).astype(float)
    if "game_type" in d:
        d["is_post"] = (d.game_type.fillna("REG") != "REG").astype(float)
    hs, as_ = (pd.to_numeric(d.get("home_score"), errors="coerce") if "home_score" in d else np.nan,
               pd.to_numeric(d.get("away_score"), errors="coerce") if "away_score" in d else np.nan)
    d["y_margin"] = hs - as_
    d["y_total"] = hs + as_
    d["y_win"] = np.where(d.y_margin > 0, 1.0, np.where(d.y_margin < 0, 0.0, np.nan))
    return d, b


def feature_groups(d):
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
    cand = [c for c in num if not LEAK.search(c) and c not in SKIP and c not in
            ("p_ml", "mkt_p", "p_script", "has_mkt")]
    g = {"market": [c for c in MARKET if c in d], "blend": [], "context": [], "script": []}
    for c in cand:
        if c in g["market"]:
            continue
        if BLEND.search(c):
            g["blend"].append(c)
        elif CONTEXT.search(c):
            g["context"].append(c)
        else:
            g["script"].append(c)
    g = {k: [c for c in v if d[c].notna().mean() > 0.02 and d[c].nunique() > 1] for k, v in g.items()}
    return g


def load(sport):
    d = pd.read_parquet(SCRIPTS / f"{sport}_games.parquet").sort_values(["date", "home"]).reset_index(drop=True)
    d, b = add_features(d)
    return d, b


# ------------------------------------------------------------------ models
def gbm_params(n):
    if n < 6000:
        return dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=2)
    return dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=2)


def _imp():
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def fit_bases(tr, F, base, C=None, w=None):
    """base = (logit col, margin col, total col) the boosters start from.  w = optional sample weights."""
    X = tr[F].to_numpy(float)
    w = np.ones(len(tr)) if w is None else np.asarray(w, float)
    yw = tr.y_win.to_numpy()
    k = ~np.isnan(yw)
    if C is None:
        lr = make_pipeline(_imp(), StandardScaler(), LogisticRegressionCV(
            Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0], cv=3, scoring="neg_log_loss", max_iter=3000))
    else:
        lr = make_pipeline(_imp(), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    step = "logisticregression" if C is not None else "logisticregressioncv"
    lr.fit(X[k], yw[k], **{f"{step}__sample_weight": w[k]})
    p = gbm_params(len(tr))
    bl, bm, bt = (tr[c].to_numpy(float) for c in base)
    gc = lgb.LGBMClassifier(**p).fit(X[k], yw[k], init_score=bl[k], sample_weight=w[k])
    ym, yt = tr.y_margin.to_numpy(float), tr.y_total.to_numpy(float)
    alphas = np.logspace(-1, 4, 11)
    rm = make_pipeline(_imp(), StandardScaler(), RidgeCV(alphas=alphas)).fit(X, ym, ridgecv__sample_weight=w)
    rt = make_pipeline(_imp(), StandardScaler(), RidgeCV(alphas=alphas)).fit(X, yt, ridgecv__sample_weight=w)
    gm = lgb.LGBMRegressor(**p).fit(X, ym - bm, sample_weight=w)
    gt = lgb.LGBMRegressor(**p).fit(X, yt - bt, sample_weight=w)
    Cfit = float(lr[-1].C_[0]) if C is None else C
    return dict(F=F, base=base, lr=lr, gc=gc, rm=rm, rt=rt, gm=gm, gt=gt, C=Cfit)


def pred_bases(M, te):
    X = te[M["F"]].to_numpy(float)
    bl, bm, bt = (te[c].to_numpy(float) for c in M["base"])
    return pd.DataFrame({
        "b_p": sigmoid(bl), "lr_p": M["lr"].predict_proba(X)[:, 1],
        "gb_p": sigmoid(M["gc"].predict(X, raw_score=True) + bl),
        "b_m": bm, "rd_m": M["rm"].predict(X), "gb_m": M["gm"].predict(X) + bm,
        "b_t": bt, "rd_t": M["rt"].predict(X), "gb_t": M["gt"].predict(X) + bt}, index=te.index)


def fit_meta(o):
    """o: OOS base predictions with outcomes. Returns (cls meta, margin meta, total meta)."""
    k = o.y_win.notna()
    Xc = np.c_[logit(o.lr_p), logit(o.gb_p), logit(o.b_p)][k]
    mc = LogisticRegression(C=10.0, max_iter=1000).fit(Xc, o.y_win[k])
    mm = LinearRegression().fit(o[["rd_m", "gb_m", "b_m"]].to_numpy(), o.y_margin)
    mt = LinearRegression().fit(o[["rd_t", "gb_t", "b_t"]].to_numpy(), o.y_total)
    return mc, mm, mt


def apply_meta(meta, P):
    mc, mm, mt = meta
    return (mc.predict_proba(np.c_[logit(P.lr_p), logit(P.gb_p), logit(P.b_p)])[:, 1],
            mm.predict(P[["rd_m", "gb_m", "b_m"]].to_numpy()), mt.predict(P[["rd_t", "gb_t", "b_t"]].to_numpy()))


# ------------------------------------------------------------------ walk-forward
def variant_spec(d, groups, v):
    if v == "mkt":
        F = groups["market"] + groups["blend"] + groups["script"] + groups["context"]
        return F, ("mkt_logit", "mkt_margin", "total_line"), d.has_mkt
    F = groups["script"] + groups["context"]
    return F, ("script_logit", "margin", "total"), pd.Series(True, index=d.index)


def walk_forward(d, F, base, rows, mode, Cs=None, log=print):
    """OOS base predictions for every row in `rows` with season >= FIRST_WF."""
    scored = d.y_margin.notna() & rows
    out, Cused = [], {}
    for S in sorted(d.loc[rows & (d.season >= FIRST_WF), "season"].unique()):
        te_all = d[rows & (d.season == S)]
        if mode == "season":
            cuts = [te_all.date.min()]
        else:
            ms = te_all.date.dt.to_period("M").dt.start_time.unique()
            cuts = sorted(ms)
        for i, c in enumerate(cuts):
            nxt = cuts[i + 1] if i + 1 < len(cuts) else pd.Timestamp.max
            te = te_all if mode == "season" else te_all[(te_all.date >= c) & (te_all.date < nxt)]
            if te.empty:
                continue
            tr = d[scored & (d.date < c)]
            M = fit_bases(tr, F, base, C=None if Cs is None else Cs.get(S))
            Cused[S] = M["C"]
            P = pred_bases(M, te)
            P["season"], P["refit_cut"] = S, c
            out.append(P)
    o = pd.concat(out)
    o = o.join(d[["y_win", "y_margin", "y_total"]])
    return o, Cused


def stack_wf(o):
    """Per-season stack/calibration fitted on OOS rows of earlier seasons (>= FIRST_WF)."""
    o = o.copy()
    for c in ["st_p", "st_m", "st_t"]:
        o[c] = np.nan
    metas = {}
    for S in sorted(o.season.unique()):
        prev = o[(o.season < S) & o.y_margin.notna()]
        if len(prev) < 100:
            continue
        meta = fit_meta(prev)
        metas[S] = meta
        idx = o.season == S
        p, m, t = apply_meta(meta, o[idx])
        o.loc[idx, "st_p"], o.loc[idx, "st_m"], o.loc[idx, "st_t"] = p, m, t
    return o, metas


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select(o, sel_mask):
    s = o[sel_mask & o.y_win.notna()]
    sc = {k: logloss(s.y_win, s[c]) for k, c in [("base", "b_p"), ("lr", "lr_p"), ("gbm", "gb_p"), ("stack", "st_p")]}
    sm = o[sel_mask & o.y_margin.notna()]
    mm = {k: float((sm.y_margin - sm[c]).abs().mean()) for k, c in
          [("base", "b_m"), ("ridge", "rd_m"), ("gbm", "gb_m"), ("stack", "st_m")]}
    mt = {k: float((sm.y_total - sm[c]).abs().mean()) for k, c in
          [("base", "b_t"), ("ridge", "rd_t"), ("gbm", "gb_t"), ("stack", "st_t")]}
    acc = {k: float(((s[c] > 0.5) == (s.y_win == 1)).mean()) for k, c in
           [("base", "b_p"), ("lr", "lr_p"), ("gbm", "gb_p"), ("stack", "st_p")]}
    return sc, mm, mt, acc


PCOL = {"base": "b_p", "lr": "lr_p", "gbm": "gb_p", "stack": "st_p"}
MCOL = {"base": "b_m", "ridge": "rd_m", "gbm": "gb_m", "stack": "st_m"}
TCOL = {"base": "b_t", "ridge": "rd_t", "gbm": "gb_t", "stack": "st_t"}


# ------------------------------------------------------------------ final predictions
def tier_of(pp):
    return pd.cut(pd.Series(pp), TIER_EDGES, labels=TIER_LABELS, right=False).astype(str).to_numpy()


def finalize(df, p, m, t, sd_m, sd_t, min_margin=1, m_ats=None, t_ou=None):
    """df needs spread_home,total_line. Returns frame with picks, consistent rounded score, cover/over probs."""
    r = pd.DataFrame(index=df.index)
    r["p_home"] = p
    r["pick_home"] = (p >= 0.5)
    r["pick_prob"] = np.where(r.pick_home, p, 1 - p)
    r["pred_margin"], r["pred_total"] = m, t
    r["pred_home_score"], r["pred_away_score"] = (t + m) / 2, (t - m) / 2
    # final integer score consistent with the predicted winner
    fm = np.round(m)
    sign = np.where(r.pick_home, 1, -1)
    fm = np.where(fm * sign < min_margin, sign * min_margin, fm)
    ft = np.round(t)
    fh = np.round((ft + fm) / 2)
    fa = fh - fm
    r["final_home"], r["final_away"] = fh, fa
    sp, tl = df.spread_home.to_numpy(float), df.total_line.to_numpy(float)
    m_ats = m if m_ats is None else m_ats
    t_ou = t if t_ou is None else t_ou
    r["ats_margin"], r["ou_total"] = m_ats, t_ou
    pc = norm.cdf((m_ats + sp) / sd_m)                                   # P(home covers)
    r["p_home_cover"] = pc
    r["spread_pick_home"] = np.where(np.isnan(sp), np.nan, pc >= 0.5)
    r["spread_prob"] = np.where(np.isnan(sp), np.nan, np.maximum(pc, 1 - pc))
    po = norm.cdf((t_ou - tl) / sd_t)
    r["p_over"] = po
    r["total_pick_over"] = np.where(np.isnan(tl), np.nan, po >= 0.5)
    r["total_prob"] = np.where(np.isnan(tl), np.nan, np.maximum(po, 1 - po))
    r["tier"] = tier_of(r.pick_prob.to_numpy())
    return r


# ------------------------------------------------------------------ report helpers
def md(df, floatfmt=3):
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else f"{x:.{floatfmt}f}"
        return str(x)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) and df[c].notna().any() and \
                np.allclose(df[c].dropna() % 1, 0) and c in ("season", "n", "days", "top-N", "n_line", "n_all", "games"):
            df[c] = df[c].astype("Int64")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.astype(object).itertuples(index=False):
        lines.append("| " + " | ".join(f(v) for v in row) + " |")
    return "\n".join(lines)


def tier_table(sub, prob, hit, be, seasons):
    """sub rows with prob col, hit col (1/0, NaN = push/no bet), be = break-even prob col or float."""
    s = sub[sub[hit].notna() & sub[prob].notna()].copy()
    s["tier"] = tier_of(s[prob].to_numpy())
    rows = []
    for t in TIER_LABELS:
        g = s[s.tier == t]
        row = {"tier": t}
        for S in seasons:
            gs = g[g.season == S]
            row[str(S)] = f"{len(gs)} / {100 * gs[hit].mean():.0f}%" if len(gs) else "0"
        row["n"] = len(g)
        row["win%"] = 100 * g[hit].mean() if len(g) else np.nan
        row["break-even%"] = 100 * (g[be].mean() if isinstance(be, str) else be) if len(g) else np.nan
        rows.append(row)
    T = pd.DataFrame(rows)
    T["n"] = T["n"].astype(int)
    ok = T[T.n >= 100]
    best = ok.loc[ok["win%"].idxmax()] if len(ok) else None
    return T, best


def parlay_table(sub, prob_col, hit_col, mkt_prob_col=None, mkt_hit_col=None, N=(1, 2, 3, 4, 5)):
    rows = []
    s = sub[sub[hit_col].notna()]
    for n in N:
        hits, exp, days, mh = [], [], 0, []
        for _, g in s.groupby("date"):
            if len(g) < n:
                continue
            days += 1
            top = g.nlargest(n, prob_col)
            hits.append(top[hit_col].all())
            exp.append(top[prob_col].prod())
            if mkt_prob_col:
                tm = g.nlargest(n, mkt_prob_col)
                mh.append(tm[mkt_hit_col].all())
        r = {"top-N": n, "days": days, "hit%": 100 * np.mean(hits) if hits else np.nan,
             "model-expected%": 100 * np.mean(exp) if exp else np.nan}
        if mkt_prob_col:
            r["market-fav top-N hit%"] = 100 * np.mean(mh) if mh else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main run
def run(sport, log=print):
    t0 = time.time()
    cfg = CFG[sport]
    OUT.mkdir(parents=True, exist_ok=True)
    d, bmap = load(sport)
    groups = feature_groups(d)
    log(f"{sport}: {len(d)} rows; groups " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    sel = (d.season >= SEL0) & (d.season <= SEL1)

    res = {}
    for v in ["mkt", "nomkt"]:
        F, base, rows = variant_spec(d, groups, v)
        o_s, Cs = walk_forward(d, F, base, rows, "season")
        o_s, metas_s = stack_wf(o_s)
        res[(v, "season")] = (o_s, metas_s)
        log(f"  {v}/season done {time.time() - t0:.0f}s")
        if v == "mkt":
            o_m, _ = walk_forward(d, F, base, rows, "month", Cs=Cs)
            o_m, metas_m = stack_wf(o_m)
            res[(v, "month")] = (o_m, metas_m)
            log(f"  {v}/month done {time.time() - t0:.0f}s")

    # ---- selection on 2018-2020
    choice, seltab = {}, []
    for v in ["mkt", "nomkt"]:
        best = None
        for mode in (["season", "month"] if v == "mkt" else ["season"]):
            o, _ = res[(v, mode)]
            msk = sel.reindex(o.index).fillna(False).to_numpy() & (o.season >= SEL0).to_numpy()
            if v == "nomkt":   # evaluate the no-market model on the rows it is actually used for when possible
                pass
            sc, mm, mt, acc = select(o, pd.Series(msk, index=o.index))
            for k in sc:
                seltab.append({"variant": v, "refit": mode, "target": "win", "model": k, "logloss": sc[k],
                               "acc%": 100 * acc[k]})
            for k in mm:
                seltab.append({"variant": v, "refit": mode, "target": "margin", "model": k, "MAE": mm[k]})
                seltab.append({"variant": v, "refit": mode, "target": "total", "model": k, "MAE": mt[k]})
            kp = min(sc, key=sc.get)
            if best is None or sc[kp] < best[1]:
                best = (mode, sc[kp], kp, min(mm, key=mm.get), min(mt, key=mt.get))
        o, _ = res[(v, best[0])]
        msk = pd.Series(((o.season >= SEL0) & (o.season <= SEL1)).to_numpy(), index=o.index)
        _, mm, mt, _ = select(o, msk)
        mm.pop("base"); mt.pop("base")
        choice[v] = dict(mode=best[0], p=best[2], m=best[3], t=best[4],
                         m_ats=min(mm, key=mm.get), t_ou=min(mt, key=mt.get))
    log(f"  choice {choice}")
    seltab = pd.DataFrame(seltab)

    # ---- assemble final OOS predictions (2018+)
    parts = []
    for v in ["mkt", "nomkt"]:
        ch = choice[v]
        o, _ = res[(v, ch["mode"])]
        q = o[o.season >= SEL0].copy()
        q["variant"] = v
        q["p_final"], q["m_final"], q["t_final"] = q[PCOL[ch["p"]]], q[MCOL[ch["m"]]], q[TCOL[ch["t"]]]
        q["m_ats"], q["t_ou"] = q[MCOL[ch["m_ats"]]], q[TCOL[ch["t_ou"]]]
        parts.append(q)
    om, on = parts
    on = on[~on.index.isin(om.index)]
    P = pd.concat([om, on]).sort_index()
    # out-of-sample residual sds from earlier seasons (per variant)
    P["sd_m"], P["sd_t"] = np.nan, np.nan
    for v in ["mkt", "nomkt"]:
        ch = choice[v]
        o, _ = res[(v, ch["mode"])]
        rm = o.y_margin - o[MCOL[ch["m_ats"]]]
        rt = o.y_total - o[TCOL[ch["t_ou"]]]
        for S in P.loc[P.variant == v, "season"].unique():
            prev = (o.season < S) & rm.notna()
            k = (P.variant == v) & (P.season == S)
            P.loc[k, "sd_m"] = float(np.sqrt(np.mean(rm[prev] ** 2)))
            P.loc[k, "sd_t"] = float(np.sqrt(np.nanmean(rt[prev] ** 2)))
    G = d.loc[P.index]
    F_ = finalize(G, P.p_final.to_numpy(), P.m_final.to_numpy(), P.t_final.to_numpy(),
                  P.sd_m.to_numpy(), P.sd_t.to_numpy(), cfg["min_margin"], P.m_ats.to_numpy(), P.t_ou.to_numpy())
    keep = ["game_id", "date", "season", "home", "away", "neutral", "home_score", "away_score", "spread_home",
            "total_line", "home_ml", "away_ml", "has_mkt", "mkt_p", "p_script", "margin", "total"]
    keep = [c for c in keep if c in G]
    preds = G[keep].rename(columns={"margin": "script_margin", "total": "script_total"}).copy()
    preds = preds.join(P[["variant", "lr_p", "gb_p", "st_p", "rd_m", "gb_m", "st_m", "rd_t", "gb_t", "st_t",
                          "sd_m", "sd_t", "y_win", "y_margin", "y_total"]]).join(F_)
    preds["refit"] = np.where(preds.variant == "mkt", choice["mkt"]["mode"], choice["nomkt"]["mode"])
    preds["pick"] = np.where(preds.pick_home, preds.home, preds.away)

    # outcome / hit columns
    pr = preds
    pr["ml_hit"] = np.where(pr.y_win.isna(), np.nan, (pr.pick_home == (pr.y_win == 1)).astype(float))
    pr["mkt_fav_home"] = pr.mkt_p >= 0.5
    pr["mkt_hit"] = np.where(pr.y_win.isna() | ~pr.has_mkt, np.nan, (pr.mkt_fav_home == (pr.y_win == 1)).astype(float))
    pr["mkt_pick_prob"] = np.maximum(pr.mkt_p, 1 - pr.mkt_p)
    pr["script_hit"] = np.where(pr.y_win.isna(), np.nan, ((pr.p_script >= 0.5) == (pr.y_win == 1)).astype(float))
    cov = pr.y_margin + pr.spread_home
    pr["spread_hit"] = np.where(pr.y_margin.isna() | pr.spread_home.isna() | (cov == 0), np.nan,
                                ((cov > 0) == (pr.spread_pick_home == 1)).astype(float))
    ou = pr.y_total - pr.total_line
    pr["total_hit"] = np.where(pr.y_total.isna() | pr.total_line.isna() | (ou == 0), np.nan,
                               ((ou > 0) == (pr.total_pick_over == 1)).astype(float))
    pick_odds = np.where(pr.pick_home, pr.home_ml, pr.away_ml)
    pr["pick_breakeven"] = np.where(np.isnan(pick_odds.astype(float)), np.nan,
                                    american_to_prob(np.nan_to_num(pick_odds.astype(float), nan=100)))

    # ---- final model for live use (fit on everything scored)
    final = {}
    for v in ["mkt", "nomkt"]:
        F, base, rows = variant_spec(d, groups, v)
        tr = d[d.y_margin.notna() & rows]
        o, metas = res[(v, choice[v]["mode"])]
        M = fit_bases(tr, F, base)
        meta = fit_meta(o[o.y_margin.notna()])
        q = pr[(pr.variant == v) & pr.y_margin.notna()]
        final[v] = dict(models=M, meta=meta, choice=choice[v],
                        sd_m=float(np.sqrt(np.mean((q.y_margin - q.ats_margin) ** 2))) if len(q) else 12.0,
                        sd_t=float(np.sqrt(np.nanmean((q.y_total - q.ou_total) ** 2))) if len(q) else 15.0)
    model = dict(sport=sport, spread_b=bmap, groups=groups, final=final, min_margin=cfg["min_margin"],
                 trained_through=str(d.loc[d.y_margin.notna(), "date"].max().date()))
    with open(OUT / f"{sport}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ---- ablation (logistic + ridge, season refits, games with a line, 2021+)
    abl = ablation(d, groups, log)
    # ---- does more history help?
    hist = history_test(d, groups, choice, res, pr, log)

    pr_out = pr.copy()
    for c in pr_out.columns:
        if pr_out[c].dtype == object and c not in ("game_id", "home", "away", "pick", "variant", "refit", "tier"):
            pr_out[c] = pd.to_numeric(pr_out[c], errors="coerce")
    pr_out["game_id"] = pr_out["game_id"].astype(str)
    pr_out.to_parquet(OUT / f"{sport}_preds.parquet", index=False)
    rep, summary = report(sport, d, pr, seltab, choice, res, final, groups, abl, hist, time.time() - t0)
    (OUT / f"{sport}_predict.md").write_text(rep)
    log(f"{sport} done in {time.time() - t0:.0f}s")
    return summary


def ablation(d, groups, log=print):
    sets = {
        "full (all groups)": ["market", "blend", "script", "context"],
        "- market (no lines)": ["script", "context"],
        "- blend": ["market", "script", "context"],
        "- script": ["market", "blend", "context"],
        "- context": ["market", "blend", "script"],
        "market only": ["market"],
        "market + script": ["market", "script"],
    }
    rows = d.has_mkt
    out = []
    test = rows & (d.season >= REPORT) & d.y_margin.notna()
    for name, gs in sets.items():
        F = sum((groups[g] for g in gs), [])
        P = []
        for S in sorted(d.loc[test, "season"].unique()):
            tr = d[rows & d.y_margin.notna() & (d.season < S)]
            te = d[test & (d.season == S)]
            X, Xt = tr[F].to_numpy(float), te[F].to_numpy(float)
            k = tr.y_win.notna().to_numpy()
            lr = make_pipeline(_imp(), StandardScaler(), LogisticRegressionCV(
                Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0], cv=3, scoring="neg_log_loss", max_iter=3000)).fit(X[k], tr.y_win[k])
            rm = make_pipeline(_imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(X, tr.y_margin)
            rt = make_pipeline(_imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(X, tr.y_total)
            P.append(pd.DataFrame({"p": lr.predict_proba(Xt)[:, 1], "m": rm.predict(Xt), "t": rt.predict(Xt)},
                                  index=te.index))
        P = pd.concat(P).join(d[["y_win", "y_margin", "y_total"]])
        k = P.y_win.notna()
        out.append({"features": name, "n_feat": len(F), "SU acc%": 100 * ((P.p[k] > 0.5) == (P.y_win[k] == 1)).mean(),
                    "logloss": logloss(P.y_win[k], P.p[k]), "MAE margin": (P.y_margin - P.m).abs().mean(),
                    "MAE total": (P.y_total - P.t).abs().mean()})
    log("  ablation done")
    return pd.DataFrame(out)


def history_test(d, groups, choice, res, pr, log=print):
    """Eval fixed at 2023+ (market variant, season refits). Retrain the chosen base models on the last k seasons
    (or with exponential season decay) and combine with that season's stack meta from the main walk-forward."""
    v = "mkt"
    ch = choice[v]
    F, base, rows = variant_spec(d, groups, v)
    _, metas = res[(v, "season")]
    pcol = ch["p"] if ch["p"] != "base" else "stack"
    mcol = ch["m"] if ch["m"] != "base" else ch["m_ats"]
    tcol = ch["t"] if ch["t"] != "base" else ch["t_ou"]
    seasons = sorted(d.loc[rows & d.y_margin.notna() & (d.season >= 2023), "season"].unique())
    first = int(d.season.min())
    configs = [(f"last {k}", k, None) for k in (1, 2, 4, 6)] + [("all prior", None, None)] + \
              [(f"all, decay half-life {h} seasons", None, h) for h in (1, 2, 4)]
    best = pr[pr.season >= REPORT]
    T, bt = tier_table(best, "pick_prob", "ml_hit", "pick_breakeven", [])
    lo = 0.75 if bt is None else TIER_EDGES[TIER_LABELS.index(bt["tier"])]
    out = []
    for name, k, h in configs:
        Ps = []
        for S in seasons:
            avail = sorted(d.loc[rows & d.y_margin.notna() & (d.season < S), "season"].unique())
            lo_s = avail[-k] if k else first          # last k seasons that HAVE line data
            tr = d[rows & d.y_margin.notna() & (d.season < S) & (d.season >= lo_s)]
            te = d[rows & d.y_margin.notna() & (d.season == S)]
            w = None if h is None else 0.5 ** ((S - 1 - tr.season.to_numpy()) / h)
            M = fit_bases(tr, F, base, w=w)
            B = pred_bases(M, te)
            if S in metas:
                B["st_p"], B["st_m"], B["st_t"] = apply_meta(metas[S], B)
            Ps.append(B)
        B = pd.concat(Ps).join(d[["y_win", "y_margin", "y_total"]])
        p = B[PCOL[pcol]]
        m = B[MCOL[mcol]]
        t = B[TCOL[tcol]]
        kk = B.y_win.notna()
        pp = np.maximum(p, 1 - p)
        hit = ((p > 0.5) == (B.y_win == 1))
        hi = kk & (pp >= lo)
        out.append({"training window": name, "SU win%": 100 * hit[kk].mean(),
                    "logloss": logloss(B.y_win[kk], p[kk]),
                    "MAE margin": (B.y_margin - m).abs().mean(), "MAE total": (B.y_total - t).abs().mean(),
                    f"P(pick)>={lo:.2f} n": int(hi.sum()), f"P(pick)>={lo:.2f} win%": 100 * hit[hi].mean()})
    log("  history test done")
    return pd.DataFrame(out), (pcol, mcol, tcol)


def report(sport, d, pr, seltab, choice, res, final, groups, abl, hist, secs):
    cfg = CFG[sport]
    name = cfg["name"]
    L = [f"# {name} game predictor (win% first)", ""]
    L.append(f"Code: `engine/predict/{sport}.py` -> `engine/predict/common_team.py`. Inputs: "
             f"`results/scripts/{sport}_games.parquet`. Runtime {secs / 60:.1f} min. Strict walk-forward: for each "
             f"season S, models are fit only on earlier games (season < S, or date < first of the month when monthly "
             f"refits were chosen); the stack/calibration layer for season S is fit only on the out-of-sample "
             f"predictions of seasons {FIRST_WF}..S-1. Candidates and refit cadence were picked on {SEL0}-{SEL1}; "
             f"**everything below is {REPORT}+ unless stated**. Ties (NFL) are excluded from win% tables.")
    L.append("")
    sc = pr[pr.season >= REPORT]
    scm = sc[sc.has_mkt & sc.y_win.notna()]
    L.append("## 0. Bottom line")
    L.append("")
    acc_m = 100 * scm.ml_hit.mean()
    acc_k = 100 * scm.mkt_hit.mean()
    L.append(f"- Chosen: market-variant win model **{choice['mkt']['p']}** ({choice['mkt']['mode']} refits), margin "
             f"**{choice['mkt']['m']}**, total **{choice['mkt']['t']}**; no-line games use the script-only variant "
             f"(win {choice['nomkt']['p']}, margin {choice['nomkt']['m']}, total {choice['nomkt']['t']}). "
             f"Spread / total picks use the best learned (non-market) margin / total model, "
             f"**{choice['mkt']['m_ats']} / {choice['mkt']['t_ou']}**, because the bare market line gives P(cover) = "
             f"P(over) = 50% by construction. `base` = the closing line itself (de-vigged ML, -spread, total).")
    L.append(f"- {REPORT}+ straight-up winner, games with a closing line (n={len(scm)}): **model {acc_m:.2f}%** vs "
             f"market favourite {acc_k:.2f}% vs script alone {100 * scm.script_hit.mean():.2f}%. "
             f"Agreement model = market favourite on {100 * (scm.pick_home == scm.mkt_fav_home).mean():.1f}% of games.")
    sa = sc[sc.y_win.notna()]
    L.append(f"- All {REPORT}+ games incl. those without a line (n={len(sa)}): model {100 * sa.ml_hit.mean():.2f}%, "
             f"script {100 * sa.script_hit.mean():.2f}%.")
    L.append(f"- Log loss (lines games): model {logloss(scm.y_win, scm.p_home):.4f} vs market "
             f"{logloss(scm.y_win, scm.mkt_p):.4f} vs script {logloss(scm.y_win, scm.p_script):.4f}.")
    L.append("")

    # 1. learning curve
    L.append("## 1. Per-season learning curve (straight-up winner, log loss, score MAE)")
    L.append("")
    L.append("Games with a closing line. `model` = final pick; `mkt` = de-vigged closing favourite; `script` = "
             "Kalman script alone. MAE margin/total: model vs closing spread/total vs script.")
    L.append("")
    rows = []
    for S, g in pr[pr.y_win.notna() | pr.y_margin.notna()].groupby("season"):
        gm = g[g.has_mkt & g.y_win.notna()]
        gg = g[g.has_mkt & g.y_margin.notna()]
        ga = g[g.y_win.notna()]
        rows.append({"season": S, "n_line": len(gm), "acc_model": 100 * gm.ml_hit.mean() if len(gm) else np.nan,
                     "acc_mkt": 100 * gm.mkt_hit.mean() if len(gm) else np.nan,
                     "acc_script": 100 * gm.script_hit.mean() if len(gm) else np.nan,
                     "ll_model": logloss(gm.y_win, gm.p_home) if len(gm) else np.nan,
                     "ll_mkt": logloss(gm.y_win, gm.mkt_p) if len(gm) else np.nan,
                     "ll_script": logloss(gm.y_win, gm.p_script) if len(gm) else np.nan,
                     "mae_m_model": (gg.y_margin - gg.pred_margin).abs().mean(),
                     "mae_m_mkt": (gg.y_margin + gg.spread_home).abs().mean(),
                     "mae_m_script": (gg.y_margin - gg.script_margin).abs().mean(),
                     "mae_t_model": (gg.y_total - gg.pred_total).abs().mean(),
                     "mae_t_mkt": (gg.y_total - gg.total_line).abs().mean(),
                     "mae_t_script": (gg.y_total - gg.script_total).abs().mean(),
                     "n_all": len(ga), "acc_all_model": 100 * ga.ml_hit.mean() if len(ga) else np.nan})
    LC = pd.DataFrame(rows)
    L.append(md(LC, 2))
    L.append("")
    L.append("Refit cadence and candidate comparison on the selection window "
             f"{SEL0}-{SEL1} (market variant = games with a line; no-market variant = all games):")
    L.append("")
    st = seltab.copy()
    L.append(md(st[st.target == "win"].drop(columns=[c for c in ["MAE"] if c in st]).reset_index(drop=True), 4))
    L.append("")
    L.append(md(st[st.target != "win"].drop(columns=[c for c in ["logloss", "acc%"] if c in st]).reset_index(drop=True), 3))
    L.append("")
    # month vs season on 2021+ (informational)
    if ("mkt", "month") in res:
        L.append(f"Monthly vs season refits on {REPORT}+ (market variant, informational, not used for selection):")
        L.append("")
        rr = []
        for mode in ["season", "month"]:
            o, _ = res[("mkt", mode)]
            q = o[(o.season >= REPORT) & o.y_win.notna()]
            for k, c in PCOL.items():
                rr.append({"refit": mode, "model": k, "acc%": 100 * ((q[c] > 0.5) == (q.y_win == 1)).mean(),
                           "logloss": logloss(q.y_win, q[c]),
                           "MAE margin": (q.y_margin - q[MCOL.get(k if k != "lr" else "ridge")]).abs().mean()})
        L.append(md(pd.DataFrame(rr), 4))
        L.append("")
    # calibration
    L.append(f"Calibration of P(home win), {REPORT}+ all games:")
    L.append("")
    q = sc[sc.y_win.notna()].copy()
    q["bin"] = pd.cut(q.p_home, np.linspace(0, 1, 11))
    cal = q.groupby("bin", observed=True).agg(n=("y_win", "size"), mean_pred=("p_home", "mean"),
                                              observed=("y_win", "mean")).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    L.append(md(cal, 3))
    L.append("")

    # 2. score accuracy
    L.append("## 2. Final-score accuracy")
    L.append("")
    b1, b2 = cfg["band"]
    rows = []
    for lab, g in [(f"{REPORT}+ with line", sc[sc.has_mkt & sc.y_margin.notna()]),
                   (f"{REPORT}+ all games", sc[sc.y_margin.notna()]),
                   (f"{REPORT}+ no line (script variant)", sc[~sc.has_mkt & sc.y_margin.notna()])]:
        if not len(g):
            continue
        hs, as_ = (g.y_total + g.y_margin) / 2, (g.y_total - g.y_margin) / 2
        fm = g.final_home - g.final_away
        right = (np.sign(fm) == np.sign(g.y_margin))
        rows.append({"games": lab, "n": len(g), "MAE home": (hs - g.pred_home_score).abs().mean(),
                     "MAE away": (as_ - g.pred_away_score).abs().mean(),
                     "MAE margin": (g.y_margin - g.pred_margin).abs().mean(),
                     "MAE total": (g.y_total - g.pred_total).abs().mean(),
                     "MAE margin (final int score)": (g.y_margin - fm).abs().mean(),
                     "MAE total (final int score)": (g.y_total - (g.final_home + g.final_away)).abs().mean(),
                     f"winner & margin within {b1}%": 100 * (right & ((g.y_margin - fm).abs() <= b1)).mean(),
                     f"winner & margin within {b2}%": 100 * (right & ((g.y_margin - fm).abs() <= b2)).mean(),
                     "exact score%": 100 * ((g.final_home == hs) & (g.final_away == as_)).mean(),
                     "mkt MAE margin": (g.y_margin + g.spread_home).abs().mean() if g.spread_home.notna().all() else np.nan,
                     "mkt MAE total": (g.y_total - g.total_line).abs().mean() if g.total_line.notna().all() else np.nan})
    L.append(md(pd.DataFrame(rows), 3))
    L.append("")
    L.append("Final predicted score = rounded predicted total and margin, with the margin forced to at least "
             f"{cfg['min_margin']} point toward the predicted winner (so the score always agrees with the pick).")
    L.append("")

    # 3. tiers
    seasons = sorted(sc.season.unique())
    L.append("## 3. Win rate by confidence tier")
    L.append("")
    L.append("Cells are `n / win%` per season. Pushes excluded. `break-even%` = moneyline: the average vigged "
             "implied probability of the picked side's closing price; spread/total: -110.")
    L.append("")
    summ = {}
    for lab, prob, hit, be, sub in [
            ("(a) Moneyline winner (games with a closing line)", "pick_prob", "ml_hit", "pick_breakeven",
             sc[sc.has_mkt]),
            ("(b) Spread pick (P(cover) from predicted margin distribution)", "spread_prob", "spread_hit", 0.5238,
             sc[sc.has_mkt]),
            ("(c) Total pick (P(over/under) from predicted total distribution)", "total_prob", "total_hit", 0.5238,
             sc[sc.has_mkt])]:
        T, best = tier_table(sub, prob, hit, be, seasons)
        L.append(f"### {lab}")
        L.append("")
        L.append(md(T, 1))
        L.append("")
        if best is not None:
            L.append(f"**Best tier with >=100 games {REPORT}+: {best['tier']}% -> {best['win%']:.1f}% "
                     f"(n={best['n']}).**")
            summ[hit] = (best["tier"], int(best["n"]), float(best["win%"]))
        L.append("")
        cum = []
        s2 = sub[sub[hit].notna() & sub[prob].notna()]
        for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            g = s2[s2[prob] >= thr]
            if len(g):
                cum.append({"P(pick) >=": thr, "n": len(g), "win%": 100 * g[hit].mean()})
        L.append("Cumulative (P(pick) at or above threshold): " + "; ".join(
            f">={c['P(pick) >=']:.2f}: {c['win%']:.1f}% (n={c['n']})" for c in cum))
        L.append("")

    # 4. parlays
    L.append("## 4. Daily top-N confidence parlays (moneyline picks)")
    L.append("")
    L.append(f"Each date with at least N games, the N most confident model picks parlayed; {REPORT}+. "
             "`model-expected%` = mean product of the model's pick probabilities (a calibration check); "
             "the last column parlays the N biggest market favourites instead.")
    L.append("")
    sub = sc[sc.has_mkt & sc.y_win.notna()].copy()
    sub["mkt_hit2"] = sub.mkt_hit
    PT = parlay_table(sub, "pick_prob", "ml_hit", "mkt_pick_prob", "mkt_hit2")
    L.append("Games with a line:")
    L.append("")
    L.append(md(PT, 1))
    L.append("")
    if (~sc.has_mkt).any():
        PA = parlay_table(sc[sc.y_win.notna()], "pick_prob", "ml_hit")
        L.append("All games (incl. no-line games):")
        L.append("")
        L.append(md(PA, 1))
        L.append("")
    top3 = PT.loc[PT["top-N"] == 3].iloc[0]

    # 5. formula
    L.append("## 5. The formula in plain English")
    L.append("")
    fm = final["mkt"]
    ch = choice["mkt"]
    o, _ = res[("mkt", ch["mode"])]
    q = o[(o.season >= SEL0) & o.y_margin.notna()].join(d[["mkt_margin", "margin", "total", "total_line",
                                                          "mkt_logit", "script_logit"]])
    Xm = np.c_[q.mkt_margin, q.margin - q.mkt_margin]
    lm = LinearRegression().fit(Xm, q.y_margin)
    Xt = np.c_[q.total_line, q.total - q.total_line]
    lt = LinearRegression().fit(Xt, q.y_total)
    k = q.y_win.notna()
    lc = LogisticRegression(C=100, max_iter=1000).fit(np.c_[q.mkt_logit, q.script_logit - q.mkt_logit][k], q.y_win[k])
    L.append(f"Compact version (OLS / logistic on all out-of-sample {SEL0}+ games with a line; this is what the "
             "full model boils down to):")
    L.append("")
    L.append(f"- **predicted home margin = {lm.intercept_:+.2f} + {lm.coef_[0]:.3f} x (market margin = -closing "
             f"spread) + {lm.coef_[1]:.3f} x (script margin - market margin)**")
    L.append(f"- **predicted total = {lt.intercept_:+.2f} + {lt.coef_[0]:.3f} x closing total + {lt.coef_[1]:.3f} x "
             f"(script total - closing total)**")
    L.append(f"- **logit P(home win) = {lc.intercept_[0]:+.3f} + {lc.coef_[0, 0]:.3f} x logit(de-vigged market "
             f"P(home)) + {lc.coef_[0, 1]:.3f} x (script logit - market logit)**")
    L.append("- home score = (total + margin) / 2, away score = (total - margin) / 2.")
    L.append("")
    mc, mmeta, mt = fm["meta"]
    L.append("Stack weights of the deployed model (fit on all out-of-sample seasons):")
    L.append("")
    L.append(f"- logit P(home) = {mc.intercept_[0]:+.3f} + {mc.coef_[0, 0]:.3f} x logit(logistic) + "
             f"{mc.coef_[0, 1]:.3f} x logit(LightGBM-from-market) + {mc.coef_[0, 2]:.3f} x logit(market)")
    L.append(f"- margin = {mmeta.intercept_:+.2f} + {mmeta.coef_[0]:.3f} x ridge + {mmeta.coef_[1]:.3f} x "
             f"LightGBM + {mmeta.coef_[2]:.3f} x market margin")
    L.append(f"- total = {mt.intercept_:+.2f} + {mt.coef_[0]:.3f} x ridge + {mt.coef_[1]:.3f} x LightGBM + "
             f"{mt.coef_[2]:.3f} x market total")
    L.append("")
    M = fm["models"]
    for lab, pipe in [("Logistic P(home win)", M["lr"]), ("Ridge margin", M["rm"]), ("Ridge total", M["rt"])]:
        imp, sclr, est = pipe[0], pipe[1], pipe[-1]
        names = list(M["F"]) + [f"missing({M['F'][i]})" for i in imp.indicator_.features_] if imp.add_indicator else list(M["F"])
        coef = est.coef_.ravel()
        raw = coef / sclr.scale_
        T = pd.DataFrame({"feature": names, "per 1 sd": coef, "per raw unit": raw})
        T = T.reindex(T["per 1 sd"].abs().sort_values(ascending=False).index).head(12)
        unit = "log-odds" if "Logistic" in lab else "points"
        L.append(f"Top features, {lab} (deployed fit; {unit}):")
        L.append("")
        L.append(md(T.reset_index(drop=True), 3))
        L.append("")
    gi = pd.Series(M["gc"].booster_.feature_importance("gain"), index=M["F"]).sort_values(ascending=False).head(10)
    L.append("LightGBM (boosted from the market logit) top gain features: " +
             ", ".join(f"{k} ({v:.0f})" for k, v in gi.items()))
    L.append("")

    # 6. ablation
    L.append("## 6. Ablation (what each feature group adds)")
    L.append("")
    L.append(f"Logistic + ridge, season refits, games with a line, {REPORT}+. Groups: market = "
             f"{', '.join(groups['market'])}; blend = script-x-market blends ({len(groups['blend'])} cols); "
             f"script = {len(groups['script'])} Kalman/enrichment cols; context = {len(groups['context'])} cols "
             f"({', '.join(groups['context'][:12])}{'...' if len(groups['context']) > 12 else ''}).")
    L.append("")
    L.append(md(abl, 4))
    L.append("")

    # 7. history
    H, (pcol, mcol, tcol) = hist
    L.append("## 7. Does more history help?")
    L.append("")
    L.append("Evaluation fixed at 2023+ seasons (games with a line). The chosen base models are retrained with only "
             "the last k seasons (that have closing lines) before each test season (or all prior seasons, optionally with exponential "
             f"recency weights by season), then combined with that season's stack layer. Win model shown: "
             f"**{pcol}**; margin **{mcol}**, total **{tcol}** (when the deployed choice is the bare market line, the "
             "learned stack/ridge is shown instead, since the market line itself has no training window). The "
             "high-confidence column uses the best moneyline tier's lower bound from section 3.")
    L.append("")
    L.append(md(H, 4))
    L.append("")
    a = H.set_index("training window")
    L.append(f"Read: win% last-1 {a.iloc[0]['SU win%']:.2f} -> all {a.loc['all prior', 'SU win%']:.2f}; log loss "
             f"{a.iloc[0]['logloss']:.4f} -> {a.loc['all prior', 'logloss']:.4f}; margin MAE "
             f"{a.iloc[0]['MAE margin']:.3f} -> {a.loc['all prior', 'MAE margin']:.3f}. " + history_verdict(H))
    L.append("")

    # 8. live
    L.append("## 8. Live use")
    L.append("")
    L.append(f"`from engine.predict.{sport} import predict; predict(slate_df)` with rows shaped like "
             f"`results/scripts/{sport}_games.parquet` (the script engine's pre-game columns for upcoming games) plus "
             f"`spread_home`, `total_line`, `home_ml`, `away_ml` (and `spread_open_home`/`total_open` if known). "
             f"Rows without `spread_home` use the script-only model. Model: `results/predict/{sport}_model.pkl` "
             f"(trained through {d.loc[d.y_margin.notna(), 'date'].max().date()}). Required script columns: "
             f"{', '.join(sorted(set(groups['script'] + groups['context'] + groups['blend'])))}.")
    L.append("")
    summary = dict(sport=name, n=len(scm), acc_model=acc_m, acc_mkt=acc_k,
                   mae_margin=(sc.y_margin - sc.pred_margin)[sc.has_mkt].abs().mean(),
                   mae_total=(sc.y_total - sc.pred_total)[sc.has_mkt].abs().mean(),
                   best_ml=summ.get("ml_hit"), top3=(float(top3["hit%"]), int(top3["days"])),
                   best_ats=summ.get("spread_hit"), best_ou=summ.get("total_hit"),
                   hist=history_verdict(H))
    return "\n".join(L), summary


def history_verdict(H):
    a = H.set_index("training window")
    wins = ["last 1", "last 2", "last 4", "last 6", "all prior"]
    ll, mae, acc = a.loc[wins, "logloss"], a.loc[wins, "MAE margin"], a.loc[wins, "SU win%"]
    k_ll = next(w for w in wins if ll[w] <= ll.min() + 0.001)
    k_m = next(w for w in wins if mae[w] <= mae.min() + 0.01)
    dec = a.loc[[i for i in a.index if "decay" in i]]
    bd = dec["logloss"].idxmin()
    span = acc.max() - acc.min()
    return (f"Win% moves only {span:.2f} pts across windows (noise level). Log loss is within 0.001 of its best "
            f"by '{k_ll}' (last-1 {ll['last 1']:.4f} vs all {ll['all prior']:.4f}); margin MAE is within 0.01 of "
            f"its best by '{k_m}' (last-1 {mae['last 1']:.3f} vs all {mae['all prior']:.3f}). "
            + ("More history keeps helping a little. " if ll.idxmin() == "all prior" and mae.idxmin() == "all prior"
               else "It plateaus: beyond that, more seasons add nothing measurable. ")
            + f"Recency weighting: best decay = {bd} (log loss {dec.loc[bd, 'logloss']:.4f}, MAE margin "
            f"{dec.loc[bd, 'MAE margin']:.3f}) vs unweighted all-prior {ll['all prior']:.4f} / "
            f"{mae['all prior']:.3f}: " + ("no gain." if dec.loc[bd, "logloss"] >= ll["all prior"] - 0.0005
                                            else "a small gain.") +
            " The model is dominated by the closing line, which already embodies the latest information, so the "
            "learned layer has little to gain from extra history.")


def _old_history_verdict(H):
    a = H.set_index("training window")
    win = [a.loc[f"last {k}", "logloss"] for k in (1, 2, 4, 6)] + [a.loc["all prior", "logloss"]]
    mae = [a.loc[f"last {k}", "MAE margin"] for k in (1, 2, 4, 6)] + [a.loc["all prior", "MAE margin"]]
    dec = a.loc[[i for i in a.index if "decay" in i]]
    bestdec = dec["logloss"].idxmin()
    gain_1_4 = win[0] - win[2]
    gain_4_all = win[2] - win[4]
    if gain_4_all > 0.002:
        s = "Log loss keeps improving past 4 seasons"
    elif gain_1_4 > 0.002:
        s = "Most of the gain comes by ~4 seasons, then it plateaus"
    else:
        s = "History length barely matters (the market feature carries the model)"
    s += (f" (log loss: last1 {win[0]:.4f}, last4 {win[2]:.4f}, all {win[4]:.4f}; margin MAE last1 {mae[0]:.3f}, "
          f"all {mae[4]:.3f}). Best recency weighting: {bestdec} (log loss {dec.loc[bestdec, 'logloss']:.4f} vs "
          f"unweighted {win[4]:.4f}).")
    return s


# ------------------------------------------------------------------ live predict
def predict(sport, slate_df, model=None):
    if model is None:
        with open(OUT / f"{sport}_model.pkl", "rb") as f:
            model = pickle.load(f)
    s = slate_df.copy().reset_index(drop=True)
    if "margin" not in s or "sd_margin" not in s or "total" not in s:
        raise ValueError("slate needs script columns margin, sd_margin, total (from the script engine)")
    d, _ = add_features(s, model["spread_b"])
    allF = set(model["final"]["mkt"]["models"]["F"]) | set(model["final"]["nomkt"]["models"]["F"])
    for c in allF:
        if c not in d:
            d[c] = np.nan
    out = []
    for v, msk in [("mkt", d.has_mkt), ("nomkt", ~d.has_mkt)]:
        if not msk.any():
            continue
        fv = model["final"][v]
        te = d[msk]
        B = pred_bases(fv["models"], te)
        B["st_p"], B["st_m"], B["st_t"] = apply_meta(fv["meta"], B)
        ch = fv["choice"]
        r = finalize(te, B[PCOL[ch["p"]]].to_numpy(), B[MCOL[ch["m"]]].to_numpy(), B[TCOL[ch["t"]]].to_numpy(),
                     fv["sd_m"], fv["sd_t"], model["min_margin"], B[MCOL[ch["m_ats"]]].to_numpy(),
                     B[TCOL[ch["t_ou"]]].to_numpy())
        r["model_variant"] = v
        out.append(r)
    r = pd.concat(out).sort_index()
    res = pd.DataFrame({"home": d.home, "away": d.away, "p_home": r.p_home,
                        "pick": np.where(r.pick_home, d.home, d.away), "pick_prob": r.pick_prob,
                        "pred_home_score": r.final_home, "pred_away_score": r.final_away,
                        "pred_margin": r.pred_margin, "pred_total": r.pred_total,
                        "spread_pick": np.where(r.spread_prob.isna(), None,
                                                np.where(r.spread_pick_home == 1, d.home, d.away)),
                        "spread_prob": r.spread_prob,
                        "total_pick": np.where(r.total_prob.isna(), None,
                                               np.where(r.total_pick_over == 1, "over", "under")),
                        "total_prob": r.total_prob, "tier": r.tier, "model_variant": r.model_variant})
    return res
