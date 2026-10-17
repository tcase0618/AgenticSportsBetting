"""UFC fight predictor: winner + method + round ("final score" of a fight). Walk-forward, see engine/predict/SPEC.md.

    python -m engine.predict.ufc     -> results/predict/ufc_predict.md, ufc_preds.parquet, ufc_model.pkl

Live: `predict(slate_df)`; slate_df rows look like results/scripts/ufc_fights.parquet rows for upcoming fights (script
columns from engine.scripts.ufc: p_A_win, p_A_KO..p_B_DEC, p_distance, p_end_r1..r5, skill_a/b, p_A_skill_only,
n_ufc_a/b, weight_class, gender, scheduled_rounds, title_bout; optional market a_ml, b_ml and de-vigged method props
mkt_A_KO..mkt_B_DEC) plus, when available, the pre-fight fighter stats of data/processed/ufc.csv (a_age, b_age,
a_reach_cm, ..., corner). Missing stats are filled with 0 and flagged.
Fighter A = red corner when corner == 'red/blue'.
"""
from __future__ import annotations

import pickle
import time
import warnings


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from engine.predict.common_count import (OUT, ROOT, TIER_LABELS, implied, logit, logloss, md, parlay_table, safe_devig,
                                         tier_of, tier_table, best_tier, win_metrics, small_cols, clipper)

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

OUTS = ["A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC"]
METH = ["KO", "SUB", "DEC"]
WCLASS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
          "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 250, "Catch Weight": 160}
STATS = ["age", "reach_cm", "height_cm", "days_since_last_fight", "fights_ufc", "win_streak", "lose_streak", "wins", "losses",
         "ko_wins", "sub_wins", "dec_wins", "ko_losses", "sig_str_landed_pm", "sig_str_absorbed_pm", "sig_str_acc", "td_avg",
         "td_acc", "td_def", "sub_avg", "kd_avg", "ctrl_share", "ufc_minutes", "title_bouts"]
LGB_UFC = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=1)

GROUPS = {
    "script": ["s_logit", "skill_diff", "skill_only_logit", "dist_logit", "ls_A_KO", "ls_A_SUB", "ls_A_DEC", "ls_B_KO", "ls_B_SUB",
               "ls_B_DEC", "itd_diff"],
    "market methods": ["lm_A_KO", "lm_A_SUB", "lm_A_DEC", "lm_B_KO", "lm_B_SUB", "lm_B_DEC"],
    "physical": ["age_diff", "age_a", "age_b", "reach_diff", "height_diff", "young_light"],
    "experience/record": ["layoff_a", "layoff_b", "fights_diff", "n_ufc_a_ln", "n_ufc_b_ln", "win_streak_diff", "lose_streak_diff",
                          "wins_diff", "losses_diff", "ko_losses_diff", "title_bouts_diff"],
    "style stats": ["slpm_diff", "sapm_diff", "stracc_diff", "td_avg_diff", "td_def_diff", "sub_avg_diff", "kd_avg_diff",
                    "ctrl_diff", "ko_wins_diff", "sub_wins_diff", "dec_wins_diff"],
    "context": ["five_rds", "title_bout", "female", "weight_lb", "corner_sign"],
}
MARKET = ["mkt_logit", "mkt_gap"]


def feats_of(drop=()):
    f = [] if "market" in drop else list(MARKET)
    for g, c in GROUPS.items():
        if g not in drop:
            f += c
    return f


def build(raw: pd.DataFrame, csv: pd.DataFrame | None = None) -> pd.DataFrame:
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    for c in ["winner", "method_cat", "finish_round", "fight_id"]:
        if c not in d:
            d[c] = np.nan if c != "fight_id" else np.arange(len(d)).astype(str)
    if csv is not None:
        cols = ["date", "fighter_a", "fighter_b", "corner"] + [f"{s}_{c}" for c in STATS for s in "ab"]
        u = csv[[c for c in cols if c in csv.columns]].drop_duplicates(["date", "fighter_a", "fighter_b"])
        d = d.merge(u, on=["date", "fighter_a", "fighter_b"], how="left")
    for c in [f"{s}_{x}" for x in STATS for s in "ab"] + ["corner", "a_ml", "b_ml"] + [f"mkt_{o}" for o in OUTS]:
        if c not in d:
            d[c] = np.nan
    d["y"] = np.where(d.winner == "A", 1.0, np.where(d.winner == "B", 0.0, np.nan))
    d["meth"] = d.method_cat.where(d.method_cat.isin(METH))
    d["y6"] = np.where(d.y.notna() & d.meth.notna(), d.winner.astype(str) + "_" + d.meth.astype(str), None)
    d["end_round"] = pd.to_numeric(d.finish_round, errors="coerce")
    p = safe_devig(d.a_ml.values, d.b_ml.values, max_over=1.15)
    broken = np.isfinite(p) & (np.abs(p - d.p_A_win.values) > 0.45)
    p[broken] = np.nan
    d["mkt_broken"] = broken.astype(int)
    d["p_mkt"] = p
    d["has_mkt"] = np.isfinite(p).astype(float)
    d["s_logit"] = logit(d.p_A_win)
    d["mkt_logit"] = np.where(d.has_mkt > 0, logit(np.nan_to_num(p, nan=0.5)), d.s_logit)
    d["mkt_gap"] = d.mkt_logit - d.s_logit
    d["be_a"] = implied(d.a_ml.values)
    d["be_b"] = implied(d.b_ml.values)
    d["skill_diff"] = d.skill_a - d.skill_b
    d["skill_only_logit"] = logit(d.p_A_skill_only)
    d["dist_logit"] = logit(d.p_distance)
    d["itd_diff"] = d.p_A_itd - d.p_B_itd
    mm = d[[f"mkt_{o}" for o in OUTS]].astype(float)
    d["has_mm"] = mm.notna().all(1).astype(float)
    for o in OUTS:
        d[f"ls_{o}"] = np.log(np.clip(d[f"p_{o}"].astype(float), 1e-3, 1))
        d[f"lm_{o}"] = np.where(d.has_mm > 0, np.log(np.clip(mm[f"mkt_{o}"].fillna(0.1), 1e-3, 1)), d[f"ls_{o}"])
    a = lambda c: pd.to_numeric(d[f"a_{c}"], errors="coerce")
    b = lambda c: pd.to_numeric(d[f"b_{c}"], errors="coerce")
    d["age_a"], d["age_b"] = a("age"), b("age")
    d["age_diff"] = d.age_a - d.age_b
    d["has_phys"] = d.age_diff.notna().astype(float)
    d["reach_diff"] = a("reach_cm") - b("reach_cm")
    d["height_diff"] = a("height_cm") - b("height_cm")
    d["layoff_a"] = np.log1p(a("days_since_last_fight").clip(lower=0))
    d["layoff_b"] = np.log1p(b("days_since_last_fight").clip(lower=0))
    d["fights_diff"] = a("fights_ufc") - b("fights_ufc")
    d["n_ufc_a_ln"] = np.log1p(d.n_ufc_a.astype(float))
    d["n_ufc_b_ln"] = np.log1p(d.n_ufc_b.astype(float))
    for c, k in [("win_streak", "win_streak_diff"), ("lose_streak", "lose_streak_diff"), ("wins", "wins_diff"),
                 ("losses", "losses_diff"), ("ko_losses", "ko_losses_diff"), ("title_bouts", "title_bouts_diff"),
                 ("sig_str_landed_pm", "slpm_diff"), ("sig_str_absorbed_pm", "sapm_diff"), ("sig_str_acc", "stracc_diff"),
                 ("td_avg", "td_avg_diff"), ("td_def", "td_def_diff"), ("sub_avg", "sub_avg_diff"), ("kd_avg", "kd_avg_diff"),
                 ("ctrl_share", "ctrl_diff"), ("ko_wins", "ko_wins_diff"), ("sub_wins", "sub_wins_diff"), ("dec_wins", "dec_wins_diff")]:
        d[k] = a(c) - b(c)
    d["five_rds"] = (d.scheduled_rounds.astype(float) >= 5).astype(float)
    d["female"] = (d.gender == "F").astype(float)
    wc = d.weight_class.astype(str).str.replace("Women's ", "", regex=False)
    d["weight_lb"] = wc.map(lambda s: next((v for k, v in sorted(WCLASS.items(), key=lambda kv: -len(kv[0])) if k in s), 160)).astype(float)
    d["corner_sign"] = (d.corner == "red/blue").astype(float)   # +1 A is red corner; 0 unknown ('alpha' = alphabetical, most of 2026)
    # KEYS.md 'youth over experience': age gap matters most at 155 lb and below (incl. all women's classes)
    d["young_light"] = d.age_diff.fillna(0) * ((d.weight_lb <= 155) | (d.female > 0)).astype(float)
    d["title_bout"] = d.title_bout.astype(float)
    for c in feats_of():
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for r in range(1, 6):
        d[f"p_end_r{r}"] = d[f"p_end_r{r}"].astype(float).fillna(0)
    return d.sort_values(["date", "fight_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- models
SWAP = [("age_a", "age_b"), ("layoff_a", "layoff_b"), ("n_ufc_a_ln", "n_ufc_b_ln")] + \
       [(f"{p}_A_{m}", f"{p}_B_{m}") for p in ("ls", "lm") for m in METH]
NEG = ["mkt_logit", "mkt_gap", "s_logit", "skill_diff", "skill_only_logit", "itd_diff", "age_diff", "young_light", "reach_diff",
       "height_diff", "corner_sign"] + [c for c in sum(GROUPS.values(), []) if c.endswith("_diff")]


def augment(tr, w=None):
    """Mirror every fight (A<->B) so the models are corner-symmetric; the corner advantage lives in corner_sign only.
    Without this the intercept learns 'A = red corner wins 57%', which is wrong for 2026 fights listed alphabetically."""
    sw = tr.copy()
    for a, b in SWAP:
        sw[a], sw[b] = tr[b].values, tr[a].values
    for c in dict.fromkeys(NEG):
        sw[c] = -tr[c].values
    sw["y"] = 1 - tr.y.values
    sw["y6"] = tr.y6.map(lambda v: v if not isinstance(v, str) else ("B" + v[1:] if v[0] == "A" else "A" + v[1:]))
    w2 = None if w is None else np.r_[w, w]
    return pd.concat([tr, sw], ignore_index=True), w2

def _lr(C, multi=False):
    return make_pipeline(StandardScaler(), clipper(), LogisticRegression(C=C, max_iter=4000))


def fit_win(tr, feats, kind, w=None):
    tr, w = augment(tr, w)
    y = tr.y.values.astype(int)
    if kind == "lr_small":
        return _lr(1.0).fit(tr[small_cols(feats)].values, y, logisticregression__sample_weight=w)
    if kind == "lr_mid":
        return _lr(0.3).fit(tr[[c for c in MID if c in feats] or ["s_logit"]].values, y, logisticregression__sample_weight=w)
    if kind == "lr_full":
        return _lr(0.03).fit(tr[feats].values, y, logisticregression__sample_weight=w)
    if kind == "lgbm":
        return lgb.LGBMClassifier(**LGB_UFC).fit(tr[feats].values, y, sample_weight=w)


def pred_win(m, d, feats, kind):
    X = d[small_cols(feats)].values if kind == "lr_small" else (d[[c for c in MID if c in feats] or ["s_logit"]].values if kind == "lr_mid" else d[feats].values)
    return m.predict_proba(X)[:, 1]


METH_SMALL = [f"ls_{o}" for o in OUTS] + [f"lm_{o}" for o in OUTS] + ["mkt_logit", "s_logit"]
MID = ["mkt_logit", "s_logit", "age_diff", "young_light", "reach_diff", "layoff_a", "layoff_b", "skill_diff"]


def meth_feats(feats):
    return list(feats)


def fit_meth(tr, feats, kind, w=None):
    tr, w = augment(tr, w)
    t = tr[tr.y6.notna()]
    ww = None if w is None else w[tr.y6.notna().values]
    X = t[METH_SMALL].values if kind == "mnl_small" else t[meth_feats(feats)].values
    m = _lr(0.3 if kind == "mnl_small" else 0.03).fit(X, t.y6.values, logisticregression__sample_weight=ww)
    return m


def pred_meth(m, d, feats, kind):
    X = d[METH_SMALL].values if kind == "mnl_small" else d[meth_feats(feats)].values
    P = m.predict_proba(X)
    cls = list(m.classes_)
    return np.column_stack([P[:, cls.index(o)] if o in cls else np.full(len(d), 1e-4) for o in OUTS])


RF = ["lr_end1", "lr_end2", "lr_end3", "lr_end4", "lr_end5", "five_rds", "female", "weight_lb", "is_sub", "dist_logit"]


def round_X(d, is_sub):
    pe = d[[f"p_end_r{r}" for r in range(1, 6)]].values
    pe = pe / np.clip(pe.sum(1, keepdims=True), 1e-9, None)
    X = pd.DataFrame(np.log(np.clip(pe, 1e-3, 1)), columns=RF[:5], index=d.index)
    X["five_rds"], X["female"], X["weight_lb"], X["dist_logit"] = d.five_rds, d.female, d.weight_lb, d.dist_logit
    X["is_sub"] = is_sub
    return X[RF].values


def fit_round(tr, w=None):
    t = tr[tr.meth.isin(["KO", "SUB"]) & tr.end_round.between(1, 5)]
    ww = None if w is None else w[(tr.meth.isin(["KO", "SUB"]) & tr.end_round.between(1, 5)).values]
    return _lr(0.3).fit(round_X(t, (t.meth == "SUB").astype(float).values), t.end_round.astype(int).values,
                        logisticregression__sample_weight=ww)


def pred_round(m, d, is_sub):
    P = m.predict_proba(round_X(d, np.full(len(d), float(is_sub))))
    cls = list(m.classes_)
    out = np.column_stack([P[:, cls.index(r)] if r in cls else np.zeros(len(d)) for r in range(1, 6)])
    out[d.scheduled_rounds.values.astype(float) < 5, 3:] = 0
    return out / out.sum(1, keepdims=True)


def script_round(d):
    pe = d[[f"p_end_r{r}" for r in range(1, 6)]].values
    return pe / np.clip(pe.sum(1, keepdims=True), 1e-9, None)


# ----------------------------------------------------------------------------------------------- walk
def walk(d, drop=(), freq="M", window=None, half_life=None, first_season=2018, verbose=True,
         win_kinds=("lr_small", "lr_mid", "lr_full", "lgbm"), meth_kinds=("mnl_small", "mnl_full"), rounds=True):
    feats = feats_of(drop)
    pr = d[d.season >= first_season]
    key = pr.date.dt.to_period("M").astype(str) if freq == "M" else pr.season.astype(str)
    res = []
    t0 = time.time()
    groups = sorted(pr.groupby(key).groups.items(), key=lambda kv: d.loc[kv[1], "date"].min())
    for i, (k, idx) in enumerate(groups):
        te = d.loc[idx]
        start = te.date.min()
        season = int(te.season.iloc[0])
        tr = d[(d.date < start) & d.y.notna()]
        w = np.ones(len(tr))
        if window is not None:
            keep = tr.season.values >= season - window
            tr, w = tr[keep], w[keep]
        if half_life is not None:
            w = 0.5 ** (np.clip(season - 1 - tr.season.values, 0, None) / half_life)
        o = pd.DataFrame(index=idx)
        o["period"] = k
        for kind in win_kinds:
            o[kind] = pred_win(fit_win(tr, feats, kind, w), te, feats, kind)
        if "lr_full" in win_kinds and "lgbm" in win_kinds:
            prev = pd.concat(res) if res else pd.DataFrame()
            if len(prev) >= 600:
                pj = prev.join(d[["y", "mkt_logit", "has_mkt"]])
                pj = pj[pj.y.notna()]
                st = LogisticRegression(max_iter=2000).fit(np.c_[logit(pj.lr_full), logit(pj.lgbm), pj.mkt_logit, pj.has_mkt], pj.y.astype(int))
                o["stack"] = st.predict_proba(np.c_[logit(o.lr_full), logit(o.lgbm), te.mkt_logit, te.has_mkt])[:, 1]
            else:
                o["stack"] = o.lr_full
        for mk in meth_kinds:
            P6 = pred_meth(fit_meth(tr, feats, mk, w), te, feats, mk)
            for j, oc in enumerate(OUTS):
                o[f"{mk}_{oc}"] = P6[:, j]
        if rounds:
            rm = fit_round(tr, w)
            for mth, flag in [("KO", 0), ("SUB", 1)]:
                R = pred_round(rm, te, flag)
                for r in range(5):
                    o[f"rd_{mth}_{r+1}"] = R[:, r]
        res.append(o)
        if verbose and i % 24 == 0:
            print(f"  [ufc] period {k} train={len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(res)
    ev = d.loc[out.index]
    out["market"] = np.where(ev.has_mkt > 0, ev.p_mkt, ev.p_A_win)
    out["script"] = ev.p_A_win.values
    out["blend"] = ev.p_A_win_blend.values
    for j, oc in enumerate(OUTS):
        out[f"script6_{oc}"] = ev[f"p_{oc}"].values
        out[f"blend6_{oc}"] = ev[f"blend_{oc}"].values
    sr = script_round(ev)
    for r in range(5):
        out[f"srd_{r+1}"] = sr[:, r]
    return out


def method_acc(w, kind, ev, mask):
    P6 = w[[f"{kind}_{o}" for o in OUTS]].values[mask]
    marg = np.c_[P6[:, 0] + P6[:, 3], P6[:, 1] + P6[:, 4], P6[:, 2] + P6[:, 5]]
    act = ev.meth.values[mask]
    ok = pd.notna(act)
    return 100 * float(np.mean(np.array(METH)[marg.argmax(1)][ok] == act[ok]))


def ll6(P, y6):
    ok = pd.notna(y6)
    P = np.asarray(P)[ok]
    P = P / P.sum(1, keepdims=True)
    idx = np.array([OUTS.index(v) for v in np.asarray(y6)[ok]])
    return float(-np.mean(np.log(np.clip(P[np.arange(len(idx)), idx], 1e-6, 1))))


def picks(d, w, win_kind, meth_kind, round_kind):
    """Combine winner prob, 6-way method probs and round model into the predicted fight result."""
    o = pd.DataFrame(index=w.index)
    p = w[win_kind].values
    P6 = w[[f"{meth_kind}_{oc}" for oc in OUTS]].values
    P6 = P6 / P6.sum(1, keepdims=True)
    pick_a = p >= 0.5
    cond = np.where(pick_a[:, None], P6[:, :3], P6[:, 3:])
    cond = cond / cond.sum(1, keepdims=True)
    mi = cond.argmax(1)
    meth = np.array(METH)[mi]
    ev = d.loc[w.index]
    sched = ev.scheduled_rounds.values.astype(int)
    if round_kind == "script":
        R = w[[f"srd_{r}" for r in range(1, 6)]].values
        Rko = Rsub = R
    else:
        Rko = w[[f"rd_KO_{r}" for r in range(1, 6)]].values
        Rsub = w[[f"rd_SUB_{r}" for r in range(1, 6)]].values
    Rm = np.where((meth == "SUB")[:, None], Rsub, Rko)
    rnd = np.where(meth == "DEC", sched, Rm.argmax(1) + 1)
    p_round = np.where(meth == "DEC", 1.0, Rm.max(1))
    o["p_A"] = p
    o["pick_A"] = pick_a.astype(int)
    o["pick"] = np.where(pick_a, ev.fighter_a, ev.fighter_b)
    o["pick_prob"] = np.maximum(p, 1 - p)
    o["pred_method"] = meth
    o["p_method_given_win"] = cond.max(1)
    o["p_win_method"] = o.pick_prob * o.p_method_given_win
    o["pred_round"] = rnd
    o["p_exact"] = o.p_win_method * p_round
    marg = np.c_[P6[:, 0] + P6[:, 3], P6[:, 1] + P6[:, 4], P6[:, 2] + P6[:, 5]]
    o["p_KO"], o["p_SUB"], o["p_DEC"] = marg[:, 0], marg[:, 1], marg[:, 2]
    o["method_mode"] = np.array(METH)[marg.argmax(1)]
    o["method_conf"] = marg.max(1)
    # alt round pick: unconditional most likely ending round (finish rounds vs distance)
    pdist = marg[:, 2]
    pr_end = (1 - pdist)[:, None] * (marg[:, [0]] * Rko + marg[:, [1]] * Rsub) / np.clip((marg[:, 0] + marg[:, 1])[:, None], 1e-9, None)
    pr_end[np.arange(len(sched)), sched - 1] += pdist
    o["round_mode"] = pr_end.argmax(1) + 1
    o["p_round_mode"] = pr_end.max(1)
    o["predicted"] = [f"{pk} by {m}" + ("" if m == "DEC" else f" R{r}") for pk, m, r in zip(o.pick, meth, rnd)]
    # outcomes
    o["y"] = ev.y.values
    o["win_hit"] = np.where(ev.y.isna(), np.nan, (o.pick_A == ev.y).astype(float))
    valid = ev.meth.notna() & ev.y.notna()
    o["meth_hit"] = np.where(ev.meth.isna(), np.nan, (meth == ev.meth.values).astype(float))
    o["wm_hit"] = np.where(valid, ((o.win_hit == 1) & (o.meth_hit == 1)).astype(float), np.nan)
    o["round_hit"] = np.where(valid, (rnd == ev.end_round.values).astype(float), np.nan)
    o["round_mode_hit"] = np.where(valid, (o.round_mode.values == ev.end_round.values).astype(float), np.nan)
    o["exact_hit"] = np.where(valid, ((o.wm_hit == 1) & (o.round_hit == 1)).astype(float), np.nan)
    o["method_mode_hit"] = np.where(ev.meth.isna(), np.nan, (o.method_mode == ev.meth.values).astype(float))
    return o


def run():
    T0 = time.time()
    raw = pd.read_parquet(ROOT / "results" / "scripts" / "ufc_fights.parquet")
    csv = pd.read_csv(ROOT / "data" / "processed" / "ufc.csv", low_memory=False)
    csv["date"] = pd.to_datetime(csv.date)
    d = build(raw, csv)
    feats = feats_of()
    print(f"[ufc] {len(d)} fights, {len(feats)} features", flush=True)
    w = walk(d)
    ev = d.loc[w.index]
    sel = ev.season.between(2018, 2020)
    rep = ev.season >= 2021
    # ----- selection
    rows = []
    for m in ["market", "script", "blend", "lr_small", "lr_mid", "lr_full", "lgbm", "stack"]:
        a = win_metrics(w.loc[sel, m], ev.loc[sel, "y"])
        b = win_metrics(w.loc[rep, m], ev.loc[rep, "y"])
        rows.append({"model": m, "ll_2018_20": a["ll"], "acc%_2018_20": a["acc"], "ll_2021+": b["ll"], "acc%_2021+": b["acc"]})
    sel_tab = pd.DataFrame(rows).set_index("model")
    win_kind = sel_tab.drop(index=["blend"]).ll_2018_20.idxmin()
    rows = []
    for mk in ["script6", "blend6", "mnl_small", "mnl_full"]:
        P = w[[f"{mk}_{o}" for o in OUTS]].values
        rows.append({"method_model": mk, "ll6_2018_20": ll6(P[sel.values], ev.loc[sel, "y6"].values),
                     "ll6_2021+": ll6(P[rep.values], ev.loc[rep, "y6"].values)})
    meth_tab = pd.DataFrame(rows).set_index("method_model")
    meth_kind = meth_tab.ll6_2018_20.idxmin()
    # round model choice: log loss of actual finish round given actual finish method, 2018-2020
    rows = []
    fin = ev.meth.isin(["KO", "SUB"]) & ev.end_round.between(1, 5)
    for rk in ["script", "lr"]:
        lls = {}
        for nm, mask in [("2018_20", sel & fin), ("2021+", rep & fin)]:
            e = ev[mask]
            if rk == "script":
                R = w.loc[mask, [f"srd_{r}" for r in range(1, 6)]].values
            else:
                R = np.where((e.meth == "SUB").values[:, None], w.loc[mask, [f"rd_SUB_{r}" for r in range(1, 6)]].values,
                             w.loc[mask, [f"rd_KO_{r}" for r in range(1, 6)]].values)
            pr = R[np.arange(len(e)), e.end_round.astype(int).values - 1]
            lls[nm] = float(-np.mean(np.log(np.clip(pr, 1e-6, 1))))
            lls[f"acc%_{nm}"] = 100 * float((R.argmax(1) + 1 == e.end_round.values).mean())
        rows.append({"round_model": rk, "ll_2018_20": lls["2018_20"], "acc%_2018_20": lls["acc%_2018_20"], "ll_2021+": lls["2021+"], "acc%_2021+": lls["acc%_2021+"]})
    round_tab = pd.DataFrame(rows).set_index("round_model")
    round_kind = round_tab.ll_2018_20.idxmin()
    print(f"[ufc] win={win_kind} method={meth_kind} round={round_kind} ({time.time()-T0:.0f}s)", flush=True)

    pk = picks(d, w, win_kind, meth_kind, round_kind)
    P = ev[["fight_id", "date", "season", "event", "fighter_a", "fighter_b", "weight_class", "scheduled_rounds", "winner",
            "method_cat", "end_round", "p_mkt", "has_mkt", "a_ml", "b_ml", "be_a", "be_b", "p_A_win", "mkt_broken"]].copy()
    for m in ["market", "script", "lr_small", "lr_mid", "lr_full", "lgbm", "stack"]:
        P[f"p_{m}"] = w[m]
    for oc in OUTS:
        P[f"p6_{oc}"] = w[f"{meth_kind}_{oc}"]
    P = P.join(pk)
    P["tier"] = tier_of(P.pick_prob).astype(str)
    P["be_pick"] = np.where(P.pick_A == 1, P.be_a, P.be_b)
    P["mkt_fav_A"] = np.where(P.has_mkt > 0, (P.p_mkt >= 0.5).astype(float), np.nan)
    P["mkt_hit"] = np.where(P.y.isna() | P.mkt_fav_A.isna(), np.nan, (P.mkt_fav_A == P.y).astype(float))
    P["mkt_pick_prob"] = np.where(P.has_mkt > 0, np.maximum(P.p_mkt, 1 - P.p_mkt), np.nan)
    P["script_hit"] = np.where(P.y.isna(), np.nan, ((P.p_A_win >= 0.5) == (P.y == 1)).astype(float))
    R = P[P.season >= 2021]
    T = P[P.season.between(2018, 2020)]

    # ----- per-season
    rows = []
    for s, g in P.groupby("season"):
        gm = g[g.has_mkt > 0]
        rows.append({"season": s, "fights": int(g.win_hit.notna().sum()), "winner%": 100 * g.win_hit.mean(), "script%": 100 * g.script_hit.mean(),
                     "fights_w_mkt": int(gm.mkt_hit.notna().sum()), "winner%_mkt_fights": 100 * gm.win_hit.mean(), "mkt_fav%": 100 * gm.mkt_hit.mean(),
                     "method%": 100 * g.meth_hit.mean(), "winner+method%": 100 * g.wm_hit.mean(), "round%": 100 * g.round_hit.mean(),
                     "exact(win+meth+rd)%": 100 * g.exact_hit.mean(), "ll_win": logloss(g.p_A[g.y.notna()], g.y.dropna())})
    season_tab = pd.DataFrame(rows).set_index("season")

    def summ(g, name):
        gm = g[g.has_mkt > 0]
        ok = g.meth_hit.notna()
        return {"sample": name, "fights": int(g.win_hit.notna().sum()), "winner%": 100 * g.win_hit.mean(), "script%": 100 * g.script_hit.mean(),
                "winner%_mkt_fights": 100 * gm.win_hit.mean(), "mkt_fav%": 100 * gm.mkt_hit.mean(), "fights_w_mkt": int(gm.mkt_hit.notna().sum()),
                "method% (method of predicted result)": 100 * g.meth_hit.mean(), "method% (most likely method overall)": 100 * g.method_mode_hit.mean(),
                "method% always-DEC baseline": 100 * (d.loc[g.index].meth[ok] == "DEC").mean(),
                "winner+method%": 100 * g.wm_hit.mean(),
                "round% (round of predicted result)": 100 * g.round_hit.mean(), "round% (most likely ending round)": 100 * g.round_mode_hit.mean(),
                "round% always-distance baseline": 100 * (d.loc[g.index].end_round[ok] == d.loc[g.index].scheduled_rounds[ok]).mean(),
                "winner+method+round%": 100 * g.exact_hit.mean(),
                "ll6 (winner x method)": ll6(g[[f"p6_{o}" for o in OUTS]].values, d.loc[g.index, "y6"].values)}
    sum_tab = pd.DataFrame([summ(T, "2018-2020"), summ(R, "2021+")]).set_index("sample").T
    # method confusion 2021+
    conf = pd.crosstab(R.pred_method, d.loc[R.index, "meth"], margins=True)
    pred_mix = R.predicted.str.replace(r"^.* by ", "", regex=True).value_counts().to_frame("n")
    pred_mix["hit%(win+meth+round)"] = [100 * R[R.predicted.str.endswith(" by " + k) | R.predicted.str.endswith("by " + k)].exact_hit.mean() for k in pred_mix.index]

    # ----- tiers
    t_win = tier_table(P.pick_prob.values, P.win_hit.values, P.season.values, P.be_pick.values, 2021)
    pm_ = P[P.has_mkt > 0]
    t_mkt = tier_table(pm_.mkt_pick_prob.values, pm_.mkt_hit.values, pm_.season.values, np.where(pm_.mkt_fav_A == 1, pm_.be_a, pm_.be_b), 2021, extra_seasons=False)
    wm_edges = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
    wm_lab = ["<20", "20-30", "30-40", "40-50", "50-60", "60+"]

    def prob_tiers(prob, hit, edges, labs):
        t = pd.DataFrame({"t": pd.cut(prob, edges, labels=labs, right=False), "hit": hit, "season": P.season.values, "p": prob})
        t = t[t.hit.notna()]
        g1 = t[t.season.between(2018, 2020)].groupby("t", observed=False).agg(n_2018_20=("hit", "size"), hit_2018_20=("hit", "mean"))
        g2 = t[t.season >= 2021].groupby("t", observed=False).agg(n_2021=("hit", "size"), avg_p_2021=("p", "mean"), hit_2021=("hit", "mean"))
        g = g1.join(g2)
        for c in ["hit_2018_20", "hit_2021", "avg_p_2021"]:
            g[c] *= 100
        return g.rename(columns={"hit_2018_20": "hit%_2018_20", "n_2021": "n_2021+", "avg_p_2021": "avg_p_2021+", "hit_2021": "hit%_2021+"})
    t_wm = prob_tiers(P.p_win_method.values, P.wm_hit.values, wm_edges, wm_lab)
    t_meth = prob_tiers(P.method_conf.values, P.method_mode_hit.values, [0, .4, .5, .6, .7, 1.01], ["<40", "40-50", "50-60", "60-70", "70+"])
    t_ex = prob_tiers(P.p_exact.values, P.exact_hit.values, [0, .15, .25, .35, .45, 1.01], ["<15", "15-25", "25-35", "35-45", "45+"])
    bt = best_tier(t_win)
    par = parlay_table(P.assign(hit=P.win_hit), "pick_prob", "hit", 2021)
    parm = parlay_table(P[P.has_mkt > 0].assign(hit=P.mkt_hit), "mkt_pick_prob", "hit", 2021)
    par = par.join(parm.rename(columns={"days": "events_mkt", "hit%": "mkt_fav_hit%", "model_expected%": "mkt_expected%"})).rename(columns={"days": "events"})
    cal = R[R.y.notna()].assign(b=pd.cut(R.p_A, [0, .2, .35, .45, .55, .65, .8, 1])).groupby("b", observed=True).agg(n=("y", "size"), p_model=("p_A", "mean"), A_won=("y", "mean"))

    # ----- final fit
    full = d[d.y.notna()]
    final = {"feats": feats, "win_kind": win_kind, "meth_kind": meth_kind, "round_kind": round_kind, "trained_through": str(d.date.max().date())}
    if win_kind == "stack":
        final["lr_full"] = fit_win(full, feats, "lr_full")
        final["lgbm"] = fit_win(full, feats, "lgbm")
        pj = P[P.y.notna()].join(d[["mkt_logit"]])
        final["stacker"] = LogisticRegression(max_iter=2000).fit(np.c_[logit(pj.p_lr_full), logit(pj.p_lgbm), pj.mkt_logit, pj.has_mkt], pj.y.astype(int))
    elif win_kind in ("lr_small", "lr_mid", "lr_full", "lgbm"):
        final["win_model"] = fit_win(full, feats, win_kind)
    if meth_kind.startswith("mnl"):
        final["meth_model"] = fit_meth(full, feats, meth_kind)
    final["round_model"] = fit_round(full)
    with open(OUT / "ufc_model.pkl", "wb") as f:
        pickle.dump(final, f)
    lr = final.get("lr_full") if win_kind == "stack" else (final["win_model"] if win_kind == "lr_full" else fit_win(full, feats, "lr_full"))
    mid_tab = None
    lm_ = final["win_model"] if win_kind == "lr_mid" else fit_win(full, feats, "lr_mid")
    sc_m = lm_.named_steps["standardscaler"]
    co_m = lm_.named_steps["logisticregression"].coef_[0]
    mid_tab = pd.DataFrame({"feature": MID, "coef_per_SD": co_m, "coef_raw_unit": co_m / sc_m.scale_}).set_index("feature")
    sc = lr.named_steps["standardscaler"]
    co = lr.named_steps["logisticregression"].coef_[0]
    fo = pd.DataFrame({"feature": feats, "coef_per_SD": co, "coef_raw_unit": co / sc.scale_, "SD": sc.scale_}) \
        .assign(a=lambda x: x.coef_per_SD.abs()).sort_values("a", ascending=False).drop(columns="a").head(14).set_index("feature")
    imp = None
    if win_kind in ("lgbm", "stack"):
        lg = final["lgbm"] if win_kind == "stack" else final["win_model"]
        imp = pd.Series(lg.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False).head(12)
        imp = (100 * imp / imp.sum()).rename("gain_share%").to_frame()
    stack_coef = dict(zip(["logit(lr_full)", "logit(lgbm)", "market logit", "has_market"], final["stacker"].coef_[0])) if win_kind == "stack" else None
    mm = final["meth_model"] if "meth_model" in final else fit_meth(full, feats, "mnl_small")
    mfeat = METH_SMALL if (meth_kind == "mnl_small" or not meth_kind.startswith("mnl")) else meth_feats(feats)
    mco = pd.DataFrame(mm.named_steps["logisticregression"].coef_.T, index=mfeat, columns=mm.classes_)
    mco = mco.assign(a=mco.abs().sum(1)).sort_values("a", ascending=False).drop(columns="a").head(10)

    # ----- ablation (season refits)
    print(f"[ufc] ablation ({time.time()-T0:.0f}s)", flush=True)
    wk_need = ("lr_full", "lgbm") if win_kind == "stack" else tuple(dict.fromkeys(((win_kind,) if win_kind in ("lr_small", "lr_mid", "lr_full", "lgbm") else ()) + ("lr_full",)))
    mk_need = (meth_kind,) if meth_kind.startswith("mnl") else ("mnl_full",)
    pkw = win_kind if win_kind in ("lr_small", "lr_mid", "lr_full", "lgbm", "stack") else "lr_full"
    abl = []
    for name, drop in [("all features", ())] + [(f"drop {g}", (g,)) for g in ["market"] + list(GROUPS)]:
        wa = walk(d, drop=drop, freq="S", verbose=False, win_kinds=wk_need, meth_kinds=mk_need, rounds=False)
        if "stack" not in wa and pkw == "stack":
            wa["stack"] = wa.lr_full
        ea = d.loc[wa.index]
        m = ea.season >= 2021
        wm_ = win_metrics(wa.loc[m, pkw], ea.loc[m, "y"])
        P6 = wa.loc[m, [f"{mk_need[0]}_{o}" for o in OUTS]].values
        marg = np.c_[P6[:, 0] + P6[:, 3], P6[:, 1] + P6[:, 4], P6[:, 2] + P6[:, 5]]
        ok = ea.loc[m, "meth"].notna().values
        macc = 100 * np.mean(np.array(METH)[marg.argmax(1)][ok] == ea.loc[m, "meth"].values[ok])
        wf = win_metrics(wa.loc[m, "lr_full"], ea.loc[m, "y"])
        abl.append({"variant": name, "n_features": len(feats_of(drop)), "winner%_2021+": wm_["acc"], "ll_win_2021+": wm_["ll"],
                    "lr_full_winner%": wf["acc"], "lr_full_ll": wf["ll"],
                    "method%_2021+": macc, "ll6_2021+": ll6(P6, ea.loc[m, "y6"].values)})
    abl = pd.DataFrame(abl).set_index("variant")
    abl["dLL_win(+ = helps)"] = abl["ll_win_2021+"] - abl.loc["all features", "ll_win_2021+"]
    abl["dLL6(+ = helps)"] = abl["ll6_2021+"] - abl.loc["all features", "ll6_2021+"]
    abl["lr_full_dLL(+ = helps)"] = abl["lr_full_ll"] - abl.loc["all features", "lr_full_ll"]

    # ----- history study
    print(f"[ufc] history ({time.time()-T0:.0f}s)", flush=True)
    hist = []
    for name, kw in [(f"last {k} season(s)", dict(window=k)) for k in (1, 2, 4, 6)] + [("all prior seasons (from 2016)", {})] + \
                    [(f"all, recency half-life {h} season(s)", dict(half_life=h)) for h in (1, 2, 4)]:
        wa = walk(d, freq="S", verbose=False, win_kinds=wk_need, meth_kinds=tuple(dict.fromkeys(mk_need + ("mnl_full",))), rounds=(round_kind == "lr"), first_season=2020, **kw)
        if "stack" not in wa and pkw == "stack":
            wa["stack"] = wa.lr_full
        mk = mk_need[0] if meth_kind.startswith("mnl") else meth_kind
        pkh = picks(d, wa, pkw, mk, round_kind)
        ea = d.loc[wa.index]
        m = (ea.season >= 2023).values
        g = pkh[m]
        hi = g.pick_prob >= 0.75
        hist.append({"training data": name, "n_eval": int(g.win_hit.notna().sum()), "winner%": 100 * g.win_hit.mean(),
                     "ll_win": logloss(g.p_A[g.y.notna()], g.y.dropna()), "method%": 100 * g.meth_hit.mean(),
                     "method%(mode)": 100 * g.method_mode_hit.mean(), "winner+method%": 100 * g.wm_hit.mean(),
                     "n_pick≥75%": int(hi.sum()), "win%_pick≥75%": 100 * g[hi].win_hit.mean(),
                     "lr_full_winner%": 100 * np.mean((wa.lr_full.values[m] >= 0.5)[g.y.notna().values] == (g.y.dropna().values == 1)),
                     "lr_full_ll": logloss(wa.lr_full.values[m][g.y.notna().values], g.y.dropna()),
                     "mnl_full_method%": method_acc(wa, "mnl_full", ea, m)})
    hist = pd.DataFrame(hist).set_index("training data")

    P.to_parquet(OUT / "ufc_preds.parquet", index=False)
    rep_md = render(d, P, sel_tab, win_kind, meth_tab, meth_kind, round_tab, round_kind, season_tab, sum_tab, conf, pred_mix,
                    t_win, t_mkt, t_wm, t_meth, t_ex, bt, par, cal, fo, imp, stack_coef, mco, abl, hist, T0, mid_tab)
    (OUT / "ufc_predict.md").write_text(rep_md)
    s = sum_tab["2021+"]
    summary = {"winner%": s["winner%"], "winner%_mkt_fights": s["winner%_mkt_fights"], "mkt_fav%": s["mkt_fav%"],
               "method%": s["method% (method of predicted result)"], "winner+method%": s["winner+method%"],
               "round%": s["round% (round of predicted result)"], "exact%": s["winner+method+round%"], "best_tier": bt,
               "parlay3": par.loc[3, "hit%"], "kinds": (win_kind, meth_kind, round_kind)}
    print(summary, f"{time.time()-T0:.0f}s", flush=True)
    return summary


def render(d, P, sel_tab, win_kind, meth_tab, meth_kind, round_tab, round_kind, season_tab, sum_tab, conf, pred_mix, t_win, t_mkt,
           t_wm, t_meth, t_ex, bt, par, cal, fo, imp, stack_coef, mco, abl, hist, T0, mid_tab=None):
    L = []
    R = P[P.season >= 2021]
    s = sum_tab["2021+"]
    L.append("# UFC fight predictor: winner + method + round\n")
    L.append(f"Generated by `python -m engine.predict.ufc` ({(time.time()-T0)/60:.1f} min). Strict walk-forward: every month from 2018 "
             "the models are refit on all fights dated before that month (2016 onward) and predict that month's fights. Models "
             f"picked on 2018-2020; reported 2021+ ({int(R.win_hit.notna().sum()):,} fights with a winner) was never used to pick anything.\n")
    L.append("* \"Final score\" of a fight = winner + method (KO/TKO, SUB, DEC) + ending round (a decision ends in the last scheduled round).")
    L.append("* Market: closing moneyline (a_ml/b_ml) de-vigged with `engine.devig.devig(a, b, 'odds_ratio')`; dropped when missing, "
             f"|odds|<100, overround outside 0.995-1.15 or >45 pts from the script ({int(P.mkt_broken.sum())} dropped). Moneylines are "
             "missing for 15-26% of 2023-24 fights and 74% of 2026 fights; those use the script (has_mkt=0). Method props: the "
             "script's power6 de-vig with its quality gate (2025-26 props are mostly broken and are gated out).")
    L.append("* Fighter stats (age, reach, record, streaks, strike/grapple rates) come from data/processed/ufc.csv and are pre-fight "
             "(checked: career win counts increment only after the fight). Fighter A is the red corner except in 2026, where most fights are listed alphabetically (corner unknown); every model is trained on mirrored copies of each fight (A<->B) so the only corner effect is the explicit `corner_sign` feature.\n")
    L.append("## Headline (2021+)\n")
    L.append("| metric | value |\n|---|---|\n"
             f"| Winner, model (all fights) | {s['winner%']:.2f}% of {int(s['fights']):,} |\n"
             f"| Winner, model vs market favourite (fights with a usable price) | {s['winner%_mkt_fights']:.2f}% vs {s['mkt_fav%']:.2f}% (n={int(s['fights_w_mkt']):,}) |\n"
             f"| Method accuracy (method of the predicted result / most likely method) | {s['method% (method of predicted result)']:.2f}% / {s['method% (most likely method overall)']:.2f}% (always-DEC {s['method% always-DEC baseline']:.2f}%) |\n"
             f"| Winner + method | {s['winner+method%']:.2f}% |\n"
             f"| Round accuracy (round of predicted result / most likely ending round) | {s['round% (round of predicted result)']:.2f}% / {s['round% (most likely ending round)']:.2f}% (always-distance {s['round% always-distance baseline']:.2f}%) |\n"
             f"| Winner + method + round (exact result) | {s['winner+method+round%']:.2f}% |\n"
             + (f"| Best winner tier (≥100 fights) | {bt[0]}: {bt[2]:.1f}% (n={bt[1]}) |\n" if bt else "")
             + f"| Per-event top-1 / top-2 / top-3 parlay | {par.loc[1,'hit%']:.1f}% / {par.loc[2,'hit%']:.1f}% / {par.loc[3,'hit%']:.1f}% |\n")
    L.append("## Model selection\n")
    L.append(f"Winner: **{win_kind}** (lowest 2018-2020 log loss; `blend` = the script's own market blend, shown for reference only).\n")
    L.append(md(sel_tab, ".4f"))
    L.append(f"\nWinner × method (6-way): **{meth_kind}**. script6 = script method probabilities; blend6 = script/market-prop log-linear pool; "
             "mnl_small = multinomial logistic on script + market-prop log-probabilities + moneyline; mnl_full = + all features.\n")
    L.append(md(meth_tab, ".4f"))
    L.append(f"\nFinish round given a finish (KO/SUB): **{round_kind}** (script = script's p_end_r normalised; lr = multinomial logistic on "
             "the script round probabilities, method, 5-rounder, weight, gender).\n")
    L.append(md(round_tab, ".4f"))
    L.append("\n## 1. Accuracy per season\n")
    L.append(md(season_tab, ".2f"))
    L.append("\n## 2. Score (result) accuracy\n")
    L.append(md(sum_tab, ".3f"))
    L.append("\nPredicted method vs actual (2021+):\n\n" + md(conf, ".0f"))
    L.append("\nPredicted result types (2021+) and exact-result hit rate:\n\n" + md(pred_mix, ".1f"))
    L.append("\nCalibration of P(A wins), 2021+:\n\n" + md(cal, ".3f"))
    L.append("\n## 3. Win rate by confidence tier\n")
    L.append("### (a) Winner (moneyline); `breakeven%` = vigged implied probability of the picked fighter at the closing price\n")
    L.append(md(t_win, ".1f"))
    L.append("\nMarket favourite by its de-vigged probability (comparison):\n\n" + md(t_mkt, ".1f"))
    L.append("\n### (b) Winner + method, by model P(winner and method)\n")
    L.append(md(t_wm, ".1f"))
    L.append("\n### (c) Method only (most likely method), by its probability\n")
    L.append(md(t_meth, ".1f"))
    L.append("\n### (d) Exact result (winner + method + round), by its probability\n")
    L.append(md(t_ex, ".1f"))
    if bt:
        L.append(f"\n**Highest-win% winner tier with ≥100 fights 2021+: {bt[0]} → {bt[2]:.1f}% (n={bt[1]}).**\n")
    L.append("\n## 4. Per-event top-N confidence parlays (2021+)\n")
    L.append("Each event (card date), the N most confident winner picks parlayed (events with ≥N fights); compared with the N biggest market favourites.\n")
    L.append(md(par, ".1f"))
    L.append("\n## 5. The formula (plain English)\n")
    if mid_tab is not None:
        L.append("Compact winner logistic `lr_mid` (market + script + age/youth-in-light-classes + reach + layoff + skill), full-data fit:\n")
        L.append(md(mid_tab, ".4f") + "\n")
    L.append("Full winner logistic `lr_full` (full-data fit), largest terms per one SD and per raw unit:\n")
    L.append(md(fo, ".4f"))
    L.append("\nIn words: logit P(A wins) ≈ " + " + ".join(f"{v:+.3f}×{k}" for k, v in fo.head(5).coef_raw_unit.items()) + " + (smaller terms). "
             "`mkt_logit` = log-odds of the de-vigged moneyline; `s_logit` = script log-odds; diffs are A minus B.\n")
    if stack_coef:
        L.append("Stack weights: " + ", ".join(f"{k} {v:+.3f}" for k, v in stack_coef.items()) + "\n")
    if imp is not None:
        L.append("LightGBM gain share:\n\n" + md(imp, ".1f") + "\n")
    L.append("Method model (multinomial logistic, per-SD coefficients for each outcome, largest features):\n")
    L.append(md(mco, ".3f"))
    L.append("\nPrediction rule: pick the winner from P(A wins); pick that fighter's most likely method from the 6-way model "
             "(renormalised within the winner); if KO/SUB, pick the most likely finishing round, else the last round.\n")
    L.append("\n## 6. Ablation (2021+, season refits, drop one feature group)\n")
    L.append(md(abl, ".4f"))
    L.append("\nGroups: " + "; ".join(f"**{g}**: {', '.join(c)}" for g, c in GROUPS.items()) + "; **market**: mkt_logit, mkt_gap (market minus script log-odds, 0 when no price).\n")
    L.append("\n## Does more history help?\n")
    L.append("Evaluation fixed at 2023+ (season refits; for season S the model sees only fights before S, restricted to the last k "
             "seasons, or all with exponential recency weights 0.5^(age/half-life)). Script features exist from 2016, so 'last 6' and "
             "'all' differ only by 2016 for the 2023 season. High-confidence tier = pick probability ≥75%.\n")
    L.append(f"Unprefixed columns = chosen models (winner **{win_kind}**, method **{meth_kind}**; a fixed blend like blend6 does not train, so its "
             "method% cannot move); `lr_full_*` / `mnl_full_*` = the full learned winner / method models.\n")
    L.append(md(hist, ".4f"))
    w_ = hist.iloc[:5]
    b = w_.ll_win.idxmin()
    rec = hist.iloc[5:]
    rb = rec.ll_win.idxmin()
    L.append(f"\nBest plain window by winner log loss: **{b}** (winner {w_.loc[b,'winner%']:.1f}%). Last-1 vs all: winner "
             f"{w_.iloc[0]['winner%']:.1f}% → {w_.iloc[4]['winner%']:.1f}%, log loss {w_.iloc[0]['ll_win']:.4f} → {w_.iloc[4]['ll_win']:.4f}, "
             f"method {w_.iloc[0]['method%']:.1f}% → {w_.iloc[4]['method%']:.1f}%. "
             + ("More history keeps helping. " if w_.iloc[2].ll_win - w_.iloc[4].ll_win > 0.001 else "Gains plateau after ~2-4 seasons. ")
             + f"Recency weighting best: {rb} (log loss {rec.loc[rb,'ll_win']:.4f} vs {w_.iloc[4]['ll_win']:.4f} unweighted). "
             f"With {int(w_.n_eval.iloc[0]):,} fights in the 2023+ window, SU differences under ~2 pts are noise (SE ≈ {100*np.sqrt(0.22/w_.n_eval.iloc[0]):.1f} pts). "
             f"Full logistic (many more parameters): last 1 → all winner {w_.iloc[0]['lr_full_winner%']:.1f}% → {w_.iloc[4]['lr_full_winner%']:.1f}%, "
             f"log loss {w_.iloc[0]['lr_full_ll']:.4f} → {w_.iloc[4]['lr_full_ll']:.4f}; learned method model (mnl_full) method% "
             f"{w_.iloc[0]['mnl_full_method%']:.1f}% → {w_.iloc[4]['mnl_full_method%']:.1f}%.\n")
    L.append("\n## Live use\n")
    L.append(f"`engine.predict.ufc.predict(slate_df)` loads `results/predict/ufc_model.pkl` (fit on all fights through {d.date.max().date()}). "
             "Returns fighter_a, fighter_b, p_A, pick, pick_prob, pred_method, pred_round, predicted (e.g. 'X by KO R1'), p_win_method, "
             "p_exact, p_KO/p_SUB/p_DEC, tier. Files: `results/predict/ufc_preds.parquet`, `ufc_model.pkl`.\n")
    return "\n".join(L)


def predict(slate_df: pd.DataFrame, csv: pd.DataFrame | None = None) -> pd.DataFrame:
    with open(OUT / "ufc_model.pkl", "rb") as f:
        model = pickle.load(f)
    d = build(slate_df, csv)
    feats = model["feats"]
    wk = model["win_kind"]
    w = pd.DataFrame(index=d.index)
    if wk == "stack":
        a = model["lr_full"].predict_proba(d[feats].values)[:, 1]
        b = model["lgbm"].predict_proba(d[feats].values)[:, 1]
        w["stack"] = model["stacker"].predict_proba(np.c_[logit(a), logit(b), d.mkt_logit, d.has_mkt])[:, 1]
    elif wk == "market":
        w[wk] = np.where(d.has_mkt > 0, d.p_mkt, d.p_A_win)
    elif wk == "script":
        w[wk] = d.p_A_win.values
    else:
        w[wk] = pred_win(model["win_model"], d, feats, wk)
    mk = model["meth_kind"]
    if mk.startswith("mnl"):
        P6 = pred_meth(model["meth_model"], d, feats, mk)
    else:
        src = "p_" if mk == "script6" else "blend_"
        P6 = d[[f"{src}{o}" for o in OUTS]].values
    for j, o in enumerate(OUTS):
        w[f"{mk}_{o}"] = P6[:, j]
    for mth, flag in [("KO", 0), ("SUB", 1)]:
        Rr = pred_round(model["round_model"], d, flag)
        for r in range(5):
            w[f"rd_{mth}_{r+1}"] = Rr[:, r]
    sr = script_round(d)
    for r in range(5):
        w[f"srd_{r+1}"] = sr[:, r]
    o = picks(d, w, wk, mk, model["round_kind"])
    o.insert(0, "fighter_b", d.fighter_b.values)
    o.insert(0, "fighter_a", d.fighter_a.values)
    o["tier"] = tier_of(o.pick_prob).astype(str)
    return o.drop(columns=[c for c in o.columns if c.endswith("_hit") or c == "y"])


if __name__ == "__main__":
    run()
