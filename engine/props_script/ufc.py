"""
UFC script-driven fight props (goes distance, winner by KO/SUB/DEC, under/over 1.5 & 2.5 rounds) and 3/5/7-leg
parlays, from the fight script (results/scripts/ufc_fights.parquet) plus fighter tendencies.

  python -m engine.props_script.ufc

Model (strictly walk-forward, refit every season on prior seasons only):
  * outcome model   6-way multinomial logistic (A/B x KO/SUB/DEC) on the script's method probabilities, the
                    market method props where they exist (de-vigged; indicator when missing) and fighter tendencies
                    from every prior UFC fight since 1994 (ufcstats via the greco dump): finish rates (KO / SUB
                    wins per fight), durability (KO / SUB losses per fight), decision rate, mean fight length,
                    knockdowns / sub attempts / takedowns per 15 min, significant strikes landed & absorbed per
                    minute, experience; weight-class finish rate (prior fights), scheduled rounds, title fight,
                    women's division. Trained on both A/B orientations; predictions averaged over orientations.
  * timing model    3-way multinomial for a finish's clock time: before 2:30 of R2 (under 1.5), between (1.5-2.5),
                    after 2:30 of R3, given the fight ends inside the distance (script timing + tendencies).
  * joint           14 cells = {A,B} x {KO,SUB} x 3 time buckets + A_DEC + B_DEC; every leg is a set of cells, so
                    same-fight parlays have an exact joint; different fights are independent.
  * calibrator      online per leg type (distance yes/no, fav/dog x method, under/over lines) + confidence bins.
Fit / choose on seasons <= 2020; report 2021+.
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from engine.props_script import common_count as C

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RS, RP = ROOT / "results" / "scripts", ROOT / "results" / "predict"
OUT = ROOT / "results" / "props_script"
FIRST_TEST, REPORT_FROM = 2018, 2021
OUTC = ["A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC"]
SWAP6 = [3, 4, 5, 0, 1, 2]
# cells: 0-5 A_KO_t1..t3, A_SUB_t1..t3 ; 6-11 B ; 12 A_DEC ; 13 B_DEC
NCELL = 14


def nrm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def cell(side, meth, t):
    return (0 if side == "A" else 6) + (0 if meth == "KO" else 3) + t


LEGS = {}
for s in "AB":
    for m in ("KO", "SUB"):
        LEGS[f"{s}_{m}"] = [cell(s, m, t) for t in range(3)]
    LEGS[f"{s}_DEC"] = [12 if s == "A" else 13]
    LEGS[f"{s}_WIN"] = [cell(s, m, t) for m in ("KO", "SUB") for t in range(3)] + [12 if s == "A" else 13]
LEGS["DIST_YES"] = [12, 13]
LEGS["DIST_NO"] = list(range(12))
LEGS["U1.5"] = [cell(s, m, 0) for s in "AB" for m in ("KO", "SUB")]
LEGS["O1.5"] = [c for c in range(NCELL) if c not in LEGS["U1.5"]]
LEGS["U2.5"] = [cell(s, m, t) for s in "AB" for m in ("KO", "SUB") for t in (0, 1)]
LEGS["O2.5"] = [c for c in range(NCELL) if c not in LEGS["U2.5"]]
PROP_LEGS = [k for k in LEGS if not k.endswith("_WIN")]
LEG_MASK = {k: np.isin(np.arange(NCELL), v) for k, v in LEGS.items()}


# ------------------------------------------------------------------ fighter histories (greco, all UFC fights)
def histories(log=print):
    from engine.scripts.ufc import load_greco
    r, st = load_greco()
    r["fid"] = r.URL.str.rsplit("/", n=1).str[-1]
    stt = st.set_index(["EVENT", "BOUT", "FIGHTER"])
    H = {}
    wc_fin = {}
    pre = {}                         # (fid, fighter name) -> feature dict
    pre_wc = {}
    prior = dict(n=0, kow=0, subw=0, decw=0, kol=0, subl=0, decl=0, secs=0.0, sl=0.0, sa=0.0, kd=0.0, suba=0.0,
                 tdl=0.0, r1f=0)
    for row in r.itertuples(index=False):
        sched = row.sched if row.sched else 3
        wc = row.wc
        w = wc_fin.setdefault(wc, [0, 0])
        pre_wc[row.fid] = (w[0] + 0.45 * 50) / (w[1] + 50)
        for f in (row.f1, row.f2):
            h = H.get(f, dict(prior))
            n = h["n"]; mins = max(h["secs"] / 60, 1e-6)
            pre[(row.fid, f)] = dict(
                n=n,
                ko_rate=(h["kow"] + 0.25 * 3) / (n + 3), sub_rate=(h["subw"] + 0.12 * 3) / (n + 3),
                dec_rate=(h["decw"] + h["decl"] + 0.45 * 3) / (n + 3),
                ko_loss=(h["kol"] + 0.15 * 3) / (n + 3), sub_loss=(h["subl"] + 0.08 * 3) / (n + 3),
                r1_fin=(h["r1f"] + 0.2 * 3) / (n + 3),
                mins_pf=(h["secs"] / 60 + 10 * 3) / (n + 3),
                sl_pm=(h["sl"] + 3.5 * 15) / (mins + 15), sa_pm=(h["sa"] + 3.5 * 15) / (mins + 15),
                kd15=(h["kd"] + 0.2 * 1) / (mins + 15) * 15, suba15=(h["suba"] + 0.5 * 1) / (mins + 15) * 15,
                td15=(h["tdl"] + 1.2 * 1) / (mins + 15) * 15)
        # update after the fight
        if row.mcat in ("KO", "SUB", "DEC") and row.win1 in (0.0, 1.0):
            win, los = (row.f1, row.f2) if row.win1 == 1 else (row.f2, row.f1)
            for f, is_w in ((win, True), (los, False)):
                h = H.setdefault(f, dict(prior))
                h["n"] += 1
                if row.mcat == "DEC":
                    h["decw" if is_w else "decl"] += 1
                else:
                    h[("kow" if row.mcat == "KO" else "subw") if is_w else ("kol" if row.mcat == "KO" else "subl")] += 1
                    if row.round == 1:
                        h["r1f"] += 1
                h["secs"] += row.secs if row.secs == row.secs else 0
                try:
                    s = stt.loc[(row.EVENT, row.BOUT, f)]
                    for c in ("sl", "sa", "kd", "suba", "tdl"):
                        v = s[c]
                        h[c] += float(v) if v == v else 0.0
                except KeyError:
                    pass
            w[1] += 1
            w[0] += int(row.mcat != "DEC")
    return r, pre, pre_wc


# ------------------------------------------------------------------ data
def load(log=print):
    t0 = time.time()
    u = pd.read_parquet(RS / "ufc_fights.parquet")
    r, pre, pre_wc = histories(log)
    g = r.set_index("fid")
    u = u[u.fight_id.isin(g.index)].copy()
    gg = g.loc[u.fight_id]
    u["f1"], u["f2"] = gg.f1.values, gg.f2.values
    u["tsec"], u["g_round"] = gg.tsec.values, gg["round"].values
    a_is_1 = (u.fighter_a.map(nrm) == u.f1.map(nrm)).values | (
        (u.fighter_a.map(nrm) != u.f2.map(nrm)).values & ((u.winner == "A").values == (gg.win1.values == 1)))
    u["ga"] = np.where(a_is_1, u.f1, u.f2)
    u["gb"] = np.where(a_is_1, u.f2, u.f1)
    keys = ["n", "ko_rate", "sub_rate", "dec_rate", "ko_loss", "sub_loss", "r1_fin", "mins_pf", "sl_pm", "sa_pm",
            "kd15", "suba15", "td15"]
    for side, col in (("a", "ga"), ("b", "gb")):
        vals = [pre.get((f, n), None) for f, n in zip(u.fight_id, u[col])]
        for k in keys:
            u[f"{side}_{k}"] = [v[k] if v else np.nan for v in vals]
    u["wc_fin"] = u.fight_id.map(pre_wc)
    u["five"] = (u.scheduled_rounds == 5).astype(int)
    u["female"] = (u.gender == "F").astype(int)
    # outcomes
    u = u[u.outcome.isin(OUTC)].copy()
    u["y6"] = u.outcome.map({c: i for i, c in enumerate(OUTC)})
    sec = (u.g_round - 1) * 300 + u.tsec
    u["tb"] = np.select([sec < 450, sec < 750], [0, 1], 2)           # 450s = 2:30 of R2, 750s = 2:30 of R3
    ycell = np.where(u.y6 == 2, 12, np.where(u.y6 == 5, 13, 0))
    fin = ~u.y6.isin([2, 5])
    side = np.where(u.y6 < 3, "A", "B")
    meth = u.outcome.str.split("_").str[1]
    u["ycell"] = np.where(fin, [cell(s, m, int(t)) if f else 0 for s, m, t, f in zip(side, meth, u.tb, fin)], ycell)
    u = u.sort_values(["date", "fight_id"]).reset_index(drop=True)
    log(f"  fights {len(u)} ({time.time()-t0:.0f}s)")
    return u


GROUPS = {
    "market": ["m_A_KO", "m_A_SUB", "m_A_DEC", "m_B_KO", "m_B_SUB", "m_B_DEC", "has_mkt"],
    "finish_tendency": ["a_ko_rate", "a_sub_rate", "b_ko_rate", "b_sub_rate", "a_r1_fin", "b_r1_fin"],
    "durability": ["a_ko_loss", "a_sub_loss", "b_ko_loss", "b_sub_loss", "a_dec_rate", "b_dec_rate", "a_mins_pf", "b_mins_pf"],
    "style": ["a_kd15", "b_kd15", "a_suba15", "b_suba15", "a_td15", "b_td15", "a_sl_pm", "b_sl_pm", "a_sa_pm", "b_sa_pm",
              "a_n", "b_n"],
    "weight_class": ["wc_fin", "female"],
    "rounds": ["five", "title_bout"],
}
SCRIPT = ["s_A_KO", "s_A_SUB", "s_A_DEC", "s_B_KO", "s_B_SUB", "s_B_DEC", "s_u15", "s_u25"]


def design(u, swap=False):
    """Feature frame in A/B orientation (swap=True: B becomes A)."""
    X = pd.DataFrame(index=u.index)
    o = OUTC if not swap else [OUTC[i] for i in SWAP6]
    for new, old in zip(OUTC, o):
        X[f"s_{new}"] = C.logit(u[f"p_{old}"].clip(0.005, 0.99))
        mk = u[f"mkt_{old}"]
        X[f"m_{new}"] = C.logit(mk.fillna(u[f"p_{old}"]).clip(0.005, 0.99))
        X[f"b_{new}"] = C.logit(u[f"blend_{old}"].clip(0.005, 0.99))
    X["has_mkt"] = u.mkt_A_KO.notna().astype(int)
    fin = (1 - u.p_distance).clip(0.02, 1)
    X["s_u15"] = C.logit((u.p_under_1_5 / fin).clip(0.01, 0.99))
    X["s_u25"] = C.logit((u.p_under_2_5 / fin).clip(0.01, 0.99))
    for k in ("ko_rate", "sub_rate", "dec_rate", "ko_loss", "sub_loss", "r1_fin", "mins_pf", "sl_pm", "sa_pm", "kd15",
              "suba15", "td15", "n"):
        a, b = (u[f"a_{k}"], u[f"b_{k}"]) if not swap else (u[f"b_{k}"], u[f"a_{k}"])
        X[f"a_{k}"], X[f"b_{k}"] = a, b
    X["a_n"], X["b_n"] = np.log1p(X.a_n), np.log1p(X.b_n)
    for c in ("wc_fin", "female", "five", "title_bout"):
        X[c] = u[c]
    return X.fillna(X.median(numeric_only=True)).fillna(0)


BLEND = ["b_A_KO", "b_A_SUB", "b_A_DEC", "b_B_KO", "b_B_SUB", "b_B_DEC"]
TIMING = ["s_u15", "s_u25"]
CONFIGS = {
    "blend + timing": BLEND + TIMING,
    "blend + timing + finish/durability/weight-class/rounds": BLEND + TIMING + GROUPS["finish_tendency"]
    + GROUPS["durability"] + GROUPS["weight_class"] + GROUPS["rounds"],
    "blend + script + all tendencies": BLEND + SCRIPT + sum(GROUPS.values(), []),
}
CHOICE = {"cols": CONFIGS["blend + timing"], "C": 0.03, "name": "blend + timing"}


def select_config(u, log=print):
    """Choose features / regularisation on seasons <= 2020 only: train <= 2018, validate 2019-2020."""
    tr, va = (u.season <= 2018).values, ((u.season >= 2019) & (u.season <= 2020)).values
    rows = []
    for name, cols in CONFIGS.items():
        for Cc in (0.03, 0.3):
            P = Model(cols, Cc).fit(u[tr]).cells(u[va])
            Lv = legs_long(u[va].reset_index(drop=True), P)
            rows.append(dict(config=name, C=Cc, val_ll_legs=C.ll(Lv.p_model, Lv.hit)))
    t = pd.DataFrame(rows)
    b = t.sort_values("val_ll_legs").iloc[0]
    CHOICE.update(cols=CONFIGS[b.config], C=float(b.C), name=b.config)
    log(f"  chosen on 2019-20 validation: {b.config} C={b.C}")
    return t


def cols_for(groups):
    c = list(SCRIPT)
    for v in groups.values():
        c += v
    return c


class Model:
    def __init__(self, cols, C_=0.3):
        self.cols, self.C_ = cols, C_

    def fit(self, u):
        X0, X1 = design(u)[self.cols], design(u, swap=True)[self.cols]
        y6 = u.y6.values
        X = pd.concat([X0, X1]); y = np.r_[y6, np.array(SWAP6)[y6]]
        self.sc = StandardScaler().fit(X)
        self.m6 = LogisticRegression(C=self.C_, max_iter=3000).fit(self.sc.transform(X), y)
        fin = ~np.isin(y6, [2, 5])
        Xt = pd.concat([X0[fin], X1[fin]]); yt = np.r_[u.tb.values[fin], u.tb.values[fin]]
        self.mt = LogisticRegression(C=self.C_, max_iter=3000).fit(self.sc.transform(Xt), yt)
        return self

    def cells(self, u):
        X0, X1 = design(u)[self.cols], design(u, swap=True)[self.cols]
        p0 = self.m6.predict_proba(self.sc.transform(X0))
        p1 = self.m6.predict_proba(self.sc.transform(X1))[:, SWAP6]
        p6 = (p0 + p1) / 2
        t = (self.mt.predict_proba(self.sc.transform(X0)) + self.mt.predict_proba(self.sc.transform(X1))) / 2
        return cells_from(p6, t)


def cells_from(p6, t):
    P = np.zeros((len(p6), NCELL))
    for j, (s, m) in enumerate([("A", "KO"), ("A", "SUB"), (None, None), ("B", "KO"), ("B", "SUB")]):
        if s is None:
            continue
        for k in range(3):
            P[:, cell(s, m, k)] = p6[:, j] * t[:, k]
    P[:, 12], P[:, 13] = p6[:, 2], p6[:, 5]
    return P


def script_cells(u):
    p6 = u[[f"p_{c}" for c in OUTC]].values
    p6 = p6 / p6.sum(1, keepdims=True)
    fin = (1 - u.p_distance).clip(0.02, 1).values
    t1 = (u.p_under_1_5.values / fin).clip(0.01, 0.98)
    t2 = (u.p_under_2_5.values / fin).clip(t1 + 0.005, 0.99) - t1
    t = np.c_[t1, t2, 1 - t1 - t2]
    return cells_from(p6, t)


def blend_cells(u):
    p6 = u[[f"blend_{c}" for c in OUTC]].values
    p6 = p6 / p6.sum(1, keepdims=True)
    fin = (1 - u.p_distance).clip(0.02, 1).values
    t1 = (u.p_under_1_5.values / fin).clip(0.01, 0.98)
    t2 = (u.p_under_2_5.values / fin).clip(t1 + 0.005, 0.99) - t1
    return cells_from(p6, np.c_[t1, t2, 1 - t1 - t2])


def walk_cells(u, log=print):
    P = script_cells(u)
    for s in sorted(u.season.unique()):
        if s < FIRST_TEST:
            continue
        tr, te = (u.season < s).values, (u.season == s).values
        m = Model(CHOICE["cols"], CHOICE["C"]).fit(u[tr])
        P[te] = m.cells(u[te])
    return P


def legs_long(u, P):
    rows = []
    pa = P[:, LEGS["A_WIN"]].sum(1)
    fav_a = pa >= 0.5
    for k in PROP_LEGS:
        p = P[:, LEG_MASK[k]].sum(1)
        hit = LEG_MASK[k][u.ycell.values].astype(float)
        side = k[0] if k[:2] in ("A_", "B_") else None
        if side:
            fav = fav_a if side == "A" else ~fav_a
            typ = np.where(fav, "fav_" + k[2:], "dog_" + k[2:])
            name = np.where(side == "A", u.fighter_a, u.fighter_b)
            label = [f"{n} by {k[2:]}" for n in name]
        else:
            typ = np.full(len(u), k)
            label = [f"{a} v {b}: {k}" for a, b in zip(u.fighter_a, u.fighter_b)]
        rows.append(pd.DataFrame(dict(row=np.arange(len(u)), leg=k, typ=typ, p_model=p, hit=hit, label=label)))
    L = pd.concat(rows, ignore_index=True)
    L["date"] = u.date.values[L.row]
    L["season"] = u.season.values[L.row]
    return L


# ------------------------------------------------------------------ parlays (exact same-fight joint)
def ticket_joint(legs, Pc, cal_ratio):
    """legs: list of (row, legname, p_cal). Exact joint within a fight from cells, scaled by the calibration ratio of
    the block's most restrictive leg; fights independent."""
    out = 1.0
    by = {}
    for L in legs:
        by.setdefault(L["row"], []).append(L)
    for r, ls in by.items():
        mask = np.ones(NCELL, bool)
        for L in ls:
            mask &= LEG_MASK[L["leg"]]
        raw = Pc[r][mask].sum()
        tight = min(ls, key=lambda L: L["p_raw"])
        out *= float(np.clip(raw * tight["p"] / max(tight["p_raw"], 1e-9), 0, 1))
    return out


def ufc_tickets(day, Pc, mlday, sizes=(3, 5, 7)):
    legs = day.to_dict("records")
    res = []
    for n in sizes:
        cands = []
        # top-N: max one leg per fight
        t, seen = [], set()
        for L in sorted(legs, key=lambda d: -d["p"]):
            if L["row"] not in seen:
                t.append(L); seen.add(L["row"])
            if len(t) == n:
                break
        if len(t) == n:
            cands.append(("top", t))
        # same-fight stacks: greedy within a fight on exact joint, non-redundant legs; continue into next fight
        best, bv = None, -1
        for r0 in {L["row"] for L in legs}:
            cur = []
            pool = [L for L in legs if L["row"] == r0]
            while len(cur) < n:
                bc, bj = None, -1
                for c in pool + ([L for L in legs if L["row"] != r0] if len(cur) and not any(
                        _nonredundant(cur, L, Pc) and ticket_joint(cur + [L], Pc, None)
                        >= ticket_joint(cur, Pc, None) * L["p_raw"] for L in pool if L not in cur) else []):
                    if c in cur or not _nonredundant(cur, c, Pc):
                        continue
                    j = ticket_joint(cur + [c], Pc, None)
                    if c["row"] == r0 and cur and j < ticket_joint(cur, Pc, None) * c["p_raw"]:
                        continue                      # same-fight legs must move together (joint > independent)
                    if j > bj:
                        bc, bj = c, j
                if bc is None:
                    break
                cur.append(bc)
            if len(cur) == n and bj > bv:
                best, bv = cur, bj
        if best:
            cands.append(("sgp", best))
        for thr in (0.85, 0.70):
            if mlday is not None and len(mlday) and mlday.p.max() >= thr:
                m = mlday.sort_values("p", ascending=False).iloc[0]
                rest, seen = [], {m["row"]}
                for L in sorted(legs, key=lambda d: -d["p"]):
                    if L["row"] not in seen:
                        rest.append(L); seen.add(L["row"])
                    if len(rest) == n - 1:
                        break
                if len(rest) == n - 1:
                    cands.append((f"mixed_ml{int(thr*100)}", [dict(m)] + rest))
        for c, t in cands:
            props = [L for L in t if L["leg"] != "ML"]
            pj = ticket_joint(props, Pc, None) * np.prod([L["p"] for L in t if L["leg"] == "ML"])
            hit = float(all(L["hit"] == 1 for L in t))
            res.append(dict(construction=c, n_legs=n, p_joint=pj, p_indep=float(np.prod([L["p"] for L in t])), hit=hit,
                            legs=" | ".join(f"{L['label']} ({L['p']:.2f})" for L in t)))
        cn = [r for r in res if r["n_legs"] == n]
        if cn:
            b = max(cn, key=lambda r: r["p_joint"])
            res.append({**b, "construction": "best_of_day"})
    return res


def _nonredundant(cur, c, Pc):
    """c adds information (shrinks the joint) and does not make an existing leg redundant."""
    same = [L for L in cur if L["row"] == c["row"]]
    if not same:
        return True
    mask = np.ones(NCELL, bool)
    for L in same:
        mask &= LEG_MASK[L["leg"]]
    new = mask & LEG_MASK[c["leg"]]
    p = Pc[c["row"]]
    if p[new].sum() >= p[mask].sum() * 0.999 or p[new].sum() <= 1e-6:
        return False
    for i, L in enumerate(same):
        m2 = LEG_MASK[c["leg"]].copy()
        for j, L2 in enumerate(same):
            if j != i:
                m2 &= LEG_MASK[L2["leg"]]
        if p[new].sum() >= p[m2].sum() * 0.999:
            return False
    return True


# ------------------------------------------------------------------ real lines: market method props
def real_lines(u, L):
    x = L[L.leg.isin(["A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC"])].copy()
    x["p_mkt"] = [u[f"mkt_{lg}"].values[r] for lg, r in zip(x.leg, x.row)]
    odds_col = {"A_KO": "a_ko_odds", "A_SUB": "a_sub_odds", "A_DEC": "a_dec_odds", "B_KO": "b_ko_odds",
                "B_SUB": "b_sub_odds", "B_DEC": "b_dec_odds"}
    x["odds"] = [u[odds_col[lg]].values[r] for lg, r in zip(x.leg, x.row)]
    x["mq"] = u.mkt_quality_ok.values[x.row]
    x = x[x.p_mkt.notna() & x.odds.notna() & x.mq]
    return x


def real_line_table(x):
    rows = []
    for w, s in (("<=2020 (fit)", x[x.season <= 2020]), ("2021+ (report)", x[x.season >= REPORT_FROM])):
        for lo, hi in ((0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)):
            b = s[(s.p_cal >= lo) & (s.p_cal < hi)]
            if len(b):
                dec = np.where(b.odds > 0, 1 + b.odds / 100, 1 + 100 / b.odds.abs())
                rows.append(dict(window=w, model_tier=f"{int(lo*100)}-{int(min(hi,1)*100)}%", n=len(b), pred=b.p_cal.mean(),
                                 hit=b.hit.mean(), mkt_implied=b.p_mkt.mean(), roi_flat_yes=float((b.hit * dec - 1).mean()),
                                 flag="SMALL" if len(b) < 100 else ""))
        # "edge" picks: model >= market + 5 pts
        e = s[s.p_cal >= s.p_mkt + 0.05]
        if len(e):
            dec = np.where(e.odds > 0, 1 + e.odds / 100, 1 + 100 / e.odds.abs())
            rows.append(dict(window=w, model_tier="model >= market+5pts", n=len(e), pred=e.p_cal.mean(), hit=e.hit.mean(),
                             mkt_implied=e.p_mkt.mean(), roi_flat_yes=float((e.hit * dec - 1).mean()),
                             flag="SMALL" if len(e) < 100 else ""))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ablation
def ablation(u, log):
    tr, te = (u.season < REPORT_FROM).values, (u.season >= REPORT_FROM).values
    rows = []

    def ev(name, P):
        L = legs_long(u.iloc[np.flatnonzero(te)].reset_index(drop=True), P[te])
        p6 = np.c_[P[te][:, LEGS["A_KO"]].sum(1), P[te][:, LEGS["A_SUB"]].sum(1), P[te][:, 12],
                   P[te][:, LEGS["B_KO"]].sum(1), P[te][:, LEGS["B_SUB"]].sum(1), P[te][:, 13]]
        y = u.y6.values[te]
        rows.append(dict(variant=name, ll_6way=float(-np.log(np.clip(p6[np.arange(len(y)), y], 1e-6, 1)).mean()),
                         ll_legs=C.ll(L.p_model, L.hit), n_ge70=int((L.p_model >= .7).sum()),
                         hit_ge70=L.hit[L.p_model >= .7].mean(), n_ge80=int((L.p_model >= .8).sum()),
                         hit_ge80=L.hit[L.p_model >= .8].mean()))

    ev("script only", script_cells(u))
    ev("script+market blend (file)", blend_cells(u))
    def fitp(cols, Cc=CHOICE["C"]):
        m = Model(cols, Cc).fit(u[tr])
        P = script_cells(u)
        P[te] = m.cells(u[te])
        return P
    ev(f"chosen: {CHOICE['name']} (C={CHOICE['C']})", fitp(CHOICE["cols"]))
    full = BLEND + SCRIPT + sum(GROUPS.values(), [])
    ev("blend + script + all tendencies", fitp(full))
    for g in GROUPS:
        ev(f"  - without {g}", fitp([c for c in full if c not in GROUPS[g]]))
    for g in ("finish_tendency", "durability", "style", "weight_class", "rounds"):
        ev(f"  chosen + {g} only", fitp(CHOICE["cols"] + GROUPS[g]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def run(log=print):
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    u = load(log)
    sel = select_config(u, log)
    P = walk_cells(u, log=log)
    L = legs_long(u, P)
    types = sorted(L.typ.unique())
    L["key"] = L.typ.map({t: i for i, t in enumerate(types)})
    dk = L.date.values.astype("datetime64[D]").astype(int)
    L["p_cal"], L["p_platt"], _ = C.walk_calibrate(dk, L.key.values, L.p_model.values, L.hit.values, n0=40.0, nbins=10)
    Ps = script_cells(u)
    Ls = legs_long(u, Ps)
    L["p_script"] = Ls.p_model.values
    L["p_script_cal"], _, _ = C.walk_calibrate(dk, L.key.values, L.p_script.values, L.hit.values, n0=40.0, nbins=10)
    # naive base: prior-season rate of that leg type
    rate = L.groupby(["typ", "season"]).hit.agg(["sum", "size"]).reset_index().sort_values("season")
    rate["base_pos"] = (rate.groupby("typ")["sum"].cumsum() - rate["sum"]) / \
                       (rate.groupby("typ")["size"].cumsum() - rate["size"]).replace(0, np.nan)
    L = L.merge(rate[["typ", "season", "base_pos"]], on=["typ", "season"], how="left")
    log(f"  model + calibrator ({time.time()-T0:.0f}s)")

    M = L[L.season >= REPORT_FROM].copy()
    tiers_all = C.tier_table(M, "p_cal", "hit", "base_pos")
    M["legtype"] = M.typ
    tiers_by = C.tier_table(M, "p_cal", "hit", "base_pos", tiers=[0.60, 0.70, 0.80], group="legtype")
    tiers_by = tiers_by[tiers_by.n > 0]
    tiers_raw = C.tier_table(M, "p_model", "hit", "base_pos")
    lc = []
    for s, x in L.groupby("season"):
        hi = x[x.p_cal >= 0.7]
        lc.append(dict(season=s, n_legs=len(x), ll_script=C.ll(x.p_script, x.hit), ll_model=C.ll(x.p_model, x.hit),
                       ll_cal=C.ll(x.p_cal, x.hit), ece_script=C.ece(x.p_script, x.hit), ece_model=C.ece(x.p_model, x.hit),
                       ece_cal=C.ece(x.p_cal, x.hit), n_ge70=len(hi), pred_ge70=hi.p_cal.mean(), hit_ge70=hi.hit.mean()))
    lc = pd.DataFrame(lc)
    abl = ablation(u, log)
    abl_cal = pd.DataFrame([
        dict(variant="script only + calibrator", ll=C.ll(M.p_script_cal, M.hit), n_ge70=int((M.p_script_cal >= .7).sum()),
             hit_ge70=M.hit[M.p_script_cal >= .7].mean()),
        dict(variant="full walk-forward + calibrator", ll=C.ll(M.p_cal, M.hit), n_ge70=int((M.p_cal >= .7).sum()),
             hit_ge70=M.hit[M.p_cal >= .7].mean())])
    rl = real_line_table(real_lines(u, L))
    log(f"  evaluated ({time.time()-T0:.0f}s); parlays...")

    # parlays per event date
    pr = pd.read_parquet(RP / "ufc_preds.parquet")
    pr = pr.merge(u[["fight_id"]].reset_index().rename(columns={"index": "row"}), on="fight_id", how="inner")
    pr = pr[pr.winner.isin(["A", "B"])]
    ml = pd.DataFrame(dict(date=pr.date, row=pr.row, leg="ML", p=pr.pick_prob, p_raw=pr.pick_prob, hit=pr.win_hit,
                           label="ML " + np.where(pr.pick_A == 1, pr.fighter_a, pr.fighter_b)))
    LL = L[L.season >= REPORT_FROM].copy()
    LL["p"], LL["p_raw"] = LL.p_cal, LL.p_model
    LL = LL.sort_values("p", ascending=False).groupby(["row"]).head(6)
    out = []
    for d, day in LL.groupby("date"):
        for r in ufc_tickets(day[["row", "leg", "p", "p_raw", "hit", "label"]], P, ml[ml.date == d]):
            out.append(dict(date=d, season=int(day.season.iloc[0]), **r))
    Pt = pd.DataFrame(out)
    Pt.to_csv(OUT / "ufc_parlays.csv", index=False)
    PS = C.parlay_summary(Pt)
    ml_avail = dict(ml85=int((ml[ml.date.dt.year >= REPORT_FROM].groupby("date").p.max() >= 0.85).sum()),
                    ml70=int((ml[ml.date.dt.year >= REPORT_FROM].groupby("date").p.max() >= 0.70).sum()),
                    days=int(LL.date.nunique()))
    S = L[L.season >= FIRST_TEST]
    S = pd.DataFrame(dict(date=S.date.values, season=S.season.values.astype(np.int16), fight_id=u.fight_id.values[S.row],
                          fighter_a=u.fighter_a.values[S.row], fighter_b=u.fighter_b.values[S.row],
                          weight_class=u.weight_class.values[S.row], leg=S.leg.values, leg_type=S.typ.values,
                          label=S.label.values, side="yes", p_script=S.p_script.values.astype(np.float32),
                          p_model=S.p_model.values.astype(np.float32), p=S.p_cal.values.astype(np.float32),
                          hit=S.hit.values.astype(np.int8), base_pos=S.base_pos.values.astype(np.float32)))
    S.to_parquet(OUT / "ufc_legs.parquet", index=False, compression="zstd")
    np.save(OUT / "ufc_cells.npy", P.astype(np.float32))
    u[["fight_id", "date"]].to_parquet(OUT / "ufc_cells_index.parquet", index=False)
    secs = time.time() - T0
    write_report(dict(sel=sel, tiers_all=tiers_all, tiers_by=tiers_by, tiers_raw=tiers_raw, lc=lc, abl=abl, abl_cal=abl_cal, rl=rl,
                      PS=PS, ml_avail=ml_avail, secs=secs))
    log(f"  done in {secs/60:.1f} min")


def write_report(R):
    L = ["# UFC script-driven fight props & parlays\n",
         f"Run time {R['secs']/60:.1f} min. Outcome + timing models refit every season on prior seasons only; online "
         f"calibrator per leg type updated after every event; report = {REPORT_FROM}+. Under/over 1.5 / 2.5 use the real "
         "finish clock (2:30 of R2 / R3, ufcstats).\n",
         "## A. Real lines: market method props (winner by KO/SUB/DEC 'yes' prices, de-vigged)\n",
         "Only the 'yes' side is priced, so the pick is always 'yes'; table is by model probability tier. "
         "`roi_flat_yes` = flat 1u at the posted price.\n", C.md(R["rl"]), "",
         "## B. Milestone-style legs, calibrated probability (report seasons)\n",
         "`base` = prior-season hit rate of that leg type (fav/dog-oriented).\n", C.md(R["tiers_all"]), "",
         "### By leg type (cumulative tiers shown for 60/70/80)\n", C.md(R["tiers_by"]), "",
         "### Raw model probabilities\n", C.md(R["tiers_raw"]), "",
         "## C. Parlays (per event date)\n",
         f"Events with an ML >= 85%: {R['ml_avail']['ml85']} of {R['ml_avail']['days']}; >= 70%: {R['ml_avail']['ml70']}. "
         "`top` = N most likely legs, max 1 per fight; `sgp` = same-fight stack on the exact cell joint (non-redundant "
         "legs only), continuing into the next fight when one fight runs out; `mixed` = best ML + top N-1 props from "
         "other fights.\n", C.md(R["PS"]), "",
         "## D. Model choice (made on seasons <= 2020 only: train <= 2018, validate 2019-20)\n",
         C.md(R["sel"], ".4f"), "",
         "## D2. Ablation (train <= 2020, test 2021+, raw probabilities)\n", C.md(R["abl"], ".4f"), "",
         C.md(R["abl_cal"], ".4f"), "", "## E. Learning curve\n", C.md(R["lc"], ".4f")]
    (OUT / "ufc_props.md").write_text("\n".join(L))


def build_tickets(date, sizes=(3, 5, 7)):
    date = pd.Timestamp(date)
    legs = pd.read_parquet(OUT / "ufc_legs.parquet", filters=[("date", "==", date)])
    if legs.empty:
        return {"date": str(date.date()), "tickets": [], "note": "no scripted fights for this date"}
    P = np.load(OUT / "ufc_cells.npy")
    idx = pd.read_parquet(OUT / "ufc_cells_index.parquet").reset_index().rename(columns={"index": "row"})
    legs = legs.merge(idx[["fight_id", "row"]], on="fight_id")
    legs["p_raw"] = legs.p_model.astype(float)
    legs["p"] = legs.p.astype(float)
    legs = legs.sort_values("p", ascending=False).groupby("row").head(6)
    pr = pd.read_parquet(RP / "ufc_preds.parquet")
    pr = pr[pr.date == date].merge(idx[["fight_id", "row"]], on="fight_id")
    ml = pd.DataFrame(dict(date=pr.date, row=pr.row, leg="ML", p=pr.pick_prob, p_raw=pr.pick_prob, hit=pr.win_hit,
                           label="ML " + np.where(pr.pick_A == 1, pr.fighter_a, pr.fighter_b)))
    return {"date": str(date.date()), "tickets": ufc_tickets(legs[["row", "leg", "p", "p_raw", "hit", "label"]], P, ml, sizes)}


if __name__ == "__main__":
    run()
