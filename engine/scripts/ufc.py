"""
UFC adapter for the script-then-learn backtest: a FIGHT SCRIPT per fight instead of team scores.

For every fight (scripted 2016-2026, state warmed up on every UFC fight since 1994 from ufcstats round data):
  P(A wins); P(A by KO/TKO), P(A by SUB), P(A by DEC), same for B; P(goes distance); finish-round distribution
  (P(end in round r)), expected finish round; and the SGP fight legs (winner x method, winner x distance,
  winner x over/under 1.5 / 2.5 rounds) as JOINT probabilities from the same script.

Two sequential layers, both updated after every fight from its errors (no refitting on the future):
  1. FIGHTER STATE (online Bayesian / Kalman)
       skill        EKF Bradley-Terry rating (mean, variance) with time-based drift
       finish style decayed Beta counts: KO/SUB/DEC wins, KO/SUB/DEC losses (chin), shrunk to the population
       striking     decayed minute-weighted sig. strikes landed / absorbed per min, accuracy, knockdowns
       grappling    takedowns landed per 15 min, takedown defence, submission attempts, control share
       physical     age at fight (DOB), reach, height; layoff, UFC experience
     Same-date fights are all scripted before any state is updated.
  2. METHOD SCRIPT = conditional-logit (6-way softmax over A_KO, A_SUB, A_DEC, B_KO, B_SUB, B_DEC) with
     method-specific weights shared across corners (u(A,KO) = theta_KO . phi(A,B), u(B,KO) = theta_KO . phi(B,A)),
     trained by online AdaGrad on each fight's error (y - p). Weight-class base rates and 5-round flags are features.
     Round model: online Dirichlet counts of finish round by (method, scheduled rounds, finish-intensity bucket)
     and of first-half-of-round finishes (for 1.5 / 2.5 round lines).

Market: real method-prop odds (6 separate yes-props per fight) + moneyline. De-vig options compared on 2016-2020,
best one used as the market benchmark (documented in the report). Blend = log-linear pool, weight chosen on
2016-2020. Method-prop betting thresholds chosen on 2016-2020, reported 2021+.

Outputs: results/scripts/ufc_fights.parquet, results/scripts/ufc_learning.md

  python -m engine.scripts.ufc
"""
from __future__ import annotations

import difflib
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRECO = ROOT / "data" / "raw" / "ufc" / "greco"
OUT = ROOT / "results" / "scripts"
START, TUNE_END, HOLD = 2016, 2020, 2021
TRAIN_FROM = pd.Timestamp("2001-01-01")
CLASSES = ["Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
           "Light Heavyweight", "Heavyweight", "Middleweight", "Welterweight", "Lightweight", "Featherweight",
           "Bantamweight", "Flyweight", "Catch Weight", "Open Weight"]
OUTC = ["A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC"]
METH = ["KO", "SUB", "DEC"]


# ------------------------------------------------------------------ parsing
def _of(x):
    try:
        a, b = str(x).split(" of ")
        return float(a), float(b)
    except Exception:
        return np.nan, np.nan


def _mmss(x):
    try:
        m, s = str(x).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def method_cat(m):
    m = str(m).strip()
    if m.startswith("KO") or m.startswith("TKO"):
        return "KO"
    if m.startswith("Submission"):
        return "SUB"
    if m.startswith("Decision"):
        return "DEC"
    return "VOID"                                     # overturned, CNC, DQ, other


def wclass(s):
    for c in CLASSES:
        if c in s:
            return c
    return "Other"


def load_greco() -> pd.DataFrame:
    r = pd.read_csv(GRECO / "ufc_fight_results.csv")
    e = pd.read_csv(GRECO / "ufc_event_details.csv")
    r["EVENT"] = r.EVENT.str.strip(); r["BOUT"] = r.BOUT.str.strip()
    e["EVENT"] = e.EVENT.str.strip()
    r = r.merge(e[["EVENT", "DATE"]].drop_duplicates("EVENT"), on="EVENT", how="left")
    r["date"] = pd.to_datetime(r.DATE, format="%B %d, %Y", errors="coerce")
    sides = r.BOUT.str.split(" vs. ", n=1, expand=True)
    r["f1"], r["f2"] = sides[0].str.strip(), sides[1].str.strip()
    r["mcat"] = r.METHOD.map(method_cat)
    oc = r.OUTCOME.str.strip()
    r["win1"] = np.select([oc == "W/L", oc == "L/W", oc == "D/D"], [1.0, 0.0, 0.5], np.nan)
    r["round"] = pd.to_numeric(r.ROUND, errors="coerce")
    r["tsec"] = r.TIME.map(_mmss)
    fmt = r["TIME FORMAT"].astype(str)
    r["sched"] = np.where(fmt.str.startswith("5 Rnd"), 5, np.where(fmt.str.startswith("3 Rnd"), 3, 0))
    r["wc"] = r.WEIGHTCLASS.astype(str).map(wclass)
    r["title"] = r.WEIGHTCLASS.astype(str).str.contains("Title").astype(int)
    r["female"] = r.WEIGHTCLASS.astype(str).str.contains("Women").astype(int)
    r["secs"] = (r["round"] - 1) * 300 + r.tsec
    # per-fight per-fighter totals from round stats
    s = pd.read_csv(GRECO / "ufc_fight_stats.csv").dropna(subset=["FIGHTER"])
    s["EVENT"] = s.EVENT.str.strip(); s["BOUT"] = s.BOUT.str.strip(); s["FIGHTER"] = s.FIGHTER.str.strip()
    s["sl"], s["sa"] = zip(*s["SIG.STR."].map(_of))
    s["tdl"], s["tda"] = zip(*s["TD"].map(_of))
    s["ctrl"] = s.CTRL.map(_mmss)
    s["kd"] = pd.to_numeric(s.KD, errors="coerce"); s["suba"] = pd.to_numeric(s["SUB.ATT"], errors="coerce")
    st = s.groupby(["EVENT", "BOUT", "FIGHTER"])[["sl", "sa", "tdl", "tda", "ctrl", "kd", "suba"]].sum(min_count=1)
    st = st.reset_index()
    r = r.dropna(subset=["date", "f1", "f2"]).sort_values(["date", "URL"]).reset_index(drop=True)
    return r, st


def load_tott():
    t = pd.read_csv(GRECO / "ufc_fighter_tott.csv")
    t["FIGHTER"] = t.FIGHTER.str.strip()
    def inch(x):
        try:
            x = str(x).replace('"', "")
            if "'" in x:
                f, i = x.split("'")
                return int(f) * 12 + int(i or 0)
            return float(x)
        except Exception:
            return np.nan
    t["height"] = t.HEIGHT.map(inch) * 2.54
    t["reach"] = t.REACH.map(inch) * 2.54
    t["dob"] = pd.to_datetime(t.DOB, format="%b %d, %Y", errors="coerce")
    t = t.drop_duplicates("FIGHTER")                  # 9 duplicate names: keep first (documented)
    return t.set_index("FIGHTER")[["height", "reach", "dob", "STANCE"]]


# ------------------------------------------------------------------ fighter state
DECAY = 0.85          # weight kept by old fights when a new one is added (style drifts)
PRIOR_FIGHTS = 3.0    # Beta prior strength for finish tendencies
PRIOR_MIN = 20.0      # minutes of prior for per-minute rates


class Fighter:
    __slots__ = ("mu", "var", "last", "n", "cnt", "mins", "acc", "fights_ufc", "fin_rounds")

    def __init__(self):
        self.mu, self.var, self.last, self.n = 0.0, 1.0, None, 0
        self.cnt = defaultdict(float)         # wKO wSUB wDEC lKO lSUB lDEC (decayed)
        self.mins = 0.0
        self.acc = defaultdict(float)         # decayed sums: sl sa sab saa_opp tdl tda otdl otda suba kd okd ctrl octrl
        self.fights_ufc = 0
        self.fin_rounds = 0.0


class Pop:
    """Population (online) rates used as shrinkage priors."""
    def __init__(self):
        self.c = defaultdict(lambda: 1.0)
        self.tot = 6.0
        self.rate = dict(sl=3.5, sab=3.5, acc=0.45, td=1.3, tdd=0.62, suba=0.5, kd=0.25, ctrl=0.18)


def fstate_features(F: Fighter, O: Fighter, pop: Pop, tott, name, oname, date, wc_rates, sched, female, title, corner):
    """phi(att=F, def=O) - only pre-fight information."""
    def shr(k):
        n = sum(F.cnt[x] for x in ("wKO", "wSUB", "wDEC", "lKO", "lSUB", "lDEC"))
        p0 = pop.c[k] / pop.tot
        return (F.cnt[k] + PRIOR_FIGHTS * p0) / (n + PRIOR_FIGHTS)

    def shr_o(k):
        n = sum(O.cnt[x] for x in ("wKO", "wSUB", "wDEC", "lKO", "lSUB", "lDEC"))
        p0 = pop.c[k] / pop.tot
        return (O.cnt[k] + PRIOR_FIGHTS * p0) / (n + PRIOR_FIGHTS)

    def rate(S, k, per, prior):
        return (S.acc[k] + prior * PRIOR_MIN / per) / (S.mins + PRIOR_MIN) * per

    lg = lambda p: math.log(max(p, 1e-3) / max(1 - p, 1e-3))
    slpm_f = rate(F, "sl", 1, pop.rate["sl"]); sapm_f = rate(F, "sab", 1, pop.rate["sab"])
    slpm_o = rate(O, "sl", 1, pop.rate["sl"]); sapm_o = rate(O, "sab", 1, pop.rate["sab"])
    acc_f = (F.acc["sl"] + 20 * pop.rate["acc"]) / (F.acc["sa"] + 20)
    td_f = rate(F, "tdl", 15, pop.rate["td"])
    tdd_o = (O.acc["otda"] - O.acc["otdl"] + 4 * pop.rate["tdd"]) / (O.acc["otda"] + 4)
    tdd_f = (F.acc["otda"] - F.acc["otdl"] + 4 * pop.rate["tdd"]) / (F.acc["otda"] + 4)
    td_o = rate(O, "tdl", 15, pop.rate["td"])
    sub_f = rate(F, "suba", 15, pop.rate["suba"])
    kd_f = rate(F, "kd", 15, pop.rate["kd"])
    ctrl_f = (F.acc["ctrl"] + 60 * PRIOR_MIN * pop.rate["ctrl"]) / (60 * (F.mins + PRIOR_MIN))
    ctrl_o = (O.acc["ctrl"] + 60 * PRIOR_MIN * pop.rate["ctrl"]) / (60 * (O.mins + PRIOR_MIN))
    tf, to = tott.get(name), tott.get(oname)
    def age(t):
        return (date - t[2]).days / 365.25 if t is not None and pd.notna(t[2]) else np.nan
    af, ao = age(tf), age(to)
    rf, ro = (tf[1] if tf is not None else np.nan), (to[1] if to is not None else np.nan)
    hf, ho = (tf[0] if tf is not None else np.nan), (to[0] if to is not None else np.nan)
    lay = (date - F.last).days if F.last is not None else 365
    wk, ws = wc_rates
    nz = lambda x: 0.0 if not np.isfinite(x) else x
    return [
        F.mu - O.mu,                                   # skill
        lg(shr("wKO")), lg(shr_o("lKO")),              # finish: KO power vs opponent chin
        lg(shr("wSUB")), lg(shr_o("lSUB")),            # finish: SUB threat vs opp sub-defence
        lg(shr("wDEC") + shr("lDEC")), lg(shr_o("wDEC") + shr_o("lDEC")),   # decision tendencies
        lg(shr("lKO")),                                 # own chin
        slpm_f, sapm_o, (slpm_f - sapm_f) - (slpm_o - sapm_o), acc_f, kd_f,   # striking
        td_f * (1 - tdd_o), sub_f, ctrl_f - ctrl_o, td_o * (1 - tdd_f),       # grappling
        nz(af - ao), nz(ao) - 30.0, nz(rf - ro), nz(hf - ho),                 # physical
        math.log1p(min(lay, 2000)), math.log1p(F.fights_ufc), math.log1p(O.fights_ufc),  # layoff / experience
        lg(wk), lg(ws), float(sched == 5), float(female), float(title), corner,          # context
    ]


FEATS = ["skill", "ko_win", "opp_ko_loss", "sub_win", "opp_sub_loss", "dec_rate", "opp_dec_rate", "own_ko_loss",
         "slpm", "opp_sapm", "strike_diff", "sig_acc", "kd_rate", "td_eff", "sub_att", "ctrl_diff", "opp_td_eff",
         "age_diff", "opp_age", "reach_diff", "height_diff", "layoff", "exp", "opp_exp",
         "class_ko", "class_sub", "five_rd", "female", "title", "corner"]
GROUPS = {
    "skill rating": ["skill"],
    "finish tendencies (power/chin/sub)": ["ko_win", "opp_ko_loss", "sub_win", "opp_sub_loss", "dec_rate",
                                           "opp_dec_rate", "own_ko_loss"],
    "striking": ["slpm", "opp_sapm", "strike_diff", "sig_acc", "kd_rate"],
    "grappling": ["td_eff", "sub_att", "ctrl_diff", "opp_td_eff"],
    "physical (age/reach/height)": ["age_diff", "opp_age", "reach_diff", "height_diff"],
    "layoff/experience": ["layoff", "exp", "opp_exp"],
    "class base rates & 5 rounds": ["class_ko", "class_sub", "five_rd", "female", "title"],
}


def build_states(r: pd.DataFrame, st: pd.DataFrame, tott: pd.DataFrame, q_day=0.0006, q_fight=0.03, s_scale=1.0):
    """Pass 1: in date order, write features for both corners before the fight, then update fighter state."""
    stats = {(a, b, c): v for (a, b, c), v in zip(zip(st.EVENT, st.BOUT, st.FIGHTER),
                                                   st[["sl", "sa", "tdl", "tda", "ctrl", "kd", "suba"]].values)}
    T = {k: (float(h), float(rc), d) for k, h, rc, d in zip(tott.index, tott.height.values, tott.reach.values, tott.dob)}
    fighters: dict[str, Fighter] = {}
    pop = Pop()
    wc = defaultdict(lambda: [3.0, 1.5, 5.0])         # KO, SUB, DEC counts per class (Dirichlet prior)
    X1, X2, meta = [], [], []
    for d, blk in r.groupby("date", sort=True):
        rows = []
        for t in blk.itertuples(index=False):
            A = fighters.setdefault(t.f1, Fighter()); B = fighters.setdefault(t.f2, Fighter())
            for F in (A, B):                             # rating drift with time out
                if F.last is not None:
                    F.var = min(F.var + q_day * (d - F.last).days + q_fight, 1.5)
            c = wc[t.wc]; s = sum(c)
            rates = (c[0] / s, c[1] / s)
            x1 = fstate_features(A, B, pop, T, t.f1, t.f2, d, rates, t.sched, t.female, t.title, 0.0)
            x2 = fstate_features(B, A, pop, T, t.f2, t.f1, d, rates, t.sched, t.female, t.title, 0.0)
            pA = 1 / (1 + math.exp(-(A.mu - B.mu) / s_scale))
            X1.append(x1); X2.append(x2)
            meta.append(dict(URL=t.URL, p_skill=pA, mu1=A.mu, mu2=B.mu, var1=A.var, var2=B.var,
                             n1=A.fights_ufc, n2=B.fights_ufc))
            rows.append(t)
        for t in rows:                                   # reveal & learn
            A, B = fighters[t.f1], fighters[t.f2]
            y = t.win1
            if y == y:                                   # EKF Bradley-Terry update (wins, losses, draws)
                p = 1 / (1 + math.exp(-(A.mu - B.mu) / s_scale))
                h = p * (1 - p) / s_scale
                S = h * h * (A.var + B.var) + p * (1 - p)
                ka, kb = A.var * h / S, B.var * h / S
                A.mu += ka * (y - p); B.mu -= kb * (y - p)
                A.var = max(A.var - ka * h * A.var, 0.02); B.var = max(B.var - kb * h * B.var, 0.02)
            m = t.mcat
            mins = t.secs / 60 if t.secs == t.secs else 0.0
            sa_ = stats.get((t.EVENT, t.BOUT, t.f1)); sb_ = stats.get((t.EVENT, t.BOUT, t.f2))
            for F, me, op, won in ((A, sa_, sb_, y == 1), (B, sb_, sa_, y == 0)):
                if y == y and y != 0.5 and m in METH:
                    for k in F.cnt:
                        F.cnt[k] *= DECAY
                    F.cnt[("w" if won else "l") + m] += 1
                    if won and m != "DEC":
                        F.fin_rounds += t["round"] if isinstance(t, dict) else t.round
                if me is not None and op is not None and mins > 0 and np.isfinite(me).all() and np.isfinite(op).all():
                    for k in list(F.acc):
                        F.acc[k] *= DECAY
                    F.mins = F.mins * DECAY + mins
                    sl, sa, tdl, tda, ctrl, kd, suba = me
                    osl, osa, otdl, otda, octrl, okd, osuba = op
                    for k, v in (("sl", sl), ("sa", sa), ("sab", osl), ("tdl", tdl), ("tda", tda), ("otdl", otdl),
                                 ("otda", otda), ("suba", suba), ("kd", kd), ("okd", okd), ("ctrl", ctrl),
                                 ("octrl", octrl)):
                        F.acc[k] += v
                F.last = d; F.fights_ufc += 1
            if y == y and y != 0.5 and m in METH:
                wc[t.wc][METH.index(m)] += 1
                for k in ("wKO", "wSUB", "wDEC"):
                    pass
                pop.c["w" + m] += 1; pop.c["l" + m] += 1; pop.tot += 2
    meta = pd.DataFrame(meta)
    return np.array(X1, float), np.array(X2, float), meta


# ------------------------------------------------------------------ online conditional-logit method script
class MethodLearner:
    def __init__(self, k, lr=0.05, l2=1e-4):
        self.th = np.zeros((3, k + 1)); self.G = np.full((3, k + 1), 1e-6); self.lr, self.l2 = lr, l2
        self.th[:, 0] = [np.log(0.32), np.log(0.18), np.log(0.5)]

    def probs(self, z1, z2):
        u = np.r_[self.th @ z1, self.th @ z2]
        u -= u.max()
        e = np.exp(u)
        return e / e.sum()

    def update(self, z1, z2, yk):
        p = self.probs(z1, z2)
        y = np.zeros(6); y[yk] = 1
        g = np.zeros_like(self.th)
        for m in range(3):
            g[m] = (p[m] - y[m]) * z1 + (p[m + 3] - y[m + 3]) * z2 + self.l2 * self.th[m]
        self.G += g * g
        self.th -= self.lr * g / np.sqrt(self.G)


class Norm:
    def __init__(self, k):
        self.n, self.m, self.s = 0, np.zeros(k), np.zeros(k)

    def z(self, x):
        if self.n < 30:
            return np.zeros_like(x)
        sd = np.sqrt(self.s / (self.n - 1)); sd[sd < 1e-9] = 1
        return np.clip((x - self.m) / sd, -4, 4)

    def upd(self, x):
        self.n += 1; d = x - self.m; self.m += d / self.n; self.s += d * (x - self.m)


def run_method(r, X1, X2, keep, lr=0.05, l2=1e-4):
    """Pass 2: online softmax. Same-date fights scripted before any update. Returns (n,6) probabilities."""
    X1, X2 = X1[:, keep], X2[:, keep]
    k = X1.shape[1]
    ml, nz = MethodLearner(k, lr, l2), Norm(k)
    P = np.zeros((len(r), 6))
    yk = r.yk.values; train = (r.date >= TRAIN_FROM).values
    dates = r.date.values
    i, n = 0, len(r)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        Z = [(np.r_[1.0, nz.z(X1[t])], np.r_[1.0, nz.z(X2[t])]) for t in range(i, j)]
        for t, (z1, z2) in zip(range(i, j), Z):
            P[t] = ml.probs(z1, z2)
        for t, (z1, z2) in zip(range(i, j), Z):
            if yk[t] >= 0 and train[t]:
                ml.update(z1, z2, yk[t]); nz.upd(X1[t]); nz.upd(X2[t])
        i = j
    return P, ml


def round_model(r, P):
    """Online Dirichlet counts of finish round | (method, scheduled rounds, intensity bucket) and of finishes in the
    first half of a round. Returns per-fight round distributions for KO and SUB finishes and the first-half share."""
    cnt = defaultdict(lambda: None)
    half = [10.0, 10.0]
    RK, RS, H = [], [], []
    fin = P[:, [0, 1, 3, 4]].sum(1)
    dates = r.date.values
    n = len(r); i = 0
    def key(m, R, f):
        return (m, R, int(min(f, 0.79) // 0.2))
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        for t in range(i, j):
            R = 5 if r.sched.iat[t] == 5 else 3
            out = []
            for m in ("KO", "SUB"):
                c = cnt[key(m, R, fin[t])]
                if c is None:
                    c = np.array([5.0, 3.0, 2.0, 1.0, 1.0][:R])
                out.append(c / c.sum())
            RK.append(out[0]); RS.append(out[1]); H.append(half[0] / sum(half))
        for t in range(i, j):
            m, R, rd = r.mcat.iat[t], (5 if r.sched.iat[t] == 5 else 3), r["round"].iat[t]
            if m in ("KO", "SUB") and r.sched.iat[t] in (3, 5) and rd == rd and 1 <= rd <= R:
                kk = key(m, R, fin[t])
                if cnt[kk] is None:
                    cnt[kk] = np.array([5.0, 3.0, 2.0, 1.0, 1.0][:R])
                cnt[kk][int(rd) - 1] += 1
                ts = r.tsec.iat[t]
                if ts == ts:
                    half[0 if ts < 150 else 1] += 1
        i = j
    return RK, RS, np.array(H)


# ------------------------------------------------------------------ market
def imp(o):
    o = np.asarray(o, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def dec(o):
    o = np.asarray(o, float)
    return np.where(o < 0, 1 + 100 / -o, 1 + o / 100)


def market_methods(u: pd.DataFrame) -> dict:
    """Candidate de-vig methods for the six method props (A/B x KO/SUB/DEC, each a separate yes-prop)."""
    q = np.c_[[imp(u[c]) for c in ["a_ko_odds", "a_sub_odds", "a_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"]]].T
    qa, qb = imp(u.a_ml), imp(u.b_ml)
    pA = qa / (qa + qb)
    out = {}
    out["mult6"] = q / q.sum(1, keepdims=True)
    # power: p_k = q_k^k with sum 1 (shrinks long-shot props more)
    lo, hi = np.ones(len(q)), np.full(len(q), 4.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        s = (q ** mid[:, None]).sum(1)
        lo, hi = np.where(s > 1, mid, lo), np.where(s > 1, hi, mid)
    out["power6"] = q ** ((lo + hi) / 2)[:, None]
    out["power6"] /= out["power6"].sum(1, keepdims=True)
    # ML-anchored: winner probability from the (more liquid) moneyline, split by each fighter's own 3 props
    for nm, base in (("ml_anchor_mult", q), ("ml_anchor_power", out["power6"])):
        a = base[:, :3] / base[:, :3].sum(1, keepdims=True) * pA[:, None]
        b = base[:, 3:] / base[:, 3:].sum(1, keepdims=True) * (1 - pA)[:, None]
        out[nm] = np.c_[a, b]
    return out


def ll6(P, yk):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(yk)), yk], 1e-6, 1))))


def ll2(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pool(Ps, Pm, w):
    L = (1 - w) * np.log(np.clip(Ps, 1e-9, 1)) + w * np.log(np.clip(Pm, 1e-9, 1))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    r, st = load_greco()
    tott = load_tott()
    r["yk"] = -1
    w1, w2 = r.win1 == 1, r.win1 == 0
    for j, m in enumerate(METH):
        r.loc[w1 & (r.mcat == m), "yk"] = j
        r.loc[w2 & (r.mcat == m), "yk"] = j + 3
    print("greco fights", len(r), f"{time.time()-t0:.0f}s")
    X1, X2, meta = build_states(r, st, tott)
    print("states", X1.shape, f"{time.time()-t0:.0f}s")

    # ---- odds / outcomes file, oriented to greco side 1
    u = pd.read_csv(ROOT / "data" / "processed" / "ufc.csv", low_memory=False)
    u = u.merge(r[["URL", "f1", "f2"]].reset_index().rename(columns={"index": "ri"}),
                left_on="ufcstats_url", right_on="URL", how="inner")
    def same(a, b):
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    u["a_is_1"] = [same(a, f1) >= same(a, f2) for a, f1, f2 in zip(u.fighter_a, u.f1, u.f2)]
    u = u.drop_duplicates("ri")
    u["date"] = pd.to_datetime(u.date)
    u["season"] = u.date.dt.year

    def oriented(P6, a_is_1):     # greco side-1/2 -> A/B
        P6 = np.asarray(P6)
        return np.where(a_is_1[:, None], P6, P6[:, [3, 4, 5, 0, 1, 2]])

    a1 = u.a_is_1.values
    ri = u.ri.values
    u["yk"] = -1
    wa, wb = u.winner == "A", u.winner == "B"
    mc = u.method.map(lambda m: "KO" if m == "KO/TKO" else "SUB" if m == "SUB" else "DEC" if "DEC" in str(m) else "VOID")
    for j, m in enumerate(METH):
        u.loc[wa & (mc == m), "yk"] = j
        u.loc[wb & (mc == m), "yk"] = j + 3
    u["mcat"] = mc
    tune = ((u.season >= START) & (u.season <= TUNE_END)).values
    hold = (u.season >= HOLD).values
    ev = (u.yk >= 0).values

    # ---- tune learner (lr, l2) on 2016-2020 6-way log loss
    best = None
    allk = list(range(len(FEATS)))
    for lr in (0.005, 0.01, 0.02, 0.05):
        for l2 in (1e-4, 1e-3):
            P, _ = run_method(r, X1, X2, allk, lr, l2)
            Pu = oriented(P[ri], a1)
            s = ll6(Pu[tune & ev], u.yk.values[tune & ev])
            if best is None or s < best[0]:
                best = (s, lr, l2)
    _, lr, l2 = best
    print("tuned", best, f"{time.time()-t0:.0f}s")
    P, learner = run_method(r, X1, X2, allk, lr, l2)
    RK, RS, H = round_model(r, P)
    Pu = oriented(P[ri], a1)

    # ---- ablation
    abl = []
    yk = u.yk.values
    def scores(Pq):
        return (ll6(Pq[tune & ev], yk[tune & ev]), ll6(Pq[hold & ev], yk[hold & ev]),
                ll2(Pq[hold & ev][:, :3].sum(1), (yk[hold & ev] < 3).astype(float)))
    base = scores(Pu)
    abl.append(dict(variant="all features", tune_ll6=base[0], hold_ll6=base[1], hold_ll_win=base[2]))
    for g, cols in GROUPS.items():
        keep = [i for i, f in enumerate(FEATS) if f not in cols]
        Pq, _ = run_method(r, X1, X2, keep, lr, l2)
        s = scores(oriented(Pq[ri], a1))
        abl.append(dict(variant=f"drop {g}", tune_ll6=s[0], hold_ll6=s[1], hold_ll_win=s[2]))
    Pq, _ = run_method(r, X1, X2, [], lr, l2)
    s = scores(oriented(Pq[ri], a1))
    abl.append(dict(variant="base rates only (intercepts)", tune_ll6=s[0], hold_ll6=s[1], hold_ll_win=s[2]))
    abl = pd.DataFrame(abl)
    abl["hold_d_ll6_vs_all"] = abl.hold_ll6 - base[1]
    print(abl.round(4).to_string(), f"{time.time()-t0:.0f}s")

    # ---- fight script table
    f = pd.DataFrame(dict(fight_id=u.ufcstats_url.str.rsplit("/", n=1).str[-1], date=u.date, season=u.season,
                          event=u.event, fighter_a=u.fighter_a, fighter_b=u.fighter_b, weight_class=u.weight_class,
                          gender=u.gender, scheduled_rounds=u.scheduled_rounds, title_bout=u.title_bout))
    for j, c in enumerate(OUTC):
        f[f"p_{c}"] = Pu[:, j]
    f["p_A_win"] = Pu[:, :3].sum(1)
    f["p_distance"] = Pu[:, 2] + Pu[:, 5]
    m = meta.iloc[ri].reset_index(drop=True)
    f["skill_a"] = np.where(a1, m.mu1, m.mu2); f["skill_b"] = np.where(a1, m.mu2, m.mu1)
    f["p_A_skill_only"] = np.where(a1, m.p_skill, 1 - m.p_skill)
    # round distributions (end round), joint legs
    R = np.where(u.scheduled_rounds.values == 5, 5, 3)
    rk = [RK[i] for i in ri]; rs = [RS[i] for i in ri]; hh = H[ri]
    pr = np.zeros((len(u), 5)); A_r = np.zeros((len(u), 5)); B_r = np.zeros((len(u), 5))
    for i in range(len(u)):
        k1, k2 = rk[i], rs[i]
        n = len(k1)
        A_r[i, :n] = f.p_A_KO.iat[i] * k1 + f.p_A_SUB.iat[i] * k2
        B_r[i, :n] = f.p_B_KO.iat[i] * k1 + f.p_B_SUB.iat[i] * k2
    pr = A_r + B_r
    for k in range(5):
        f[f"p_end_r{k+1}"] = pr[:, k]
    rr = np.arange(1, 6)
    fin = pr.sum(1)
    f["exp_finish_round_if_finish"] = (pr * rr).sum(1) / np.maximum(fin, 1e-9)
    f["exp_end_round"] = (pr * rr).sum(1) + R * f.p_distance
    # over/under rounds: line 1.5 = before 2:30 of R2; 2.5 = before 2:30 of R3 (first-half share learned online)
    def under(Mr, line):
        full = int(line)            # rounds completely before the line
        return Mr[:, :full].sum(1) + hh * Mr[:, full]
    for side, Mr, win in (("A", A_r, f.p_A_win), ("B", B_r, 1 - f.p_A_win)):
        f[f"p_{side}_itd"] = f[f"p_{side}_KO"] + f[f"p_{side}_SUB"]
        f[f"p_{side}_win_distance"] = f[f"p_{side}_DEC"]
        for line in (1.5, 2.5):
            un = under(Mr, line)
            ls = str(line).replace(".", "_")
            f[f"p_{side}_win_under_{ls}"] = un
            f[f"p_{side}_win_over_{ls}"] = win - un
    for line in (1.5, 2.5):
        f[f"p_under_{str(line).replace('.', '_')}"] = under(pr, line)
    # market
    mk = market_methods(u)
    has6 = np.isfinite(u[["a_ko_odds", "a_sub_odds", "a_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds",
                          "a_ml", "b_ml"]].values).all(1)
    # data-quality gate (uses prices only, never outcomes): the six props must carry a plausible overround and
    # agree with the moneyline on who wins. 2025-26 rows in the source data often fail this (sum of implied
    # probabilities < 1, i.e. stale / mislabelled prices) and are excluded from the market benchmark and bets.
    q6 = np.c_[[imp(u[c]) for c in ["a_ko_odds", "a_sub_odds", "a_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"]]].T
    ovr = q6.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dis = np.abs(q6[:, :3].sum(1) / ovr - imp(u.a_ml) / (imp(u.a_ml) + imp(u.b_ml)))
    has6_raw = has6.copy()
    has6 = has6 & (ovr >= 1.08) & (ovr <= 1.45) & (dis <= 0.08)
    qual = pd.DataFrame(dict(season=u.season, raw=has6_raw, kept=has6)).groupby("season").sum()
    qual = qual[qual.index >= START]
    dv = {}
    for nm, Pm in mk.items():
        sel = tune & ev & has6
        dv[nm] = ll6(Pm[sel], yk[sel])
    best_dv = min(dv, key=dv.get)
    Pm = mk[best_dv]
    for j, c in enumerate(OUTC):
        f[f"mkt_{c}"] = np.where(has6, Pm[:, j], np.nan)
    qa, qb = imp(u.a_ml), imp(u.b_ml)
    f["mkt_A_win_ml"] = qa / (qa + qb)
    for c in ["a_ml", "b_ml", "a_ko_odds", "a_sub_odds", "a_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"]:
        f[c] = u[c].values
    # blend weight (log-linear pool) chosen on 2016-2020
    sel_t = tune & ev & has6
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    ll_w = {w: ll6(pool(Pu[sel_t], Pm[sel_t], w), yk[sel_t]) for w in ws}
    w_bl = min(ll_w, key=ll_w.get)
    Pb = np.where(has6[:, None], pool(Pu, np.where(has6[:, None], Pm, Pu), w_bl), Pu)
    for j, c in enumerate(OUTC):
        f[f"blend_{c}"] = Pb[:, j]
    f["w_mkt_method"] = w_bl
    # win blend vs ML
    ok_ml = np.isfinite(f.mkt_A_win_ml.values)
    yw = (u.winner == "A").astype(float).values
    evw = u.winner.isin(["A", "B"]).values
    selw = tune & evw & ok_ml
    lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    ll_ww = {w: ll2(1 / (1 + np.exp(-((1 - w) * lgt(f.p_A_win.values[selw]) + w * lgt(f.mkt_A_win_ml.values[selw])))), yw[selw]) for w in ws}
    w_win = min(ll_ww, key=ll_ww.get)
    f["p_A_win_blend"] = np.where(ok_ml, 1 / (1 + np.exp(-((1 - w_win) * lgt(f.p_A_win.values) + w_win * lgt(np.nan_to_num(f.mkt_A_win_ml.values, nan=0.5))))), f.p_A_win)
    # outcomes
    f["winner"] = u.winner.values; f["method"] = u.method.values; f["method_cat"] = u.mcat.values
    f["finish_round"] = u.finish_round.values; f["outcome_k"] = yk
    f["outcome"] = [OUTC[k] if k >= 0 else None for k in yk]
    f["went_distance"] = np.where(ev, (u.mcat == "DEC").astype(float), np.nan)
    f["n_ufc_a"] = np.where(a1, m.n1, m.n2); f["n_ufc_b"] = np.where(a1, m.n2, m.n1)
    f["mkt_quality_ok"] = has6
    out = f[f.season >= START].reset_index(drop=True)
    out.to_parquet(OUT / "ufc_fights.parquet", index=False)
    report(out, abl, dv, best_dv, ll_w, w_bl, ll_ww, w_win, (lr, l2), learner, u, has6, f, qual, rs)
    print(f"done {time.time()-t0:.0f}s")


def bets_table(f, tune, hold):
    """Bet a method prop when model EV > tau. Source and tau chosen on 2016-2020 (>=100 bets), reported 2021+."""
    odds_cols = dict(A_KO="a_ko_odds", A_SUB="a_sub_odds", A_DEC="a_dec_odds", B_KO="b_ko_odds",
                     B_SUB="b_sub_odds", B_DEC="b_dec_odds")
    rows = []
    for c, oc in odds_cols.items():
        o = f[oc].values
        ok = np.isfinite(o) & np.isfinite(f[f"mkt_{c}"].values)
        void = f.outcome_k.values < 0
        drawish = f.winner.isin(["draw"]).values
        won = (f.outcome.values == c)
        pnl = np.where(won, dec(np.nan_to_num(o, nan=100)) - 1, -1.0)
        pnl = np.where(void & ~drawish, 0.0, pnl)       # NC / overturned / DQ / CNC -> void; draw -> loss
        for src in ("p", "blend"):
            p = f[f"{src}_{c}"].values
            evv = p * dec(np.nan_to_num(o, nan=100)) - 1
            rows.append(pd.DataFrame(dict(leg=c, src=src, ev=evv, pnl=pnl, ok=ok, season=f.season.values,
                                          odds=o, p=p, mkt=f[f"mkt_{c}"].values, won=won.astype(float))))
    b = pd.concat(rows)
    b = b[b.ok]
    res = []
    for src in ("p", "blend"):
        for tau in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            for nm, m in (("tune", (b.season >= START) & (b.season <= TUNE_END)), ("hold", b.season >= HOLD),
                          ("hold21_24", (b.season >= HOLD) & (b.season <= 2024)), ("hold25_26", b.season >= 2025)):
                x = b[m & (b.src == src) & (b.ev > tau)]
                res.append(dict(src=src, tau=tau, split=nm, bets=len(x), roi=x.pnl.mean() if len(x) else np.nan,
                                hit=x.won.mean() if len(x) else np.nan, avg_odds=x.odds.median() if len(x) else np.nan))
    return pd.DataFrame(res), b


def report(f, abl, dv, best_dv, ll_w, w_bl, ll_ww, w_win, hp, learner, u, has6, fall, qual, rs):
    L = ["# UFC fight scripts: learning report\n"]
    ev = f.outcome_k >= 0
    m6 = ev & f.mkt_A_KO.notna()
    tune = (f.season >= START) & (f.season <= TUNE_END); hold = f.season >= HOLD
    L.append(f"Fights scripted 2016-2026: {len(f):,} ({ev.sum():,} with a KO/SUB/DEC result; {m6.sum():,} also have "
             f"all six method props + moneyline). Fighter state warmed up on every UFC fight since 1993 "
             f"(ufcstats round stats, greco dump); the method learner trains online from 2001. Learner AdaGrad "
             f"lr/l2 chosen on 2016-2020: {hp}.\n")
    L.append("## De-vigging the method props\n")
    L.append("Each fighter's KO/TKO, SUB and DEC props are separate yes-props, but the six (A/B x 3) partition "
             "every non-draw outcome, so their implied probabilities sum to 1 + overround (median overround "
             f"{np.nanmedian(np.c_[[imp(u[c]) for c in ['a_ko_odds','a_sub_odds','a_dec_odds','b_ko_odds','b_sub_odds','b_dec_odds']]].T.sum(1)[has6]) - 1:.1%}). "
             "Candidates (6-way log loss on 2016-2020 fights with all six props):\n")
    L.append("* `mult6` - divide the six implied probabilities by their sum\n* `power6` - p_k = q_k^k, k solved so sum = 1 "
             "(removes more vig from long shots)\n* `ml_anchor_mult` / `ml_anchor_power` - fighter win probability "
             "from the de-vigged moneyline (more liquid, lower margin), split across that fighter's three methods in "
             "proportion to their (raw / power-de-vigged) prop prices\n")
    L.append(pd.Series(dv, name="ll6_2016_20").round(4).to_frame().to_markdown())
    L.append("\n**Data-quality gate** (prices only, no outcomes): keep a fight's props only if the six implied "
             "probabilities sum to 1.08-1.45 and their implied winner probability is within 0.08 of the de-vigged "
             "moneyline. In 2016-2024 this removes ~1-2% of fights; in 2025-26 the source odds are badly broken "
             "(a third of fights have the six props summing to < 1, e.g. a -270 favourite at +1700 to win by KO "
             "but -110 by decision; blind-betting every KO prop in 2025 'returns' +50-80%), so most 2025-26 fights "
             "fail. Fights with all six props / kept after the gate:\n")
    L.append(qual.T.to_markdown())
    L.append(f"\nChosen market benchmark: **{best_dv}**. Draws (~1%) are not priced by the six props and are excluded "
             "from log loss; NC/overturned/DQ/CNC are void.\n")
    # ---- per season table
    L.append("## Per-season learning curve (log loss; lower is better)\n")
    rows = []
    yk = f.outcome_k.values
    for s, x in f[m6].groupby("season"):
        k = x.outcome_k.values
        S = x[[f"p_{c}" for c in OUTC]].values; M = x[[f"mkt_{c}" for c in OUTC]].values
        B = x[[f"blend_{c}" for c in OUTC]].values
        yw = (k < 3).astype(float); yd = np.isin(k, [2, 5]).astype(float)
        rows.append(dict(season=s, fights=len(x), ll6_script=ll6(S, k), ll6_market=ll6(M, k), ll6_blend=ll6(B, k),
                         win_script=ll2(x.p_A_win, yw), win_market_ml=ll2(x.mkt_A_win_ml, yw),
                         win_blend=ll2(x.p_A_win_blend, yw),
                         dist_script=ll2(x.p_distance, yd), dist_market=ll2(x.mkt_A_DEC + x.mkt_B_DEC, yd),
                         dist_blend=ll2(x.blend_A_DEC + x.blend_B_DEC, yd)))
    cur = pd.DataFrame(rows)
    L.append(cur.round(4).to_markdown(index=False))
    for nm, mm in (("2016-2020 (tuning)", tune), ("2021-2026 holdout", hold)):
        x = f[m6 & mm]; k = x.outcome_k.values
        S = x[[f"p_{c}" for c in OUTC]].values; M = x[[f"mkt_{c}" for c in OUTC]].values
        B = x[[f"blend_{c}" for c in OUTC]].values
        meth3 = lambda P: np.c_[P[:, 0] + P[:, 3], P[:, 1] + P[:, 4], P[:, 2] + P[:, 5]]
        k3 = k % 3
        L.append(f"\n**{nm}** ({len(x):,} fights): 6-way winner x method log loss script {ll6(S,k):.4f} / market "
                 f"{ll6(M,k):.4f} / blend {ll6(B,k):.4f};  3-way method (KO/SUB/DEC) script {ll6(meth3(S),k3):.4f} / "
                 f"market {ll6(meth3(M),k3):.4f} / blend {ll6(meth3(B),k3):.4f};  winner script "
                 f"{ll2(x.p_A_win,(k<3).astype(float)):.4f} / ML {ll2(x.mkt_A_win_ml,(k<3).astype(float)):.4f} / "
                 f"blend {ll2(x.p_A_win_blend,(k<3).astype(float)):.4f}")
    base6 = None
    L.append(f"\nBlend weights chosen on 2016-2020: method log-linear pool w_market = **{w_bl}**, winner w_market = "
             f"**{w_win}**.\n")
    # ---- calibration of script by outcome type (holdout)
    x = f[ev & hold]
    cal = []
    for c in OUTC + ["distance"]:
        if c == "distance":
            p, y = x.p_distance, x.went_distance
            pm = (x.mkt_A_DEC + x.mkt_B_DEC)
        else:
            p, y = x[f"p_{c}"], (x.outcome == c).astype(float); pm = x[f"mkt_{c}"]
        cal.append(dict(outcome=c, mean_script=p.mean(), mean_market=pm.mean(), actual=y.mean()))
    L.append("Holdout calibration-in-the-large (2021+):\n")
    L.append(pd.DataFrame(cal).round(3).to_markdown(index=False))
    # rounds
    rd = x[(x.method_cat.isin(["KO", "SUB"])) & x.finish_round.between(1, 5)]
    pr = rd[[f"p_end_r{i}" for i in range(1, 6)]].values
    prn = pr / pr.sum(1, keepdims=True)
    k = rd.finish_round.values.astype(int) - 1
    base_r = np.array([np.mean(f[tune & f.method_cat.isin(['KO','SUB']) & (f.scheduled_rounds == R)].finish_round == i + 1)
                       for R in (3,) for i in range(5)])
    L.append(f"\nFinish-round distribution given a finish (2021+, {len(rd):,} finishes): script log loss "
             f"{ll6(prn, k):.4f} vs static 2016-20 base rates {ll6(np.tile(np.clip(base_r,1e-3,1)/np.clip(base_r,1e-3,1).sum(), (len(k),1)), k):.4f}; "
             f"expected finish round MAE {np.mean(np.abs(rd.exp_finish_round_if_finish - rd.finish_round)):.3f}.\n")
    # ---- ablation
    L.append("## Ablation (method learner re-run without each feature group)\n")
    L.append("ll6 = 6-way winner x method log loss; `hold_d_ll6_vs_all` > 0 means dropping the group HURT the 2021+ "
             "holdout (the group helps).\n")
    L.append(abl.round(4).to_markdown(index=False))
    th = learner.th
    L.append("\nFinal learned weights (standardised features) per method:\n")
    L.append(pd.DataFrame(th.T, index=["intercept"] + FEATS, columns=["KO", "SUB", "DEC"]).round(3).to_markdown())
    # ---- error attribution
    L.append("\n## What the script got wrong most (2021+)\n")
    x = f[m6 & hold].copy()
    k = x.outcome_k.values
    S = x[[f"p_{c}" for c in OUTC]].values; M = x[[f"mkt_{c}" for c in OUTC]].values
    x["ll_s"] = -np.log(np.clip(S[np.arange(len(k)), k], 1e-6, 1)); x["ll_m"] = -np.log(np.clip(M[np.arange(len(k)), k], 1e-6, 1))
    seg = []
    for nm, mm in (("all", x.season > 0), ("5-round", x.scheduled_rounds == 5), ("women", x.gender == "F"),
                   ("heavyweight", x.weight_class == "Heavyweight"), ("flyweight/bantam", x.weight_class.isin(["Flyweight", "Bantamweight"])),
                   ("A or B UFC debut", (x.n_ufc_a == 0) | (x.n_ufc_b == 0)), ("both 5+ UFC fights", (x.n_ufc_a >= 5) & (x.n_ufc_b >= 5)),
                   ("result KO", x.method_cat == "KO"), ("result SUB", x.method_cat == "SUB"), ("result DEC", x.method_cat == "DEC"),
                   ("underdog won (ML)", ((x.mkt_A_win_ml > .5) & (k >= 3)) | ((x.mkt_A_win_ml < .5) & (k < 3)))):
        y = x[mm]
        seg.append(dict(segment=nm, fights=len(y), ll6_script=y.ll_s.mean(), ll6_market=y.ll_m.mean(), gap=y.ll_s.mean() - y.ll_m.mean()))
    L.append(pd.DataFrame(seg).round(4).to_markdown(index=False))
    # split the 6-way gap into winner part and method-given-winner part
    pw_s = np.where(k < 3, S[:, :3].sum(1), S[:, 3:].sum(1)); pw_m = np.where(k < 3, M[:, :3].sum(1), M[:, 3:].sum(1))
    pm_s = S[np.arange(len(k)), k] / pw_s; pm_m = M[np.arange(len(k)), k] / pw_m
    L.append(f"\nGap decomposition 2021+: winner part {np.mean(-np.log(pw_s)) - np.mean(-np.log(pw_m)):+.4f}, "
             f"method-given-winner part {np.mean(-np.log(pm_s)) - np.mean(-np.log(pm_m)):+.4f} nats/fight "
             "(positive = market better).\n")
    # ---- betting
    L.append("## Betting method props where the script disagrees with the market\n")
    res, b = bets_table(f, tune, hold)
    t = res[(res.split == "tune") & (res.bets >= 100)].sort_values("roi")
    if len(t):
        bs = t.iloc[-1]
        h = res[(res.split == "hold") & (res.src == bs.src) & (res.tau == bs.tau)].iloc[0]
        h1 = res[(res.split == "hold21_24") & (res.src == bs.src) & (res.tau == bs.tau)].iloc[0]
        h2 = res[(res.split == "hold25_26") & (res.src == bs.src) & (res.tau == bs.tau)].iloc[0]
        L.append(f"Rule chosen on 2016-2020 (max ROI, >=100 bets): source=`{bs.src}`, EV threshold tau={bs.tau}: "
                 f"tune {int(bs.bets)} bets ROI {bs.roi:+.1%}; **holdout 2021+ {int(h.bets)} bets ROI {h.roi:+.1%}** "
                 f"(hit {h.hit:.1%}, median odds {h.avg_odds:+.0f}); of which 2021-2024 {int(h1.bets)} bets ROI {h1.roi:+.1%}, "
                 f"2025-26 (suspect odds even after the gate) {int(h2.bets)} bets ROI {h2.roi:+.1%}. Flat 1u; "
                 "NC/DQ/overturned void, draws lose.\n")
        hb = b[(b.season >= HOLD) & (b.season <= 2024) & (b.src == bs.src) & (b.ev > bs.tau)]
        if len(hb):
            boot = [hb.pnl.sample(frac=1, replace=True, random_state=i).mean() for i in range(500)]
            L.append(f"2021-2024 holdout ROI bootstrap 90% interval: [{np.percentile(boot,5):+.1%}, {np.percentile(boot,95):+.1%}]. "
                     "By leg:\n")
            L.append(hb.groupby("leg").agg(bets=("pnl", "size"), roi=("pnl", "mean"), hit=("won", "mean")).round(3).to_markdown())
    L.append("\nFull grid:\n")
    L.append(res.round(3).to_markdown(index=False))
    # ---- SGP legs
    L.append("\n## SGP fight legs: joint probabilities from the script (2021+)\n")
    x = f[ev & hold].copy()
    yA = (x.outcome_k < 3).astype(float)
    itdA = x.outcome.isin(["A_KO", "A_SUB"]).astype(float)
    dist = x.went_distance
    # over 1.5 rounds actual: fight lasts past 2:30 of R2
    fr = x.finish_round
    endsec = np.where(x.method_cat == "DEC", 99999, np.nan)
    uu = u.set_index(u.ufcstats_url.str.rsplit("/", n=1).str[-1])
    rows = []
    legs = [("A wins & inside distance", x.p_A_itd, x.p_A_win * (x.p_A_itd + x.p_B_itd), itdA,
             x.mkt_A_KO + x.mkt_A_SUB),
            ("A wins & goes distance", x.p_A_win_distance, x.p_A_win * x.p_distance, yA * dist, x.mkt_A_DEC),
            ("B wins & inside distance", x.p_B_itd, (1 - x.p_A_win) * (x.p_A_itd + x.p_B_itd),
             x.outcome.isin(["B_KO", "B_SUB"]).astype(float), x.mkt_B_KO + x.mkt_B_SUB),
            ("B wins & goes distance", x.p_B_win_distance, (1 - x.p_A_win) * x.p_distance, (1 - yA) * dist, x.mkt_B_DEC),
            ("A wins by KO", x.p_A_KO, x.p_A_win * (x.p_A_KO + x.p_B_KO), (x.outcome == "A_KO").astype(float), x.mkt_A_KO),
            ("favourite (ML) wins & goes distance",
             np.where(x.mkt_A_win_ml >= .5, x.p_A_DEC, x.p_B_DEC),
             np.where(x.mkt_A_win_ml >= .5, x.p_A_win, 1 - x.p_A_win) * x.p_distance,
             np.where(x.mkt_A_win_ml >= .5, x.outcome == "A_DEC", x.outcome == "B_DEC").astype(float),
             np.where(x.mkt_A_win_ml >= .5, x.mkt_A_DEC, x.mkt_B_DEC)),
            ("favourite (ML) wins inside distance",
             np.where(x.mkt_A_win_ml >= .5, x.p_A_itd, x.p_B_itd),
             np.where(x.mkt_A_win_ml >= .5, x.p_A_win, 1 - x.p_A_win) * (x.p_A_itd + x.p_B_itd),
             np.where(x.mkt_A_win_ml >= .5, x.outcome.isin(["A_KO", "A_SUB"]), x.outcome.isin(["B_KO", "B_SUB"])).astype(float),
             np.where(x.mkt_A_win_ml >= .5, x.mkt_A_KO + x.mkt_A_SUB, x.mkt_B_KO + x.mkt_B_SUB))]
    for nm, pj, pind, y, pmk in legs:
        pj, pind, y, pmk = map(lambda v: np.asarray(v, float), (pj, pind, y, pmk))
        okm = np.isfinite(pmk)
        rows.append(dict(leg=nm, n=len(y), actual=y.mean(), script_joint=pj.mean(), independent_product=pind.mean(),
                         ll_joint=ll2(pj, y), ll_independent=ll2(pind, y),
                         ll_market=ll2(pmk[okm], y[okm]), ll_joint_same_fights=ll2(pj[okm], y[okm])))
    L.append("`script_joint` = the script's joint probability; `independent_product` = P(winner) x P(method) as if the "
             "legs were independent (how a naive parlay is priced); `ll_market` = the market's own priced joint "
             "(winner-by-method props), on fights where it exists.\n")
    L.append(pd.DataFrame(rows).round(4).to_markdown(index=False))
    L.append(f"\nRound-based SGP legs are in the parquet (`p_A_win_over_1_5`, `p_A_win_under_2_5`, `p_under_1_5`, ...): "
             f"no round-prop prices exist in the data, so they are evaluated only for calibration: mean P(fight ends "
             f"under 2.5 rounds) {x['p_under_2_5'].mean():.3f}.\n")
    L.append("## Verdict\n")
    hx = f[m6 & hold]; k = hx.outcome_k.values
    S = hx[[f"p_{c}" for c in OUTC]].values; M = hx[[f"mkt_{c}" for c in OUTC]].values
    B = hx[[f"blend_{c}" for c in OUTC]].values
    L.append(f"On 2021+ the method market (6-way ll {ll6(M,k):.4f}) is "
             f"{'more' if ll6(M,k) < ll6(S,k) else 'less'} accurate than the stand-alone script ({ll6(S,k):.4f}); the "
             f"2016-20-chosen blend scores {ll6(B,k):.4f}. "
             "See the betting section for whether disagreement is profitable after the vig; treat any holdout "
             "ROI with the bootstrap interval in mind.")
    (OUT / "ufc_learning.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
