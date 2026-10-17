"""
Shared machinery for the NBA / NFL script-driven props + parlay pipeline (engine/props_script/nba.py, nfl.py).

  * grouped EWMs of PRIOR values only (player form, dispersion, opponent-vs-position)
  * CDFModel: distributional LightGBM. Each player-game is expanded over thresholds k and the model learns
    P(stat >= k | script, tendencies) with monotone (decreasing) constraints in k. One model per sport answers
    milestones ("k+") and real lines (over x.5 == >= x+1) alike.
  * OnlineCal: per (group, probability-bin) bias correction learned from every past hit/miss in date order
    (exponentially decayed counts, shrunk to the raw probability). Strictly prior dates only.
  * Correlations (teammate / opponent / player-vs-game) estimated from residuals of PRIOR seasons, Gaussian
    copula joint probabilities, and the 3/5/7-leg ticket constructions.
  * Report helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def american_to_dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / np.abs(a))


def dec_to_american(d):
    d = np.asarray(d, float)
    return np.where(d >= 2, (d - 1) * 100.0, -100.0 / (d - 1))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=20):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / max(len(p), 1))


# ============================================================ grouped EWMs (prior values only)
def grp_ewm_prev(df: pd.DataFrame, keys: list[str], col: str, hl: float) -> np.ndarray:
    """EWM (half-life hl rows) of PRIOR values of `col` within groups. df must be sorted by keys + time."""
    codes = df.groupby(keys, sort=False, observed=True, dropna=False).ngroup().to_numpy()
    prev = df[col].groupby(codes).shift(1)
    out = prev.groupby(codes).ewm(halflife=hl, ignore_na=True).mean()
    out = out.reset_index(level=0, drop=True)
    return out.reindex(df.index).to_numpy()


def grp_count_prev(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    return df.groupby(keys, sort=False).cumcount().to_numpy()


def date_grp_ewm_prev(df: pd.DataFrame, keys: list[str], col: str, hl: float, date="date") -> np.ndarray:
    """EWM over prior DATES of the per-(keys, date) mean of col (no same-date leakage), merged back."""
    a = df.groupby(keys + [date], sort=True, observed=True)[col].mean().reset_index()
    a = a.sort_values(keys + [date]).reset_index(drop=True)
    a["_v"] = grp_ewm_prev(a, keys, col, hl)
    return df[keys + [date]].merge(a[keys + [date, "_v"]], on=keys + [date], how="left")["_v"].to_numpy()


# ============================================================ distributional LightGBM
K_FEATS = ["k", "zk", "rk"]


def add_k_feats(X: pd.DataFrame, k: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> pd.DataFrame:
    X = X.copy()
    X["k"] = k.astype(np.float32)
    X["zk"] = ((k - 0.5 - mu) / np.maximum(sd, 0.25)).astype(np.float32)
    X["rk"] = np.log((k + 0.5) / (np.maximum(mu, 0) + 0.5)).astype(np.float32)
    return X


@dataclass
class CDFModel:
    feats: list[str]
    cats: list[str] = field(default_factory=list)
    n_estimators: int = 300
    params: dict = field(default_factory=dict)
    model: lgb.Booster | None = None

    def _X(self, df, k):
        X = df[self.feats].astype(np.float32)
        return add_k_feats(X, np.asarray(k, float), df["mu"].to_numpy(float), df["sd"].to_numpy(float))

    def fit(self, df: pd.DataFrame, ks: list[np.ndarray], y_fn, max_rows=1_000_000, seed=0):
        """ks: list of threshold arrays aligned with df (NaN = skip). y_fn(df, k) -> 0/1."""
        parts, ys = [], []
        for k in ks:
            m = ~np.isnan(k)
            parts.append(self._X(df[m], k[m]))
            ys.append(y_fn(df[m], k[m]))
        X = pd.concat(parts, ignore_index=True)
        y = np.concatenate(ys)
        if len(X) > max_rows:
            idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
            X, y = X.iloc[idx], y[idx]
        cols = list(X.columns)
        mono = [-1 if c in K_FEATS else 0 for c in cols]
        p = dict(objective="binary", learning_rate=0.06, num_leaves=63, min_data_in_leaf=300,
                 feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 monotone_constraints=mono, num_threads=2, max_bin=63,
                 verbose=-1, seed=seed)
        p.update(self.params)
        ds = lgb.Dataset(X, y, categorical_feature=[c for c in self.cats if c in cols], free_raw_data=True)
        self.model = lgb.train(p, ds, num_boost_round=self.n_estimators)
        return self

    def predict(self, df: pd.DataFrame, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, float)
        out = np.full(len(df), np.nan)
        m = ~np.isnan(k)
        if m.any():
            out[m] = self.model.predict(self._X(df[m], k[m]))
        return out


def param_cdf(mu, sd, k):
    """Script-only parametric P(X >= k) for integer-valued stats (normal + continuity correction)."""
    return 1 - norm.cdf((np.asarray(k, float) - 0.5 - mu) / np.maximum(sd, 0.25))


# ============================================================ online calibrator
class OnlineCal:
    """p_cal = p + sum_w(y - p) / (sum_w + m) in the (group, logit-bin) cell, counts decayed by lam per date.
    Vectorised: for each row only legs from STRICTLY EARLIER dates enter its cell."""

    def __init__(self, nbins=24, m=150.0, lam=0.9993, span=4.6):
        self.edges = np.linspace(-span, span, nbins - 1)
        self.m, self.lam = m, lam
        self.state: pd.DataFrame | None = None   # final per-cell decayed sums (for live use)
        self.t_last = 0

    def bins(self, p):
        return np.digitize(logit(p), self.edges)

    def run(self, date, grp, p, y) -> np.ndarray:
        d = pd.DataFrame({"t": pd.to_datetime(date).values.astype("datetime64[D]").astype(np.int64),
                          "g": np.asarray(grp).astype(str), "b": self.bins(p), "p": p, "y": y})
        d["t"] -= d.t.min()
        t0 = int(d.t.min())
        w = self.lam ** (-(d.t - t0).to_numpy(float))
        d["wr"] = w * (d.y - d.p)
        d["wn"] = w
        a = d.groupby(["g", "b", "t"], sort=True)[["wr", "wn"]].sum().reset_index()
        a[["cr", "cn"]] = a.groupby(["g", "b"])[["wr", "wn"]].cumsum().to_numpy() - a[["wr", "wn"]].to_numpy()
        x = d[["g", "b", "t"]].merge(a[["g", "b", "t", "cr", "cn"]], on=["g", "b", "t"], how="left")
        scale = self.lam ** (d.t.to_numpy(float) - t0)
        r, n = x.cr.to_numpy() * scale, x.cn.to_numpy() * scale
        last = a.groupby(["g", "b"]).agg(r=("wr", "sum"), n=("wn", "sum")).reset_index()
        tl = int(d.t.max())
        last["r"] *= self.lam ** (tl - t0)
        last["n"] *= self.lam ** (tl - t0)
        self.state = last
        return np.clip(p + r / (n + self.m), 0.001, 0.999)

    def apply_state(self, grp, p):
        """Calibrate new legs with the final state (live use)."""
        if self.state is None:
            return p
        d = pd.DataFrame({"g": np.asarray(grp).astype(str), "b": self.bins(p)})
        x = d.merge(self.state, on=["g", "b"], how="left").fillna(0)
        return np.clip(p + x.r.to_numpy() / (x.n.to_numpy() + self.m), 0.001, 0.999)


# ============================================================ correlations
def pair_corr(z: pd.DataFrame, stats: list[str]) -> dict:
    """z: columns game_id, team, stat, z (standardised residual), only 'relevant' players.
    Returns {'tm': {(a,b): rho}, 'op': {(a,b): rho}} = mean product over distinct-player pairs."""
    w = z.pivot_table(index=["game_id", "team", "pid"], columns="stat", values="z", aggfunc="first")
    w = w.reindex(columns=stats)
    out = {"tm": {}, "op": {}}
    gt = w.groupby(level=[0, 1])
    S = gt.sum(min_count=1)
    N = gt.count()
    for i, a in enumerate(stats):
        for b in stats[i:]:
            both = w[[a, b]].dropna() if a != b else w[[a]].dropna()
            if a == b:
                cross = (both[a] ** 2).groupby(level=[0, 1]).sum()
                nab = both[a].groupby(level=[0, 1]).count()
            else:
                cross = (both[a] * both[b]).groupby(level=[0, 1]).sum()
                nab = both[a].groupby(level=[0, 1]).count()
            sa, sb = S[a].fillna(0), S[b].fillna(0)
            na, nb = N[a], N[b]
            cr = cross.reindex(sa.index).fillna(0)
            nn = nab.reindex(sa.index).fillna(0)
            num = (sa * sb - cr).sum()
            den = (na * nb - nn).sum()
            out["tm"][(a, b)] = out["tm"][(b, a)] = float(num / den) if den > 0 else 0.0
            # opponents: team sums of a times opponent team sums of b
            ta = sa.reset_index()
            tb = sb.reset_index()
            ta.columns = ["game_id", "team", "sa"]
            tb.columns = ["game_id", "team2", "sb"]
            na_ = na.reset_index(name="na")
            nb_ = nb.reset_index(name="nb")
            ta = ta.merge(na_, on=["game_id", "team"])
            tb = tb.merge(nb_.rename(columns={"team": "team2"}), on=["game_id", "team2"])
            j = ta.merge(tb, on="game_id")
            j = j[j.team != j.team2]
            num = (j.sa * j.sb).sum()
            den = (j.na * j.nb).sum()
            r = float(num / den) if den > 0 else 0.0
            out["op"][(a, b)] = out["op"][(b, a)] = r
    return out


def game_corr(z: pd.DataFrame, stats: list[str]) -> dict:
    """z with columns stat, z, mres (team margin residual std), tres (game total residual std)."""
    out = {"m": {}, "t": {}}
    for s in stats:
        x = z[z.stat == s]
        out["m"][s] = float(np.nanmean(x.z * x.mres)) if len(x) else 0.0
        out["t"][s] = float(np.nanmean(x.z * x.tres)) if len(x) else 0.0
    return out


# ============================================================ copula joint + tickets
_RNG = np.random.default_rng(12345)
_ZBANK = _RNG.standard_normal((20000, 12))


def leg_rho(a: dict, b: dict, C: dict) -> float:
    if a["game_id"] != b["game_id"]:
        return 0.0
    ka, kb = a["kind"], b["kind"]
    if ka == "T" and kb == "T" or ka == "M" and kb == "M":
        return 0.0
    if ka == "P" and kb == "P":
        if a["pid"] == b["pid"]:
            return 0.6 * a["sgn"] * b["sgn"]
        tbl = C["pp"]["tm"] if a["team"] == b["team"] else C["pp"]["op"]
        return a["sgn"] * b["sgn"] * tbl.get((a["stat"], b["stat"]), 0.0)
    if kb == "P":
        a, b = b, a
        ka, kb = kb, ka
    if ka == "P" and kb == "M":
        r = C["g"]["m"].get(a["stat"], 0.0)
        return a["sgn"] * (r if a["team"] == b["team"] else -r)
    if ka == "P" and kb == "T":
        return a["sgn"] * b["sgn"] * C["g"]["t"].get(a["stat"], 0.0)
    return 0.0   # M-T


def corr_matrix(legs: list[dict], C: dict) -> np.ndarray:
    n = len(legs)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = leg_rho(legs[i], legs[j], C)
    return R


def _psd(R):
    w, V = np.linalg.eigh(R)
    if w.min() > 1e-6:
        return R
    w = np.clip(w, 1e-6, None)
    R2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def joint_prob(p: np.ndarray, R: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    n = len(p)
    if n == 0:
        return 1.0
    if np.allclose(R, np.eye(n)):
        return float(np.prod(p))
    L = np.linalg.cholesky(_psd(R))
    Z = _ZBANK[:, :n] @ L.T
    a = norm.ppf(1 - p)
    return float(np.mean(np.all(Z > a, axis=1)))


def ticket_joint(legs: list[dict], C: dict) -> float:
    """Product over games of within-game copula joints."""
    by = {}
    for l in legs:
        by.setdefault(l["game_id"], []).append(l)
    out = 1.0
    for g, ls in by.items():
        p = np.array([l["p"] for l in ls])
        out *= joint_prob(p, corr_matrix(ls, C)) if len(ls) > 1 else p[0]
    return out


def pair_approx_gain(cur: list[dict], cand: dict, C: dict) -> float:
    """log of the multiplicative effect of adding cand: p_c * prod_j (1 + rho phi phi / (p p))."""
    pc = min(max(cand["p"], 1e-4), 1 - 1e-4)
    g = math.log(pc)
    for l in cur:
        r = leg_rho(l, cand, C)
        if r:
            pl = min(max(l["p"], 1e-4), 1 - 1e-4)
            g += math.log(max(1e-3, 1 + r * cand["phi"] * l["phi"] / (pc * pl)))
    return g


def top_n(legs: pd.DataFrame, n: int, one_per="pid") -> pd.DataFrame | None:
    x = legs.sort_values("p", ascending=False)
    if one_per:
        x = x.drop_duplicates(one_per)
    return x.head(n) if len(x) >= n else None


def greedy_sgp(game_legs: list[dict], n: int, C: dict) -> list[dict] | None:
    """Pick n legs from one game maximising the (pairwise-approx) correlated joint; one leg per player."""
    if len(game_legs) < n:
        return None
    cand = sorted(game_legs, key=lambda l: -l["p"])[:max(4 * n, 16)]
    cur: list[dict] = []
    used = set()
    for _ in range(n):
        best, bg = None, -1e9
        for c in cand:
            key = c.get("pid") if c["kind"] == "P" else c["kind"]
            if key in used:
                continue
            g = pair_approx_gain(cur, c, C)
            if g > bg:
                best, bg = c, g
        if best is None:
            return None
        cur.append(best)
        used.add(best.get("pid") if best["kind"] == "P" else best["kind"])
    return cur


def leg_dicts(df: pd.DataFrame) -> list[dict]:
    cols = ["game_id", "kind", "pid", "team", "stat", "sgn", "p", "hit", "dec", "label"]
    phi = norm.pdf(norm.ppf(1 - np.clip(df.p.to_numpy(float), 1e-4, 1 - 1e-4)))
    return [dict(zip(cols, r), phi=f) for r, f in zip(df[cols].itertuples(index=False, name=None), phi)]


def build_day(props: pd.DataFrame, glegs: pd.DataFrame, C: dict, sizes=(3, 5, 7), real: pd.DataFrame | None = None,
              ml_min=0.85) -> list[dict]:
    """All ticket constructions for one day. props/glegs/real: leg frames with the columns in leg_dicts.
    Returns rows: construction, n_legs, p_joint, p_indep, hit, dec_price, legs(label string)."""
    rows = []

    def rec(cons, n, legs):
        pj = ticket_joint(legs, C)
        rows.append({"construction": cons, "n_legs": n, "p_joint": pj,
                     "p_indep": float(np.prod([l["p"] for l in legs])),
                     "hit": int(all(l["hit"] == 1 for l in legs)),
                     "dec": float(np.prod([l["dec"] for l in legs])),
                     "legs": " | ".join(f"{l['label']} ({l['p']:.2f})" for l in legs)})
        return pj

    allP = leg_dicts(props)
    for n in sizes:
        cands = []
        # 1. top-N across the day, max 1 per player
        t = top_n(props, n)
        if t is not None:
            pj = rec("1 top-N cross-game", n, leg_dicts(t))
            cands.append((pj, "1"))
        if real is not None and len(real):
            t = top_n(real, n)
            if t is not None:
                rec("1R top-N real lines", n, leg_dicts(t))
        # 2. script-correlated same-game stack (props + game ML/total legs), best game of the day
        best = None
        gl = leg_dicts(glegs) if glegs is not None and len(glegs) else []
        games = props.groupby("game_id").p.apply(lambda s: np.sort(s.to_numpy())[::-1][:n].prod()).sort_values(ascending=False)
        for gid in games.index[:4]:
            legs = [l for l in allP if l["game_id"] == gid] + [l for l in gl if l["game_id"] == gid]
            s = greedy_sgp(legs, n, C)
            if s is None:
                continue
            pj = ticket_joint(s, C)
            if best is None or pj > best[0]:
                best = (pj, s)
        if best is not None:
            rec("2 same-game script stack", n, best[1])
        # 3. mixed: best 85%+ ML of the day + top n-1 props
        if gl:
            mls = [l for l in gl if l["kind"] == "M" and l["p"] >= ml_min]
            if mls:
                ml = max(mls, key=lambda l: l["p"])
                t = top_n(props, n - 1)
                if t is not None:
                    rec("3 mixed 85%+ ML + props", n, [ml] + leg_dicts(t))
    # 4. most likely ticket of the day per size (max predicted joint over constructions 1-3)
    df = pd.DataFrame(rows)
    if len(df):
        m = df[df.construction.str[0].isin(["1", "2", "3"]) & ~df.construction.str.startswith("1R")]
        for n, x in m.groupby("n_legs"):
            r = x.loc[x.p_joint.idxmax()].to_dict()
            r["construction"] = "4 most-likely ticket"
            rows.append(r)
    return rows


# ============================================================ report helpers
def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        vals = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                vals.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                vals.append(str(v))
        out += "| " + " | ".join(vals) + " |\n"
    return out


def tier_table(df: pd.DataFrame, p="p", y="hit", cuts=(0.5, 0.55, 0.6, 0.65, 0.7, 1.01), extra=None) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        x = df[(df[p] >= lo) & (df[p] < hi)]
        r = {"tier": f"{lo:.2f}-{min(hi, 1):.2f}", "n": len(x), "pred": x[p].mean() if len(x) else np.nan,
             "hit": x[y].mean() if len(x) else np.nan}
        if extra:
            for k, f in extra.items():
                r[k] = f(x) if len(x) else np.nan
        rows.append(r)
    for lo in cuts[1:-1]:
        x = df[df[p] >= lo]
        r = {"tier": f">= {lo:.2f}", "n": len(x), "pred": x[p].mean() if len(x) else np.nan,
             "hit": x[y].mean() if len(x) else np.nan}
        if extra:
            for k, f in extra.items():
                r[k] = f(x) if len(x) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t["se"] = np.sqrt(t.hit * (1 - t.hit) / t.n.clip(lower=1))
    return t


def parlay_summary(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, n), x in T.groupby(["construction", "n_legs"]):
        am = dec_to_american(np.median(x.dec))
        rows.append({"construction": c, "legs": n, "n_days": len(x), "pred_joint": x.p_joint.mean(),
                     "pred_indep": x.p_indep.mean(), "realized": x.hit.mean(),
                     "se": math.sqrt(max(x.hit.mean() * (1 - x.hit.mean()), 1e-9) / len(x)),
                     "typical_price": f"{am:+.0f}"})
    return pd.DataFrame(rows)


def tendency_table(P: pd.DataFrame, specs: list[tuple[str, str, list]], stats: list[str]) -> pd.DataFrame:
    """Mean script residual z = (actual - mu)/sd by tendency bucket and stat (what the script misses and the
    distributional model can learn). specs: (label, column, bin edges)."""
    rows = []
    for label, col, edges in specs:
        b = pd.cut(P[col], edges, include_lowest=True)
        for (bk, st), x in P.groupby([b, P.stat], observed=True):
            if st in stats and len(x) >= 200:
                rows.append({"tendency": label, "bucket": str(bk), "stat": st, "n": len(x),
                             "mean_z": x.z.mean(), "actual/mu": x.actual.sum() / x.mu.sum()})
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    w = t.pivot_table(index=["tendency", "bucket"], columns="stat", values="mean_z", sort=False).reset_index()
    n = t.groupby(["tendency", "bucket"], sort=False).n.sum().reset_index()
    return n.merge(w, on=["tendency", "bucket"])
