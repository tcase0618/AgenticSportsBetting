"""
Shared machinery for the script-driven props / parlay engines of the count sports (NHL, MLB) and UFC.

  * count tails         nb_ge(k, mu, r) = P(X >= k) for a negative binomial (r = inf -> Poisson)
  * dispersion          fit_r_by_mu(): method-of-moments NB size per mu-bin (training rows only)
  * tendencies layer    fit_offset_gbm(): LightGBM Poisson on top of the script mean (init_score = log mu)
  * online calibrator   OnlineCalibrator: per key (stat x threshold) online Platt (AdaGrad) + per
                        (key x confidence-bin) decayed residual table; walk_calibrate() runs it strictly in
                        date order (a day's legs are predicted with state built from earlier days only)
  * copula correlations latent (Gaussian-copula) correlations between legs estimated from randomized-PIT
                        normal scores of prior seasons (pair types: teammates / opponents / own-team margin /
                        game total), joint P(all legs) by Monte Carlo with shared draws
  * parlays             top-N cross-game, script-correlated same-game stacks, ML + props mixed; per-day
                        predicted joint vs realised
  * reports             tier tables, learning curves (ECE by season), markdown helpers
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from engine.scripts.sgp_builder import bvn_cdf

TIERS = [0.70, 0.75, 0.80, 0.85, 0.90]


# ------------------------------------------------------------------ small math
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def nb_ge(k, mu, r):
    """P(X >= k), X ~ NB(mean mu, size r); r >= 1e6 or inf -> Poisson."""
    k, mu, r = np.broadcast_arrays(np.asarray(k, float), np.asarray(mu, float), np.asarray(r, float))
    mu = np.maximum(mu, 1e-6)
    out = np.empty(k.shape)
    pois = ~np.isfinite(r) | (r >= 1e6)
    out[pois] = poisson.sf(k[pois] - 1, mu[pois])
    nb = ~pois
    out[nb] = nbinom.sf(k[nb] - 1, r[nb], r[nb] / (r[nb] + mu[nb]))
    return out


def nb_cdf(x, mu, r):
    x, mu, r = np.broadcast_arrays(np.asarray(x, float), np.asarray(mu, float), np.asarray(r, float))
    mu = np.maximum(mu, 1e-6)
    out = np.empty(x.shape)
    pois = ~np.isfinite(r) | (r >= 1e6)
    out[pois] = poisson.cdf(x[pois], mu[pois])
    out[~pois] = nbinom.cdf(x[~pois], r[~pois], r[~pois] / (r[~pois] + mu[~pois]))
    return out


def pit_z(x, mu, r, seed=0):
    """Randomized PIT normal score of a count outcome under NB(mu, r)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    lo = np.where(x > 0, nb_cdf(x - 1, mu, r), 0.0)
    hi = nb_cdf(x, mu, r)
    u = lo + rng.random(len(x)) * (hi - lo)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def fit_r_by_mu(y, mu, nbins=8, rmin=1.0, rmax=500.0):
    """Method-of-moments NB size per mu-quantile bin -> (edges, r per bin). r >= rmax means ~Poisson."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    edges = np.unique(np.quantile(mu, np.linspace(0, 1, nbins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, len(edges) - 2)
    rs = []
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() < 50:
            rs.append(rmax); continue
        exc = np.mean((y[m] - mu[m]) ** 2 - mu[m])
        rs.append(float(np.clip(np.mean(mu[m] ** 2) / exc, rmin, rmax)) if exc > 1e-9 else rmax)
    return edges, np.array(rs)


def apply_r(mu, edges, rs):
    b = np.clip(np.searchsorted(edges, np.asarray(mu, float), side="right") - 1, 0, len(rs) - 1)
    r = rs[b].astype(float)
    return np.where(r >= 499.0, np.inf, r)


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(p, y, bins=20):
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    n = np.bincount(b, minlength=bins)
    sp = np.bincount(b, p, minlength=bins)
    sy = np.bincount(b, y, minlength=bins)
    m = n > 0
    return float(np.sum(np.abs(sp[m] - sy[m])) / max(len(p), 1))


def fair_american(p):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    d = 1 / p
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"{-100 / (d - 1):.0f}"


# ------------------------------------------------------------------ tendencies layer
def fit_offset_gbm(X, y, offset, params=None, rounds=250):
    import lightgbm as lgb
    p = dict(objective="poisson", learning_rate=0.08, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, max_bin=63, verbose=-1, num_threads=2, seed=7)
    if params:
        p.update(params)
    ds = lgb.Dataset(X, y, init_score=offset, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


def gbm_mu(model, X, offset):
    return np.exp(model.predict(X, raw_score=True) + offset)


def walk_forward_gbm(df, feats, target, offset_col, season_col, first_test, rounds=250, params=None,
                     min_train_rows=5000, log=print, refit_every=1):
    """Yearly expanding refit: season s predicted by a model trained on seasons < s (refit every `refit_every`
    seasons; in between, the latest model — trained on strictly earlier seasons — is reused). Seasons before
    first_test get the plain offset (script) mean."""
    mu = np.exp(df[offset_col].values).copy()
    seasons = sorted(df[season_col].unique())
    imp, m, k = None, None, 0
    for s in seasons:
        if s < first_test:
            continue
        tr = (df[season_col] < s).values
        te = (df[season_col] == s).values
        if tr.sum() < min_train_rows:
            continue
        if m is None or k % refit_every == 0 or s == seasons[-1]:
            m = fit_offset_gbm(df.loc[tr, feats], df.loc[tr, target].values, df.loc[tr, offset_col].values,
                               params=params, rounds=rounds)
            imp = pd.Series(m.feature_importance("gain"), index=feats)
            log(f"    {target}: season {s} trained on {tr.sum()} rows")
        k += 1
        mu[te] = gbm_mu(m, df.loc[te, feats], df.loc[te, offset_col].values)
    return mu, imp


# ------------------------------------------------------------------ online calibrator
class OnlineCalibrator:
    """Per key: p1 = sigmoid(a + b*logit(p)) (online Platt, AdaGrad, starts at identity); then a decayed
    residual table per (key, confidence bin): p2 = p1 + sum(y - p1) / (n + n0)."""

    def __init__(self, n_keys, nbins=20, lr=0.03, n0=150.0, decay=0.997):
        self.a = np.zeros(n_keys); self.b = np.ones(n_keys)
        self.Ga = np.full(n_keys, 1e-8); self.Gb = np.full(n_keys, 1e-8)
        self.res = np.zeros((n_keys, nbins)); self.cnt = np.zeros((n_keys, nbins))
        self.nbins, self.lr, self.n0, self.decay, self.K = nbins, lr, n0, decay, n_keys

    def _p1(self, key, p):
        return sigmoid(self.a[key] + self.b[key] * logit(p))

    def _bin(self, p1):
        return np.clip((p1 * self.nbins).astype(int), 0, self.nbins - 1)

    def predict(self, key, p):
        p1 = self._p1(key, p)
        bi = self._bin(p1)
        adj = self.res[key, bi] / (self.cnt[key, bi] + self.n0)
        return np.clip(p1 + adj, 1e-4, 1 - 1e-4), p1

    def update(self, key, p, y):
        z = logit(p)
        p1 = self._p1(key, p)
        e = y - p1
        n = np.bincount(key, minlength=self.K).astype(float)
        ga = np.bincount(key, e, minlength=self.K) / np.maximum(n, 1)
        gb = np.bincount(key, e * z, minlength=self.K) / np.maximum(n, 1)
        self.Ga += ga ** 2; self.Gb += gb ** 2
        self.a += self.lr * ga / np.sqrt(self.Ga)
        self.b += self.lr * gb / np.sqrt(self.Gb)
        self.res *= self.decay; self.cnt *= self.decay
        bi = self._bin(p1)
        flat = key * self.nbins + bi
        self.res += np.bincount(flat, e, minlength=self.K * self.nbins).reshape(self.K, self.nbins)
        self.cnt += np.bincount(flat, minlength=self.K * self.nbins).reshape(self.K, self.nbins)


def walk_calibrate(dates, keys, p, y, **kw):
    """Run the calibrator strictly in date order. Returns (p_cal, p_platt, calibrator). Legs whose outcome is
    NaN (void / not yet played) are predicted but never used for updates."""
    dates = np.asarray(dates); keys = np.asarray(keys, int); p = np.asarray(p, float); y = np.asarray(y, float)
    order = np.argsort(dates, kind="stable")
    cal = OnlineCalibrator(int(keys.max()) + 1, **kw)
    out = np.empty(len(p)); out1 = np.empty(len(p))
    d_sorted = dates[order]
    bounds = np.flatnonzero(np.r_[True, d_sorted[1:] != d_sorted[:-1], True])
    for i in range(len(bounds) - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        out[idx], out1[idx] = cal.predict(keys[idx], p[idx])
        ok = idx[np.isfinite(y[idx])]
        if len(ok):
            cal.update(keys[ok], p[ok], y[ok])
    return out, out1, cal


# ------------------------------------------------------------------ correlation (Gaussian copula)
def team_pair_rho(df, group_cols, z_a, z_b=None, same_ent_col=None):
    """Mean product of normal scores over all distinct within-group pairs (teammates): an estimate of the latent
    correlation when the scores are ~N(0,1). z_b given -> cross-stat pairs (excluding the same player)."""
    if z_b is None:
        g = df.assign(_z=df[z_a], _z2=df[z_a] ** 2).groupby(group_cols).agg(s=("_z", "sum"), q=("_z2", "sum"),
                                                                            n=("_z", "size"))
        num = ((g.s ** 2 - g.q) / 2).sum()
        den = (g.n * (g.n - 1) / 2).sum()
    else:
        g = df.assign(_za=df[z_a], _zb=df[z_b], _p=df[z_a] * df[z_b]).groupby(group_cols).agg(
            sa=("_za", "sum"), sb=("_zb", "sum"), sp=("_p", "sum"), n=("_za", "size"))
        num = (g.sa * g.sb - g.sp).sum()
        den = (g.n * g.n - g.n).sum()
    return float(num / den) if den > 0 else 0.0


def opp_pair_rho(df, game_col, team_col, z_a, z_b=None):
    """Mean product over cross-team pairs within a game."""
    z_b = z_b or z_a
    g = df.groupby([game_col, team_col]).agg(sa=(z_a, "sum"), sb=(z_b, "sum"), n=(z_a, "size")).reset_index()
    t = g.groupby(game_col)
    rows = []
    for gid, x in t:
        if len(x) != 2:
            continue
        a, b = x.iloc[0], x.iloc[1]
        rows.append((a.sa * b.sb + b.sa * a.sb, 2 * a.n * b.n))
    if not rows:
        return 0.0
    r = np.array(rows)
    return float(r[:, 0].sum() / r[:, 1].sum())


class RhoBook:
    """Latent correlation lookup for leg pairs. Legs carry: kind ('prop'|'ml'|'total'), gid, team, stat, sign
    (+1 = leg wins when its latent is high: X+ props, a team ML, a game over; -1 for unders)."""

    def __init__(self, table: dict | None = None):
        self.t = dict(table or {})

    def get(self, key):
        return self.t.get(key, 0.0)

    def pair(self, A, B):
        if A["gid"] != B["gid"]:
            return 0.0
        ka, kb = A["kind"], B["kind"]
        if ka == "prop" and kb == "prop":
            s1, s2 = sorted([A["stat"], B["stat"]])
            if A.get("ent") == B.get("ent"):
                key = f"pl|{s1}|{s2}"
            else:
                key = f"{'tm' if A['team'] == B['team'] else 'op'}|{s1}|{s2}"
            return A["sign"] * B["sign"] * self.get(key)
        if {ka, kb} == {"ml", "prop"}:
            M, P = (A, B) if ka == "ml" else (B, A)
            s = 1 if P["team"] == M["team"] else -1
            return s * P["sign"] * self.get(f"ml|{P['stat']}")
        if {ka, kb} == {"total", "prop"}:
            T, P = (A, B) if ka == "total" else (B, A)
            return T["sign"] * P["sign"] * self.get(f"tot|{P['stat']}")
        if {ka, kb} == {"ml", "total"}:
            M, T = (A, B) if ka == "ml" else (B, A)
            s = 1 if M.get("team_home", 0) == 1 else -1
            return s * T["sign"] * self.get("mltot")
        if ka == kb == "ml":
            return 1.0 if A["team"] == B["team"] else -1.0
        return 0.0

    def matrix(self, legs):
        k = len(legs)
        R = np.eye(k)
        for i, j in itertools.combinations(range(k), 2):
            R[i, j] = R[j, i] = self.pair(legs[i], legs[j])
        return R


_BASE = np.random.default_rng(12345).standard_normal((40000, 12))


def joint_prob(p, R):
    """P(all legs win) under a Gaussian copula (leg wins <=> latent below Phi^-1(p)); independent blocks multiply."""
    p = np.asarray(p, float)
    k = len(p)
    if k == 0:
        return 1.0
    if np.allclose(R, np.eye(k)):
        return float(np.prod(p))
    z = norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    # split into connected blocks
    adj = np.abs(R) > 1e-9
    seen, out = np.zeros(k, bool), 1.0
    for s in range(k):
        if seen[s]:
            continue
        stack, blk = [s], []
        seen[s] = True
        while stack:
            i = stack.pop(); blk.append(i)
            for j in np.flatnonzero(adj[i] & ~seen):
                seen[j] = True; stack.append(j)
        blk = sorted(blk)
        if len(blk) == 1:
            out *= p[blk[0]]
        elif len(blk) == 2:
            out *= float(bvn_cdf(z[blk[0]], z[blk[1]], R[blk[0], blk[1]]))
        else:
            Ri = R[np.ix_(blk, blk)]
            w, V = np.linalg.eigh(Ri)
            Ri = V @ np.diag(np.clip(w, 1e-4, None)) @ V.T
            d = np.sqrt(np.diag(Ri)); Ri = Ri / np.outer(d, d)
            L = np.linalg.cholesky(Ri)
            Z = _BASE[:, :len(blk)] @ L.T
            out *= float(np.mean(np.all(Z < z[blk], axis=1)))
    return out


# ------------------------------------------------------------------ parlay constructions
def _approx_gain(cur, cand, rho: RhoBook):
    """log-joint increment of adding cand to cur (pairwise BVN correction)."""
    g = math.log(max(cand["p"], 1e-9))
    for L in cur:
        if L["gid"] != cand["gid"]:
            continue
        r = rho.pair(L, cand)
        if r != 0.0:
            j = float(bvn_cdf(L["z"], cand["z"], r))
            g += math.log(max(j, 1e-12)) - math.log(max(L["p"] * cand["p"], 1e-12))
    return g


def _ok_add(cur, cand, max_per_game=None):
    for L in cur:
        if L["ent"] == cand["ent"]:
            return False
    if max_per_game is not None and sum(1 for L in cur if L["gid"] == cand["gid"]) >= max_per_game:
        return False
    return True


def top_n(legs, n, max_per_game=None):
    out = []
    for L in sorted(legs, key=lambda d: -d["p"]):
        if _ok_add(out, L, max_per_game):
            out.append(L)
        if len(out) == n:
            break
    return out if len(out) == n else None


def greedy_from(seed, pool, n, rho, max_per_game=None):
    cur = list(seed)
    while len(cur) < n:
        best, bg = None, -1e18
        for c in pool:
            if not _ok_add(cur, c, max_per_game):
                continue
            g = _approx_gain(cur, c, rho)
            if g > bg:
                best, bg = c, g
        if best is None:
            return None
        cur.append(best)
    return cur


def _pair_matrix(pool, rho: RhoBook):
    """W[i,j] = log(P(i and j) / (p_i p_j)) for same-game pairs (copula), 0 otherwise; vectorized per game."""
    n = len(pool)
    W = np.zeros((n, n))
    gids = np.array([str(L["gid"]) for L in pool])
    z = np.array([L["z"] for L in pool]); p = np.array([L["p"] for L in pool])
    for g in np.unique(gids):
        idx = np.flatnonzero(gids == g)
        if len(idx) < 2:
            continue
        I, J = np.triu_indices(len(idx), 1)
        a, b = idx[I], idx[J]
        r = np.array([rho.pair(pool[i], pool[j]) for i, j in zip(a, b)])
        nz = r != 0
        if nz.any():
            j = bvn_cdf(z[a[nz]], z[b[nz]], r[nz])
            v = np.log(np.maximum(j, 1e-12)) - np.log(np.maximum(p[a[nz]] * p[b[nz]], 1e-12))
            W[a[nz], b[nz]] = v; W[b[nz], a[nz]] = v
    return W


def _greedy_idx(seed, cand_idx, n, logp, W, ent, positive_only=False):
    cur = list(seed)
    used = {ent[i] for i in cur}
    while len(cur) < n:
        c = [i for i in cand_idx if ent[i] not in used and i not in cur]
        if positive_only and cur:                     # same-game phase: only legs the script moves together
            c = [i for i in c if W[cur, i].sum() >= 0]
        if not c:
            return cur
        c = np.array(c)
        gain = logp[c] + (W[np.ix_(cur, c)].sum(0) if cur else 0.0)
        b = int(c[np.argmax(gain)])
        cur.append(b); used.add(ent[b])
    return cur


def sgp_stack(legs, n, rho, top_pool=250):
    """Script-correlated same-game stack: for each game, greedily build the max-joint ticket from that game's legs
    (pairwise copula correction); when a game runs out of eligible legs the ticket continues greedily into the rest
    of the day's legs. Returns the game-anchored ticket with the highest approximate joint."""
    pool = sorted(legs, key=lambda d: -d["p"])[:top_pool]
    if len(pool) < n:
        return None
    W = _pair_matrix(pool, rho)
    logp = np.log(np.maximum([L["p"] for L in pool], 1e-9))
    ent = [L["ent"] for L in pool]
    gids = np.array([str(L["gid"]) for L in pool])
    allidx = list(range(len(pool)))
    best, bv = None, -1e18
    for g in np.unique(gids):
        gi = list(np.flatnonzero(gids == g))
        t = _greedy_idx([], gi, n, logp, W, ent, positive_only=True)
        if len(t) < n:
            t = _greedy_idx(t, allidx, n, logp, W, ent)
        if len(t) < n:
            continue
        v = logp[t].sum() + W[np.ix_(t, t)].sum() / 2
        if v > bv:
            best, bv = t, v
    return [pool[i] for i in best] if best is not None else None


def tickets_for_day(prop_legs, ml_legs, total_legs, rho, sizes=(3, 5, 7), ml_thresholds=(0.85, 0.70)):
    """All constructions for one day. Returns list of dict(construction, n_legs, legs, p_joint, p_indep)."""
    out = []
    ml_sorted = sorted(ml_legs, key=lambda d: -d["p"])
    for n in sizes:
        t = top_n(prop_legs, n)
        if t:
            out.append(("top", n, t))
        t = sgp_stack(prop_legs + ml_legs + total_legs, n, rho)
        if t:
            out.append(("sgp", n, t))
        for thr in ml_thresholds:
            if ml_sorted and ml_sorted[0]["p"] >= thr:
                ml = ml_sorted[0]
                rest = top_n([L for L in prop_legs], n - 1)
                if rest:
                    out.append((f"mixed_ml{int(thr * 100)}", n, [ml] + rest))
    res = []
    for c, n, t in out:
        R = rho.matrix(t)
        pj = joint_prob([L["p"] for L in t], R)
        hits = [L["hit"] for L in t]
        hit = np.nan if any(h != h for h in hits) else float(all(h == 1 for h in hits))
        res.append(dict(construction=c, n_legs=n, p_joint=pj, p_indep=float(np.prod([L["p"] for L in t])), hit=hit,
                        legs=" | ".join(f"{L['label']} ({L['p']:.2f})" for L in t)))
    # most likely ticket of the day per size (max predicted joint among the constructions)
    for n in sizes:
        cand = [r for r in res if r["n_legs"] == n]
        if cand:
            b = max(cand, key=lambda r: r["p_joint"])
            res.append({**b, "construction": "best_of_day"})
    return res


def parlay_summary(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, n), x in P.dropna(subset=["hit"]).groupby(["construction", "n_legs"]):
        se = math.sqrt(max(x.p_joint.mean() * (1 - x.p_joint.mean()), 1e-9) / len(x))
        rows.append(dict(construction=c, legs=n, n_days=len(x), pred_joint=x.p_joint.mean(),
                         pred_indep=x.p_indep.mean(), realized=x.hit.mean(),
                         z=(x.hit.mean() - x.p_joint.mean()) / se, typical_price=fair_american(x.p_joint.median())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ reports
def tier_table(df, p_col, y_col, base_col=None, base2_col=None, tiers=TIERS, group=None):
    rows = []
    groups = [(None, df)] if group is None else list(df.groupby(group))
    for gk, x in groups:
        for i, t in enumerate(tiers):
            hi = tiers[i + 1] if i + 1 < len(tiers) else 1.01
            for kind, m in (("band", (x[p_col] >= t) & (x[p_col] < hi)), ("cum", x[p_col] >= t)):
                s = x[m]
                if kind == "band" and i == len(tiers) - 1:
                    continue
                r = dict(tier=f"{int(t*100)}-{int(hi*100)}%" if kind == "band" else f">={int(t*100)}%",
                         n=len(s), pred=s[p_col].mean() if len(s) else np.nan,
                         hit=s[y_col].mean() if len(s) else np.nan)
                if base_col:
                    r["base"] = s[base_col].mean() if len(s) else np.nan
                    r["lift_vs_base"] = r["hit"] - r["base"] if len(s) else np.nan
                if base2_col:
                    r["base_player"] = s[base2_col].mean() if len(s) else np.nan
                if group is not None:
                    r = {"group": gk, **r}
                rows.append(r)
    return pd.DataFrame(rows)


def md(df, floatfmt=".3f"):
    if df is None or len(df) == 0:
        return "_(none)_"
    return df.to_markdown(index=False, floatfmt=floatfmt)


def small_flag(n, k=100):
    return " (SMALL SAMPLE)" if n < k else ""


# ------------------------------------------------------------------ standardized leg frames -> daily tickets
LEG_COLS = ["date", "season", "gid", "team", "team_home", "ent", "stat", "kind", "sign", "label", "p", "hit"]


def to_leg_dicts(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    d = df[LEG_COLS].to_dict("records")
    for L in d:
        L["z"] = float(norm.ppf(np.clip(L["p"], 1e-6, 1 - 1e-6)))
    return d


def run_parlays(props: pd.DataFrame, ml: pd.DataFrame, tot: pd.DataFrame, rho_for_season, per_ent=2,
                sizes=(3, 5, 7), log=print) -> pd.DataFrame:
    """Daily tickets over every date in `props` (already restricted to eligible legs). per_ent: keep each
    entity's top legs only (by p) to bound the search."""
    props = props.sort_values("p", ascending=False).groupby(["date", "ent"], sort=False).head(per_ent)
    gp = dict(tuple(props.groupby("date")))
    gm = dict(tuple(ml.groupby("date"))) if ml is not None and len(ml) else {}
    gt = dict(tuple(tot.groupby("date"))) if tot is not None and len(tot) else {}
    out = []
    for i, (d, x) in enumerate(sorted(gp.items())):
        rho = rho_for_season(int(x.season.iloc[0]))
        res = tickets_for_day(to_leg_dicts(x), to_leg_dicts(gm.get(d)), to_leg_dicts(gt.get(d)), rho, sizes=sizes)
        for r in res:
            out.append(dict(date=d, season=int(x.season.iloc[0]), **r))
        if i % 300 == 0:
            log(f"    parlays: {i}/{len(gp)} days")
    return pd.DataFrame(out)


def pair_moments(df, season, game, team, zcols, prefix_ent=True):
    """Per-season numerators/denominators of latent-correlation estimates (mean products of normal scores):
    tm|a|b teammates, op|a|b opponents, pl|a|b same player (a != b). Returns {key: DataFrame(season,num,den)}."""
    out = {}
    g = df.groupby([season, game, team])
    S = g[zcols].sum(); n = g.size().rename("n")
    pairs = [(a, b) for i, a in enumerate(zcols) for b in zcols[i:]]
    for a, b in pairs:
        sa, sb = a.replace("z_", ""), b.replace("z_", "")
        k1, k2 = sorted([sa, sb])
        q = df.assign(_p=df[a] * df[b]).groupby([season, game, team])._p.sum()
        tm_num = (S[a] * S[b] - q).groupby(level=0).sum()
        tm_den = (n * n - n).groupby(level=0).sum()
        out[f"tm|{k1}|{k2}"] = pd.DataFrame(dict(num=tm_num, den=tm_den))
        # opponents: pivot the two teams of each game
        T = pd.DataFrame(dict(sa=S[a], sb=S[b], n=n)).reset_index()
        T["slot"] = T.groupby([season, game]).cumcount()
        T = T[T.slot < 2]
        A = T[T.slot == 0].set_index([season, game]); B = T[T.slot == 1].set_index([season, game])
        j = A.join(B, lsuffix="_1", rsuffix="_2", how="inner")
        num = (j.sa_1 * j.sb_2 + j.sb_1 * j.sa_2).groupby(level=0).sum()
        den = (2 * j.n_1 * j.n_2).groupby(level=0).sum()
        out[f"op|{k1}|{k2}"] = pd.DataFrame(dict(num=num, den=den))
        if a != b:
            pl = df.assign(_p=df[a] * df[b]).groupby(season)._p.agg(["sum", "size"])
            out[f"pl|{k1}|{k2}"] = pd.DataFrame(dict(num=pl["sum"], den=pl["size"]))
    return out


def row_moments(df, season, za, zb, key):
    m = df.assign(_p=df[za] * df[zb]).dropna(subset=["_p"]).groupby(season)._p.agg(["sum", "size"])
    return {key: pd.DataFrame(dict(num=m["sum"], den=m["size"]))}


def rho_tables_by_season(moments: dict, seasons) -> dict:
    """season -> {key: rho} using only seasons strictly before it."""
    out = {}
    for s in seasons:
        t = {}
        for k, m in moments.items():
            x = m[m.index < s]
            if len(x) and x.den.sum() > 0:
                t[k] = float(np.clip(x.num.sum() / x.den.sum(), -0.9, 0.9))
        out[int(s)] = t
    return out
