"""
Edge hunt: props_market -- calibrated prop probabilities vs REAL prop prices (protocol: engine/edge_hunt/PROTOCOL.md).

Tests
  T1  model probability vs real (alt-ladder / main) price: bet when p * decimal >= 1 + e
        NFL 2026 alt ladders (DK/FD/Bovada/...), NBA points 2024-26 (main + alt), NHL 2026-27, MLB Ks 2026.
        Model = results/props_script/<sport>_legs.parquet (walk-forward, calibrated milestone probabilities);
        NFL/NBA ladders are interpolated per player-game with a probit-linear fit in the threshold k
        (P(X>=k) = Phi(a + b k)); NBA also tested with the GBM projection of backtest/props_nba.py.
  T2  cross-book line shopping vs leave-one-book-out (LOO) consensus
        NBA points (re-verifies 25% model / 75% LOO, EV>=4%), NFL 2026 main + alt, MLB Ks (2 books -> other-book).
        Consensus only from quotes in the SAME scrape (same snapshot) as the quote bet -> no look-ahead.
  T3  DFS pick'em (PrizePicks / Underdog / Pick6) NFL lines at a per-leg price of -122 (5/6-pick flex equivalent).
  T4  UFC method props: model >= market + d / EV >= e, threshold chosen on <=2022 only, lockbox 2023+.

Discovery / lockbox: data that starts after 2022 -> first 60% of dates = discovery, last 40% = lockbox
(UFC: <=2022 / 2023+). One configuration per family is chosen on discovery (max discovery z with n >= nmin)
and evaluated on the lockbox once. Every knob tried is written to results/edge_hunt/props_market_candidates.csv.

    python -m engine.edge_hunt.props_market            # full backtest -> md + candidates csv
    python -c "from engine.edge_hunt.props_market import scan; print(scan())"   # live FanDuel scan
"""
from __future__ import annotations

import math
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
PS = ROOT / "results" / "props_script"
OUT = ROOT / "results" / "edge_hunt"
DFS_PRICE = -122.0          # conservative per-leg equivalent of PrizePicks/Underdog 5/6-pick flex
CANDS: list[dict] = []
LOCK: list[dict] = []


# ------------------------------------------------------------------ helpers
def dec(am):
    am = np.asarray(am, dtype=float)
    return np.where(am > 0, 1 + am / 100, 1 + 100 / np.abs(am))


def ip(am):
    am = np.asarray(am, dtype=float)
    return np.where(am < 0, -am / (-am + 100), 100 / (am + 100))


def nrm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def disc_mask(dates: pd.Series, frac=0.6) -> pd.Series:
    ud = np.sort(pd.to_datetime(dates).unique())
    cut = ud[max(int(math.ceil(len(ud) * frac)) - 1, 0)]
    return pd.to_datetime(dates) <= cut


def perf(pnl) -> dict:
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    if n < 2:
        return dict(n=n, roi=np.nan, z=np.nan, p=np.nan, units=float(pnl.sum()) if n else 0.0, max_dd=np.nan)
    roi, sd = pnl.mean(), pnl.std(ddof=1)
    z = roi / (sd / math.sqrt(n)) if sd > 0 else np.nan
    cum = np.cumsum(pnl)
    return dict(n=n, roi=float(roi), z=float(z), p=float(0.5 * math.erfc(z / math.sqrt(2))) if z == z else np.nan,
                units=float(pnl.sum()), max_dd=float((np.maximum.accumulate(cum) - cum).max()))


def pick(Q: pd.DataFrame, thr: float, max_odds: float = 1e9, min_odds: float = -1e9, key="key") -> pd.DataFrame:
    """Q rows = candidate quotes with p (prob the side wins), dec, res (+1 win / 0 push / -1 loss), odds.
    Keep EV >= thr, one bet per key (best EV)."""
    q = Q[(Q.ev >= thr) & (Q.odds <= max_odds) & (Q.odds >= min_odds)]
    q = q.sort_values("ev", ascending=False).drop_duplicates(key)
    q = q.sort_values("date")
    return q.assign(pnl=np.where(q.res > 0, q.dec - 1, np.where(q.res < 0, -1.0, 0.0)))


def cluster_z(B: pd.DataFrame, cl: str):
    """Cluster-robust z (bets in the same game/date are correlated)."""
    if len(B) < 3 or cl not in B:
        return np.nan, np.nan
    m = B.pnl.mean()
    r = (B.pnl - m).groupby(B[cl]).sum()
    se = math.sqrt(float((r ** 2).sum())) / len(B)
    z = m / se if se > 0 else np.nan
    return float(z), float(0.5 * math.erfc(z / math.sqrt(2))) if z == z else np.nan


def blocks_pos(B: pd.DataFrame, freq="M") -> str:
    if not len(B):
        return "0/0"
    g = B.groupby(pd.to_datetime(B.date).dt.to_period(freq)).pnl.sum()
    return f"{int((g > 0).sum())}/{len(g)}"


def run_family(name: str, test: str, Q: pd.DataFrame, grid: list[dict], mech: str, nmin=50, blk="M",
               extra_lock=None, cl="date"):
    """grid: list of dict(cfg=str, f=callable(Q)->bets). Discovery choose by z; lockbox once."""
    Q = Q.copy()
    Q["disc"] = disc_mask(Q.date) if "disc" not in Q else Q["disc"]
    best = None
    for g in grid:
        B = g["f"](Q[Q.disc])
        r = perf(B.pnl) if len(B) else dict(n=0, roi=np.nan, z=np.nan, p=np.nan)
        CANDS.append(dict(test=test, family=name, config=g["cfg"], disc_n=r["n"], disc_roi=r["roi"], disc_z=r["z"]))
        if r["n"] >= nmin and r["z"] == r["z"] and (best is None or r["z"] > best[1]["z"]):
            best = (g, r)
    if best is None:
        LOCK.append(dict(test=test, family=name, config="(none: no config with n>=%d)" % nmin, disc_n=0, disc_roi=np.nan,
                         lock_n=0, lock_roi=np.nan, lock_z=np.nan, lock_p=np.nan, lock_z_cl=np.nan, lock_p_cl=np.nan,
                         units=np.nan, max_dd=np.nan,
                         blocks_pos="", mech=mech))
        return None
    g, r = best
    BL = g["f"](Q[~Q.disc])
    BA = g["f"](Q)
    rl = perf(BL.pnl) if len(BL) else dict(n=0, roi=np.nan, z=np.nan, p=np.nan, units=np.nan, max_dd=np.nan)
    zc, pc = cluster_z(BL, cl)
    row = dict(test=test, family=name, config=g["cfg"], disc_n=r["n"], disc_roi=r["roi"], lock_n=rl["n"],
               lock_roi=rl["roi"], lock_z=rl["z"], lock_p=rl["p"], lock_z_cl=zc, lock_p_cl=pc, cluster=cl,
               units=rl.get("units"), max_dd=rl.get("max_dd"),
               blocks_pos=blocks_pos(BA, blk), mech=mech)
    if extra_lock:
        row.update(extra_lock(g, Q[~Q.disc]))
    LOCK.append(row)
    for c in CANDS:
        if c["family"] == name and c["config"] == g["cfg"]:
            c["chosen"] = True
            c.update(lock_n=rl["n"], lock_roi=rl["roi"], lock_z=rl["z"])
    return BL


def ev_grid(sides=("over", "under", "both"), thrs=(0.0, 0.03, 0.05, 0.08, 0.12, 0.20), caps=(150, 300, 1000)):
    grid = []
    for s in sides:
        for t in thrs:
            for c in caps:
                def f(Q, s=s, t=t, c=c):
                    q = Q if s == "both" else Q[Q.side == s]
                    return pick(q, t, max_odds=c)
                grid.append(dict(cfg=f"side={s} ev>={t} odds<=+{c}", f=f))
    return grid


def expand(df: pd.DataFrame, p_over: str, two_sided: pd.Series | None = None, settle_over=None) -> pd.DataFrame:
    """quotes -> long candidate frame of (over, under) bets with p, dec, res, ev."""
    o = df.assign(side="over", p=df[p_over], odds=df.over_odds, res=settle_over)
    rows = [o]
    if "under_odds" in df:
        u = df[df.under_odds.notna()]
        u = u.assign(side="under", p=1 - u[p_over], odds=u.under_odds, res=-settle_over[u.index])
        rows.append(u)
    q = pd.concat(rows, ignore_index=True)
    q = q[q.odds.notna() & q.p.notna() & q.res.notna()]
    q["dec"] = dec(q.odds)
    q["ev"] = q.p * q.dec - 1
    return q


# ------------------------------------------------------------------ ladder interpolation from milestone legs
def ladder_fit(L: pd.DataFrame, keys: list[str], kcol="k") -> pd.DataFrame:
    """Per group: probit(P(X>=k)) = a + b k, least squares over the milestone grid."""
    x = L[keys + [kcol, "p"]].copy()
    x["z"] = norm.ppf(x.p.clip(0.003, 0.997))
    g = x.groupby(keys)
    m = g.agg(km=(kcol, "mean"), zm=("z", "mean"), n=(kcol, "size"), kmin=(kcol, "min"), kmax=(kcol, "max"))
    x = x.join(m[["km", "zm"]], on=keys)
    x["dk"], x["dz"] = x[kcol] - x.km, x.z - x.zm
    s = x.assign(kk=x.dk * x.dk, kz=x.dk * x.dz).groupby(keys)[["kk", "kz"]].sum()
    m["b"] = s.kz / s.kk.replace(0, np.nan)
    m["a"] = m.zm - m.b * m.km
    m = m[(m.n >= 2) & (m.b < 0)]
    m["sd"] = -1 / m.b
    m["mu"] = m.a * m.sd          # P(X>=k)=Phi((mu-k)/sd)
    return m.reset_index()


def p_ge(fit_row_a, fit_row_b, t):
    return norm.cdf(fit_row_a + fit_row_b * t)


# ------------------------------------------------------------------ NFL
NFL_MKTS = ["receiving_yards", "receptions", "rushing_yards", "passing_yards"]


def nfl_quotes() -> pd.DataFrame:
    a = pd.read_csv(PROC / "nfl_props_alt.csv.gz")
    m = pd.read_csv(PROC / "nfl_props.csv")
    m = m[m.season == 2026]
    q = pd.concat([a.assign(file="alt"), m.assign(file="main")], ignore_index=True)
    q = q[q.actual.notna() & q.snapshot_utc.notna()]
    q["snap"] = pd.to_datetime(q.snapshot_utc, utc=True)
    q["com"] = pd.to_datetime(q.commence_utc, utc=True)
    q = q[q.snap <= q.com]
    q = q.drop_duplicates(["snap", "game_id", "player_id", "market", "book", "line", "over_odds", "under_odds"])
    q["date"] = pd.to_datetime(q.date)
    # FanDuel integer, over-only lines are milestones "L+" (price matches DK's L-0.5 ladder; verified)
    q["plus"] = (q.line % 1 == 0) & q.under_odds.isna()
    q = q[(q.line % 1 != 0) | q.plus]
    q["thr"] = np.where(q.plus, q.line, np.floor(q.line) + 1)       # over wins iff actual >= thr
    q["res_over"] = np.where(q.actual >= q.thr, 1, -1)
    q["dfs"] = q.source_type == "dfs"
    return q


def nfl_model() -> pd.DataFrame:
    L = pd.read_parquet(PS / "nfl_legs.parquet")
    L = L[(L.season == 2026) & (L.universe == "milestone") & L.stat.isin(NFL_MKTS)]
    f = ladder_fit(L.rename(columns={"stat": "market"}), ["game_id", "player_id", "market"])
    return f


def attach_model(q, f, keys, pad):
    q = q.merge(f[keys + ["a", "b", "sd", "mu", "kmin", "kmax"]], on=keys, how="left")
    span = (q.kmax - q.kmin).fillna(0)
    ok = (q.thr >= q.kmin - pad * span.clip(lower=1)) & (q.thr <= q.kmax + pad * span.clip(lower=1))
    q["p_mod"] = np.where(ok, norm.cdf(q.a + q.b * q.thr), np.nan)
    return q


def loo_consensus(q: pd.DataFrame, grp: list[str], sd_col: str, min_other=2) -> pd.DataFrame:
    """Two-sided quotes -> per-book implied mean; LOO median of OTHER books within the same group (same scrape)."""
    t = q[q.over_odds.notna() & q.under_odds.notna() & ~q.dfs & q[sd_col].notna()].copy()
    po, pu = ip(t.over_odds), ip(t.under_odds)
    t["pnv"] = po / (po + pu)
    t["mu_i"] = (t.thr - 0.5) + t[sd_col] * norm.ppf(t.pnv.clip(0.02, 0.98))
    agg = t.groupby(grp + ["book"]).mu_i.mean().rename("mu_b").reset_index()
    vals = agg.groupby(grp).mu_b.agg(list)
    agg = agg.join(vals.rename("all"), on=grp)
    agg["n_books"] = agg["all"].map(len)
    def loo(r):
        o = list(r["all"]); o.remove(r["mu_b"])
        return np.median(o) if len(o) >= min_other else np.nan
    agg["mu_loo"] = agg.apply(loo, axis=1)
    agg["mu_all"] = agg["all"].map(lambda v: np.median(v) if len(v) >= min_other else np.nan)
    q = q.merge(agg[grp + ["book", "mu_loo", "n_books"]], on=grp + ["book"], how="left")
    q = q.merge(agg.drop_duplicates(grp)[grp + ["mu_all"]], on=grp, how="left")
    q["mu_mkt"] = q.mu_loo.fillna(q.mu_all)       # books with no two-sided quote here (alt-only, DFS) -> all books
    q["p_mkt"] = 1 - norm.cdf((q.thr - 0.5 - q.mu_mkt) / q[sd_col])
    return q


def nfl_tests():
    q = nfl_quotes()
    f = nfl_model()
    q = attach_model(q, f, ["game_id", "player_id", "market"], pad=0.25)
    sd_med = f.groupby("market").sd.median()
    q["sd_c"] = q.sd.fillna(q.market.map(sd_med))
    q = q[q.market.isin(NFL_MKTS)].reset_index(drop=True)
    mech_t1 = ("Alt ladders are priced off the book's own main line with a generic distribution; a player-specific "
               "script model could know the shape (snap share, game script) better than the book's ladder.")
    # ---- T1: model vs sportsbook price, latest pre-kick snapshot per quote
    sb = q[~q.dfs].sort_values("snap").drop_duplicates(["game_id", "player_id", "market", "book", "line", "plus"], keep="last")
    sb = sb[sb.p_mod.notna()].reset_index(drop=True)
    sb["key"] = sb.game_id + sb.player_id + sb.market
    Q = expand(sb, "p_mod", settle_over=sb.res_over)
    run_family("NFL alt+main ladder vs model", "T1", Q, ev_grid(), mech_t1, nmin=40, blk="W", cl="game_id")
    Qa = Q[Q.file == "alt"]
    run_family("NFL alt-ladder only vs model", "T1", Qa, ev_grid(sides=("over", "both")), mech_t1, nmin=40, blk="W",
               cl="game_id")
    # ---- T2: LOO consensus within the same scrape
    grp = ["snap", "game_id", "player_id", "market"]
    c = loo_consensus(q, grp, "sd_c")
    c = c[c.p_mkt.notna() & ~c.dfs].reset_index(drop=True)
    c["key"] = c.game_id + c.player_id + c.market
    mech_t2 = ("Books copy each other's main line; when one book's quote (often an auto-generated alt rung) lags the "
               "others after news, the other books' consensus is the sharper price and the laggard is beatable.")
    grid = []
    for w in (0.0, 0.25, 0.5):
        for t in (0.02, 0.04, 0.06, 0.08, 0.12):
            for scope in ("main", "alt", "all"):
                def fn(Qx, w=w, t=t, scope=scope):
                    x = Qx if scope == "all" else Qx[Qx.file == scope]
                    x = x.assign(p=np.where(x.side == "over", 1, -1) * ((1 - w) * x.pm_side_over + w * x.pmod_side_over)
                                 + np.where(x.side == "over", 0, 1))
                    x = x[x.p.notna()]
                    x = x.assign(ev=x.p * x.dec - 1)
                    # earliest scrape where it qualifies, best quote in that scrape
                    x = x[(x.ev >= t) & (x.odds <= 300)].sort_values(["snap", "ev"], ascending=[True, False])
                    x = x.drop_duplicates("key").sort_values("date")
                    return x.assign(pnl=np.where(x.res > 0, x.dec - 1, -1.0))
                grid.append(dict(cfg=f"w_model={w} ev>={t} scope={scope}", f=fn))
    c["p_mod_f"] = c.p_mod.fillna(c.p_mkt)
    Qc = expand(c, "p_mkt", settle_over=c.res_over)
    Qc["pm_side_over"] = np.where(Qc.side == "over", Qc.p, 1 - Qc.p)
    Qc["pmod_side_over"] = Qc.p_mod_f
    run_family("NFL LOO consensus line-shop (main+alt)", "T2", Qc, grid, mech_t2, nmin=40, blk="W", cl="game_id")
    # ---- T3: DFS pick'em lines at -122/leg vs consensus / model
    d = loo_consensus(q, grp, "sd_c")
    d = d[d.dfs].reset_index(drop=True)
    d["key"] = d.game_id + d.player_id + d.market
    d["over_odds"], d["under_odds"] = DFS_PRICE, DFS_PRICE
    d["p_mod_f"] = d.p_mod
    rows = []
    for src in ("p_mkt", "p_mod", "blend"):
        dd = d.copy()
        dd["pp"] = dd.p_mkt if src == "p_mkt" else dd.p_mod if src == "p_mod" else 0.75 * dd.p_mkt + 0.25 * dd.p_mod
        dd = dd[dd.pp.notna()].reset_index(drop=True)
        x = expand(dd, "pp", settle_over=dd.res_over).assign(src=src)
        rows.append(x)
    Qd = pd.concat(rows, ignore_index=True)
    grid = []
    for src in ("p_mkt", "p_mod", "blend"):
        for t in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12):
            def fd(Qx, src=src, t=t):
                x = Qx[Qx.src == src].sort_values(["snap", "ev"], ascending=[True, False])
                x = x[x.ev >= t].drop_duplicates("key").sort_values("date")
                return x.assign(pnl=np.where(x.res > 0, x.dec - 1, -1.0))
            grid.append(dict(cfg=f"prob={src} ev>={t} @-122", f=fd))
    def lock119(g, QL):
        B = g["f"](QL)
        if not len(B):
            return {}
        d119 = dec(-119.0)
        pn = np.where(B.res > 0, d119 - 1, -1.0)
        return dict(note=f"at -119/leg: ROI {pn.mean():+.3f}; hit {(B.res > 0).mean():.3f} (n={len(B)})")
    mech_t3 = ("DFS pick'em posts one line per player and moves it slowly; when sportsbooks move after news the DFS "
               "line is stale and the side the books moved toward wins more than the ~55% flex breakeven.")
    run_family("NFL DFS pick'em vs book consensus/model", "T3", Qd, grid, mech_t3, nmin=40, blk="W", extra_lock=lock119,
               cl="game_id")
    return q


# ------------------------------------------------------------------ NBA
def nba_tests():
    fr = pd.read_csv(ROOT / "results" / "nba_props_frame.csv.gz", low_memory=False)
    fr["date"] = pd.to_datetime(fr.date)
    fr = fr[fr.result.isin(["over", "under"])].reset_index(drop=True)
    fr["thr"] = np.floor(fr.line) + 1
    fr["res_over"] = np.where(fr.result == "over", 1, -1)
    fr["key"] = fr.date.dt.strftime("%Y%m%d") + "_" + fr.pid.astype(str)
    fr["file"] = np.where(fr.alt_line == 1, "alt", "main")
    # legs-model ladder (pts 10/15/20/25 milestone probabilities, walk-forward + calibrated)
    L = pd.read_parquet(PS / "nba_legs.parquet", columns=["date", "player_id", "stat", "k", "p", "universe"])
    L = L[(L.universe == "milestone") & (L.stat == "pts") & (L.date >= "2024-10-01")]
    f = ladder_fit(L, ["date", "player_id"])
    f = f.rename(columns={"player_id": "pid"})
    fr = fr.merge(f[["date", "pid", "a", "b", "kmin", "kmax"]], on=["date", "pid"], how="left")
    ok = (fr.thr >= fr.kmin - 3) & (fr.thr <= fr.kmax + 3)
    fr["p_legs"] = np.where(ok, norm.cdf(fr.a + fr.b * fr.thr), np.nan)
    mech_t1 = ("Model sees usage redistribution / minutes context the line may not fully price; alt rungs are set off "
               "the main line with a generic spread.")
    for col, nm in (("p_legs", "NBA pts lines vs props_script legs model"), ("p_mod", "NBA pts lines vs GBM projection")):
        x = fr[fr[col].notna()].reset_index(drop=True)
        Q = expand(x, col, settle_over=x.res_over)
        run_family(nm, "T1", Q, ev_grid(), mech_t1, nmin=100, blk="M")
    # ---- T2 LOO (prior: 25% model / 75% LOO, EV>=4%)
    x = fr.reset_index(drop=True)
    Q = expand(x, "p_mkt", settle_over=x.res_over)
    Q["pm_over"] = np.where(Q.side == "over", Q.p, 1 - Q.p)
    grid = []
    for w in (0.0, 0.25, 0.5, 1.0):
        for t in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
            for scope in ("all", "main", "alt"):
                def fn(Qx, w=w, t=t, scope=scope):
                    q = Qx if scope == "all" else Qx[Qx.file == scope]
                    po = (1 - w) * q.pm_over + w * q.p_mod
                    q = q.assign(p=np.where(q.side == "over", po, 1 - po))
                    q = q.assign(ev=q.p * q.dec - 1)
                    return pick(q, t, max_odds=200)
                grid.append(dict(cfg=f"w_model={w} ev>={t} scope={scope} odds<=+200", f=fn))
    mech_t2 = ("Soft books (Bovada/BetRivers alt rungs, slow movers) hang lines that lag the other books' consensus "
               "~12h before tip; the consensus of the other books is the sharper estimate.")
    run_family("NBA pts LOO consensus line-shop", "T2", Q, grid, mech_t2, nmin=200, blk="M")
    # prior exact rule re-verified on protocol split (not chosen here -> reported as a fixed pre-registered rule)
    po = 0.75 * Q.pm_over + 0.25 * Q.p_mod
    Qp = Q.assign(p=np.where(Q.side == "over", po, 1 - po))
    Qp = Qp.assign(ev=Qp.p * Qp.dec - 1)
    Qp["disc"] = disc_mask(Qp.date)
    return fr, Qp


def nba_prior_rule(Qp):
    out = {}
    for nm, m in (("disc(first 60% dates)", Qp.disc), ("lockbox(last 40%)", ~Qp.disc),
                  ("2024-25", Qp.season == 2025), ("2025-26", Qp.season == 2026)):
        for scope in ("all", "main", "alt"):
            q = Qp[m] if scope == "all" else Qp[m & (Qp.file == scope)]
            B = pick(q, 0.04, max_odds=200)
            out[(nm, scope)] = dict(**perf(B.pnl), months_pos=blocks_pos(B))
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ NHL / MLB (direct threshold lookup)
def nhl_tests():
    h = pd.read_csv(PROC / "nhl_props.csv")
    h = h[h.result.isin(["over", "under"]) & h.market.isin(["shots_on_goal", "points"])].copy()
    h["stat"] = h.market.map({"shots_on_goal": "sog", "points": "points"})
    h["thr"] = (np.floor(h.line) + 1).astype(int)
    L = pd.read_parquet(PS / "nhl_legs.parquet")
    L = L[L.season >= 2027][["game_id", "player_id", "stat", "threshold", "p"]]
    L["stat"] = L.stat.astype(str)
    h = h.merge(L.rename(columns={"threshold": "thr", "p": "p_mod"}), on=["game_id", "player_id", "stat", "thr"], how="left")
    h = h[h.p_mod.notna()].reset_index(drop=True)
    h["date"] = pd.to_datetime(h.date)
    h["res_over"] = np.where(h.result == "over", 1, -1)
    h["key"] = h.game_id.astype(str) + h.player_id.astype(str) + h.stat
    Q = expand(h, "p_mod", settle_over=h.res_over)
    run_family("NHL SOG/points vs model (best price across books)", "T1", Q, ev_grid(), "Model TOI/PP role vs books' "
               "generic SOG ladders.", nmin=30, blk="W", cl="game_id")


def mlb_tests():
    m = pd.read_csv(PROC / "mlb_props.csv")
    m = m[m.result.isin(["over", "under"])].copy()
    m["date"] = pd.to_datetime(m.date)
    m["thr"] = (np.floor(m.line) + 1).astype(int)
    L = pd.read_parquet(PS / "mlb_legs.parquet")
    L = L[L.season == 2026][["date", "player", "threshold", "p"]]
    L["nk"] = L.player.map(lambda s: nrm(" ".join(reversed(s.split(", ")))))
    f = ladder_fit(L.rename(columns={"threshold": "k"}), ["date", "nk"])
    m["nk"] = m.player.map(nrm)
    m = m.merge(f[["date", "nk", "a", "b"]], on=["date", "nk"], how="left")
    m = m.merge(L.rename(columns={"threshold": "thr", "p": "p_direct"})[["date", "nk", "thr", "p_direct"]],
                on=["date", "nk", "thr"], how="left")
    m["p_mod"] = m.p_direct.fillna(pd.Series(norm.cdf(m.a + m.b * m.thr), index=m.index).where(m.thr.between(2, 9)))
    m["res_over"] = np.where(m.result == "over", 1, -1)
    m["key"] = m.date.astype(str) + m.nk
    m["dfs"] = False
    mm = m[m.p_mod.notna()].reset_index(drop=True)
    print(f"  MLB matched {len(mm)}/{len(m)} quotes to model")
    Q = expand(mm, "p_mod", settle_over=mm.res_over)
    run_family("MLB Ks vs model (DK/FD, am+late snaps)", "T1", Q, ev_grid(), "Pitcher-leash model vs books' K lines.",
               nmin=40, blk="W")
    # T2: other-book (only 2 books) -- stale-book within the same snapshot
    m["sd_c"] = np.where(m.b.notna() & (m.b < 0), -1 / m.b, 2.0)
    t = m.copy()
    po, pu = ip(t.over_odds), ip(t.under_odds)
    t["mu_i"] = (t.thr - 0.5) + t.sd_c * norm.ppf((po / (po + pu)).clip(0.02, 0.98))
    oth = t.groupby(["date", "nk", "snapshot", "book"]).mu_i.mean().unstack("book")
    t = t.join(oth, on=["date", "nk", "snapshot"])
    t["mu_o"] = np.where(t.book == "draftkings", t.fanduel, t.draftkings)
    t["p_mkt"] = 1 - norm.cdf((t.thr - 0.5 - t.mu_o) / t.sd_c)
    t = t[t.p_mkt.notna()].reset_index(drop=True)
    Q = expand(t, "p_mkt", settle_over=t.res_over)
    Q["pmo"] = np.where(Q.side == "over", Q.p, 1 - Q.p)
    grid = []
    for w in (0.0, 0.25, 0.5):
        for th in (0.0, 0.02, 0.04, 0.06, 0.08):
            def fn(Qx, w=w, th=th):
                po = (1 - w) * Qx.pmo + w * Qx.p_mod.fillna(Qx.pmo)
                q = Qx.assign(p=np.where(Qx.side == "over", po, 1 - po))
                return pick(q.assign(ev=q.p * q.dec - 1), th, max_odds=300)
            grid.append(dict(cfg=f"w_model={w} ev>={th} (other-book)", f=fn))
    run_family("MLB Ks DK-vs-FD stale book", "T2", Q, grid, "Two books disagreeing within a scrape: the other book's "
               "price as fair value.", nmin=40, blk="W")


# ------------------------------------------------------------------ UFC
def ufc_tests():
    L = pd.read_parquet(PS / "ufc_legs.parquet")
    L = L[L.leg.isin(["A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC"])]
    u = pd.read_parquet(ROOT / "results" / "scripts" / "ufc_fights.parquet")
    oc = {"A_KO": "a_ko_odds", "A_SUB": "a_sub_odds", "A_DEC": "a_dec_odds", "B_KO": "b_ko_odds", "B_SUB": "b_sub_odds",
          "B_DEC": "b_dec_odds"}
    rows = []
    for lg, col in oc.items():
        x = L[L.leg == lg].merge(u[["fight_id", f"mkt_{lg}", col, "mkt_quality_ok"]], on="fight_id", how="left")
        rows.append(x.rename(columns={f"mkt_{lg}": "p_mkt", col: "odds"}))
    x = pd.concat(rows, ignore_index=True)
    x = x[x.p_mkt.notna() & x.odds.notna() & x.mkt_quality_ok.astype(bool)].reset_index(drop=True)
    x["dec"], x["res"], x["side"] = dec(x.odds), np.where(x.hit == 1, 1, -1), "yes"
    x["ev"] = x.p * x.dec - 1
    x["key"] = x.fight_id + x.leg
    x["disc"] = x.season <= 2022
    x["meth"] = x.leg.str[2:]
    grid = []
    for d in (0.0, 0.02, 0.03, 0.05, 0.07, 0.10):
        for meth in ("all", "KO", "SUB", "DEC"):
            for cap in (400, 1000, 1e9):
                def fn(Q, d=d, meth=meth, cap=cap):
                    q = Q[(Q.p >= Q.p_mkt + d) & (Q.odds <= cap)]
                    q = q if meth == "all" else q[q.meth == meth]
                    q = q.sort_values("date")
                    return q.assign(pnl=np.where(q.res > 0, q.dec - 1, -1.0))
                grid.append(dict(cfg=f"model>=mkt+{d} meth={meth} odds<=+{int(min(cap, 99999))}", f=fn))
    for e in (0.0, 0.05, 0.10, 0.20, 0.30):
        def fe(Q, e=e):
            q = Q[Q.ev >= e].sort_values("date")
            return q.assign(pnl=np.where(q.res > 0, q.dec - 1, -1.0))
        grid.append(dict(cfg=f"EV>={e}", f=fe))
    # prior lead: >= +5pts, all methods, any price -> pre-registered fixed rule
    run_family("UFC method props model vs market", "T4", x, grid,
               "Method-of-victory props are thin, high-hold markets priced mostly off the moneyline; fighter finish/"
               "durability tendencies are under-weighted.", nmin=100, blk="Y")
    pr = x[x.p >= x.p_mkt + 0.05]
    pr = pr.assign(pnl=np.where(pr.res > 0, pr.dec - 1, -1.0))
    return x, pr


# ------------------------------------------------------------------ report
def add_fixed(test, name, cfg, Bd, Bl, Ball, blk, cl, mech):
    rd, rl = perf(Bd.pnl), perf(Bl.pnl)
    zc, pc = cluster_z(Bl, cl)
    LOCK.append(dict(test=test, family=name, config=cfg, disc_n=rd["n"], disc_roi=rd["roi"], lock_n=rl["n"],
                     lock_roi=rl["roi"], lock_z=rl["z"], lock_p=rl["p"], lock_z_cl=zc, lock_p_cl=pc, cluster=cl,
                     units=rl["units"], max_dd=rl["max_dd"], blocks_pos=blocks_pos(Ball, blk), mech=mech))
    CANDS.append(dict(test=test, family=name, config=cfg, disc_n=rd["n"], disc_roi=rd["roi"], disc_z=rd["z"], chosen=True,
                      lock_n=rl["n"], lock_roi=rl["roi"], lock_z=rl["z"]))


def bh(pvals, q=0.10):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    o = np.argsort(pv)
    m = len(pv)
    adj = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    r = np.empty(m); r[o] = np.minimum(adj, 1)
    out[ok] = r
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("NFL..."); nfl_tests()
    print("NBA..."); fr, Qp = nba_tests()
    prior = nba_prior_rule(Qp)
    print(prior.round(4).to_string())
    print("NHL..."); nhl_tests()
    print("MLB..."); mlb_tests()
    print("UFC..."); x, pr = ufc_tests()
    # pre-registered prior rules evaluated once on the lockbox (count in the multiple-testing family)
    Bd, Bl = pick(Qp[Qp.disc], 0.04, max_odds=200), pick(Qp[~Qp.disc], 0.04, max_odds=200)
    nba_fam = [r for r in LOCK if r["family"] == "NBA pts LOO consensus line-shop"]
    if not (nba_fam and nba_fam[0]["config"].startswith("w_model=0.25 ev>=0.04 scope=all")):
        add_fixed("T2", "NBA prior rule (fixed, pre-registered)", "w_model=0.25 ev>=0.04 odds<=+200 main+alt", Bd, Bl,
              pd.concat([Bd, Bl]), "M", "date", "same as NBA LOO family")
    add_fixed("T4", "UFC prior lead (fixed, pre-registered)", "model>=mkt+0.05 all methods any price",
              pr[pr.season <= 2022], pr[pr.season >= 2023], pr, "Y", "date", "same as UFC family")
    C = pd.DataFrame(CANDS)
    C.to_csv(OUT / "props_market_candidates.csv", index=False)
    V = pd.DataFrame(LOCK)
    nf = int((V.lock_n > 0).sum())
    V["p_use"] = V[["lock_p", "lock_p_cl"]].max(axis=1)
    V["q_bh"] = bh(V.p_use.values)
    V["bonf_p"] = 0.05 / max(nf, 1)
    def verdict(r):
        if not (r.lock_n and r.lock_n > 0) or not (r.lock_roi > 0):
            return "FALSE"
        sig = (r.p_use < r.bonf_p) or (r.q_bh < 0.10)
        bp = str(r.blocks_pos)
        stab = False
        if "/" in bp and bp.split("/")[1] != "0":
            a_, b_ = map(int, bp.split("/"))
            stab = a_ / b_ >= 0.6
        return "TRUE" if (sig and stab and r.disc_roi > 0 and r.lock_n >= 100) else "FALSE"
    V["verdict"] = V.apply(verdict, axis=1)
    V.to_csv(OUT / "props_market_verdicts.csv", index=False)
    # UFC prior lead fixed rule
    pr_rows = []
    for nm, m in (("<=2022 (disc)", pr.season <= 2022), ("2023+ (lockbox)", pr.season >= 2023), ("2021+ (prior report)", pr.season >= 2021)):
        r = perf(pr[m].pnl)
        pr_rows.append(dict(window=nm, n=r["n"], roi=r["roi"], z=r["z"], p=r["p"], hit=pr[m].hit.mean(),
                            implied=pr[m].p_mkt.mean()))
    PR = pd.DataFrame(pr_rows)
    by_season = pr.groupby("season").pnl.agg(["size", "mean", "sum"]).round(3)
    write_md(V, C, prior, PR, by_season)
    print(V[["test", "family", "config", "disc_n", "disc_roi", "lock_n", "lock_roi", "lock_z", "lock_p", "blocks_pos",
             "verdict"]].round(4).to_string())


def write_md(V, C, prior, PR, by_season):
    f = lambda v, d=3: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.{d}f}" if isinstance(v, float) else str(v))
    L = ["# Edge hunt: props_market — calibrated prop probabilities vs REAL prop prices\n",
         "Protocol: engine/edge_hunt/PROTOCOL.md. Code: engine/edge_hunt/props_market.py. Every knob tried: "
         "props_market_candidates.csv (n_tests = %d). Discovery = first 60%% of dates (UFC <= 2022); lockbox = last 40%% "
         "(UFC 2023+), evaluated ONCE for the single discovery-chosen config per family (max discovery z, n >= nmin). "
         "Significance: one-sided p < 0.05 / %d (families with lockbox bets) or BH q < 0.10, using the LARGER of the "
         "naive and the cluster-robust (by game / date) p; TRUE also needs disc ROI > 0, lockbox n >= 100 and >= 60%% "
         "positive blocks (months NBA, weeks NFL/NHL/MLB, years UFC) over the full sample.\n" % (len(C), int((V.lock_n > 0).sum())),
         "## Verdicts\n",
         "| test | candidate (discovery-chosen config) | disc n / ROI | lockbox n / ROI / z / p | clustered z / p | units / maxDD | blocks + | BH q | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in V.iterrows():
        L.append(f"| {r.test} | {r.family}: `{r.config}` | {int(r.disc_n) if r.disc_n == r.disc_n else ''} / {f(r.disc_roi)} | "
                 f"{int(r.lock_n) if r.lock_n == r.lock_n else 0} / {f(r.lock_roi)} / {f(r.lock_z, 2)} / "
                 f"{'' if not r.lock_p == r.lock_p else f'{r.lock_p:.3f}'} | {f(r.lock_z_cl, 2)} / "
                 f"{'' if not r.lock_p_cl == r.lock_p_cl else f'{r.lock_p_cl:.3f}'} ({r.cluster}) | {f(r.units, 1)} / "
                 f"{'' if not r.max_dd == r.max_dd else f'{r.max_dd:.1f}'} | {r.blocks_pos} | "
                 f"{'' if not r.q_bh == r.q_bh else f'{r.q_bh:.3f}'} | **{r.verdict}** |")
    notes = V[V.get("note").notna()] if "note" in V else V.iloc[:0]
    for _, r in notes.iterrows():
        L.append(f"\n{r.family} sensitivity: {r.note}")
    L += ["\n## Mechanism (one line each)\n"]
    for _, r in V.iterrows():
        L.append(f"* **{r.family}** — {r.mech}")
    L += ["\n## Re-verification of the prior NBA rule (25% GBM model / 75% LOO consensus, EV >= 4%, odds <= +200)\n",
          "Prior claim (results/SUPER_ENGINE.md): +3.9% on 3,059 bets in 2025-26 (season split, discovery = 2024-25). "
          "Under the protocol's date split most of Oct-Dec 2025 moves into discovery. Fixed rule, no re-tuning:\n",
          prior[["n", "roi", "z", "p", "units", "max_dd", "months_pos"]].astype(object).to_markdown(floatfmt=".4f"),
          "\n## UFC prior lead re-derived ('model >= market + 5 pts', all methods, posted price)\n",
          PR.to_markdown(index=False, floatfmt=".4f"), "\nBy season:\n", by_season.to_markdown(),
          ]
    L += ["\n## Bottom line and caveats\n", NOTES]
    (OUT / "props_market.md").write_text("\n".join(L) + "\n")


NOTES = """
**No TRUE edge.** Nothing clears the protocol (multiple-testing significance on the lockbox + discovery-positive +
stable + n >= 100). Closest calls, all to be treated as *forward-test only*, not bets:

* **NFL alt ladders vs props_script model** (+10.8% on 350 lockbox bets, clustered z 1.80, BH q 0.14): the lockbox is
  only 2 NFL weeks (33 games), discovery z was 1.05 and the config is the best of 36 tried. On ALL alt-ladder over
  quotes the model is *under*-confident by ~2-4 pts in every probability bin and the book's vigged price has the
  better Brier (0.1911 vs 0.1939) -- the model is not sharper than the ladder; selection picks up noise.
* **NBA GBM model, unders with EV >= 12%** (+5.5%, n 1,659, clustered p 0.018, BH q 0.12): fails BH/Bonferroni.
* **Prior NBA rule 25% model / 75% LOO consensus, EV >= 4%**: re-verified on the protocol date split it is +3.7%
  (disc, n 2,618) and +3.2% (lockbox, n 2,281, z 1.30-1.41, p 0.08-0.10). The previous "+3.9% on 3,059 in 2025-26"
  is reproduced exactly with the season split, but it is not significant at the protocol's bar, and only 7/14
  months are positive. The discovery search over 84 configs independently picked the same rule. Live it needs
  multi-book quotes (ODDS_API_KEY / The Odds API); FanDuel alone cannot form a leave-one-out consensus.
* **UFC 'model >= market + 5 pts'** (prior untested lead): discovery (<=2022) is NEGATIVE (-7.4%, n 1,132);
  2023+ is +28.6% (n 509, p 0.002) but 2026 alone contributes +54u on 33 bets, only 5/9 seasons are positive, the
  rule was found on a 2021+ report (not blind), and the method-prop overround in the source drops from ~1.22 to
  ~1.14 in 2025-26 (odds source/book mix changed), so the late-sample profit is partly a price-source artefact.
  Re-deriving the threshold on <=2022 only picks `DEC, +3 pts` -> lockbox +6.7% (z 0.98). FALSE.
* **DFS pick'em** (PrizePicks/Underdog/Pick6 at -122/leg): ~2.9k DFS legs have a same-scrape sportsbook
  consensus but only 121 clear the chosen EV bar; disc -0.9% (n 90), lockbox +17% on 31 (noise). The dataset's dfs_*_odds column is a flat -137 placeholder.
* **NHL / MLB** 2026 real lines: model-vs-price and DK-vs-FD stale-book both negative on the lockbox.

Data notes: (1) In nfl_props_alt the FanDuel integer, over-only lines are milestones "L+" (their price equals
DraftKings' L-0.5 rung); the file labels actual == L as "push" -- here they are settled as wins. (2) NFL quotes are
the last pre-kick scrape per book/line; LOO consensus uses only quotes from the SAME scrape (no look-ahead).
(3) NBA quotes are one scrape ~12 h before tip. (4) NHL prices are "best price across books" (not one book).
(5) NFL/NBA ladder probabilities are interpolated from the 4-point milestone grid (probit-linear in k), lines
outside the grid +/-25% (NFL) / +/-3 pts (NBA) are dropped.
"""


# ------------------------------------------------------------------ live scan
def scan(min_ev=0.04) -> pd.DataFrame:
    """Live: FanDuel NBA points O/U (sbapi) + whatever consensus is available.
    With ODDS_API_KEY set, uses nightly.live.live_nba_props (all US books, LOO consensus 75% + GBM 25%, EV>=4%).
    Without it, FanDuel alone cannot form a leave-one-book-out consensus -> returns FanDuel quotes with the GBM
    model probability only (informational; model-only rules FAILED the lockbox, see props_market.md)."""
    import os
    from nightly import providers as PV
    if os.environ.get("ODDS_API_KEY"):
        from nightly.live import live_nba_props
        B = live_nba_props(pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize())
        return B[B.ev >= min_ev] if len(B) else B
    try:
        board = PV.fanduel_board("nba")
        q = PV.fanduel_nba_points(board) if len(board) else pd.DataFrame()
    except Exception as e:  # network / geo block
        print("scan: FanDuel unavailable:", e)
        return pd.DataFrame()
    if q.empty:
        return q
    q["note"] = "FanDuel only: no LOO consensus without ODDS_API_KEY; no verified edge (see props_market.md)"
    return q


if __name__ == "__main__":
    main()
