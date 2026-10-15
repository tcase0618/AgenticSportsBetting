"""
NBA player-points props: projection model + sharp-consensus EV + parlays.

Discovery season: 2024-25 (season=2025). Holdout: 2025-26 (season=2026).
Every threshold is chosen on discovery and only *reported* on holdout.

Pipeline
  1. projections(): leak-free features from box scores (shifted rolling stats), gradient-boosted
     regression of points, retrained walk-forward (model for season S sees only seasons < S).
     Residual SD is modelled as a function of the projection.
  2. consensus(): de-vig every main-line quote, convert each book's (line, p_over) to an implied
     mean via the normal model, take the leave-one-out median across the OTHER books -> fair
     P(over L) for any line L, i.e. what the sharp market thinks, excluding the book being bet.
  3. strategies: (a) consensus-EV line shopping, (b) model-vs-market edges, (c) blended, each with
     EV thresholds; one bet per player-side per day at the best qualifying quote.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from .core import american_to_decimal, american_to_prob, perf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"
DISC, HOLD = 2025, 2026


def proc_path(name: str) -> Path:
    """Large processed files are stored gzipped in git; accept either form."""
    for ext in (".csv", ".csv.gz"):
        f = PROC / f"{name}{ext}"
        if f.exists():
            return f
    raise FileNotFoundError(name)


# ------------------------------------------------------------------ projections
def box_features(upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """upcoming: rows (date, season, game_id, team, opp, home, player_id, player) for unplayed games;
    they get pre-game features from all prior games and NaN stats."""
    b = pd.read_csv(proc_path("nba_player_box"), low_memory=False,
                    usecols=["date", "season", "season_type", "game_id", "team", "team_name", "opp", "home", "player_id", "player",
                             "starter", "did_not_play", "minutes", "pts", "fga", "fta", "fg3a", "ast", "reb",
                             "team_score", "opp_score"])
    b["date"] = pd.to_datetime(b["date"])
    b = b[b.season >= 2012]
    played = b[(b.did_not_play != True) & (b.minutes.fillna(0) > 0)].copy()  # noqa: E712
    played["upcoming"] = False
    if upcoming is not None and len(upcoming):
        u = upcoming.copy()
        u["date"] = pd.to_datetime(u["date"])
        u["upcoming"] = True
        played = pd.concat([played, u], ignore_index=True)
    played = played.sort_values(["player_id", "date"])
    gp = played.groupby("player_id", sort=False)
    played["usage"] = played.fga + 0.44 * played.fta
    for c in ["pts", "minutes", "usage", "fg3a", "ast", "reb"]:
        for n in (5, 10, 25):
            played[f"{c}_r{n}"] = gp[c].transform(lambda s: s.rolling(n, min_periods=1).mean().shift(1))
    played["pts_sd10"] = gp["pts"].transform(lambda s: s.rolling(10, min_periods=3).std().shift(1))
    played["ppm_r25"] = played.pts_r25 / played.minutes_r25.replace(0, np.nan)
    gs = played.groupby(["player_id", "season"], sort=False)
    played["pts_season"] = gs["pts"].transform(lambda s: s.expanding().mean().shift(1))
    played["min_season"] = gs["minutes"].transform(lambda s: s.expanding().mean().shift(1))
    played["gp_season"] = gs.cumcount()
    played["starter_r5"] = gp["starter"].transform(lambda s: s.astype(float).rolling(5, min_periods=1).mean().shift(1))
    played["prev_date"] = gp["date"].shift(1)
    played["rest"] = (played.date - played.prev_date).dt.days.clip(upper=10)
    # team context: opponent points allowed & game pace proxy (pre-game, team level)
    tb = b if upcoming is None or not len(upcoming) else pd.concat(
        [b, upcoming.drop_duplicates(["game_id", "team"])], ignore_index=True)
    t = tb.drop_duplicates(["game_id", "team"])[["date", "game_id", "team", "opp", "team_score", "opp_score"]].copy()
    t = t.sort_values(["team", "date"])
    gt = t.groupby("team", sort=False)
    t["allowed_r10"] = gt["opp_score"].transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
    t["total_r10"] = gt["team_score"].transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1)) + t["allowed_r10"]
    opp = t[["game_id", "team", "allowed_r10", "total_r10"]].rename(
        columns={"team": "opp", "allowed_r10": "opp_allowed_r10", "total_r10": "opp_total_r10"})
    own = t[["game_id", "team", "total_r10"]].rename(columns={"total_r10": "team_total_r10"})
    played = played.merge(opp, on=["game_id", "opp"], how="left").merge(own, on=["game_id", "team"], how="left")
    return played


FEATS = ["pts_r5", "pts_r10", "pts_r25", "minutes_r5", "minutes_r10", "minutes_r25", "usage_r5", "usage_r10",
         "usage_r25", "fg3a_r10", "ast_r10", "reb_r10", "pts_sd10", "ppm_r25", "pts_season", "min_season",
         "gp_season", "starter_r5", "rest", "home", "opp_allowed_r10", "opp_total_r10", "team_total_r10"]


def projections(F: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward projected points + residual SD for seasons DISC and HOLD."""
    out = []
    for s in (DISC, HOLD):
        tr = F[(F.season < s) & (F.season >= s - 8) & F.pts_r10.notna()]
        te = F[F.season == s].copy()
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0)
        m.fit(tr[FEATS], tr.pts)
        te["proj"] = m.predict(te[FEATS])
        # residual sd as a function of projection, fit on last training season (out-of-fold-ish)
        last = tr[tr.season == s - 1].copy()
        last["proj"] = m.predict(last[FEATS])
        last["bin"] = pd.qcut(last.proj, 20, duplicates="drop")
        sd = last.groupby("bin").apply(lambda x: pd.Series({"mu": x.proj.mean(), "sd": (x.pts - x.proj).std()}))
        a, c = np.polyfit(np.sqrt(sd.mu.clip(lower=0.5)), sd.sd, 1)
        te["proj_sd"] = np.maximum(a * np.sqrt(te.proj.clip(lower=0.5)) + c, 2.0)
        out.append(te)
        print(f"  model season {s}: train n={len(tr)}, sd(mu)={a:.2f}*sqrt(mu)+{c:.2f}")
    return pd.concat(out)


def projections_for(F: pd.DataFrame, rows: pd.DataFrame, season: int) -> pd.DataFrame:
    """Train on the 8 seasons before `season`, project `rows` (e.g. tonight's upcoming rows)."""
    tr = F[(F.season < season) & (F.season >= season - 8) & F.pts_r10.notna() & F.pts.notna()]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                      min_samples_leaf=200, l2_regularization=1.0).fit(tr[FEATS], tr.pts)
    last = tr[tr.season == tr.season.max()].copy()
    last["proj"] = m.predict(last[FEATS])
    last["bin"] = pd.qcut(last.proj, 20, duplicates="drop")
    sd = last.groupby("bin").apply(lambda x: pd.Series({"mu": x.proj.mean(), "sd": (x.pts - x.proj).std()}))
    a, c = np.polyfit(np.sqrt(sd.mu.clip(lower=0.5)), sd.sd, 1)
    out = rows[["player_id"]].copy()
    out["proj"] = m.predict(rows[FEATS])
    out["proj_sd"] = np.maximum(a * np.sqrt(out.proj.clip(lower=0.5)) + c, 2.0)
    return out


# ------------------------------------------------------------------ props + consensus
def sd_from_line(L):
    """Market-side SD guess when no projection exists (fit roughly to NBA points)."""
    return np.maximum(1.55 * np.sqrt(np.maximum(L, 0.5)) + 0.6, 2.0)


def load_props() -> pd.DataFrame:
    p = pd.read_csv(proc_path("nba_props"), low_memory=False)
    p = p[p.season.isin([DISC, HOLD]) & p.result.isin(["over", "under"]) & (p.market == "pts")]
    p = p[p.over_odds.notna() & p.under_odds.notna()]
    p["date"] = pd.to_datetime(p["date"])
    p["pid"] = p.player_id.astype("Int64")
    return p


def consensus(p: pd.DataFrame, sd_col="sd_c") -> pd.DataFrame:
    """Leave-one-book-out market mean (mu_mkt) for every quote."""
    p = p.copy()
    main = p[p.alt_line == 0].copy()
    po, pu = american_to_prob(main.over_odds), american_to_prob(main.under_odds)
    main["p_over_nv"] = po / (po + pu)
    main["mu_i"] = main.line + main[sd_col] * norm.ppf(main.p_over_nv.clip(0.02, 0.98))
    key = ["date", "pid"]
    agg = main.groupby(key + ["book"]).mu_i.mean().rename("mu_b").reset_index()
    # leave-one-out median across other books
    def loo(s):
        vals = s.values
        return pd.Series([np.median(np.delete(vals, i)) if len(vals) >= 3 else np.nan
                          for i in range(len(vals))], index=s.index)
    gk = agg.groupby(key).mu_b
    agg["mu_loo"] = gk.transform(loo)
    agg["n_books"] = gk.transform("size")
    agg["mu_all"] = gk.transform("median")
    p = p.merge(agg[key + ["book", "mu_loo", "n_books", "mu_all"]], on=key + ["book"], how="left")
    # books with only alt lines: use the all-book median
    p["mu_mkt"] = p.mu_loo.fillna(p.mu_all)
    return p


def build_frame() -> pd.DataFrame:
    print("features...")
    F = box_features()
    print("projections...")
    P = projections(F)
    p = load_props()
    P = P[["date", "player_id", "proj", "proj_sd", "pts_r10", "minutes_r10", "rest", "starter_r5", "gp_season"]]
    P = P.rename(columns={"player_id": "pid"})
    P["pid"] = P.pid.astype("Int64")
    p = p.merge(P, on=["date", "pid"], how="left")
    p["sd_c"] = p.proj_sd.fillna(sd_from_line(p.line))
    p = consensus(p)
    p["dec_o"], p["dec_u"] = american_to_decimal(p.over_odds), american_to_decimal(p.under_odds)
    p["p_mkt"] = 1 - norm.cdf((p.line - p.mu_mkt) / p.sd_c)          # fair P(over) from other books
    p["p_mod"] = 1 - norm.cdf((p.line - p.proj) / p.sd_c)            # model P(over)
    p["y"] = (p.result == "over").astype(int)
    return p


# ------------------------------------------------------------------ strategies
def pick_bets(p: pd.DataFrame, prob_col: str, thr: float, max_odds=200) -> pd.DataFrame:
    """For each player-day, best EV quote across books & sides; keep if EV >= thr."""
    q = p[p[prob_col].notna()].copy()
    q["ev_o"] = q[prob_col] * q.dec_o - 1
    q["ev_u"] = (1 - q[prob_col]) * q.dec_u - 1
    o = q.assign(side="over", ev=q.ev_o, dec=q.dec_o, price=q.over_odds, win=(q.y == 1))
    u = q.assign(side="under", ev=q.ev_u, dec=q.dec_u, price=q.under_odds, win=(q.y == 0))
    a = pd.concat([o, u])
    a = a[(a.ev >= thr) & (a.price.abs() <= 1000) & (a.price <= max_odds)]
    a = a.sort_values("ev", ascending=False).drop_duplicates(["date", "pid"])
    a["pnl"] = np.where(a.win, a.dec - 1, -1.0)
    return a


def evaluate(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p = p.copy()
    for w in (0.0, 0.25, 0.5, 1.0):
        col = f"p_blend_{w}"
        p[col] = (1 - w) * p.p_mkt + w * p.p_mod
        for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
            B = pick_bets(p, col, thr)
            for s in (DISC, HOLD):
                x = B[B.season == s]
                r = perf(x.pnl) if len(x) else dict(n=0)
                rows.append(dict(model_weight=w, ev_thr=thr, season=s, n=r.get("n", 0), roi=r.get("roi"),
                                 z=r.get("z"), win=r.get("win_pct"), avg_price=x.price.median() if len(x) else None))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = build_frame()
    p.to_parquet(OUT / "nba_props_frame.parquet") if False else None
    p.to_csv(OUT / "nba_props_frame.csv.gz", index=False)
    # calibration sanity: model & market Brier on main lines
    m = p[p.alt_line == 0]
    for c in ("p_mkt", "p_mod"):
        mm = m[m[c].notna()]
        print(f"{c}: brier={np.mean((mm[c]-mm.y)**2):.4f}  mean_pred={mm[c].mean():.3f} mean_y={mm.y.mean():.3f} n={len(mm)}")
    E = evaluate(p)
    E.to_csv(OUT / "nba_props_strategies.csv", index=False)
    pd.set_option("display.width", 200)
    print(E.pivot_table(index=["model_weight", "ev_thr"], columns="season", values=["n", "roi", "z"]).round(3).to_string())
