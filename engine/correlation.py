"""
Leg-correlation engine for same-game parlays, fit on real historical results.

Every bet outcome is modelled as a latent standard normal crossing a threshold (Gaussian copula):
   leg wins  <=>  Z_leg > Phi^-1(1 - p_leg)
so a parlay's true probability is a multivariate-normal orthant probability given the legs' marginal
probabilities p_i and pairwise latent correlations rho_ij. The rho's are estimated from data:

  game legs (all team sports): home-ATS residual vs total residual, by favorite side and spread size
     (e.g. NFL: does the favorite covering go with the OVER?), from closing lines + finals
  NBA player points (2024-26 real prop lines): player OVER vs game OVER, vs own team covering,
     vs a teammate's OVER, vs an opponent's OVER

Validation: correlations are fit on discovery seasons and the predicted joint probabilities are
checked against realized joint hit rates on holdout seasons (results/sgp_correlation_validation.csv).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RHO_FILE = RES / "sgp_correlations.json"


# ------------------------------------------------------------------ copula math
def joint_prob(p: list[float], R: np.ndarray) -> float:
    """P(all legs win) for marginal win probs p and latent correlation matrix R."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    k = len(p)
    if k == 1:
        return float(p[0])
    z = norm.ppf(p)  # leg wins <=> -Z < z  (orientation absorbed into sign of rho)
    R = np.asarray(R, float)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)  # project to nearest PSD
    R = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    return float(multivariate_normal(mean=np.zeros(k), cov=R, allow_singular=True).cdf(z))


def tetrachoric(a: np.ndarray, b: np.ndarray) -> float:
    """Latent correlation from two binary win indicators (Gaussian copula, solved numerically)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    pa, pb, pab = a.mean(), b.mean(), (a & b).mean()
    if min(pa, pb) <= 0 or max(pa, pb) >= 1:
        return 0.0
    lo, hi = -0.99, 0.99
    for _ in range(50):
        mid = (lo + hi) / 2
        if joint_prob([pa, pb], np.array([[1, mid], [mid, 1]])) < pab:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


# ------------------------------------------------------------------ game-leg correlations
SPREAD_BUCKETS = [0, 3.5, 7.5, 14.5, 99]


def _bucket(x):
    return pd.cut(np.abs(x), SPREAD_BUCKETS, labels=["<=3", "3.5-7", "7.5-14", "14.5+"], include_lowest=True)


def game_frame(sport: str) -> pd.DataFrame:
    from backtest import run_team as rt
    g = rt.load(sport)
    ml_sport = sport in ("mlb", "nhl")   # fixed +/-1.5 run/puck lines: define favorite & legs by moneyline
    g = g.dropna(subset=["home_score", "away_score", "total"] + (["home_ml", "away_ml"] if ml_sport else ["spread_home"]))
    g = g.copy()
    margin = g.home_score - g.away_score
    if ml_sport:
        g["spread_home"] = np.where(g.home_ml < g.away_ml, -0.5, 0.5)   # ML favorite, 'cover' = win
    g["fav_home"] = g.spread_home < 0
    fav_margin = np.where(g.fav_home, margin, -margin)
    fav_line = -np.abs(g.spread_home)
    g["fav_cover"] = (fav_margin + fav_line) > 0
    g["fav_win"] = fav_margin > 0
    g["over"] = (g.home_score + g.away_score) > g.total
    push = ((fav_margin + fav_line) == 0) | ((g.home_score + g.away_score) == g.total)
    g = g[~push]
    g["bucket"] = _bucket(g.spread_home)
    return g


def fit_game_rho(sport: str, split: int) -> dict:
    """rho(fav covers, over) and rho(fav wins ML, over) by spread bucket, discovery seasons only."""
    g = game_frame(sport)
    tr = g[g.season < split]
    out = {}
    for b, x in tr.groupby("bucket"):
        if len(x) < 150:
            continue
        out[str(b)] = dict(n=int(len(x)), cover_over=tetrachoric(x.fav_cover, x.over),
                           win_over=tetrachoric(x.fav_win, x.over))
    allx = tr
    out["all"] = dict(n=int(len(allx)), cover_over=tetrachoric(allx.fav_cover, allx.over),
                      win_over=tetrachoric(allx.fav_win, allx.over))
    return out


def validate_game_rho(sport: str, split: int, rho: dict) -> pd.DataFrame:
    """Holdout: predicted P(fav cover & over) etc. vs realized, using market-fair marginals (~realized
    marginal rates per bucket) so the test isolates the correlation, not the side/total edge."""
    g = game_frame(sport)
    ho = g[g.season >= split]
    rows = []
    for b, x in ho.groupby("bucket"):
        r = rho.get(str(b), rho["all"])
        for a_name, a_col in (("fav_cover", "fav_cover"), ("fav_win", "fav_win")):
            key = "cover_over" if a_col == "fav_cover" else "win_over"
            for over_side in (True, False):
                pa, po = x[a_col].mean(), (x.over == over_side).mean()
                s = 1 if over_side else -1
                pred = joint_prob([pa, po], np.array([[1, s * r[key]], [s * r[key], 1]]))
                indep = pa * po
                real = (x[a_col] & (x.over == over_side)).mean()
                rows.append(dict(sport=sport, bucket=str(b), combo=f"{a_name}+{'over' if over_side else 'under'}",
                                 n=len(x), independent=indep, predicted=pred, realized=real))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ NBA player-prop correlations
def fit_nba_prop_rho(split_season=2026) -> dict:
    """From real 2024-25 prop lines (discovery): latent correlations between a player's points OVER and
    game OVER / own-team cover / teammate OVER / opponent OVER. Validated on 2025-26."""
    from backtest.props_nba import proc_path
    p = pd.read_csv(proc_path("nba_props"), low_memory=False)
    p = p[(p.market == "pts") & (p.alt_line == 0) & p.result.isin(["over", "under"])]
    p = p.sort_values("book").drop_duplicates(["date", "player_id"])            # one main line per player-game
    p["y"] = (p.result == "over")
    g = pd.read_csv(ROOT / "data" / "processed" / "nba.csv")
    box = pd.read_csv(proc_path("nba_player_box"), usecols=["game_id", "team", "team_name"], low_memory=False)
    abbr = box.drop_duplicates("team").set_index("team")["team_name"].to_dict()
    p["team_name"] = p.team.map(abbr)
    g = g.dropna(subset=["home_score", "total", "spread_home"])
    g["game_over"] = (g.home_score + g.away_score) > g.total
    g["home_cover"] = (g.home_score - g.away_score + g.spread_home) > 0
    gh = g[["date", "home", "game_over", "home_cover"]].rename(columns={"home": "team_name"})
    gh["team_cover"] = gh.home_cover
    ga = g[["date", "away", "game_over", "home_cover"]].rename(columns={"away": "team_name"})
    ga["team_cover"] = ~ga.home_cover
    G = pd.concat([gh, ga])[["date", "team_name", "game_over", "team_cover"]]
    q = p.merge(G, on=["date", "team_name"], how="inner")
    res = {}
    for label, sub in (("disc", q[q.season < split_season]), ("hold", q[q.season >= split_season])):
        d = dict(n=int(len(sub)), player_over__game_over=tetrachoric(sub.y, sub.game_over),
                 player_over__team_cover=tetrachoric(sub.y, sub.team_cover))
        # teammate / opponent pairs (same game)
        pairs_t, pairs_o = [], []
        for (dt, gid), x in sub.groupby(["date", "game_id"]):
            for (i, a), (j, b) in [(u, v) for u in x.iterrows() for v in x.iterrows() if u[0] < v[0]]:
                (pairs_t if a.team == b.team else pairs_o).append((a.y, b.y))
        if pairs_t:
            t = np.array(pairs_t)
            d["player_over__teammate_over"] = tetrachoric(t[:, 0], t[:, 1])
            d["n_teammate_pairs"] = len(t)
        if pairs_o:
            o = np.array(pairs_o)
            d["player_over__opponent_over"] = tetrachoric(o[:, 0], o[:, 1])
            d["n_opponent_pairs"] = len(o)
        res[label] = d
    return res


SPLITS = {"nfl": 2016, "ncaaf": 2018, "nba": 2019, "ncaam": 2020, "mlb": 2019, "nhl": 2019}

if __name__ == "__main__":
    rho, val = {}, []
    for sp, split in SPLITS.items():
        rho[sp] = fit_game_rho(sp, split)
        v = validate_game_rho(sp, split, rho[sp])
        val.append(v)
        print(f"{sp}: rho(fav cover, over) all={rho[sp]['all']['cover_over']:+.3f}  by bucket: "
              + ", ".join(f"{b}={r['cover_over']:+.3f}" for b, r in rho[sp].items() if b != 'all'))
    rho["nba_props"] = fit_nba_prop_rho()
    print("nba props:", json.dumps(rho["nba_props"], indent=1))
    V = pd.concat(val)
    V["err_pred"] = (V.predicted - V.realized).abs()
    V["err_indep"] = (V.independent - V.realized).abs()
    V.to_csv(RES / "sgp_correlation_validation.csv", index=False)
    # only trust a fitted correlation where it beat independence on the holdout seasons
    for sp, x in V.groupby("sport"):
        better = np.average(x.err_pred, weights=x.n) < np.average(x.err_indep, weights=x.n)
        rho[sp]["use"] = bool(better)
    RHO_FILE.write_text(json.dumps(rho, indent=1))
    w = V.groupby("sport").apply(lambda x: pd.Series({
        "holdout_combos": len(x), "mae_independent": np.average(x.err_indep, weights=x.n),
        "mae_copula": np.average(x.err_pred, weights=x.n)}))
    print("\nHOLDOUT: joint-probability error, independence assumption vs fitted correlation")
    print(w.round(4).to_string())
