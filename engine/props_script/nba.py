"""
NBA script-driven player props + 3/5/7-leg parlays (spec: engine/props_script/SPEC.md).

  python -m engine.props_script.nba            # full walk-forward run + report (~15-20 min)
  from engine.props_script.nba import build_tickets; build_tickets("2026-04-10")

Model, all strictly pre-game / walk-forward:
  1. Script  : results/scripts/nba_players.parquet (usage share x scripted team stat x minutes, blowout-aware).
  2. Tendencies (box-score derived, prior games only):
       teammates who missed the team's PREVIOUS game (vacated minutes / stat share), opponent absences,
       player's own return from absence, opponent defence vs position (EWM of script residuals allowed),
       rest / b2b / 3-in-4 for both teams, home/away, recent form (EWM residuals, minutes vs expectation),
       player-specific bias and dispersion (long EWM of residual and residual^2), blowout risk / spread.
  3. Distributional LightGBM P(stat >= k) refit each season on the previous 4 seasons (threshold-expanded).
  4. Online calibrator per stat x threshold x probability bin, learning from every settled leg in date order.
Fit/choose on seasons <= 2020, report 2021+. Real lines (points 2021-22, 2024-25, 2025-26) are never trained on
(only the online calibrator sees them, date by date).
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from engine.props_script import common_hoops_football as H

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "props_script"
STATS = ["pts", "reb", "ast", "fg3m"]
LADDER = {"pts": [10, 15, 20, 25], "reb": [4, 6, 8, 10], "ast": [2, 4, 6, 8], "fg3m": [1, 2, 3]}
REPORT_FROM = 2021
TRAIN_YEARS = 4

GROUPS = {
    "script": ["mu", "sd", "mu_nm", "sd_nm", "share_of_team", "team_stat_script", "stat_c", "playoff", "gnum"],
    "minutes_blowout": ["min_exp", "min_e", "blowout_p", "tier_c", "t_margin", "g_total", "abs_margin",
                        "spread_team", "pace_t", "pace_o"],
    "teammates_out": ["vac_min", "vac_sh", "n_vac", "n_vac_st", "opp_vac_min", "self_missed_prev", "gap_games",
                      "miss_min_t", "miss_st_t", "miss_min_o"],
    "opp_def_vs_pos": ["opp_pos_z", "opp_z", "pos_c"],
    "rest_b2b": ["rest", "b2b", "g3in4", "opp_rest", "opp_b2b"],
    "home_away": ["home"],
    "recent_form": ["f_z3", "f_z10", "minr5", "starter_prev"],
    "player_bias_dispersion": ["f_z40", "zsd", "n_prev"],
}
CATS = ["stat_c", "tier_c", "pos_c"]


# ============================================================ data
def load_box() -> pd.DataFrame:
    cols = ["date", "season", "season_type", "game_id", "team", "opp", "home", "player_id", "position", "starter",
            "did_not_play", "minutes", "pts", "reb", "ast", "fg3m"]
    b = pd.read_csv(ROOT / "data/processed/nba_player_box.csv.gz", usecols=cols, low_memory=False)
    b = b[(b.season >= 2015) & b.player_id.notna()].copy()
    b["date"] = pd.to_datetime(b.date)
    b["played"] = ((b.did_not_play != 1) & (b.minutes.fillna(0) > 0)).astype(int)
    return b


def box_features(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team-game features (rest, absences of rotation players in the PREVIOUS team game) and
    player-game features (own return from absence, prior start rate)."""
    tg = b.groupby(["game_id", "team"]).agg(date=("date", "first"), season=("season", "first"),
                                            opp=("opp", "first"), home=("home", "first")).reset_index()
    tg = tg.sort_values(["team", "date"]).reset_index(drop=True)
    tg["ti"] = tg.groupby("team").cumcount()
    d = tg.groupby("team").date.diff().dt.days
    tg["rest"] = d.clip(upper=7).fillna(7)
    tg["b2b"] = (tg.rest == 1).astype(int)
    d2 = (tg.date - tg.groupby("team").date.shift(2)).dt.days
    tg["g3in4"] = (d2 <= 3).astype(int)
    tg["gnum"] = tg.groupby(["team", "season"]).cumcount()

    pl = b[b.played == 1].merge(tg[["game_id", "team", "ti"]], on=["game_id", "team"])
    tot = pl.groupby(["game_id", "team"])[STATS].transform("sum")
    for s in STATS:
        pl[f"sh_{s}"] = pl[s] / tot[s].replace(0, np.nan)
    pl = pl.sort_values(["player_id", "team", "date"]).reset_index(drop=True)
    g = pl.groupby(["player_id", "team"])
    # post-game EWM state (includes this game) - what a LATER game may use
    pl["min_post"] = g.minutes.transform(lambda s: s.ewm(halflife=5).mean())
    for s in STATS:
        pl[f"shp_{s}"] = g[f"sh_{s}"].transform(lambda x: x.ewm(halflife=8, ignore_na=True).mean())
    pl["nxt"] = g.ti.shift(-1)
    pl["prv"] = g.ti.shift(1)
    seas = tg.set_index(["team", "ti"]).season
    # vacated: player last appeared at s, did not play t-1, t in [s+2, s+6], same season, t <= next appearance
    rows = []
    for dt in range(2, 7):
        x = pl[["team", "ti", "season", "nxt", "min_post"] + [f"shp_{s}" for s in STATS]].copy()
        x["t"] = x.ti + dt
        x = x[(x.nxt.isna()) | (x.nxt >= x.t)]
        rows.append(x)
    V = pd.concat(rows)
    V = V[V.min_post >= 15]
    V["t_season"] = seas.reindex(pd.MultiIndex.from_arrays([V.team, V.t])).to_numpy()
    V = V[V.t_season == V.season]
    V["st"] = (V.min_post >= 26).astype(int)
    agg = V.groupby(["team", "t"]).agg(vac_min=("min_post", "sum"), n_vac=("min_post", "size"),
                                         n_vac_st=("st", "sum"),
                                         **{f"vac_{s}": (f"shp_{s}", "sum") for s in STATS}).reset_index()
    tg = tg.merge(agg.rename(columns={"t": "ti"}), on=["team", "ti"], how="left")
    for c in ["vac_min", "n_vac", "n_vac_st"] + [f"vac_{s}" for s in STATS]:
        tg[c] = tg[c].fillna(0)
    opp = tg[["game_id", "team", "rest", "b2b", "vac_min"]].rename(
        columns={"team": "opp", "rest": "opp_rest", "b2b": "opp_b2b", "vac_min": "opp_vac_min"})
    tg = tg.merge(opp, on=["game_id", "opp"], how="left")

    pl["self_missed_prev"] = ((pl.ti - pl.prv) > 1).astype(int)
    pl["gap_games"] = (pl.ti - pl.prv - 1).clip(0, 10).fillna(0)
    pl["starter_prev"] = g.starter.transform(lambda s: s.shift(1).ewm(halflife=5).mean())
    pf = pl[["game_id", "player_id", "self_missed_prev", "gap_games", "starter_prev", "position"]]
    return tg, pf


def load_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "results/scripts/nba_games.parquet")
    keep = ["game_id", "home", "away", "margin", "total", "sd_margin", "sd_total", "spread_home", "total_line",
            "home_score", "away_score", "pace_home_e", "pace_away_e", "miss_min_home", "miss_min_away",
            "miss_st_home", "miss_st_away", "playoff"]
    return g[keep]


def load_players() -> pd.DataFrame:
    P = pd.read_parquet(ROOT / "results/scripts/nba_players.parquet")
    P = P[P.player_id.notna()].copy()
    P["player_id"] = P.player_id.astype(np.int64)
    return P


def build_frame(P=None, b=None) -> pd.DataFrame:
    t0 = time.time()
    P = load_players() if P is None else P
    b = load_box() if b is None else b
    tg, pf = box_features(b)
    pf = pf.copy()
    pf["player_id"] = pf.player_id.astype(np.int64)
    G = load_games()
    P = P.merge(tg[["game_id", "team", "home", "rest", "b2b", "g3in4", "gnum", "vac_min", "n_vac", "n_vac_st",
                    "opp_rest", "opp_b2b", "opp_vac_min"] + [f"vac_{s}" for s in STATS]],
                on=["game_id", "team"], how="left")
    P = P.merge(pf.drop_duplicates(["game_id", "player_id"]), on=["game_id", "player_id"], how="left")
    P = P.merge(G, on="game_id", how="left")
    # 'home' collides (box flag vs games home team code)
    P = P.rename(columns={"home_x": "home", "home_y": "home_team"})
    P["home"] = (P.team == P.home_team).astype(int)
    sgn = np.where(P.home == 1, 1.0, -1.0)
    P["t_margin"] = sgn * P.margin
    P["g_total"] = P.total
    P["abs_margin"] = P.margin.abs()
    P["spread_team"] = sgn * P.spread_home
    P["pace_t"] = np.where(P.home == 1, P.pace_home_e, P.pace_away_e)
    P["pace_o"] = np.where(P.home == 1, P.pace_away_e, P.pace_home_e)
    P["miss_min_t"] = np.where(P.home == 1, P.miss_min_home, P.miss_min_away)
    P["miss_min_o"] = np.where(P.home == 1, P.miss_min_away, P.miss_min_home)
    P["miss_st_t"] = np.where(P.home == 1, P.miss_st_home, P.miss_st_away)
    act_m = np.where(P.home == 1, P.home_score - P.away_score, P.away_score - P.home_score)
    P["mres"] = (act_m - P.t_margin) / P.sd_margin
    P["tres"] = (P.home_score + P.away_score - P.total) / P.sd_total
    P["vac_sh"] = 0.0
    for s in STATS:
        m = P.stat == s
        P.loc[m, "vac_sh"] = P.loc[m, f"vac_{s}"]
    pos = P.position.fillna("F").str[-1].map({"G": 0, "F": 1, "C": 2}).fillna(1)
    P["pos_c"] = pos.astype(int)
    P["stat_c"] = P.stat.map({s: i for i, s in enumerate(STATS)}).astype(int)
    P["tier_c"] = P.tier.astype("category").cat.codes
    P["playoff"] = P.playoff.fillna(0)

    # ---- residual-based tendencies (prior games only)
    P = P.sort_values(["player_id", "stat", "date", "game_id"]).reset_index(drop=True)
    P["z"] = (P.actual - P.mu) / P.sd.clip(lower=0.25)
    P["z2"] = P.z ** 2
    P["minr"] = (P.minutes / P.min_exp.clip(lower=4)).clip(0, 2.5)
    k = ["player_id", "stat"]
    P["f_z3"] = H.grp_ewm_prev(P, k, "z", 3)
    P["f_z10"] = H.grp_ewm_prev(P, k, "z", 10)
    P["f_z40"] = H.grp_ewm_prev(P, k, "z", 40)
    e2 = H.grp_ewm_prev(P, k, "z2", 25)
    P["zsd"] = np.sqrt(np.clip(e2 - P.f_z40.fillna(0) ** 2, 0.05, None))
    P["n_prev"] = H.grp_count_prev(P, k)
    P["minr5"] = H.grp_ewm_prev(P, k, "minr", 5)
    for j in range(4):
        kk = P.stat.map(lambda s: LADDER[s][j] if j < len(LADDER[s]) else np.nan).to_numpy(float)
        P[f"k{j}"] = kk
        P[f"ind{j}"] = np.where(np.isnan(kk), np.nan, (P.actual >= kk).astype(float))
        P[f"naive{j}"] = H.grp_ewm_prev(P, k, f"ind{j}", 10)
    # opponent defence vs position: residuals allowed by this opponent to this position group, prior dates
    P["opp_pos_z"] = H.date_grp_ewm_prev(P, ["opp", "pos_c", "stat"], "z", 12)
    P["opp_z"] = H.date_grp_ewm_prev(P, ["opp", "stat"], "z", 12)
    print(f"  frame built {len(P):,} rows in {time.time() - t0:.0f}s", flush=True)
    return P


def all_feats(groups=None):
    groups = groups or list(GROUPS)
    f = []
    for g in groups:
        f += GROUPS[g]
    return f


def train_ks(df, rng):
    ks = [df[f"k{j}"].to_numpy(float) for j in range(4)]
    for _ in range(2):
        r = np.round(df.mu.to_numpy() + df.sd.to_numpy() * rng.standard_normal(len(df)))
        ks.append(np.clip(r, 1, None))
    return ks


def y_ge(df, k):
    return (df.actual.to_numpy() >= k).astype(np.int8)


# ============================================================ real lines
def load_lines() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/processed/nba_props.csv.gz", low_memory=False)
    p = p[(p.market == "pts") & (p.alt_line == 0) & p.result.isin(["over", "under"]) & p.player_id.notna()].copy()
    p["player_id"] = p.player_id.astype(np.int64)
    po = 1 / H.american_to_dec(p.over_odds.fillna(-110))
    pu = 1 / H.american_to_dec(p.under_odds.fillna(-110))
    p["both"] = p.over_odds.notna() & p.under_odds.notna()
    p["p_mkt"] = np.where(p.both, po / (po + pu), np.nan)
    p["bal"] = np.where(p.both, (p.p_mkt - 0.5).abs(), 0.25)
    p["mbt"] = p.minutes_before_tip.fillna(1e6)
    p["mbt_min"] = p.groupby(["game_id", "player_id"]).mbt.transform("min")
    p = p[p.mbt <= p.mbt_min + 30]
    p = p.sort_values(["game_id", "player_id", "bal", "mbt"]).drop_duplicates(["game_id", "player_id"])
    p = p[(p.line % 1) != 0]                       # half-point main lines only (no pushes)
    p["dec_over"] = H.american_to_dec(p.over_odds.fillna(-110))
    p["dec_under"] = H.american_to_dec(p.under_odds.fillna(-110))
    return p[["game_id", "player_id", "line", "p_mkt", "dec_over", "dec_under", "book", "both"]]


# ============================================================ walk-forward
def walk_forward(P: pd.DataFrame, feats, seasons, rng_seed=0, n_est=260, max_rows=800_000, lines=None, verbose=True):
    """Season-by-season refit on the previous TRAIN_YEARS seasons. Returns milestone legs + line probabilities."""
    rng = np.random.default_rng(rng_seed)
    legs, lp = [], []
    models = {}
    for S in seasons:
        t0 = time.time()
        tr = P[(P.season < S) & (P.season >= S - TRAIN_YEARS)]
        te = P[P.season == S]
        m = H.CDFModel(feats=feats, cats=[c for c in CATS if c in feats], n_estimators=n_est)
        m.fit(tr, train_ks(tr, rng), y_ge, max_rows=max_rows, seed=int(S))
        models[S] = m
        for j in range(4):
            k = te[f"k{j}"].to_numpy(float)
            ok = ~np.isnan(k)
            x = te[ok]
            legs.append(pd.DataFrame({
                "date": x.date.values, "season": S, "game_id": x.game_id.values, "player_id": x.player_id.values,
                "player": x.player.values, "team": x.team.values, "opp": x.opp.values, "stat": x.stat.values,
                "k": k[ok], "tier": x.tier.values, "pos_c": x.pos_c.values,
                "p_lgb": m.predict(x, k[ok]), "p_param": H.param_cdf(x.mu.values, x.sd.values, k[ok]),
                "naive_pl": x[f"naive{j}"].values, "hit": x[f"ind{j}"].values.astype(np.int8),
                "min_exp": x.min_exp.values}))
        if lines is not None:
            x = te[te.stat == "pts"].merge(lines, on=["game_id", "player_id"])
            if len(x):
                kk = np.floor(x.line.to_numpy()) + 1
                lp.append(pd.DataFrame({"date": x.date.values, "season": S, "game_id": x.game_id.values,
                                        "player_id": x.player_id.values, "player": x.player.values,
                                        "team": x.team.values, "line": x.line.values, "actual": x.actual.values,
                                        "p_over_lgb": m.predict(x, kk),
                                        "p_over_param": H.param_cdf(x.mu.values, x.sd.values, kk),
                                        "p_mkt": x.p_mkt.values, "dec_over": x.dec_over.values,
                                        "dec_under": x.dec_under.values, "mu": x.mu.values}))
        if verbose:
            print(f"  season {S}: train {len(tr):,} rows, {time.time() - t0:.0f}s", flush=True)
    L = pd.concat(legs, ignore_index=True)
    LP = pd.concat(lp, ignore_index=True) if lp else None
    return L, LP, models


def base_rates(L: pd.DataFrame) -> np.ndarray:
    """Naive group base rate: hit rate of the same stat x threshold x role tier in all PRIOR seasons."""
    a = L.groupby(["stat", "k", "tier", "season"]).hit.agg(["sum", "count"]).reset_index()
    a = a.sort_values("season")
    a[["cs", "cn"]] = a.groupby(["stat", "k", "tier"])[["sum", "count"]].cumsum().to_numpy() - a[["sum", "count"]].to_numpy()
    a["base"] = a.cs / a.cn.replace(0, np.nan)
    return L.merge(a[["stat", "k", "tier", "season", "base"]], on=["stat", "k", "tier", "season"], how="left")["base"].to_numpy()


def calibrate_milestones(L: pd.DataFrame, col="p_lgb") -> tuple[np.ndarray, H.OnlineCal]:
    L = L.sort_values("date", kind="stable")
    cal = H.OnlineCal()
    grp = L.stat + "|" + L.k.astype(int).astype(str)
    pc = cal.run(L.date.values, grp.values, L[col].to_numpy(), L.hit.to_numpy())
    out = pd.Series(pc, index=L.index)
    return out, cal


def calibrate_lines(LP: pd.DataFrame, col="p_over_lgb"):
    LP = LP.sort_values("date", kind="stable")
    cal = H.OnlineCal(nbins=18, m=50, span=1.2)   # fine bins around 0.5 (p 0.23-0.77)
    y = (LP.actual > LP.line).astype(int).to_numpy()
    pc = cal.run(LP.date.values, np.array(["over"] * len(LP)), LP[col].to_numpy(), y)
    return pd.Series(pc, index=LP.index), cal


def market_stack(LP: pd.DataFrame) -> np.ndarray:
    """Walk-forward monthly logistic stack of [logit p_model_cal, logit p_mkt] (rows with both odds)."""
    from sklearn.linear_model import LogisticRegression
    out = np.full(len(LP), np.nan)
    ok = LP.p_mkt.notna().to_numpy()
    mon = LP.date.dt.to_period("M")
    y = (LP.actual > LP.line).astype(int).to_numpy()
    X = np.c_[H.logit(LP.p_over_cal), H.logit(LP.p_mkt.fillna(0.5))]
    for mth in sorted(mon[ok].unique()):
        tr = ok & (LP.date < mth.start_time).to_numpy()
        te = ok & (mon == mth).to_numpy()
        if tr.sum() < 1500:
            continue
        clf = LogisticRegression(C=1.0).fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


# ============================================================ correlations by season (prior seasons only)
def season_corrs(P: pd.DataFrame, seasons) -> dict:
    Z = P[(P.min_exp >= 20)][["season", "game_id", "team", "player_id", "stat", "z", "mres", "tres"]].copy()
    Z["z"] = Z.groupby(["season", "stat"]).z.transform(lambda s: (s - s.mean()) / s.std())
    Z = Z.rename(columns={"player_id": "pid"})
    out = {}
    for S in seasons:
        z = Z[(Z.season < S) & (Z.season >= S - 4)]
        out[S] = {"pp": H.pair_corr(z, STATS), "g": H.game_corr(z.dropna(subset=["mres", "tres"]), STATS)}
    return out


# ============================================================ parlays
def game_legs() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "results/predict/nba_preds.parquet")
    g["game_id"] = g.game_id.astype(np.int64)
    ml = pd.DataFrame({"date": g.date, "game_id": g.game_id, "kind": "M", "pid": -1,
                       "team": np.where(g.pick_home, g.home, g.away), "stat": "ml", "sgn": 1.0,
                       "p": g.pick_prob, "hit": g.ml_hit,
                       "dec": H.american_to_dec(np.where(g.pick_home, g.home_ml, g.away_ml)),
                       "label": np.where(g.pick_home, g.home, g.away) + " ML"})
    tot = pd.DataFrame({"date": g.date, "game_id": g.game_id, "kind": "T", "pid": -2, "team": g.home,
                        "stat": "total", "sgn": np.where(g.total_pick_over == 1, 1.0, -1.0),
                        "p": g.total_prob, "hit": g.total_hit, "dec": 1.909,
                        "label": np.where(g.total_pick_over == 1, "o", "u") + g.total_line.astype(str) + " " + g.away + "@" + g.home})
    out = pd.concat([ml, tot])
    out = out[out.p.notna() & out.hit.notna() & np.isfinite(out.dec)]
    return out


def prop_legs(L: pd.DataFrame, pmin=0.55) -> pd.DataFrame:
    x = L[L.p >= pmin]
    q = np.clip(x.naive.fillna(x.base).fillna(x.p).to_numpy() * 1.045, 0.05, 0.985)
    return pd.DataFrame({"date": x.date.values, "game_id": x.game_id.values, "kind": "P", "pid": x.player_id.values,
                         "team": x.team.values, "stat": x.stat.values, "sgn": 1.0, "p": x.p.values,
                         "hit": x.hit.values, "dec": 1 / q,
                         "label": x.player.values + " " + x.k.astype(int).astype(str) + "+ " + x.stat.values})


def real_legs(LP: pd.DataFrame) -> pd.DataFrame:
    over = LP.p_over_cal >= 0.5
    p = np.where(over, LP.p_over_cal, 1 - LP.p_over_cal)
    hit = np.where(over, LP.actual > LP.line, LP.actual < LP.line).astype(int)
    return pd.DataFrame({"date": LP.date.values, "game_id": LP.game_id.values, "kind": "P", "pid": LP.player_id.values,
                         "team": LP.team.values, "stat": "pts", "sgn": np.where(over, 1.0, -1.0), "p": p, "hit": hit,
                         "dec": np.where(over, LP.dec_over, LP.dec_under),
                         "label": LP.player.values + np.where(over, " o", " u") + LP.line.astype(str) + " pts"})


def run_parlays(L, LP, GL, corrs, dates, sizes=(3, 5, 7), constructions=None) -> pd.DataFrame:
    PL = prop_legs(L)
    RL = real_legs(LP) if LP is not None else None
    byP = dict(tuple(PL.groupby("date")))
    byG = dict(tuple(GL.groupby("date")))
    byR = dict(tuple(RL.groupby("date"))) if RL is not None else {}
    rows = []
    for d in dates:
        if d not in byP:
            continue
        props = byP[d]
        season = SEASON_OF.get(d)
        C = corrs[season]
        r = H.build_day(props, byG.get(d), C, sizes=sizes, real=byR.get(d))
        for x in r:
            x["date"] = d
            x["season"] = season
        rows += r
    return pd.DataFrame(rows)


SEASON_OF: dict = {}


# ============================================================ ablation
def ablation(P, lines, corrs, GL, test_from=REPORT_FROM, frac_dates=0.35):
    """Single split: train seasons 2017-2020, test a fixed random 35% of 2021+ dates."""
    rng = np.random.default_rng(7)
    tr = P[(P.season >= test_from - TRAIN_YEARS) & (P.season < test_from)]
    dates = np.array(sorted(P[P.season >= test_from].date.unique()))
    pick = set(rng.choice(dates, int(len(dates) * frac_dates), replace=False))
    te = P[P.date.isin(list(pick))]
    full = list(GROUPS)
    configs = [("full model", full)] + [(f"drop {g}", [x for x in full if x != g]) for g in full if g != "script"] \
        + [("script features only (LGBM)", ["script"])]
    res = []
    for name, groups in configs + [("script parametric (no LGBM)", None)]:
        t0 = time.time()
        if groups is not None:
            m = H.CDFModel(feats=all_feats(groups), cats=[c for c in CATS if c in all_feats(groups)], n_estimators=200)
            m.fit(tr, train_ks(tr, np.random.default_rng(1)), y_ge, max_rows=500_000, seed=1)
        legs = []
        for j in range(4):
            k = te[f"k{j}"].to_numpy(float)
            ok = ~np.isnan(k)
            x = te[ok]
            p = m.predict(x, k[ok]) if groups is not None else H.param_cdf(x.mu.values, x.sd.values, k[ok])
            legs.append(pd.DataFrame({"date": x.date.values, "season": x.season.values, "game_id": x.game_id.values,
                                      "player_id": x.player_id.values, "player": x.player.values,
                                      "team": x.team.values, "stat": x.stat.values, "k": k[ok], "p_lgb": p,
                                      "hit": x[f"ind{j}"].values.astype(np.int8), "naive_pl": x[f"naive{j}"].values}))
        Lx = pd.concat(legs, ignore_index=True).sort_values("date").reset_index(drop=True)
        pc, _ = calibrate_milestones(Lx)
        Lx["p"] = pc
        variants = [(name, "p")] + ([("full model, no online calibrator", "p_lgb")] if name == "full model" else [])
        # real lines on the same dates
        xl = te[te.stat == "pts"].merge(lines, on=["game_id", "player_id"])
        kk = np.floor(xl.line.to_numpy()) + 1
        po = m.predict(xl, kk) if groups is not None else H.param_cdf(xl.mu.values, xl.sd.values, kk)
        yl = (xl.actual > xl.line).to_numpy()
        xl = xl.assign(p_over_lgb=po).sort_values("date")
        LPc, _ = calibrate_lines(xl.assign(actual=xl.actual))
        for vn, col in variants:
            pp = Lx[col].to_numpy()
            y = Lx.hit.to_numpy()
            sel8, sel9 = pp >= 0.8, pp >= 0.9
            nv = Lx.naive_pl.to_numpy()
            # daily top-5 (one leg per player), independent product
            t5 = Lx.assign(pp=pp).sort_values("pp", ascending=False).drop_duplicates(["date", "player_id"])
            t5 = t5.groupby("date").head(5).groupby("date").agg(n=("hit", "size"), h=("hit", "prod"))
            t5 = t5[t5.n == 5]
            plc = LPc.loc[xl.index].to_numpy() if vn != "full model, no online calibrator" else xl.p_over_lgb.to_numpy()
            ylc = (xl.actual > xl.line).to_numpy()
            side = np.maximum(plc, 1 - plc)
            hitl = np.where(plc >= 0.5, ylc, ~ylc)
            res.append({"config": vn, "ms_logloss": H.logloss(pp, y), "ms_ece": H.ece(pp, y),
                        "n>=0.8": int(sel8.sum()), "hit>=0.8": y[sel8].mean(),
                        "lift_vs_player_naive>=0.8": y[sel8].mean() - np.nanmean(nv[sel8]),
                        "n>=0.9": int(sel9.sum()), "hit>=0.9": y[sel9].mean(),
                        "top5_daily_hit": t5.h.mean(), "line_hit_all": hitl.mean(),
                        "line_n>=0.58": int((side >= 0.58).sum()), "line_hit>=0.58": hitl[side >= 0.58].mean()})
        print(f"  ablation {name}: {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(res)


# ============================================================ main
def main(do_ablation=True):
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P = build_frame()
    lines = load_lines()
    seasons = list(range(2017, 2027))
    feats = all_feats()
    L, LP, models = walk_forward(P, feats, seasons, lines=lines)
    print(f"walk-forward done {time.time() - T0:.0f}s", flush=True)
    L = L.sort_values(["date", "game_id"]).reset_index(drop=True)
    pc, cal_ms = calibrate_milestones(L)
    L["p"] = pc
    L["base"] = base_rates(L)
    # naive player rate shrunk to the group base (5 pseudo-games)
    L["naive"] = L.naive_pl.fillna(L.base)
    LP = LP.sort_values("date").reset_index(drop=True)
    pcl, cal_ln = calibrate_lines(LP)
    LP["p_over_cal"] = pcl
    LP["p_stack"] = market_stack(LP)
    corrs = season_corrs(P, seasons)
    GL = game_legs()
    for d, s in L[["date", "season"]].drop_duplicates().itertuples(index=False):
        SEASON_OF[d] = int(s)
    rep_dates = sorted(L[L.season >= REPORT_FROM].date.unique())
    print(f"calibration/corr done {time.time() - T0:.0f}s", flush=True)
    T = run_parlays(L[L.season >= REPORT_FROM], LP, GL, corrs, rep_dates)
    print(f"parlays done {time.time() - T0:.0f}s", flush=True)
    ABL = ablation(P, lines, corrs, GL) if do_ablation else None
    print(f"ablation done {time.time() - T0:.0f}s", flush=True)

    # ---- save
    keep = L[L.season >= REPORT_FROM]
    legs_out = keep[["date", "season", "game_id", "player_id", "player", "team", "stat", "k", "p", "p_lgb",
                     "naive", "base", "hit"]].copy()
    legs_out["side"] = "over(k+)"
    legs_out["universe"] = "milestone"
    for c in ["p", "p_lgb", "naive", "base", "k"]:
        legs_out[c] = legs_out[c].astype(np.float32)
    lo = LP.assign(universe="real_line", side=np.where(LP.p_over_cal >= 0.5, "over", "under"),
                   p=np.maximum(LP.p_over_cal, 1 - LP.p_over_cal),
                   hit=np.where(LP.p_over_cal >= 0.5, LP.actual > LP.line, LP.actual < LP.line).astype(np.int8),
                   stat="pts", k=LP.line)
    lo = lo[["date", "season", "game_id", "player_id", "player", "team", "stat", "k", "p", "p_over_lgb", "p_over_cal",
             "p_mkt", "p_stack", "hit", "side", "universe"]]
    allg = pd.concat([legs_out, lo], ignore_index=True)
    for c in ["player", "team", "stat", "side", "universe"]:
        allg[c] = allg[c].astype("category")
    allg["season"] = allg.season.astype(np.int16)
    allg.to_parquet(OUT / "nba_legs.parquet", index=False, compression="zstd")
    T.to_csv(OUT / "nba_parlays.csv", index=False)
    with open(OUT / "nba_state.pkl", "wb") as f:
        pickle.dump({"model": models[max(models)], "feats": feats, "cal_ms": cal_ms, "cal_ln": cal_ln,
                     "corr": corrs[max(corrs)]}, f)
    TT = H.tendency_table(P[P.season >= REPORT_FROM], TENDENCY_SPECS, STATS)
    write_report(L, LP, T, ABL, time.time() - T0, TT, models)
    print(f"total {time.time() - T0:.0f}s")


TENDENCY_SPECS = [
    ("teammates out prev game: vacated stat share", "vac_sh", [-0.01, 0.0, 0.1, 0.25, 0.5, 5]),
    ("teammates out prev game: vacated minutes", "vac_min", [-1, 0, 25, 50, 100, 500]),
    ("player returning from absence", "self_missed_prev", [-0.5, 0.5, 1.5]),
    ("opponent rotation minutes out", "opp_vac_min", [-1, 0, 30, 60, 500]),
    ("team on back-to-back", "b2b", [-0.5, 0.5, 1.5]),
    ("opponent on back-to-back", "opp_b2b", [-0.5, 0.5, 1.5]),
    ("home", "home", [-0.5, 0.5, 1.5]),
    ("opp defence vs position (prior z allowed)", "opp_pos_z", [-9, -0.15, -0.05, 0.05, 0.15, 9]),
    ("recent form (EWM z, hl 10)", "f_z10", [-9, -0.3, -0.1, 0.1, 0.3, 30]),
    ("player long-run bias (EWM z, hl 40)", "f_z40", [-9, -0.2, -0.05, 0.05, 0.2, 30]),
    ("minutes vs expectation, last ~5", "minr5", [0, 0.85, 0.95, 1.05, 1.15, 3]),
    ("blowout probability", "blowout_p", [0, 0.28, 0.33, 0.4, 1]),
    ("script team margin", "t_margin", [-40, -8, -3, 3, 8, 40]),
]


def write_report(L, LP, T, ABL, secs, TT=None, models=None):
    R = L[L.season >= REPORT_FROM]
    s = ["# NBA script-driven props + 3/5/7-leg parlays (walk-forward, report 2021+)\n",
         f"Code: `engine/props_script/nba.py` (+ `common_hoops_football.py`). Runtime {secs / 60:.1f} min. "
         "Distributional LightGBM P(stat >= k) refit each season on the prior 4 seasons from the pre-game script "
         "plus tendencies; online per stat x threshold x probability-bin calibrator updated date by date. "
         "Seasons 2017-2020 = fit/choose, 2021-2026 = reported. NBA season label = year it ends.\n"]
    # (a) real lines
    s.append("## (a) Real points lines: hit rate by model confidence (side = higher calibrated prob)\n")
    s.append("One main line per player-game (latest snapshot, line closest to even odds, half-points only). "
             "Real lines were never used to train the model; only the online calibrator sees them, date by date.\n")
    LP2 = LP.copy()
    LP2["pside"] = np.maximum(LP2.p_over_cal, 1 - LP2.p_over_cal)
    LP2["hit"] = np.where(LP2.p_over_cal >= 0.5, LP2.actual > LP2.line, LP2.actual < LP2.line).astype(int)
    for sea in sorted(LP2.season.unique()):
        x = LP2[LP2.season == sea]
        s.append(f"\n**Season {sea}** (n = {len(x):,})\n\n")
        s.append(H.md(H.tier_table(x, "pside", "hit", cuts=(0.5, 0.53, 0.56, 0.6, 0.65, 1.01))))
    x = LP2[LP2.p_stack.notna()].copy()
    if len(x):
        x["pside"] = np.maximum(x.p_stack, 1 - x.p_stack)
        x["hit"] = np.where(x.p_stack >= 0.5, x.actual > x.line, x.actual < x.line).astype(int)
        x["mside"] = np.maximum(x.p_mkt, 1 - x.p_mkt)
        s.append(f"\n**Variant: model + de-vigged market, walk-forward monthly logistic stack** "
                 f"(rows with both prices, n = {len(x):,}; all seasons)\n\n")
        s.append(H.md(H.tier_table(x, "pside", "hit", cuts=(0.5, 0.53, 0.56, 0.6, 0.65, 1.01))))
        y = (x.actual > x.line).astype(int)
        s.append(f"\nLog loss on these rows: model-cal {H.logloss(x.p_over_cal, y):.4f}, market {H.logloss(x.p_mkt, y):.4f}, "
                 f"stack {H.logloss(x.p_stack, y):.4f}.\n")
    # (b) milestones
    s.append("\n## (b) Milestones (k+ ladders), 2021-2026: realized hit rate by confidence tier vs naive base rates\n")
    s.append("`group_base` = hit rate of the same stat x threshold x role tier in prior seasons; `player_naive` = the "
             "player's own trailing (EWM, hl 10 games) frequency of reaching k. Lift = hit - naive.\n\n")
    ext = {"group_base": lambda x: x.base.mean(), "player_naive": lambda x: x.naive.mean(),
           "lift_vs_group": lambda x: x.hit.mean() - x.base.mean(),
           "lift_vs_player": lambda x: x.hit.mean() - x.naive.mean()}
    s.append(H.md(H.tier_table(R, "p", "hit", cuts=(0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01), extra=ext)))
    s.append("\nBy stat, legs with p >= 0.80:\n\n")
    rows = []
    for (st, k), x in R[R.p >= 0.8].groupby(["stat", "k"]):
        rows.append({"stat": st, "k": int(k), "n": len(x), "pred": x.p.mean(), "hit": x.hit.mean(),
                     "group_base": x.base.mean(), "player_naive": x.naive.mean(),
                     "all_legs_base(k)": R[(R.stat == st) & (R.k == k)].hit.mean()})
    s.append(H.md(pd.DataFrame(rows)))
    s.append("\nSelection skill at equal naive rate: legs where the player's naive rate is 0.80-0.90, split by model p:\n\n")
    x = R[(R.naive >= 0.8) & (R.naive < 0.9)]
    s.append(H.md(H.tier_table(x, "p", "hit", cuts=(0.0, 0.7, 0.8, 0.9, 1.01),
                               extra={"player_naive": lambda q: q.naive.mean()})))
    # (c) parlays
    s.append("\n## (c) Parlays, one ticket per day per construction, 2021-2026\n")
    s.append("Predicted = Gaussian-copula joint of calibrated leg probabilities with correlations learned from "
             "prior seasons' residuals (teammate/opponent stat pairs, player vs team margin and game total). "
             "`typical_price` = median ticket odds: real odds for real-line and ML legs; milestone legs priced at the "
             "player's naive rate plus 4.5% margin; totals -110.\n\n")
    s.append(H.md(H.parlay_summary(T)))
    s.append("\nBy season (realized / predicted joint), construction 1 and 4:\n\n")
    rows = []
    for (c, n, sea), x in T[T.construction.str[0].isin(["1", "4"])].groupby(["construction", "n_legs", "season"]):
        rows.append({"construction": c, "legs": n, "season": sea, "n_days": len(x), "pred": x.p_joint.mean(),
                     "realized": x.hit.mean()})
    s.append(H.md(pd.DataFrame(rows)))
    # (d) ablation
    if ABL is not None:
        s.append("\n## (d) Which tendencies mattered (ablation)\n")
        s.append("Single split: train 2017-2020, test a fixed random 35% of 2021-2026 dates (same dates for every row). "
                 "Each row refits the LightGBM without that feature group, then runs the same online calibrator. "
                 "Lower log loss = better.\n\n")
        s.append(H.md(ABL, "{:.4f}"))
    if TT is not None and len(TT):
        s.append("\n**Tendency evidence (2021-2026):** mean script residual z = (actual - script mu)/sd by bucket. "
                 "Non-zero means the coupled script under/over-shoots there and the LightGBM layer has something to "
                 "learn (the script already renormalises usage for teammates missing the previous game with exponent "
                 "0.5, so these are residual effects).\n\n")
        s.append(H.md(TT))
    if models is not None:
        m = models[max(models)].model
        imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        imp = (imp / imp.sum()).sort_values(ascending=False).head(20).reset_index()
        imp.columns = ["feature", "gain_share"]
        s.append("\nTop features (gain share, 2026 model; k/zk/rk = threshold terms):\n\n" + H.md(imp, "{:.4f}"))
    # (e) learning curve
    s.append("\n## (e) Learning curve by season (milestone legs; log loss / calibration error)\n\n")
    rows = []
    for sea, x in L.groupby("season"):
        r = {"season": sea, "n_legs": len(x), "ll_script_param": H.logloss(x.p_param, x.hit),
             "ll_lgbm": H.logloss(x.p_lgb, x.hit), "ll_calibrated": H.logloss(x.p, x.hit),
             "ece_param": H.ece(x.p_param, x.hit), "ece_lgbm": H.ece(x.p_lgb, x.hit), "ece_cal": H.ece(x.p, x.hit),
             "hit>=0.8": x[x.p >= 0.8].hit.mean(), "pred>=0.8": x[x.p >= 0.8].p.mean()}
        y = LP2[LP2.season == sea]
        r["line_n"] = len(y)
        r["line_hit_all"] = y.hit.mean() if len(y) else np.nan
        rows.append(r)
    s.append(H.md(pd.DataFrame(rows), "{:.4f}"))
    s.append("\nReal-line hit rate by month of the calibrator's life (all picks):\n\n")
    m = LP2.groupby(LP2.date.dt.to_period("Q").astype(str)).agg(n=("hit", "size"), hit=("hit", "mean"),
                                                               pred=("pside", "mean")).reset_index()
    s.append(H.md(m))
    (OUT / "nba_props.md").write_text("".join(s))


# ============================================================ live API
def build_tickets(date, sizes=(3, 5, 7)) -> dict:
    """Tonight's tickets from the pre-game script rows for `date` (players parquet must contain them).
    Returns {construction: [{n_legs, p_joint, legs}]}; legs carry their calibrated probabilities."""
    date = pd.Timestamp(date)
    st = pickle.load(open(OUT / "nba_state.pkl", "rb"))
    P = load_players()
    P = P[P.date <= date]
    F = build_frame(P)
    te = F[F.date == date]
    if not len(te):
        return {}
    legs = []
    for j in range(4):
        k = te[f"k{j}"].to_numpy(float)
        ok = ~np.isnan(k)
        x = te[ok]
        p = st["model"].predict(x, k[ok])
        grp = x.stat + "|" + pd.Series(k[ok], index=x.index).astype(int).astype(str)
        legs.append(pd.DataFrame({"date": x.date.values, "game_id": x.game_id.values, "kind": "P",
                                  "pid": x.player_id.values, "team": x.team.values, "stat": x.stat.values, "sgn": 1.0,
                                  "p": st["cal_ms"].apply_state(grp.values, p), "hit": 0, "dec": np.nan,
                                  "label": x.player.values + " " + k[ok].astype(int).astype(str) + "+ " + x.stat.values}))
    PL = pd.concat(legs)
    PL = PL[PL.p >= 0.55]
    try:
        GL = game_legs()
        GL = GL[GL.date == date]
    except Exception:
        GL = None
    rows = H.build_day(PL, GL, st["corr"], sizes=sizes)
    out = {}
    for r in rows:
        out.setdefault(r["construction"], []).append({"n_legs": r["n_legs"], "p_joint": r["p_joint"], "legs": r["legs"]})
    return out


if __name__ == "__main__":
    main(do_ablation="--no-ablation" not in sys.argv)
