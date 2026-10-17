"""
NFL script-driven player props + 3/5/7-leg parlays (spec: engine/props_script/SPEC.md).

  python -m engine.props_script.nfl
  from engine.props_script.nfl import build_tickets; build_tickets("2026-10-12")

Same architecture as engine/props_script/nba.py (shared code in common_hoops_football.py):
  script (results/scripts/nfl_players.parquet: share x scripted team pass/rush volume) + tendencies
  (rotation teammates who missed the team's previous game -> vacated target / carry share, QB change,
  opponent defence vs position (EWM of residuals allowed) + opponent EPA/pressure, rest, home/away,
  weather/roof, recent form (residual EWM, target/carry/snap share trends), player bias & dispersion,
  game-script / blowout terms) -> distributional LightGBM P(stat >= k), refit each season on the prior
  4 seasons -> online calibrator per stat x threshold x probability bin, date by date.
Fit/choose on seasons <= 2020; report 2021+ (2026 = weeks 1-5 only).
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
STATS = ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "completions"]
SHORT = {"receiving_yards": "rec yds", "receptions": "rec", "rushing_yards": "rush yds",
         "passing_yards": "pass yds", "completions": "cmp"}
LADDER = {"receiving_yards": [25, 40, 60, 80], "rushing_yards": [25, 40, 60, 80], "receptions": [2, 3, 4, 5],
          "passing_yards": [200, 250], "completions": []}
REPORT_FROM = 2021
TRAIN_YEARS = 4
REL_MIN = {"passing_yards": 150, "completions": 12, "rushing_yards": 25, "receiving_yards": 20, "receptions": 2}

GROUPS = {
    "script": ["mu", "sd", "share_of_team", "team_pass_att_s", "team_rush_att_s", "cal", "mu_raw", "prior_games",
               "stat_c", "pos_c", "playoff", "week"],
    "game_script_blowout": ["t_margin", "g_total", "abs_margin", "spread_team", "imp_total"],
    "teammates_out": ["vac_tgt", "vac_car", "n_vac", "self_missed_prev", "gap_games", "qb_changed", "qb_delta"],
    "opp_def_vs_pos": ["opp_pos_z", "opp_z", "opp_d_epa_pass", "opp_d_epa_rush", "opp_d_pres"],
    "rest": ["rest", "opp_rest"],
    "home_away": ["home"],
    "weather": ["wind", "temp", "dome", "div_game"],
    "recent_form": ["f_z2", "f_z5", "tgt3", "car3", "snap3", "snap_last"],
    "player_bias_dispersion": ["f_z20", "zsd", "n_prev"],
}
CATS = ["stat_c", "pos_c"]


# ============================================================ data
def load_week() -> pd.DataFrame:
    w = pd.read_csv(ROOT / "data/processed/nfl_player_week.csv.gz", low_memory=False)
    w = w[w.season >= 2014].copy()
    w["date"] = pd.to_datetime(w.gameday)
    return w


def week_features(w: pd.DataFrame):
    tg = w.groupby(["game_id", "team"]).agg(date=("date", "first"), season=("season", "first")).reset_index()
    tg = tg.sort_values(["team", "date"]).reset_index(drop=True)
    tg["ti"] = tg.groupby("team").cumcount()
    x = w.merge(tg[["game_id", "team", "ti"]], on=["game_id", "team"])
    tc = x.groupby(["game_id", "team"]).carries.transform("sum").replace(0, np.nan)
    tt = x.groupby(["game_id", "team"]).targets.transform("sum").replace(0, np.nan)
    x["tsh"] = x.targets / tt
    x["csh"] = x.carries / tc
    x = x.sort_values(["player_id", "team", "date"]).reset_index(drop=True)
    g = x.groupby(["player_id", "team"])
    x["tsh_post"] = g.tsh.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    x["csh_post"] = g.csh.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    x["nxt"] = g.ti.shift(-1)
    x["prv"] = g.ti.shift(1)
    seas = tg.set_index(["team", "ti"]).season
    rows = []
    for dt in range(2, 5):
        v = x[["team", "ti", "season", "nxt", "tsh_post", "csh_post", "position"]].copy()
        v["t"] = v.ti + dt
        v = v[(v.nxt.isna()) | (v.nxt >= v.t)]
        rows.append(v)
    V = pd.concat(rows)
    V = V[((V.tsh_post >= 0.12) | (V.csh_post >= 0.25)) & (V.position != "QB")]
    V["t_season"] = seas.reindex(pd.MultiIndex.from_arrays([V.team, V.t])).to_numpy()
    V = V[V.t_season == V.season]
    agg = V.groupby(["team", "t"]).agg(vac_tgt=("tsh_post", "sum"), vac_car=("csh_post", "sum"),
                                         n_vac=("tsh_post", "size")).reset_index().rename(columns={"t": "ti"})
    tg = tg.merge(agg, on=["team", "ti"], how="left").fillna({"vac_tgt": 0, "vac_car": 0, "n_vac": 0})
    x["self_missed_prev"] = ((x.ti - x.prv) > 1).astype(int)
    x["gap_games"] = (x.ti - x.prv - 1).clip(0, 8).fillna(0)
    x["tgt3"] = g.tsh.transform(lambda s: s.shift(1).ewm(halflife=2, ignore_na=True).mean())
    x["car3"] = g.csh.transform(lambda s: s.shift(1).ewm(halflife=2, ignore_na=True).mean())
    x["snap3"] = g.snap_pct.transform(lambda s: s.shift(1).ewm(halflife=2, ignore_na=True).mean())
    x["snap_last"] = g.snap_pct.shift(1)
    pf = x[["game_id", "player_id", "self_missed_prev", "gap_games", "tgt3", "car3", "snap3", "snap_last"]]
    return tg[["game_id", "team", "vac_tgt", "vac_car", "n_vac"]], pf.drop_duplicates(["game_id", "player_id"])


def load_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "results/scripts/nfl_games.parquet")
    return g


def build_frame(P=None, w=None) -> pd.DataFrame:
    t0 = time.time()
    P = pd.read_parquet(ROOT / "results/scripts/nfl_players.parquet") if P is None else P
    w = load_week() if w is None else w
    tgf, pf = week_features(w)
    G = load_games()
    P = P.merge(tgf, on=["game_id", "team"], how="left").merge(pf, on=["game_id", "player_id"], how="left")
    gk = G[["game_id", "home", "away", "margin", "total", "sd_margin", "sd_total", "spread_home", "total_line",
            "home_score", "away_score", "home_rest", "away_rest", "wind", "temp", "roof", "div_game", "game_type",
            "home_qb_name", "away_qb_name", "home_qb_delta", "away_qb_delta", "home_d_epa_pass", "away_d_epa_pass",
            "home_d_epa_rush", "away_d_epa_rush", "home_d_pres", "away_d_pres"]]
    P = P.merge(gk, on="game_id", how="left")
    ih = (P.team == P.home).to_numpy()
    sgn = np.where(ih, 1.0, -1.0)
    P["home"] = ih.astype(int)
    P["t_margin"] = sgn * P.margin
    P["g_total"] = P.total
    P["abs_margin"] = P.margin.abs()
    P["spread_team"] = sgn * P.spread_home
    P["imp_total"] = P.total_line / 2 - P.spread_team / 2
    P["rest"] = np.where(ih, P.home_rest, P.away_rest)
    P["opp_rest"] = np.where(ih, P.away_rest, P.home_rest)
    P["dome"] = P.roof.isin(["dome", "closed"]).astype(int)
    P["wind"] = np.where(P.dome == 1, 0, P.wind)
    P["playoff"] = (P.game_type != "REG").astype(int)
    P["qb_delta"] = np.where(ih, P.home_qb_delta, P.away_qb_delta)
    P["opp_d_epa_pass"] = np.where(ih, P.away_d_epa_pass, P.home_d_epa_pass)
    P["opp_d_epa_rush"] = np.where(ih, P.away_d_epa_rush, P.home_d_epa_rush)
    P["opp_d_pres"] = np.where(ih, P.away_d_pres, P.home_d_pres)
    # QB change vs the team's previous game
    tq = pd.concat([G[["game_id", "date", "home", "home_qb_name"]].set_axis(["game_id", "date", "team", "qb"], axis=1),
                    G[["game_id", "date", "away", "away_qb_name"]].set_axis(["game_id", "date", "team", "qb"], axis=1)])
    tq = tq.sort_values(["team", "date"])
    tq["qb_changed"] = (tq.qb != tq.groupby("team").qb.shift(1)).astype(int)
    P = P.merge(tq[["game_id", "team", "qb_changed"]], on=["game_id", "team"], how="left")
    act_m = np.where(ih, P.home_score - P.away_score, P.away_score - P.home_score)
    P["mres"] = (act_m - P.t_margin) / P.sd_margin
    P["tres"] = (P.home_score + P.away_score - P.total) / P.sd_total
    P["stat_c"] = P.stat.map({s: i for i, s in enumerate(STATS)}).astype(int)
    P["pos_c"] = P.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3}).fillna(2).astype(int)
    for c in ["vac_tgt", "vac_car", "n_vac"]:
        P[c] = P[c].fillna(0)

    P = P.sort_values(["player_id", "stat", "date", "game_id"]).reset_index(drop=True)
    P["z"] = (P.actual - P.mu) / P.sd.clip(lower=1)
    P["z2"] = P.z ** 2
    k = ["player_id", "stat"]
    P["f_z2"] = H.grp_ewm_prev(P, k, "z", 2)
    P["f_z5"] = H.grp_ewm_prev(P, k, "z", 5)
    P["f_z20"] = H.grp_ewm_prev(P, k, "z", 20)
    e2 = H.grp_ewm_prev(P, k, "z2", 12)
    P["zsd"] = np.sqrt(np.clip(e2 - P.f_z20.fillna(0) ** 2, 0.05, None))
    P["n_prev"] = H.grp_count_prev(P, k)
    for j in range(4):
        kk = P.stat.map(lambda s: LADDER[s][j] if j < len(LADDER[s]) else np.nan).to_numpy(float)
        P[f"k{j}"] = kk
        P[f"ind{j}"] = np.where(np.isnan(kk), np.nan, (P.actual >= kk).astype(float))
        P[f"naive{j}"] = H.grp_ewm_prev(P, k, f"ind{j}", 6)
    P["opp_pos_z"] = H.date_grp_ewm_prev(P, ["opp", "pos_c", "stat"], "z", 6)
    P["opp_z"] = H.date_grp_ewm_prev(P, ["opp", "stat"], "z", 6)
    P["tier"] = P.position
    print(f"  frame built {len(P):,} rows in {time.time() - t0:.0f}s", flush=True)
    return P


def all_feats(groups=None):
    groups = groups or list(GROUPS)
    return [c for g in groups for c in GROUPS[g]]


def train_ks(df, rng):
    ks = [df[f"k{j}"].to_numpy(float) for j in range(4)]
    for _ in range(3):
        r = np.round(df.mu.to_numpy() + df.sd.to_numpy() * rng.standard_normal(len(df)))
        ks.append(np.clip(r, 1, None))
    return ks


def y_ge(df, k):
    return (df.actual.to_numpy() >= k).astype(np.int8)


# ============================================================ real lines
def load_lines() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/processed/nfl_props.csv")
    p = p[p.result.isin(["over", "under"]) & p.player_id.notna()].copy()
    p["sb"] = (p.source_type == "sportsbook") & p.over_odds.notna() & p.under_odds.notna()
    po = 1 / H.american_to_dec(p.over_odds.fillna(-110))
    pu = 1 / H.american_to_dec(p.under_odds.fillna(-110))
    p["p_mkt"] = np.where(p.sb, po / (po + pu), np.nan)
    p["bal"] = np.where(p.sb, (p.p_mkt - 0.5).abs(), 0.5)
    p["snap"] = pd.to_datetime(p.snapshot_utc, errors="coerce")
    p["snap_max"] = p.groupby(["game_id", "player_id", "market"]).snap.transform("max")
    p = p[p.snap.isna() | (p.snap >= p.snap_max - pd.Timedelta(hours=3))]
    p = p.sort_values(["game_id", "player_id", "market", "bal"]).drop_duplicates(["game_id", "player_id", "market"])
    p = p[(p.line % 1) != 0]
    p["dec_over"] = H.american_to_dec(p.over_odds.fillna(-110))
    p["dec_under"] = H.american_to_dec(p.under_odds.fillna(-110))
    return p.rename(columns={"market": "stat"})[["game_id", "player_id", "stat", "line", "p_mkt", "dec_over",
                                                 "dec_under", "book", "sb"]]


# ============================================================ walk-forward
def walk_forward(P, feats, seasons, lines=None, n_est=300, verbose=True, seed=0):
    rng = np.random.default_rng(seed)
    legs, lp, models = [], [], {}
    for S in seasons:
        t0 = time.time()
        tr = P[(P.season < S) & (P.season >= S - TRAIN_YEARS)]
        te = P[P.season == S]
        m = H.CDFModel(feats=feats, cats=[c for c in CATS if c in feats], n_estimators=n_est,
                       params=dict(min_data_in_leaf=150, num_leaves=31, learning_rate=0.05))
        m.fit(tr, train_ks(tr, rng), y_ge, max_rows=900_000, seed=int(S))
        models[S] = m
        legs.append(ms_legs(te, m, S))
        if lines is not None:
            x = te.merge(lines, on=["game_id", "player_id", "stat"])
            if len(x):
                kk = np.floor(x.line.to_numpy()) + 1
                lp.append(pd.DataFrame({"date": x.date.values, "season": S, "game_id": x.game_id.values,
                                        "player_id": x.player_id.values, "player": x.player.values,
                                        "team": x.team.values, "stat": x.stat.values, "line": x.line.values,
                                        "actual": x.actual.values, "p_over_lgb": m.predict(x, kk),
                                        "p_over_param": H.param_cdf(x.mu.values, x.sd.values, kk),
                                        "p_mkt": x.p_mkt.values, "dec_over": x.dec_over.values,
                                        "dec_under": x.dec_under.values, "mu": x.mu.values}))
        if verbose:
            print(f"  season {S}: train {len(tr):,}, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(legs, ignore_index=True), (pd.concat(lp, ignore_index=True) if lp else None), models


def ms_legs(te, m, S, param=False):
    legs = []
    for j in range(4):
        k = te[f"k{j}"].to_numpy(float)
        ok = ~np.isnan(k)
        if not ok.any():
            continue
        x = te[ok]
        p = H.param_cdf(x.mu.values, x.sd.values, k[ok]) if param else m.predict(x, k[ok])
        legs.append(pd.DataFrame({
            "date": x.date.values, "season": x.season.values, "game_id": x.game_id.values,
            "player_id": x.player_id.values, "player": x.player.values, "team": x.team.values, "opp": x.opp.values,
            "stat": x.stat.values, "k": k[ok], "tier": x.tier.values, "p_lgb": p,
            "p_param": H.param_cdf(x.mu.values, x.sd.values, k[ok]), "naive_pl": x[f"naive{j}"].values,
            "hit": x[f"ind{j}"].values.astype(np.int8)}))
    return pd.concat(legs, ignore_index=True)


def base_rates(L):
    a = L.groupby(["stat", "k", "tier", "season"]).hit.agg(["sum", "count"]).reset_index().sort_values("season")
    a[["cs", "cn"]] = a.groupby(["stat", "k", "tier"])[["sum", "count"]].cumsum().to_numpy() - a[["sum", "count"]].to_numpy()
    a["base"] = a.cs / a.cn.replace(0, np.nan)
    return L.merge(a[["stat", "k", "tier", "season", "base"]], on=["stat", "k", "tier", "season"], how="left")["base"].to_numpy()


def calibrate_milestones(L, col="p_lgb"):
    L = L.sort_values("date", kind="stable")
    cal = H.OnlineCal(m=100)
    grp = L.stat + "|" + L.k.astype(int).astype(str)
    pc = cal.run(L.date.values, grp.values, L[col].to_numpy(), L.hit.to_numpy())
    return pd.Series(pc, index=L.index), cal


def calibrate_lines(LP, col="p_over_lgb"):
    LP = LP.sort_values("date", kind="stable")
    cal = H.OnlineCal(nbins=14, m=50, span=1.2)
    y = (LP.actual > LP.line).astype(int).to_numpy()
    pc = cal.run(LP.date.values, LP.stat.values, LP[col].to_numpy(), y)
    return pd.Series(pc, index=LP.index), cal


def season_corrs(P, seasons):
    rel = P.mu >= P.stat.map(REL_MIN)
    Z = P[rel][["season", "game_id", "team", "player_id", "stat", "z", "mres", "tres"]].copy()
    Z["z"] = Z.groupby(["season", "stat"]).z.transform(lambda s: (s - s.mean()) / s.std())
    Z = Z.rename(columns={"player_id": "pid"})
    out = {}
    for S in seasons:
        z = Z[(Z.season < S) & (Z.season >= S - 5)]
        out[S] = {"pp": H.pair_corr(z, STATS), "g": H.game_corr(z.dropna(subset=["mres", "tres"]), STATS)}
    return out


# ============================================================ parlays
def game_legs() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "results/predict/nfl_preds.parquet")
    ml = pd.DataFrame({"date": g.date, "game_id": g.game_id, "kind": "M", "pid": "-1",
                       "team": np.where(g.pick_home, g.home, g.away), "stat": "ml", "sgn": 1.0,
                       "p": g.pick_prob, "hit": g.ml_hit,
                       "dec": H.american_to_dec(np.where(g.pick_home, g.home_ml, g.away_ml)),
                       "label": np.where(g.pick_home, g.home, g.away) + " ML"})
    tot = pd.DataFrame({"date": g.date, "game_id": g.game_id, "kind": "T", "pid": "-2", "team": g.home,
                        "stat": "total", "sgn": np.where(g.total_pick_over == 1, 1.0, -1.0),
                        "p": g.total_prob, "hit": g.total_hit, "dec": 1.909,
                        "label": np.where(g.total_pick_over == 1, "o", "u") + g.total_line.astype(str) + " " + g.away + "@" + g.home})
    out = pd.concat([ml, tot])
    return out[out.p.notna() & out.hit.notna() & np.isfinite(out.dec)]


def prop_legs(L, pmin=0.5):
    x = L[L.p >= pmin]
    q = np.clip(x.naive.fillna(x.base).fillna(x.p).to_numpy() * 1.045, 0.05, 0.985)
    return pd.DataFrame({"date": x.date.values, "game_id": x.game_id.values, "kind": "P", "pid": x.player_id.values,
                         "team": x.team.values, "stat": x.stat.values, "sgn": 1.0, "p": x.p.values, "hit": x.hit.values,
                         "dec": 1 / q, "label": x.player.values + " " + x.k.astype(int).astype(str) + "+ " + x.stat.map(SHORT).values})


def real_legs(LP):
    over = LP.p_over_cal >= 0.5
    return pd.DataFrame({"date": LP.date.values, "game_id": LP.game_id.values, "kind": "P", "pid": LP.player_id.values,
                         "team": LP.team.values, "stat": LP.stat.values, "sgn": np.where(over, 1.0, -1.0),
                         "p": np.where(over, LP.p_over_cal, 1 - LP.p_over_cal),
                         "hit": np.where(over, LP.actual > LP.line, LP.actual < LP.line).astype(int),
                         "dec": np.where(over, LP.dec_over, LP.dec_under),
                         "label": LP.player.values + np.where(over, " o", " u") + LP.line.astype(str) + " " + LP.stat.map(SHORT).values})


def run_parlays(L, LP, GL, corrs, sizes=(3, 5, 7)):
    PL = prop_legs(L)
    RL = real_legs(LP) if LP is not None and len(LP) else None
    byG = dict(tuple(GL.groupby("date")))
    byR = dict(tuple(RL.groupby("date"))) if RL is not None else {}
    rows = []
    for d, props in PL.groupby("date"):
        season = int(L.loc[L.date == d, "season"].iloc[0])
        r = H.build_day(props, byG.get(d), corrs[season], sizes=sizes, real=byR.get(d))
        for x in r:
            x["date"] = d
            x["season"] = season
        rows += r
    return pd.DataFrame(rows)


# ============================================================ ablation
def ablation(P, lines):
    tr = P[(P.season >= REPORT_FROM - TRAIN_YEARS) & (P.season < REPORT_FROM)]
    te = P[P.season >= REPORT_FROM]
    full = list(GROUPS)
    configs = [("full model", full)] + [(f"drop {g}", [x for x in full if x != g]) for g in full if g != "script"] \
        + [("script features only (LGBM)", ["script"]), ("script parametric (no LGBM)", None)]
    res = []
    for name, groups in configs:
        t0 = time.time()
        m = None
        if groups is not None:
            fe = all_feats(groups)
            m = H.CDFModel(feats=fe, cats=[c for c in CATS if c in fe], n_estimators=300,
                           params=dict(min_data_in_leaf=150, num_leaves=31, learning_rate=0.05))
            m.fit(tr, train_ks(tr, np.random.default_rng(1)), y_ge, max_rows=900_000, seed=1)
        Lx = ms_legs(te, m, 0, param=groups is None).sort_values("date").reset_index(drop=True)
        Lx["p"] = calibrate_milestones(Lx)[0]
        variants = [(name, "p")] + ([("full model, no online calibrator", "p_lgb")] if name == "full model" else [])
        for vn, col in variants:
            pp = Lx[col].to_numpy()
            y = Lx.hit.to_numpy()
            s8, s9 = pp >= 0.8, pp >= 0.9
            t5 = Lx.assign(pp=pp).sort_values("pp", ascending=False).drop_duplicates(["date", "player_id"])
            t5 = t5.groupby("date").head(5).groupby("date").agg(n=("hit", "size"), h=("hit", "prod"))
            t5 = t5[t5.n == 5]
            res.append({"config": vn, "ms_logloss": H.logloss(pp, y), "ms_ece": H.ece(pp, y),
                        "n>=0.8": int(s8.sum()), "hit>=0.8": y[s8].mean(),
                        "lift_vs_player_naive>=0.8": y[s8].mean() - np.nanmean(Lx.naive_pl.to_numpy()[s8]),
                        "n>=0.9": int(s9.sum()), "hit>=0.9": y[s9].mean() if s9.any() else np.nan,
                        "top5_daily_hit": t5.h.mean()})
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
    print(f"walk-forward {time.time() - T0:.0f}s", flush=True)
    L = L.sort_values(["date", "game_id"]).reset_index(drop=True)
    pc, cal_ms = calibrate_milestones(L)
    L["p"] = pc
    L["base"] = base_rates(L)
    L["naive"] = L.naive_pl.fillna(L.base)
    LP = LP.sort_values("date").reset_index(drop=True)
    pcl, cal_ln = calibrate_lines(LP)
    LP["p_over_cal"] = pcl
    corrs = season_corrs(P, seasons)
    GL = game_legs()
    T = run_parlays(L[L.season >= REPORT_FROM], LP[LP.season >= REPORT_FROM], GL, corrs)
    print(f"parlays {time.time() - T0:.0f}s", flush=True)
    ABL = ablation(P, lines) if do_ablation else None
    keep = L[L.season >= 2017][["date", "season", "game_id", "player_id", "player", "team", "opp", "stat", "k", "p",
                                "p_lgb", "p_param", "naive", "base", "hit"]].assign(side="over(k+)", universe="milestone")
    lo = LP.assign(universe="real_line", side=np.where(LP.p_over_cal >= 0.5, "over", "under"),
                   p=np.maximum(LP.p_over_cal, 1 - LP.p_over_cal),
                   hit=np.where(LP.p_over_cal >= 0.5, LP.actual > LP.line, LP.actual < LP.line).astype(np.int8),
                   k=LP.line)[["date", "season", "game_id", "player_id", "player", "team", "stat", "k", "p",
                               "p_over_lgb", "p_over_cal", "p_mkt", "hit", "side", "universe"]]
    pd.concat([keep, lo], ignore_index=True).to_parquet(OUT / "nfl_legs.parquet", index=False)
    T.to_csv(OUT / "nfl_parlays.csv", index=False)
    with open(OUT / "nfl_state.pkl", "wb") as f:
        pickle.dump({"model": models[max(models)], "feats": feats, "cal_ms": cal_ms, "cal_ln": cal_ln,
                     "corr": corrs[max(corrs)]}, f)
    TT = H.tendency_table(P[P.season >= REPORT_FROM], TENDENCY_SPECS, STATS[:4])
    write_report(L, LP, T, ABL, models, time.time() - T0, TT)
    print(f"total {time.time() - T0:.0f}s")


TENDENCY_SPECS = [
    ("teammates out prev game: vacated target share", "vac_tgt", [-0.01, 0.0, 0.15, 0.3, 5]),
    ("teammates out prev game: vacated carry share", "vac_car", [-0.01, 0.0, 0.3, 0.6, 5]),
    ("player returning from absence", "self_missed_prev", [-0.5, 0.5, 1.5]),
    ("QB changed vs previous game", "qb_changed", [-0.5, 0.5, 1.5]),
    ("home", "home", [-0.5, 0.5, 1.5]),
    ("rest days", "rest", [0, 6, 7, 30]),
    ("opp defence vs position (prior z allowed)", "opp_pos_z", [-9, -0.2, -0.05, 0.05, 0.2, 9]),
    ("recent form (EWM z, hl 2)", "f_z2", [-9, -0.4, -0.1, 0.1, 0.4, 30]),
    ("player long-run bias (EWM z, hl 20)", "f_z20", [-9, -0.2, -0.05, 0.05, 0.2, 30]),
    ("snap share last game", "snap_last", [0, 0.4, 0.6, 0.8, 1.01]),
    ("wind (mph, outdoors)", "wind", [-1, 5, 10, 15, 80]),
    ("script team margin", "t_margin", [-40, -7, -3, 3, 7, 40]),
]


def write_report(L, LP, T, ABL, models, secs, TT=None):
    R = L[L.season >= REPORT_FROM]
    s = ["# NFL script-driven props + 3/5/7-leg parlays (walk-forward, report 2021+)\n",
         f"Code: `engine/props_script/nfl.py` (+ `common_hoops_football.py`). Runtime {secs / 60:.1f} min. "
         "Distributional LightGBM P(stat >= k) refit each season on the prior 4 seasons (script + tendencies), "
         "online calibrator per stat x threshold x probability bin, date by date. 2017-2020 fit/choose, 2021-2026 "
         "reported (2026 = weeks 1-5).\n"]
    s.append("\n## (a) Real lines: hit rate by model confidence (side = higher calibrated prob)\n")
    s.append("One line per player-game-market (latest snapshot, sportsbook line closest to even odds, half-points). "
             "2017-18 = completions only (propbets.football); 2026 = weeks 1-5, all five markets. "
             "Lines never used for training; the online calibrator (per market) learns from them date by date. "
             "**Small samples.**\n")
    LP2 = LP.copy()
    LP2["pside"] = np.maximum(LP2.p_over_cal, 1 - LP2.p_over_cal)
    LP2["hit"] = np.where(LP2.p_over_cal >= 0.5, LP2.actual > LP2.line, LP2.actual < LP2.line).astype(int)
    for sea in sorted(LP2.season.unique()):
        x = LP2[LP2.season == sea]
        s.append(f"\n**Season {sea}** (n = {len(x):,})\n\n")
        s.append(H.md(H.tier_table(x, "pside", "hit", cuts=(0.5, 0.53, 0.56, 0.6, 0.65, 1.01))))
    x = LP2[LP2.season == 2026]
    if len(x):
        s.append("\n2026 by market (all picks; p >= 0.58):\n\n")
        rows = []
        for st, y in x.groupby("stat"):
            yy = y[y.pside >= 0.58]
            rows.append({"market": st, "n": len(y), "hit": y.hit.mean(), "n>=0.58": len(yy),
                         "hit>=0.58": yy.hit.mean() if len(yy) else np.nan})
        s.append(H.md(pd.DataFrame(rows)))
        y = x[x.p_mkt.notna()]
        yo = (y.actual > y.line).astype(int)
        s.append(f"\nLog loss on 2026 rows with both prices (n={len(y)}): model-cal {H.logloss(y.p_over_cal, yo):.4f}, "
                 f"de-vigged market {H.logloss(y.p_mkt, yo):.4f}.\n")
    s.append("\n## (b) Milestones, 2021-2026: realized hit rate by confidence tier vs naive base rates\n")
    s.append("`group_base` = hit rate of the same stat x threshold x position in prior seasons; `player_naive` = the "
             "player's own trailing (EWM, hl 6 games) frequency of reaching k.\n\n")
    ext = {"group_base": lambda x: x.base.mean(), "player_naive": lambda x: x.naive.mean(),
           "lift_vs_group": lambda x: x.hit.mean() - x.base.mean(),
           "lift_vs_player": lambda x: x.hit.mean() - x.naive.mean()}
    s.append(H.md(H.tier_table(R, "p", "hit", cuts=(0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01), extra=ext)))
    s.append("\nBy stat x threshold, legs with p >= 0.75:\n\n")
    rows = []
    for (st, k), x in R[R.p >= 0.75].groupby(["stat", "k"]):
        rows.append({"stat": st, "k": int(k), "n": len(x), "pred": x.p.mean(), "hit": x.hit.mean(),
                     "group_base": x.base.mean(), "player_naive": x.naive.mean(),
                     "all_legs_base(k)": R[(R.stat == st) & (R.k == k)].hit.mean()})
    s.append(H.md(pd.DataFrame(rows)))
    s.append("\n## (c) Parlays, one ticket per game-day per construction, 2021-2026\n")
    s.append("Predicted = Gaussian-copula joint (correlations from prior seasons' residuals). `typical_price` = median "
             "ticket odds (real odds for real-line and ML legs; milestone legs priced at player naive rate + 4.5%).\n\n")
    s.append(H.md(H.parlay_summary(T)))
    if ABL is not None:
        s.append("\n## (d) Which tendencies mattered (ablation; train 2017-2020, test all 2021-2026)\n\n")
        s.append(H.md(ABL, "{:.4f}"))
    if TT is not None and len(TT):
        s.append("\n**Tendency evidence (2021-2026):** mean script residual z = (actual - mu)/sd by bucket and stat.\n\n")
        s.append(H.md(TT))
    s.append("\n## (e) Learning curve by season\n\n")
    rows = []
    for sea, x in L.groupby("season"):
        y = LP2[LP2.season == sea]
        rows.append({"season": sea, "n_legs": len(x), "ll_script_param": H.logloss(x.p_param, x.hit),
                     "ll_lgbm": H.logloss(x.p_lgb, x.hit), "ll_calibrated": H.logloss(x.p, x.hit),
                     "ece_param": H.ece(x.p_param, x.hit), "ece_lgbm": H.ece(x.p_lgb, x.hit),
                     "ece_cal": H.ece(x.p, x.hit), "hit>=0.75": x[x.p >= 0.75].hit.mean(),
                     "pred>=0.75": x[x.p >= 0.75].p.mean(), "line_n": len(y),
                     "line_hit": y.hit.mean() if len(y) else np.nan})
    s.append(H.md(pd.DataFrame(rows), "{:.4f}"))
    m = models[max(models)].model
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15).reset_index()
    imp.columns = ["feature", "gain_share"]
    s.append("\nTop features (gain share, 2026 model):\n\n" + H.md(imp, "{:.4f}"))
    (OUT / "nfl_props.md").write_text("".join(s))


# ============================================================ live API
def build_tickets(date, sizes=(3, 5, 7)) -> dict:
    date = pd.Timestamp(date)
    st = pickle.load(open(OUT / "nfl_state.pkl", "rb"))
    P = pd.read_parquet(ROOT / "results/scripts/nfl_players.parquet")
    F = build_frame(P[P.date <= date])
    te = F[F.date == date]
    if not len(te):
        return {}
    L = ms_legs(te, st["model"], 0)
    L["p"] = st["cal_ms"].apply_state((L.stat + "|" + L.k.astype(int).astype(str)).values, L.p_lgb.to_numpy())
    L["naive"] = L.naive_pl
    L["base"] = np.nan
    PL = prop_legs(L, pmin=0.55)
    GL = game_legs()
    rows = H.build_day(PL, GL[GL.date == date], st["corr"], sizes=sizes)
    out = {}
    for r in rows:
        out.setdefault(r["construction"], []).append({"n_legs": r["n_legs"], "p_joint": r["p_joint"], "legs": r["legs"]})
    return out


if __name__ == "__main__":
    main(do_ablation="--no-ablation" not in sys.argv)
