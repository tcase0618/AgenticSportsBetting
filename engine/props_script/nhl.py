"""
NHL script-driven player props (SOG / points, plus goals & assists for real lines) and 3/5/7-leg parlays.

  python -m engine.props_script.nhl            # full walk-forward run + report (~10-15 min)

Pipeline (strictly walk-forward):
  1. script mean   results/scripts/nhl_players.parquet: SOG mu = learned shot share x scripted team shots,
                   points mu = points-per-team-goal share x scripted team goals (lineups as dressed).
  2. tendencies    LightGBM Poisson on top of log(script mu), refit every season on prior seasons only. Player
                   tendencies built from the NHL boxscore cache (data/raw/nhl_api), every value shifted (prior
                   games only): TOI expectation (EWMA / last-3 / trend), line rank inside the team by expected TOI
                   (F lines 1-4, D pairs 1-3), PP role (PP-goal rate & share of team PP goals), per-60 shot /
                   point / goal / assist rates, recent form (last 5), rest; opponent shots-allowed & goals-allowed
                   EWMA, own shots-for, opposing starting goalie (shift-safe save% EWMA, experience, script goalie
                   effect, backup flag), own / opponent back-to-back, home/away, game script (expected total,
                   own expected goals, own win probability, scripted team shots).
  3. dispersion    NB size by mu-bin from prior seasons' out-of-sample residuals.
  4. calibrator    online (stat x threshold) Platt + confidence-bin residual table, updated after every date.
  5. parlays       top-N cross-game / script-correlated same-game stacks / ML + props, joint probability by
                   Gaussian copula with latent correlations estimated from prior seasons' PIT scores.
Fit / choose on seasons <= 2020 (season = year the season ends), report 2021+.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from engine.props_script import common_count as C

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RS, RP, PROC = ROOT / "results" / "scripts", ROOT / "results" / "predict", ROOT / "data" / "processed"
OUT = ROOT / "results" / "props_script"
BOX = ROOT / "data" / "raw" / "nhl_api"
FIRST_TEST, REPORT_FROM = 2018, 2021
MIN_GP = 5
LEGS = {"sog": [1, 2, 3, 4, 5], "points": [1, 2, 3], "goals": [1, 2], "assists": [1, 2]}
MILESTONES = [("sog", 1), ("sog", 2), ("sog", 3), ("sog", 4), ("points", 1)]
STAT_LABEL = {"sog": "SOG", "points": "PTS", "goals": "G", "assists": "A"}

GROUPS = {
    "toi_role": ["toi_ewm", "toi_l3", "toi_trend", "line_rank", "ppg_ewm", "pp_share"],
    "rates": ["sog60", "pts60", "g60", "a60", "goal_frac", "szn_sog_pg"],
    "form": ["sog_l5", "pts_l5", "rest_days"],
    "opponent": ["opp_sa_ewm", "opp_ga_ewm", "own_sf_ewm"],
    "goalie": ["opp_gsv", "opp_g_n", "opp_g_eff", "opp_g_backup"],
    "rest_home": ["home", "own_b2b", "opp_b2b"],
    "game_script": ["g_total", "mu_own", "p_win_own", "team_shots"],
}
BASE_FEATS = ["log_mu", "share", "n_prior", "is_d", "gp_prior"]


def season_of(d):
    d = pd.to_datetime(d)
    return np.where(d.dt.month >= 8, d.dt.year + 1, d.dt.year)


def _lag(df, key, col, fn):
    return df.groupby(key, sort=False)[col].transform(lambda s: fn(s.shift()))


# ------------------------------------------------------------------ data
def load(log=print):
    t0 = time.time()
    bp = pd.read_parquet(BOX / "nhl_box_players.parquet")
    bp["date"] = pd.to_datetime(bp.date)
    bg = pd.read_parquet(BOX / "nhl_box_games.parquet")
    bg["date"] = pd.to_datetime(bg.date)
    pl = pd.read_parquet(RS / "nhl_players.parquet")
    gm = pd.read_parquet(RS / "nhl_games.parquet")
    log(f"  loaded box {len(bp)}, players {len(pl)}, games {len(gm)} ({time.time()-t0:.0f}s)")

    # ---- team-level shift-safe pace / defence
    rows = []
    for r in bg.itertuples(index=False):
        rows.append((r.game_id, r.date, r.home, r.home_sog, r.away_sog, r.home_score, r.away_score))
        rows.append((r.game_id, r.date, r.away, r.away_sog, r.home_sog, r.away_score, r.home_score))
    tb = pd.DataFrame(rows, columns=["game_id", "date", "team", "sf", "sa", "gf", "ga"]).sort_values(["date", "game_id"])
    for c in ("sf", "sa", "ga"):
        tb[f"{c}_ewm"] = _lag(tb, "team", c, lambda s: s.ewm(halflife=15, min_periods=1).mean())
    sk = bp[bp.pos != "G"].copy()
    tpp = sk.groupby(["game_id", "team"]).ppg.sum().rename("team_ppg").reset_index()
    tb = tb.merge(tpp, on=["game_id", "team"], how="left")
    tb["team_ppg_ewm"] = _lag(tb, "team", "team_ppg", lambda s: s.ewm(halflife=30, min_periods=1).mean())

    # ---- goalies: starters' shift-safe save% EWMA
    g = bp[(bp.pos == "G") & (bp.starter == 1)].sort_values(["date", "game_id"]).copy()
    g["sa_e"] = _lag(g, "player_id", "shots_against", lambda s: s.ewm(halflife=25, min_periods=1).mean())
    g["ga_e"] = _lag(g, "player_id", "goals_against", lambda s: s.ewm(halflife=25, min_periods=1).mean())
    g["g_n"] = g.groupby("player_id").cumcount()
    g["gsv"] = 1 - (g.ga_e * g.g_n.clip(upper=40) + 0.095 * 10) / (g.sa_e * g.g_n.clip(upper=40) + 10)
    g.loc[g.g_n == 0, "gsv"] = 0.905
    gk = g.drop_duplicates(["game_id", "team"])[["game_id", "team", "gsv", "g_n"]]

    # ---- skater tendencies (shift-safe)
    sk = sk[sk.toi_s > 0].sort_values(["date", "game_id"]).reset_index(drop=True)
    sk["season"] = season_of(sk.date)
    key = "player_id"
    ew = lambda hl: (lambda s: s.ewm(halflife=hl, min_periods=1).mean())
    sk["toi_m"] = sk.toi_s / 60
    sk["toi_ewm"] = _lag(sk, key, "toi_m", ew(8))
    sk["toi_l3"] = _lag(sk, key, "toi_m", lambda s: s.rolling(3, min_periods=1).mean())
    sk["toi_trend"] = sk.toi_l3 - sk.toi_ewm
    for c, hl in (("sog", 20), ("points", 25), ("goals", 25), ("assists", 25), ("ppg", 30)):
        sk[f"{c}_e"] = _lag(sk, key, c, ew(hl))
    sk["toi_e20"] = _lag(sk, key, "toi_m", ew(20))
    sk["sog60"] = sk.sog_e / sk.toi_e20 * 60
    sk["pts60"] = sk.points_e / sk.toi_e20 * 60
    sk["g60"] = sk.goals_e / sk.toi_e20 * 60
    sk["a60"] = sk.assists_e / sk.toi_e20 * 60
    sk["is_d"] = (sk.pos == "D").astype(int)
    prior_gf = np.where(sk.is_d == 1, 0.22, 0.40)
    sk["goal_frac"] = (sk.goals_e * 10 + prior_gf * 2) / (sk.points_e * 10 + 2)
    sk["ppg_ewm"] = sk.ppg_e
    sk["sog_l5"] = _lag(sk, key, "sog", lambda s: s.rolling(5, min_periods=1).mean())
    sk["pts_l5"] = _lag(sk, key, "points", lambda s: s.rolling(5, min_periods=1).mean())
    sk["gp_prior"] = sk.groupby(key).cumcount()
    sk["rest_days"] = sk.groupby(key).date.diff().dt.days.clip(upper=30)
    sk["szn_sog_pg"] = sk.groupby([key, "season"]).sog.transform(lambda s: s.shift().expanding().mean())
    for k in (1, 2, 3, 4, 5):
        sk[f"_h_sog{k}"] = (sk.sog >= k).astype(float)
        sk[f"bp_sog{k}"] = _lag(sk, key, f"_h_sog{k}", lambda s: s.rolling(10, min_periods=5).mean())
    for st, ks in (("points", (1, 2, 3)), ("goals", (1, 2)), ("assists", (1, 2))):
        for k in ks:
            sk[f"_h_{st}{k}"] = (sk[st] >= k).astype(float)
            sk[f"bp_{st}{k}"] = _lag(sk, key, f"_h_{st}{k}", lambda s: s.rolling(10, min_periods=5).mean())
    sk = sk.drop(columns=[c for c in sk.columns if c.startswith("_h_")])
    # line rank inside team by expected TOI (pre-game)
    sk["posgrp"] = np.where(sk.is_d == 1, "D", "F")
    sk["toi_rank"] = sk.groupby(["game_id", "team", "posgrp"]).toi_ewm.rank(ascending=False, method="first",
                                                                           na_option="bottom")
    sk["line_rank"] = np.where(sk.is_d == 1, np.ceil(sk.toi_rank / 2), np.ceil(sk.toi_rank / 3))
    # team / opponent / goalie context
    sk = sk.merge(tb[["game_id", "team", "sf_ewm", "team_ppg_ewm"]].rename(columns={"sf_ewm": "own_sf_ewm"}),
                  on=["game_id", "team"], how="left")
    sk = sk.merge(tb[["game_id", "team", "sa_ewm", "ga_ewm"]].rename(
        columns={"team": "opp", "sa_ewm": "opp_sa_ewm", "ga_ewm": "opp_ga_ewm"}), on=["game_id", "opp"], how="left")
    sk["pp_share"] = (sk.ppg_e / sk.team_ppg_ewm.replace(0, np.nan)).clip(0, 1)
    sk = sk.merge(gk.rename(columns={"team": "opp", "gsv": "opp_gsv", "g_n": "opp_g_n"}), on=["game_id", "opp"],
                  how="left")

    # ---- script layer
    w = pl.pivot_table(index=["nhl_id", "player_id"], columns="stat",
                       values=["mu", "nb_r", "share_of_team", "team_script"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    npr = pl[pl.stat == "sog"][["nhl_id", "player_id", "n_prior"]]
    w = w.merge(npr, on=["nhl_id", "player_id"])
    d = sk.merge(w.rename(columns={"nhl_id": "game_id"}), on=["game_id", "player_id"], how="inner")
    gs = gm.copy()
    gs["game_id"] = gs.game_id.astype(int)
    cols = ["game_id", "home", "total", "mu_home", "mu_away", "p_home_blend", "home_b2b", "away_b2b",
            "home_g_backup", "away_g_backup", "goalie_home_eff", "goalie_away_eff", "margin", "home_score",
            "away_score"]
    d = d.merge(gs[cols].rename(columns={"home": "home_team"}), on="game_id", how="left")
    h = d.home == 1
    d["g_total"] = d.total
    d["mu_own"] = np.where(h, d.mu_home, d.mu_away)
    d["p_win_own"] = np.where(h, d.p_home_blend, 1 - d.p_home_blend)
    d["own_b2b"] = np.where(h, d.home_b2b, d.away_b2b)
    d["opp_b2b"] = np.where(h, d.away_b2b, d.home_b2b)
    d["opp_g_backup"] = np.where(h, d.away_g_backup, d.home_g_backup)
    d["opp_g_eff"] = np.where(h, d.goalie_away_eff, d.goalie_home_eff)
    d["margin_own_pred"] = np.where(h, d.margin, -d.margin)
    d["margin_own_act"] = np.where(h, d.home_score - d.away_score, d.away_score - d.home_score)
    d["total_act"] = d.home_score + d.away_score
    d["team_shots"] = d.team_script_sog
    d["share"] = d.share_of_team_sog
    d["season"] = season_of(d.date)
    d = d.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)
    log(f"  model frame {len(d)} rows, seasons {d.season.min()}-{d.season.max()} ({time.time()-t0:.0f}s)")
    return d


# ------------------------------------------------------------------ model
def offsets(d):
    gf = d.goal_frac.fillna(pd.Series(np.where(d.is_d == 1, 0.22, 0.40), index=d.index)).clip(0.05, 0.8)
    return {"sog": np.log(d.mu_sog.clip(lower=0.02)), "points": np.log(d.mu_points.clip(lower=0.01)),
            "goals": np.log((d.mu_points * gf).clip(lower=0.005)),
            "assists": np.log((d.mu_points * (1 - gf)).clip(lower=0.005))}


def feats_for(stat, groups=GROUPS):
    f = list(BASE_FEATS)
    for v in groups.values():
        f += v
    return f


def fit_all(d, log=print, groups=GROUPS, stats=("sog", "points", "goals", "assists"), rounds=150):
    off = offsets(d)
    mu, imps = {}, {}
    for st in stats:
        x = d.assign(log_mu=off[st], _off=off[st])
        feats = feats_for(st, groups)
        # SOG / points (milestones): refit every season; goals / assists (real lines only): every 3rd season
        mu[st], imps[st] = C.walk_forward_gbm(x, feats, st, "_off", "season", FIRST_TEST, rounds=rounds, log=log,
                                              refit_every=1 if st in ("sog", "points") else 3)
    return mu, imps


def dispersion(d, stat, mu):
    """season -> (edges, r) fitted on prior seasons' out-of-sample (walk-forward) means."""
    out = {}
    y = d[stat].values
    for s in sorted(d.season.unique()):
        tr = (d.season < s).values
        if tr.sum() < 20000:
            tr = (d.season <= s).values if s == d.season.min() else tr
        out[s] = C.fit_r_by_mu(y[tr], mu[tr]) if tr.sum() >= 1000 else C.fit_r_by_mu(y, mu)
    return out


def r_for(d, disp, mu):
    r = np.empty(len(d))
    for s, (e, rs) in disp.items():
        m = (d.season == s).values
        r[m] = C.apply_r(mu[m], e, rs)
    return r


def make_legs(d, mu, r, base_mu_r):
    """Long leg table: one row per player-game x (stat, k)."""
    parts = []
    for st, ks in LEGS.items():
        for k in ks:
            bmu, br = base_mu_r[st]
            x = pd.DataFrame(dict(row=np.arange(len(d)), stat=st, k=k,
                                  p_base=C.nb_ge(k, bmu, br), p_model=C.nb_ge(k, mu[st], r[st]),
                                  hit=(d[st].values >= k).astype(float),
                                  base_player=d[f"bp_{st}{k}"].values))
            parts.append(x)
    L = pd.concat(parts, ignore_index=True)
    L["date"] = d.date.values[L.row]
    L["season"] = d.season.values[L.row]
    L["posgrp"] = np.where(d.is_d.values[L.row] == 1, "D", "F")
    L["gp_prior"] = d.gp_prior.values[L.row]
    # naive base rate: league rate for (stat, k, position group) from prior seasons
    rate = L.groupby(["stat", "k", "posgrp", "season"]).hit.agg(["sum", "size"]).reset_index()
    rate = rate.sort_values("season")
    rate["cs"] = rate.groupby(["stat", "k", "posgrp"])["sum"].cumsum() - rate["sum"]
    rate["cn"] = rate.groupby(["stat", "k", "posgrp"])["size"].cumsum() - rate["size"]
    rate["base_pos"] = rate.cs / rate.cn.replace(0, np.nan)
    L = L.merge(rate[["stat", "k", "posgrp", "season", "base_pos"]], on=["stat", "k", "posgrp", "season"], how="left")
    L["base_player"] = L.base_player.fillna(L.base_pos)
    keys = {sk: i for i, sk in enumerate((s, k) for s, ks in LEGS.items() for k in ks)}
    L["key"] = [keys[(s, k)] for s, k in zip(L.stat, L.k)]
    return L


# ------------------------------------------------------------------ correlations
def rho_book(d, mu, r):
    x = d[["season", "game_id", "team", "margin_own_pred", "margin_own_act", "total", "total_act"]].copy()
    x["z_sog"] = C.pit_z(d.sog.values, mu["sog"], r["sog"], seed=1)
    x["z_points"] = C.pit_z(d.points.values, mu["points"], r["points"], seed=2)
    em = x.margin_own_act - x.margin_own_pred
    et = x.total_act - x.total
    x["z_m"] = (em - em.mean()) / em.std()
    x["z_t"] = (et - et.mean()) / et.std()
    mom = C.pair_moments(x, "season", "game_id", "team", ["z_sog", "z_points"])
    for st in ("sog", "points"):
        mom.update(C.row_moments(x, "season", f"z_{st}", "z_m", f"ml|{st}"))
        mom.update(C.row_moments(x, "season", f"z_{st}", "z_t", f"tot|{st}"))
    hm = x[d.home.values == 1].drop_duplicates("game_id")
    mom.update(C.row_moments(hm, "season", "z_m", "z_t", "mltot"))
    seasons = sorted(d.season.unique()) + [int(d.season.max()) + 1]
    return C.rho_tables_by_season(mom, seasons)


# ------------------------------------------------------------------ real lines
def real_lines(d, L):
    pr = pd.read_csv(PROC / "nhl_props.csv")
    pr = pr.dropna(subset=["line", "actual"])
    pr["stat"] = pr.market.map({"shots_on_goal": "sog", "points": "points", "goals": "goals", "assists": "assists"})
    pr["game_id"] = pr.game_id.astype(int)
    pr["k"] = (np.floor(pr.line) + 1).astype(int)
    idx = d[["game_id", "player_id"]].reset_index().rename(columns={"index": "row"})
    pr = pr.merge(idx, on=["game_id", "player_id"], how="inner")
    pr = pr.merge(L[["row", "stat", "k", "p_cal", "p_model", "p_base"]], on=["row", "stat", "k"], how="inner")
    pr["over"] = (pr.actual > pr.line).astype(float)
    pr["p_mkt"] = C.sigmoid(0)  # placeholder
    from engine.scripts.sgp_builder import fair2
    pr["p_mkt"] = fair2(pr.over_odds.values, pr.under_odds.values)
    pr["pick_over"] = pr.p_cal >= 0.5
    pr["p_pick"] = np.where(pr.pick_over, pr.p_cal, 1 - pr.p_cal)
    pr["hit"] = np.where(pr.pick_over, pr.over, 1 - pr.over)
    pr["mkt_pick_over"] = pr.p_mkt >= 0.5
    pr["mkt_hit"] = np.where(pr.mkt_pick_over, pr.over, 1 - pr.over)
    pr["p_mkt_side"] = np.where(pr.pick_over, pr.p_mkt, 1 - pr.p_mkt)
    return pr


def real_line_table(pr):
    rows = []
    bands = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for grp, x in [("all", pr)] + list(pr.groupby("market")):
        for lo, hi in bands:
            s = x[(x.p_pick >= lo) & (x.p_pick < hi)]
            if len(s) == 0:
                continue
            rows.append(dict(market=grp, tier=f"{int(lo*100)}-{int(min(hi,1)*100)}%", n=len(s), pred=s.p_pick.mean(),
                             hit=s.hit.mean(), mkt_implied_same_side=s.p_mkt_side.mean(),
                             flag="SMALL" if len(s) < 100 else ""))
        rows.append(dict(market=grp, tier="ALL picks", n=len(x), pred=x.p_pick.mean(), hit=x.hit.mean(),
                         mkt_implied_same_side=x.p_mkt_side.mean(), flag="SMALL" if len(x) < 100 else ""))
        rows.append(dict(market=grp, tier="market-favoured side (baseline)", n=len(x),
                         pred=np.where(x.mkt_pick_over, x.p_mkt, 1 - x.p_mkt).mean(), hit=x.mkt_hit.mean(),
                         mkt_implied_same_side=np.nan, flag="SMALL" if len(x) < 100 else ""))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ parlay legs
def prop_leg_frame(d, L):
    m = L[L.set_index(["stat", "k"]).index.isin(MILESTONES) & (L.gp_prior >= MIN_GP)]
    x = pd.DataFrame(dict(date=m.date.values, season=m.season.values, gid=d.game_id.values[m.row].astype(str),
                          team=d.team.values[m.row], team_home=d.home.values[m.row], ent=d.player_id.values[m.row],
                          stat=m.stat.values, kind="prop", sign=1,
                          label=[f"{n} {STAT_LABEL[s]} {k}+" for n, s, k in zip(d.player.values[m.row], m.stat, m.k)],
                          p=m.p_cal.values, hit=m.hit.values))
    return x


def game_leg_frames(seasons_from):
    p = pd.read_parquet(RP / "nhl_preds.parquet")
    p["season"] = season_of(p.date)
    p = p[p.season >= seasons_from]
    ml = pd.DataFrame(dict(date=p.date, season=p.season, gid=p.game_id.astype(str), team=p.pick,
                           team_home=p.pick_home, ent="ML" + p.game_id.astype(str), stat="ml", kind="ml", sign=1,
                           label="ML " + p.pick, p=p.pick_prob, hit=p.pick_hit))
    t = p.dropna(subset=["total_prob"])
    tot = pd.DataFrame(dict(date=t.date, season=t.season, gid=t.game_id.astype(str), team=None, team_home=0,
                            ent="TOT" + t.game_id.astype(str), stat="total", kind="total",
                            sign=np.where(t.total_pick == "OVER", 1, -1),
                            label=t.total_pick + " " + t.tl.astype(str) + " " + t.away + "@" + t.home,
                            p=t.total_prob, hit=t.total_hit))
    return ml, tot


# ------------------------------------------------------------------ main
def run(log=print, quick=False):
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(log)
    if quick:                                     # smoke test: 1 game in 6
        d = d[d.game_id % 6 == 0].reset_index(drop=True)
    log("  fitting tendencies layer (yearly walk-forward)...")
    mu, imps = fit_all(d, log=log)
    log(f"  fitted ({time.time()-T0:.0f}s)")
    r, disp = {}, {}
    for st in LEGS:
        disp[st] = dispersion(d, st, mu[st])
        r[st] = r_for(d, disp[st], mu[st])
    off = offsets(d)
    base_mu_r = {"sog": (d.mu_sog.values, d.nb_r_sog.values), "points": (d.mu_points.values, np.full(len(d), np.inf)),
                 "goals": (np.exp(off["goals"].values), np.full(len(d), np.inf)),
                 "assists": (np.exp(off["assists"].values), np.full(len(d), np.inf))}
    L = make_legs(d, mu, r, base_mu_r)
    L["p_cal"], L["p_platt"], _ = C.walk_calibrate(L.date.values.astype("datetime64[D]").astype(int), L.key.values,
                                                   L.p_model.values, L.hit.values)
    L["p_base_cal"], _, _ = C.walk_calibrate(L.date.values.astype("datetime64[D]").astype(int), L.key.values,
                                             L.p_base.values, L.hit.values)
    log(f"  legs {len(L)} calibrated ({time.time()-T0:.0f}s)")

    rho = rho_book(d, mu, r)
    rho_fn = lambda s: C.RhoBook(rho.get(s, rho[max(rho)]))

    # ---- milestone evaluation (report seasons, eligible players)
    M = L[L.set_index(["stat", "k"]).index.isin(MILESTONES) & (L.season >= REPORT_FROM) & (L.gp_prior >= MIN_GP)].copy()
    M["leg"] = M.stat.map(STAT_LABEL) + " " + M.k.astype(str) + "+"
    tiers_all = C.tier_table(M, "p_cal", "hit", "base_pos", "base_player")
    tiers_by = C.tier_table(M, "p_cal", "hit", "base_pos", "base_player", tiers=[0.70, 0.80, 0.90], group="leg")
    tiers_raw = C.tier_table(M, "p_model", "hit", "base_pos", "base_player")

    # ---- learning curve
    lc = []
    for s, x in L[L.set_index(["stat", "k"]).index.isin(MILESTONES) & (L.gp_prior >= MIN_GP)].groupby("season"):
        hi = x[x.p_cal >= 0.8]
        lc.append(dict(season=s, n_legs=len(x), ll_script=C.ll(x.p_base, x.hit), ll_model=C.ll(x.p_model, x.hit),
                       ll_cal=C.ll(x.p_cal, x.hit), ece_script=C.ece(x.p_base, x.hit), ece_model=C.ece(x.p_model, x.hit),
                       ece_cal=C.ece(x.p_cal, x.hit), n_ge80=len(hi), pred_ge80=hi.p_cal.mean(), hit_ge80=hi.hit.mean()))
    lc = pd.DataFrame(lc)

    # ---- ablation (single split: train <= 2020, test 2021+; raw model probs, no calibrator)
    log("  ablation...")
    abl = ablation(d, base_mu_r, log)
    # calibrated script-only vs calibrated full model on the same legs
    abl_cal = pd.DataFrame([
        dict(variant="script only + calibrator", ll=C.ll(M.p_base_cal, M.hit), brier=float(((M.p_base_cal - M.hit) ** 2).mean()),
             n_ge80=int((M.p_base_cal >= .8).sum()), hit_ge80=M.hit[M.p_base_cal >= .8].mean()),
        dict(variant="full walk-forward + calibrator", ll=C.ll(M.p_cal, M.hit), brier=float(((M.p_cal - M.hit) ** 2).mean()),
             n_ge80=int((M.p_cal >= .8).sum()), hit_ge80=M.hit[M.p_cal >= .8].mean())])
    log(f"  ablation done ({time.time()-T0:.0f}s)")

    # ---- real lines
    pr = real_lines(d, L)
    rl = real_line_table(pr)

    # ---- parlays
    log("  parlays...")
    props = prop_leg_frame(d, L[L.season >= REPORT_FROM])
    ml, tot = game_leg_frames(REPORT_FROM)
    P = C.run_parlays(props, ml, tot, rho_fn, log=log)
    P.to_csv(OUT / "nhl_parlays.csv", index=False)
    PS = C.parlay_summary(P)
    ml_avail = dict(ml85=int((ml.groupby("date").p.max() >= 0.85).sum()), ml70=int((ml.groupby("date").p.max() >= 0.70).sum()),
                    days=int(props.date.nunique()))
    log(f"  parlays done ({time.time()-T0:.0f}s)")

    # ---- save legs
    S = L[L.set_index(["stat", "k"]).index.isin(MILESTONES) & (L.season >= REPORT_FROM - 1)].copy()   # <40MB
    S = pd.DataFrame(dict(date=S.date.values, season=S.season.values.astype(np.int16),
                          game_id=d.game_id.values[S.row].astype(np.int64), player_id=d.player_id.values[S.row].astype(np.int64),
                          player=pd.Categorical(d.player.values[S.row]), team=pd.Categorical(d.team.values[S.row]),
                          home=d.home.values[S.row].astype(np.int8), stat=pd.Categorical(S.stat.values),
                          threshold=S.k.values.astype(np.int8), side="over",
                          mu=np.where(S.stat.values == "sog", mu["sog"][S.row], mu["points"][S.row]).astype(np.float32),
                          p_script=S.p_base.values.astype(np.float32), p_model=S.p_model.values.astype(np.float32),
                          p=S.p_cal.values.astype(np.float32), hit=S.hit.values.astype(np.int8),
                          base_pos=S.base_pos.values.astype(np.float32), gp_prior=S.gp_prior.values.astype(np.int16)))
    S.to_parquet(OUT / "nhl_legs.parquet", index=False, compression="zstd")
    pr.to_csv(OUT / "nhl_real_lines.csv", index=False)
    state = dict(rho={str(k): v for k, v in rho.items()},
                 dispersion={st: {str(s): [list(map(float, e[1:-1])), list(map(float, rs))] for s, (e, rs) in disp[st].items()}
                             for st in disp},
                 sgp_state_reference={k: v for k, v in json.load(open(RS / "sgp_state.json"))["rho"].items() if k.startswith("nhl")})
    json.dump(state, open(OUT / "nhl_state.json", "w"), indent=1)
    secs = time.time() - T0
    write_report(dict(tiers_all=tiers_all, tiers_by=tiers_by, tiers_raw=tiers_raw, lc=lc, abl=abl, abl_cal=abl_cal,
                      rl=rl, pr=pr, PS=PS, ml_avail=ml_avail, rho=rho, imps=imps, M=M, secs=secs, P=P))
    log(f"  done in {secs/60:.1f} min")


def ablation(d, base_mu_r, log):
    rows = []
    test = (d.season >= REPORT_FROM).values & (d.gp_prior >= MIN_GP).values
    trn = (d.season < REPORT_FROM).values
    off = offsets(d)

    def evaluate(name, mus, rs):
        ps, ys = [], []
        for st, k in MILESTONES:
            ps.append(C.nb_ge(k, mus[st][test], rs[st][test])); ys.append((d[st].values[test] >= k).astype(float))
        p, y = np.concatenate(ps), np.concatenate(ys)
        rows.append(dict(variant=name, ll=C.ll(p, y), brier=float(((p - y) ** 2).mean()), n_ge80=int((p >= .8).sum()),
                         hit_ge80=float(y[p >= .8].mean()) if (p >= .8).any() else np.nan,
                         n_ge90=int((p >= .9).sum()), hit_ge90=float(y[p >= .9].mean()) if (p >= .9).any() else np.nan))

    def fit_variant(groups):
        mus, rs = {}, {}
        for st in ("sog", "points"):
            x = d.assign(log_mu=off[st])
            f = feats_for(st, groups)
            m = C.fit_offset_gbm(x.loc[trn, f], d.loc[trn, st].values, off[st].values[trn], rounds=150)
            mu = C.gbm_mu(m, x[f], off[st].values)
            e, r = C.fit_r_by_mu(d[st].values[trn], mu[trn])     # in-sample on train (same for every variant)
            mus[st], rs[st] = mu, C.apply_r(mu, e, r)
        return mus, rs

    # script only
    evaluate("script only (share x scripted team volume)", {k: v[0] for k, v in base_mu_r.items()},
             {k: v[1] for k, v in base_mu_r.items()})
    evaluate("full tendencies", *fit_variant(GROUPS))
    for gname in GROUPS:
        g = {k: v for k, v in GROUPS.items() if k != gname}
        evaluate(f"  - without {gname}", *fit_variant(g))
        log(f"    ablation {gname}")
    return pd.DataFrame(rows)


def write_report(R):
    L = ["# NHL script-driven props & parlays\n",
         f"Run time {R['secs']/60:.1f} min. Walk-forward: tendencies layer refit every season on prior seasons only; "
         f"online calibrator updated after every date; correlations from prior seasons only. Report = seasons "
         f"{REPORT_FROM}+ (season = year it ends); players with >= {MIN_GP} prior games.\n",
         "## A. Real lines (2026-27, Oct 1-13 2026 only — SMALL SAMPLE)\n",
         "Pick side = higher calibrated model probability. `mkt_implied_same_side` = de-vigged market probability of the "
         "side we picked (best price across books).\n", C.md(R["rl"]), "",
         "## B. Milestones (X+ ladders), calibrated model probability, report seasons\n",
         "`base` = naive league rate for that leg and position group (F/D) from prior seasons; `base_player` = the "
         "player's own last-10-game hit rate (prior games).\n", C.md(R["tiers_all"]), "",
         "### By leg\n", C.md(R["tiers_by"]), "",
         "### Same tiers on raw model probabilities (before the online calibrator)\n", C.md(R["tiers_raw"]), "",
         "## C. Parlays (one ticket per construction per day, report seasons)\n",
         f"Days with an ML >= 85%: {R['ml_avail']['ml85']} of {R['ml_avail']['days']}; >= 70%: {R['ml_avail']['ml70']}. "
         "Constructions: `top` = N most likely legs across the day (max 1 leg per player); `sgp` = script-correlated "
         "same-game stack (greedy on copula joint; ML and game-total legs allowed; fills into the next game when one "
         "runs out); `mixed_mlXX` = best game ML >= XX% + top N-1 props; `best_of_day` = highest predicted joint among "
         "them. `pred_joint` uses learned correlations; `pred_indep` = product of legs. `typical_price` = fair odds of "
         "the median predicted joint.\n", C.md(R["PS"]), "",
         "## D. Ablation of tendencies (train seasons <= 2020, test 2021+, raw probabilities, milestone legs)\n",
         C.md(R["abl"], ".4f"), "", C.md(R["abl_cal"], ".4f"), "",
         "## E. Learning curve (milestone legs, by season)\n", C.md(R["lc"], ".4f"), "",
         "## Learned latent correlations (latest season table)\n",
         C.md(pd.DataFrame(sorted(R["rho"][max(R["rho"])].items()), columns=["pair", "rho"]), ".3f"), "",
         "## Feature importance (gain, last refit)\n"]
    for st, imp in R["imps"].items():
        if imp is not None:
            L.append(f"**{st}**: " + ", ".join(f"{k} {v/imp.sum():.2f}" for k, v in imp.sort_values(ascending=False).head(10).items()) + "\n")
    (OUT / "nhl_props.md").write_text("\n".join(L))


# ------------------------------------------------------------------ tonight
def build_tickets(date, sizes=(3, 5, 7)):
    """3/5/7-leg tickets for a date from the saved legs (results/props_script/nhl_legs.parquet) and saved state.
    Legs exist for dates whose player script rows exist (lineups known); outcomes of future games are NaN."""
    date = pd.Timestamp(date)
    legs = pd.read_parquet(OUT / "nhl_legs.parquet", filters=[("date", "==", date)])
    if legs.empty:
        return {"date": str(date.date()), "tickets": [], "note": "no scripted player rows for this date"}
    st = json.load(open(OUT / "nhl_state.json"))
    s = int(legs.season.iloc[0])
    rho = C.RhoBook(st["rho"].get(str(s), st["rho"][max(st["rho"], key=int)]))
    legs = legs[legs.gp_prior >= MIN_GP]
    props = pd.DataFrame(dict(date=legs.date, season=legs.season, gid=legs.game_id.astype(str), team=legs.team.astype(str),
                              team_home=legs.home, ent=legs.player_id, stat=legs.stat.astype(str), kind="prop", sign=1,
                              label=legs.player.astype(str) + " " + legs.stat.astype(str).map(STAT_LABEL) + " "
                              + legs.threshold.astype(str) + "+", p=legs.p.astype(float), hit=legs.hit.astype(float)))
    ml, tot = game_leg_frames(s)
    ml, tot = ml[ml.date == date], tot[tot.date == date]
    props = props.sort_values("p", ascending=False).groupby("ent").head(2)
    res = C.tickets_for_day(C.to_leg_dicts(props), C.to_leg_dicts(ml), C.to_leg_dicts(tot), rho, sizes=sizes)
    return {"date": str(date.date()), "tickets": res}


if __name__ == "__main__":
    run(quick="quick" in sys.argv)
