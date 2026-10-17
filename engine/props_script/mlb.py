"""
MLB script-driven starting-pitcher strikeout props (3+..7+ ladders, real 2026 K lines) and 3/5/7-leg parlays.

  python -m engine.props_script.mlb            # full walk-forward run + report

Pipeline (strictly walk-forward):
  1. script mean   results/scripts/mlb_players.parquet: K mu = expected batters faced x per-PA K probability
                   (pitcher K rate x opponent K rate), already coupled to the game script.
  2. tendencies    LightGBM Poisson on top of log(script mu), refit every season on prior seasons only. Every
                   tendency is shift-safe (prior games / prior dates only), from Statcast pitcher games:
                   leash (pitch counts last start / last 3 / EWMA, batters faced last 5 / season, innings depth,
                   days of rest, starts this season, own bullpen usage last 3 days), pitcher skill (K/BF last 5 /
                   season / career shrunk, whiff and called+swinging-strike rates per pitch), recent form,
                   opponent K% vs the pitcher's handedness (season-to-date, shrunk to the prior season) and last
                   30 days, park strikeout factor (venue K/BF vs league, all prior dates), home/away, game script
                   (expected total, own win probability, opponent / own expected runs). Umpire: not available.
  3. dispersion    NB size by mu-bin from prior seasons' out-of-sample residuals.
  4. calibrator    online (threshold) Platt + confidence-bin residual table, updated after every date.
  5. parlays       top-N cross-game / script-correlated same-game stacks / ML + props.
Fit / choose on seasons <= 2020, report 2021+.
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

from engine.props_script import common_count as C

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RS, RP, PROC = ROOT / "results" / "scripts", ROOT / "results" / "predict", ROOT / "data" / "processed"
OUT = ROOT / "results" / "props_script"
FIRST_TEST, REPORT_FROM = 2018, 2021
MIN_GP = 3
KS = [2, 3, 4, 5, 6, 7, 8, 9]
MILE = [3, 4, 5, 6, 7]
LG_K = 0.22

GROUPS = {
    "leash": ["pitches_last", "pitches_l3", "pitches_ewm", "bf_l5", "bf_szn", "depth_l5", "days_rest", "starts_szn",
              "own_pen3", "bf_exp"],
    "pitcher_skill": ["k_rate_l5", "k_rate_szn", "k_rate_car", "whiff_l5", "whiff_car", "csw_l5", "csw_car"],
    "form": ["k_l3", "k_trend"],
    "opponent": ["opp_k_hand", "opp_k30", "throws_l", "opp_k"],
    "park": ["park_k"],
    "home": ["home"],
    "game_script": ["g_total", "p_win_own", "mu_opp_runs", "mu_own_runs"],
}
BASE_FEATS = ["log_mu", "n_prior_starts", "k_rate", "p_k"]


def norm_name(s: str) -> str:
    s = str(s)
    if "," in s:
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower())
    return re.sub(r"[^a-z]", "", s)


def _lag(df, key, col, fn):
    return df.groupby(key, sort=False)[col].transform(lambda s: fn(s.shift()))


def _prior_by_date(daily, keys, cols):
    """daily: one row per (keys, date) with sums; returns cumulative sums over strictly earlier dates."""
    daily = daily.sort_values("date")
    g = daily.groupby(keys, sort=False)
    for c in cols:
        daily[f"{c}_cum"] = g[c].cumsum() - daily[c]
    return daily


# ------------------------------------------------------------------ data
def load(log=print):
    t0 = time.time()
    g = pd.read_csv(PROC / "mlb_statcast_pitcher_games.csv.gz", parse_dates=["game_date"]).rename(columns={"game_date": "date"})
    g["season"] = g.date.dt.year
    g["venue"] = np.where(g.home == 1, g.pitch_team, g.opp)
    # ---- opponent (batting team) K% vs pitcher hand: season-to-date (prior dates) shrunk to prior season
    oh = g.groupby(["opp", "p_throws", "season", "date"])[["k", "bf"]].sum().reset_index()
    oh = _prior_by_date(oh, ["opp", "p_throws", "season"], ["k", "bf"])
    tot = oh.groupby(["opp", "p_throws", "season"])[["k", "bf"]].sum().reset_index()
    tot["prev_rate"] = tot.k / tot.bf
    tot["season"] += 1
    oh = oh.merge(tot[["opp", "p_throws", "season", "prev_rate"]], on=["opp", "p_throws", "season"], how="left")
    oh["opp_k_hand"] = (oh.k_cum + oh.prev_rate.fillna(LG_K) * 250) / (oh.bf_cum + 250)
    # last 30 days (both hands)
    od = g.groupby(["opp", "date"])[["k", "bf"]].sum().reset_index().sort_values("date")
    def roll30(x):
        x = x.set_index("date")
        return pd.DataFrame({"opp_k30": ((x.k.rolling("30D").sum() - x.k) / (x.bf.rolling("30D").sum() - x.bf)).values},
                            index=x.index)
    o30 = od.groupby("opp", group_keys=True).apply(roll30).reset_index()
    # ---- park K factor: venue K/BF over all prior dates relative to league over the same dates (shrunk)
    vd = g.groupby(["venue", "date"])[["k", "bf"]].sum().reset_index()
    vd = _prior_by_date(vd, ["venue"], ["k", "bf"])
    lg = g.groupby("date")[["k", "bf"]].sum().sort_index()
    lg["lg_rate"] = (lg.k.cumsum() - lg.k) / (lg.bf.cumsum() - lg.bf)
    vd = vd.merge(lg[["lg_rate"]].reset_index(), on="date", how="left")
    vd["park_k"] = ((vd.k_cum + vd.lg_rate * 3000) / (vd.bf_cum + 3000)) / vd.lg_rate
    # ---- starter tendencies
    s = g[g.starter == 1].sort_values(["date", "game_pk"]).reset_index(drop=True)
    key = "pitcher"
    for c in ("k", "bf", "pitches", "whiffs", "csw"):
        s[f"{c}_l5"] = _lag(s, key, c, lambda x: x.rolling(5, min_periods=1).sum())
        s[f"{c}_car"] = _lag(s, key, c, lambda x: x.expanding().sum())
    gs = s.groupby([key, "season"], sort=False)
    s["k_sz"] = gs.k.transform(lambda x: x.shift().expanding().sum())
    s["bf_sz"] = gs.bf.transform(lambda x: x.shift().expanding().sum())
    s["starts_szn"] = gs.cumcount()
    s["k_rate_car"] = (s.k_car.fillna(0) + LG_K * 150) / (s.bf_car.fillna(0) + 150)
    s["k_rate_l5"] = (s.k_l5.fillna(0) + LG_K * 40) / (s.bf_l5.fillna(0) + 40)
    s["k_rate_szn"] = (s.k_sz.fillna(0) + LG_K * 80) / (s.bf_sz.fillna(0) + 80)
    s["whiff_car"] = (s.whiffs_car.fillna(0) + 0.11 * 600) / (s.pitches_car.fillna(0) + 600)
    s["csw_car"] = (s.csw_car.fillna(0) + 0.29 * 600) / (s.pitches_car.fillna(0) + 600)
    s["whiff_l5"] = (s.whiffs_l5.fillna(0) + 0.11 * 200) / (s.pitches_l5.fillna(0) + 200)
    s["csw_l5"] = (s.csw_l5.fillna(0) + 0.29 * 200) / (s.pitches_l5.fillna(0) + 200)
    s["pitches_last"] = _lag(s, key, "pitches", lambda x: x)
    s["pitches_l3"] = _lag(s, key, "pitches", lambda x: x.rolling(3, min_periods=1).mean())
    s["pitches_ewm"] = _lag(s, key, "pitches", lambda x: x.ewm(halflife=5, min_periods=1).mean())
    s["bf_l5"] = _lag(s, key, "bf", lambda x: x.rolling(5, min_periods=1).mean())
    s["bf_szn"] = gs.bf.transform(lambda x: x.shift().expanding().mean())
    s["depth_l5"] = _lag(s, key, "max_inning", lambda x: x.rolling(5, min_periods=1).mean())
    s["days_rest"] = s.groupby(key).date.diff().dt.days.clip(upper=30)
    s["k_l3"] = _lag(s, key, "k", lambda x: x.rolling(3, min_periods=1).mean())
    s["k_l10"] = _lag(s, key, "k", lambda x: x.rolling(10, min_periods=1).mean())
    s["k_trend"] = s.k_l3 - s.k_l10
    s["throws_l"] = (s.p_throws == "L").astype(int)
    s["gp_prior"] = s.groupby(key).cumcount()
    for k in KS:
        s[f"_h{k}"] = (s.k >= k).astype(float)
        s[f"bp_{k}"] = _lag(s, key, f"_h{k}", lambda x: x.rolling(10, min_periods=3).mean())
    s = s.drop(columns=[c for c in s.columns if c.startswith("_h")])
    s = s.merge(oh[["opp", "p_throws", "date", "opp_k_hand"]], on=["opp", "p_throws", "date"], how="left")
    s = s.merge(o30, on=["opp", "date"], how="left")
    s = s.merge(vd[["venue", "date", "park_k"]], on=["venue", "date"], how="left")
    # ---- script layer
    pl = pd.read_parquet(RS / "mlb_players.parquet")
    d = pl.merge(s.drop(columns=["season", "home", "opp"]).rename(columns={"pitcher": "pitcher_id"}),
                 on=["game_pk", "pitcher_id"], how="inner", suffixes=("", "_sc"))
    d["date"] = pd.to_datetime(d.date)
    gm = pd.read_parquet(RS / "mlb_games.parquet")
    cols = ["game_id", "home", "total", "mu_home", "mu_away", "p_home_blend", "h_opp_pen3", "a_opp_pen3", "margin",
            "home_score", "away_score"]
    d = d.merge(gm[cols].rename(columns={"home": "home_team"}), on="game_id", how="left")
    d["home"] = (d.team == d.home_team).astype(int)
    h = d.home == 1
    d["g_total"] = d.total
    d["p_win_own"] = np.where(h, d.p_home_blend, 1 - d.p_home_blend)
    d["mu_own_runs"] = np.where(h, d.mu_home, d.mu_away)
    d["own_pen3"] = np.where(h, d.a_opp_pen3, d.h_opp_pen3)        # home side's opponent pen = away pen
    d["margin_own_pred"] = np.where(h, d.margin, -d.margin)
    d["margin_own_act"] = np.where(h, d.home_score - d.away_score, d.away_score - d.home_score)
    d["total_act"] = d.home_score + d.away_score
    d["season"] = d.date.dt.year
    d = d.sort_values(["date", "game_id", "pitcher_id"]).reset_index(drop=True)
    log(f"  model frame {len(d)} starts, {d.season.min()}-{d.season.max()} ({time.time()-t0:.0f}s)")
    return d


def feats_for(groups=GROUPS):
    f = list(BASE_FEATS)
    for v in groups.values():
        f += v
    return f


def dispersion(d, mu):
    out = {}
    y = d.actual.values.astype(float)
    for s in sorted(d.season.unique()):
        tr = (d.season < s).values
        if tr.sum() < 3000:
            tr = np.ones(len(d), bool) & (d.season <= d.season.min() + 1).values
        out[s] = C.fit_r_by_mu(y[tr], mu[tr], nbins=6)
    return out


def make_legs(d, mu, r):
    parts = []
    for k in KS:
        parts.append(pd.DataFrame(dict(row=np.arange(len(d)), k=k, p_base=C.nb_ge(k, d.mu.values, d.nb_r.values),
                                       p_model=C.nb_ge(k, mu, r), hit=(d.actual.values >= k).astype(float),
                                       base_player=d[f"bp_{k}"].values)))
    L = pd.concat(parts, ignore_index=True)
    L["date"] = d.date.values[L.row]
    L["season"] = d.season.values[L.row]
    L["gp_prior"] = d.gp_prior.values[L.row]
    rate = L.groupby(["k", "season"]).hit.agg(["sum", "size"]).reset_index().sort_values("season")
    rate["base_pos"] = (rate.groupby("k")["sum"].cumsum() - rate["sum"]) / \
                       (rate.groupby("k")["size"].cumsum() - rate["size"]).replace(0, np.nan)
    L = L.merge(rate[["k", "season", "base_pos"]], on=["k", "season"], how="left")
    L["base_player"] = L.base_player.fillna(L.base_pos)
    L["key"] = L.k - KS[0]
    return L


def rho_book(d, mu, r):
    x = d[["season", "game_id", "team"]].copy()
    x["z_k"] = C.pit_z(d.actual.values, mu, r, seed=3)
    em = d.margin_own_act - d.margin_own_pred
    et = d.total_act - d.total
    x["z_m"] = ((em - em.mean()) / em.std()).values
    x["z_t"] = ((et - et.mean()) / et.std()).values
    mom = C.pair_moments(x, "season", "game_id", "team", ["z_k"])
    mom.update(C.row_moments(x, "season", "z_k", "z_m", "ml|k"))
    mom.update(C.row_moments(x, "season", "z_k", "z_t", "tot|k"))
    hm = x[d.home.values == 1].drop_duplicates("game_id")
    mom.update(C.row_moments(hm, "season", "z_m", "z_t", "mltot"))
    seasons = sorted(d.season.unique()) + [int(d.season.max()) + 1]
    return C.rho_tables_by_season(mom, seasons)


def real_lines(d, L):
    pr = pd.read_csv(PROC / "mlb_props.csv", parse_dates=["date"])
    pr = pr[pr.result.isin(["over", "under"])].dropna(subset=["line", "actual"])
    raw_n = pr.groupby(["date", "player"]).ngroups
    # one line per pitcher-start: morning snapshot, DraftKings first, then FanDuel
    pr["_o"] = (pr.snapshot != "am").astype(int) * 2 + (pr.book != "draftkings").astype(int)
    pr = pr.sort_values("_o").drop_duplicates(["date", "player"]).drop(columns="_o")
    pr["key"] = pr.player.map(norm_name)
    idx = d[["date", "player"]].reset_index().rename(columns={"index": "row"})
    idx["key"] = idx.player.map(norm_name)
    pr = pr.merge(idx[["date", "key", "row"]], on=["date", "key"], how="left")
    unmatched = int(pr.row.isna().sum())
    pr = pr.dropna(subset=["row"])
    pr["row"] = pr.row.astype(int)
    pr["k"] = (np.floor(pr.line) + 1).astype(int)
    pr = pr.merge(L[["row", "k", "p_cal", "p_model", "p_base"]], on=["row", "k"], how="inner")
    from engine.scripts.sgp_builder import fair2
    pr["p_mkt"] = fair2(pr.over_odds.values, pr.under_odds.values)
    pr["over"] = (pr.actual > pr.line).astype(float)
    pr["pick_over"] = pr.p_cal >= 0.5
    pr["p_pick"] = np.where(pr.pick_over, pr.p_cal, 1 - pr.p_cal)
    pr["hit"] = np.where(pr.pick_over, pr.over, 1 - pr.over)
    pr["mkt_pick_over"] = pr.p_mkt >= 0.5
    pr["mkt_hit"] = np.where(pr.mkt_pick_over, pr.over, 1 - pr.over)
    pr["p_mkt_side"] = np.where(pr.pick_over, pr.p_mkt, 1 - pr.p_mkt)
    pr["window"] = np.where(pr.date < "2026-08-20", "first half", "second half")
    return pr, dict(raw_pitcher_dates=raw_n, unmatched=unmatched, matched=len(pr))


def real_line_table(pr):
    rows = []
    bands = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.01)]
    for grp, x in [("all", pr)] + list(pr.groupby("window")):
        for lo, hi in bands:
            s = x[(x.p_pick >= lo) & (x.p_pick < hi)]
            if len(s):
                rows.append(dict(window=grp, tier=f"{int(lo*100)}-{int(min(hi,1)*100)}%", n=len(s), pred=s.p_pick.mean(),
                                 hit=s.hit.mean(), mkt_implied_same_side=s.p_mkt_side.mean(),
                                 flag="SMALL" if len(s) < 100 else ""))
        rows.append(dict(window=grp, tier="ALL picks", n=len(x), pred=x.p_pick.mean(), hit=x.hit.mean(),
                         mkt_implied_same_side=x.p_mkt_side.mean(), flag="SMALL" if len(x) < 100 else ""))
        rows.append(dict(window=grp, tier="market-favoured side (baseline)", n=len(x),
                         pred=np.where(x.mkt_pick_over, x.p_mkt, 1 - x.p_mkt).mean(), hit=x.mkt_hit.mean(),
                         mkt_implied_same_side=np.nan, flag="SMALL" if len(x) < 100 else ""))
        agree = x[x.pick_over == x.mkt_pick_over]
        rows.append(dict(window=grp, tier="model & market agree", n=len(agree), pred=agree.p_pick.mean(),
                         hit=agree.hit.mean(), mkt_implied_same_side=agree.p_mkt_side.mean(),
                         flag="SMALL" if len(agree) < 100 else ""))
    return pd.DataFrame(rows)


def prop_leg_frame(d, L):
    m = L[L.k.isin(MILE) & (L.gp_prior >= MIN_GP)]
    return pd.DataFrame(dict(date=m.date.values, season=m.season.values, gid=d.game_id.values[m.row],
                             team=d.team.values[m.row], team_home=d.home.values[m.row], ent=d.pitcher_id.values[m.row],
                             stat="k", kind="prop", sign=1,
                             label=[f"{n} {k}+ K" for n, k in zip(d.player.values[m.row], m.k)],
                             p=m.p_cal.values, hit=m.hit.values))


def game_leg_frames(seasons_from):
    p = pd.read_parquet(RP / "mlb_preds.parquet")
    p = p[p.season >= seasons_from]
    ml = pd.DataFrame(dict(date=p.date, season=p.season, gid=p.game_id.astype(str), team=p.pick, team_home=p.pick_home,
                           ent="ML" + p.game_id.astype(str), stat="ml", kind="ml", sign=1, label="ML " + p.pick,
                           p=p.pick_prob, hit=p.pick_hit))
    t = p.dropna(subset=["total_prob"])
    tot = pd.DataFrame(dict(date=t.date, season=t.season, gid=t.game_id.astype(str), team=None, team_home=0,
                            ent="TOT" + t.game_id.astype(str), stat="total", kind="total",
                            sign=np.where(t.total_pick == "OVER", 1, -1),
                            label=t.total_pick + " " + t.tl.astype(str) + " " + t.away + "@" + t.home,
                            p=t.total_prob, hit=t.total_hit))
    return ml, tot


def ablation(d, log):
    rows = []
    test = (d.season >= REPORT_FROM).values & (d.gp_prior >= MIN_GP).values
    trn = (d.season < REPORT_FROM).values
    off = np.log(d.mu.clip(lower=0.1)).values
    y = d.actual.values.astype(float)

    def evaluate(name, mu, r):
        p = np.concatenate([C.nb_ge(k, mu[test], r[test]) for k in MILE])
        yy = np.concatenate([(y[test] >= k).astype(float) for k in MILE])
        rows.append(dict(variant=name, ll=C.ll(p, yy), brier=float(((p - yy) ** 2).mean()),
                         mae_k=float(np.abs(y[test] - mu[test]).mean()),
                         n_ge80=int((p >= .8).sum()), hit_ge80=float(yy[p >= .8].mean()) if (p >= .8).any() else np.nan,
                         n_ge90=int((p >= .9).sum()), hit_ge90=float(yy[p >= .9].mean()) if (p >= .9).any() else np.nan))

    def fit(groups):
        f = feats_for(groups)
        x = d.assign(log_mu=off)
        m = C.fit_offset_gbm(x.loc[trn, f], y[trn], off[trn], rounds=200, params=dict(min_data_in_leaf=100))
        mu = C.gbm_mu(m, x[f], off)
        e, rr = C.fit_r_by_mu(y[trn], mu[trn], nbins=6)
        return mu, C.apply_r(mu, e, rr)

    e, rr = C.fit_r_by_mu(y[trn], d.mu.values[trn], nbins=6)
    evaluate("script only, file dispersion", d.mu.values, d.nb_r.values)
    evaluate("script only, refit dispersion", d.mu.values, C.apply_r(d.mu.values, e, rr))
    evaluate("full tendencies", *fit(GROUPS))
    for gname in GROUPS:
        evaluate(f"  - without {gname}", *fit({k: v for k, v in GROUPS.items() if k != gname}))
    log("    ablation done")
    return pd.DataFrame(rows)


def run(log=print):
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(log)
    off = np.log(d.mu.clip(lower=0.1))
    x = d.assign(log_mu=off, _off=off)
    mu, imp = C.walk_forward_gbm(x, feats_for(), "actual", "_off", "season", FIRST_TEST, rounds=200,
                                 params=dict(min_data_in_leaf=100), log=lambda *a: None)
    disp = dispersion(d, mu)
    r = np.empty(len(d))
    for s, (e, rs) in disp.items():
        m = (d.season == s).values
        r[m] = C.apply_r(mu[m], e, rs)
    log(f"  fitted ({time.time()-T0:.0f}s)")
    L = make_legs(d, mu, r)
    dk = L.date.values.astype("datetime64[D]").astype(int)
    L["p_cal"], L["p_platt"], _ = C.walk_calibrate(dk, L.key.values, L.p_model.values, L.hit.values, n0=60.0)
    L["p_base_cal"], _, _ = C.walk_calibrate(dk, L.key.values, L.p_base.values, L.hit.values, n0=60.0)
    rho = rho_book(d, mu, r)
    rho_fn = lambda s: C.RhoBook(rho.get(s, rho[max(rho)]))

    M = L[L.k.isin(MILE) & (L.season >= REPORT_FROM) & (L.gp_prior >= MIN_GP)].copy()
    M["leg"] = M.k.astype(str) + "+ K"
    tiers_all = C.tier_table(M, "p_cal", "hit", "base_pos", "base_player")
    tiers_by = C.tier_table(M, "p_cal", "hit", "base_pos", "base_player", tiers=[0.70, 0.80, 0.90], group="leg")
    tiers_raw = C.tier_table(M, "p_model", "hit", "base_pos", "base_player")
    lc = []
    for s, xx in L[L.k.isin(MILE) & (L.gp_prior >= MIN_GP)].groupby("season"):
        hi = xx[xx.p_cal >= 0.8]
        lc.append(dict(season=s, n_legs=len(xx), ll_script=C.ll(xx.p_base, xx.hit), ll_model=C.ll(xx.p_model, xx.hit),
                       ll_cal=C.ll(xx.p_cal, xx.hit), ece_script=C.ece(xx.p_base, xx.hit),
                       ece_model=C.ece(xx.p_model, xx.hit), ece_cal=C.ece(xx.p_cal, xx.hit), n_ge80=len(hi),
                       pred_ge80=hi.p_cal.mean(), hit_ge80=hi.hit.mean()))
    lc = pd.DataFrame(lc)
    abl = ablation(d, log)
    abl_cal = pd.DataFrame([
        dict(variant="script only + calibrator", ll=C.ll(M.p_base_cal, M.hit), brier=float(((M.p_base_cal - M.hit) ** 2).mean()),
             n_ge80=int((M.p_base_cal >= .8).sum()), hit_ge80=M.hit[M.p_base_cal >= .8].mean()),
        dict(variant="full walk-forward + calibrator", ll=C.ll(M.p_cal, M.hit), brier=float(((M.p_cal - M.hit) ** 2).mean()),
             n_ge80=int((M.p_cal >= .8).sum()), hit_ge80=M.hit[M.p_cal >= .8].mean())])
    pr, rl_info = real_lines(d, L)
    rl = real_line_table(pr)
    log(f"  evaluated ({time.time()-T0:.0f}s); parlays...")
    props = prop_leg_frame(d, L[L.season >= REPORT_FROM])
    ml, tot = game_leg_frames(REPORT_FROM)
    P = C.run_parlays(props, ml, tot, rho_fn, log=log)
    P.to_csv(OUT / "mlb_parlays.csv", index=False)
    PS = C.parlay_summary(P)
    ml_avail = dict(ml85=int((ml.groupby("date").p.max() >= 0.85).sum()), ml70=int((ml.groupby("date").p.max() >= 0.70).sum()),
                    days=int(props.date.nunique()))
    S = L[L.k.isin(MILE) & (L.season >= FIRST_TEST)]
    S = pd.DataFrame(dict(date=S.date.values, season=S.season.values.astype(np.int16), game_id=d.game_id.values[S.row],
                          pitcher_id=d.pitcher_id.values[S.row], player=d.player.values[S.row], team=d.team.values[S.row],
                          home=d.home.values[S.row].astype(np.int8), stat="strikeouts", threshold=S.k.values.astype(np.int8),
                          side="over", mu=mu[S.row].astype(np.float32), p_script=S.p_base.values.astype(np.float32),
                          p_model=S.p_model.values.astype(np.float32), p=S.p_cal.values.astype(np.float32),
                          hit=S.hit.values.astype(np.int8), base_pos=S.base_pos.values.astype(np.float32),
                          gp_prior=S.gp_prior.values.astype(np.int16)))
    S.to_parquet(OUT / "mlb_legs.parquet", index=False, compression="zstd")
    pr.to_csv(OUT / "mlb_real_lines.csv", index=False)
    json.dump(dict(rho={str(k): v for k, v in rho.items()},
                   sgp_state_reference={k: v for k, v in json.load(open(RS / "sgp_state.json"))["rho"].items() if k.startswith("mlb")}),
              open(OUT / "mlb_state.json", "w"), indent=1)
    secs = time.time() - T0
    write_report(dict(tiers_all=tiers_all, tiers_by=tiers_by, tiers_raw=tiers_raw, lc=lc, abl=abl, abl_cal=abl_cal, rl=rl,
                      rl_info=rl_info, PS=PS, ml_avail=ml_avail, rho=rho, imp=imp, secs=secs))
    log(f"  done in {secs/60:.1f} min")


def write_report(R):
    imp = R["imp"]
    L = ["# MLB script-driven strikeout props & parlays\n",
         f"Run time {R['secs']/60:.1f} min. Walk-forward: tendencies layer refit every season on prior seasons only; online "
         f"calibrator updated after every date; correlations from prior seasons only. Report = seasons {REPORT_FROM}+; "
         f"starters with >= {MIN_GP} prior starts.\n",
         "## A. Real lines (2026 K lines, Jul 10 - Sep 27 matched — SMALL SAMPLE)\n",
         f"Pitcher-starts with a line: {R['rl_info']['raw_pitcher_dates']}; matched to a scripted start "
         f"{R['rl_info']['matched']} (unmatched {R['rl_info']['unmatched']}: postseason after Sep 27 / name mismatch). One "
         "line per start (morning snapshot, DraftKings first). The model never saw these lines; `first half` / "
         "`second half` split at 2026-08-20.\n", C.md(R["rl"]), "",
         "## B. Milestones (3+..7+ K), calibrated probability, report seasons\n",
         "`base` = league hit rate of that threshold over prior seasons; `base_player` = pitcher's own last-10-start rate.\n",
         C.md(R["tiers_all"]), "", "### By leg\n", C.md(R["tiers_by"]), "",
         "### Raw model probabilities (before the online calibrator)\n", C.md(R["tiers_raw"]), "",
         "## C. Parlays\n",
         f"Days with an ML >= 85%: {R['ml_avail']['ml85']} of {R['ml_avail']['days']}; >= 70%: {R['ml_avail']['ml70']}. "
         "Same-game MLB stacks hold at most 4 legs (2 starters + ML + total), so 5/7-leg `sgp` tickets continue into the "
         "next most-correlated game.\n", C.md(R["PS"]), "",
         "## D. Ablation (train <= 2020, test 2021+, raw probabilities)\n", C.md(R["abl"], ".4f"), "",
         C.md(R["abl_cal"], ".4f"), "", "## E. Learning curve\n", C.md(R["lc"], ".4f"), "",
         "## Learned latent correlations (latest)\n",
         C.md(pd.DataFrame(sorted(R["rho"][max(R["rho"])].items()), columns=["pair", "rho"])), "",
         "## Feature importance (gain, last refit)\n",
         ", ".join(f"{k} {v/imp.sum():.2f}" for k, v in imp.sort_values(ascending=False).head(15).items()) if imp is not None else ""]
    (OUT / "mlb_props.md").write_text("\n".join(L))


def build_tickets(date, sizes=(3, 5, 7)):
    date = pd.Timestamp(date)
    legs = pd.read_parquet(OUT / "mlb_legs.parquet", filters=[("date", "==", date)])
    if legs.empty:
        return {"date": str(date.date()), "tickets": [], "note": "no scripted starter rows for this date"}
    st = json.load(open(OUT / "mlb_state.json"))
    s = int(legs.season.iloc[0])
    rho = C.RhoBook(st["rho"].get(str(s), st["rho"][max(st["rho"], key=int)]))
    legs = legs[legs.gp_prior >= MIN_GP]
    props = pd.DataFrame(dict(date=legs.date, season=legs.season, gid=legs.game_id, team=legs.team, team_home=legs.home,
                              ent=legs.pitcher_id, stat="k", kind="prop", sign=1,
                              label=legs.player + " " + legs.threshold.astype(str) + "+ K", p=legs.p.astype(float),
                              hit=legs.hit.astype(float)))
    ml, tot = game_leg_frames(s)
    ml, tot = ml[ml.date == date], tot[tot.date == date]
    props = props.sort_values("p", ascending=False).groupby("ent").head(2)
    return {"date": str(date.date()),
            "tickets": C.tickets_for_day(C.to_leg_dicts(props), C.to_leg_dicts(ml), C.to_leg_dicts(tot), rho, sizes=sizes)}


if __name__ == "__main__":
    run()
