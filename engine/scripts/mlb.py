"""MLB script-then-learn adapter (runs per side, count model) + starting-pitcher strikeout layer.

Script for each game (only information before first pitch):
  core          team offense / run-prevention ratings, league level, home edge (base.ScriptEngine, count model)
  sp            starting-pitcher run-prevention rating: Kalman effect on the OPPONENT's runs, updated after each
                start from the runs his side allowed vs the script (innovation)
  park          per home ballpark run effect (both sides), learned online
  RLS features  (online recursive least squares on the per-side residual)
                  opp SP strikeout rate (Statcast, shrunk, decayed) and CSW% vs league; SP has Statcast history;
                  SP new (<3 prior starts); SP days of rest; SP depth (batters faced per start);
                  bullpen fatigue (opp relievers' pitches in previous 3 days, Statcast); played yesterday
  learned blend own script + de-vigged closing moneyline (online logistic), totals via core Bates-Granger weight
Player layer: starter strikeouts = expected batters faced x log5(pitcher K%, opponent K%), coupled to the
scripted opponent runs via an online RLS, negative-binomial counts; tested on 2026 real K lines.

Tuning on 2016-2020 only; reporting 2021+. Run:  python -m engine.scripts.mlb
"""
from __future__ import annotations

import math
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from engine.scripts.base import Params
from engine.scripts.countsport import (AdapterConfig, KalmanComponent, RLS, OnlineLogit, bet_sim, devig_pair, ll,
                                       md_table, run_adapter, season_table, summary, logit, sigmoid)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PROC, LIVE, RES = ROOT / "data" / "processed", ROOT / "data" / "live", ROOT / "results" / "scripts"
SC_MAP = {"AZ": "ARI", "CWS": "CHW"}
LG_K, LG_CSW, LG_BF = 0.215, 0.285, 23.0


def norm(s) -> str | None:
    if s is None or (isinstance(s, float) and s != s):
        return None
    s = str(s)
    if "," in s:
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower())
    return re.sub(r"[^a-z]", "", s) or None


# ------------------------------------------------------------------------------------------- data
def load_statcast() -> pd.DataFrame:
    s = pd.read_csv(PROC / "mlb_statcast_pitcher_games.csv.gz")
    s["pitch_team"] = s.pitch_team.replace(SC_MAP)
    s["opp"] = s.opp.replace(SC_MAP)
    s["date"] = s.game_date
    return s.sort_values(["date", "game_pk"]).reset_index(drop=True)


def load_games(sc: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(PROC / "mlb.csv")
    d["game_id"] = d.game_id.astype(str)
    d["neutral"] = d.neutral_site.fillna(0).astype(int)
    d["order"] = d.dh_game.clip(lower=1)
    l = pd.read_csv(LIVE / "mlb.csv")
    teams = set(d.home)
    first_rs = sc[sc.date >= "2026-01-01"].date.min()
    l = l[(l.date >= first_rs) & l.home.isin(teams) & l.away.isin(teams) & l.completed.astype(bool)].copy()
    l["season"] = 2026
    l["game_id"] = "ESPN" + l.espn_id.astype(str)
    l["neutral"] = l.neutral.fillna(0).astype(int)
    l = l.sort_values(["date", "start"])
    l["order"] = l.groupby(["date", "home"]).cumcount() + 1
    l["home_sp"] = None; l["away_sp"] = None
    cols = ["game_id", "date", "season", "home", "away", "home_score", "away_score", "home_ml", "away_ml", "total",
            "over_odds", "under_odds", "spread_home", "spread_home_odds", "spread_away_odds", "playoff", "home_sp",
            "away_sp", "neutral", "order"]
    g = pd.concat([d[cols], l[cols]], ignore_index=True)
    g = g.dropna(subset=["home_score", "away_score"]).sort_values(["date", "order", "game_id"]).reset_index(drop=True)
    return g


def statcast_starter_features(sc: pd.DataFrame) -> pd.DataFrame:
    """Per Statcast START: the pitcher's pre-start K%, CSW%, BF/start (decayed, shrunk), days rest, n prior starts.
    Uses only that pitcher's earlier starts."""
    st = sc[sc.starter == 1].sort_values(["date", "game_pk"]).copy()
    dec = 0.94
    acc = {}
    rows = []
    for r in st.itertuples(index=False):
        a = acc.get(r.pitcher)
        if a is None:
            a = dict(k=0.0, bf=0.0, csw=0.0, pit=0.0, n=0, nbf=0.0, last=None, kl=0.0, bfl=0.0, wh=0.0, pitl=0.0)
            acc[r.pitcher] = a
        kr = (a["k"] + LG_K * 120) / (a["bf"] + 120)
        cr = (a["csw"] + LG_CSW * 400) / (a["pit"] + 400)
        bfs = (a["nbf"] + LG_BF * 3) / (a["n_w"] + 3) if "n_w" in a else LG_BF
        rest = (pd.Timestamp(r.date) - pd.Timestamp(a["last"])).days if a["last"] else np.nan
        krl = (a["kl"] + LG_K * 200) / (a["bfl"] + 200)
        wr = (a["wh"] + 0.11 * 400) / (a["pitl"] + 400)
        rows.append((r.game_pk, r.pitcher, kr, cr, bfs, a["n"], rest, krl, wr))
        a["kl"] = a["kl"] * 0.985 + r.k; a["bfl"] = a["bfl"] * 0.985 + r.bf
        a["wh"] = a["wh"] * 0.985 + r.whiffs; a["pitl"] = a["pitl"] * 0.985 + r.pitches
        for key, v in (("k", r.k), ("bf", r.bf), ("csw", r.csw), ("pit", r.pitches)):
            a[key] = a[key] * dec + v
        a["nbf"] = a["nbf"] * dec + r.bf
        a["n_w"] = a.get("n_w", 0.0) * dec + 1
        a["n"] += 1
        a["last"] = r.date
    f = pd.DataFrame(rows, columns=["game_pk", "pitcher", "sp_k", "sp_csw", "sp_bf", "sp_nsc", "sp_rest",
                                   "sp_k_long", "sp_whiff"])
    return st.merge(f, on=["game_pk", "pitcher"])


def team_k_rates(sc: pd.DataFrame) -> pd.DataFrame:
    """Batting team's strikeout rate before each date (decayed by game), from all pitchers it faced."""
    td = sc.groupby(["opp", "date"]).agg(k=("k", "sum"), bf=("bf", "sum")).reset_index().sort_values("date")
    out = []
    for t, x in td.groupby("opp"):
        k = bf = 0.0
        for r in x.itertuples(index=False):
            out.append((t, r.date, (k + LG_K * 300) / (bf + 300)))
            k = k * 0.985 + r.k; bf = bf * 0.985 + r.bf
    return pd.DataFrame(out, columns=["team", "date", "bat_k"])


def attach_features(g: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    stf = statcast_starter_features(sc)
    # order statcast games within (date, team) like the schedule's doubleheader order
    gp = sc[["date", "game_pk", "pitch_team"]].drop_duplicates().sort_values(["date", "game_pk"])
    gp["order"] = gp.groupby(["date", "pitch_team"]).cumcount() + 1
    stf = stf.merge(gp, on=["date", "game_pk", "pitch_team"])
    stf_cols = ["date", "pitch_team", "order", "game_pk", "pitcher", "player_name", "sp_k", "sp_csw", "sp_bf", "sp_nsc",
                "sp_rest", "k", "bf"]
    for side in ("home", "away"):
        x = stf[stf_cols].rename(columns={c: f"{side}_sc_{c}" for c in stf_cols[3:]})
        x = x.rename(columns={"pitch_team": side})
        g = g.merge(x, on=["date", side, "order"], how="left")
    # starter identity key: processed names; 2026 from Statcast names translated via 2024-25 alias map
    alias = {}
    for side in ("home", "away"):
        z = g[g[f"{side}_sp"].notna() & g[f"{side}_sc_player_name"].notna()]
        for a, b in zip(z[f"{side}_sc_player_name"].map(norm), z[f"{side}_sp"].map(norm)):
            alias[a] = b
    for side in ("home", "away"):
        k = g[f"{side}_sp"].map(norm)
        k2 = g[f"{side}_sc_player_name"].map(norm).map(lambda v: alias.get(v, v) if v else None)
        g[f"{side}_sp_key"] = k.where(k.notna(), k2)
    # bullpen: relievers' pitches on each of the 3 previous calendar days
    rp = sc[sc.starter == 0].groupby(["pitch_team", "date"]).pitches.sum()
    rpd = rp.to_dict()
    def pen3(team, date):
        dt = pd.Timestamp(date)
        return sum(rpd.get((team, str((dt - pd.Timedelta(days=j)).date())), 0) for j in (1, 2, 3))
    sc_dates = set(sc.date.unique())
    has_sc_season = set(pd.to_datetime(sc.date).dt.year.unique())
    g["has_pen"] = g.season.isin(has_sc_season).astype(int)
    for side in ("home", "away"):
        g[f"{side}_pen3"] = [pen3(t, d) if h else np.nan for t, d, h in zip(g[side], g.date, g.has_pen)]
    # played yesterday (schedule only)
    played = set(zip(g.home, g.date)) | set(zip(g.away, g.date))
    for side in ("home", "away"):
        g[f"{side}_yday"] = [int((t, str((pd.Timestamp(d) - pd.Timedelta(days=1)).date())) in played)
                             for t, d in zip(g[side], g.date)]
    # starter count of prior starts in our schedule (all eras), for the 'new pitcher' flag
    cnt = defaultdict(int)
    for side in ("home", "away"):
        g[f"{side}_sp_prior"] = 0
    hp, ap = [], []
    for kh, ka in zip(g.home_sp_key, g.away_sp_key):
        hp.append(cnt[kh] if kh else 0); ap.append(cnt[ka] if ka else 0)
        if kh: cnt[kh] += 1
        if ka: cnt[ka] += 1
    g["home_sp_prior"], g["away_sp_prior"] = hp, ap
    # team batting K rate
    tk = team_k_rates(sc)
    for side in ("home", "away"):
        g = g.merge(tk.rename(columns={"team": side, "bat_k": f"{side}_bat_k"}), on=["date", side], how="left")
    return g


FEATS = ["opp_sp_k", "opp_sp_csw", "opp_sp_sc", "opp_sp_new", "opp_sp_rest", "opp_sp_depth", "opp_pen3",
         "opp_yday", "own_yday", "own_bat_k"]
GROUPS = {"statcast_sp": ["opp_sp_k", "opp_sp_csw", "opp_sp_sc", "opp_sp_depth"],
          "sp_context": ["opp_sp_new", "opp_sp_rest"],
          "bullpen": ["opp_pen3", "opp_yday", "own_yday"],
          "lineup_k": ["own_bat_k"]}


def build_side_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for side, opp, pre in (("home", "away", "h"), ("away", "home", "a")):
        has = g[f"{opp}_sc_sp_nsc"].fillna(0) > 0
        g[f"{pre}_opp_sp_k"] = np.where(has, (g[f"{opp}_sc_sp_k"] - LG_K) * 10, 0.0)
        g[f"{pre}_opp_sp_csw"] = np.where(has, (g[f"{opp}_sc_sp_csw"] - LG_CSW) * 10, 0.0)
        g[f"{pre}_opp_sp_depth"] = np.where(has, (g[f"{opp}_sc_sp_bf"] - LG_BF) / 5, 0.0)
        g[f"{pre}_opp_sp_sc"] = has.astype(float)
        g[f"{pre}_opp_sp_new"] = (g[f"{opp}_sp_prior"] < 3).astype(float)
        rest = g[f"{opp}_sc_sp_rest"]
        g[f"{pre}_opp_sp_rest"] = np.where(rest.notna(), (rest.clip(3, 10) - 5) / 5, 0.0)
        g[f"{pre}_opp_pen3"] = np.where(g.has_pen == 1, (g[f"{opp}_pen3"].fillna(0) - 150) / 100, 0.0)
        g[f"{pre}_opp_yday"] = g[f"{opp}_yday"].astype(float)
        g[f"{pre}_own_yday"] = g[f"{side}_yday"].astype(float)
        g[f"{pre}_own_bat_k"] = ((g[f"{side}_bat_k"] - LG_K) * 10).fillna(0.0)
    # Kalman component keys: sp acting on HOME runs = AWAY starter
    g["sp_h_key"], g["sp_a_key"] = g.away_sp_key, g.home_sp_key
    park = g.home.where(~((g.home == "ATH") & (g.season >= 2025)), "ATH_SAC")
    park = park.where(~((park == "TB") & (g.season == 2025)), "TB_2025")
    park = park.where(g.neutral == 0, None)
    g["park_h_key"], g["park_a_key"] = park, park
    pm = devig_pair(g.home_ml, g.away_ml)
    g["p_mkt"] = np.where(g.home_ml.notna() & g.away_ml.notna(), pm, np.nan)
    g["total_line"] = g.total
    return g


def make_cfg(feats=FEATS, use_sp=True, use_park=True, use_rls=True, q_game=0.0004, prior_var=0.12, sp_var=0.35,
             sp_gain=1.0) -> AdapterConfig:
    comps = {}
    if use_sp:
        comps["sp"] = KalmanComponent(prior_mean=0.15, prior_var=sp_var, q=0.002, carry=0.85, gain_scale=sp_gain)
    if use_park:
        comps["park"] = KalmanComponent(prior_mean=0.0, prior_var=0.25, q=0.0, carry=0.95)
    p = Params(init_level=4.4, init_hfa=0.15, init_sd=3.1, prior_var=prior_var, q_game=q_game, carry=0.6,
               season_var=0.0, lr_level=0.002, count_model=True, min_sd=1.5)
    return AdapterConfig(params=p, components=comps, feature_names=list(feats), use_rls=use_rls and len(feats) > 0,
                         tie_home=0.52, margin_sd=4.4, n_sim=3000)


# ------------------------------------------------------------------------------------------- K layer
def k_layer(g_res: pd.DataFrame, g: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    """Starter strikeouts coupled to the script: E[K] = E[BF] x log5(p_sp, p_opp), with E[BF] and an online
    RLS correction depending on the scripted opponent runs; NB dispersion learned online."""
    stf = statcast_starter_features(sc)
    tk = team_k_rates(sc)
    stf = stf.merge(tk.rename(columns={"team": "opp", "bat_k": "opp_bat_k"}), on=["date", "opp"], how="left")
    # link to script rows: (date, pitch_team, order)
    gp = sc[["date", "game_pk", "pitch_team"]].drop_duplicates().sort_values(["date", "game_pk"])
    gp["order"] = gp.groupby(["date", "pitch_team"]).cumcount() + 1
    stf = stf.merge(gp, on=["date", "game_pk", "pitch_team"])
    sr = g_res.merge(g[["game_id", "order"]], on="game_id")
    a = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = sr[["game_id", "date", side, "order", f"mu_{opp}", "level", f"mu_{side}"]].rename(
            columns={side: "pitch_team", f"mu_{opp}": "mu_opp_runs", f"mu_{side}": "mu_own_runs"})
        a.append(x)
    sr = pd.concat(a)
    k = stf.merge(sr, on=["date", "pitch_team", "order"], how="inner").sort_values(["date", "game_pk"])
    rls = RLS(9, lam=0.9995, p0=0.05, noise=4.0)
    disp_num, disp_den = 1.0, 0.1     # running E[resid^2 - mu] and E[mu^2] for NB 1/r
    lg = LG_K
    out = []
    for r in k.itertuples(index=False):
        p_sp, p_op = r.sp_k, r.opp_bat_k if r.opp_bat_k == r.opp_bat_k else lg
        odds = (p_sp / (1 - p_sp)) * (p_op / (1 - p_op)) / (lg / (1 - lg))
        p = odds / (1 + odds)
        bf = r.sp_bf
        rest = (min(max(r.sp_rest, 3), 10) - 5) / 5 if r.sp_rest == r.sp_rest else 0.0
        x = np.array([1.0, (r.mu_opp_runs - r.level), (p - lg) * 10, (bf - LG_BF) / 5, (r.mu_own_runs - r.level),
                      (r.sp_k_long - p_sp) * 10, (r.sp_whiff - 0.11) * 10, (r.sp_csw - LG_CSW) * 10, rest])
        mu0 = bf * p
        mu = max(mu0 + rls.predict(x), 0.3)
        inv_r = max(disp_num / disp_den, 0.01)
        rr = 1 / inv_r
        out.append(dict(game_id=r.game_id, game_pk=r.game_pk, date=r.date, player=r.player_name, pitcher_id=r.pitcher,
                        team=r.pitch_team, opp=r.opp, stat="strikeouts", mu=mu, sd=math.sqrt(mu + mu * mu * inv_r),
                        nb_r=rr, k_rate=p_sp, opp_k=p_op, p_k=p, bf_exp=bf, mu_opp_runs=r.mu_opp_runs,
                        n_prior_starts=r.sp_nsc, actual=r.k, bf_actual=r.bf))
        # learn
        rls.update(x, r.k - mu0)
        res2 = (r.k - mu) ** 2 - mu
        disp_num = 0.998 * disp_num + 0.002 * res2
        disp_den = 0.998 * disp_den + 0.002 * mu * mu
        lg = 0.999 * lg + 0.001 * (r.k / max(r.bf, 1))
    return pd.DataFrame(out)


def k_props_eval(kp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pr = pd.read_csv(PROC / "mlb_props.csv")
    pr = pr[pr.market == "pitcher_strikeouts"].dropna(subset=["line", "over_odds", "under_odds", "actual"])
    pr["snap_rank"] = pr.snapshot.map({"late": 0, "am": 1})
    pr = pr.sort_values("snap_rank").drop_duplicates(["date", "player", "book"])
    pr["key"] = pr.player.map(norm)
    kp = kp.copy(); kp["key"] = kp.player.map(norm)
    m = pr.merge(kp[["date", "key", "mu", "nb_r", "actual", "team"]].rename(columns={"actual": "k_sc"}),
                 on=["date", "key"], how="inner")
    m = m[m.k_sc == m.actual]                     # sanity: same game
    pr_n = m.nb_r; p_nb = pr_n / (pr_n + m.mu)
    m["p_over_model"] = 1 - nbinom.cdf(np.floor(m.line), pr_n, p_nb)
    m["p_over_mkt"] = devig_pair(m.over_odds, m.under_odds)
    m["over"] = (m.actual > m.line).astype(int)
    m = m[m.actual != m.line]
    m = m.sort_values("date")
    # online logistic blend learned sequentially through the 2026 sample (predict, then learn)
    bl = OnlineLogit([0.0, 0.3, 0.7], p0=0.3, lam=1.0)
    pb = []
    for r in m.itertuples(index=False):
        xb = np.array([1.0, float(logit(r.p_over_model)), float(logit(r.p_over_mkt))])
        pb.append(bl.predict(xb)); bl.update(xb, r.over)
    m["p_over_blend"] = pb
    # betting: model edge > 5% vs de-vigged, at posted odds
    def dec(o): return 1 + (o / 100 if o > 0 else 100 / -o)
    pnl = []
    for r in m.itertuples(index=False):
        if r.p_over_model - r.p_over_mkt > 0.05:
            pnl.append(dec(r.over_odds) - 1 if r.over else -1)
        elif (1 - r.p_over_model) - (1 - r.p_over_mkt) > 0.05:
            pnl.append(dec(r.under_odds) - 1 if not r.over else -1)
    pnl = np.array(pnl)
    res = dict(rows=len(m), pitcher_games=m.groupby(["date", "key"]).ngroups,
               ll_model=ll(m.p_over_model, m.over).mean(), ll_market=ll(m.p_over_mkt, m.over).mean(),
               ll_blend=ll(m.p_over_blend, m.over).mean(),
               mae_mu=(m.mu - m.actual).abs().mean(), mae_line=(m.line + 0.0 - m.actual).abs().mean(),
               bias_model=(m.p_over_model - m.over).mean(), bias_mkt=(m.p_over_mkt - m.over).mean(),
               bets=len(pnl), roi=pnl.mean() if len(pnl) else np.nan,
               roi_z=(pnl.mean() / (pnl.std(ddof=1) / math.sqrt(len(pnl)))) if len(pnl) > 2 else np.nan)
    return m, res


# ------------------------------------------------------------------------------------------- main
_G = None


def _job(args):
    kw, max_season, since = args
    gg = _G if max_season is None else _G[_G.season <= max_season]
    return summary(run_adapter(gg, make_cfg(**kw)), since)


def run_all(quick: bool = False):
    global _G
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    sc = load_statcast()
    g = attach_features(load_games(sc), sc)
    g = build_side_features(g)
    print(f"games {len(g)}  statcast seasons {sorted(pd.to_datetime(sc.date).dt.year.unique())}  prep {time.time()-t0:.0f}s", flush=True)

    # --- tuning on 2016-2020 only (small grid on the core drift / pitcher prior)
    grid = [dict(q_game=0.0004, prior_var=0.12, sp_var=0.10, sp_gain=1.0), dict(q_game=0.0004, prior_var=0.12, sp_var=0.10, sp_gain=0.5),
            dict(q_game=0.0002, prior_var=0.06, sp_var=0.10, sp_gain=0.5), dict(q_game=0.0002, prior_var=0.06, sp_var=0.06, sp_gain=1.0)]
    _G = g
    grid = grid[:1] if quick else grid
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(_job, [(kw, 2020, 2016) for kw in grid]))
    tune = pd.DataFrame([dict(**kw, ll_script=s["ll_script"], ll_market=s["ll_market"], mae_total=s["mae_total_script"])
                         for kw, s in zip(grid, res)])
    print(tune, f"{time.time()-t0:.0f}s", flush=True)
    best = tune.sort_values("ll_script").iloc[0][["q_game", "prior_var", "sp_var", "sp_gain"]].to_dict()
    print("best", best)

    full = run_adapter(g, make_cfg(**best))
    full = full.merge(g[["game_id", "home_ml", "away_ml", "spread_home", "home_sp_key", "away_sp_key", "over_odds",
                         "under_odds", "playoff"] + [c for c in g.columns if c.startswith(("h_", "a_"))]],
                      on="game_id", how="left")
    print(f"full run {time.time()-t0:.0f}s", flush=True)

    # --- ablation on 2021+
    abl = {"full": summary(full, 2021)}
    variants = {"no_sp_rating": dict(use_sp=False), "no_park": dict(use_park=False), "no_rls(all features)": dict(use_rls=False),
                "core_only": dict(use_sp=False, use_park=False, use_rls=False)}
    for gname, fs in GROUPS.items():
        variants[f"no_{gname}"] = dict(feats=[f for f in FEATS if f not in fs])
    if quick:
        variants = {"core_only": variants["core_only"]}
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(_job, [({**best, **kw}, None, 2021) for kw in variants.values()]))
    abl.update(dict(zip(variants, res)))
    abl = pd.DataFrame(abl).T
    print(abl.round(4), f"{time.time()-t0:.0f}s", flush=True)

    # --- player layer
    kp = k_layer(full, g, sc)
    km, kres = k_props_eval(kp)
    print("K props", kres)

    out = full[full.season >= 2016].copy()
    out["date"] = pd.to_datetime(out.date)
    out.to_parquet(RES / "mlb_games.parquet", index=False)
    kp2 = kp.copy(); kp2["date"] = pd.to_datetime(kp2.date)
    kp2.to_parquet(RES / "mlb_players.parquet", index=False)
    write_report(full, g, tune, best, abl, kp, km, kres, time.time() - t0)
    print(f"done {time.time()-t0:.0f}s")
    return full, abl, kres


def write_report(full, g, tune, best, abl, kp, km, kres, secs):
    d = full[full.season >= 2016]
    st = season_table(d)
    s21 = summary(full, 2021)
    rlsd = d.dropna(subset=["home_win"])
    # error attribution on 2021+: decompose squared error of runs per side
    x = d[d.season >= 2021].dropna(subset=["err_home"])
    eh = pd.concat([x.err_home, x.err_away])
    # which situations the script misses most (|total error| by bucket)
    x = x.assign(abs_tot=x.err_total.abs(), tot_bias=x.err_total, new_sp=((x.h_opp_sp_new + x.a_opp_sp_new) > 0),
                 month=pd.to_datetime(x.date).dt.month, coors=x.home.eq("COL"))
    att_rows = []
    for name, mask in (("all", x.index == x.index), ("a starter with <3 prior starts", x.new_sp),
                       ("April", x.month == 4), ("Sep/Oct", x.month >= 9), ("Coors Field", x.coors),
                       ("playoffs", x.playoff == 1), ("mkt total >= 9.5", x.total_line >= 9.5),
                       ("mkt total <= 7.5", x.total_line <= 7.5)):
        z = x[mask]
        if len(z) == 0:
            continue
        att_rows.append(dict(bucket=name, games=len(z), mae_total_script=z.abs_tot.mean(),
                             mae_total_market=z.err_total_mkt.abs().mean(), bias_total_script=z.tot_bias.mean(),
                             bias_total_market=z.err_total_mkt.mean(),
                             ll_script=ll(z.dropna(subset=["home_win", "p_home_mkt"]).p_home_script, z.dropna(subset=["home_win", "p_home_mkt"]).home_win).mean(),
                             ll_market=ll(z.dropna(subset=["home_win", "p_home_mkt"]).p_home_mkt, z.dropna(subset=["home_win", "p_home_mkt"]).home_win).mean()))
    att = pd.DataFrame(att_rows).set_index("bucket")
    # variance decomposition: share of script-vs-market disagreement explained by components
    y = x.dropna(subset=["p_home_mkt"])
    comp_var = {"sp (home-minus-away runs)": np.var(y.sp_home_eff - y.sp_away_eff),
                "park (total runs)": np.var(y.park_home_eff + y.park_away_eff)}
    # final RLS coefficients are not stored; report correlation of rls term with residual
    bets = {f"script edge>3% ({lab})": bet_sim(full[(full.season >= a) & (full.season <= b)], "p_home_script", 0.03)
            for lab, a, b in (("2016-20", 2016, 2020), ("2021+", 2021, 2026))}
    bets.update({f"blend edge>2% ({lab})": bet_sim(full[(full.season >= a) & (full.season <= b)], "p_home_blend", 0.02)
                 for lab, a, b in (("2016-20", 2016, 2020), ("2021+", 2021, 2026))})
    bets = pd.DataFrame(bets).T
    calib = y.assign(b=pd.cut(y.p_home_script, [0, .35, .45, .5, .55, .65, 1])).groupby("b", observed=True).agg(
        n=("home_win", "size"), p_script=("p_home_script", "mean"), p_mkt=("p_home_mkt", "mean"), won=("home_win", "mean"))
    sp_cov = d.groupby("season").apply(lambda z: (z.h_opp_sp_sc > 0).mean()).rename("statcast_sp_coverage")
    lines = [
        "# MLB script-then-learn: learning report",
        "",
        f"Generated by `python -m engine.scripts.mlb` ({secs/60:.1f} min). Games 2010-2026 run sequentially "
        f"(2010-2015 warm-up), reported 2016+. Script = runs per side from the shared core (count model) + learned "
        "starting-pitcher and park Kalman effects + an online RLS correction on pre-game features. "
        "Win probability from `simulate_scores(count_model=True)` (ties = extra innings, 52% home), then an online "
        "logistic calibration; blend = online logistic of script and de-vigged closing moneyline. "
        "Market total = closing total line. Tuning (grid below) used only 2016-2020.",
        "",
        "Data notes: 2010-2025 closing lines from data/processed/mlb.csv; 2026 results from data/live/mlb.csv with "
        "closing lines only for Sep 14-Oct 14 2026 (210 games), starters for 2026 taken from Statcast. "
        "Statcast pitcher-game data (K, BF, CSW) coverage by season:",
        "",
        md_table(sp_cov.to_frame()),
        "",
        "Weather: not used (no historical weather source wired in). Bullpen fatigue only where Statcast exists.",
        "",
        "## Tuning (2016-2020 only)",
        md_table(tune.set_index("q_game")), "",
        f"Chosen: {best}",
        "",
        "## Learning curve per season (games with closing moneyline)",
        "Log loss/Brier: home win probability. MAE: total runs (script mean vs closing total line) and margin "
        "(market margin translated from the de-vigged moneyline).",
        "",
        md_table(st),
        "",
        f"**2021+ summary** (n={s21['n']}): log loss script {s21['ll_script']:.4f} vs market {s21['ll_market']:.4f} vs "
        f"blend {s21['ll_blend']:.4f}; total MAE script {s21['mae_total_script']:.3f} vs market {s21['mae_total_market']:.3f} "
        f"vs blend {s21['mae_total_blend']:.3f}.",
        "",
        "## Calibration of script win probability (2021+)",
        md_table(calib), "",
        "## Ablation (re-run with one enrichment removed; metrics 2021+)",
        md_table(abl[["n", "ll_script", "ll_market", "ll_blend", "mae_total_script", "mae_total_market", "rmse_total_script", "mae_home_runs"]], 4),
        "",
        "## Error attribution (2021+)",
        md_table(att), "",
        "Spread of learned component effects (variance, runs^2): "
        + ", ".join(f"{k}: {v:.3f}" for k, v in comp_var.items()),
        "",
        "## Does anything beat the closing line?",
        "Flat 1u moneyline bets at the closing price when the model's probability exceeds the de-vigged close.",
        "",
        md_table(bets, 3),
        "",
        "## Player layer: starting-pitcher strikeouts vs 2026 real K lines",
        f"Projection for every Statcast start ({len(kp)} starter-games); E[K] = expected BF x log5(pitcher K%, "
        "opponent K%) + online RLS correction on scripted opponent/own runs; NB dispersion learned online. "
        f"Matched {kres['rows']} book lines ({kres['pitcher_games']} pitcher-games, DK/FD, late snapshot preferred), "
        "Jul 10 - Oct 14 2026.",
        "",
        md_table(pd.DataFrame([kres]).T.rename(columns={0: "value"}), 4),
        "",
    ]
    lines += verdict_lines(s21, abl, kres, bets)
    (RES / "mlb_learning.md").write_text("\n".join(lines))


def verdict_lines(s21, abl, kres, bets):
    out = ["## Verdict", ""]
    gap = s21["ll_script"] - s21["ll_market"]
    out.append(f"* Moneyline: the closing market is more accurate than the script by {gap:.4f} log loss per game on 2021+ "
               f"({s21['ll_script']:.4f} vs {s21['ll_market']:.4f})." if gap > 0 else
               f"* Moneyline: the script beat the market by {-gap:.4f} log loss on 2021+.")
    bgap = s21["ll_blend"] - s21["ll_market"]
    out.append(f"* Learned blend vs market: {bgap:+.4f} log loss (negative = blend better).")
    out.append(f"* Totals: script MAE {s21['mae_total_script']:.3f} vs closing total {s21['mae_total_market']:.3f}.")
    full = abl.loc["full"]
    helped = [(n, full.ll_script - r.ll_script, full.mae_total_script - r.mae_total_script) for n, r in abl.iterrows() if n != "full"]
    out.append("* Ablation (delta when removed; positive log-loss delta = component helped): " + "; ".join(
        f"{n}: ll {-dl:+.4f}, total MAE {-dm:+.3f}" for n, dl, dm in helped))
    out.append(f"* K props: model log loss {kres['ll_model']:.4f} vs market {kres['ll_market']:.4f} vs blend "
               f"{kres['ll_blend']:.4f}; edge>5% bets: {kres['bets']} at ROI {kres['roi']:+.3f} (z {kres['roi_z']:.2f}).")
    return out


if __name__ == "__main__":
    run_all(quick="--quick" in sys.argv)
