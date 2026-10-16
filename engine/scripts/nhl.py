"""NHL script-then-learn adapter (goals per side, count model) + skater SOG / points layer.

Script for each game (only information before puck drop):
  core          team offense / defense goal ratings, league level, home edge (base.ScriptEngine, count model)
  goalie        starting-goalie rating: Kalman effect on the OPPONENT's goals (goals allowed vs scripted),
                updated after every start
  RLS features  (online recursive least squares on the per-side residual)
                  back-to-back (own / opp), opp goalie is a backup (<35% of team's last 10 starts), opp goalie new
                  (<5 prior starts), shot pace: own shots-for and opp shots-against (decayed, NHL API boxscores),
                  special teams: own power-play goals for, opp power-play goals against (decayed)
  learned blend own script + de-vigged closing moneyline (online logistic), totals via core Bates-Granger weight
Scripted team shots = decayed shots-for/against pace + an online RLS on the scripted goals (coupling).
Player layer: skater SOG = learned shot share x scripted team shots; points = learned points-per-team-goal share x
scripted team goals; negative binomial / Poisson; tested on the 2026-27 prop lines in data/processed/nhl_props.csv.

Tuning on 2016-2020 only; reporting 2021+. Run:  python -m engine.scripts.nhl
"""
from __future__ import annotations

import math
import re
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from engine.scripts.base import Params
from engine.scripts.countsport import (AdapterConfig, KalmanComponent, RLS, OnlineLogit, bet_sim, devig_pair, ll,
                                       md_table, run_adapter, season_table, summary, logit)
from engine.scripts.nhl_fetch import load_box

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PROC, LIVE, RES = ROOT / "data" / "processed", ROOT / "data" / "live", ROOT / "results" / "scripts"
LG_SOG, LG_PPG = 30.5, 0.6


def norm(s):
    if s is None or (isinstance(s, float) and s != s):
        return None
    return re.sub(r"[^a-z]", "", str(s).lower()) or None


# ------------------------------------------------------------------------------------------- data
def load_games(bg: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(PROC / "nhl.csv")
    d["neutral"] = 0
    d["nhl_id"] = d.game_id.astype(int)
    l = pd.read_csv(LIVE / "nhl.csv")
    l = l[l.completed.astype(bool)].copy()
    # 2026-27 regular season from the NHL API (drops preseason rows in the ESPN feed); goalies from boxscores
    bx = bg[bg.game_id >= 2026020000][["game_id", "date", "home", "away"]].rename(columns={"date": "bdate"})
    l["d"] = pd.to_datetime(l.date)
    m = l.merge(bx, on=["home", "away"])
    m = m[(pd.to_datetime(m.bdate) - m.d).dt.days.abs() <= 1]
    m["season"] = 2027
    m["nhl_id"] = m.game_id
    m["date"] = m.bdate
    m["neutral"] = m.neutral.fillna(0).astype(int)
    # goalie names for live games: NHL player id -> full name learned from earlier seasons
    gs = bp[(bp.pos == "G") & (bp.starter == 1)][["game_id", "team", "player_id", "player"]]
    hist = gs.merge(d[["nhl_id", "home", "away", "home_goalie", "away_goalie"]], left_on="game_id", right_on="nhl_id")
    hist["full"] = np.where(hist.team == hist.home, hist.home_goalie, hist.away_goalie)
    id2name = hist.dropna(subset=["full"]).groupby("player_id").full.agg(lambda x: x.mode().iloc[0]).to_dict()
    for side in ("home", "away"):
        z = gs.rename(columns={"team": side, "game_id": "nhl_id"})
        z[f"{side}_goalie"] = [id2name.get(pid, p) for pid, p in zip(z.player_id, z.player)]
        m = m.merge(z[["nhl_id", side, f"{side}_goalie"]], on=["nhl_id", side], how="left")
    cols = ["nhl_id", "date", "season", "home", "away", "home_score", "away_score", "home_ml", "away_ml", "total",
            "over_odds", "under_odds", "spread_home", "playoff", "home_goalie", "away_goalie", "neutral"]
    g = pd.concat([d[cols], m[cols]], ignore_index=True).dropna(subset=["home_score", "away_score"])
    g = g.drop_duplicates("nhl_id").sort_values(["date", "nhl_id"]).reset_index(drop=True)
    g["game_id"] = g.nhl_id.astype(str)
    return g


def team_box(bg: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: shots for/against, power-play goals for/against (regular season boxscores)."""
    sk = bp[bp.pos != "G"].groupby(["game_id", "team"]).ppg.sum().rename("ppg").reset_index()
    rows = []
    for r in bg.itertuples(index=False):
        rows.append((r.game_id, r.date, r.home, r.home_sog, r.away_sog))
        rows.append((r.game_id, r.date, r.away, r.away_sog, r.home_sog))
    t = pd.DataFrame(rows, columns=["nhl_id", "date", "team", "sf", "sa"])
    t = t.merge(sk.rename(columns={"game_id": "nhl_id"}), on=["nhl_id", "team"], how="left")
    opp = t[["nhl_id", "team", "ppg"]].rename(columns={"team": "opp_team", "ppg": "ppga"})
    t = t.merge(opp, on="nhl_id")
    t = t[t.team != t.opp_team].drop(columns="opp_team")
    return t.sort_values(["date", "nhl_id"])


def attach_features(g: pd.DataFrame, tb: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    # decayed pace / special-teams state per team, read BEFORE each game, updated after (box games only)
    tbi = {(r.nhl_id, r.team): r for r in tb.itertuples(index=False)}
    st = defaultdict(lambda: dict(sf=LG_SOG, sa=LG_SOG, ppg=LG_PPG, ppga=LG_PPG, n=0))
    lg = dict(sf=LG_SOG, ppg=LG_PPG)
    dec = 0.93
    feats = defaultdict(list)
    last_date = {}
    goalie_hist = defaultdict(lambda: deque(maxlen=10))
    gstarts = defaultdict(int)
    for r in g.itertuples(index=False):
        dt = pd.Timestamp(r.date)
        for side in ("home", "away"):
            t = getattr(r, side)
            s = st[t]
            feats[f"{side}_sf"].append(s["sf"] - lg["sf"]); feats[f"{side}_sa"].append(s["sa"] - lg["sf"])
            feats[f"{side}_ppg"].append(s["ppg"] - lg["ppg"]); feats[f"{side}_ppga"].append(s["ppga"] - lg["ppg"])
            feats[f"{side}_sf_abs"].append(s["sf"]); feats[f"{side}_sa_abs"].append(s["sa"])
            ld = last_date.get(t)
            feats[f"{side}_b2b"].append(int(ld is not None and (dt - ld).days == 1))
            gk = norm(getattr(r, f"{side}_goalie"))
            h = goalie_hist[t]
            share = (sum(1 for x in h if x == gk) / len(h)) if len(h) >= 5 else 1.0
            feats[f"{side}_g_backup"].append(int(share < 0.35))
            feats[f"{side}_g_new"].append(int(gstarts[gk] < 5))
            feats[f"{side}_g_key"].append(gk)
        feats["lg_sf"].append(lg["sf"])
        # update after the game
        for side in ("home", "away"):
            t = getattr(r, side)
            last_date[t] = dt
            gk = norm(getattr(r, f"{side}_goalie"))
            goalie_hist[t].append(gk); gstarts[gk] += 1
            b = tbi.get((r.nhl_id, t))
            if b is not None and b.sf == b.sf:
                s = st[t]
                for k in ("sf", "sa", "ppg", "ppga"):
                    v = getattr(b, k)
                    if v == v:
                        s[k] = dec * s[k] + (1 - dec) * v
                lg["sf"] = 0.998 * lg["sf"] + 0.002 * b.sf
                lg["ppg"] = 0.998 * lg["ppg"] + 0.002 * (b.ppg if b.ppg == b.ppg else lg["ppg"])
    for k, v in feats.items():
        g[k] = v
    g["has_box"] = g.nhl_id.isin(set(tb.nhl_id)).astype(int)
    return g


FEATS = ["own_b2b", "opp_b2b", "opp_g_backup", "opp_g_new", "own_sf", "opp_sa", "own_ppg", "opp_ppga", "pace_known"]
GROUPS = {"rest": ["own_b2b", "opp_b2b"], "goalie_context": ["opp_g_backup", "opp_g_new"],
          "shots_pace": ["own_sf", "opp_sa"], "special_teams": ["own_ppg", "opp_ppga"]}


def build_side_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    known = (g.season >= 2017).astype(float)          # box data from 2015-16; first season is the pace warm-up
    for side, opp, pre in (("home", "away", "h"), ("away", "home", "a")):
        g[f"{pre}_own_b2b"] = g[f"{side}_b2b"].astype(float)
        g[f"{pre}_opp_b2b"] = g[f"{opp}_b2b"].astype(float)
        g[f"{pre}_opp_g_backup"] = g[f"{opp}_g_backup"].astype(float)
        g[f"{pre}_opp_g_new"] = g[f"{opp}_g_new"].astype(float)
        g[f"{pre}_own_sf"] = g[f"{side}_sf"] / 3 * known
        g[f"{pre}_opp_sa"] = g[f"{opp}_sa"] / 3 * known
        g[f"{pre}_own_ppg"] = g[f"{side}_ppg"] / 0.2 * known
        g[f"{pre}_opp_ppga"] = g[f"{opp}_ppga"] / 0.2 * known
        g[f"{pre}_pace_known"] = known
    g["goalie_h_key"], g["goalie_a_key"] = g.away_g_key, g.home_g_key
    pm = devig_pair(g.home_ml, g.away_ml)
    g["p_mkt"] = np.where(g.home_ml.notna() & g.away_ml.notna(), pm, np.nan)
    g["total_line"] = g.total
    return g


def make_cfg(feats=FEATS, use_goalie=True, use_rls=True, q_game=0.0003, prior_var=0.06, g_var=0.06,
             g_gain=1.0) -> AdapterConfig:
    comps = {}
    if use_goalie:
        comps["goalie"] = KalmanComponent(prior_mean=0.05, prior_var=g_var, q=0.0005, carry=0.8, gain_scale=g_gain)
    p = Params(init_level=2.9, init_hfa=0.2, init_sd=1.6, prior_var=prior_var, q_game=q_game, carry=0.6,
               season_var=0.0, lr_level=0.002, count_model=True, min_sd=1.0)
    return AdapterConfig(params=p, components=comps, feature_names=list(feats), use_rls=use_rls and len(feats) > 0,
                         tie_home=0.5, margin_sd=2.4, n_sim=3000)


# ------------------------------------------------------------------------------------------- player layer
def team_shots_script(res: pd.DataFrame, g: pd.DataFrame, tb: pd.DataFrame) -> pd.DataFrame:
    """Scripted shots per side = league + own shots-for pace + opp shots-against pace, plus an online RLS on the
    scripted goals (coupling shots to the goal script). Learned only from earlier games."""
    x = res.merge(g[["game_id", "nhl_id", "home_sf_abs", "away_sf_abs", "home_sa_abs", "away_sa_abs", "lg_sf"]], on="game_id")
    act = tb.set_index(["nhl_id", "team"]).sf.to_dict()
    rls = RLS(3, lam=0.9995, p0=0.5, noise=40.0)
    out = []
    for r in x.itertuples(index=False):
        row = {}
        for side, opp in (("home", "away"), ("away", "home")):
            base = getattr(r, f"{side}_sf_abs") + getattr(r, f"{opp}_sa_abs") - r.lg_sf
            f = np.array([1.0, getattr(r, f"mu_{side}") - r.level, 1.0 if side == "home" else -1.0])
            mu = base + rls.predict(f)
            row[f"shots_{side}"] = mu
            row[f"_f_{side}"], row[f"_b_{side}"] = f, base
        for side in ("home", "away"):
            a = act.get((r.nhl_id, getattr(r, side)))
            if a is not None and a == a:
                rls.update(row[f"_f_{side}"], a - row[f"_b_{side}"])
        out.append(dict(game_id=r.game_id, shots_home=row["shots_home"], shots_away=row["shots_away"],
                        shots_home_act=act.get((r.nhl_id, r.home)), shots_away_act=act.get((r.nhl_id, r.away))))
    return pd.DataFrame(out)


def player_layer(res: pd.DataFrame, shots: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    """Skater SOG = decayed shot share x scripted team shots; points = decayed points-per-team-goal share x scripted
    team goals. Shares are updated after every game (Bayesian shrinkage to a position prior)."""
    sk = bp[bp.pos != "G"].copy()
    sk["nhl_id"] = sk.game_id
    tg = sk.groupby(["nhl_id", "team"]).agg(team_sog=("sog", "sum"), team_goals=("goals", "sum")).reset_index()
    sk = sk.merge(tg, on=["nhl_id", "team"])
    s = res[["game_id", "date", "home", "away", "mu_home", "mu_away"]].merge(shots, on="game_id")
    s["nhl_id"] = s.game_id.astype(int)
    lk = {}
    for r in s.itertuples(index=False):
        lk[(r.nhl_id, r.home)] = (r.shots_home, r.mu_home)
        lk[(r.nhl_id, r.away)] = (r.shots_away, r.mu_away)
    sk = sk[[k in lk for k in zip(sk.nhl_id, sk.team)]].sort_values(["date", "nhl_id"])
    prior_sog = {"D": 0.065, "C": 0.095, "L": 0.095, "R": 0.095}
    prior_pts = {"D": 0.25, "C": 0.45, "L": 0.42, "R": 0.42}
    st = {}
    dec = 0.95
    disp = [0.15]
    out = []
    for r in sk.itertuples(index=False):
        tsh, tgl = lk[(r.nhl_id, r.team)]
        a = st.get(r.player_id)
        if a is None:
            a = dict(s=0.0, ts=0.0, p=0.0, tg=0.0, n=0)
            st[r.player_id] = a
        ps, pp = prior_sog.get(r.pos, 0.09), prior_pts.get(r.pos, 0.4)
        share = (a["s"] + ps * 60) / (a["ts"] + 60)
        pshare = (a["p"] + pp * 8) / (a["tg"] + 8)
        mu_sog = share * tsh
        mu_pts = pshare * tgl
        out.append((r.nhl_id, r.date, r.player, r.player_id, r.team, r.pos, share, mu_sog, 1 / disp[0], r.sog,
                    pshare, mu_pts, r.points, r.goals, a["n"], tsh, tgl))
        # learn
        if r.toi_s is not None and r.toi_s == r.toi_s and r.toi_s > 0:
            a["s"] = dec * a["s"] + r.sog; a["ts"] = dec * a["ts"] + r.team_sog
            a["p"] = dec * a["p"] + r.points; a["tg"] = dec * a["tg"] + r.team_goals
            a["n"] += 1
            disp[0] = float(np.clip(0.999 * disp[0] + 0.001 * (((r.sog - mu_sog) ** 2 - mu_sog) / max(mu_sog ** 2, 0.05)), 0.01, 1.0))
    p = pd.DataFrame(out, columns=["nhl_id", "date", "player", "player_id", "team", "pos", "share_of_team", "mu_sog",
                                   "nb_r", "sog", "share_pts", "mu_pts", "points", "goals", "n_prior", "team_shots_script",
                                   "team_goals_script"])
    a = p[["nhl_id", "date", "player", "player_id", "team", "pos", "share_of_team", "n_prior"]].assign(
        stat="sog", mu=p.mu_sog, nb_r=p.nb_r, sd=np.sqrt(p.mu_sog + p.mu_sog ** 2 / p.nb_r), actual=p.sog,
        team_script=p.team_shots_script)
    b = p[["nhl_id", "date", "player", "player_id", "team", "pos", "n_prior"]].assign(
        share_of_team=p.share_pts, stat="points", mu=p.mu_pts, nb_r=np.inf, sd=np.sqrt(p.mu_pts), actual=p.points,
        team_script=p.team_goals_script)
    out = pd.concat([a, b], ignore_index=True)
    out["game_id"] = out.nhl_id.astype(str)
    return out


def p_over(stat, mu, nb_r, line):
    k = np.floor(line)
    if stat == "sog":
        return 1 - nbinom.cdf(k, nb_r, nb_r / (nb_r + mu))
    return 1 - poisson.cdf(k, mu)


def props_eval(pl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pr = pd.read_csv(PROC / "nhl_props.csv")
    pr = pr[pr.market.isin(["shots_on_goal", "points"])].dropna(subset=["line", "over_odds", "under_odds", "actual"])
    pr["stat"] = pr.market.map({"shots_on_goal": "sog", "points": "points"})
    pr["nhl_id"] = pr.game_id.astype(int)
    m = pr.merge(pl[["nhl_id", "player_id", "stat", "mu", "nb_r", "actual", "n_prior"]].rename(columns={"actual": "act_box"}),
                 on=["nhl_id", "player_id", "stat"])
    m = m[m.actual != m.line]
    m["p_over_model"] = [p_over(s, mu, r, ln) for s, mu, r, ln in zip(m.stat, m.mu, m.nb_r, m.line)]
    m["p_over_mkt"] = devig_pair(m.over_odds, m.under_odds)
    m["over"] = (m.actual > m.line).astype(int)
    rows = []
    for s, x in m.groupby("stat"):
        def dec(o): return 1 + (o / 100 if o > 0 else 100 / -o)
        pnl = [(dec(r.over_odds) - 1 if r.over else -1) if r.p_over_model - r.p_over_mkt > 0.05 else
               ((dec(r.under_odds) - 1 if not r.over else -1) if r.p_over_mkt - r.p_over_model > 0.05 else None)
               for r in x.itertuples(index=False)]
        pnl = np.array([v for v in pnl if v is not None])
        rows.append(dict(stat=s, lines=len(x), ll_model=ll(x.p_over_model, x.over).mean(), ll_market=ll(x.p_over_mkt, x.over).mean(),
                         ll_avg=ll((x.p_over_model + x.p_over_mkt) / 2, x.over).mean(),
                         over_rate=x.over.mean(), p_model=x.p_over_model.mean(), p_mkt=x.p_over_mkt.mean(),
                         box_matches=(x.act_box == x.actual).mean(), bets=len(pnl), roi=pnl.mean() if len(pnl) else np.nan))
    return m, pd.DataFrame(rows).set_index("stat")


def player_accuracy(pl: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample per-player-game accuracy vs naive baselines (2021+, players with >=10 prior games)."""
    rows = []
    x = pl[(pd.to_datetime(pl.date).dt.year >= 2021) & (pl.n_prior >= 10)]
    for s, z in x.groupby("stat"):
        # naive: share-free baseline = player's decayed average is what the share encodes; compare to league avg
        rows.append(dict(stat=s, n=len(z), mae_script=(z.mu - z.actual).abs().mean(),
                         mae_league_mean=(z.actual.mean() - z.actual).abs().mean(),
                         corr=np.corrcoef(z.mu, z.actual)[0, 1], bias=(z.mu - z.actual).mean()))
    return pd.DataFrame(rows).set_index("stat")


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
    bg, bp = load_box()
    bg = bg[bg.game_id.astype(str).str[4:6] == "02"]
    tb = team_box(bg, bp)
    g = load_games(bg, bp)
    g = build_side_features(attach_features(g, tb))
    print(f"games {len(g)} box games {len(bg)} prep {time.time()-t0:.0f}s", flush=True)
    _G = g
    grid = [dict(q_game=0.0003, prior_var=0.06, g_var=0.06), dict(q_game=0.0008, prior_var=0.06, g_var=0.06),
            dict(q_game=0.0003, prior_var=0.06, g_var=0.02), dict(q_game=0.0001, prior_var=0.04, g_var=0.06)]
    grid = grid[:1] if quick else grid
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(_job, [(kw, 2020, 2016) for kw in grid]))
    tune = pd.DataFrame([dict(**kw, ll_script=s["ll_script"], ll_market=s["ll_market"], mae_total=s["mae_total_script"])
                         for kw, s in zip(grid, res)])
    print(tune, f"{time.time()-t0:.0f}s", flush=True)
    best = tune.sort_values("ll_script").iloc[0][["q_game", "prior_var", "g_var"]].to_dict()

    full = run_adapter(g, make_cfg(**best))
    keep = ["game_id", "nhl_id", "home_ml", "away_ml", "spread_home", "over_odds", "under_odds", "playoff", "home_goalie",
            "away_goalie", "home_b2b", "away_b2b", "home_g_backup", "away_g_backup", "has_box"]
    full = full.merge(g[keep + [c for c in g.columns if c.startswith(("h_", "a_"))]], on="game_id", how="left")
    print(f"full {time.time()-t0:.0f}s", flush=True)

    variants = {"no_goalie_rating": dict(use_goalie=False), "no_rls(all features)": dict(use_rls=False),
                "core_only": dict(use_goalie=False, use_rls=False)}
    for gname, fs in GROUPS.items():
        variants[f"no_{gname}"] = dict(feats=[f for f in FEATS if f not in fs])
    if quick:
        variants = {"core_only": variants["core_only"]}
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(_job, [({**best, **kw}, None, 2021) for kw in variants.values()]))
    abl = {"full": summary(full, 2021)}
    abl.update(dict(zip(variants, res)))
    abl = pd.DataFrame(abl).T
    print(abl.round(4), f"{time.time()-t0:.0f}s", flush=True)

    shots = team_shots_script(full, g, tb)
    full = full.merge(shots, on="game_id", how="left")
    pl = player_layer(full, shots, bp)
    pm, pres = props_eval(pl)
    pacc = player_accuracy(pl)
    print(pres, pacc, sep="\n")

    out = full[full.season >= 2016].copy()
    out["date"] = pd.to_datetime(out.date)
    out.to_parquet(RES / "nhl_games.parquet", index=False)
    pl2 = pl.copy(); pl2["date"] = pd.to_datetime(pl2.date)
    pl2.to_parquet(RES / "nhl_players.parquet", index=False)
    write_report(full, tune, best, abl, pres, pacc, time.time() - t0)
    print(f"done {time.time()-t0:.0f}s")


def write_report(full, tune, best, abl, pres, pacc, secs):
    d = full[full.season >= 2016]
    st = season_table(d)
    s21 = summary(full, 2021)
    x = d[d.season >= 2021].dropna(subset=["err_home"])
    def bucket(name, z):
        zz = z.dropna(subset=["home_win", "p_home_mkt"])
        return dict(bucket=name, games=len(z), mae_total_script=z.err_total.abs().mean(),
                    mae_total_market=z.err_total_mkt.abs().mean(), bias_total_script=z.err_total.mean(),
                    bias_total_market=z.err_total_mkt.mean(), ll_script=ll(zz.p_home_script, zz.home_win).mean(),
                    ll_market=ll(zz.p_home_mkt, zz.home_win).mean())
    att = pd.DataFrame([bucket(n, z) for n, z in (
        ("all", x), ("either side back-to-back", x[(x.home_b2b + x.away_b2b) > 0]),
        ("a backup goalie starts", x[(x.home_g_backup + x.away_g_backup) > 0]),
        ("goalie with <5 prior starts", x[(x.h_opp_g_new + x.a_opp_g_new) > 0]),
        ("October", x[pd.to_datetime(x.date).dt.month == 10]), ("playoffs", x[x.playoff == 1]),
        ("mkt total >= 6.5", x[x.total_line >= 6.5]), ("mkt total <= 5.5", x[x.total_line <= 5.5]))
        if len(z)]).set_index("bucket")
    y = x.dropna(subset=["p_home_mkt"])
    calib = y.assign(b=pd.cut(y.p_home_script, [0, .35, .45, .5, .55, .65, 1])).groupby("b", observed=True).agg(
        n=("home_win", "size"), p_script=("p_home_script", "mean"), p_mkt=("p_home_mkt", "mean"), won=("home_win", "mean"))
    bets = {f"script edge>3% ({lab})": bet_sim(full[(full.season >= a) & (full.season <= b)], "p_home_script", 0.03)
            for lab, a, b in (("2016-20", 2016, 2020), ("2021+", 2021, 2027))}
    bets.update({f"blend edge>2% ({lab})": bet_sim(full[(full.season >= a) & (full.season <= b)], "p_home_blend", 0.02)
                 for lab, a, b in (("2016-20", 2016, 2020), ("2021+", 2021, 2027))})
    bets = pd.DataFrame(bets).T
    sh = d.dropna(subset=["shots_home_act"])
    sh21 = sh[sh.season >= 2021]
    shot_line = (f"Scripted team shots (2021+, n={2*len(sh21)} team-games): MAE {pd.concat([sh21.shots_home - sh21.shots_home_act, sh21.shots_away - sh21.shots_away_act]).abs().mean():.2f} "
                 f"vs league-mean baseline {pd.concat([sh21.shots_home_act, sh21.shots_away_act]).sub(pd.concat([sh21.shots_home_act, sh21.shots_away_act]).mean()).abs().mean():.2f}.")
    lines = [
        "# NHL script-then-learn: learning report", "",
        f"Generated by `python -m engine.scripts.nhl` ({secs/60:.1f} min). Seasons labelled by end year (2016 = 2015-16; "
        "2027 = the 2026-27 season in progress). Games 2008-2027 run sequentially (2008-2015 warm-up), reported 2016+. "
        "Script = goals per side from the shared core (count model) + learned starting-goalie Kalman effect + online RLS "
        "correction on pre-game features (back-to-back, backup/new goalie, decayed shots-for/against pace and power-play "
        "goals from NHL API boxscores, regular season 2015-16 onward). Win probability from "
        "`simulate_scores(count_model=True)` (ties = OT/SO, split 50/50), then online logistic calibration; blend = online "
        "logistic of script and de-vigged closing moneyline. Tuning used only 2016-2020.", "",
        "Data notes: closing moneylines missing for most of 2022-23 (342 of 1400 games) and half of 2025-26; 2026-27 "
        "lines from the ESPN feed. Special teams = power-play goals for/against only (PP opportunities are not in the "
        "boxscore endpoint).", "",
        "## Tuning (2016-2020 only)", md_table(tune.set_index("q_game")), "", f"Chosen: {best}", "",
        "## Learning curve per season (games with closing moneyline)",
        "Log loss/Brier: home win probability (incl. OT/SO). MAE: total goals (script mean vs closing total) and margin "
        "(market margin translated from the de-vigged moneyline).", "",
        md_table(st), "",
        f"**2021+ summary** (n={s21['n']}): log loss script {s21['ll_script']:.4f} vs market {s21['ll_market']:.4f} vs "
        f"blend {s21['ll_blend']:.4f}; total MAE script {s21['mae_total_script']:.3f} vs market {s21['mae_total_market']:.3f} "
        f"vs blend {s21['mae_total_blend']:.3f}.", "",
        "## Calibration of script win probability (2021+)", md_table(calib), "",
        "## Ablation (re-run with one enrichment removed; metrics 2021+)",
        md_table(abl[["n", "ll_script", "ll_market", "ll_blend", "mae_total_script", "mae_total_market", "rmse_total_script", "mae_home_runs"]].rename(columns={"mae_home_runs": "mae_side_goals"}), 4), "",
        "## Error attribution (2021+)", md_table(att), "",
        shot_line, "",
        "## Does anything beat the closing line?",
        "Flat 1u moneyline bets at the closing price when the model's probability exceeds the de-vigged close.", "",
        md_table(bets, 3), "",
        "## Player layer: skater SOG and points",
        "SOG mean = learned shot share x scripted team shots (NB, dispersion learned online); points mean = learned "
        "points-per-team-goal share x scripted team goals (Poisson). Lineups are taken as known (players who dressed).", "",
        "Out-of-sample accuracy, 2021+ player-games with >=10 prior games:", "", md_table(pacc, 3), "",
        "Against real 2026-27 prop lines (data/processed/nhl_props.csv, best price across books, Oct 1-13 2026). "
        "Small sample: treat as a sanity check, not evidence.", "", md_table(pres, 4), "",
    ]
    gap = s21["ll_script"] - s21["ll_market"]
    full_r = abl.loc["full"]
    lines += ["## Verdict", "",
              (f"* Moneyline: the closing market is more accurate than the script by {gap:.4f} log loss per game on 2021+ "
               f"({s21['ll_script']:.4f} vs {s21['ll_market']:.4f})." if gap > 0 else
               f"* Moneyline: script beat the market by {-gap:.4f} log loss on 2021+."),
              f"* Learned blend vs market: {s21['ll_blend'] - s21['ll_market']:+.4f} log loss (negative = blend better).",
              f"* Totals: script MAE {s21['mae_total_script']:.3f} vs closing total {s21['mae_total_market']:.3f}.",
              "* Ablation (delta when removed; positive = component helped): " + "; ".join(
                  f"{n}: ll {r.ll_script - full_r.ll_script:+.4f}, total MAE {r.mae_total_script - full_r.mae_total_script:+.3f}"
                  for n, r in abl.iterrows() if n != "full")]
    (RES / "nhl_learning.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_all(quick="--quick" in sys.argv)
