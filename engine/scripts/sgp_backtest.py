"""
Walk-forward backtest of script-fit same-game parlays (SGP), every sport, 2016+.

  python -m engine.scripts.sgp_backtest            # all sports  (~15-25 min)
  python -m engine.scripts.sgp_backtest nba nfl    # subset

For every game, from the PRE-GAME script only (results/scripts/<sport>_games / _players parquet):
  1. legs in the script's direction vs the line (side/ML, total, player props), 2-4 leg candidates
     (engine/scripts/sgp_builder.candidate_sets). Prop lines: real lines where they exist (NBA points 2024-25 &
     2025-26, NFL 2026 wks 1-5, NHL 2026-27, MLB 2026 Ks); otherwise a MARKET PROXY = trailing 10-game average
     (prior games only, min 5; NFL min 3) rounded to x.5, priced at the historical over-rate of proxy lines
     on prior dates. Proxy-line SGPs are reported separately and are NOT evidence of a bettable edge.
  2. joint probability three ways: script Monte Carlo, market-fair (de-vigged marginals + Gaussian copula with
     correlations learned online from prior settled games), independent product; plus 'hybrid' (script
     marginals + learned correlations). Realised outcome recorded (pushes void the SGP).
  3. learning loop, date by date: after each date settles, update the correlation table, the online logistic
     calibration of the script joint (p_script_cal) and a script+market combiner (p_comb).
  4. edge test: rule (ratio p_script_cal / p_market >= k) chosen on 2016-2020 only, reported on 2021+;
     SGP price = market-fair joint price reduced by hold 0/10/20/30% (no historical SGP prices exist anywhere).
     Real-leg-price version where real leg prices exist (NBA points 2025-26, UFC method props).
Outputs: results/scripts/sgp_candidates.parquet, sgp_learning_curves.csv, sgp_edge.csv, sgp_state.json,
sgp_backtest.md (written by write_report()).
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.core import american_to_decimal
from engine.scripts import sgp_builder as B

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RS = ROOT / "results" / "scripts"
TRAIN_END = 2020
TOP_PROPS = 4
N_SIM = 4000
HOLDS = [0.0, 0.10, 0.20, 0.30]
KGRID = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.0]

STAT_MIN = {
    "nba": {"pts": 8, "reb": 3, "ast": 2, "fg3m": 1},
    "nfl": {"passing_yards": 150, "rushing_yards": 25, "receiving_yards": 20, "receptions": 2},
    "nhl": {"sog": 1.8, "points": 0.4},
    "mlb": {"strikeouts": 0},
}
DIST = {"nhl": {"sog": "nb", "points": "poisson"}, "mlb": {"strikeouts": "nb"}}
STRUCTURAL = {("nba", "pts"), ("nhl", "points")}        # share-of-scoring stats: beta = share*sd_team/sd_player


# ================================================================== games
def load_games(sport: str) -> pd.DataFrame:
    g = pd.read_parquet(RS / f"{sport}_games.parquet")
    g = g[(g.season >= 2016) & g.home_score.notna() & g.away_score.notna()].copy()
    for c in ["over_odds", "under_odds", "home_ml", "away_ml", "spread_home", "total_line"]:
        if c not in g:
            g[c] = np.nan
    has = g.total_line.notna() & (g.spread_home.notna() if sport not in B.ML_SPORTS else g.home_ml.notna())
    g = g[has]
    if sport == "nhl":
        g["game_id"] = g["nhl_id"]
    g["date"] = pd.to_datetime(g.date)
    return g.sort_values("date").reset_index(drop=True)


# ================================================================== real prop lines
def _consolidate(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Many quotes -> one main line per key: modal line, mean de-vigged P(over) there, median decimal prices."""
    d = d[d.over_odds.notna() & d.under_odds.notna() & (d.over_odds.abs() >= 100) & (d.under_odds.abs() >= 100)].copy()
    d["p"] = B.fair2(d.over_odds.values, d.under_odds.values)
    d["dec_o"] = american_to_decimal(d.over_odds.values)
    d["dec_u"] = american_to_decimal(d.under_odds.values)
    d["cnt"] = d.groupby(keys + ["line"]).p.transform("size")
    d["dist5"] = (d.p - 0.5).abs()
    d = d.sort_values(keys + ["cnt", "dist5"], ascending=[True] * len(keys) + [False, True])
    main = d.drop_duplicates(keys)[keys + ["line"]]
    d = d.merge(main, on=keys + ["line"])
    return d.groupby(keys + ["line"], as_index=False).agg(p_over_real=("p", "mean"), dec_o=("dec_o", "median"),
                                                          dec_u=("dec_u", "median"))


def real_lines(sport: str) -> tuple[pd.DataFrame, list[str]] | tuple[None, None]:
    P = ROOT / "data" / "processed"
    if sport == "nba":
        f = pd.read_csv(ROOT / "results" / "nba_props_frame.csv.gz", low_memory=False,
                        usecols=["date", "player_id", "market", "line", "alt_line", "over_odds", "under_odds", "p_mkt"])
        f = f[(f.alt_line == 0) & (f.market == "pts")].copy()
        f["date"] = pd.to_datetime(f.date)
        f["stat"] = "pts"
        r = _consolidate(f, ["date", "player_id", "stat"])
        pm = f.merge(r[["date", "player_id", "line"]], on=["date", "player_id", "line"]).groupby(["date", "player_id"]).p_mkt.mean()
        r = r.merge(pm.rename("p_cons").reset_index(), on=["date", "player_id"], how="left")
        r["p_over_real"] = r.p_cons.fillna(r.p_over_real)          # leave-one-book-out consensus where available
        return r.drop(columns="p_cons"), ["date", "player_id", "stat"]
    if sport == "nfl":
        f = pd.read_csv(P / "nfl_props.csv", low_memory=False)
        f = f[(f.season == 2026) & (f.source_type == "sportsbook")].rename(columns={"market": "stat"})
        f = f[f.stat.isin(STAT_MIN["nfl"])]
        return _consolidate(f, ["game_id", "player_id", "stat"]), ["game_id", "player_id", "stat"]
    if sport == "nhl":
        f = pd.read_csv(P / "nhl_props.csv")
        f["stat"] = f.market.map({"shots_on_goal": "sog", "points": "points"})
        f = f[f.stat.notna()]
        return _consolidate(f, ["game_id", "player_id", "stat"]), ["game_id", "player_id", "stat"]
    if sport == "mlb":
        f = pd.read_csv(P / "mlb_props.csv")
        f = f[f.market == "pitcher_strikeouts"].copy()
        f["date"] = pd.to_datetime(f.date)
        f["stat"] = "strikeouts"
        f["pname"] = f.player.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
        late = f.groupby(["date", "pname"]).snapshot.transform(lambda s: (s == "late").any())
        f = f[(f.snapshot == "late") | ~late]
        return _consolidate(f, ["date", "pname", "stat"]), ["date", "pname", "stat"]
    return None, None


# ================================================================== players
def load_players(sport: str, games: pd.DataFrame) -> pd.DataFrame | None:
    if sport not in STAT_MIN:
        return None
    p = pd.read_parquet(RS / f"{sport}_players.parquet")
    p = p[p.stat.isin(STAT_MIN[sport])].copy()
    if sport == "nhl":
        p["game_id"] = p["nhl_id"]
    if sport == "mlb":
        p["pid"] = p.pitcher_id
        last_first = p.player.str.split(",", n=1)
        p["pname"] = (last_first.str[1].fillna("") + " " + last_first.str[0]).str.lower() \
            .str.replace(r"[^a-z ]", "", regex=True).str.strip()
        p["n_prior"] = p.n_prior_starts
    else:
        p["pid"] = p.player_id
    if sport == "nfl":
        p["n_prior"] = p.prior_games
    if "nb_r" not in p:
        p["nb_r"] = np.nan
    p["date"] = pd.to_datetime(p.date)
    p["actual"] = p.actual.astype(float).fillna(0)
    # trailing average of PRIOR games (market proxy line)
    p = p.sort_values(["pid", "stat", "date"]).reset_index(drop=True)
    grp = p.groupby(["pid", "stat"], sort=False)
    p["cs"] = grp.actual.cumsum()
    c = grp.cumcount()
    prev = grp.cs.shift(1)
    prev11 = grp.cs.shift(11).fillna(0)
    nmin = 3 if sport == "nfl" else 5
    p["n_prev"] = c
    p["trail_avg"] = np.where(c >= nmin, (prev - prev11) / c.clip(upper=10).replace(0, 1), np.nan)
    # join to the game script
    gcols = ["game_id", "season", "home", "away", "mu_home", "mu_away", "sd_home", "sd_away", "home_score", "away_score"]
    gg = games[gcols].copy()
    if sport == "nhl" or sport == "mlb" or sport == "nba":
        gg["game_id"] = gg.game_id.astype(str)
        p["game_id"] = p.game_id.astype(str)
    if "season" in p:
        p = p.drop(columns="season")
    p = p.merge(gg, on="game_id", how="inner")
    p = p[(p.team == p.home) | (p.team == p.away)].copy()
    p["team_is_home"] = p.team == p.home
    p["couple_home"] = ~p.team_is_home if sport == "mlb" else p.team_is_home     # Ks <-> OPPONENT runs
    ch = p.couple_home
    p["mu_team"] = np.where(ch, p.mu_home, p.mu_away)
    p["sd_team"] = np.where(ch, p.sd_home, p.sd_away)
    p["team_act"] = np.where(ch, p.home_score, p.away_score)
    # eligibility (pre-game info only)
    mins = p.stat.map(STAT_MIN[sport])
    ok = (p.mu >= mins) & p.trail_avg.notna()
    if sport == "nba":
        ok &= p.min_exp >= 24
    if sport in ("nfl",):
        ok &= p.n_prior >= 3
    if sport == "nhl":
        ok &= p.n_prior >= 10
    if sport == "mlb":
        ok &= p.n_prior >= 3
    p = p[ok].copy()
    # real lines
    p["real"] = False
    p["line"] = np.floor(p.trail_avg) + 0.5
    p["proxy_line"] = p.line
    p["p_over_mkt"] = np.nan
    p["dec_o"] = np.nan
    p["dec_u"] = np.nan
    rl, keys = real_lines(sport)
    if rl is not None and len(rl):
        if "game_id" in keys:
            rl["game_id"] = rl.game_id.astype(str)
        if "player_id" in keys:
            rl["player_id"] = pd.to_numeric(rl.player_id, errors="coerce") if sport != "nfl" else rl.player_id
            if sport != "nfl":
                p["player_id"] = pd.to_numeric(p.player_id, errors="coerce")
        p = p.merge(rl.rename(columns={"line": "rline", "dec_o": "rdo", "dec_u": "rdu"}), on=keys, how="left")
        m = p.rline.notna()
        p.loc[m, "real"] = True
        p.loc[m, "line"] = p.loc[m, "rline"]
        p.loc[m, "p_over_mkt"] = p.loc[m, "p_over_real"]
        p.loc[m, "dec_o"] = p.loc[m, "rdo"]
        p.loc[m, "dec_u"] = p.loc[m, "rdu"]
        print(f"  {sport}: real prop lines matched {int(m.sum())} player-games")
    # proxy market: over-rate of proxy lines on PRIOR dates (by stat & line level), Beta(20,20) prior
    p["pkey"] = p.stat + "|" + np.where(p.proxy_line < 10, p.proxy_line.round(1).astype(str), "hi")
    p["proxy_over"] = (p.actual > p.proxy_line).astype(float)
    agg = p.groupby(["pkey", "date"]).proxy_over.agg(["sum", "size"]).reset_index().sort_values(["pkey", "date"])
    agg["cs"] = agg.groupby("pkey")["sum"].cumsum() - agg["sum"]
    agg["cn"] = agg.groupby("pkey")["size"].cumsum() - agg["size"]
    agg["proxy_rate"] = (agg.cs + B.PROXY_N0 / 2) / (agg.cn + B.PROXY_N0)
    p = p.merge(agg[["pkey", "date", "proxy_rate"]], on=["pkey", "date"], how="left")
    p.loc[~p.real, "p_over_mkt"] = p.loc[~p.real, "proxy_rate"]
    # script marginal at the line + latent threshold
    p["thr"], p["p_over_script"] = np.nan, np.nan
    for st, x in p.groupby("stat"):
        t, ps = B.player_threshold(DIST.get(sport, {}).get(st, "normal"), x.mu.values, x.sd.values, x.nb_r.values, x.line.values)
        p.loc[x.index, "thr"] = t
        p.loc[x.index, "p_over_script"] = ps
    # coupling beta: structural for share-of-scoring stats; else learned from PRIOR seasons' residuals
    p["zp"] = (p.actual - p.mu) / p.sd.clip(lower=1e-6)
    p["zt"] = (p.team_act - p.mu_team) / p.sd_team.clip(lower=1e-6)
    p["beta"] = np.nan
    beta_last = {}
    for st, x in p.groupby("stat"):
        if (sport, st) in STRUCTURAL:
            b = (x.share_of_team * x.sd_team / x.sd.clip(lower=1e-6)).clip(0, 0.9)
            p.loc[x.index, "beta"] = b
            beta_last[st] = float(b.median())
            continue
        for s in sorted(x.season.unique()):
            prior = x[x.season < s]
            b = float(np.corrcoef(prior.zp, prior.zt)[0, 1]) if len(prior) > 500 else 0.0
            p.loc[x.index[x.season == s], "beta"] = b
        allb = float(np.corrcoef(x.zp, x.zt)[0, 1])
        beta_last[st] = allb                       # for live use only (all seasons now in the past)
    p.attrs["beta_last"] = beta_last
    p["outcome_over"] = (p.actual > p.line)
    # direction = script lean vs market; keep leaning legs, top-K per game, one leg per player
    p["pick_over"] = p.p_over_script >= p.p_over_mkt
    p["p_s_pick"] = np.where(p.pick_over, p.p_over_script, 1 - p.p_over_script)
    p["p_m_pick"] = np.where(p.pick_over, p.p_over_mkt, 1 - p.p_over_mkt)
    p["ratio"] = p.p_s_pick / p.p_m_pick.clip(lower=1e-6)
    p = p[p.p_s_pick > p.p_m_pick].sort_values(["game_id", "real", "ratio"], ascending=[True, False, False])
    p = p.drop_duplicates(["game_id", "pid"]).groupby("game_id").head(TOP_PROPS)
    return p


# ================================================================== per-sport walk-forward
def _prop_leg(r) -> dict:
    return dict(kind="prop", player=r.player, stat=r.stat, line=float(r.line), thr=float(r.thr), beta=float(r.beta),
                team_is_home=bool(r.team_is_home), couple_home=bool(r.couple_home), pick_home=bool(r.pick_over),
                o=1 if r.pick_over else -1, real=bool(r.real), p_script=float(r.p_s_pick), p_mkt=float(r.p_m_pick),
                dec=float(r.dec_o if r.pick_over else r.dec_u), outcome=bool(r.outcome_over) == bool(r.pick_over),
                over=bool(r.outcome_over), push=False)


def _game_outcomes(lg, g):
    h, a = g.home_score, g.away_score
    if lg["kind"] == "side":
        m = h - a + lg["line"]
        home_w, push = m > 0, m == 0
    elif lg["kind"] == "ml":
        home_w, push = h > a, False
    else:
        home_w, push = (h + a) > lg["line"], (h + a) == lg["line"]
    return bool(home_w), bool(push)


def run_sport(sport: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    G = load_games(sport)
    P = load_players(sport, G)
    pools = {gid: list(x.itertuples(index=False)) for gid, x in P.groupby("game_id")} if P is not None else {}
    print(f"  {sport}: {len(G)} games, {0 if P is None else len(P)} pooled prop legs ({time.time()-t0:.0f}s)")
    rho = B.RhoTable()
    cal = B.OnlineLogit([0.0, 1.0, 0.0])
    comb = B.OnlineLogit([0.0, 0.5, 0.5, 0.0])
    base = np.random.default_rng(7).standard_normal((20000, 4))
    rows = []
    for di, (date, gd) in enumerate(G.groupby("date", sort=True)):
        batch, obs = [], []
        for g in gd.itertuples(index=False):
            gr = g._asdict()
            glegs = B.game_legs(sport, gr)
            if not glegs:
                continue
            plegs = [_prop_leg(r) for r in pools.get(str(g.game_id), [])]
            rng = np.random.default_rng(zlib.crc32(f"{sport}|{g.game_id}".encode()))
            for lg in glegs:
                lg["pick_home"] = True
            win, push = B.simulate_game(sport, gr, glegs, plegs, n=N_SIM, rng=rng)
            ps_home = [float(win[~push[:, j], j].mean()) if (~push[:, j]).any() else 0.5 for j in range(len(glegs))]
            B.finalize_game_legs(glegs, ps_home)
            for j, lg in enumerate(glegs):
                if not lg["pick_home"]:
                    win[:, j] = ~win[:, j] & ~push[:, j]
                hw, pu = _game_outcomes(lg, g)
                lg["push"] = pu
                lg["home_event"] = hw
                lg["outcome"] = (hw if lg["pick_home"] else (not hw)) and not pu
                lg["label"] = B.leg_label(lg, gr)
            for lg in plegs:
                lg["label"] = B.leg_label(lg, gr)
            legs = glegs + plegs
            for idx in B.candidate_sets(glegs, plegs):
                L = [legs[i] for i in idx]
                void = any(lg["push"] for lg in L)
                batch.append(dict(sport=sport, game_id=str(g.game_id), date=date, season=int(g.season),
                                  n_legs=len(L), kinds="+".join(lg["kind"] for lg in L),
                                  legs=" | ".join(lg["label"] for lg in L),
                                  has_prop=any(lg["kind"] == "prop" for lg in L),
                                  all_real=all(lg["real"] for lg in L),
                                  p_script=B.script_joint(win, push, list(idx)),
                                  p_script_indep=float(np.prod([lg["p_script"] for lg in L])),
                                  _pm=[lg["p_mkt"] for lg in L], _ps=[lg["p_script"] for lg in L],
                                  _R=B.market_R(sport, L, rho),
                                  dec_legs=float(np.prod([lg["dec"] for lg in L])),
                                  y=np.nan if void else float(all(lg["outcome"] for lg in L))))
            # canonical events of every pooled leg, for the correlation table (observed AFTER the date is scored)
            can = []
            for lg in glegs:
                if lg["push"]:
                    continue
                fav_ev = lg["home_event"] if (lg["kind"] == "total" or lg["fav_home"]) else not lg["home_event"]
                can.append(({**lg, "o": 1}, fav_ev))
            for lg in plegs:
                can.append(({**lg, "o": 1}, lg["over"]))
            obs.append(can)
        if not batch:
            continue
        # ---- market-fair / hybrid joints with the correlation table as of BEFORE this date
        for k in (2, 3, 4):
            ii = [i for i, b in enumerate(batch) if b["n_legs"] == k]
            if not ii:
                continue
            pm = np.array([batch[i]["_pm"] for i in ii])
            ps = np.array([batch[i]["_ps"] for i in ii])
            R = np.array([batch[i]["_R"] for i in ii])
            jm = B.mvn_orthant(pm, R, base)
            jh = B.mvn_orthant(ps, R, base)
            for i, a, h in zip(ii, jm, jh):
                batch[i]["p_mkt"] = float(a)
                batch[i]["p_hybrid"] = float(h)
                batch[i]["p_indep"] = float(np.prod(batch[i]["_pm"]))
        bd = pd.DataFrame(batch).drop(columns=["_pm", "_ps", "_R"])
        bd = bd[bd.p_script.notna()]
        bd["p_script_cal"] = cal.predict(B.cal_features(bd.p_script.values, bd.n_legs.values))
        bd["p_comb"] = comb.predict(B.comb_features(bd.p_script.values, bd.p_mkt.values, bd.n_legs.values))
        rows.append(bd)
        # ---- learn from this date's wins and losses
        s = bd[bd.y.notna()]
        cal.update(B.cal_features(s.p_script.values, s.n_legs.values), s.y.values)
        comb.update(B.comb_features(s.p_script.values, s.p_mkt.values, s.n_legs.values), s.y.values)
        for can in obs:
            for i in range(len(can)):
                for j in range(i + 1, len(can)):
                    key, sgn = B.pair_key(sport, can[i][0], can[j][0])
                    if key is None:
                        continue
                    A, Bv = can[i], can[j]
                    # orient so the stored pair is (first leg in pair_key order, second)
                    order = {"side": 0, "ml": 0, "total": 1, "prop": 2}
                    if order[A[0]["kind"]] > order[Bv[0]["kind"]]:
                        A, Bv = Bv, A
                    ea = A[1] if sgn > 0 else not A[1]
                    rho.observe(key, bool(ea), bool(Bv[1]))
        rho.refresh()
    df = pd.concat(rows, ignore_index=True)
    state = dict(rho=rho.to_json(), cal=cal.w.tolist(), comb=comb.w.tolist(),
                 beta=(P.attrs.get("beta_last", {}) if P is not None else {}))
    if P is not None:
        last = P.sort_values("date").groupby("pkey").proxy_rate.last()
        state["proxy_rate"] = {k: round(float(v), 4) for k, v in last.items()}
    print(f"  {sport}: {len(df)} candidate SGPs in {time.time()-t0:.0f}s")
    return df, state


# ================================================================== UFC
def run_ufc() -> tuple[pd.DataFrame, dict]:
    u = pd.read_parquet(RS / "ufc_fights.parquet")
    u = u[(u.season >= 2016) & (u.mkt_quality_ok == True) & u.mkt_A_win_ml.notna()].copy()
    u["date"] = pd.to_datetime(u.date)
    u = u.sort_values("date")
    cal = B.OnlineLogit([0.0, 1.0, 0.0])
    comb = B.OnlineLogit([0.0, 0.5, 0.5, 0.0])
    rows = []
    od = lambda o: float(american_to_decimal(o)) if np.isfinite(o) else np.nan
    for date, fd in u.groupby("date"):
        batch = []
        for f in fd.itertuples(index=False):
            W = "A" if f.p_A_win >= f.mkt_A_win_ml else "B"
            g = lambda pre, m: getattr(f, f"{pre}_{W}_{m}")
            ps = {m: g("p", m) for m in ("KO", "SUB", "DEC")}
            pm = {m: g("mkt", m) for m in ("KO", "SUB", "DEC")}
            odds = {m: getattr(f, f"{W.lower()}_{m.lower()}_odds") for m in ("KO", "SUB", "DEC")}
            p_win_m = f.mkt_A_win_ml if W == "A" else 1 - f.mkt_A_win_ml
            dist_s, dist_m = f.p_distance, f.mkt_A_DEC + f.mkt_B_DEC
            void = f.winner not in ("A", "B") or f.method_cat not in ("KO", "SUB", "DEC")
            name = f.fighter_a if W == "A" else f.fighter_b
            cands = []
            if dist_s >= dist_m:
                cands.append(("goes distance", ["DEC"], dist_m))
            else:
                cands.append(("inside distance", ["KO", "SUB"], 1 - dist_m))
            best = max(("KO", "SUB", "DEC"), key=lambda m: ps[m] / max(pm[m], 1e-6))
            if not (best == "DEC" and dist_s >= dist_m):
                cands.append((f"by {best}", [best], np.nan))
            for lab, ms, pdist in cands:
                dec = 1 / sum(1 / od(odds[m]) for m in ms) if all(np.isfinite(odds[m]) for m in ms) else np.nan
                p_ind = p_win_m * (pdist if np.isfinite(pdist) else sum(pm[m] for m in ms) / max(p_win_m, 1e-6))
                y = np.nan if void else float(f.winner == W and f.method_cat in ms)
                batch.append(dict(sport="ufc", game_id=str(f.fight_id), date=date, season=int(f.season), n_legs=2,
                                  kinds="winner+" + ("distance" if len(ms) > 1 or lab.startswith("goes") else "method"),
                                  legs=f"{name} wins | {lab}", has_prop=False, all_real=True,
                                  p_script=float(sum(ps[m] for m in ms)),
                                  p_script_indep=float((f.p_A_win if W == "A" else 1 - f.p_A_win) *
                                                       (dist_s if lab == "goes distance" else (1 - dist_s) if lab == "inside distance"
                                                        else sum(ps[m] for m in ms) / max(f.p_A_win if W == "A" else 1 - f.p_A_win, 1e-6))),
                                  p_mkt=float(sum(pm[m] for m in ms)), p_hybrid=float(sum(ps[m] for m in ms)),
                                  p_indep=float(p_ind), dec_legs=dec, y=y))
        bd = pd.DataFrame(batch)
        bd["p_script_cal"] = cal.predict(B.cal_features(bd.p_script.values, bd.n_legs.values))
        bd["p_comb"] = comb.predict(B.comb_features(bd.p_script.values, bd.p_mkt.values, bd.n_legs.values))
        rows.append(bd)
        s = bd[bd.y.notna()]
        cal.update(B.cal_features(s.p_script.values, s.n_legs.values), s.y.values)
        comb.update(B.comb_features(s.p_script.values, s.p_mkt.values, s.n_legs.values), s.y.values)
    df = pd.concat(rows, ignore_index=True)
    # UFC's 'real price' is the exact method-prop price (the market sells this joint directly)
    df["dec_real"] = df.dec_legs
    print(f"  ufc: {len(df)} candidate SGPs")
    return df, dict(cal=cal.w.tolist(), comb=comb.w.tolist())


# ================================================================== evaluation
def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(p, y, bins=10):
    if len(p) < bins * 5:
        return np.nan
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame(dict(p=p, y=y, q=q)).groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    return float(np.average((d.p - d.y).abs(), weights=d.n))


PROBS = ["p_script", "p_script_cal", "p_hybrid", "p_comb", "p_mkt", "p_indep"]


def learning_curves(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    d = df[df.y.notna()]
    for (sp, s, kind), x in d.groupby(["sport", "season", "lineset"]):
        r = dict(sport=sp, season=s, lineset=kind, n=len(x), hit=x.y.mean())
        for p in PROBS:
            r[f"ll_{p}"] = _ll(x[p].values, x.y.values)
            r[f"ece_{p}"] = _ece(x[p].values, x.y.values)
            r[f"mean_{p}"] = x[p].mean()
        out.append(r)
    return pd.DataFrame(out)


def _dd(pnl: np.ndarray) -> float:
    c = np.cumsum(pnl)
    return float(np.max(np.maximum.accumulate(np.r_[0, c])[1:] - c)) if len(c) else 0.0


def _stats(x: pd.DataFrame, price: np.ndarray) -> dict:
    pnl = x.y.values * price - 1
    n = len(pnl)
    sd = pnl.std(ddof=1) if n > 1 else np.nan
    return dict(n=n, roi=pnl.mean() if n else np.nan, z=pnl.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else np.nan,
                maxdd=_dd(pnl), units=pnl.sum())


def pick_per_game(df: pd.DataFrame, prob: str) -> pd.DataFrame:
    d = df[df.y.notna()].copy()
    d["ratio"] = d[prob] / d.p_mkt.clip(lower=1e-6)
    return d.sort_values(["game_id", "ratio"], ascending=[True, False]).drop_duplicates(["sport", "game_id"])


def edge_test(df: pd.DataFrame, prob="p_script_cal", subset="real", train=lambda s: s <= TRAIN_END,
              test=lambda s: s > TRAIN_END, label="") -> list[dict]:
    """Choose k on train seasons (max ROI at 0% hold == max at any hold), report test under every hold."""
    out = []
    for sp, d0 in df.groupby("sport"):
        d0 = d0[d0.all_real] if subset == "real" else d0[~d0.all_real] if subset == "proxy" else d0
        if not len(d0):
            continue
        d = pick_per_game(d0, prob).sort_values("date")
        tr, te = d[d.season.map(train)], d[d.season.map(test)]
        if len(tr) < 50 or len(te) < 20:
            continue
        best, bk = -9, None
        nmin = max(100, int(0.05 * len(tr)))
        for k in KGRID:
            x = tr[tr.ratio >= k]
            if len(x) < nmin:
                continue
            roi = (x.y / x.p_mkt).mean() - 1
            if roi > best:
                best, bk = roi, k
        if bk is None:
            continue
        x = te[te.ratio >= bk]
        r = dict(sport=sp, label=label, prob=prob, subset=subset, k=bk, n_train=int((tr.ratio >= bk).sum()),
                 roi_train_h0=best, n_test=len(x), hit=x.y.mean(), p_dec_mean=x[prob].mean(), p_mkt_mean=x.p_mkt.mean(),
                 p_script_mean=x.p_script.mean())
        for h in HOLDS:
            s = _stats(x, (1 - h) / x.p_mkt.clip(lower=1e-6).values)
            r.update({f"roi_h{int(h*100)}": s["roi"], f"z_h{int(h*100)}": s["z"], f"dd_h{int(h*100)}": s["maxdd"]})
        if "dec_real" in x and x.dec_real.notna().any():
            xr = x[x.dec_real.notna()]
            s = _stats(xr, xr.dec_real.values)
            r.update(n_real=len(xr), roi_real=s["roi"], z_real=s["z"], dd_real=s["maxdd"])
        out.append(r)
    return out


# ================================================================== main
def main(sports):
    t0 = time.time()
    allc, state = [], {"rho": {}, "cal": {}, "comb": {}, "beta": {}, "proxy_rate": {}}
    for sp in sports:
        print(f"[{sp}] ...", flush=True)
        if sp == "ufc":
            d, st = run_ufc()
        else:
            d, st = run_sport(sp)
            state["rho"].update(st["rho"])
            state["beta"][sp] = st.get("beta", {})
            state["proxy_rate"][sp] = st.get("proxy_rate", {})
            # real-leg price with correlation adjustment: product of real leg prices x indep/joint (market-fair rho)
            d["dec_real"] = np.where(d.all_real & d.has_prop, d.dec_legs * d.p_indep / d.p_mkt.clip(lower=1e-6), np.nan)
        state["cal"][sp], state["comb"][sp] = st["cal"], st["comb"]
        allc.append(d)
    df = pd.concat(allc, ignore_index=True)
    df["lineset"] = np.where(df.has_prop, np.where(df.all_real, "props_real", "props_proxy"), "game_only")
    for c in ["p_script", "p_script_indep", "p_mkt", "p_hybrid", "p_indep", "p_script_cal", "p_comb", "dec_legs", "dec_real"]:
        df[c] = df[c].astype("float32")
    df.to_parquet(RS / "sgp_candidates.parquet", index=False)
    (RS / "sgp_state.json").write_text(json.dumps(state, indent=0))
    lc = learning_curves(df)
    lc.to_csv(RS / "sgp_learning_curves.csv", index=False)
    res = []
    res += edge_test(df, "p_script_cal", "real", label="primary: calibrated script, real-market legs")
    res += edge_test(df, "p_hybrid", "real", label="secondary: hybrid (script marginals + learned rho)")
    res += edge_test(df, "p_script", "real", label="raw script joint (uncalibrated)")
    res += edge_test(df, "p_script_cal", "proxy", label="proxy-line props (no real market)")
    nb = df[(df.sport == "nba") & (df.lineset == "props_real")]
    res += edge_test(nb, "p_script_cal", "real", train=lambda s: s == 2025, test=lambda s: s == 2026,
                     label="NBA real points props: choose 2025, report 2026")
    E = pd.DataFrame(res)
    E.to_csv(RS / "sgp_edge.csv", index=False)
    print(E.round(3).to_string())
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return df, lc, E


if __name__ == "__main__":
    sp = sys.argv[1:] or ["nfl", "nba", "ncaaf", "ncaam", "nhl", "mlb", "ufc"]
    main(sp)
