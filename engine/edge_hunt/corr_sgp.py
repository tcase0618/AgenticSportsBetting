"""
Edge hunt: corr_sgp -- does a book that prices same-game parlays as the PRODUCT of the legs' prices
(no correlation adjustment; e.g. Courtside, a sweepstakes social book) underprice positively correlated legs?

Protocol: engine/edge_hunt/PROTOCOL.md.  Every leg is priced at its REAL historical market price:
  * NBA points props   : one main line per player-game (alt_line==0, latest snapshot, quote nearest -110/-110)
  * NBA spread/total   : ESPN pickcenter juice (2021-22 consensus, 2025-26 DraftKings close); 2024-25 has no
                          juice anywhere -> -110 assumed and FLAGGED (juice_flag=1).  ML: real closing ML.
  * NFL props 2026     : one quote per player-game-market (FanDuel > DraftKings > other, latest, nearest even);
                          QB passing yards supplemented from the DraftKings alt ladder (over side only, line priced
                          nearest -112 within [-130,+110]); NFL 2017-18 completions (propbets.football O/U prices)
  * NFL spread/total/ML: nflverse closing prices with real juice.
The parlay pays prod(decimal_i); a pushed leg drops out; a void (DNP / missing actual) leg is dropped from data.

Discovery = first 60% of dates of each dataset; lockbox = the rest, evaluated ONCE for the single discovery-chosen
config of each family.  Every knob is logged in results/edge_hunt/corr_sgp_candidates.csv.

usage:  python -m engine.edge_hunt.corr_sgp            # full study -> results/edge_hunt/corr_sgp.{md,_candidates.csv}
        from engine.edge_hunt.corr_sgp import build    # build(slate) for one game's lines
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "edge_hunt"
STATE = OUT / "corr_sgp_state.json"
VIGS = [0, 1, 2, 3, 4]          # extra implied-prob points of vig per leg for the break-even table


# ----------------------------------------------------------------------------- helpers
def dec(o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, 1 + o / 100.0, 1 + 100.0 / np.abs(o))


def american(d):
    d = float(d)
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def devig2(oa, ob):
    """multiplicative devig of a two-way market -> fair prob of side a (nan if b missing)."""
    ia, ib = 1 / dec(oa), 1 / dec(ob)
    return ia / (ia + ib)


def outcome(x, line):
    """1 win (x > line), 0 loss, -1 push."""
    x, line = np.asarray(x, float), np.asarray(line, float)
    return np.where(x > line, 1, np.where(x < line, 0, -1))


def flip(o):
    return np.where(o == -1, -1, 1 - o)


def date_split(dates: pd.Series, frac=0.6):
    u = np.sort(pd.Series(dates.unique()))
    return u[int(len(u) * frac) - 1]


# ----------------------------------------------------------------------------- NBA data
def _espn_nba_lines() -> pd.DataFrame:
    rows = []
    for yr, prov in [(2022, "consensus"), (2026, "DraftKings")]:
        f = ROOT / "data" / "raw" / "nba" / "espn_pickcenter" / f"nba_{yr}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            pcs = [p for p in r.get("pickcenter", []) if p.get("provider", {}).get("name") == prov] or []
            if not pcs:
                continue
            p = pcs[0]
            h, a = p.get("homeTeamOdds", {}), p.get("awayTeamOdds", {})
            rows.append(dict(game_id=int(r["game_id"]), e_spread=p.get("spread"), e_total=p.get("overUnder"),
                             e_over=p.get("overOdds"), e_under=p.get("underOdds"),
                             e_hso=h.get("spreadOdds"), e_aso=a.get("spreadOdds"),
                             e_hml=h.get("moneyLine"), e_aml=a.get("moneyLine")))
    return pd.DataFrame(rows)


def nba_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "results" / "scripts" / "nba_games.parquet")
    g = g[g.season >= 2022][["game_id", "date", "season", "home", "away", "spread_home", "total_line", "home_ml",
                             "away_ml", "home_score", "away_score"]].copy()
    g["game_id"] = g.game_id.astype(int)
    e = _espn_nba_lines()
    g = g.merge(e, on="game_id", how="left")
    ok = g.e_over.notna() & g.e_under.notna() & g.e_total.notna()
    g["total"] = np.where(ok, g.e_total, g.total_line)
    g["over_odds"] = np.where(ok, g.e_over, -110.0)
    g["under_odds"] = np.where(ok, g.e_under, -110.0)
    g["tot_flag"] = (~ok).astype(int)
    oks = g.e_hso.notna() & g.e_aso.notna() & g.e_spread.notna()
    g["spread_h"] = np.where(oks, g.e_spread, g.spread_home)
    g["hso"] = np.where(oks, g.e_hso, -110.0)
    g["aso"] = np.where(oks, g.e_aso, -110.0)
    g["spr_flag"] = (~oks).astype(int)
    g["hml"] = g.home_ml.fillna(g.e_hml)
    g["aml"] = g.away_ml.fillna(g.e_aml)
    return g.dropna(subset=["home_score", "total", "spread_h"])


def nba_props_main() -> pd.DataFrame:
    p = pd.read_csv(DATA / "nba_props.csv.gz", low_memory=False)
    p = p[(p.alt_line == 0) & p.over_odds.notna() & p.actual.notna() & p.game_id.notna() & p.player_id.notna()]
    p = p[(p.minutes_before_tip.isna()) | (p.minutes_before_tip >= 0)]
    p["game_id"] = p.game_id.astype(int)
    p["snap"] = pd.to_datetime(p.snapshot_ts, utc=True, errors="coerce")
    p["snap_rank"] = p.groupby(["game_id", "player_id"]).snap.rank(method="dense", ascending=False).fillna(1)
    p = p[p.snap_rank == 1]
    io = 1 / dec(p.over_odds)
    iu = 1 / dec(p.under_odds.fillna(-110))
    p["dist"] = np.abs(io - 0.5238) + np.abs(iu - 0.5238) + p.under_odds.isna().values * 0.2
    p = p.sort_values("dist").drop_duplicates(["game_id", "player_id"])
    p = p[["date", "season", "game_id", "player", "player_id", "team", "opp", "line", "over_odds", "under_odds",
           "actual", "book"]].copy()
    p["p_over"] = np.where(p.under_odds.notna(), devig2(p.over_odds, p.under_odds.fillna(-110)), np.nan)
    # walk-forward calibrated model prob (results/props_script/nba_legs.parquet, real_line universe)
    L = pd.read_parquet(ROOT / "results" / "props_script" / "nba_legs.parquet",
                        columns=["game_id", "player_id", "k", "p_over_cal", "universe"])
    L = L[L.universe == "real_line"].drop(columns="universe")
    L["game_id"], L["player_id"] = L.game_id.astype(int), L.player_id.astype(float)
    p = p.merge(L, on=["game_id", "player_id"], how="left")
    p.loc[(p.k - p.line).abs() > 0.01, "p_over_cal"] = np.nan
    # team-level line rank (top1 = highest posted line on his team: pre-game info)
    p["rank_tm"] = p.groupby(["game_id", "team"]).line.rank(method="first", ascending=False)
    p["o"] = outcome(p.actual, p.line)
    return p


# ----------------------------------------------------------------------------- NFL data
def nfl_games() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data" / "raw" / "nfl_games.csv")
    g = g[g.season.isin([2017, 2018, 2026])].dropna(subset=["home_score"])
    g = g[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score",
           "spread_line", "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds",
           "home_moneyline", "away_moneyline"]].copy()
    g["spread_h"] = -g.spread_line
    g = g.rename(columns={"total_line": "total", "home_spread_odds": "hso", "away_spread_odds": "aso",
                          "home_moneyline": "hml", "away_moneyline": "aml", "home_team": "home", "away_team": "away",
                          "gameday": "gdate"})
    g["tot_flag"] = (g.over_odds.isna() | g.under_odds.isna()).astype(int)
    g["spr_flag"] = (g.hso.isna() | g.aso.isna()).astype(int)
    for c in ["over_odds", "under_odds", "hso", "aso"]:
        g[c] = g[c].fillna(-110.0)
    return g


def nfl_props() -> pd.DataFrame:
    m = pd.read_csv(DATA / "nfl_props.csv")
    m = m[m.over_odds.notna() & m.under_odds.notna() & m.actual.notna() & m.team.notna()]
    m["prio"] = m.book.map({"FanDuel": 0, "DraftKings": 1}).fillna(2)
    m["snap"] = pd.to_datetime(m.snapshot_utc, utc=True, errors="coerce")
    m["dist"] = np.abs(1 / dec(m.over_odds) - 0.5238) + np.abs(1 / dec(m.under_odds) - 0.5238)
    m = m.sort_values(["prio", "snap", "dist"], ascending=[True, False, True])
    m = m.drop_duplicates(["game_id", "player", "market"])
    keep = ["season", "week", "date", "game_id", "player", "player_id", "position", "team", "opp", "market", "line",
            "over_odds", "under_odds", "actual", "book"]
    m = m[keep]
    # QB passing yards from DraftKings alt ladder (over side only) where no main quote exists
    a = pd.read_csv(DATA / "nfl_props_alt.csv.gz")
    a = a[(a.market == "passing_yards") & (a.book == "DraftKings") & a.over_odds.between(-130, 110) & a.actual.notna()
          & a.team.notna()]
    a["snap"] = pd.to_datetime(a.snapshot_utc, utc=True, errors="coerce")
    a["dist"] = (a.over_odds + 112).abs()
    a = a.sort_values(["snap", "dist"], ascending=[False, True]).drop_duplicates(["game_id", "player"])
    a["under_odds"] = np.nan
    have = set(zip(m.game_id, m.player, m.market))
    a = a[[k not in have for k in zip(a.game_id, a.player, a.market)]][keep]
    m = pd.concat([m, a], ignore_index=True)
    m["p_over"] = np.where(m.under_odds.notna(), devig2(m.over_odds, m.under_odds.fillna(-110)), np.nan)
    m["o"] = outcome(m.actual, m.line)
    L = pd.read_parquet(ROOT / "results" / "props_script" / "nfl_legs.parquet",
                        columns=["game_id", "player_id", "stat", "k", "p_over_cal", "universe"])
    L = L[L.universe == "real_line"].rename(columns={"stat": "market"}).drop(columns="universe")
    L = L.drop_duplicates(["game_id", "player_id", "market"])
    m = m.merge(L, on=["game_id", "player_id", "market"], how="left")
    m.loc[(m.k - m.line).abs() > 0.01, "p_over_cal"] = np.nan
    m["rank_tm"] = m.groupby(["game_id", "team", "market"]).line.rank(method="first", ascending=False)
    return m


# ----------------------------------------------------------------------------- game legs
def game_legs(g: pd.DataFrame, team: pd.Series, gid: pd.DataFrame) -> dict:
    """For each row of gid (with game columns merged) and team, the game-line legs from the team's perspective."""
    home = (team.values == gid.home.values)
    tm_sc = np.where(home, gid.home_score, gid.away_score)
    op_sc = np.where(home, gid.away_score, gid.home_score)
    spr = np.where(home, gid.spread_h, -gid.spread_h)
    so, oso = np.where(home, gid.hso, gid.aso), np.where(home, gid.aso, gid.hso)
    ml, oml = np.where(home, gid.hml, gid.aml), np.where(home, gid.aml, gid.hml)
    tot = gid.home_score.values + gid.away_score.values
    cov = outcome(tm_sc - op_sc + spr, 0)
    win = outcome(tm_sc - op_sc, 0)
    over = outcome(tot, gid.total.values)
    return dict(
        gameO=(dec(gid.over_odds), over, devig2(gid.over_odds, gid.under_odds), gid.tot_flag.values),
        gameU=(dec(gid.under_odds), flip(over), devig2(gid.under_odds, gid.over_odds), gid.tot_flag.values),
        cover=(dec(so), cov, devig2(so, oso), gid.spr_flag.values),
        oppcover=(dec(oso), flip(cov), devig2(oso, so), gid.spr_flag.values),
        ML=(dec(ml), win, devig2(ml, oml), np.zeros(len(gid))),
    )


# ----------------------------------------------------------------------------- parlay evaluation
def settle(D: np.ndarray, O: np.ndarray, extra_vig: float = 0.0, haircut: float = 0.0) -> np.ndarray:
    """D, O: (n, k) decimal prices and outcomes (1/0/-1 push).  Product pricing.  Returns profit per 1u."""
    Dv = 1 / (1 / D + extra_vig / 100.0)
    lose = (O == 0).any(1)
    pay = np.where(O == 1, Dv, 1.0).prod(1) * (1 - haircut)
    allpush = (O == -1).all(1)
    pay = np.where(allpush, 1.0, pay)
    return np.where(lose, -1.0, pay - 1.0)


def cluster_z(pnl: np.ndarray, cl: np.ndarray):
    n = len(pnl)
    if n < 2:
        return np.nan, np.nan
    m = pnl.mean()
    s = pd.Series(pnl - m).groupby(cl).sum().values
    G = len(s)
    se = math.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / n          # CR1 small-cluster correction
    z = m / se if se > 0 else np.nan
    # one-sided p from t with G-1 df (few clusters -> conservative)
    return z, (1 - stats.t.cdf(z, df=max(G - 1, 1))) if z == z else np.nan


def summarize(B: pd.DataFrame, k: int, blockcol: str) -> dict:
    """B has columns d0..d{k-1}, o0..o{k-1}, pf0.., date, game_id, block."""
    if len(B) == 0:
        return dict(n=0)
    D = B[[f"d{i}" for i in range(k)]].values
    O = B[[f"o{i}" for i in range(k)]].values
    P = B[[f"pf{i}" for i in range(k)]].values
    pnl = settle(D, O)
    nopush = ~(O == -1).any(1)
    Dn, On, Pn = D[nopush], O[nopush], P[nopush]
    joint = On.all(1).mean() if nopush.any() else np.nan
    prod_real = np.prod(On.mean(0)) if nopush.any() else np.nan
    prod_impl = (1 / Dn).prod(1).mean() if nopush.any() else np.nan
    prod_fair = np.nanmean(np.where(np.isnan(Pn), 1 / Dn, Pn).prod(1)) if nopush.any() else np.nan
    z, p = cluster_z(pnl, B.date.values)
    zg, pg = cluster_z(pnl, B.game_id.values)
    zb, pb = cluster_z(pnl, B[blockcol].values)
    blk = pd.Series(pnl).groupby(B[blockcol].values).sum()
    cum = np.cumsum(pnl)
    out = dict(n=len(B), n_games=B.game_id.nunique(), n_dates=B.date.nunique(), roi=pnl.mean(), units=pnl.sum(),
               z=z, p=p, z_game=zg, p_game=pg, z_block=zb, p_block=pb, maxdd=float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max()),
               joint=joint, prod_real=prod_real, prod_impl=prod_impl, prod_fair=prod_fair,
               lift_real=joint / prod_real if prod_real else np.nan,
               lift_fair=joint / prod_fair if prod_fair else np.nan,
               blocks_pos=int((blk > 0).sum()), blocks=len(blk),
               flag_share=float(B.flag.mean()) if "flag" in B else 0.0)
    for x in VIGS[1:]:
        out[f"roi_v{x}"] = settle(D, O, extra_vig=x).mean()
    # break-even extra vig per leg (implied-prob points) and break-even multiplier haircut
    f = lambda x: settle(D, O, extra_vig=x).mean()
    if f(0) > 0:
        lo, hi = 0.0, 25.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        out["be_vig"] = lo
        out["be_haircut"] = out["roi"] / (1 + out["roi"])
    else:
        out["be_vig"], out["be_haircut"] = 0.0, 0.0
    return out


# ----------------------------------------------------------------------------- pair construction
def _legs_from_props(P, side: str):
    """side 'O'/'U' -> (dec, outcome, fair prob) arrays for a prop frame."""
    if side == "O":
        return dec(P.over_odds), P.o.values, P.p_over.values
    return dec(P.under_odds), flip(P.o.values), 1 - P.p_over.values


def _frame(parts: list, base: pd.DataFrame, flags: list) -> pd.DataFrame:
    B = base[["date", "game_id", "block"]].reset_index(drop=True).copy()
    for i, (d, o, pf) in enumerate(parts):
        B[f"d{i}"], B[f"o{i}"], B[f"pf{i}"] = np.asarray(d, float), np.asarray(o, int), np.asarray(pf, float)
    B["flag"] = np.max(np.vstack([np.asarray(f, float) for f in flags] + [np.zeros(len(B))]), 0)
    B = B[np.isfinite(B[[c for c in B if c.startswith("d") and c[1:].isdigit()]]).all(1)]
    return B


def _gid(X):
    return X.drop(columns=["over_odds", "under_odds"]).rename(columns={"g_over": "over_odds", "g_under": "under_odds"})


def nba_scope(P, scope, side):
    if scope == "top1":
        P = P[P.rank_tm == 1]
    elif scope == "line20":
        P = P[P.line >= 20]
    elif scope == "line25":
        P = P[P.line >= 25]
    if side == "U":
        P = P[P.under_odds.notna()]
    return P


def nba_model(P, model, side):
    if model == "none":
        return P
    P = P[P.p_over_cal.notna() & P.p_over.notna()]
    e = P.p_over_cal - P.p_over
    return P[e >= 0] if side == "O" else P[e <= 0]


NBA_FAMILIES = {
    # name: (prop side, [game legs], kind)   kind 'pair' or 'stack'
    "nba_ptsO_gameO": ("O", ["gameO"]),
    "nba_ptsO_cover": ("O", ["cover"]),
    "nba_ptsO_ML": ("O", ["ML"]),
    "nba_ptsU_gameU": ("U", ["gameU"]),
    "nba_ptsU_oppcover": ("U", ["oppcover"]),
    "nba_ptsO_cover_gameO": ("O", ["cover", "gameO"]),
    "ctrl_nba_ptsO_gameU": ("O", ["gameU"]),
    "ctrl_nba_ptsO_oppcover": ("O", ["oppcover"]),
}


def nba_candidates(P: pd.DataFrame, G: pd.DataFrame):
    X = P.merge(G[["game_id", "home", "away", "home_score", "away_score", "spread_h", "hso", "aso", "hml", "aml",
                   "total", "over_odds", "under_odds", "tot_flag", "spr_flag"]].rename(
        columns={"over_odds": "g_over", "under_odds": "g_under"}), on="game_id", how="inner")
    X = X[(X.team == X.home) | (X.team == X.away)]
    X = X.reset_index(drop=True)
    gl = game_legs(None, X.team, _gid(X))
    out = {}
    for fam, (side, glegs) in NBA_FAMILIES.items():
        for scope in ["all", "top1", "line20", "line25"]:
            for model in ["none", "agree"]:
                Q = nba_model(nba_scope(X, scope, side), model, side)
                if len(Q) == 0:
                    continue
                idx = Q.index.values
                parts = [_legs_from_props(Q, side)]
                flags = []
                for gname in glegs:
                    d, o, pf, fl = gl[gname]
                    parts.append((d[idx], o[idx], pf[idx]))
                    flags.append(fl[idx])
                out[(fam, f"scope={scope} model={model}")] = (_frame(parts, Q, flags), 1 + len(glegs))
    # 3-leg: both teams' top scorer OVER + game OVER  /  both UNDER + game UNDER
    T = X[X.rank_tm == 1]
    for fam, side, gname in [("nba_2topO_gameO", "O", "gameO"), ("nba_2topU_gameU", "U", "gameU")]:
        for model in ["none", "agree"]:
            Q = nba_model(T if side == "O" else T[T.under_odds.notna()], model, side)
            h, a = Q[Q.team == Q.home], Q[Q.team == Q.away]
            M = h.merge(a, on="game_id", suffixes=("", "_b"))
            if len(M) == 0:
                continue
            ph, pa = _legs_from_props(M, side), _legs_from_props(
                M[[c for c in M if c.endswith("_b")]].rename(columns=lambda c: c[:-2]), side)
            g2 = game_legs(None, M.team, _gid(M))[gname]
            out[(fam, f"scope=top1 model={model}")] = (_frame([ph, pa, g2[:3]], M, [g2[3]]), 3)
    return out


NFL_FAMILIES = {
    # (QB/prop A spec, prop B spec or None, game legs)
    "nfl_QBpyO_RECydsO": (("QB", "passing_yards", "O"), (("WR", "TE"), "receiving_yards", "O"), []),
    "nfl_QBpyU_RECydsU": (("QB", "passing_yards", "U"), (("WR", "TE"), "receiving_yards", "U"), []),
    "nfl_QBpyO_RECrecO": (("QB", "passing_yards", "O"), (("WR", "TE"), "receptions", "O"), []),
    "nfl_QBpyO_RECydsO_gameO": (("QB", "passing_yards", "O"), (("WR", "TE"), "receiving_yards", "O"), ["gameO"]),
    "nfl_RBrushO_cover": (("RB", "rushing_yards", "O"), None, ["cover"]),
    "nfl_RBrushO_ML": (("RB", "rushing_yards", "O"), None, ["ML"]),
    "nfl_RBrushU_oppcover": (("RB", "rushing_yards", "U"), None, ["oppcover"]),
    "nfl_RECydsO_gameO": ((("WR", "TE"), "receiving_yards", "O"), None, ["gameO"]),
    "nfl_QBpyO_gameO": (("QB", "passing_yards", "O"), None, ["gameO"]),
    "nfl_QBcmpO_gameO": (("QB", "completions", "O"), None, ["gameO"]),
    "nfl_RBrushO_cover_gameO": (("RB", "rushing_yards", "O"), None, ["cover", "gameO"]),
    "ctrl_nfl_RBrushO_oppcover": (("RB", "rushing_yards", "O"), None, ["oppcover"]),
    "ctrl_nfl_RECydsO_gameU": ((("WR", "TE"), "receiving_yards", "O"), None, ["gameU"]),
}


def _sel(M, pos, market, side, scope, model):
    pos = (pos,) if isinstance(pos, str) else pos
    Q = M[M.position.isin(pos) & (M.market == market)]
    if side == "U":
        Q = Q[Q.under_odds.notna()]
    if scope == "top1":
        Q = Q[Q.rank_tm == 1]
    if model == "agree":
        Q = Q[Q.p_over_cal.notna() & Q.p_over.notna()]
        e = Q.p_over_cal - Q.p_over
        Q = Q[e >= 0] if side == "O" else Q[e <= 0]
    return Q


def nfl_candidates(M: pd.DataFrame, G: pd.DataFrame):
    M = M.merge(G[["game_id", "home", "away", "home_score", "away_score", "spread_h", "hso", "aso", "hml", "aml",
                   "total", "over_odds", "under_odds", "tot_flag", "spr_flag"]].rename(
        columns={"over_odds": "g_over", "under_odds": "g_under"}), on="game_id", how="inner")
    M = M[(M.team == M.home) | (M.team == M.away)]
    out = {}
    for fam, (a, b, glegs) in NFL_FAMILIES.items():
        for scope in ["all", "top1"]:
            for model in ["none", "agree"]:
                A = _sel(M, *a, "all", model if b is None else "none")
                if b is None:
                    A = _sel(M, *a, scope, model)
                    X = A.reset_index(drop=True)
                    parts = [_legs_from_props(X, a[2])]
                else:
                    Bq = _sel(M, *b, scope, model)
                    X = A.merge(Bq[["game_id", "team", "over_odds", "under_odds", "o", "p_over"]],
                                on=["game_id", "team"], suffixes=("", "_b")).reset_index(drop=True)
                    parts = [_legs_from_props(X, a[2]),
                             _legs_from_props(X[["over_odds_b", "under_odds_b", "o_b", "p_over_b"]].rename(
                                 columns=lambda c: c[:-2]), b[2])]
                if len(X) == 0:
                    continue
                gl = game_legs(None, X.team, _gid(X))
                flags = []
                for gname in glegs:
                    d, o, pf, fl = gl[gname]
                    parts.append((d, o, pf))
                    flags.append(fl)
                X = X.copy()
                X["date"] = pd.to_datetime(X.date).dt.strftime("%Y-%m-%d")
                X["block"] = X.season.astype(str) + "-w" + X.week.astype(int).astype(str).str.zfill(2)
                out[(fam, f"scope={scope} model={model}")] = (_frame(parts, X, flags), len(parts))
    return out


# ----------------------------------------------------------------------------- main study
def tetra(joint, pa, pb):
    """tetrachoric rho such that P(A,B)=joint for latent normals with marginals pa,pb."""
    from scipy.stats import multivariate_normal
    ta, tb = stats.norm.ppf(pa), stats.norm.ppf(pb)
    f = lambda r: multivariate_normal.cdf([ta, tb], cov=[[1, r], [r, 1]]) - joint
    lo, hi = -0.95, 0.95
    if f(lo) * f(hi) > 0:
        return np.nan
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


# long-run learned latent correlations (results/scripts/sgp_state.json, many seasons, shrunk n/(n+300))
PRIOR_KEY = {
    "nba_ptsO_gameO": ("nba|total-prop|pts", 1), "nba_ptsU_gameU": ("nba|total-prop|pts", 1),
    "nba_ptsO_cover": ("nba|side-prop|pts", 1), "nba_ptsU_oppcover": ("nba|side-prop|pts", 1),
    "nba_ptsO_ML": ("nba|ml-prop|pts", 1),
    "ctrl_nba_ptsO_gameU": ("nba|total-prop|pts", -1), "ctrl_nba_ptsO_oppcover": ("nba|side-prop|pts", -1),
    "nfl_QBpyO_RECydsO": ("nfl|pp-tm|passing_yards|receiving_yards", 1),
    "nfl_QBpyU_RECydsU": ("nfl|pp-tm|passing_yards|receiving_yards", 1),
    "nfl_QBpyO_RECrecO": ("nfl|pp-tm|passing_yards|receptions", 1),
    "nfl_RBrushO_cover": ("nfl|side-prop|rushing_yards", 1), "nfl_RBrushU_oppcover": ("nfl|side-prop|rushing_yards", 1),
    "nfl_RBrushO_ML": ("nfl|ml-prop|rushing_yards", 1),
    "nfl_RECydsO_gameO": ("nfl|total-prop|receiving_yards", 1), "nfl_QBpyO_gameO": ("nfl|total-prop|passing_yards", 1),
    "ctrl_nfl_RBrushO_oppcover": ("nfl|side-prop|rushing_yards", -1),
    "ctrl_nfl_RECydsO_gameU": ("nfl|total-prop|receiving_yards", -1),
}


def prior_ev(B: pd.DataFrame, rho: float) -> dict:
    """Expected ROI of the 2-leg sample at product pricing if the TRUE latent correlation is the long-run learned rho
    and the de-vigged market marginals are fair (no luck in the realized outcomes)."""
    from scipy.stats import multivariate_normal
    D = B[["d0", "d1"]].values
    P = B[["pf0", "pf1"]].values
    P = np.where(np.isnan(P), (1 / D) / 1.0476, P).clip(0.02, 0.98)
    cache = {}
    pj = np.empty(len(P))
    for i, (a, b) in enumerate(np.round(P, 3)):
        key = (a, b)
        if key not in cache:
            cache[key] = multivariate_normal.cdf([stats.norm.ppf(a), stats.norm.ppf(b)], cov=[[1, rho], [rho, 1]])
        pj[i] = cache[key]
    ev = lambda x: float((pj * (1 / (1 / D + x / 100.0)).prod(1)).mean() - 1)
    out = dict(rho_prior=rho, ev_prior=ev(0), lift_prior=float((pj / P.prod(1)).mean()))
    if ev(0) > 0:
        lo, hi = 0.0, 25.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ev(mid) > 0 else (lo, mid)
        out["be_vig_prior"] = lo
    else:
        out["be_vig_prior"] = 0.0
    out["be_haircut_prior"] = max(out["ev_prior"], 0) / (1 + out["ev_prior"])
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    global SGP_RHO
    SGP_RHO = json.load(open(ROOT / "results" / "scripts" / "sgp_state.json"))["rho"]
    rows, store = [], {}
    # ---------------- NBA
    P = nba_props_main()
    G = nba_games()
    P["date"] = pd.to_datetime(P.date).dt.strftime("%Y-%m-%d")
    P["block"] = P.date.str[:7]
    cut = {"nba": date_split(P.date)}
    for key, (B, k) in nba_candidates(P, G).items():
        store[("nba",) + key] = (B, k)
    # ---------------- NFL
    M = nfl_props()
    GN = nfl_games()
    for key, (B, k) in nfl_candidates(M, GN).items():
        store[("nfl",) + key] = (B, k)
    nfl26 = pd.to_datetime(M[M.season == 2026].date).dt.strftime("%Y-%m-%d")
    nfl17 = pd.to_datetime(M[M.season < 2019].date).dt.strftime("%Y-%m-%d")
    cut["nfl26"], cut["nfl17"] = date_split(nfl26), date_split(nfl17)

    def cutoff(sport, B):
        if sport == "nba":
            return cut["nba"]
        return cut["nfl17"] if B.date.min() < "2020" else cut["nfl26"]

    nmin = {"nba": 200, "nfl": 25}
    finals = []
    for (sport, fam, cfg), (B, k) in store.items():
        c = cutoff(sport, B)
        Dd, Dl = B[B.date <= c], B[B.date > c]
        s = summarize(Dd, k, "block")
        rows.append(dict(sport=sport, family=fam, config=cfg, legs=k, split=f"disc<= {c}",
                         **{f"disc_{a}": s.get(a) for a in ["n", "roi", "z", "joint", "prod_real", "prod_impl",
                                                             "lift_real", "lift_fair", "be_vig", "flag_share"]}))
    cand = pd.DataFrame(rows)
    # choose one config per family on discovery only
    cand["chosen"] = False
    for (sport, fam), grp in cand.groupby(["sport", "family"]):
        ok = grp[(grp.disc_n >= nmin[sport])]
        if len(ok) == 0:
            ok = grp[grp.disc_n >= 10]
        if len(ok) == 0:
            continue
        best = ok.sort_values("disc_roi", ascending=False).index[0]
        cand.loc[best, "chosen"] = True
    # lockbox: evaluated ONCE per chosen config
    fin = []
    for i, r in cand[cand.chosen].iterrows():
        B, k = store[(r.sport, r.family, r.config)]
        c = cutoff(r.sport, B)
        sl = summarize(B[B.date > c], k, "block")
        sd = summarize(B[B.date <= c], k, "block")
        sf = summarize(B, k, "block")
        rho = np.nan
        if k == 2 and sf["n"] > 30:
            O = B[["o0", "o1"]].values
            O = O[~(O == -1).any(1)]
            rho = tetra(O.all(1).mean(), O[:, 0].mean(), O[:, 1].mean())
        for a in ["n", "roi", "z", "p", "z_game", "p_game", "z_block", "p_block", "units", "maxdd", "joint", "prod_real", "lift_real",
                  "lift_fair", "be_vig", "be_haircut", "flag_share"] + [f"roi_v{x}" for x in VIGS[1:]]:
            cand.loc[i, f"lock_{a}"] = sl.get(a)
        pr = {}
        if r.family in PRIOR_KEY:
            kk, sg = PRIOR_KEY[r.family]
            pr = prior_ev(B, sg * SGP_RHO[kk]["rho"])
            pr["prior_n"] = SGP_RHO[kk]["n"]
        fin.append(dict(sport=r.sport, family=r.family, config=r.config, legs=k, control=r.family.startswith("ctrl"),
                        disc=sd, lock=sl, full=sf, rho=rho, prior=pr))
    # significance across finals (controls excluded from the family count)
    F = [f for f in fin if not f["control"]]
    nF = len(F)
    pv = np.array([max(f["lock"].get("p", 1) if f["lock"].get("n", 0) else 1,
                       f["lock"].get("p_game", 1) if f["lock"].get("n", 0) else 1) for f in F], float)
    pv = np.nan_to_num(pv, nan=1.0)
    order = np.argsort(pv)
    q = np.empty(nF)
    q[order] = np.minimum.accumulate((pv[order] * nF / np.arange(1, nF + 1))[::-1])[::-1]
    for f, p_, q_ in zip(F, pv, q):
        f["p_use"], f["q"] = p_, min(q_, 1)
        L, D_, Fu = f["lock"], f["disc"], f["full"]
        f["verdict"] = bool(D_.get("roi", -1) > 0 and L.get("n", 0) >= 50 and L.get("roi", -1) > 0
                            and (p_ < 0.05 / nF or q_ < 0.10)
                            and Fu.get("blocks", 0) and Fu["blocks_pos"] / Fu["blocks"] >= 0.6)
    for f in fin:
        f.setdefault("verdict", False)
        f.setdefault("p_use", np.nan)
        f.setdefault("q", np.nan)
    cand.to_csv(OUT / "corr_sgp_candidates.csv", index=False)
    state = {f"{f['sport']}|{f['family']}": dict(config=f["config"], rho=f["rho"], verdict=f["verdict"],
                                                 rho_prior=f["prior"].get("rho_prior"),
                                                 lift=f["full"].get("lift_real"), be_vig=f["full"].get("be_vig"))
             for f in fin}
    json.dump(state, open(STATE, "w"), indent=1, default=float)
    report(fin, cand, cut, nF)
    return fin, cand


def _f(x, fmt="{:+.3f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


NOTES = """
## Data actually available (same-game pairs with REAL leg prices)
- **NBA points** (one main line per player-game; 2021-22 BettingPros consensus, 2024-25 and 2025-26 Odds-API books):
  31.1k player-games, ~8.3k usable pairs per game-leg family. Game total/spread juice real in 2021-22 and 2025-26,
  assumed -110 in 2024-25 (flag column). ML always real.
- **NFL 2026 weeks 1-6 only** (~79 graded games): receiving/rushing/receptions mains, QB passing yards weeks 3-6
  only (FanDuel main + DraftKings alt ladder overs). QB+WR1 pairs: ~95. NFL 2017-18 has only QB completions
  (no receiver props), so QB+WR stacks cannot be tested before 2026; completions were paired with the game total.
- NHL/MLB 2026 prop files (1.7k / 3.1k rows, a few weeks) were not tested: too few same-game pairs with a graded,
  priced game leg to clear any multiple-testing bar.

## Knobs (all logged in corr_sgp_candidates.csv)
NBA: scope in {all, top1 (highest posted line on the team), line>=20, line>=25} x model in {none, agree (walk-forward
calibrated p_over from results/props_script/nba_legs.parquet on the leg's side of the de-vigged price)}; 3-leg stacks:
both teams' top scorer O + game O, both U + game U. NFL: scope in {all, top1 (WR1/RB1 by posted line)} x model in
{none, agree (nfl_legs p_over_cal)}. All selectors use pre-game information only.

## How to read the two break-even columns
*Sample* columns use realized outcomes (includes luck in the individual legs). *Long-run rho* columns assume the
de-vigged leg prices are fair and the true latent correlation equals the one learned in results/scripts/sgp_state.json
over thousands of games; that is the expected edge a product-pricing book concedes. Rule of thumb: with two -110 legs
the product price already carries ~9.8% parlay overround, so the joint must beat independence by a lift > 1.098.
NBA points vs game total/side (rho 0.13-0.17) produces a lift of only ~1.06-1.09 at fair marginals -> NEGATIVE EV
even with zero extra vig. NFL RB rushing vs team side/ML (rho ~0.22) and QB passing vs game total (rho ~0.26) give
~+3-5% EV at market leg prices, which dies after ~0.8-1.3 implied-prob points of extra vig per leg, or a ~3-5%
parlay-multiplier haircut.

## Mechanism
A book that multiplies leg prices treats the legs as independent; game script ties an RB's carries to his team
leading (and a QB's yardage to the game total), so the joint outcome is more likely than the product. The other side
is the book's pricing engine; FanDuel/DraftKings correlate SGPs explicitly, so the edge exists only at a book that
truly multiplies market-priced legs -- unverified for Courtside.

## * Caveat on the TRUE verdict
PROTOCOL rule 1 requires a price you could actually bet. Courtside's SGP prices are NOT observed: the parlay price
here is the ASSUMED product of other books' market leg prices. The statistical gates are met for NFL RB rushing UNDER +
opponent covers, but on 2026 weeks 1-6 only (lockbox = 33 games, 7 dates); the realized lockbox ROI (+31%) is ~10x
the +3% expected from the long-run correlation, i.e. mostly favourable variance on top of a small real effect.
Treat it as: real structural edge of ~+3% per pair IF Courtside multiplies market-level leg prices with <= ~0.8 pt of
extra vig per leg and no correlation haircut; otherwise none.
"""


def report(fin, cand, cut, nF):
    L = ["# Edge hunt: corr_sgp -- product-priced same-game parlays (Courtside hypothesis)", "",
         "Protocol: engine/edge_hunt/PROTOCOL.md. Code: engine/edge_hunt/corr_sgp.py (`build(slate)` for live use). "
         f"Every knob: corr_sgp_candidates.csv (n_tests = {len(cand)}). Discovery = first 60% of dates per dataset "
         f"(NBA <= {cut['nba']}, NFL-2026 <= {cut['nfl26']}, NFL-2017/18 <= {cut['nfl17']}); lockbox = rest, evaluated "
         f"ONCE for the single discovery-chosen config (max discovery ROI, n >= 200 NBA / 25 NFL) per family. "
         f"Significance: one-sided p < 0.05/{nF} or BH q < 0.10 across the {nF} non-control finals, using the LARGER "
         "of the date-clustered and game-clustered p. TRUE also needs disc ROI > 0, lockbox n >= 50, >= 60% positive "
         "blocks (months NBA, weeks NFL) over the full sample.", "",
         "**Pricing assumption.** Courtside's own SGP prices are unknown; each leg is priced at its real historical "
         "market price and the parlay pays the product of decimals. NBA spread/total juice is real for 2021-22 (ESPN "
         "consensus) and 2025-26 (DraftKings close) but MISSING for 2024-25 -> -110 assumed there (`flag` = share of "
         "bets with an assumed -110 leg). NFL game juice is real (nflverse). Pushed legs drop out; DNP voids dropped.",
         "", "## Verdicts", "",
         "| candidate (discovery-chosen config) | disc n / ROI | lockbox n / ROI / z / p (week/month-cluster p) | joint vs naive lift "
         "(full; joint / prod realized marginals) | max per-leg vig before edge dies (full / lockbox) | blocks + | BH q | "
         "-110 flag | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for f in fin:
        D_, Lk, Fu = f["disc"], f["lock"], f["full"]
        ver = "control" if f["control"] else ("**TRUE*** (conditional)" if f["verdict"] else "FALSE")
        L.append(f"| {f['sport'].upper()} {f['family']} `{f['config']}` | {D_.get('n', 0)} / {_f(D_.get('roi'))} | "
                 f"{Lk.get('n', 0)} / {_f(Lk.get('roi'))} / {_f(Lk.get('z'), '{:+.2f}')} / "
                 f"{_f(f['p_use'] if not f['control'] else Lk.get('p'), '{:.3f}')} ({_f(Lk.get('p_block'), '{:.3f}')}) | "
                 f"{_f(Fu.get('lift_real'), '{:.3f}')} (joint {_f(Fu.get('joint'), '{:.3f}')} vs "
                 f"{_f(Fu.get('prod_real'), '{:.3f}')}; implied {_f(Fu.get('prod_impl'), '{:.3f}')}) | "
                 f"{_f(Fu.get('be_vig'), '{:.2f}')} / {_f(Lk.get('be_vig'), '{:.2f}')} pts | "
                 f"{Fu.get('blocks_pos', 0)}/{Fu.get('blocks', 0)} | {_f(f['q'], '{:.3f}')} | "
                 f"{_f(Fu.get('flag_share'), '{:.0%}')} | {ver} |")
    L += ["", "## Break-even table (full sample of each final; ROI at product pricing with x extra implied-prob points "
          "of vig per leg; haircut = parlay-multiplier cut that zeroes ROI)", "",
          "| candidate | legs | n | ROI +0 | +1 pt | +2 pt | +3 pt | +4 pt | break-even vig/leg | break-even haircut | "
          "latent rho (sample) | long-run rho (n) | EV at long-run rho | break-even vig/leg at long-run rho | "
          "break-even haircut at long-run rho |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fin:
        Fu = f["full"]
        L.append(f"| {f['sport'].upper()} {f['family']} | {f['legs']} | {Fu.get('n', 0)} | {_f(Fu.get('roi'))} | "
                 + " | ".join(_f(Fu.get(f"roi_v{x}")) for x in VIGS[1:])
                 + f" | {_f(Fu.get('be_vig'), '{:.2f}')} pts | {_f(Fu.get('be_haircut'), '{:.1%}')} | "
                   f"{_f(f['rho'], '{:+.2f}')} | {_f(f['prior'].get('rho_prior'), '{:+.2f}')} "
                   f"({f['prior'].get('prior_n', '-')}) | {_f(f['prior'].get('ev_prior'))} | "
                   f"{_f(f['prior'].get('be_vig_prior'), '{:.2f}')} pts | {_f(f['prior'].get('be_haircut_prior'), '{:.1%}')} |")
    (OUT / "corr_sgp.md").write_text("\n".join(L) + "\n" + NOTES)
    print("\n".join(L))


# ----------------------------------------------------------------------------- live: build(slate)
def _bvn(pa, pb, rho):
    from scipy.stats import multivariate_normal
    return float(multivariate_normal.cdf([stats.norm.ppf(pa), stats.norm.ppf(pb)], cov=[[1, rho], [rho, 1]]))


def build(slate: dict, include_failed: bool = False) -> pd.DataFrame:
    """slate = one game's lines:
        dict(sport='nba'|'nfl', home='BOS', away='NY', spread_home=-5.5, spread_home_odds=-110, spread_away_odds=-110,
             total=221.5, over_odds=-110, under_odds=-110, home_ml=-220, away_ml=180,
             props=[dict(player, team, position (NFL), market ('pts' NBA; NFL markets), line, over_odds, under_odds), ...])
    Fair joint uses the LONG-RUN learned latent rho (results/scripts/sgp_state.json) where one exists.
    Returns the correlated pairs whose family PASSED the lockbox (all families if include_failed) with the fair joint
    prob (devig marginals + latent-normal copula at the learned rho), the product-priced odds a product-pricing book
    pays, and the minimum price needed (fair joint, i.e. break-even).  Pair 'ok' = product price >= minimum price."""
    st = json.load(open(STATE)) if STATE.exists() else {}
    sport = slate["sport"]
    rows = []
    gm = {"gameO": (slate.get("over_odds", -110), slate.get("under_odds", -110))}

    def gleg(name, team):
        home = team == slate["home"]
        so, oso = (slate.get("spread_home_odds", -110), slate.get("spread_away_odds", -110))
        if not home:
            so, oso = oso, so
        ml, oml = (slate.get("home_ml"), slate.get("away_ml")) if home else (slate.get("away_ml"), slate.get("home_ml"))
        m = {"gameO": (slate.get("over_odds", -110), slate.get("under_odds", -110)),
             "gameU": (slate.get("under_odds", -110), slate.get("over_odds", -110)),
             "cover": (so, oso), "oppcover": (oso, so), "ML": (ml, oml)}[name]
        return None if m[0] is None or m[1] is None else (float(dec(m[0])), float(devig2(m[0], m[1])))

    for p in slate.get("props", []):
        po = float(devig2(p["over_odds"], p["under_odds"])) if p.get("under_odds") is not None else 0.5
        for key, v in st.items():
            sp, fam = key.split("|")
            if sp != sport or fam.startswith("ctrl") or (not v.get("verdict") and not include_failed):
                continue
            # conservative: long-run learned rho (sgp_state, thousands of games) when available -- the small 2026
            # sample rhos (e.g. +0.43 RB/cover) are noisy-high; else the sample rho shrunk by half
            rho = v.get("rho_prior")
            if rho is None or not np.isfinite(rho):
                rho = v.get("rho")
                if rho is None or not np.isfinite(rho):
                    continue
                rho = 0.5 * rho
            if sport == "nba":
                spec = NBA_FAMILIES.get(fam)
                if not spec or len(spec[1]) != 1 or p.get("market", "pts") != "pts":
                    continue
                side, (gname,) = spec[0], spec[1]
            else:
                spec = NFL_FAMILIES.get(fam)
                if not spec or spec[1] is not None or len(spec[2]) != 1:
                    continue
                (pos, market, side), (gname,) = spec[0], spec[2]
                pos = (pos,) if isinstance(pos, str) else pos
                if p.get("position") not in pos or p.get("market") != market:
                    continue
            if side == "U" and p.get("under_odds") is None:
                continue
            d1 = float(dec(p["over_odds"] if side == "O" else p["under_odds"]))
            p1 = po if side == "O" else 1 - po
            g = gleg(gname, p["team"])
            if g is None:
                continue
            d2, p2 = g
            pj = _bvn(p1, p2, rho)
            prod = d1 * d2
            rows.append(dict(family=fam, player=p["player"], team=p["team"], leg1=f"{p['market']} {side} {p['line']}",
                             leg2=gname, rho=rho, p_leg1=p1, p_leg2=p2, p_joint_fair=pj, p_indep=p1 * p2,
                             product_decimal=prod, product_american=american(prod),
                             min_decimal=1 / pj, min_american=american(1 / pj), ev=pj * prod - 1,
                             ok=prod >= 1 / pj, family_passed=bool(v.get("verdict"))))
    return pd.DataFrame(rows).sort_values("ev", ascending=False) if rows else pd.DataFrame()


def scan(sport: str = "nba") -> pd.DataFrame:
    """Live: FanDuel board (sbapi) -> build() per game.  Only families that PASSED are returned (none did)."""
    try:
        from nightly import providers as PV
        board = PV.fanduel_board(sport)
        props = PV.fanduel_nba_points(board) if sport == "nba" and len(board) else pd.DataFrame()
    except Exception as e:
        print("scan: FanDuel unavailable:", e)
        return pd.DataFrame()
    out = []
    for _, g in board.iterrows():
        pr = props[(props.home_team == g.home_name)] if len(props) else props
        sl = dict(sport=sport, home=g.home_name, away=g.away_name, spread_home=g.get("spread_home"),
                  spread_home_odds=g.get("spread_home_odds"), spread_away_odds=g.get("spread_away_odds"),
                  total=g.get("total"), over_odds=g.get("over_odds"), under_odds=g.get("under_odds"),
                  home_ml=g.get("home_ml"), away_ml=g.get("away_ml"),
                  props=[dict(player=r.player, team=g.home_name, market="pts", line=r.line, over_odds=r.over_odds,
                              under_odds=r.under_odds) for r in pr.itertuples()])  # team unknown on FD feed
        b = build(sl)
        if len(b):
            out.append(b.assign(game=f"{g.away_name}@{g.home_name}"))
    return pd.concat(out) if out else pd.DataFrame()


if __name__ == "__main__":
    run()
