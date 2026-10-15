"""
NFL game-script model: each team's points, plays, pass attempts and rush attempts
from both teams' pre-game tendencies + context. Walk-forward by season (fit on
seasons < S, predict S). Two variants:
  * pure : no market inputs (used for side/total betting vs the close)
  * mkt  : adds the closing spread/total as inputs (used to condition player props)

usage: python -m engine.nfl_script.script
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest.core import perf, settle
from .tendencies import ROOT, load_games, tendencies

OUT = ROOT / "results"
TRAIN_FROM = 2002
DISC = (2006, 2015)
HOLD = (2016, 2025)
TARGETS = ["pts", "plays", "pass_att", "rush_att"]
KEY = ["epa_pass", "epa_rush", "sr", "sr_pass", "sr_rush", "expl", "neu_pass", "proe", "spp", "plays",
       "pass_att", "rush_att", "rz_td", "sack_rate", "to_rate", "pts"]
WIN = ["std", "l4", "l8"]
K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
MIN_N = 100


def rows() -> pd.DataFrame:
    """One row per (offense team, game) with own-offense / opp-defense / own-defense / opp-offense features."""
    T = tendencies()
    g = load_games()
    feat = [c for c in T.columns if any(c.endswith("_" + w) for w in WIN) and c != "games_std"]
    me = T[["game_id", "team", "opp", "season", "week", "game_type", "date", "is_home", "played",
            "pts_for", "off_n_play", "off_pass_att", "off_rush_att", "games_std"] + feat]
    op = T[["game_id", "team"] + feat].rename(columns={"team": "opp", **{c: "o_" + c for c in feat}})
    r = me.merge(op, on=["game_id", "opp"], how="left")
    r = r.rename(columns={"pts_for": "y_pts", "off_n_play": "y_plays", "off_pass_att": "y_pass_att",
                          "off_rush_att": "y_rush_att"})
    pc = [c for c in r.columns if "_proe_" in c]
    r[pc] = r[pc].fillna(0.0)  # xpass only exists from 2006: pre-2006 PROE = league average (0)
    ctx = g[["game_id", "location", "home_rest", "away_rest", "roof", "wind", "temp", "div_game",
             "spread_line", "total_line"]]
    r = r.merge(ctx, on="game_id", how="left")
    hm = r["is_home"].eq(1)
    r["neutral"] = r["location"].eq("Neutral").astype(int)
    r["home_adv"] = np.where(r["neutral"].eq(1), 0, np.where(hm, 1, -1))
    r["rest_diff"] = np.where(hm, r["home_rest"] - r["away_rest"], r["away_rest"] - r["home_rest"]).clip(-7, 7)
    r["short_rest"] = (np.where(hm, r["home_rest"], r["away_rest"]) <= 5).astype(int)
    r["dome"] = r["roof"].isin(["dome", "closed"]).astype(int)
    r["wind_x"] = np.where(r["dome"].eq(1), 0, r["wind"].fillna(8)).clip(0, 35)
    r["wind15"] = (r["wind_x"] >= 15).astype(int)
    r["cold"] = np.where(r["dome"].eq(1), 0, (r["temp"].fillna(60) < 32).astype(int))
    r["div"] = r["div_game"].fillna(0)
    r["playoff"] = r["game_type"].ne("REG").astype(int)
    r["early"] = (r["week"] <= 4).astype(int)
    # market-implied team script (only used by the 'mkt' variant)
    r["mkt_margin"] = np.where(hm, r["spread_line"], -r["spread_line"])
    r["mkt_total"] = r["total_line"]
    r["mkt_team_pts"] = (r["mkt_total"] + r["mkt_margin"]) / 2
    return r


def features(r: pd.DataFrame, variant: str) -> list[str]:
    f = []
    for k in KEY:
        for w in WIN:
            f += [f"off_{k}_{w}", f"o_def_{k}_{w}", f"def_{k}_{w}", f"o_off_{k}_{w}"]
    f += ["home_adv", "rest_diff", "short_rest", "dome", "wind_x", "wind15", "cold", "div", "playoff", "early"]
    if variant == "mkt":
        f += ["mkt_margin", "mkt_total", "mkt_team_pts"]
    return [c for c in f if c in r]


def walk_forward(r: pd.DataFrame, variant: str, first: int = 2006, alpha: float = 300.0) -> pd.DataFrame:
    F = features(r, variant)
    preds = []
    for S in sorted(r.season.unique()):
        if S < first:
            continue
        tr = r[(r.season >= TRAIN_FROM) & (r.season < S) & r.played]
        if variant == "mkt":
            tr = tr[tr["mkt_total"].notna()]
        te = r[r.season == S].copy()
        Xte = te[F].fillna(tr[F].median())
        for t in TARGETS:
            y = tr[f"y_{t}"]
            ok = y.notna()
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            m.fit(tr.loc[ok, F].fillna(tr[F].median()), y[ok])
            te[f"p_{t}"] = m.predict(Xte)
            te[f"res_sd_{t}"] = float(np.std(y[ok] - m.predict(tr.loc[ok, F].fillna(tr[F].median()))))
        preds.append(te)
    return pd.concat(preds, ignore_index=True)


def game_table(P: pd.DataFrame, Pm: pd.DataFrame | None = None) -> pd.DataFrame:
    g = load_games()[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
                      "home_score", "away_score", "spread_line", "total_line", "home_spread_odds",
                      "away_spread_odds", "over_odds", "under_odds", "wind", "roof"]]
    cols = ["game_id", "team"] + [f"p_{t}" for t in TARGETS]
    out = g
    for side in ("home", "away"):
        x = P[cols].rename(columns={"team": f"{side}_team", **{f"p_{t}": f"{side}_{t}" for t in TARGETS}})
        out = out.merge(x, on=["game_id", f"{side}_team"], how="left")
        if Pm is not None:
            xm = Pm[cols].rename(columns={"team": f"{side}_team", **{f"p_{t}": f"{side}_{t}_mkt" for t in TARGETS}})
            out = out.merge(xm, on=["game_id", f"{side}_team"], how="left")
    out["model_margin"] = out["home_pts"] - out["away_pts"]
    out["model_total"] = out["home_pts"] + out["away_pts"]
    out["actual_margin"] = out["home_score"] - out["away_score"]
    out["actual_total"] = out["home_score"] + out["away_score"]
    return out[out["home_pts"].notna()].reset_index(drop=True)


def bets(G: pd.DataFrame, k: float, market: str) -> pd.DataFrame:
    d = G.dropna(subset=["actual_margin"])
    if market == "side":
        d = d.dropna(subset=["spread_line", "home_spread_odds", "away_spread_odds"])
        e = d["model_margin"] - d["spread_line"]
        home = e >= k
        away = e <= -k
        d = d[home | away].copy()
        h = home[d.index]
        cover = np.sign(d["actual_margin"] - d["spread_line"])
        d["outcome"] = np.where(h, cover, -cover)
        d["odds"] = np.where(h, d["home_spread_odds"], d["away_spread_odds"])
        d["pick"] = np.where(h, "home", "away")
    else:
        d = d.dropna(subset=["total_line", "over_odds", "under_odds"])
        e = d["model_total"] - d["total_line"]
        ov, un = e >= k, e <= -k
        d = d[ov | un].copy()
        o = ov[d.index]
        s = np.sign(d["actual_total"] - d["total_line"])
        d["outcome"] = np.where(o, s, -s)
        d["odds"] = np.where(o, d["over_odds"], d["under_odds"])
        d["pick"] = np.where(o, "over", "under")
    d["pnl"] = settle(d["outcome"], d["odds"])
    return d


def info_beyond_market(G: pd.DataFrame, lo: int, hi: int, market: str) -> dict:
    d = G[(G.season >= lo) & (G.season <= hi)].dropna(subset=["actual_margin", "spread_line", "total_line"])
    if market == "side":
        y, x = d["actual_margin"] - d["spread_line"], d["model_margin"] - d["spread_line"]
        mae_m, mae_c = (d["actual_margin"] - d["model_margin"]).abs().mean(), (d["actual_margin"] - d["spread_line"]).abs().mean()
    else:
        y, x = d["actual_total"] - d["total_line"], d["model_total"] - d["total_line"]
        mae_m, mae_c = (d["actual_total"] - d["model_total"]).abs().mean(), (d["actual_total"] - d["total_line"]).abs().mean()
    X = np.c_[np.ones(len(x)), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    se = math.sqrt(res.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return dict(n=len(d), mae_model=mae_m, mae_close=mae_c, slope=b[1], t=b[1] / se)


def run():
    r = rows()
    P = walk_forward(r, "pure")
    Pm = walk_forward(r, "mkt")
    G = game_table(P, Pm)
    lines = []
    # volume / points accuracy per team-game
    for name, X in (("pure", P), ("mkt", Pm)):
        X = X[X.played & X.season.between(2006, 2025)]
        mae = {t: (X[f"y_{t}"] - X[f"p_{t}"]).abs().mean() for t in TARGETS}
        lines.append(f"team-game MAE ({name}): " + ", ".join(f"{t} {v:.2f}" for t, v in mae.items()))
    res = {}
    for market in ("side", "total"):
        for lab, (lo, hi) in (("disc", DISC), ("hold", HOLD)):
            res[(market, lab, "info")] = info_beyond_market(G, lo, hi, market)
        # choose k on discovery only
        best = None
        for k in K_GRID:
            b = bets(G, k, market)
            bd = b[b.season.between(*DISC)]
            if len(bd) < MIN_N:
                continue
            roi = bd["pnl"].mean()
            if best is None or roi > best[1]:
                best = (k, roi)
        k = best[0]
        b = bets(G, k, market)
        for lab, (lo, hi) in (("disc", DISC), ("hold", HOLD), ("2026", (2026, 2026))):
            x = b[b.season.between(lo, hi)]
            res[(market, lab, "bet")] = dict(k=k, **{kk: perf(x["pnl"]).get(kk) for kk in ("n", "roi", "z")})
        # every k on holdout for transparency
        res[(market, "grid")] = {kk: (perf(bets(G, kk, market).query("season>=2016 and season<=2025")["pnl"]).get("roi"),
                                      perf(bets(G, kk, market).query("season>=2006 and season<=2015")["pnl"]).get("roi"))
                                 for kk in K_GRID}
        # cross-check with the existing wind>=15 unders key
        if market == "total":
            bh = b[b.season.between(2006, 2025)]
            w = bh[(bh.wind >= 15) & ~bh.roof.isin(["dome", "closed"])]
            res[("total", "wind15")] = dict(n=len(w), under_share=float((w.pick == "under").mean()) if len(w) else np.nan,
                                            roi=float(w.pnl.mean()) if len(w) else np.nan)
    OUT.mkdir(exist_ok=True)
    keep = ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score",
            "home_pts", "away_pts", "model_margin", "model_total", "home_plays", "away_plays", "home_pass_att",
            "away_pass_att", "home_rush_att", "away_rush_att", "home_pts_mkt", "away_pts_mkt", "home_plays_mkt",
            "away_plays_mkt", "home_pass_att_mkt", "away_pass_att_mkt", "home_rush_att_mkt", "away_rush_att_mkt",
            "spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
            "actual_margin", "actual_total"]
    G[keep].round(3).to_csv(OUT / "nfl_script_games.csv", index=False)
    P.to_parquet(ROOT / "data" / "raw" / "nfl_pbp" / "_cache" / "script_pure.parquet")
    Pm.to_parquet(ROOT / "data" / "raw" / "nfl_pbp" / "_cache" / "script_mkt.parquet")
    return G, res, lines


if __name__ == "__main__":
    G, res, lines = run()
    print("\n".join(lines))
    for k, v in res.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v)
