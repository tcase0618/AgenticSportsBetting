"""
Script-conditional NBA player points projection.

Two projections, both walk-forward (season S uses only seasons < S):

  proj_mech   mechanistic:  share-of-team-points (shifted rolling)  x  script-implied team points
              x  blowout minutes adjustment  x  opponent positional-defence multiplier^k,
              then a linear recalibration fit on prior seasons.
  proj_script props_nba's HistGradientBoosting points model (same FEATS, same hyper-parameters,
              same 8-season window) PLUS script features (team/opp implied points, pace, blowout
              probability, spread, positional defence, share, proj_mech).  So the only difference vs the
              baseline `proj` is the game script.
  proj_script_nm  same, but the script comes from the tendency-only variant (no betting line at all) ->
              robustness check against any timing advantage of closing lines.

Market lines used by the script ('pre' variant, see script.py): OPENING spread/total when nba.csv has it,
else the closing line.  Openers are missing for seasons 2023-25 and 19% of 2026, so for 2025 the script
sees closing lines, which post-date the ~12h-pre-tip prop snapshots.  Treat 2025 'pre' results as
slightly optimistic; compare with proj_script_nm.

Blowout minutes effect: quantified on 2012-2016 only (pre-dates every training window used for the props
seasons), by role tier: ratio of actual minutes/points to the player's own rolling average in games that
ended |margin| >= 15 vs closer games.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from backtest.props_nba import DISC, HOLD, FEATS, OUT, box_features
from .script import game_frame, script_predictions, team_script
from .tendencies import POS_GROUP, load_box, tendencies

BLOW_FIT_SEASONS = (2012, 2016)
SCRIPT_FEATS = ["team_pts_pred", "opp_pts_pred", "total_pred", "margin_pred", "pace_pred", "blowout_p",
                "posdef_mult", "share_r10", "share_r25", "proj_mech", "team_pts_r10", "starter_share_ewm"]


def _tier(d):
    starter = d.starter_r5.fillna(0) >= 0.5
    mins = pd.cut(d.minutes_r10.fillna(0), [-1, 18, 28, 34, 60], labels=["<18", "18-28", "28-34", "34+"])
    return np.where(starter, "S", "B") + mins.astype(str)


def blowout_effects(F: pd.DataFrame) -> pd.DataFrame:
    """Minutes and points in blowouts vs close games, relative to the player's own rolling form."""
    lo, hi = BLOW_FIT_SEASONS
    d = F[F.season.between(lo, hi) & (F.minutes_r10 >= 5) & (F.pts_r10 >= 1)].copy()
    d["blow"] = (d.team_score - d.opp_score).abs() >= 15
    d["min_ratio"], d["pts_ratio"] = d.minutes / d.minutes_r10, d.pts / d.pts_r10
    E = d.groupby(["tier", "blow"])[["min_ratio", "pts_ratio"]].mean().unstack("blow")
    E.columns = [f"{a}_{'blow' if b else 'close'}" for a, b in E.columns]
    E["n"] = d.groupby("tier").size()
    E["blow_rate"] = d.groupby("tier").blow.mean()
    return E


def player_frame(cache: str | None = None) -> pd.DataFrame:
    t0 = time.time()
    F = box_features()
    F = F[F.pts.notna()].copy()
    print(f"  box_features {len(F)} rows ({time.time() - t0:.0f}s)")
    # position group + share of team points (shifted)
    pos = load_box(["date", "game_id", "player_id", "position", "season_type"]).drop_duplicates(["game_id", "player_id"])
    F = F.merge(pos[["game_id", "player_id", "position"]], on=["game_id", "player_id"], how="left")
    F["pg"] = F.position.map(POS_GROUP).fillna("F")
    F = F.sort_values(["player_id", "date"])
    F["share"] = F.pts / F.team_score.replace(0, np.nan)
    gp = F.groupby("player_id", sort=False)
    for n in (10, 25):
        F[f"share_r{n}"] = gp["share"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    F["team_pts_r10"] = gp["team_score"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    # team tendencies + script
    T = tendencies()
    g = game_frame(T)
    P = script_predictions(g)
    P.to_csv(OUT / "nba_script_games.csv.gz", index=False)
    for v in ("pre", "none"):
        L = team_script(P, v)
        if v == "none":
            L = L[["game_id", "team", "total_pred", "margin_pred", "team_pts_pred", "opp_pts_pred", "pace_pred",
                   "blowout_p"]].add_suffix("_nm").rename(columns={"game_id_nm": "game_id", "team_nm": "team"})
        F = F.merge(L, on=["game_id", "team"], how="left")
    od = T[["game_id", "team", "posdef_G", "posdef_F", "posdef_C"]].rename(columns={"team": "opp"})
    F = F.merge(od, on=["game_id", "opp"], how="left")
    F["posdef_mult"] = np.select([F.pg == "G", F.pg == "C"], [F.posdef_G, F.posdef_C], F.posdef_F)
    own = T[["game_id", "team", "starter_share_ewm"]]
    F = F.merge(own, on=["game_id", "team"], how="left")
    F["tier"] = _tier(F)
    print(f"  player frame ready ({time.time() - t0:.0f}s)")
    if cache:
        F.to_parquet(cache)
    return F


def add_mech(F: pd.DataFrame, E: pd.DataFrame, suffix: str = "") -> tuple[pd.DataFrame, float]:
    """Mechanistic script projection (before recalibration)."""
    bp = F[f"blowout_p{suffix}"]
    e = E.reindex(F.tier)
    base = e.blow_rate.values
    r_b, r_c = e.pts_ratio_blow.values, e.pts_ratio_close.values
    blow_adj = (bp * r_b + (1 - bp) * r_c) / (base * r_b + (1 - base) * r_c)
    share = 0.6 * F.share_r10 + 0.4 * F.share_r25
    raw = share * F[f"team_pts_pred{suffix}"] * blow_adj
    # opponent positional defence exponent k, fit on the blowout-fit seasons only
    lo, hi = BLOW_FIT_SEASONS
    m = F.season.between(lo, hi) & raw.gt(2) & F.posdef_mult.gt(0) & F.pts_r10.notna()
    k = LinearRegression(fit_intercept=True).fit(np.log(F.loc[m, ["posdef_mult"]]),
                                                 np.log((F.pts[m] + 1) / (raw[m] + 1))).coef_[0]
    k = float(np.clip(k, 0, 1.5))
    F = F.copy()
    F["blow_adj"] = blow_adj
    F["proj_mech_raw"] = raw * F.posdef_mult.fillna(1.0).clip(0.7, 1.4) ** k
    return F, k


def _sd_fit(last, pred_col):
    last = last.copy()
    last["bin"] = pd.qcut(last[pred_col], 20, duplicates="drop")
    sd = last.groupby("bin", observed=True).apply(
        lambda x: pd.Series({"mu": x[pred_col].mean(), "sd": (x.pts - x[pred_col]).std()}))
    a, c = np.polyfit(np.sqrt(sd.mu.clip(lower=0.5)), sd.sd, 1)
    return a, c


def _hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)


def project(F: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    F, k = add_mech(F, E)
    print(f"  positional-defence exponent k={k:.2f}")
    out = []
    for s in (DISC, HOLD):
        tr = F[(F.season < s) & (F.season >= s - 8) & F.pts_r10.notna()].copy()
        te = F[F.season == s].copy()
        # mechanistic recalibration on the training window
        mm = tr.proj_mech_raw.notna()
        lr = LinearRegression().fit(tr.loc[mm, ["proj_mech_raw"]], tr.pts[mm])
        for d in (tr, te):
            d["proj_mech"] = np.where(d.proj_mech_raw.notna(), lr.predict(d[["proj_mech_raw"]].fillna(0)), np.nan)
        res = {}
        for name, X in (("script", FEATS + SCRIPT_FEATS),
                        ("script_nm", FEATS + [c + "_nm" if c in ("team_pts_pred", "opp_pts_pred", "total_pred",
                                                                  "margin_pred", "pace_pred", "blowout_p") else c
                                               for c in SCRIPT_FEATS if c != "proj_mech"]),
                        ("base", FEATS)):
            m = _hgb().fit(tr[X], tr.pts)
            te[f"proj_{name}"] = m.predict(te[X])
            last = tr[tr.season == s - 1].copy()
            last[f"proj_{name}"] = m.predict(last[X])
            a, c = _sd_fit(last, f"proj_{name}")
            te[f"proj_{name}_sd"] = np.maximum(a * np.sqrt(te[f"proj_{name}"].clip(lower=0.5)) + c, 2.0)
        last = tr[(tr.season == s - 1) & tr.proj_mech.notna()]
        a, c = _sd_fit(last, "proj_mech")
        te["proj_mech_sd"] = np.maximum(a * np.sqrt(te.proj_mech.clip(lower=0.5)) + c, 2.0)
        for c_ in ("proj_base", "proj_script", "proj_script_nm", "proj_mech"):
            x = te[te[c_].notna()]
            res[c_] = f"{(x.pts - x[c_]).abs().mean():.3f}"
        print(f"  season {s}: train n={len(tr)}  MAE {res}")
        out.append(te)
    return pd.concat(out, ignore_index=True)


KEEP = ["date", "season", "game_id", "player_id", "player", "team", "opp", "pg", "tier", "starter_r5", "minutes_r10",
        "pts", "minutes", "team_score", "opp_score", "proj_base", "proj_base_sd", "proj_script", "proj_script_sd",
        "proj_script_nm", "proj_script_nm_sd", "proj_mech", "proj_mech_sd", "blowout_p", "blowout_p_nm", "pace_pred",
        "team_pts_pred", "opp_pts_pred", "total_pred", "margin_pred", "spread_team", "pre_total", "pre_src",
        "lg_pace", "posdef_mult", "blow_adj", "share_r10"]


def run(cache: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import os
    F = pd.read_parquet(cache) if cache and os.path.exists(cache) else player_frame(cache)
    E = blowout_effects(F)
    print(E.round(3).to_string())
    Pj = project(F, E)
    return Pj[KEEP], E


if __name__ == "__main__":
    import sys
    Pj, E = run(sys.argv[1] if len(sys.argv) > 1 else None)
    print(Pj.describe().T.round(2).to_string())
