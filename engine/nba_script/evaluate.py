"""
Evaluate script-aware NBA points projections against real sportsbook prop lines.

Joins player_script projections to results/nba_props_frame.csv.gz on (date, pid) and compares
  p_mkt     leave-one-book-out market consensus (from props_nba)
  p_mod     props_nba HGB model
  p_script  script-aware HGB (same model + game-script features; 'pre' lines = open, else close)
  p_script_nm  script-aware HGB with the tendency-only script (no betting lines at all)
  p_mech    mechanistic share x team points x blowout x positional defence
and blends.  Blend weights and filter thresholds are chosen ONLY on season 2025 (discovery);
2026 (holdout) is reported as-is.

Usage: python -m engine.nba_script.evaluate [projection_cache.parquet]
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest.core import perf
from backtest.props_nba import DISC, HOLD, OUT, pick_bets
from .script import validate

THRS = (0.02, 0.04, 0.06, 0.08)


def load(cache: str | None) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        Pj = pd.read_parquet(cache)
    else:
        from .player_script import run
        Pj, _ = run(os.environ.get("NBA_SCRIPT_FRAME_CACHE"))   # optional player-frame cache
        if cache:
            Pj.to_parquet(cache)
    p = pd.read_csv(OUT / "nba_props_frame.csv.gz", low_memory=False)
    p["date"] = pd.to_datetime(p["date"])
    Pj = Pj.rename(columns={"player_id": "pid"}).drop(columns=["season", "game_id", "player", "team", "opp", "pts", "minutes", "minutes_r10"])
    Pj["pid"] = Pj.pid.astype("int64")
    p["pid"] = p.pid.astype("int64")
    p = p.drop(columns=["starter_r5"]).merge(Pj, on=["date", "pid"], how="left")   # same value, recomputed
    for name, mu, sd in (("script", "proj_script", "proj_script_sd"), ("script_nm", "proj_script_nm", "proj_script_nm_sd"),
                         ("mech", "proj_mech", "proj_mech_sd"), ("base", "proj_base", "proj_base_sd")):
        p[f"p_{name}"] = 1 - norm.cdf((p.line - p[mu]) / p[sd])
        p[f"p_{name}"] = p[f"p_{name}"].fillna(p.p_mod)          # (few) rows w/o script -> baseline model
    p["p_blend"] = 0.75 * p.p_mkt + 0.25 * p.p_mod
    return p


def scores(p, cols):
    rows = []
    for (s, main), x in p.groupby([p.season, p.alt_line == 0]):
        for c in cols:
            q = x[c].clip(1e-4, 1 - 1e-4)
            rows.append(dict(season=s, main_only=main, prob=c, n=len(x), brier=np.mean((q - x.y) ** 2),
                             logloss=-np.mean(x.y * np.log(q) + (1 - x.y) * np.log(1 - q))))
    return pd.DataFrame(rows)


def roi_table(p, cols, thrs=THRS, main_only=False):
    q = p[p.alt_line == 0] if main_only else p
    rows = []
    for c, t in itertools.product(cols, thrs):
        B = pick_bets(q, c, t)
        for s in (DISC, HOLD):
            x = B[B.season == s]
            r = perf(x.pnl)
            rows.append(dict(prob=c, thr=t, season=s, n=r.get("n", 0), roi=r.get("roi"), z=r.get("z")))
    return pd.DataFrame(rows)


def choose_blend(p):
    """Grid over weights (mkt, mod, script) on 2025 only: by main-line log-loss and by ROI@EV>=0.04."""
    d = p[p.season == DISC]
    dm = d[d.alt_line == 0]
    res = []
    for a in np.arange(0, 0.65, 0.05):
        for b in np.arange(0, 0.65 - a, 0.05):
            pb = (1 - a - b) * d.p_mkt + a * d.p_mod + b * d.p_script
            q = pb[dm.index].clip(1e-4, 1 - 1e-4)
            ll = -np.mean(dm.y * np.log(q) + (1 - dm.y) * np.log(1 - q))
            B = pick_bets(d.assign(pb=pb), "pb", 0.04)
            res.append(dict(w_mod=round(a, 2), w_script=round(b, 2), logloss=ll, roi=B.pnl.mean(), n=len(B)))
    R = pd.DataFrame(res)
    return R, R.sort_values("logloss").iloc[0], R[R.n >= 500].sort_values("roi", ascending=False).iloc[0]


def side_split(p, col, thr=0.04):
    B = pick_bets(p, col, thr)
    return B.groupby(["season", "side"]).pnl.agg(n="size", roi="mean").reset_index().assign(prob=col)


def script_fit(p, col="p_blend", thr=0.04):
    """Does agreeing with the game script improve picked bets?  Thresholds from 2025 quotes."""
    d25 = p[p.season == DISC]
    bp_lo, bp_hi = d25.blowout_p.quantile([1 / 3, 2 / 3])
    p = p.assign(pace_rel=p.pace_pred - p.lg_pace)
    pr_hi = p.loc[p.season == DISC, "pace_rel"].quantile(2 / 3)
    B = pick_bets(p, col, thr)
    starter = B.starter_r5.fillna(0) >= 0.5
    over = B.side == "over"
    rule = np.where(over, (B.blowout_p <= bp_lo) | (B.pace_rel >= pr_hi), (B.blowout_p >= bp_hi) & starter)
    adj = np.where(over, B.proj_script > B.proj, B.proj_script < B.proj)
    rows = []
    for name, f in (("rule_fit", rule), ("script_agrees", adj)):
        for s in (DISC, HOLD):
            for val in (True, False):
                x = B[(B.season == s) & (f == val)]
                rows.append(dict(base=col, filter=name, season=s, fit=val, n=len(x), roi=x.pnl.mean() if len(x) else np.nan))
    return pd.DataFrame(rows), dict(bp_lo=bp_lo, bp_hi=bp_hi, pace_rel_hi=pr_hi)


def main(cache=None):
    pd.set_option("display.width", 220)
    p = load(cache)
    print(f"joined {len(p)} quotes; script coverage {p.proj_script.notna().mean():.3f}")
    probs = ["p_mkt", "p_mod", "p_base", "p_script", "p_script_nm", "p_mech", "p_blend"]
    S = scores(p, probs)
    print(S.pivot_table(index="prob", columns=["season", "main_only"], values="logloss").round(5))
    R, best_ll, best_roi = choose_blend(p)
    print("best by 2025 log-loss:", best_ll.to_dict(), "\nbest by 2025 ROI@0.04:", best_roi.to_dict())
    p["p_blend_script"] = 0.75 * p.p_mkt + 0.25 * p.p_script
    p["p_blend_script_nm"] = 0.75 * p.p_mkt + 0.25 * p.p_script_nm
    p["p_sel_ll"] = (1 - best_ll.w_mod - best_ll.w_script) * p.p_mkt + best_ll.w_mod * p.p_mod + best_ll.w_script * p.p_script
    p["p_sel_roi"] = (1 - best_roi.w_mod - best_roi.w_script) * p.p_mkt + best_roi.w_mod * p.p_mod + best_roi.w_script * p.p_script
    S2 = scores(p, ["p_blend", "p_blend_script", "p_blend_script_nm", "p_sel_ll", "p_sel_roi"])
    S = pd.concat([S, S2])
    roi_cols = ["p_mkt", "p_mod", "p_script", "p_script_nm", "p_blend", "p_blend_script", "p_blend_script_nm",
                "p_sel_ll", "p_sel_roi"]
    RA = roi_table(p, roi_cols).assign(lines="all")
    RM = roi_table(p, roi_cols, main_only=True).assign(lines="main")
    RT = pd.concat([RA, RM])
    print(RT.pivot_table(index=["lines", "prob", "thr"], columns="season", values=["n", "roi"]).round(3).to_string())
    SS = pd.concat([side_split(p, c) for c in ("p_blend", "p_blend_script", "p_sel_roi")])
    print(SS.round(3).to_string())
    FF, fq = pd.concat([script_fit(p, c)[0] for c in ("p_blend", "p_blend_script")]), script_fit(p)[1]
    print(fq); print(FF.round(3).to_string())
    games = pd.read_csv(OUT / "nba_script_games.csv.gz", low_memory=False)
    V = validate(games)
    # outputs
    fr = p[["date", "pid", "book", "line", "alt_line", "p_script", "proj_script", "proj_script_sd", "blowout_p",
            "pace_pred", "team_pts_pred", "p_script_nm", "proj_mech"]].copy()
    fr["date"] = fr.date.dt.date
    fr.round(4).to_csv(OUT / "nba_script_frame.csv.gz", index=False)
    with open(OUT.parent / "logs" / "nba_script_eval.txt", "w") as f:
        for name, df in (("scores", S), ("blend_grid", R), ("roi", RT), ("sides", SS), ("fit", FF),
                         ("val_by_season", V["by_season"]), ("totals_grid", V["totals_grid"]), ("totals_chosen", V["chosen"])):
            f.write(f"\n## {name}\n{df.round(4).to_string()}\n")
        f.write(f"\nblend best_ll={best_ll.to_dict()} best_roi={best_roi.to_dict()} fit_thresholds={fq}\n")
    return p, S, RT, SS, FF, V, (best_ll, best_roi)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
