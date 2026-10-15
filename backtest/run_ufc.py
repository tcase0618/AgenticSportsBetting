"""
UFC strategy search. Same discovery/holdout protocol as team sports, but the
unit is a fighter-vs-fighter moneyline. Each fight is viewed from both corners;
a strategy bets on (or against) fighter F when a pre-fight condition on F holds
and does NOT also hold for the opponent.

usage: python -m backtest.run_ufc
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import settle, no_vig_probs
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

NUM = ["age", "height_cm", "reach_cm", "wins", "losses", "win_streak", "lose_streak",
       "sig_str_landed_pm", "sig_str_absorbed_pm", "td_avg", "td_def", "sub_avg",
       "fights_ufc", "days_since_last_fight", "sig_str_acc", "sig_str_def", "td_acc", "ko_wins", "sub_wins"]


def fighter_elo(f: pd.DataFrame, k=32.0) -> pd.DataFrame:
    """Pre-fight Elo from results only (chronological)."""
    r = {}
    ea, eb = np.zeros(len(f)), np.zeros(len(f))
    for i, row in enumerate(f.itertuples(index=False)):
        ra, rb = r.get(row.fighter_a, 1500.0), r.get(row.fighter_b, 1500.0)
        ea[i], eb[i] = ra, rb
        exp = 1 / (1 + 10 ** (-(ra - rb) / 400))
        res = {"A": 1.0, "B": 0.0}.get(row.winner)
        if res is None:
            continue
        r[row.fighter_a], r[row.fighter_b] = ra + k * (res - exp), rb - k * (res - exp)
    f = f.copy()
    f["elo_a"], f["elo_b"] = ea, eb
    f["p_a_elo"] = 1 / (1 + 10 ** (-(f.elo_a - f.elo_b) / 400))
    return f


def prep(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date").reset_index(drop=True)
    f["season"] = f["date"].dt.year
    for side in "ab":
        for c in NUM:
            col = f"{side}_{c}"
            f[col] = pd.to_numeric(f[col], errors="coerce") if col in f else np.nan
    f["a_ml"], f["b_ml"] = pd.to_numeric(f["a_ml"], errors="coerce"), pd.to_numeric(f["b_ml"], errors="coerce")
    for c in ["a_ml", "b_ml"]:
        f.loc[f[c].abs() < 100, c] = np.nan
    f = fighter_elo(f)
    pa, pb, vig = no_vig_probs(f["a_ml"], f["b_ml"])
    f["p_a"], f["p_b"], f["overround"] = pa, pb, vig
    f["title_bout"] = pd.to_numeric(f.get("title_bout", 0), errors="coerce").fillna(0)
    f["scheduled_rounds"] = pd.to_numeric(f.get("scheduled_rounds", 3), errors="coerce")
    return f


def corner_view(f: pd.DataFrame, side: str) -> pd.DataFrame:
    me, op = ("a_", "b_") if side == "a" else ("b_", "a_")
    v = pd.DataFrame(index=f.index)
    for c in NUM:
        v[c], v["o_" + c] = f[me + c], f[op + c]
    for c in ["stance"]:
        v[c] = f.get(me + c)
        v["o_" + c] = f.get(op + c)
    v["ml"] = f[me + "ml"]
    v["p_mkt"] = f["p_" + side]
    v["p_elo"] = f["p_a_elo"] if side == "a" else 1 - f["p_a_elo"]
    v["fav"] = v["p_mkt"] > 0.5
    return v


def conditions() -> dict:
    return {
        "any": lambda v: pd.Series(True, v.index),
        "younger5": lambda v: (v.o_age - v.age) >= 5,
        "younger8": lambda v: (v.o_age - v.age) >= 8,
        "age35plus": lambda v: v.age >= 35,
        "opp_age35plus": lambda v: (v.o_age >= 35) & (v.age < 33),
        "reach_adv8cm": lambda v: (v.reach_cm - v.o_reach_cm) >= 8,
        "reach_adv13cm": lambda v: (v.reach_cm - v.o_reach_cm) >= 13,
        "height_adv8cm": lambda v: (v.height_cm - v.o_height_cm) >= 8,
        "layoff_1yr": lambda v: v.days_since_last_fight >= 365,
        "opp_layoff_1yr": lambda v: (v.o_days_since_last_fight >= 365) & (v.days_since_last_fight < 250),
        "quick_turnaround": lambda v: v.days_since_last_fight <= 60,
        "ufc_debut": lambda v: v.fights_ufc == 0,
        "vs_debutant": lambda v: (v.o_fights_ufc == 0) & (v.fights_ufc >= 3),
        "more_exp_10": lambda v: (v.fights_ufc - v.o_fights_ufc) >= 10,
        "win_streak3": lambda v: v.win_streak >= 3,
        "win_streak5": lambda v: v.win_streak >= 5,
        "lose_streak2": lambda v: v.lose_streak >= 2,
        "opp_lose_streak2": lambda v: (v.o_lose_streak >= 2) & (v.lose_streak == 0),
        "striking_diff_edge": lambda v: ((v.sig_str_landed_pm - v.sig_str_absorbed_pm) -
                                         (v.o_sig_str_landed_pm - v.o_sig_str_absorbed_pm)) >= 2,
        "wrestler_vs_poor_tdd": lambda v: (v.td_avg >= 2.5) & (v.o_td_def <= 0.55),
        "td_edge": lambda v: (v.td_avg - v.o_td_avg) >= 2,
        "sub_threat": lambda v: (v.sub_avg >= 1.0) & (v.o_sub_avg < 0.3),
        "southpaw_vs_orth": lambda v: (v.stance == "Southpaw") & (v.o_stance == "Orthodox"),
        "more_pro_wins10": lambda v: (v.wins - v.o_wins) >= 10,
        "elo_edge_5": lambda v: (v.p_elo - v.p_mkt) >= 0.05,
        "elo_edge_10": lambda v: (v.p_elo - v.p_mkt) >= 0.10,
        "elo_edge_15": lambda v: (v.p_elo - v.p_mkt) >= 0.15,
    }


def roles() -> dict:
    return {
        "any": lambda v: pd.Series(True, v.index),
        "fav": lambda v: v.fav,
        "dog": lambda v: ~v.fav,
        "heavy_fav": lambda v: v.p_mkt >= 0.75,
        "mid_fav": lambda v: (v.p_mkt >= 0.6) & (v.p_mkt < 0.75),
        "small_fav": lambda v: (v.p_mkt > 0.5) & (v.p_mkt < 0.6),
        "small_dog": lambda v: (v.p_mkt >= 0.4) & (v.p_mkt <= 0.5),
        "mid_dog": lambda v: (v.p_mkt >= 0.25) & (v.p_mkt < 0.4),
        "big_dog": lambda v: v.p_mkt < 0.25,
    }


def phases(f: pd.DataFrame) -> dict:
    wc = f.get("weight_class", pd.Series("", f.index)).astype(str).str.lower()
    g = f.get("gender", pd.Series("", f.index)).astype(str).str.lower()
    return {
        "all": pd.Series(True, f.index),
        "heavyweight": wc.str.contains("heavy") & ~wc.str.contains("light"),
        "lightweight_down": wc.str.contains("light|feather|bantam|fly|straw") & ~wc.str.contains("heavy"),
        "women": g.str.startswith("f") | wc.str.contains("women"),
        "five_rounds": f["scheduled_rounds"] == 5,
        "three_rounds": f["scheduled_rounds"] == 3,
    }


def strategies(f: pd.DataFrame) -> dict:
    out_a = pd.Series(np.select([f.winner == "A", f.winner == "B"], [1, -1], 0), f.index)
    pnl_a = pd.Series(np.where(f.a_ml.notna() & f.b_ml.notna(), settle(out_a, f.a_ml.fillna(-110)), np.nan), f.index)
    pnl_b = pd.Series(np.where(f.a_ml.notna() & f.b_ml.notna(), settle(-out_a, f.b_ml.fillna(-110)), np.nan), f.index)
    va, vb = corner_view(f, "a"), corner_view(f, "b")
    C, Rl, P = conditions(), roles(), phases(f)
    res = {}
    for cn, cf in C.items():
        ca, cb = cf(va).fillna(False).astype(bool), cf(vb).fillna(False).astype(bool)
        if cn != "any":
            ca, cb = ca & ~cb, cb & ~ca
        for rn, rf in Rl.items():
            if cn == "any" and rn == "any":
                continue
            ma = ca & rf(va).fillna(False).astype(bool)
            mb = cb & rf(vb).fillna(False).astype(bool)
            if cn == "any":
                mb &= ~ma
            for d in ("on", "vs"):
                s = pd.concat([pnl_a[ma], pnl_b[mb]]) if d == "on" else pd.concat([pnl_b[ma], pnl_a[mb]])
                s = s.dropna().sort_index()
                for pn, pm in P.items():
                    ss = s[pm.loc[s.index].values]
                    if len(ss):
                        res[("ml", cn, rn, d, pn)] = ss
    return res, pnl_a, pnl_b


def run(split: int | None = None):
    f = prep(pd.read_csv(ROOT / "data" / "processed" / "ufc.csv"))
    f = f[f.a_ml.notna() & f.b_ml.notna()].copy()
    yrs = np.sort(f.season.unique())
    split = split or int(yrs[int(len(yrs) * 0.62)])
    strats, pa, pb = strategies(f)
    R = evaluate(strats, f, split, min_train=150, min_hold=80)
    OUT.mkdir(exist_ok=True)
    R.to_csv(OUT / "ufc_strategies.csv", index=False)
    fav_win = np.where(f.p_a > 0.5, f.winner == "A", f.winner == "B").mean()
    meta = dict(sport="ufc", seasons=[int(yrs.min()), int(yrs.max())], split=split, fights=len(f),
                n_strategies=len(strats), n_evaluated=len(R), n_discovered=int(R.discovered.sum()),
                n_validated=int(R.validated.sum()), fav_win_rate=float(fav_win),
                avg_overround=float(f.overround.mean()),
                baseline={"bet_A": float(pa.mean()), "bet_B": float(pb.mean())})
    (OUT / "ufc_meta.json").write_text(json.dumps(meta, indent=2))
    return f, strats, R, meta


if __name__ == "__main__":
    f, strats, R, meta = run()
    print(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250)
    cols = ["market", "condition", "role", "direction", "phase", "n_train", "roi_train", "p_train",
            "n_hold", "roi_hold", "z_hold", "q_hold", "roi_all", "pct_pos_seasons"]
    print(R[R.validated].head(30)[cols].to_string())
