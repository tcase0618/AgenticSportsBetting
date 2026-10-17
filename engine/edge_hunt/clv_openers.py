"""
Edge hunter "clv_openers" — can we predict the open->close line move and bet the OPENER?

Hypothesis: a model that knows where the closing line will go (script projection vs opener) lets us bet the
opening number and capture closing-line value (CLV). We test, per sport x market (spread / total / moneyline):

  A. "model" family: walk-forward OLS predicting (close - open) from pre-opener info only
     (script/core projection minus opener, opener level). Bet the opener when |predicted move| >= T.
     Settled at the OPENING price against the actual result.
  B. "raw" family: same but the signal is the raw script-minus-opener disagreement (no regression).
  C. "move" family (public, live-able rule): line moved >= k from the opener to the (near-)close;
     FOLLOW or FADE that move at the closing price (live: compare current FanDuel to the ESPN opener).

Protocol (engine/edge_hunt/PROTOCOL.md): every knob is chosen on discovery (season <= 2022, or first 60% of
dates when opener data begin later / have no 2023+ rows), the single chosen rule per (sport, market, family)
is evaluated ONCE on the lockbox. Every test is logged to results/edge_hunt/clv_openers_candidates.csv.

Prices:
  * NBA 2025-26 and NCAAM 2025-26: real DraftKings opening AND closing juice/ML from ESPN PickCenter JSON.
  * NCAAF 2016-25 and NBA 2016-22 spreads/totals: NO opening juice in the data -> assumed -110 (flagged).
    (Closing-price "move" rules on those rows also assume -110.)
  * MLB 2022-25 and NHL 2016-22 moneylines: real SBR/SBRO opening and closing American prices.
  * NCAAF moneyline: openers exist only 2016-19 (CFBD), none in the lockbox -> not testable.
  * NFL: no openers anywhere in data -> not testable.

usage:  python -m engine.edge_hunt.clv_openers            # full backtest -> results/edge_hunt/clv_openers.md
        python -c "from engine.edge_hunt.clv_openers import scan; print(scan('ncaaf'))"
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest.core import american_to_decimal, no_vig_probs, perf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "edge_hunt"
SCR = ROOT / "results" / "scripts"
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

MAX_MOVE = {"spread": 10.0, "total": 15.0}   # |close-open| beyond this = bad record (CFBD has 299-pt typos)


# ======================================================================== loading
def _am(s):
    if s is None:
        return np.nan
    s = str(s).strip().lower().replace("o", "").replace("u", "")
    if s in ("even", "ev", "pk", "pick"):
        return 100.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pickcenter(league: str, season: int) -> pd.DataFrame:
    """Real DraftKings open/close lines and juice from ESPN PickCenter (2025-26 season files)."""
    f = RAW / league / "espn_pickcenter" / f"{'mbb' if league == 'ncaam' else 'nba'}_{season}.jsonl"
    rows = []
    for line in open(f):
        r = json.loads(line)
        for p in (r.get("pickcenter") or [])[:1]:
            ps, tt, ml = p.get("pointSpread", {}), p.get("total", {}), p.get("moneyline", {})
            rows.append(dict(
                game_id=int(r["game_id"]),
                pc_sp_open=_am(_dig(ps, "home", "open", "line")), pc_sp_close=_am(_dig(ps, "home", "close", "line")),
                sp_open_h_odds=_am(_dig(ps, "home", "open", "odds")), sp_open_a_odds=_am(_dig(ps, "away", "open", "odds")),
                sp_close_h_odds=_am(_dig(ps, "home", "close", "odds")), sp_close_a_odds=_am(_dig(ps, "away", "close", "odds")),
                pc_tot_open=_am(_dig(tt, "over", "open", "line")), pc_tot_close=_am(_dig(tt, "over", "close", "line")),
                o_open_odds=_am(_dig(tt, "over", "open", "odds")), u_open_odds=_am(_dig(tt, "under", "open", "odds")),
                o_close_odds=_am(_dig(tt, "over", "close", "odds")), u_close_odds=_am(_dig(tt, "under", "close", "odds")),
                ml_h_open=_am(_dig(ml, "home", "open", "odds")), ml_a_open=_am(_dig(ml, "away", "open", "odds")),
                ml_h_close=_am(_dig(ml, "home", "close", "odds")), ml_a_close=_am(_dig(ml, "away", "close", "odds")),
            ))
    return pd.DataFrame(rows).drop_duplicates("game_id")


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def _poisson_home(mh, ma, kmax=25):
    """P(home wins) from independent Poisson scores, ties split 50/50 (OT/SO, extra innings)."""
    from scipy.stats import poisson
    k = np.arange(kmax)[:, None]
    ph, pa = poisson.pmf(k, np.asarray(mh, float)[None, :]), poisson.pmf(k, np.asarray(ma, float)[None, :])
    cdf_a = np.cumsum(pa, axis=0)
    p_win = (ph[1:] * cdf_a[:-1]).sum(axis=0)
    p_tie = (ph * pa).sum(axis=0)
    return p_win + 0.5 * p_tie


def load(sport: str) -> pd.DataFrame:
    """One row per game with: scores, sp_open/sp_close (home), tot_open/tot_close, ML open/close, juice
    (NaN = unknown), script features (pre-game), and split labels per market."""
    if sport in ("ncaaf", "nba", "ncaam"):
        d = pd.read_parquet(SCR / f"{sport}_games.parquet")
        d = d.rename(columns={"spread_open_home": "sp_open", "spread_home": "sp_close", "total_open": "tot_open",
                              "total_line": "tot_close", "home_ml": "ml_h_close", "away_ml": "ml_a_close"})
        d["s_margin"], d["c_margin"] = d["margin"], d["margin_core"]
        d["s_total"], d["c_total"] = d["total"], d["total_core"]
        sdm = d["sd_margin"] if "sd_margin" in d else np.hypot(d["sd_home"], d["sd_away"])
        d["p_script"] = _phi(d["s_margin"] / sdm)
        d["p_core"] = _phi(d["c_margin"] / sdm)
        for c in ["sp_open_h_odds", "sp_open_a_odds", "sp_close_h_odds", "sp_close_a_odds", "o_open_odds",
                  "u_open_odds", "o_close_odds", "u_close_odds", "ml_h_open", "ml_a_open"]:
            d[c] = np.nan
        if sport in ("nba", "ncaam"):
            pc = _pickcenter(sport, 2026)
            d = d.merge(pc, on="game_id", how="left", suffixes=("", "_pc"))
            m = d["season"] == 2026
            for c in ["sp_open_h_odds", "sp_open_a_odds", "sp_close_h_odds", "sp_close_a_odds", "o_open_odds",
                      "u_open_odds", "o_close_odds", "u_close_odds", "ml_h_open", "ml_a_open"]:
                d.loc[m, c] = d.loc[m, c + "_pc"]
            # use PickCenter's own open/close numbers (same DK feed as the juice) where present
            for a, b in [("sp_open", "pc_sp_open"), ("sp_close", "pc_sp_close"), ("tot_open", "pc_tot_open"),
                         ("tot_close", "pc_tot_close")]:
                d.loc[m & d[b].notna(), a] = d.loc[m & d[b].notna(), b]
            mm = m & d["ml_h_close_pc"].notna()
            d.loc[mm, "ml_h_close"], d.loc[mm, "ml_a_close"] = d.loc[mm, "ml_h_close_pc"], d.loc[mm, "ml_a_close_pc"]
            d = d.drop(columns=[c for c in d.columns if c.endswith("_pc") or c.startswith("pc_")])
    else:  # mlb, nhl: moneyline openers from processed CSVs
        d = pd.read_parquet(SCR / f"{sport}_games.parquet")
        c = pd.read_csv(PROC / f"{sport}.csv", low_memory=False).dropna(subset=["home_ml_open", "away_ml_open"])
        c = c[["date", "home", "away", "home_ml_open", "away_ml_open", "home_ml", "away_ml"]]
        c = c.drop_duplicates(["date", "home", "away"], keep=False)
        d["date"] = d["date"].astype(str).str[:10]
        c["date"] = c["date"].astype(str).str[:10]
        d = d.drop(columns=["home_ml", "away_ml"]).merge(c, on=["date", "home", "away"], how="inner")
        d = d.rename(columns={"home_ml_open": "ml_h_open", "away_ml_open": "ml_a_open", "home_ml": "ml_h_close",
                              "away_ml": "ml_a_close"})
        d["s_margin"] = d["margin"]
        d["c_margin"] = d["mu_home_core"] - d["mu_away_core"]
        d["p_core"] = _poisson_home(d["mu_home_core"].values, d["mu_away_core"].values)
        d["p_script"] = d["p_home_script"]
        for k in ["sp_open", "sp_close", "tot_open", "tot_close"]:
            d[k] = np.nan
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["hs"], d["as_"] = d["home_score"].astype(float), d["away_score"].astype(float)
    d = d.dropna(subset=["hs", "as_"]).sort_values("date").reset_index(drop=True)
    # de-vigged home probability at open and close
    for t in ("open", "close"):
        ok = d[f"ml_h_{t}"].notna() & d[f"ml_a_{t}"].notna()
        d[f"pdv_{t}"] = np.nan
        d.loc[ok, f"pdv_{t}"] = no_vig_probs(d.loc[ok, f"ml_h_{t}"], d.loc[ok, f"ml_a_{t}"])[0]
    d["sport"] = sport
    return d


# ======================================================================== market frames
def market_frame(d: pd.DataFrame, market: str) -> pd.DataFrame:
    """Rows with both opener and closer for this market + target move + candidate features."""
    sport = d["sport"].iat[0]
    if market == "spread":
        x = d.dropna(subset=["sp_open", "sp_close"]).copy()
        x["move"] = x["sp_close"] - x["sp_open"]                       # <0 = moved toward HOME
        x = x[x["move"].abs() <= MAX_MOVE["spread"]]
        # script-implied home spread minus opener (<0 = script likes home more than the opener)
        x["f_script"] = -x["s_margin"] - x["sp_open"]
        x["f_core"] = -x["c_margin"] - x["sp_open"]
        x["f_level"] = x["sp_open"]
    elif market == "total":
        x = d.dropna(subset=["tot_open", "tot_close"]).copy()
        x["move"] = x["tot_close"] - x["tot_open"]
        x = x[x["move"].abs() <= MAX_MOVE["total"]]
        x["f_script"] = x["s_total"] - x["tot_open"]
        x["f_core"] = x["c_total"] - x["tot_open"]
        x["f_level"] = x["tot_open"]
    else:  # ml, move in de-vigged home probability
        x = d.dropna(subset=["pdv_open", "pdv_close"]).copy()
        x["move"] = x["pdv_close"] - x["pdv_open"]
        x = x[x["move"].abs() <= 0.25]
        x["f_script"] = x["p_script"] - x["pdv_open"]
        x["f_core"] = x["p_core"] - x["pdv_open"]
        x["f_level"] = x["pdv_open"] - 0.5
    x = x.dropna(subset=["f_script"]).reset_index(drop=True)
    x["split"] = split_labels(x, sport, market)
    return x


def split_labels(x: pd.DataFrame, sport: str, market: str) -> pd.Series:
    """Protocol split: season <= 2022 discovery / 2023+ lockbox; if the opener data start after 2022 or have
    no 2023+ rows, discovery = first 60% of dates, lockbox = last 40%."""
    disc = x["season"] <= 2022
    if disc.all() or (~disc).all():
        dates = np.sort(x["date"].unique())
        cut = dates[int(len(dates) * 0.6)]
        return pd.Series(np.where(x["date"] < cut, "disc", "lock"), index=x.index)
    return pd.Series(np.where(disc, "disc", "lock"), index=x.index)


def split_rule(x: pd.DataFrame) -> str:
    if (x["season"] <= 2022).all() or (x["season"] > 2022).all():
        return "60/40 dates"
    return "<=2022 / 2023+"


# ======================================================================== walk-forward model
FEATSETS = {"script": ["f_script"], "core": ["f_core"], "script+core+level": ["f_script", "f_core", "f_level"]}
MIN_TRAIN = 300


def wf_predict(x: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Expanding-window OLS refit at the start of each calendar month; only earlier months train."""
    pred = pd.Series(np.nan, index=x.index)
    for mth in sorted(x["month"].unique()):
        tr = x[x["month"] < mth]
        if len(tr) < MIN_TRAIN:
            continue
        te = x["month"] == mth
        A = np.column_stack([np.ones(len(tr))] + [tr[f].values for f in feats])
        beta = np.linalg.lstsq(A, tr["move"].values, rcond=None)[0]
        B = np.column_stack([np.ones(te.sum())] + [x.loc[te, f].values for f in feats])
        pred[te] = B @ beta
    return pred


# ======================================================================== settlement
def _settle_side(x: pd.DataFrame, market: str, side: np.ndarray, at: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """side: +1 = home/over, -1 = away/under, 0 = no bet. at: 'open' or 'close' price/line.
    Returns pnl, odds, clv (pts or prob in bet's favour), juice_real flag."""
    side = np.asarray(side)
    margin = (x["hs"] - x["as_"]).values
    if market == "spread":
        line = x[f"sp_{at}"].values
        res = margin + line                                          # >0 home covers
        out = np.sign(res) * side
        oh, oa = x[f"sp_{at}_h_odds"].values, x[f"sp_{at}_a_odds"].values
        odds = np.where(side > 0, oh, oa)
        clv = side * (x["sp_open"].values - x["sp_close"].values)
    elif market == "total":
        line = x[f"tot_{at}"].values
        res = (x["hs"] + x["as_"]).values - line
        out = np.sign(res) * side
        odds = np.where(side > 0, x[f"o_{at}_odds"].values, x[f"u_{at}_odds"].values)
        clv = side * (x["tot_close"].values - x["tot_open"].values)
    else:
        out = np.where(margin > 0, 1, np.where(margin < 0, -1, 0)) * side
        odds = np.where(side > 0, x[f"ml_h_{at}"].values, x[f"ml_a_{at}"].values)
        clv = side * (x["pdv_close"].values - x["pdv_open"].values)
    real = ~np.isnan(odds.astype(float))
    if market == "ml":
        odds_f = odds.astype(float)
    else:
        odds_f = np.where(real, odds.astype(float), -110.0)
    dec = american_to_decimal(odds_f)
    pnl = np.where(out > 0, dec - 1, np.where(out < 0, -1.0, 0.0))
    m = side != 0
    if market == "ml":
        m = m & real
    idx = x.index[m]
    return (pd.Series(pnl[m], idx), pd.Series(odds_f[m], idx), pd.Series(clv[m], idx), pd.Series(real[m], idx))


def evaluate(x: pd.DataFrame, market: str, side: np.ndarray, at: str, part: str | None) -> dict:
    sel = np.ones(len(x), bool) if part is None else (x["split"] == part).values
    s = np.where(sel, side, 0)
    pnl, odds, clv, real = _settle_side(x, market, s, at)
    if len(pnl) == 0:
        return dict(n=0)
    blk = x.loc[pnl.index, "season"] if x["season"].nunique() > 2 else x.loc[pnl.index, "month"]
    r = perf(pnl, odds, blk)
    r["clv"] = float(clv.mean())
    r["clv_beat"] = float((clv > 0).mean())
    r["juice_real_frac"] = float(real.mean())
    return r


# ======================================================================== search
GRID = {
    "spread": {"model": [0.25, 0.5, 0.75, 1.0, 1.5], "raw": [2, 3, 4, 6, 8], "move": [1, 1.5, 2, 3]},
    "total": {"model": [0.5, 1.0, 1.5, 2.0, 3.0], "raw": [2, 4, 6, 8, 10], "move": [1, 2, 3, 4]},
    "ml": {"model": [0.005, 0.01, 0.015, 0.02, 0.03], "raw": [0.02, 0.04, 0.06, 0.08, 0.10], "move": [0.02, 0.03, 0.05, 0.08]},
}
MIN_DISC_N = 150
SPORT_MARKETS = {"ncaaf": ["spread", "total"], "nba": ["spread", "total", "ml"],
                 "ncaam": ["spread", "total", "ml"], "mlb": ["ml"], "nhl": ["ml"]}


def _sides(x, family, feat, T, direction):
    if family == "model":
        s = x[f"pred_{feat}"]
    elif family == "raw":
        s = x["f_core"] if feat == "core" else x["f_script"]
    else:  # move: observed open->close move (home perspective sign)
        s = x["move"]
    s = s.values.astype(float)
    if x.attrs["market"] == "spread" and family != "move":
        sig = -s                     # predicted/implied spread falling => home
    elif x.attrs["market"] == "spread":
        sig = -s                     # spread moved down => move was toward home
    else:
        sig = s
    side = np.where(np.isnan(sig), 0, np.where(sig >= T, 1, np.where(sig <= -T, -1, 0)))
    return side * direction


# Pre-registered feature sets (main pass) vs the CLEAN post-hoc pass. The main pass's "script" projections for
# NBA (actual missing minutes / who played) and NHL (confirmed starting goalie) use information that is NOT
# available when the opener is posted -> those rules are flagged LEAK and cannot be TRUE. The clean pass uses
# the team-rating core only; it was added AFTER the lockbox had been seen, so its lockbox is burned (diagnostic).
LEAK = {("nba", "script"): "NBA script uses actual missing minutes (post-opener lineup info)",
        ("nhl", "script"): "NHL script uses the confirmed starting goalie (post-opener info)"}
CLEAN_FEATSETS = {"core": ["f_core"], "core+level": ["f_core", "f_level"]}


def _uses_script(family, feat):
    return family == "raw" and feat != "core" or family == "model" and "script" in feat


def run(sports=None, clean=False):
    sports = sports or list(SPORT_MARKETS)
    featsets = CLEAN_FEATSETS if clean else FEATSETS
    families = ("model", "raw") if clean else ("model", "raw", "move")
    tag = "POSTHOC-clean" if clean else "main"
    tests, finals, diag = [], [], []
    for sp in sports:
        d = load(sp)
        for mk in SPORT_MARKETS[sp]:
            x = market_frame(d, mk)
            x.attrs["market"] = mk
            if len(x) < 400 or (x["split"] == "lock").sum() == 0:
                diag.append(dict(sport=sp, market=mk, n=len(x), note="insufficient opener data"))
                continue
            for name, feats in featsets.items():
                x[f"pred_{name}"] = wf_predict(x, feats)
            # diagnostics: correlations (discovery and lockbox)
            for part in ("disc", "lock"):
                q = x[x.split == part]
                row = dict(pass_=tag, sport=sp, market=mk, part=part, n=len(q), move_sd=q["move"].std(),
                           corr_raw_script=q["f_script"].corr(q["move"]), corr_raw_core=q["f_core"].corr(q["move"]),
                           juice_real=float(_real_juice(q, mk)))
                for name in featsets:
                    if f"pred_{name}" in q:
                        row[f"corr_{name}"] = q[f"pred_{name}"].corr(q["move"])
                diag.append(row)
            for family in families:
                best = None
                if family == "model":
                    feats_iter = list(featsets)
                elif family == "raw":
                    feats_iter = ["core"] if clean else ["-"]
                else:
                    feats_iter = ["-"]
                dirs = [1, -1] if family == "move" else [1]
                at = "close" if family == "move" else "open"
                for feat in feats_iter:
                    for T in GRID[mk][family]:
                        for dr in dirs:
                            side = _sides(x, family, feat, T, dr)
                            r = evaluate(x, mk, side, at, "disc")
                            rec = dict(pass_=tag, sport=sp, market=mk, family=family, feat=feat, thresh=T,
                                       direction=("follow" if dr == 1 else "fade") if family == "move" else "-",
                                       price=at, split_rule=split_rule(x), stage="discovery",
                                       n=r.get("n", 0), roi=r.get("roi"), z=r.get("z"), clv=r.get("clv"),
                                       units=r.get("units"))
                            tests.append(rec)
                            if r.get("n", 0) >= min(MIN_DISC_N, max(60, int(0.05 * (x.split == "disc").sum()))):
                                if best is None or r["roi"] > best[0]["roi"]:
                                    best = (r, feat, T, dr, side)
                if best is None:
                    continue
                r0, feat, T, dr, side = best
                lk = evaluate(x, mk, side, at, "lock")
                full = evaluate(x, mk, side, at, None)
                tests.append(dict(pass_=tag, sport=sp, market=mk, family=family, feat=feat, thresh=T,
                                  direction=("follow" if dr == 1 else "fade") if family == "move" else "-",
                                  price=at, split_rule=split_rule(x), stage="LOCKBOX(final)",
                                  n=lk.get("n", 0), roi=lk.get("roi"), z=lk.get("z"), clv=lk.get("clv"),
                                  units=lk.get("units")))
                leak = LEAK.get((sp, "script"), "") if _uses_script(family, feat) else ""
                finals.append(dict(
                    pass_=tag, leak=leak, sport=sp, market=mk, family=family, feat=feat, thresh=T,
                    direction=("follow" if dr == 1 else "fade") if family == "move" else "-",
                    price=at, split_rule=split_rule(x),
                    juice=("real" if _real_juice(x[x.split == "lock"], mk, at) > 0.95 else "assumed -110"),
                    d_n=r0["n"], d_roi=r0["roi"], d_clv=r0["clv"],
                    l_n=lk.get("n", 0), l_roi=lk.get("roi", np.nan), l_z=lk.get("z", np.nan),
                    l_p=lk.get("p", np.nan), l_units=lk.get("units", np.nan), l_dd=lk.get("max_dd", np.nan),
                    l_clv=lk.get("clv", np.nan), l_clv_beat=lk.get("clv_beat", np.nan),
                    seasons=full.get("seasons"), pos_seasons=full.get("pos_seasons"),
                    pct_pos=full.get("pct_pos_seasons")))
    return pd.DataFrame(tests), pd.DataFrame(finals), pd.DataFrame(diag)


def _real_juice(q, mk, at="open"):
    if mk == "spread":
        c = f"sp_{at}_h_odds"
    elif mk == "total":
        c = f"o_{at}_odds"
    else:
        c = f"ml_h_{at}"
    return q[c].notna().mean() if len(q) else 0.0


def verdicts(finals: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_verdicts(g) for _, g in finals.groupby("pass_", sort=False)], ignore_index=True)


def _verdicts(finals: pd.DataFrame) -> pd.DataFrame:
    f = finals.copy().reset_index(drop=True)
    k = int((f["l_n"] > 0).sum())
    f["alpha_bonf"] = 0.05 / max(k, 1)
    # Benjamini-Hochberg across all lockbox tests
    p = f["l_p"].fillna(1.0).values
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * len(p) / rank)
        q[i] = prev
    f["bh_q"] = q
    sig = (f["l_roi"] > 0) & ((f["l_p"] < f["alpha_bonf"]) | (f["bh_q"] < 0.10))
    stable = f["pct_pos"].fillna(0) >= 0.6
    real_price = f["juice"] == "real"
    ok = sig & stable & real_price & (f["l_n"] >= 50) & (f["leak"] == "") & (f["pass_"] == "main")
    f["verdict"] = np.where(ok, "TRUE", "FALSE")
    why = []
    for _, r in f.iterrows():
        w = []
        if r["pass_"] != "main":
            w.append("post-hoc (lockbox burned)")
        if r["leak"]:
            w.append("LEAK: " + r["leak"])
        if r["juice"] != "real":
            w.append("juice assumed -110 (not in data)")
        if not (r["l_roi"] > 0 and (r["l_p"] < r["alpha_bonf"] or r["bh_q"] < 0.10)):
            w.append("not significant")
        if not (r["pct_pos"] >= 0.6):
            w.append("unstable (<60% seasons/months positive)")
        if r["split_rule"] != "<=2022 / 2023+":
            w.append("60/40 date split (no 2023+ openers)" if r["sport"] == "nhl" else "single-season 60/40 split")
        why.append("; ".join(w))
    f["why"] = why
    return f


# ======================================================================== report
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1, f1, g1 = run()
    t2, f2, g2 = run(["nba", "nhl", "mlb"], clean=True)
    tests, finals, diag = pd.concat([t1, t2]), pd.concat([f1, f2]), pd.concat([g1, g2])
    tests.to_csv(OUT / "clv_openers_candidates.csv", index=False)
    v = verdicts(finals)
    v.to_csv(OUT / "clv_openers_finals.csv", index=False)
    diag.to_csv(OUT / "clv_openers_diag.csv", index=False)
    print(v.to_string())
    print(diag.to_string())
    return tests, v, diag


# ======================================================================== live scan
ESPN = {"ncaaf": "football/college-football", "nfl": "football/nfl", "nba": "basketball/nba",
        "ncaam": "basketball/mens-college-basketball", "nhl": "hockey/nhl", "mlb": "baseball/mlb"}


def espn_openers(sport: str, date: str | None = None) -> pd.DataFrame:
    """ESPN scoreboard DraftKings odds: opener and current line/juice for each pre-game event."""
    from nightly.providers import _get
    params = {"limit": 500}
    if date:
        params["dates"] = date.replace("-", "")
    if sport == "ncaaf":
        params["groups"] = 80
    if sport == "ncaam":
        params["groups"] = 50
    js = _get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[sport]}/scoreboard", params)
    rows = []
    for e in js.get("events", []):
        c = e["competitions"][0]
        if e.get("status", {}).get("type", {}).get("state") != "pre" or not c.get("odds"):
            continue
        o = c["odds"][0]
        teams = {t["homeAway"]: t["team"]["displayName"] for t in c["competitors"]}
        ps, tt, ml = o.get("pointSpread", {}), o.get("total", {}), o.get("moneyline", {})
        rows.append(dict(
            espn_id=e["id"], commence=e["date"], home=teams.get("home"), away=teams.get("away"),
            provider=o.get("provider", {}).get("name"),
            sp_open=_am(_dig(ps, "home", "open", "line")), sp_open_h_odds=_am(_dig(ps, "home", "open", "odds")),
            sp_open_a_odds=_am(_dig(ps, "away", "open", "odds")), sp_now_dk=_am(_dig(ps, "home", "close", "line")),
            tot_open=_am(_dig(tt, "over", "open", "line")), o_open_odds=_am(_dig(tt, "over", "open", "odds")),
            u_open_odds=_am(_dig(tt, "under", "open", "odds")), tot_now_dk=_am(_dig(tt, "over", "close", "line")),
            ml_h_open=_am(_dig(ml, "home", "open", "odds")), ml_a_open=_am(_dig(ml, "away", "open", "odds")),
            ml_h_now_dk=_am(_dig(ml, "home", "close", "odds")), ml_a_now_dk=_am(_dig(ml, "away", "close", "odds")),
        ))
    return pd.DataFrame(rows)


def _finals_table() -> pd.DataFrame:
    f = OUT / "clv_openers_finals.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def scan(sport: str, date: str | None = None, projections: pd.DataFrame | None = None,
         only_true: bool = False) -> pd.DataFrame:
    """Today's instances of the frozen candidate rules.

    * 'move' rules: compare the CURRENT FanDuel line (nightly.providers.fanduel_board) with the ESPN/DraftKings
      opener; flag games whose line moved >= k (follow/fade per the frozen rule, bet at the current price).
    * 'raw'/'model' opener rules need a pre-game script projection: pass `projections` with columns
      home, away, s_margin, s_total (and optionally p_script); signal = script - opener vs the frozen threshold.
    Returns one row per signal with the candidate's lockbox verdict so only TRUE rules are acted on.
    """
    from nightly.providers import fanduel_board, _same_team
    fin = _finals_table()
    if fin.empty:
        raise RuntimeError("run `python -m engine.edge_hunt.clv_openers` first (frozen rules live in the finals csv)")
    fin = fin[fin.sport == sport]
    fin = fin[fin["pass_"] == "main"]
    if only_true:
        fin = fin[fin.verdict == "TRUE"]
    if fin.empty:
        return pd.DataFrame()
    op = espn_openers(sport, date)
    if op.empty:
        return pd.DataFrame()
    try:
        fd = fanduel_board(sport)
    except Exception:
        fd = pd.DataFrame()
    for c in ["fd_spread", "fd_spread_h_odds", "fd_spread_a_odds", "fd_total", "fd_over", "fd_under", "fd_ml_h", "fd_ml_a"]:
        op[c] = np.nan
    for i, r in op.iterrows():
        for _, q in fd.iterrows():
            if _same_team(r["home"], q["home_name"]) and _same_team(r["away"], q["away_name"]):
                op.loc[i, ["fd_spread", "fd_spread_h_odds", "fd_spread_a_odds", "fd_total", "fd_over", "fd_under",
                           "fd_ml_h", "fd_ml_a"]] = [q.get(k) for k in ["spread_home", "spread_home_odds",
                                                                         "spread_away_odds", "total", "over_odds",
                                                                         "under_odds", "home_ml", "away_ml"]]
                break
    # current line: FanDuel if listed, else DraftKings current
    op["sp_now"] = op["fd_spread"].fillna(op["sp_now_dk"])
    op["tot_now"] = op["fd_total"].fillna(op["tot_now_dk"])
    op["ml_h_now"], op["ml_a_now"] = op["fd_ml_h"].fillna(op["ml_h_now_dk"]), op["fd_ml_a"].fillna(op["ml_a_now_dk"])
    if projections is not None and len(projections):
        op = op.merge(projections, on=["home", "away"], how="left")
    out = []
    for _, rule in fin.iterrows():
        mk, fam, T = rule["market"], rule["family"], float(rule["thresh"])
        dr = -1 if rule["direction"] == "fade" else 1
        for _, g in op.iterrows():
            sig = np.nan
            if fam == "move":
                if mk == "spread":
                    sig = -(g["sp_now"] - g["sp_open"])
                elif mk == "total":
                    sig = g["tot_now"] - g["tot_open"]
                else:
                    if all(pd.notna([g["ml_h_now"], g["ml_a_now"], g["ml_h_open"], g["ml_a_open"]])):
                        sig = float(no_vig_probs(g["ml_h_now"], g["ml_a_now"])[0] - no_vig_probs(g["ml_h_open"], g["ml_a_open"])[0])
            elif "s_margin" in g and pd.notna(g.get("s_margin")):
                if mk == "spread":
                    sig = g["sp_open"] + g["s_margin"]          # = -(script spread - opener)
                elif mk == "total":
                    sig = g["s_total"] - g["tot_open"]
                elif pd.notna(g.get("p_script")):
                    sig = g["p_script"] - float(no_vig_probs(g["ml_h_open"], g["ml_a_open"])[0])
                if fam == "model":
                    sig = np.nan  # model rules need the fitted coefficients; raw signal shown only for 'raw'
            if pd.isna(sig) or abs(sig) < T:
                continue
            side = int(np.sign(sig)) * dr
            if mk == "spread":
                pick = f"{g['home'] if side > 0 else g['away']} {('%+.1f' % (g['sp_open'] if side > 0 else -g['sp_open'])) if fam != 'move' else ('%+.1f' % (g['sp_now'] if side > 0 else -g['sp_now']))}"
            elif mk == "total":
                pick = f"{'Over' if side > 0 else 'Under'} {g['tot_open'] if fam != 'move' else g['tot_now']}"
            else:
                pick = f"{g['home'] if side > 0 else g['away']} ML"
            out.append(dict(sport=sport, market=mk, family=fam, rule=f"{rule['direction']} >= {T}",
                            game=f"{g['away']} @ {g['home']}", commence=g["commence"], signal=round(float(sig), 3),
                            pick=pick, price_at=("current FanDuel/DK" if fam == "move" else "ESPN opener"),
                            verdict=rule["verdict"], lockbox_roi=rule["l_roi"], lockbox_n=rule["l_n"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
