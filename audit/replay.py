"""
THE PROOF: walk-forward replay of the auditor over history, strictly no lookahead.

  python -m audit.replay            # everything -> results/audit_*.csv + printed summary

At each date the learner / calibrator / CLV model know ONLY bets settled on earlier dates.  Stakes
for every bet on date d are fixed from that state, then the day's results are revealed and learned.
Retired / zero-stake bets are still observed (paper-tracked) — the strategy keeps firing signals.

Policies compared (same bets, different stakes):
  static       tier stakes A=1, B=0.75, C=0.5, props 0.5 (nightly/card.py STAKE)
  learner      static x Learner multiplier (strategy x segment posterior, state machine)
  learner_cal  learner, and for props additionally x clip(ev_calibrated / ev_raw, 0, 1.5)
  cal_only     static, props x clip(ev_cal/ev_raw) only (isolates the recalibrator)

Also:
  * cold-start holdout: learner sees ONLY holdout-season bets (no warm start from the discovery
    seasons, whose results were used to select the strategies and are therefore optimistic).
  * regime-flip test: after a strategy's holdout start, replace its bets with a bootstrap that
    over-samples its losing bets until true ROI = target; measure bets until stake is cut / retired.
  * historical CLV study where open & close lines exist.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from backtest.core import american_to_decimal, american_to_prob
from backtest.parlay import load_ledger

from .calibration import OnlineRecalibrator, calibration_summary, reliability
from .confidence import GRADES, grade_of
from .journal import LINE_SIGMA, PHI0, default_segment, price_bucket
from .learner import Learner, clv_prior

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATA = ROOT / "data"
STAKE = {"A": 1.0, "B": 0.75, "C": 0.5, "prop": 0.5}
PROP_STRAT = "NBA points: blend EV"

SPEC = "spec (prior -4.5%+/-3%, half-life 1500)"
CONFIGS = {
    SPEC: dict(),
    "spec, half-life 500": dict(half_life=500.0),
    "no forgetting": dict(half_life=None),
    "validated prior (0%+/-3%)": dict(prior_mean=0.0),
}


# =================================================================== data
def _market(pick: str) -> str:
    u = str(pick).upper()
    if u.startswith("OVER") or u.startswith("UNDER"):
        return "tot"
    return "ml" if u.endswith(" ML") else "ats"


def load_bets() -> pd.DataFrame:
    L = load_ledger()   # one leg per event per day, as the card bets
    L["market"] = L.pick.map(_market)
    L["segment"] = [default_segment(s, m, p, pr) for s, m, p, pr in zip(L.sport, L.market, L.pick, L.price)]
    L["unit"] = L.pnl.astype(float)
    L["model_prob"] = np.nan
    L["y"] = np.where(L.outcome > 0, 1.0, np.where(L.outcome < 0, 0.0, np.nan))
    L["cons_gap"] = np.nan
    P = pd.read_csv(RES / "nba_props_bets_p_blend_0.04.csv", parse_dates=["date"])
    over = P.side == "over"
    p_side = np.where(over, P.p_blend, 1 - P.p_blend)
    p_cons = np.where(over, P.p_mkt, 1 - P.p_mkt)
    p_book = (1 / P.dec) / (1 / P.dec_o + 1 / P.dec_u)
    props = pd.DataFrame(dict(
        date=P.date, season=P.season, sport="nba-prop", strategy=PROP_STRAT,
        event=P.team + " vs " + P.opp, pick=P.player + " " + P.side.str.upper() + " " + P.line.astype(str),
        price=P.price.astype(float), outcome=np.where(P.win, 1, -1), pnl=P.pnl, dec=P.dec, tier="prop",
        holdout=P.season >= 2026, market="prop",
        segment=["prop|%s|%s|%s" % (s, "alt" if a else "main", price_bucket(pr))
                 for s, a, pr in zip(P.side, P.alt_line, P.price)],
        unit=P.pnl.astype(float), model_prob=p_side, y=P.win.astype(float), cons_gap=p_cons - p_book))
    B = pd.concat([L, props], ignore_index=True).sort_values(["date", "strategy"]).reset_index(drop=True)
    B["base"] = B.tier.map(STAKE)
    return B


# =================================================================== metrics
def metrics(stake, unit, dates) -> dict:
    stake, unit = np.asarray(stake, float), np.asarray(unit, float)
    pnl = stake * unit
    daily = pd.Series(pnl).groupby(np.asarray(dates)).sum()
    cum = np.concatenate([[0.0], daily.cumsum().values])
    staked = stake.sum()
    return dict(bets=int((stake > 0).sum()), staked=round(staked, 1), units=round(pnl.sum(), 1),
                roi=pnl.sum() / staked if staked > 0 else np.nan,
                max_dd=float((np.maximum.accumulate(cum) - cum).max()),
                sharpe_daily=float(daily.mean() / daily.std()) if daily.std() > 0 else np.nan,
                days=int(len(daily)))


def paired_t(a, b, unit, dates) -> float:
    """t-stat of mean daily P&L difference (policy a minus policy b)."""
    d = pd.Series((np.asarray(a) - np.asarray(b)) * np.asarray(unit)).groupby(np.asarray(dates)).sum()
    return float(d.mean() / (d.std() / math.sqrt(len(d)))) if d.std() > 0 else np.nan


# =================================================================== walk-forward
def walk(B: pd.DataFrame, learner_kw: dict | None = None, clv_rows: pd.DataFrame | None = None,
         cal_method="platt", cal_window=1000, clv_k=50.0, priors=None) -> pd.DataFrame:
    """Return B with stake columns + pre-bet grade.  No lookahead: state used on date d is built from
    dates < d only."""
    learner = Learner(priors=priors, **(learner_kw or {}))
    cal = OnlineRecalibrator(cal_method, window=cal_window)
    ph, yh = [], []
    clv_sum, clv_n = {}, {}
    clv_by_date = {} if clv_rows is None else {d: g for d, g in clv_rows.groupby("date")}
    n = len(B)
    out = {k: np.zeros(n) for k in ("s_learner", "s_cal", "s_calonly", "p_cal", "p_edge", "ev_cal", "exp_clv")}
    grade = np.empty(n, dtype=object)
    for date, idx in B.groupby("date", sort=True).indices.items():
        day = B.iloc[idx]
        if (day.strategy == PROP_STRAT).any():
            cal.fit(ph, yh)
        for j, r in zip(idx, day.itertuples(index=False)):
            post = learner.segment_posterior(r.strategy, r.segment)
            m = learner.multiplier(r.strategy, r.segment)
            if r.strategy == PROP_STRAT:
                p_raw = r.model_prob
                p_cal = float(cal.transform([p_raw])[0])
                ev_raw, ev_cal = p_raw * r.dec - 1, p_cal * r.dec - 1
                adj = float(np.clip(ev_cal / ev_raw, 0, 1.5)) if ev_raw > 0 else 0.0
                exp_clv = 0.5 * r.cons_gap
            else:
                p_cal = (1 + post["post_mean"]) / r.dec
                ev_cal, adj = post["post_mean"], 1.0
                exp_clv = clv_sum.get(r.strategy, 0.0) / (clv_n.get(r.strategy, 0) + clv_k)
            out["s_learner"][j] = r.base * m
            out["s_cal"][j] = r.base * m * adj
            out["s_calonly"][j] = r.base * adj
            out["p_cal"][j], out["p_edge"][j], out["ev_cal"][j], out["exp_clv"][j] = p_cal, post["p_pos"], ev_cal, exp_clv
            grade[j] = grade_of(post["p_pos"], ev_cal, exp_clv, post["state"])
        # ---- results revealed: learn
        for r in day.itertuples(index=False):
            learner.update(r.strategy, r.unit, r.dec, r.segment, date)
            if r.strategy == PROP_STRAT:
                ph.append(r.model_prob)
                yh.append(r.y)
        if date in clv_by_date:
            for s, c in zip(clv_by_date[date].strategy, clv_by_date[date].clv_prob):
                if not np.isnan(c):
                    clv_sum[s] = clv_sum.get(s, 0.0) + c
                    clv_n[s] = clv_n.get(s, 0) + 1
    R = B.copy()
    for k, v in out.items():
        R[k] = v
    R["s_static"] = R.base
    R["grade"] = grade
    R.attrs["learner"] = learner
    return R


def summarize(R: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for period, mask in (("all", np.ones(len(R), bool)), ("in-sample", ~R.holdout.values), ("holdout", R.holdout.values)):
        X = R[mask]
        if X.empty:
            continue
        for pol, col in (("static", "s_static"), ("learner", "s_learner"), ("learner_cal", "s_cal"), ("cal_only", "s_calonly")):
            d = metrics(X[col], X.unit, X.date)
            d.update(config=label, period=period, policy=pol,
                     t_vs_static=paired_t(X[col], X.s_static, X.unit, X.date) if pol != "static" else np.nan)
            rows.append(d)
    return rows


# =================================================================== regime flip
def regime_test(B: pd.DataFrame, strategy: str, target_roi: float, seeds=20, horizon=1000,
                learner_kw: dict | None = None) -> dict:
    """Strategies are learned independently, so only this strategy's bets are replayed.  Pre-flip = real
    in-sample bets; post-flip = `horizon` synthetic bets bootstrapped from the strategy's own bets
    with losing bets over-sampled so that E[unit] = target_roi."""
    S = B[B.strategy == strategy].sort_values("date")
    pre = S[~S.holdout]
    if len(pre) < 50:
        pre = S.iloc[: len(S) // 2]
    m_all = S.unit.mean()
    q = max((m_all - target_roi) / (m_all + 1), 0.0)
    losers = S[S.unit < 0]
    base = S.base.iloc[0]
    per_year = len(S) / max((S.date.max() - S.date.min()).days / 365.25, 1.0)
    res = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        lr = Learner(**(learner_kw or {}))
        for r in pre.itertuples(index=False):
            lr.update(strategy, r.unit, r.dec, r.segment)
        m0 = lr.multiplier(strategy)
        pick_l = rng.random(horizon) < q
        ia = rng.integers(0, len(S), horizon)
        il = rng.integers(0, len(losers), horizon)
        u = np.where(pick_l, losers.unit.values[il], S.unit.values[ia])
        dd = np.where(pick_l, losers.dec.values[il], S.dec.values[ia])
        sg = np.where(pick_l, losers.segment.values[il], S.segment.values[ia])
        lag_watch = lag_zero = lag_ret = np.nan
        st_static = st_learn = 0.0
        for k in range(horizon):
            mult = lr.multiplier(strategy, sg[k])
            state = lr.strat[strategy].state
            if np.isnan(lag_watch) and state != "ACTIVE":
                lag_watch = k
            if np.isnan(lag_zero) and mult == 0:
                lag_zero = k
            if np.isnan(lag_ret) and state == "RETIRED":
                lag_ret = k
            if k < 300:
                st_static += base * u[k]
                st_learn += base * mult * u[k]
            lr.update(strategy, u[k], dd[k], sg[k])
        res.append(dict(mult_at_flip=m0, lag_watch=lag_watch, lag_zero=lag_zero, lag_retired=lag_ret,
                        units_300_static=st_static, units_300_learner=st_learn, realized=u.mean()))
    D = pd.DataFrame(res)
    return dict(strategy=strategy, target_roi=target_roi, bets_per_year=round(per_year, 0),
                mult_at_flip=round(D.mult_at_flip.mean(), 2),
                realized_roi=round(D.realized.mean(), 3),
                med_bets_to_watch=D.lag_watch.median(), med_bets_to_zero_stake=D.lag_zero.median(),
                med_bets_to_retired=D.lag_retired.median(),
                pct_retired_within_horizon=float(D.lag_retired.notna().mean()),
                yrs_to_zero_stake=round(D.lag_zero.median() / per_year, 2) if D.lag_zero.notna().any() else np.nan,
                units_first300_static=round(D.units_300_static.mean(), 1),
                units_first300_learner=round(D.units_300_learner.mean(), 1))


# =================================================================== CLV study
def _devig(a, b):
    pa, pb = american_to_prob(a), american_to_prob(b)
    return pa / (pa + pb)


def clv_study(raw_ledger: pd.DataFrame | None = None):
    """Per-bet CLV if each strategy bet had been placed at the OPEN line, vs the CLOSE.
    clv_prob: fair probability of our exact open bet at close minus at open (prob points).
    unit_open: realised unit P&L had we bet the open number (spreads/totals at -110)."""
    L = raw_ledger if raw_ledger is not None else pd.read_csv(RES / "ledger.csv", parse_dates=["date"])
    L = L.copy()
    L[["away", "home"]] = L.event.str.split(" @ ", n=1, expand=True)
    rows = []

    def load(sp):
        g = pd.read_csv(DATA / "processed" / f"{sp}.csv", parse_dates=["date"], low_memory=False)
        return g.drop(columns=[c for c in ("season", "game_id") if c in g])

    # NBA totals
    g = load("nba")
    x = L[L.sport == "nba"].merge(g, on=["date", "home", "away"], how="inner").dropna(subset=["total_open", "total"])
    over = x.pick.str.upper().str.startswith("OVER")
    pts = np.where(over, x.total - x.total_open, x.total_open - x.total)
    res = (x.home_score + x.away_score - x.total_open) * np.where(over, 1, -1)
    rows.append(pd.DataFrame(dict(date=x.date, season=x.season, sport="nba", market="tot", strategy=x.strategy,
                                  clv_pts=pts, clv_prob=PHI0 / LINE_SIGMA[("nba", "tot")] * pts,
                                  unit_open=np.where(res > 0, 100 / 110, np.where(res < 0, -1.0, 0.0)),
                                  unit_close=x.pnl, dec_open=1 + 100 / 110)))
    # NCAAF / NCAAM spreads (NCAAM picks are ML: CLV measured on the spread move, ROI-at-open unavailable)
    for sp in ("ncaaf", "ncaam"):
        g = load(sp)
        x = L[L.sport == sp].merge(g, on=["date", "home", "away"], how="inner").dropna(subset=["spread_open_home", "spread_home"])
        team = x.pick.str.replace(r"\s+(ML|[+-]?\d+(\.\d+)?)$", "", regex=True)
        hs = (team == x.home).values
        open_l = np.where(hs, x.spread_open_home, -x.spread_open_home)
        close_l = np.where(hs, x.spread_home, -x.spread_home)
        margin = np.where(hs, x.home_score - x.away_score, x.away_score - x.home_score)
        pts = open_l - close_l
        sig = LINE_SIGMA[(sp, "ats")]
        if sp == "ncaaf":
            clvp = PHI0 / sig * pts
            r = margin + open_l
            uo = np.where(r > 0, 100 / 110, np.where(r < 0, -1.0, 0.0))
        else:  # ML bet: local density of the margin distribution at the closing spread
            clvp = stats.norm.pdf(close_l / sig) / sig * pts
            uo = np.full(len(x), np.nan)
        rows.append(pd.DataFrame(dict(date=x.date, season=x.season, sport=sp, market="ats" if sp == "ncaaf" else "ml",
                                      strategy=x.strategy, clv_pts=pts, clv_prob=clvp, unit_open=uo,
                                      unit_close=x.pnl, dec_open=1 + 100 / 110 if sp == "ncaaf" else np.nan)))
    # NHL moneylines (open & close prices)
    g = load("nhl")
    x = L[L.sport == "nhl"].merge(g, on=["date", "home", "away"], how="inner").dropna(subset=["home_ml_open", "away_ml_open", "home_ml", "away_ml"])
    hs = (x.pick.str.replace(" ML", "", regex=False) == x.home).values
    p_open = np.where(hs, _devig(x.home_ml_open, x.away_ml_open), _devig(x.away_ml_open, x.home_ml_open))
    p_close = np.where(hs, _devig(x.home_ml, x.away_ml), _devig(x.away_ml, x.home_ml))
    o_price = np.where(hs, x.home_ml_open, x.away_ml_open)
    won = np.where(hs, x.home_score > x.away_score, x.away_score > x.home_score)
    dec_o = american_to_decimal(o_price)
    rows.append(pd.DataFrame(dict(date=x.date, season=x.season, sport="nhl", market="ml", strategy=x.strategy,
                                  clv_pts=np.nan, clv_prob=p_close - p_open,
                                  unit_open=np.where(won, dec_o - 1, -1.0), unit_close=x.pnl, dec_open=dec_o,
                                  p_close_nhl=p_close)))
    C = pd.concat(rows, ignore_index=True)
    # CLV-implied ROI: worth of the open bet at the (efficient) closing fair probability.
    # spreads/totals: close is ~50/50 at the closing number, so our open number is 0.5 + clv_prob.
    C["p_close_fair"] = np.where(C.market == "ml", np.nan, 0.5 + C.clv_prob)
    C.loc[C.sport == "nhl", "p_close_fair"] = C.loc[C.sport == "nhl", "p_close_nhl"]
    C["clv_roi"] = C.p_close_fair * C.dec_open - 1
    C = C.drop(columns="p_close_nhl")
    return C


def market_clv(min_season=None) -> pd.DataFrame:
    """Market-wide sanity check: every game side with open & close.  Bet at OPEN the side the line later
    moved toward (positive CLV) vs the side it moved away from.  Shows CLV -> ROI in this data."""
    out = []

    def load(sp):
        return pd.read_csv(DATA / "processed" / f"{sp}.csv", parse_dates=["date"], low_memory=False)
    for sp, mk in (("nba", "tot"), ("nba", "ats"), ("ncaaf", "ats"), ("ncaaf", "tot")):
        g = load(sp)
        oc, cc = ("total_open", "total") if mk == "tot" else ("spread_open_home", "spread_home")
        g = g.dropna(subset=[oc, cc, "home_score"])
        mv = g[cc] - g[oc]           # tot: + = moved up (over side gained); ats home line: + = home got more points
        if mk == "tot":
            res = g.home_score + g.away_score - g[oc]          # over at open
        else:
            res = g.home_score - g.away_score + g[oc]          # home at open
        # side the line moved toward: tot up -> over had value at open; spread_home up -> away was bet (away value)
        side = np.sign(mv) if mk == "tot" else -np.sign(mv)
        m = side != 0
        r = (res * side)[m]
        unit = np.where(r > 0, 100 / 110, np.where(r < 0, -1.0, 0.0))
        out.append(dict(sport=sp, market=mk, n_moved=int(m.sum()), pct_moved=float(m.mean()),
                        roi_with_move=float(unit.mean()), roi_against_move=float((-np.where(r < 0, -100 / 110, np.where(r > 0, 1.0, 0.0))).mean()),
                        mean_abs_move=float(mv[m].abs().mean())))
    for sp in ("nhl", "mlb"):
        g = load(sp).dropna(subset=["home_ml_open", "away_ml_open", "home_ml", "away_ml", "home_score"])
        ph_o, ph_c = _devig(g.home_ml_open, g.away_ml_open), _devig(g.home_ml, g.away_ml)
        d = ph_c - ph_o
        hw = g.home_score > g.away_score
        m = np.abs(d) > 0.005
        side_home = d > 0
        price = np.where(side_home, g.home_ml_open, g.away_ml_open)
        won = np.where(side_home, hw, ~hw)
        unit = np.where(won, american_to_decimal(price) - 1, -1.0)[m]
        price_b = np.where(side_home, g.away_ml_open, g.home_ml_open)
        unit_b = np.where(~won, american_to_decimal(price_b) - 1, -1.0)[m]
        out.append(dict(sport=sp, market="ml", n_moved=int(m.sum()), pct_moved=float(m.mean()),
                        roi_with_move=float(unit.mean()), roi_against_move=float(unit_b.mean()),
                        mean_abs_move=float(np.abs(d[m]).mean())))
    return pd.DataFrame(out)


def clv_correlations(C: pd.DataFrame, min_n=15) -> dict:
    G = C.groupby(["strategy", "season"]).agg(n=("clv_prob", "size"), clv=("clv_prob", "mean"),
                                               roi_open=("unit_open", "mean"), roi_close=("unit_close", "mean")).reset_index()
    G = G[G.n >= min_n]
    out = {}
    for col in ("roi_open", "roi_close"):
        h = G.dropna(subset=[col])
        if len(h) >= 5:
            pr = stats.pearsonr(h.clv, h[col])
            sr = stats.spearmanr(h.clv, h[col])
            out[f"group_{col}"] = dict(n_groups=len(h), pearson=round(pr.statistic, 3), p=round(pr.pvalue, 4),
                                       spearman=round(sr.statistic, 3), sp_p=round(sr.pvalue, 4))
    b = C.dropna(subset=["unit_open"])
    pr = stats.pearsonr(b.clv_prob, b.unit_open)
    out["per_bet_unit_open"] = dict(n=len(b), pearson=round(pr.statistic, 4), p=round(pr.pvalue, 5))
    # ROI at open by CLV sign
    s = np.sign(b.clv_prob.round(6))
    out["roi_open_by_clv_sign"] = {str(int(k)): dict(n=int(v.size), roi=round(v.mean(), 4))
                                   for k, v in b.unit_open.groupby(s)}
    return out, G


# =================================================================== production helper
def warm_learner(journal_dir=None, **learner_kw) -> Learner:
    """Learner state for tonight's card: all historical ledger + props bets, then every settled
    journal bet (date order).  Unit P&L, so zero-stake paper bets count too."""
    from .journal import load as load_journal
    lr = Learner(**learner_kw)
    for r in load_bets().itertuples(index=False):
        lr.update(r.strategy, r.unit, r.dec, r.segment, r.date)
    j = load_journal(journal_dir)
    j = j[j.result.notna()].sort_values("settled_at")
    for r in j.itertuples(index=False):
        dec = float(american_to_decimal(r.price_taken))
        x = (dec - 1.0) if r.result > 0 else (-1.0 if r.result < 0 else 0.0)
        lr.update(r.strategy, x, dec, r.segment, r.settled_at)
    return lr


# =================================================================== main
def main():
    B = load_bets()
    print(f"bets: {len(B)}  ({B.date.min().date()} .. {B.date.max().date()}), holdout {B.holdout.sum()}")
    C = clv_study()
    raw = pd.read_csv(RES / "ledger.csv", parse_dates=["date"])

    # ---------------- CLV study
    clv_tab = C.groupby(["sport", "strategy"]).agg(
        n=("clv_prob", "size"), seasons=("season", lambda s: f"{s.min()}-{s.max()}"),
        mean_clv_pts=("clv_pts", "mean"), mean_clv_prob=("clv_prob", "mean"),
        pct_pos=("clv_prob", lambda v: (v > 1e-9).mean()), pct_neg=("clv_prob", lambda v: (v < -1e-9).mean()),
        t=("clv_prob", lambda v: v.mean() / (v.std() / math.sqrt(len(v)))),
        clv_implied_roi=("clv_roi", "mean"), roi_if_open=("unit_open", "mean"), roi_at_close=("unit_close", "mean")
    ).reset_index()
    corr, G = clv_correlations(C)
    mk = market_clv()
    print("\n=== CLV study (bets placed at OPEN vs CLOSE)\n", clv_tab.round(4).to_string(index=False))
    print("\ncorrelations:", json.dumps(corr, indent=1, default=float))
    print("\nmarket-wide: bet the side the line moved toward, at open\n", mk.round(4).to_string(index=False))
    clv_tab.to_csv(RES / "audit_clv_strategies.csv", index=False)
    G.to_csv(RES / "audit_clv_groups.csv", index=False)
    mk.to_csv(RES / "audit_clv_market.csv", index=False)

    # CLV rows for the walk (team strategies, keyed by strategy/date; CLV realised at close, known pre-result)
    clv_rows = C[["date", "strategy", "clv_prob"]]

    # ---------------- walk-forward
    rows, keep = [], {}
    for label, kw in CONFIGS.items():
        R = walk(B, kw, clv_rows)
        rows += summarize(R, label)
        keep[label] = R
    # CLV-informed prior: per strategy, CLV-implied ROI from IN-SAMPLE seasons only
    pri = {}
    for s, g in C.merge(raw[["date", "strategy", "season"]].drop_duplicates(), how="left").groupby("strategy"):
        hold_start = B[(B.strategy == s) & B.holdout].date.min()
        pm, ps = clv_prior(g.loc[g.date < hold_start, "clv_roi"].values)
        pri[s] = (pm, ps)
    R = walk(B, {}, clv_rows, priors=pri)
    rows += summarize(R, "spec + CLV prior")
    keep["spec + CLV prior"] = R
    # cold start on holdout only
    H = B[B.holdout].reset_index(drop=True)
    Rh = walk(H, {}, clv_rows)
    for d in summarize(Rh, "spec, COLD START (holdout only)"):
        if d["period"] == "all":
            d["period"] = "holdout"
            rows.append(d)
    S = pd.DataFrame(rows)
    cols = ["config", "period", "policy", "bets", "staked", "units", "roi", "max_dd", "sharpe_daily", "t_vs_static"]
    S = S[cols]
    S.to_csv(RES / "audit_replay.csv", index=False)
    print("\n=== walk-forward replay\n", S.round(4).to_string(index=False))

    R0 = keep[SPEC]
    # per-strategy holdout breakdown (spec)
    ps = []
    for s, X in R0[R0.holdout].groupby("strategy"):
        a, b = metrics(X.s_static, X.unit, X.date), metrics(X.s_cal, X.unit, X.date)
        ps.append(dict(strategy=s, n=len(X), roi=X.unit.mean(), static_units=a["units"], learner_units=b["units"],
                       learner_staked=b["staked"], static_staked=a["staked"]))
    ps = pd.DataFrame(ps)
    print("\nholdout per strategy (spec):\n", ps.round(3).to_string(index=False))
    ps.to_csv(RES / "audit_replay_by_strategy.csv", index=False)
    lt = R0.attrs["learner"]
    states = pd.DataFrame([dict(strategy=k, transitions=[(str(pd.Timestamp(d).date()) if d is not None else None, a, b)
                                                         for d, a, b in v.history], **lt.strategy_posterior(k))
                           for k, v in lt.strat.items()])
    print("\nfinal learner state (spec):\n", states.round(3).to_string(index=False))
    states.to_csv(RES / "audit_learner_state.csv", index=False)

    # ---------------- does the pre-bet grade predict? (holdout, unit ROI by grade)
    gr = R0[R0.holdout].groupby("grade").agg(n=("unit", "size"), roi=("unit", "mean"),
                                              mean_p_edge=("p_edge", "mean")).reindex(GRADES).dropna(how="all")
    print("\nholdout unit ROI by pre-bet grade (spec):\n", gr.round(4).to_string())
    gr.to_csv(RES / "audit_grades.csv")

    # ---------------- calibration (props), out-of-sample walk-forward
    Pp = R0[R0.strategy == PROP_STRAT]
    cal_rows = []
    for season, X in Pp.groupby("season"):
        for name, col in (("raw p_blend (side)", "model_prob"), ("walk-forward Platt", "p_cal")):
            d = calibration_summary(X[col], X.y)
            d.update(season=int(season), prob=name)
            cal_rows.append(d)
    Rc = walk(B[B.strategy == PROP_STRAT].reset_index(drop=True), {}, None, cal_method="isotonic")
    for season, X in Rc.groupby("season"):
        d = calibration_summary(X.p_cal, X.y)
        d.update(season=int(season), prob="walk-forward isotonic")
        cal_rows.append(d)
    CAL = pd.DataFrame(cal_rows)[["season", "prob", "n", "mean_p", "win_rate", "brier", "brier_const", "ece",
                                  "cal_slope", "cal_intercept"]]
    print("\n=== props calibration\n", CAL.round(4).to_string(index=False))
    CAL.to_csv(RES / "audit_calibration.csv", index=False)
    reliability(Pp.model_prob, Pp.y, 8).to_csv(RES / "audit_reliability_props.csv", index=False)
    print(reliability(Pp[Pp.holdout].model_prob, Pp[Pp.holdout].y, 8).round(3).to_string())
    # props: does the learner+cal ROI per staked unit beat static? already in replay table

    # ---------------- regime flip
    reg = []
    for cfg, kw in ((SPEC, {}), ("validated prior (0%+/-3%)", dict(prior_mean=0.0))):
        for s in B.strategy.unique():
            for tgt in (-0.045, -0.15):
                reg.append(dict(config=cfg, **regime_test(B, s, tgt, seeds=20, learner_kw=kw)))
    REG = pd.DataFrame(reg)
    print("\n=== regime flip\n", REG.drop(columns="config").to_string(index=False))
    REG.to_csv(RES / "audit_regime.csv", index=False)
    return S, clv_tab, corr, mk, REG, CAL, gr


if __name__ == "__main__":
    main()
