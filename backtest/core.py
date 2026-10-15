"""
Core backtest utilities: odds math, bet settlement, performance statistics.

Design rules (enforced everywhere in this package):
  * Every bet is settled at the CLOSING price that was actually available.
    When a dataset lacks juice for spreads/totals we assume standard -110.
  * Pushes return stake (0 units). All results are in units risked = 1.
  * No feature may use information from the game being bet (see features.py).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

DEFAULT_JUICE = -110.0


# ---------------------------------------------------------------- odds math
def american_to_decimal(odds):
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, 1 + o / 100.0, 1 + 100.0 / np.abs(o))


def decimal_to_american(dec):
    """Decimal odds -> American (+150 / -120). Scalar or array."""
    d = np.asarray(dec, dtype=float)
    return np.where(d >= 2.0, (d - 1) * 100.0, -100.0 / (d - 1))


def prob_to_american(p):
    """Price-as-probability (Polymarket/Kalshi share price, 0-1) -> American odds you actually get."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return decimal_to_american(1.0 / p)


def fmt_american(x) -> str:
    """+150 / -120 display; rounds to whole cents like a sportsbook."""
    v = float(x)
    return f"{int(round(v)):+d}"


def american_to_prob(odds):
    """Raw (vigged) implied probability."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, 100.0 / (o + 100.0), np.abs(o) / (np.abs(o) + 100.0))


def no_vig_probs(odds_a, odds_b):
    """Multiplicative de-vig of a two-way market -> (p_a, p_b, overround)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    s = pa + pb
    return pa / s, pb / s, s


def win_profit(odds):
    """Profit on a 1u winning bet at American odds."""
    return american_to_decimal(odds) - 1.0


# ---------------------------------------------------------------- settlement
def settle(outcome, odds):
    """outcome: +1 win, 0 push, -1 loss (array). Returns unit P&L."""
    outcome = np.asarray(outcome, dtype=float)
    prof = win_profit(odds)
    return np.where(outcome > 0, prof, np.where(outcome < 0, -1.0, 0.0))


def sign_outcome(x):
    """Margin vs line -> +1/0/-1 (handles NaN as NaN)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x)


# ---------------------------------------------------------------- stats
def perf(pnl: pd.Series, odds: pd.Series | None = None, seasons: pd.Series | None = None) -> dict:
    """Summary stats for a vector of per-bet unit P&L."""
    pnl = pd.Series(pnl, dtype=float).dropna()
    n = len(pnl)
    if n == 0:
        return dict(n=0)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    pushes = n - wins - losses
    roi = pnl.mean()
    sd = pnl.std(ddof=1) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    z = roi / se if se and se > 0 else float("nan")
    # one-sided p-value that true ROI > 0 (normal approx on unit P&L)
    p = 0.5 * math.erfc(z / math.sqrt(2)) if not math.isnan(z) else float("nan")
    cum = pnl.cumsum()
    max_dd = float((cum.cummax() - cum).max())
    out = dict(
        n=n, wins=wins, losses=losses, pushes=pushes,
        win_pct=wins / max(wins + losses, 1),
        units=float(pnl.sum()), roi=float(roi), z=float(z), p=float(p),
        ci_lo=float(roi - 1.96 * se), ci_hi=float(roi + 1.96 * se),
        max_dd=max_dd,
    )
    if odds is not None:
        o = pd.Series(odds).loc[pnl.index]
        out["avg_dec_odds"] = float(np.mean(american_to_decimal(o)))
        # break-even win% at the average price actually bet
        out["breakeven"] = float(np.mean(american_to_prob(o)))
    if seasons is not None:
        s = pd.Series(seasons).loc[pnl.index]
        by = pnl.groupby(s).sum()
        out["seasons"] = int(len(by))
        out["pos_seasons"] = int((by > 0).sum())
        out["pct_pos_seasons"] = float((by > 0).mean())
    return out


def shrunk_roi(roi, n, sd=1.0, prior_sd=0.03, prior_mean=-0.045):
    """Empirical-Bayes shrinkage of an observed ROI toward the typical
    'pay the vig' outcome. prior_sd ~3% reflects how rare true edges are."""
    var_obs = sd ** 2 / max(n, 1)
    w = prior_sd ** 2 / (prior_sd ** 2 + var_obs)
    return prior_mean + w * (roi - prior_mean)


def kelly_fraction(p_win, odds):
    b = win_profit(odds)
    return np.maximum((b * p_win - (1 - p_win)) / b, 0.0)
