"""
Bayesian edge tracker.

Each strategy (and each strategy x segment) has an unknown true per-bet edge mu (expected unit P&L
per 1u risked).  Prior mu ~ N(prior_mean, prior_sd^2) with prior_mean = -0.045 (you pay the vig) and
prior_sd = 0.03 (true edges are rare and small).  Every settled bet's unit P&L x_i is treated as
x_i ~ N(mu, s^2); Normal-Normal conjugacy gives the posterior in closed form:

    precision = 1/prior_sd^2 + W/s^2,   mean = (prior_mean/prior_sd^2 + Sx/s^2) / precision

W and Sx are (optionally exponentially-forgotten) counts and sums, so an old edge that died is
eventually forgotten (half_life in bets).  s^2 blends the theoretical variance of a unit bet at its
price (dec - 1 when p ~ 1/dec) with the realised variance, so a few lucky wins do not shrink s.

From the posterior:  mean edge, P(edge > 0), fractional-Kelly stake multiplier:
    mult = 0                                   if P(edge>0) < gate (0.6)
         = clip(mean / ref_edge, 0, cap=1.5)   otherwise           (ref_edge = edge that earns 1x the tier stake)

Strategy state machine with hysteresis.  Two trackers per strategy:
    slow : the sizing posterior above (skeptical prior, long half-life, 1500 bets)
    fast : a change detector, half-life 100 bets, near-neutral prior N(0, 0.05^2), so it measures
           RECENT EVIDENCE rather than prior skepticism.
State changes require evidence (v0 of this design retired every strategy after 30 bets purely
because the skeptical prior starts at P(edge>0)=0.07 -- see results/audit_report.md):
    ACTIVE  -> WATCH    if P_fast < 0.10, or (P_slow < 0.20 and n >= 200)
    WATCH   -> ACTIVE   if P_fast >= 0.50 and P_slow >= 0.40            (after >= min_watch bets)
    WATCH   -> RETIRED  if P_fast < 0.20 and P_slow < 0.30               (after >= min_watch bets)
    RETIRED -> WATCH    if P_fast >= 0.85 and P_slow >= 0.50  (retired strategies are paper-tracked)
Stake: ACTIVE -> mult, WATCH -> 0.5*mult, RETIRED -> 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

def _sf0(mean: float, sd: float) -> float:
    """P(N(mean, sd^2) > 0)."""
    return 0.5 * math.erfc(-mean / (sd * math.sqrt(2.0)))


PRIOR_MEAN = -0.045
PRIOR_SD = 0.03


@dataclass
class EdgeTracker:
    prior_mean: float = PRIOR_MEAN
    prior_sd: float = PRIOR_SD
    half_life: float | None = 1500.0     # bets; None = never forget
    n0_var: float = 50.0                 # pseudo-count on the theoretical variance
    n: int = 0                           # raw count (no decay)
    W: float = 0.0                       # decayed count
    Sx: float = 0.0
    Sx2: float = 0.0
    Stheo: float = 0.0                   # decayed sum of theoretical variances

    def _lam(self) -> float:
        return 1.0 if not self.half_life else 0.5 ** (1.0 / self.half_life)

    def update(self, unit_pnl: float, dec: float) -> None:
        lam = self._lam()
        self.W = self.W * lam + 1.0
        self.Sx = self.Sx * lam + unit_pnl
        self.Sx2 = self.Sx2 * lam + unit_pnl * unit_pnl
        self.Stheo = self.Stheo * lam + max(dec - 1.0, 0.05)
        self.n += 1

    def update_many(self, unit_pnl, dec) -> None:
        for x, d in zip(np.asarray(unit_pnl, float), np.asarray(dec, float)):
            self.update(x, d)

    # ---------------------------------------------------------------- posterior
    def s2(self) -> float:
        if self.W <= 0:
            return 1.0
        theo = self.Stheo / self.W
        m = self.Sx / self.W
        emp = max(self.Sx2 / self.W - m * m, 1e-6)
        return (self.n0_var * theo + self.W * emp) / (self.n0_var + self.W)

    def posterior(self) -> tuple[float, float]:
        s2 = self.s2()
        prec = 1.0 / self.prior_sd ** 2 + self.W / s2
        mean = (self.prior_mean / self.prior_sd ** 2 + self.Sx / s2) / prec
        return mean, math.sqrt(1.0 / prec)

    def p_pos(self) -> float:
        m, sd = self.posterior()
        return _sf0(m, sd)

    def summary(self) -> dict:
        m, sd = self.posterior()
        return dict(n=self.n, n_eff=self.W, post_mean=m, post_sd=sd, p_pos=self.p_pos(),
                    raw_roi=self.Sx / self.W if self.W else np.nan)


def stake_multiplier(post_mean: float, p_pos: float, gate: float = 0.6, ref_edge: float = 0.03,
                     cap: float = 1.5) -> float:
    if p_pos < gate:
        return 0.0
    return float(np.clip(post_mean / ref_edge, 0.0, cap))


def clv_prior(clv_implied_roi, prior_mean: float = PRIOR_MEAN, prior_sd: float = PRIOR_SD,
              z_min: float = 2.0, weight: float = 0.7) -> tuple[float, float]:
    """CLV-informed prior.  clv_implied_roi_i = closing_prob_i * dec_i - 1 (the ROI the bet was worth
    at the closing, efficient price).  If its mean is significantly > 0 (z >= z_min), move the prior
    mean `weight` of the way to that CLV-implied edge (but never above it); prior_sd unchanged.
    CLV is known before results and is ~10x less noisy than P&L, so this is the 'knows before it is
    right' channel for a new strategy."""
    x = np.asarray(clv_implied_roi, float)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return prior_mean, prior_sd
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    if se <= 0 or m / se < z_min:
        return prior_mean, prior_sd
    return prior_mean + weight * (m - prior_mean), prior_sd


# ------------------------------------------------------------------ strategy / segment learner
STATES = ("ACTIVE", "WATCH", "RETIRED")


@dataclass
class StrategyState:
    slow: EdgeTracker
    fast: EdgeTracker
    state: str = "ACTIVE"
    in_state: int = 0
    history: list = field(default_factory=list)   # (date, from, to)


class Learner:
    def __init__(self, prior_mean=PRIOR_MEAN, prior_sd=PRIOR_SD, half_life=1500.0, fast_half_life=100.0,
                 fast_prior=(0.0, 0.05), seg_tau=0.02, gate=0.6, ref_edge=0.03, cap=1.5, min_bets=50,
                 min_watch=50,
                 priors: dict | None = None, use_segments=True):
        self.kw = dict(prior_mean=prior_mean, prior_sd=prior_sd)
        self.half_life, self.fast_half_life, self.fast_prior = half_life, fast_half_life, fast_prior
        self.seg_tau, self.gate, self.ref_edge, self.cap = seg_tau, gate, ref_edge, cap
        self.min_bets, self.min_watch = min_bets, min_watch
        self.priors = priors or {}                 # strategy -> (mean, sd) e.g. from clv_prior
        self.use_segments = use_segments
        self.strat: dict[str, StrategyState] = {}
        self.seg: dict[tuple, EdgeTracker] = {}

    def _get(self, strategy: str) -> StrategyState:
        if strategy not in self.strat:
            pm, ps = self.priors.get(strategy, (self.kw["prior_mean"], self.kw["prior_sd"]))
            self.strat[strategy] = StrategyState(
                slow=EdgeTracker(pm, ps, self.half_life), fast=EdgeTracker(*self.fast_prior, self.fast_half_life))
        return self.strat[strategy]

    def _seg(self, strategy, segment) -> EdgeTracker:
        k = (strategy, segment)
        if k not in self.seg:
            self.seg[k] = EdgeTracker(0.0, 1.0, self.half_life)   # likelihood only; prior set at query time
        return self.seg[k]

    # ---------------------------------------------------------------- updates
    def update(self, strategy: str, unit_pnl: float, dec: float, segment: str | None = None, date=None):
        S = self._get(strategy)
        S.slow.update(unit_pnl, dec)
        S.fast.update(unit_pnl, dec)
        if segment is not None and self.use_segments:
            self._seg(strategy, segment).update(unit_pnl, dec)
        S.in_state += 1
        self._transition(strategy, S, date)

    def _transition(self, name, S: StrategyState, date):
        if S.slow.n < self.min_bets:
            return
        ps, pf = S.slow.p_pos(), S.fast.p_pos()
        new = S.state
        if S.state == "ACTIVE" and (pf < 0.10 or (ps < 0.20 and S.slow.n >= 200)):
            new = "WATCH"
        elif S.state == "WATCH" and S.in_state >= self.min_watch:
            if pf >= 0.50 and ps >= 0.40:
                new = "ACTIVE"
            elif pf < 0.20 and ps < 0.30:
                new = "RETIRED"
        elif S.state == "RETIRED" and pf >= 0.85 and ps >= 0.50:
            new = "WATCH"
        if new != S.state:
            S.history.append((date, S.state, new))
            S.state, S.in_state = new, 0

    # ---------------------------------------------------------------- queries
    def strategy_posterior(self, strategy: str) -> dict:
        S = self._get(strategy)
        d = S.slow.summary()
        d.update(state=S.state, p_pos_fast=S.fast.p_pos())
        return d

    def segment_posterior(self, strategy: str, segment: str | None) -> dict:
        """Partial pooling: segment prior = strategy posterior with extra between-segment sd seg_tau."""
        sp = self.strategy_posterior(strategy)
        if segment is None or not self.use_segments or (strategy, segment) not in self.seg:
            return dict(sp, seg_n=0)
        t = self.seg[(strategy, segment)]
        pm, psd = sp["post_mean"], math.sqrt(sp["post_sd"] ** 2 + self.seg_tau ** 2)
        s2 = t.s2()
        prec = 1 / psd ** 2 + t.W / s2
        mean = (pm / psd ** 2 + t.Sx / s2) / prec
        sd = math.sqrt(1 / prec)
        return dict(sp, post_mean=mean, post_sd=sd, p_pos=_sf0(mean, sd), seg_n=t.n)

    def multiplier(self, strategy: str, segment: str | None = None) -> float:
        S = self._get(strategy)
        if S.state == "RETIRED":
            return 0.0
        post = self.segment_posterior(strategy, segment)
        m = stake_multiplier(post["post_mean"], post["p_pos"], self.gate, self.ref_edge, self.cap)
        return m * (0.5 if S.state == "WATCH" else 1.0)

    def table(self):
        import pandas as pd
        return pd.DataFrame([dict(strategy=k, **self.strategy_posterior(k), mult=self.multiplier(k))
                             for k in self.strat]).sort_values("p_pos", ascending=False)

    # ---------------------------------------------------------------- from journal
    @classmethod
    def from_journal(cls, journal, **kw) -> "Learner":
        """Rebuild from settled journal rows (date order). Uses unit P&L (per 1u risked) so stake
        decisions never feed back into the edge estimate.  Zero-stake (paper) bets count too."""
        from backtest.core import american_to_decimal
        L = cls(**kw)
        j = journal[journal["result"].notna()].sort_values("settled_at")
        for _, r in j.iterrows():
            dec = float(american_to_decimal(r["price_taken"]))
            x = (dec - 1.0) if r["result"] > 0 else (-1.0 if r["result"] < 0 else 0.0)
            L.update(r["strategy"], x, dec, r.get("segment"), r.get("settled_at"))
        return L
