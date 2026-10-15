"""
Pre-bet confidence: "does it know before it's right?"

Three independent, *pre-result* signals are combined:
  1. p_cal      calibrated win probability (model_prob through the OnlineRecalibrator; for strategies
                with no model probability, the market-fair prob shifted by the learner's posterior edge:
                p = (1 + post_mean) / dec)
  2. p_edge     posterior P(edge > 0) of the bet's strategy x segment (Learner)
  3. exp_clv    expected closing-line value in probability points (CLVModel: shrunk historical mean CLV
                of the strategy / sport-market where open+close lines exist; for props a consensus-gap
                proxy: other books' fair prob minus this book's de-vigged prob, times a decay factor)

Grade (all must hold; first match wins)
  A+ : p_edge >= 0.90, ev_cal >= 0.03, exp_clv >= 0
  A  : p_edge >= 0.80, ev_cal >= 0.02, exp_clv >= -0.005
  B  : p_edge >= 0.70, ev_cal >= 0.01
  C  : p_edge >= 0.60, ev_cal >  0
  PASS otherwise (or strategy RETIRED)
Recommended stake = tier base stake x learner multiplier x clip(ev_cal / ev_raw, 0, 1.5) (props), 0 on PASS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.core import american_to_decimal

TIER_STAKE = {"A": 1.0, "B": 0.75, "C": 0.5, "prop": 0.5}
GRADES = ["A+", "A", "B", "C", "PASS"]


class CLVModel:
    """Expected CLV (prob points) by strategy, falling back to sport|market, then 0.
    Group means are shrunk toward 0 with pseudo-count k (CLV is noisy per bet, ~1-3 pp sd)."""

    def __init__(self, k: float = 50.0, prop_gap_weight: float = 0.5):
        self.k, self.prop_gap_weight = k, prop_gap_weight
        self.by_strat: dict = {}
        self.by_mkt: dict = {}

    def fit(self, rows: pd.DataFrame) -> "CLVModel":
        """rows: columns strategy, sport, market, clv_prob (e.g. replay.clv_study() per-bet output or
        the settled journal)."""
        r = rows.dropna(subset=["clv_prob"])
        for key, store in ((["strategy"], self.by_strat), (["sport", "market"], self.by_mkt)):
            g = r.groupby(key)["clv_prob"].agg(["sum", "size"])
            for idx, v in g.iterrows():
                store[idx] = v["sum"] / (v["size"] + self.k)
        return self

    def predict(self, strategy=None, sport=None, market=None, consensus_prob=None, market_prob=None) -> float:
        if consensus_prob is not None and market_prob is not None and not pd.isna(consensus_prob):
            return self.prop_gap_weight * (float(consensus_prob) - float(market_prob))
        if strategy in self.by_strat:
            return float(self.by_strat[strategy])
        if (sport, market) in self.by_mkt:
            return float(self.by_mkt[(sport, market)])
        return 0.0


def grade_of(p_edge: float, ev_cal: float, exp_clv: float, state: str = "ACTIVE") -> str:
    if state == "RETIRED":
        return "PASS"
    if p_edge >= 0.90 and ev_cal >= 0.03 and exp_clv >= 0:
        return "A+"
    if p_edge >= 0.80 and ev_cal >= 0.02 and exp_clv >= -0.005:
        return "A"
    if p_edge >= 0.70 and ev_cal >= 0.01:
        return "B"
    if p_edge >= 0.60 and ev_cal > 0:
        return "C"
    return "PASS"


def confidence(leg: dict, learner, calibrator=None, clv_model: CLVModel | None = None,
               segment: str | None = None) -> dict:
    """leg: card leg dict (sport, strategy, tier, market, price, optional model_prob / ev /
    consensus_prob / market_prob).  Returns p_cal, p_edge, exp_clv, ev_raw, ev_cal, score, grade, stake."""
    dec = float(american_to_decimal(leg["price"]))
    post = learner.segment_posterior(leg["strategy"], segment)
    state = post.get("state", "ACTIVE")
    mp = leg.get("model_prob")
    if mp is None and leg.get("ev") is not None:
        mp = (1 + float(leg["ev"])) / dec
    if mp is not None and not pd.isna(mp):
        p_raw = float(mp)
        p_cal = float(calibrator.transform([p_raw])[0]) if calibrator is not None else p_raw
    else:  # no model prob: fair price + posterior edge
        p_raw = p_cal = (1.0 + post["post_mean"]) / dec
    ev_raw, ev_cal = p_raw * dec - 1, p_cal * dec - 1
    exp_clv = clv_model.predict(leg["strategy"], leg["sport"], leg.get("market"),
                                leg.get("consensus_prob"), leg.get("market_prob")) if clv_model else 0.0
    g = grade_of(post["p_pos"], ev_cal, exp_clv, state)
    mult = learner.multiplier(leg["strategy"], segment)
    adj = float(np.clip(ev_cal / ev_raw, 0, 1.5)) if (mp is not None and ev_raw > 0) else 1.0
    stake = 0.0 if g == "PASS" else TIER_STAKE.get(leg.get("tier", "C"), 0.5) * mult * adj
    # single 0-100 score for sorting (not used for decisions)
    score = 100 * (0.5 * post["p_pos"] + 0.3 * float(np.clip(ev_cal / 0.06, -1, 1)) * 0.5 + 0.15
                   + 0.2 * float(np.clip(exp_clv / 0.02, -1, 1)) * 0.5)
    return dict(p_raw=p_raw, p_cal=p_cal, p_edge=post["p_pos"], post_mean=post["post_mean"], state=state,
                exp_clv=exp_clv, ev_raw=ev_raw, ev_cal=ev_cal, score=round(score, 1), grade=g,
                mult=mult, stake=round(stake, 3))
