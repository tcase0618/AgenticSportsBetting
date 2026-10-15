# audit/ — the auditor / self-improvement loop

The loop learns from signals that are informative: closing-line value, calibration, and a slowly updated Bayesian posterior of each strategy's edge. It does not react to individual wins or losses.

| module | role |
|---|---|
| `journal.py` | Append-only event log, `data/journal/events.csv`, with place, close and settle events. `load()` folds the log into one row per bet; `snapshot()` writes it to parquet. |
| `learner.py` | Normal-Normal edge posterior per strategy and strategy×segment, with prior -4.5% ± 3%. Produces P(edge>0), a fractional-Kelly multiplier (0 below P=0.6, capped at 1.5x), the ACTIVE / WATCH / RETIRED states with hysteresis, and `clv_prior()`. |
| `calibration.py` | Reliability table, Brier, ECE and calibration slope. `OnlineRecalibrator` refits Platt or isotonic on past settled bets only. |
| `confidence.py` | Pre-bet grade (A+/A/B/C/PASS) and recommended stake, from calibrated p, P(edge>0) and expected CLV (`CLVModel`). |
| `replay.py` | Walk-forward proof, regime-switch test and CLV study. `warm_learner()` builds tonight's learner state. Output is `results/audit_*.csv`; the summary is `results/audit_report.md`. |

## Nightly loop
```python
from audit import journal as J
from audit.replay import warm_learner, load_bets
from audit.calibration import OnlineRecalibrator
from audit.confidence import confidence, CLVModel

# 1. build the card (nightly/card.py), then grade and size each leg
lr  = warm_learner()                                     # ledger history + settled journal bets
cal = OnlineRecalibrator("platt")
P   = load_bets().query("strategy == 'NBA points: blend EV'")
cal.fit(P.model_prob, P.y)                               # add settled journal props when available
clv = CLVModel().fit(J.load().dropna(subset=["clv_prob"]))
for L in legs:
    seg = J.default_segment(L["sport"], L["market"], L["pick"], L["price"])
    L.update(confidence(L, lr, cal, clv, segment=seg))   # adds grade, stake, p_cal, p_edge, exp_clv

# 2. place bets -> journal (opp_price, if known, gives an exact de-vig)
J.record(J.journal_from_card_legs(legs, placed_at=now))

# 3. at lock / tip-off -> closing line (CLV is known before the result)
J.attach_close(bet_id, closing_price=-118, closing_line=3.0, closing_opp_price=-102)

# 4. next morning -> settle
J.settle(bet_id, result=+1)          # +1 win, 0 push, -1 loss

# 5. the next card's warm_learner() / cal.fit() / CLVModel().fit() pick these up automatically
```
`clv_prob` is the fair probability of the exact bet at the close, minus the de-vigged probability when it was placed. A line difference counts too: it is converted at roughly phi(0)/sigma per point, using `journal.LINE_SIGMA`.

## Rules the loop enforces
* The learner sees **unit** P&L, so stake decisions never feed back into the edge estimate. Zero-stake and RETIRED bets are still journaled as paper bets, so a strategy can earn its way back.
* The calibrator and the CLV model only use bets settled before the date being priced.
* State changes need evidence from the fast detector, so the skeptical prior cannot retire a strategy on its own.
* Do not judge news-fade strategies, such as NBA Load-Management Overs, on CLV, because their trigger is the line move. See `results/audit_report.md`.
