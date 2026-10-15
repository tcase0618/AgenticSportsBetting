# Auditor replay: what it showed

To reproduce, run `python -m audit.replay`. The tables are in `results/audit_*.csv`. The replay walks forward with no lookahead: stakes on date *d* use only bets settled before *d*. It covers 17,647 bets from 1999 to 2026: the 8 ledger strategies plus NBA props. The **holdout** is 10,162 bets from seasons that strategy discovery never saw.

## 1. Adaptive staking did not beat static out of sample

| Holdout policy | staked | units | ROI | max DD | daily Sharpe | t vs static (units) |
|---|---|---|---|---|---|---|
| static tiers (A 1 / B .75 / C .5 / prop .5) | 5,845 | **+223.3** | 3.82% | 22.4 | **0.095** | – |
| learner, spec prior (-4.5% ± 3%) | 2,540 | +105.7 | 4.16% | 20.3 | 0.063 | -3.4 |
| learner + props recalibration | 2,534 | +111.6 | 4.41% | 20.3 | 0.064 | -3.1 |
| same, cold start (holdout data only) | 633 | +34.6 | 5.47% | 13.3 | 0.040 | -4.0 |
| learner + cal, "validated" prior (0% ± 3%) | 4,487 | +208.6 | **4.65%** | 24.1 | 0.079 | -0.5 |
| props recalibration only | 5,421 | +213.7 | 3.94% | 20.7 | 0.091 | -0.5 |

* The adaptive policies earn **0.3–1.6 pp more ROI per unit staked**, which is within noise. They lose on total units and on Sharpe because they bet much less. The skeptical prior cannot confirm small edges that turned out to be real. Every one of the 9 strategies was profitable in its holdout, so any de-risking cost money: Blowout Believers fell from +35u to +6.5u, Series Carryover from +27u to +7u, and props from +60u to +29u. Forgetting faster made it worse (half-life 500 bets: +41.6u).
* Disclosure: the first version of the state machine retired every strategy after 30 bets. That was the prior alone, since P(edge>0) starts at 0.07, with no evidence behind it; it staked +8.5u in total. Now a state change needs evidence from a fast detector with a near-neutral prior.
* Pre-bet grades in the holdout: A+ and A (1,938 bets) returned +5.4%, against +2.8 to 3.2% for B, C and PASS. The direction is right, but the gap is not statistically significant. The engine does **not** yet "know before it's right" for individual bets.

## 2. Where the learner earns its keep: detecting a dead strategy

The test switches each strategy to a bootstrapped negative regime that oversamples its losing bets, starting at its holdout. Figures are medians over 20 seeds, using the learner that was actually staking at the switch.
* At **-15% ROI**, stake is cut to zero after **43–250 bets** (Blowout Believers was already near zero) and the strategy is retired after **158–505 bets**. Over the first 300 bets after the switch, losses fall from 21–48u (static) to 2–36u.
* At **-4.5%** (dead, just paying the vig), stake reaches zero after 82–532 bets (Blowout Believers was again already near zero), and only 35–90% of runs retire within 1,000 bets.
* In calendar time, that is days for props, about 1 year for NHL, and **7–10 years for the NFL keys** at 16–26 bets per year. P&L alone cannot police low-volume strategies.

## 3. Calibration (NBA props)
The raw side probability is overconfident on placed bets, as the winner's curse predicts. Mean p was 0.526 against a 0.503 hit rate in 2025, and 0.464 against 0.447 in 2026; calibration slope was 0.77 and 0.95. Past-only Platt recalibration fixes the mean (0.454 in 2026), but Brier is unchanged (0.2407 to 0.2410) and isotonic is worse (0.2464). The model does discriminate: its Brier is 0.2407 against 0.2472 for the base rate.

## 4. Closing-line value (open → close, where both exist)
* **CLV is real in this data.** Across 63k game-sides, betting at the open the side the line later moved toward returned +8.0% (NBA totals), +3.6% (NBA ATS), +4.6% (NCAAF ATS), +6.6% (NCAAF totals), +2.3% (NHL ML) and +1.2% (MLB ML). The other side returned -9% to -17%.
* On the strategies' own bets placed at the open (2,087 bets), corr(CLV, P&L) is 0.095 (p=1e-5). ROI is **+9.6% when CLV > 0 and -2.9% when CLV < 0**.
* By strategy:
  * NHL Blowout Bounce-Back: +0.46 pp CLV (t=5.4).
  * NCAAF Rested Dogs and NCAAM Blowout Believers: about 0.
  * NBA Load-Management Overs: -6.7 pp on 100% of bets. The trigger is the rest news that drops the total, so the strategy fades the move. It returns +10.6% at the close and -4.0% at the open. **Do not use CLV to judge news-fade strategies; bet them at the close.**
* Across 30 strategy×season groups, corr(mean CLV, ROI at open) is 0.22 (p=0.24), which is too few groups to validate. The CLV prior never triggered. All ledger bets are priced at the close, so their CLV is 0 by construction. Real CLV only accrues once the journal records bets when they are placed.

**Recommendation:** keep the static tiers for sizing. Use the learner as a **kill switch** (WATCH and RETIRED cut stake), not as an upsizer. Journal every bet with its closing line, and treat CLV as the fast early warning for bets placed early (props, NHL). Revisit Bayesian sizing after about a season of journaled CLV.
