# Super Engine: What It Is, What Won, What Didn't

One command runs the whole daily loop: `python -m engine.super card | close | settle | status`
(automated by `.github/workflows/nightly-card.yml`; needs an `ODDS_API_KEY` repo secret).

## The layers, and their out-of-sample verdicts

| Layer | What it does | Holdout result | In production? |
|---|---|---|---|
| 7 strategies + NFL bounce-back dogs | Rule-based situational edges per sport | +3% to +10% ROI per strategy (see KEYS.md) | **Yes**, core |
| Parlays | 2-leg round robins of Tier A/B legs, different games, one book | A-tier RR2 **+15.2%** on holdout (791 tickets) | **Yes** |
| NBA points props | 25% projection model / 75% leave-one-book-out market consensus, EV ≥ 4% | **+3.9%** on 3,059 bets in 2025-26 (discovery +2.8%) | **Yes** |
| Full nightly card (all of the above) | Every night Oct 2024 – Jun 2026 | **+162u** (1u = 1% bankroll), 15/20 winning months, max DD 27u | **Yes** |
| NBA game-script engine | Pace, ratings, positional defense, blowout-minutes → script-conditional projections | Props +2.6% vs +3.9% baseline; totals model loses to the close | No (research) |
| NFL game-script engine | Play-by-play EPA, pace, PROE → team points, plays, attempts; script-fit props | Every holdout threshold loses to the close | No (research) |
| DFS pick'em vs sportsbooks | PrizePicks/Underdog legs where books disagree | Legs hit 43–47% (n small) | No |
| Adaptive (Bayesian) staking | Stake ∝ posterior edge | +112u to +209u vs **+223u static** | **Kill switch only** |
| Pre-bet grades | Calibrated p + P(edge>0) + expected CLV | A+/A +5.4% vs +3% for others (not significant) | Displayed, not used to size |
| CLV tracking | Line movement after the bet, known before the result | +9.6% ROI when CLV>0 vs −2.9% when CLV<0 (p=1e-5) | **Yes**, the main learning signal |

## The honest lessons

1. **Closing lines are very efficient.** Two full tendency engines (EPA, pace, positional defense) were *less* accurate than the
   closing number. Edges live in **situations the market misprices** (age, wind, rest, overreaction), not in out-modeling it.
2. **Parlays multiply edges. They don't create them.** Parlaying +7% legs → +15%. Parlaying ~+1% prop legs → ~+2%. Parlaying
   plus-money alt-line longshots → −12%.
3. **Learning from wins and losses is mostly learning noise.** The engine learns from CLV first (known within hours), calibration second,
   and realized P&L last and slowly. A losing bet that beat the closing line counts as a good bet.
4. **Self-improvement that's safe:** the auditor can *retire* a strategy whose edge disappears (a −15% regime is caught within
   ~43–250 bets for stake → 0), but it can't overfit by inflating stakes on a hot streak.

## "Knows before it's right"
Every live bet is journaled at placement. At ~6:30 PM ET the engine records the closing line → **CLV**, which predicts long-run
results long before the bets settle. As the journal grows, `CLVModel` learns which strategies and segments reliably beat the
close, and the pre-bet grade sharpens. That's the realistic version of knowing before the result: not certainty on a single bet,
but a calibrated, verified read on whether the process is sharp.
