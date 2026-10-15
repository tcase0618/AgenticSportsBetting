# NFL game-script engine: results

Code: `engine/nfl_script/{tendencies,script,props_script}.py`. Data: nflverse pbp 2001-2026 (week 5). No leakage: every tendency uses only earlier games (rolling L4/L8 and season-to-date, shrunk toward the prior season's league and team values). Models are fit walk-forward by season. The bet threshold k was chosen on 2006-15 only, and ridge alpha=300 was checked on 2006-15 only.

## Game script (team points, plays, pass att, rush att; ridge on matchup tendencies + context)
| | 2006-15 (discovery) | 2016-25 (holdout) | 2026 wk1-5 |
|---|---|---|---|
| Team-points MAE: model / close-implied | 7.79 / 7.63 | 7.46 / 7.26 | 7.70 / 7.31 |
| Margin MAE: model / close | 10.97 / 10.71 | 10.14 / 9.78 | 10.19 / 9.78 |
| Total MAE: model / close | 10.84 / 10.62 | 10.71 / 10.44 | 11.68 / 10.63 |
| Info beyond close, margin (slope, t) | 0.10, t=1.3 | 0.07, t=1.0 | - |
| Info beyond close, total (slope, t) | 0.27, **t=3.5** | 0.005, t=0.06 | - |

Volume MAE per team-game (2006-25): plays 6.6, pass att 6.3, rush att 5.7. Adding the closing line as an input moves these by less than 0.05. The closing line is the better forecaster on every target.

**Betting at the actual closing odds** (bet when |model - close| >= k; k chosen on discovery from 1 to 6):
- Sides, k=6: discovery n=224, ROI +7.7%, z=1.21. **Holdout n=211, ROI -5.7%, z=-0.85.** 2026: n=11, -3.1%.
- Totals, k=6: discovery n=200, ROI +10.1%, z=1.47. **Holdout n=115, ROI -0.3%, z=-0.04.**
- Every k from 1 to 6 is negative on the sides holdout (-0.1% to -8.6%), and every k is negative on the totals holdout (-0.3% to -5.4%). The 2006-15 totals signal (t=3.5) did not survive into 2016-25: the market absorbed it, or it was period-specific.
- Cross-check against the existing wind>=15 under key: 90% of the model's holdout total bets in wind>=15 games were unders, but ROI was -7.4% (n=60). The model does not add to that key.

**Verdict: the game script does not beat the closing side or total. Use it as a context and volume engine, not as a side or total signal.**

## Player props (share x script volume x opponent-adjusted efficiency; empirical distribution calibrated on prior seasons)
One bet per player-game-market, at the consensus line, priced at the median book price (decimal) and the de-vigged median-book p_over. DFS rows are excluded from betting because they carry no real prices. QB markets are conditional on the listed starting QB.

| Set | n (unique props) | Brier model vs market | Model-only EV>=5% | Model-only EV>=10% |
|---|---|---|---|---|
| 2017-18 completions (1 book) | 953 | .2556 vs .2504 | n=615, +3.9%, z=1.0 | n=469, +1.7%, z=0.4 |
| 2026 wk1-5, all 4 markets | 1,918 | worse in every market | n=1229, -1.3%, z=-0.5 | n=947, -1.9%, z=-0.6 |

The market's Brier score beats the model's in every market and season.

Blend p = w*model + (1-w)*market, with w and the EV threshold chosen on the calibration window:
- **Completions** (chosen on 2017: w=0.25, EV>=2%; tested on 2018): n=73, ROI +6.1%, z=0.54. Unders went 63 bets at +14.3% and overs 10 bets at -45%.
- **2026** (chosen on weeks 1-2, tested on weeks 3-5): no setting was profitable in weeks 1-2 (best was -7.7%), so the rule says **no bets**. For transparency, the best setting applied to weeks 3-5 gives n=802, +1.4%, z=0.43. Receptions unders: n=218, +9.9%, z=1.55. Rushing yards: +10-12%, z<=1.3. Receiving yards: negative. Passing yards (FanDuel only, n=92): strongly negative.

**2026 is a very small sample (5 weeks, one season, z<2 everywhere)**, so none of these splits is evidence. The only recurring pattern is model-driven *unders* on receptions and completions: positive in 2018, in 2026 receptions, and in the completions blend test. It is not significant and needs more seasons before it is used.

## Bottom line
- Nothing here beats the market out of sample with significance.
- The engine is useful as an input for volume (plays, pass and rush attempts per team) and for building projections that fit the game script.
- The "receptions/completions under" pattern is worth tracking forward in paper trading only.

Files: `results/nfl_script_games.csv`, `results/nfl_script_props.csv`.
