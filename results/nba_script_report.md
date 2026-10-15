# NBA game-script engine: results

Code: `engine/nba_script/` (tendencies → script → player_script → evaluate). Discovery is 2025 and holdout is 2026. Everything is walk-forward, and every weight and threshold was picked on discovery only. Full tables are in `logs/nba_script_eval.txt`.

**Bottom line: the game script does not reliably beat the current props blend.** It makes the projection slightly more accurate (holdout MAE 4.563 → 4.549), but the market already prices that information. On holdout, swapping the script into the blend lowers ROI.

## 1. Game-script model (walk-forward linear models, 2012-2026, n≈18k games)
| MAE (game-weighted) | totals, 2012-18 | totals, 2019-26 | margin, 2012-18 | margin, 2019-26 |
|---|---|---|---|---|
| Closing line | **13.66** | **14.33** | **9.34** | **10.44** |
| Tendencies only (no line) | 14.09 | 15.00 | 9.67 | 10.85 |
| Tendencies + close | 13.72 | 14.34 | 9.36 | 10.45 |

- **Information beyond the close:** the residual correlations corr(model − close, actual − close) are about zero. For totals they are 0.016 / −0.009 with no line and 0.039 / 0.011 with the close; for margins they are 0.018 / 0.018.
- **Totals betting at −110:** the threshold was chosen on 2012-18 with n≥300 (it picked 4 pts, ROI 0.0%). On 2019+ it fires only 13 times, for −27%. With no line used, every threshold loses about 4-5% in both periods.
- **P(blowout ≥15):** Brier 0.213 vs 0.217 for the climatology baseline (2019-26). The signal is small but real.
- **Pace MAE:** 3.8 possessions.

## 2. Blowout minutes (quantified on 2012-16 only)
Ratio of actual to rolling average, close game vs blowout:
- **Starters at 34+ min:** minutes 1.00 → 0.875, points 1.01 → 0.88.
- **Starters at 28-34 min:** minutes 1.03 → 0.91.
- **Bench under 18 min:** points 1.14 → 1.27.

The opponent positional-defence exponent fits to only k = 0.09, so it is a weak effect.

## 3. Props: log-loss on main lines and ROI via `pick_bets`
| Probability | LL 2025 | LL 2026 | ROI@EV≥.04, 2025 | ROI@EV≥.04, 2026 | Main-line ROI@.04, 2025 / 2026 |
|---|---|---|---|---|---|
| p_mkt | .6916 | .6919 | +4.2% (486) | 0.0% (801) | +2.6 / −0.3% |
| p_mod (baseline model) | .7070 | .7072 | −2.4% | −1.1% | −2.6 / −0.6% |
| p_script | .7069 | .7069 | −1.7% | −1.1% | −1.7 / −0.4% |
| p_script_nm (no lines) | .7072 | .7064 | −2.0% | −1.2% | −2.0 / −0.5% |
| **Baseline blend .75 mkt + .25 mod** | .6919 | .6916 | **+2.8% (1840)** | **+3.9% (3059)** | +1.5 / +2.4% |
| .75 mkt + .25 script | .6917 | .6915 | +2.9% (1889) | +2.6% (3051) | +2.7 / +2.0% |
| .75 mkt + .25 script_nm | .6918 | .6915 | +2.5% | +1.5% | +2.0 / +2.1% |
| Chosen on 2025 log-loss (.9 mkt + .1 script) | .6914 | .6914 | +5.1% (591) | +3.4% (960) | +5.0 / −1.3% |
| Chosen on 2025 ROI (.9 mkt + .1 mod, no script) | .6914 | .6915 | +6.5% (599) | +4.7% (956) | +6.7 / 0.0% |

- **Thresholds 0.02-0.08:** the script blend is below the baseline blend on holdout at every threshold: +1.0 / 2.6 / 4.7 / 3.8% vs +1.7 / 3.9 / 4.8 / 4.5%.
- **Over/under split (baseline blend, 2026):** overs +0.3%, unders +8.1%. With the script blend the unders drop to +5.1%. The edge is in the unders in both seasons.

## 4. Script-fit filter on the baseline blend's EV≥.04 bets
| Filter | 2025 fit / not fit | 2026 fit / not fit |
|---|---|---|
| Rule: over only if low blowout risk or high pace; under only if a starter in a likely blowout | **+8.8% (680)** / −0.8% | +2.2% (1175) / **+5.0%** |
| Script projection agrees with the bet's side | +2.2% / +3.2% | +7.3% (1261) / +1.5% |

The rule looked strong on discovery and **reversed on holdout**. The agreement filter failed discovery, so it would not have been adopted. Its holdout gain should be treated as noise until 2027 confirms it.

## Caveats
- **Line timing:** the script's market input is the opening line, falling back to the close. `nba.csv` has no openers for 2023-25 or for 19% of 2026, so 2025 uses closing lines, which come after the prop snapshots taken about 12h before tip. The no-line variant (`_nm`) behaves the same, so this does not drive the results.
- **Sample size:** at n≈1-3k bets, an ROI difference of ±2% is within one standard error.
