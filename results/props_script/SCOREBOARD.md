# Script-Driven Props + Parlays — Hit-Rate Scoreboard (2021+ out-of-sample, fit on ≤2020)

## Milestone ("X+") legs: predicted → realized hit, vs base rate
| Sport | ≥80% tier n | hit | base rate | ≥90% tier hit |
|---|---|---|---|---|
| NBA (pts/reb/ast/3PM) | 198k | 89.9% | 66.0% | 94.9% |
| NFL (yds/rec) | 15.2k | 87.8% | 54.6% | 92.9% |
| NHL (SOG/pts) | 123k | 87.8% | 74.4% | 93.3% |
| MLB (Ks) | 26.7k | 88.2% | 74.9% | 93.7% |
| UFC (distance/method/rounds) | 636 | 83.6% | 65.0% | (n=30) |

## Daily parlay hit rate (one ticket per day), predicted → realized
| Sport | Construction | 3-leg | 5-leg | 7-leg |
|---|---|---|---|---|
| NBA | Top-N cross-game | .965→.969 | .930→.925 | .884→.890 |
| NBA | 85%+ ML + props | .873→.885 | .852→.860 | .828→.850 |
| NBA | Same-game script stack | .954→.943 | .883→.864 | .758→.742 |
| NHL | Top-N | .928→.922 | .860→.858 | .784→.784 |
| NFL | Top-N | .833→.831 | .681→.680 | .526→.535 |
| NFL | 85%+ ML + props | .819→.851 | .739→.776 | .655→.672 |
| MLB | Top-N | .864→.867 | .742→.748 | .613→.617 |
| UFC | Top-N (per event) | .552→.587 | .315→.348 | .161→.166 |
| NBA | Top-N on REAL posted lines | .240→.133 | .086→.028 | .030→.002 |

Typical price of a top-N milestone ticket: NBA 3/5/7 ≈ -2156/-1274/-896; NFL ≈ -1454/-470/-196.

## Real posted lines (single props)
NBA points 35k lines: 50.7–53.2% overall; ≥.60 tier 50–60% by season; market log loss ties/beats model.
NFL/NHL/MLB real-line samples small; none beats the market-favoured side reliably.

## What tendencies mattered
NBA: teammates-out (usage redistribution) + recent form; NFL: recent form, snap share, teammates-out, QB change
(+0.70 sd passing yds); NHL: TOI/line/PP role; MLB: pitcher leash (pitch-count/BF). Home/away & rest ≈ 0
once the game script is in. Calibrator: error ≈.02–.04 → ≈.002–.009 after first season.
