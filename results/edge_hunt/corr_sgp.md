# Edge hunt: corr_sgp -- product-priced same-game parlays (Courtside hypothesis)

Protocol: engine/edge_hunt/PROTOCOL.md. Code: engine/edge_hunt/corr_sgp.py (`build(slate)` for live use). Every knob: corr_sgp_candidates.csv (n_tests = 120). Discovery = first 60% of dates per dataset (NBA <= 2025-04-05, NFL-2026 <= 2026-09-28, NFL-2017/18 <= 2018-09-27); lockbox = rest, evaluated ONCE for the single discovery-chosen config (max discovery ROI, n >= 200 NBA / 25 NFL) per family. Significance: one-sided p < 0.05/19 or BH q < 0.10 across the 19 non-control finals, using the LARGER of the date-clustered and game-clustered p. TRUE also needs disc ROI > 0, lockbox n >= 50, >= 60% positive blocks (months NBA, weeks NFL) over the full sample.

**Pricing assumption.** Courtside's own SGP prices are unknown; each leg is priced at its real historical market price and the parlay pays the product of decimals. NBA spread/total juice is real for 2021-22 (ESPN consensus) and 2025-26 (DraftKings close) but MISSING for 2024-25 -> -110 assumed there (`flag` = share of bets with an assumed -110 leg). NFL game juice is real (nflverse). Pushed legs drop out; DNP voids dropped.

## Verdicts

| candidate (discovery-chosen config) | disc n / ROI | lockbox n / ROI / z / p (week/month-cluster p) | joint vs naive lift (full; joint / prod realized marginals) | max per-leg vig before edge dies (full / lockbox) | blocks + | BH q | -110 flag | verdict |
|---|---|---|---|---|---|---|---|---|
| NBA nba_ptsO_gameO `scope=line20 model=none` | 3369 / +0.013 | 2628 / -0.028 / -0.69 / 0.764 (0.690) | 1.153 (joint 0.277 vs 0.240; implied 0.278) | 0.00 / 0.00 pts | 10/18 | 0.925 | 56% | FALSE |
| NBA nba_ptsO_cover `scope=top1 model=agree` | 476 / +0.041 | 592 / -0.059 / -0.91 / 0.828 (0.928) | 1.142 (joint 0.273 vs 0.239; implied 0.278) | 0.00 / 0.00 pts | 8/14 | 0.925 | 58% | FALSE |
| NBA nba_ptsO_ML `scope=top1 model=agree` | 476 / +0.003 | 592 / -0.061 / -0.79 / 0.789 (0.779) | 1.114 (joint 0.270 vs 0.242; implied 0.275) | 0.00 / 0.00 pts | 5/14 | 0.925 | 0% | FALSE |
| NBA nba_ptsU_gameU `scope=line25 model=agree` | 688 / +0.094 | 727 / +0.125 / +1.92 / 0.029 (0.097) | 1.170 (joint 0.310 vs 0.265; implied 0.279) | 2.83 / 3.19 pts | 11/14 | 0.181 | 61% | FALSE |
| NBA nba_ptsU_oppcover `scope=line25 model=agree` | 688 / +0.040 | 727 / -0.073 / -1.34 / 0.910 (0.926) | 1.060 (joint 0.276 vs 0.260; implied 0.279) | 0.00 / 0.00 pts | 5/14 | 0.960 | 61% | FALSE |
| NBA nba_ptsO_cover_gameO `scope=top1 model=agree` | 476 / +0.115 | 592 / +0.018 / +0.19 / 0.427 (0.397) | 1.213 (joint 0.154 vs 0.127; implied 0.145) | 1.06 / 0.32 pts | 6/14 | 0.675 | 58% | FALSE |
| NBA ctrl_nba_ptsO_gameU `scope=all model=agree` | 2944 / -0.218 | 5334 / -0.161 / -5.03 / 1.000 (0.998) | 0.907 (joint 0.226 vs 0.250; implied 0.277) | 0.00 / 0.00 pts | 1/14 | - | 50% | control |
| NBA ctrl_nba_ptsO_oppcover `scope=all model=agree` | 2944 / -0.177 | 5334 / -0.172 / -8.41 / 1.000 (1.000) | 0.896 (joint 0.228 vs 0.255; implied 0.277) | 0.00 / 0.00 pts | 0/14 | - | 50% | control |
| NBA nba_2topO_gameO `scope=top1 model=none` | 1713 / +0.060 | 1222 / -0.000 / -0.00 / 0.502 (0.501) | 1.296 (joint 0.153 vs 0.118; implied 0.148) | 0.61 / 0.00 pts | 10/18 | 0.733 | 47% | FALSE |
| NBA nba_2topU_gameU `scope=top1 model=agree` | 575 / +0.068 | 574 / +0.222 / +2.11 / 0.021 (0.045) | 1.322 (joint 0.170 vs 0.129; implied 0.149) | 2.45 / 3.65 pts | 10/14 | 0.181 | 61% | FALSE |
| NFL nfl_QBpyO_RECydsO `scope=top1 model=none` | 31 / +0.952 | 64 / +0.222 / +2.55 / 0.178 (0.101) | 1.378 (joint 0.411 vs 0.298; implied 0.281) | 11.05 / 5.58 pts | 3/3 | 0.392 | 0% | FALSE |
| NFL nfl_QBpyU_RECydsU `scope=top1 model=none` | 30 / -0.053 | 62 / +0.203 / +1.11 / 0.186 (0.185) | 1.526 (joint 0.315 vs 0.207; implied 0.282) | 3.09 / 5.14 pts | 2/3 | 0.392 | 0% | FALSE |
| NFL nfl_QBpyO_RECrecO `scope=top1 model=none` | 29 / +0.731 | 61 / +0.091 / +1.50 / 0.367 (0.078) | 1.437 (joint 0.367 vs 0.255; implied 0.280) | 7.36 / 2.37 pts | 3/3 | 0.634 | 0% | FALSE |
| NFL nfl_QBpyO_RECydsO_gameO `scope=top1 model=none` | 31 / +0.942 | 64 / +0.392 / +1.83 / 0.151 (0.194) | 1.539 (joint 0.232 vs 0.150; implied 0.147) | 8.59 / 6.14 pts | 3/3 | 0.392 | 0% | FALSE |
| NFL nfl_RBrushO_cover `scope=top1 model=none` | 92 / +0.041 | 63 / +0.077 / +0.63 / 0.320 (0.340) | 1.282 (joint 0.294 vs 0.229; implied 0.278) | 1.51 / 2.12 pts | 3/5 | 0.607 | 0% | FALSE |
| NFL nfl_RBrushO_ML `scope=all model=none` | 163 / +0.055 | 125 / +0.273 / +1.41 / 0.115 (0.149) | 1.338 (joint 0.299 vs 0.224; implied 0.277) | 3.36 / 5.77 pts | 4/5 | 0.392 | 0% | FALSE |
| NFL nfl_RBrushU_oppcover `scope=all model=none` | 163 / +0.167 | 125 / +0.310 / +4.90 / 0.001 (0.020) | 1.252 (joint 0.351 vs 0.280; implied 0.278) | 5.94 / 8.01 pts | 5/5 | 0.026 | 0% | **TRUE*** (conditional) |
| NFL nfl_RECydsO_gameO `scope=top1 model=none` | 93 / +0.126 | 63 / -0.079 / -0.50 / 0.684 (0.794) | 1.125 (joint 0.288 vs 0.256; implied 0.278) | 1.13 / 0.00 pts | 3/5 | 0.925 | 0% | FALSE |
| NFL nfl_QBpyO_gameO `scope=top1 model=none` | 30 / +0.422 | 62 / +0.291 / +2.47 / 0.156 (0.075) | 1.207 (joint 0.370 vs 0.306; implied 0.278) | 8.16 / 7.15 pts | 3/3 | 0.392 | 0% | FALSE |
| NFL nfl_QBcmpO_gameO `scope=all model=none` | 555 / -0.071 | 368 / -0.259 / -2.90 / 0.998 (0.998) | 1.101 (joint 0.235 vs 0.214; implied 0.279) | 0.00 / 0.00 pts | 10/34 | 0.998 | 0% | FALSE |
| NFL nfl_RBrushO_cover_gameO `scope=all model=agree` | 29 / +0.423 | 24 / +1.032 / +1.85 / 0.069 (0.032) | 1.724 (joint 0.260 vs 0.151; implied 0.145) | 10.13 / 13.95 pts | 4/5 | 0.328 | 0% | FALSE |
| NFL ctrl_nfl_RBrushO_oppcover `scope=top1 model=none` | 92 / -0.274 | 63 / -0.476 / -9.47 / 1.000 (0.979) | 0.722 (joint 0.168 vs 0.232; implied 0.278) | 0.00 / 0.00 pts | 0/5 | - | 0% | control |
| NFL ctrl_nfl_RECydsO_gameU `scope=all model=agree` | 77 / +0.074 | 95 / -0.128 / -1.58 / 0.917 (0.960) | 0.919 (joint 0.267 vs 0.291; implied 0.278) | 0.00 / 0.00 pts | 1/5 | - | 0% | control |

## Break-even table (full sample of each final; ROI at product pricing with x extra implied-prob points of vig per leg; haircut = parlay-multiplier cut that zeroes ROI)

| candidate | legs | n | ROI +0 | +1 pt | +2 pt | +3 pt | +4 pt | break-even vig/leg | break-even haircut | latent rho (sample) | long-run rho (n) | EV at long-run rho | break-even vig/leg at long-run rho | break-even haircut at long-run rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NBA nba_ptsO_gameO | 2 | 5997 | -0.005 | -0.042 | -0.076 | -0.109 | -0.140 | 0.00 pts | 0.0% | +0.23 | +0.17 (17856) | -0.005 | 0.00 pts | 0.0% |
| NBA nba_ptsO_cover | 2 | 1068 | -0.014 | -0.050 | -0.085 | -0.117 | -0.148 | 0.00 pts | 0.0% | +0.21 | +0.13 (17758) | -0.026 | 0.00 pts | 0.0% |
| NBA nba_ptsO_ML | 2 | 1068 | -0.032 | -0.073 | -0.111 | -0.146 | -0.179 | 0.00 pts | 0.0% | +0.17 | +0.15 (18009) | -0.008 | 0.00 pts | 0.0% |
| NBA nba_ptsU_gameU | 2 | 1415 | +0.110 | +0.069 | +0.030 | -0.006 | -0.041 | 2.83 pts | 9.9% | +0.28 | +0.17 (17856) | -0.007 | 0.00 pts | 0.0% |
| NBA nba_ptsU_oppcover | 2 | 1415 | -0.018 | -0.054 | -0.088 | -0.120 | -0.151 | 0.00 pts | 0.0% | +0.10 | +0.13 (17758) | -0.027 | 0.00 pts | 0.0% |
| NBA nba_ptsO_cover_gameO | 3 | 1068 | +0.061 | +0.003 | -0.051 | -0.101 | -0.147 | 1.06 pts | 5.8% | - | - (-) | - | - pts | - |
| NBA ctrl_nba_ptsO_gameU | 2 | 8278 | -0.182 | -0.212 | -0.240 | -0.267 | -0.293 | 0.00 pts | 0.0% | -0.14 | -0.17 (17856) | -0.202 | 0.00 pts | 0.0% |
| NBA ctrl_nba_ptsO_oppcover | 2 | 8278 | -0.174 | -0.204 | -0.233 | -0.260 | -0.286 | 0.00 pts | 0.0% | -0.17 | -0.13 (17758) | -0.182 | 0.00 pts | 0.0% |
| NBA nba_2topO_gameO | 3 | 2935 | +0.035 | -0.022 | -0.074 | -0.122 | -0.168 | 0.61 pts | 3.4% | - | - (-) | - | - pts | - |
| NBA nba_2topU_gameU | 3 | 1149 | +0.145 | +0.083 | +0.025 | -0.029 | -0.080 | 2.45 pts | 12.7% | - | - (-) | - | - pts | - |
| NFL nfl_QBpyO_RECydsO | 2 | 95 | +0.460 | +0.407 | +0.356 | +0.308 | +0.262 | 11.05 pts | 31.5% | +0.67 | +0.17 (453) | -0.016 | 0.00 pts | 0.0% |
| NFL nfl_QBpyU_RECydsU | 2 | 92 | +0.120 | +0.079 | +0.040 | +0.003 | -0.032 | 3.09 pts | 10.7% | +0.65 | +0.17 (453) | -0.016 | 0.00 pts | 0.0% |
| NFL nfl_QBpyO_RECrecO | 2 | 90 | +0.297 | +0.250 | +0.205 | +0.162 | +0.121 | 7.36 pts | 22.9% | +0.68 | +0.23 (256) | +0.011 | 0.30 pts | 1.1% |
| NFL nfl_QBpyO_RECydsO_gameO | 3 | 95 | +0.572 | +0.486 | +0.406 | +0.332 | +0.263 | 8.59 pts | 36.4% | - | - (-) | - | - pts | - |
| NFL nfl_RBrushO_cover | 2 | 155 | +0.056 | +0.018 | -0.017 | -0.051 | -0.082 | 1.51 pts | 5.3% | +0.40 | +0.23 (1943) | +0.030 | 0.80 pts | 3.0% |
| NFL nfl_RBrushO_ML | 2 | 288 | +0.149 | +0.101 | +0.057 | +0.015 | -0.025 | 3.36 pts | 13.0% | +0.46 | +0.22 (1991) | +0.035 | 0.84 pts | 3.4% |
| NFL nfl_RBrushU_oppcover | 2 | 288 | +0.229 | +0.185 | +0.144 | +0.104 | +0.067 | 5.94 pts | 18.7% | +0.43 | +0.23 (1943) | +0.031 | 0.80 pts | 3.0% |
| NFL nfl_RECydsO_gameO | 2 | 156 | +0.044 | +0.005 | -0.032 | -0.066 | -0.099 | 1.13 pts | 4.2% | +0.20 | +0.11 (5259) | -0.037 | 0.00 pts | 0.0% |
| NFL nfl_QBpyO_gameO | 2 | 92 | +0.334 | +0.285 | +0.238 | +0.194 | +0.152 | 8.16 pts | 25.0% | +0.40 | +0.26 (479) | +0.049 | 1.27 pts | 4.6% |
| NFL nfl_QBcmpO_gameO | 2 | 923 | -0.146 | -0.177 | -0.206 | -0.234 | -0.260 | 0.00 pts | 0.0% | +0.14 | - (-) | - | - pts | - |
| NFL nfl_RBrushO_cover_gameO | 3 | 53 | +0.699 | +0.605 | +0.518 | +0.437 | +0.362 | 10.13 pts | 41.1% | - | - (-) | - | - pts | - |
| NFL ctrl_nfl_RBrushO_oppcover | 2 | 155 | -0.356 | -0.378 | -0.399 | -0.419 | -0.438 | 0.00 pts | 0.0% | -0.40 | -0.23 (1943) | -0.232 | 0.00 pts | 0.0% |
| NFL ctrl_nfl_RECydsO_gameU | 2 | 172 | -0.038 | -0.073 | -0.107 | -0.138 | -0.169 | 0.00 pts | 0.0% | -0.15 | -0.11 (5259) | -0.165 | 0.00 pts | 0.0% |

## Data actually available (same-game pairs with REAL leg prices)
- **NBA points** (one main line per player-game; 2021-22 BettingPros consensus, 2024-25 and 2025-26 Odds-API books):
  31.1k player-games, ~8.3k usable pairs per game-leg family. Game total/spread juice real in 2021-22 and 2025-26,
  assumed -110 in 2024-25 (flag column). ML always real.
- **NFL 2026 weeks 1-6 only** (~79 graded games): receiving/rushing/receptions mains, QB passing yards weeks 3-6
  only (FanDuel main + DraftKings alt ladder overs). QB+WR1 pairs: ~95. NFL 2017-18 has only QB completions
  (no receiver props), so QB+WR stacks cannot be tested before 2026; completions were paired with the game total.
- NHL/MLB 2026 prop files (1.7k / 3.1k rows, a few weeks) were not tested: too few same-game pairs with a graded,
  priced game leg to clear any multiple-testing bar.

## Knobs (all logged in corr_sgp_candidates.csv)
NBA: scope in {all, top1 (highest posted line on the team), line>=20, line>=25} x model in {none, agree (walk-forward
calibrated p_over from results/props_script/nba_legs.parquet on the leg's side of the de-vigged price)}; 3-leg stacks:
both teams' top scorer O + game O, both U + game U. NFL: scope in {all, top1 (WR1/RB1 by posted line)} x model in
{none, agree (nfl_legs p_over_cal)}. All selectors use pre-game information only.

## How to read the two break-even columns
*Sample* columns use realized outcomes (includes luck in the individual legs). *Long-run rho* columns assume the
de-vigged leg prices are fair and the true latent correlation equals the one learned in results/scripts/sgp_state.json
over thousands of games; that is the expected edge a product-pricing book concedes. Rule of thumb: with two -110 legs
the product price already carries ~9.8% parlay overround, so the joint must beat independence by a lift > 1.098.
NBA points vs game total/side (rho 0.13-0.17) produces a lift of only ~1.06-1.09 at fair marginals -> NEGATIVE EV
even with zero extra vig. NFL RB rushing vs team side/ML (rho ~0.22) and QB passing vs game total (rho ~0.26) give
~+3-5% EV at market leg prices, which dies after ~0.8-1.3 implied-prob points of extra vig per leg, or a ~3-5%
parlay-multiplier haircut.

## Mechanism
A book that multiplies leg prices treats the legs as independent; game script ties an RB's carries to his team
leading (and a QB's yardage to the game total), so the joint outcome is more likely than the product. The other side
is the book's pricing engine; FanDuel/DraftKings correlate SGPs explicitly, so the edge exists only at a book that
truly multiplies market-priced legs -- unverified for Courtside.

## * Caveat on the TRUE verdict
PROTOCOL rule 1 requires a price you could actually bet. Courtside's SGP prices are NOT observed: the parlay price
here is the ASSUMED product of other books' market leg prices. The statistical gates are met for NFL RB rushing UNDER +
opponent covers, but on 2026 weeks 1-6 only (lockbox = 33 games, 7 dates); the realized lockbox ROI (+31%) is ~10x
the +3% expected from the long-run correlation, i.e. mostly favourable variance on top of a small real effect.
Treat it as: real structural edge of ~+3% per pair IF Courtside multiplies market-level leg prices with <= ~0.8 pt of
extra vig per leg and no correlation haircut; otherwise none.
