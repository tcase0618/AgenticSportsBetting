# NCAAF game predictor (win% first)

Code: `engine/predict/ncaaf.py` -> `engine/predict/common_team.py`. Inputs: `results/scripts/ncaaf_games.parquet`. Runtime 1.5 min. Strict walk-forward: for each season S, models are fit only on earlier games (season < S, or date < first of the month when monthly refits were chosen); the stack/calibration layer for season S is fit only on the out-of-sample predictions of seasons 2017..S-1. Candidates and refit cadence were picked on 2018-2020; **everything below is 2021+ unless stated**. Ties (NFL) are excluded from win% tables.

## 0. Bottom line

- Chosen: market-variant win model **base** (season refits), margin **base**, total **base**; no-line games use the script-only variant (win lr, margin base, total ridge). Spread / total picks use the best learned (non-market) margin / total model, **stack / stack**, because the bare market line gives P(cover) = P(over) = 50% by construction. `base` = the closing line itself (de-vigged ML, -spread, total).
- 2021+ straight-up winner, games with a closing line (n=4891): **model 76.20%** vs market favourite 76.20% vs script alone 75.14%. Agreement model = market favourite on 100.0% of games.
- All 2021+ games incl. those without a line (n=4918): model 76.29%, script 75.21%.
- Log loss (lines games): model 0.4664 vs market 0.4664 vs script 0.4910.

## 1. Per-season learning curve (straight-up winner, log loss, score MAE)

Games with a closing line. `model` = final pick; `mkt` = de-vigged closing favourite; `script` = Kalman script alone. MAE margin/total: model vs closing spread/total vs script.

| season | n_line | acc_model | acc_mkt | acc_script | ll_model | ll_mkt | ll_script | mae_m_model | mae_m_mkt | mae_m_script | mae_t_model | mae_t_mkt | mae_t_script | n_all | acc_all_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 790 | 77.47 | 77.47 | 77.34 | 0.44 | 0.44 | 0.47 | 12.18 | 12.18 | 12.82 | 12.91 | 12.91 | 13.68 | 845 | 77.51 |
| 2019 | 785 | 77.83 | 77.83 | 78.73 | 0.44 | 0.44 | 0.46 | 11.96 | 11.96 | 12.57 | 12.94 | 12.94 | 13.32 | 848 | 77.83 |
| 2020 | 541 | 75.23 | 75.23 | 74.12 | 0.51 | 0.51 | 0.52 | 12.77 | 12.77 | 13.06 | 13.85 | 13.85 | 13.93 | 542 | 75.28 |
| 2021 | 849 | 74.68 | 74.68 | 73.97 | 0.49 | 0.49 | 0.50 | 12.59 | 12.59 | 13.25 | 12.37 | 12.37 | 12.61 | 849 | 74.68 |
| 2022 | 849 | 75.85 | 75.85 | 73.14 | 0.49 | 0.49 | 0.51 | 12.21 | 12.21 | 13.06 | 12.08 | 12.08 | 12.73 | 854 | 75.88 |
| 2023 | 905 | 76.46 | 76.46 | 76.13 | 0.46 | 0.46 | 0.50 | 12.08 | 12.08 | 12.76 | 12.64 | 12.64 | 13.08 | 910 | 76.48 |
| 2024 | 904 | 73.78 | 73.78 | 73.34 | 0.49 | 0.49 | 0.51 | 11.96 | 11.96 | 12.74 | 12.65 | 12.65 | 12.85 | 919 | 74.21 |
| 2025 | 932 | 76.72 | 76.72 | 76.07 | 0.47 | 0.47 | 0.48 | 11.90 | 11.90 | 12.23 | 12.41 | 12.41 | 12.71 | 934 | 76.77 |
| 2026 | 452 | 82.96 | 82.96 | 80.75 | 0.35 | 0.35 | 0.40 | 11.70 | 11.70 | 12.88 | 11.89 | 11.89 | 12.29 | 452 | 82.96 |

Refit cadence and candidate comparison on the selection window 2018-2020 (market variant = games with a line; no-market variant = all games):

| variant | refit | target | model | logloss | acc% |
|---|---|---|---|---|---|
| mkt | season | win | base | 0.4547 | 77.0321 |
| mkt | season | win | lr | 0.4570 | 77.2212 |
| mkt | season | win | gbm | 0.4638 | 76.4650 |
| mkt | season | win | stack | 0.4590 | 77.3629 |
| mkt | month | win | base | 0.4547 | 77.0321 |
| mkt | month | win | lr | 0.4596 | 76.8904 |
| mkt | month | win | gbm | 0.4643 | 76.5595 |
| mkt | month | win | stack | 0.4577 | 76.9376 |
| nomkt | season | win | base | 0.4775 | 76.9128 |
| nomkt | season | win | lr | 0.4730 | 76.7785 |
| nomkt | season | win | gbm | 0.4759 | 75.0783 |
| nomkt | season | win | stack | 0.4765 | 76.1074 |

| variant | refit | target | model | MAE |
|---|---|---|---|---|
| mkt | season | margin | base | 12.248 |
| mkt | season | total | base | 13.164 |
| mkt | season | margin | ridge | 12.337 |
| mkt | season | total | ridge | 13.383 |
| mkt | season | margin | gbm | 12.295 |
| mkt | season | total | gbm | 13.257 |
| mkt | season | margin | stack | 12.252 |
| mkt | season | total | stack | 13.199 |
| mkt | month | margin | base | 12.248 |
| mkt | month | total | base | 13.164 |
| mkt | month | margin | ridge | 12.378 |
| mkt | month | total | ridge | 13.243 |
| mkt | month | margin | gbm | 12.346 |
| mkt | month | total | gbm | 13.269 |
| mkt | month | margin | stack | 12.266 |
| mkt | month | total | stack | 13.178 |
| nomkt | season | margin | base | 12.799 |
| nomkt | season | total | base | 13.575 |
| nomkt | season | margin | ridge | 12.822 |
| nomkt | season | total | ridge | 13.536 |
| nomkt | season | margin | gbm | 12.811 |
| nomkt | season | total | gbm | 13.640 |
| nomkt | season | margin | stack | 12.805 |
| nomkt | season | total | stack | 13.549 |

Monthly vs season refits on 2021+ (market variant, informational, not used for selection):

| refit | model | acc% | logloss | MAE margin |
|---|---|---|---|---|
| season | base | 76.1603 | 0.4664 | 12.1001 |
| season | lr | 76.1603 | 0.4680 | 12.1388 |
| season | gbm | 75.7718 | 0.4709 | 12.1640 |
| season | stack | 76.2216 | 0.4666 | 12.0985 |
| month | base | 76.1603 | 0.4664 | 12.1001 |
| month | lr | 76.0581 | 0.4684 | 12.1362 |
| month | gbm | 75.6696 | 0.4709 | 12.1877 |
| month | stack | 76.2830 | 0.4667 | 12.0992 |

Calibration of P(home win), 2021+ all games:

| bin | n | mean_pred | observed |
|---|---|---|---|
| (0.0, 0.1] | 157 | 0.058 | 0.089 |
| (0.1, 0.2] | 261 | 0.156 | 0.153 |
| (0.2, 0.3] | 386 | 0.249 | 0.272 |
| (0.3, 0.4] | 391 | 0.349 | 0.379 |
| (0.4, 0.5] | 381 | 0.445 | 0.472 |
| (0.5, 0.6] | 510 | 0.554 | 0.541 |
| (0.6, 0.7] | 524 | 0.650 | 0.647 |
| (0.7, 0.8] | 545 | 0.753 | 0.728 |
| (0.8, 0.9] | 572 | 0.851 | 0.867 |
| (0.9, 1.0] | 1191 | 0.966 | 0.968 |

## 2. Final-score accuracy

| games | n | MAE home | MAE away | MAE margin | MAE total | MAE margin (final int score) | MAE total (final int score) | winner & margin within 3% | winner & margin within 7% | exact score% | mkt MAE margin | mkt MAE total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021+ with line | 4891 | 9.184 | 8.163 | 12.100 | 12.385 | 12.098 | 12.385 | 18.871 | 34.615 | 0.184 | 12.100 | 12.385 |
| 2021+ all games | 4918 | 9.183 | 8.152 | 12.096 | 12.373 | 12.092 | 12.374 | 18.910 | 34.608 | 0.183 |  |  |
| 2021+ no line (script variant) | 27 | 9.003 | 6.211 | 11.336 | 10.298 | 11.111 | 10.296 | 25.926 | 33.333 | 0.000 |  |  |

Final predicted score = rounded predicted total and margin, with the margin forced to at least 1 point toward the predicted winner (so the score always agrees with the pick).

## 3. Win rate by confidence tier

Cells are `n / win%` per season. Pushes excluded. `break-even%` = moneyline: the average vigged implied probability of the picked side's closing price; spread/total: -110.

### (a) Moneyline winner (games with a closing line)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 70 / 47% | 47 / 68% | 73 / 47% | 96 / 51% | 79 / 53% | 26 / 35% | 391 | 50.9 | 55.1 |
| 55-60 | 73 / 49% | 84 / 55% | 95 / 61% | 105 / 47% | 97 / 62% | 42 / 69% | 496 | 56.0 | 59.4 |
| 60-65 | 92 / 58% | 90 / 57% | 78 / 59% | 83 / 64% | 98 / 65% | 27 / 63% | 468 | 60.7 | 64.5 |
| 65-70 | 69 / 70% | 64 / 61% | 99 / 69% | 75 / 72% | 101 / 63% | 37 / 62% | 445 | 66.5 | 69.6 |
| 70-75 | 64 / 62% | 90 / 76% | 76 / 59% | 74 / 74% | 81 / 70% | 24 / 92% | 409 | 70.2 | 74.4 |
| 75-80 | 88 / 75% | 94 / 77% | 104 / 80% | 104 / 68% | 87 / 76% | 45 / 73% | 522 | 74.9 | 79.6 |
| 80-85 | 80 / 86% | 80 / 78% | 83 / 88% | 74 / 76% | 94 / 86% | 33 / 91% | 444 | 83.6 | 84.9 |
| 85+ | 313 / 92% | 300 / 91% | 297 / 96% | 293 / 96% | 295 / 95% | 218 / 97% | 1716 | 94.5 | 93.9 |

**Best tier with >=100 games 2021+: 85+% -> 94.5% (n=1716).**

Cumulative (P(pick) at or above threshold): >=0.50: 76.2% (n=4891); >=0.55: 78.4% (n=4500); >=0.60: 81.2% (n=4004); >=0.65: 83.9% (n=3536); >=0.70: 86.4% (n=3091); >=0.75: 88.9% (n=2682); >=0.80: 92.2% (n=2160); >=0.85: 94.5% (n=1716); >=0.90: 96.1% (n=1331)

### (b) Spread pick (P(cover) from predicted margin distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 835 / 55% | 835 / 51% | 889 / 52% | 889 / 50% | 914 / 49% | 446 / 52% | 4808 | 51.5 | 52.4 |
| 55-60 | 0 | 1 / 0% | 0 | 0 | 0 | 0 | 1 | 0.0 | 52.4 |
| 60-65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 51.5% (n=4808).**

Cumulative (P(pick) at or above threshold): >=0.50: 51.4% (n=4809); >=0.55: 0.0% (n=1)

### (c) Total pick (P(over/under) from predicted total distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 841 / 52% | 838 / 50% | 896 / 54% | 884 / 53% | 903 / 50% | 450 / 50% | 4812 | 51.8 | 52.4 |
| 55-60 | 0 | 0 | 0 | 5 / 100% | 23 / 57% | 2 / 50% | 30 | 63.3 | 52.4 |
| 60-65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 51.8% (n=4812).**

Cumulative (P(pick) at or above threshold): >=0.50: 51.8% (n=4842); >=0.55: 63.3% (n=30)

## 4. Daily top-N confidence parlays (moneyline picks)

Each date with at least N games, the N most confident model picks parlayed; 2021+. `model-expected%` = mean product of the model's pick probabilities (a calibration check); the last column parlays the N biggest market favourites instead.

Games with a line:

| top-N | days | hit% | model-expected% | market-fav top-N hit% |
|---|---|---|---|---|
| 1 | 363 | 83.5 | 83.3 | 83.5 |
| 2 | 289 | 68.9 | 69.3 | 68.9 |
| 3 | 196 | 63.3 | 66.5 | 63.3 |
| 4 | 138 | 68.1 | 69.6 | 68.1 |
| 5 | 115 | 70.4 | 71.2 | 70.4 |

All games (incl. no-line games):

| top-N | days | hit% | model-expected% |
|---|---|---|---|
| 1 | 365 | 83.3 | 83.1 |
| 2 | 289 | 68.9 | 69.3 |
| 3 | 197 | 62.9 | 66.3 |
| 4 | 138 | 68.1 | 69.7 |
| 5 | 115 | 70.4 | 71.4 |

## 5. The formula in plain English

Compact version (OLS / logistic on all out-of-sample 2018+ games with a line; this is what the full model boils down to):

- **predicted home margin = +0.05 + 1.015 x (market margin = -closing spread) + 0.046 x (script margin - market margin)**
- **predicted total = +7.30 + 0.877 x closing total + -0.058 x (script total - closing total)**
- **logit P(home win) = +0.056 + 0.970 x logit(de-vigged market P(home)) + -0.000 x (script logit - market logit)**
- home score = (total + margin) / 2, away score = (total - margin) / 2.

Stack weights of the deployed model (fit on all out-of-sample seasons):

- logit P(home) = +0.030 + 0.195 x logit(logistic) + 0.011 x logit(LightGBM-from-market) + 0.793 x logit(market)
- margin = +0.01 + 0.156 x ridge + 0.174 x LightGBM + 0.685 x market margin
- total = +6.19 + -0.073 x ridge + 0.062 x LightGBM + 0.906 x market total

Top features, Logistic P(home win) (deployed fit; log-odds):

| feature | per 1 sd | per raw unit |
|---|---|---|
| script_vs_mkt | -0.291 | -0.059 |
| mkt_margin | 0.245 | 0.015 |
| mkt_logit | 0.234 | 0.111 |
| margin_b | 0.233 | 0.014 |
| p_home_win_b | 0.206 | 0.814 |
| script_logit | 0.174 | 0.111 |
| margin | 0.162 | 0.010 |
| mu_away | -0.152 | -0.018 |
| mu_home | 0.149 | 0.017 |
| margin_core | 0.148 | 0.011 |
| p_home_win | 0.112 | 0.445 |
| conf_diff | 0.104 | 0.010 |

Top features, Ridge margin (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| script_logit | 2.637 | 1.691 |
| mkt_margin | 2.600 | 0.158 |
| script_vs_mkt | -2.490 | -0.505 |
| p_home_win_b | 2.345 | 9.250 |
| margin_b | 2.191 | 0.134 |
| mu_away | -2.020 | -0.244 |
| margin | 1.910 | 0.120 |
| mu_home | 1.533 | 0.171 |
| mkt_logit | 1.523 | 0.719 |
| missing(total_move) | 1.085 | 3.135 |
| missing(spread_move) | -1.068 | -3.056 |
| total_core | 1.031 | 0.121 |

Top features, Ridge total (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| total_line | 1.779 | 0.225 |
| total_b | 1.767 | 0.226 |
| total_vs_line | -1.643 | -0.384 |
| total_core | 1.132 | 0.133 |
| total | 1.073 | 0.163 |
| total_move | 0.566 | 0.254 |
| mu_home | 0.515 | 0.058 |
| p_home_win_b | -0.504 | -1.988 |
| rest_h | -0.386 | -0.005 |
| week | 0.354 | 0.085 |
| neutral | -0.334 | -1.686 |
| pass_match | -0.327 | -2.569 |

LightGBM (boosted from the market logit) top gain features: plays_a (1423), pass_rate_a (1130), plays_h (1089), pass_rate_h (1017), epa_sum (961), total_vs_line (958), total_move (915), rush_match (882), sd_margin (807), plays_script (780)

## 6. Ablation (what each feature group adds)

Logistic + ridge, season refits, games with a line, 2021+. Groups: market = mkt_logit, mkt_margin, total_line, spread_move, total_move, script_vs_mkt, total_vs_line, has_ml; blend = script-x-market blends (3 cols); script = 22 Kalman/enrichment cols; context = 8 cols (week, neutral, postseason, rest_h, rest_a, conf_diff, p5_vs_g5, fbs_vs_fcs).

| features | n_feat | SU acc% | logloss | MAE margin | MAE total |
|---|---|---|---|---|---|
| full (all groups) | 41 | 76.1603 | 0.4680 | 12.1388 | 12.4078 |
| - market (no lines) | 30 | 74.7700 | 0.4901 | 12.8048 | 12.8340 |
| - blend | 38 | 76.0990 | 0.4678 | 12.1342 | 12.4236 |
| - script | 19 | 76.1398 | 0.4679 | 12.1499 | 12.3592 |
| - context | 33 | 76.2830 | 0.4690 | 12.1518 | 12.4318 |
| market only | 8 | 76.0376 | 0.4675 | 12.1303 | 12.3792 |
| market + script | 30 | 76.2625 | 0.4687 | 12.1498 | 12.4350 |

## 7. Does more history help?

Evaluation fixed at 2023+ seasons (games with a line). The chosen base models are retrained with only the last k seasons (that have closing lines) before each test season (or all prior seasons, optionally with exponential recency weights by season), then combined with that season's stack layer. Win model shown: **stack**; margin **stack**, total **stack** (when the deployed choice is the bare market line, the learned stack/ridge is shown instead, since the market line itself has no training window). The high-confidence column uses the best moneyline tier's lower bound from section 3.

| training window | SU win% | logloss | MAE margin | MAE total | P(pick)>=0.85 n | P(pick)>=0.85 win% |
|---|---|---|---|---|---|---|
| last 1 | 76.7303 | 0.4556 | 11.9276 | 12.4549 | 1032 | 96.4147 |
| last 2 | 76.8556 | 0.4548 | 11.9314 | 12.4531 | 1066 | 96.2477 |
| last 4 | 76.8243 | 0.4542 | 11.9397 | 12.4554 | 1072 | 96.1754 |
| last 6 | 76.7930 | 0.4542 | 11.9374 | 12.4537 | 1072 | 96.0821 |
| all prior | 76.8243 | 0.4541 | 11.9376 | 12.4504 | 1074 | 96.2756 |
| all, decay half-life 1 seasons | 76.8556 | 0.4543 | 11.9372 | 12.4570 | 1077 | 96.1931 |
| all, decay half-life 2 seasons | 76.8869 | 0.4542 | 11.9369 | 12.4568 | 1076 | 96.2825 |
| all, decay half-life 4 seasons | 76.7930 | 0.4542 | 11.9337 | 12.4523 | 1071 | 96.2652 |

Read: win% last-1 76.73 -> all 76.82; log loss 0.4556 -> 0.4541; margin MAE 11.928 -> 11.938. Win% moves only 0.13 pts across windows (noise level). Log loss is within 0.001 of its best by 'last 2' (last-1 0.4556 vs all 0.4541); margin MAE is within 0.01 of its best by 'last 1' (last-1 11.928 vs all 11.938). It plateaus: beyond that, more seasons add nothing measurable. Recency weighting: best decay = all, decay half-life 2 seasons (log loss 0.4542, MAE margin 11.937) vs unweighted all-prior 0.4541 / 11.938: no gain. The model is dominated by the closing line, which already embodies the latest information, so the learned layer has little to gain from extra history.

## 8. Live use

`from engine.predict.ncaaf import predict; predict(slate_df)` with rows shaped like `results/scripts/ncaaf_games.parquet` (the script engine's pre-game columns for upcoming games) plus `spread_home`, `total_line`, `home_ml`, `away_ml` (and `spread_open_home`/`total_open` if known). Rows without `spread_home` use the script-only model. Model: `results/predict/ncaaf_model.pkl` (trained through 2026-10-14). Required script columns: conf_diff, epa_net, epa_sum, fbs_vs_fcs, margin, margin_b, margin_core, mu_away, mu_home, neutral, p5_vs_g5, p_home_win, p_home_win_b, pass_match, pass_rate_a, pass_rate_h, plays_a, plays_h, plays_script, postseason, rest_a, rest_h, rho, rush_match, script_logit, sd_away, sd_home, sd_margin, sd_total, total, total_b, total_core, week.
