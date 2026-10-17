# NCAAM game predictor (win% first)

Code: `engine/predict/ncaam.py` -> `engine/predict/common_team.py`. Inputs: `results/scripts/ncaam_games.parquet`. Runtime 4.9 min. Strict walk-forward: for each season S, models are fit only on earlier games (season < S, or date < first of the month when monthly refits were chosen); the stack/calibration layer for season S is fit only on the out-of-sample predictions of seasons 2017..S-1. Candidates and refit cadence were picked on 2018-2020; **everything below is 2021+ unless stated**. Ties (NFL) are excluded from win% tables.

## 0. Bottom line

- Chosen: market-variant win model **stack** (month refits), margin **base**, total **base**; no-line games use the script-only variant (win lr, margin stack, total ridge). Spread / total picks use the best learned (non-market) margin / total model, **stack / stack**, because the bare market line gives P(cover) = P(over) = 50% by construction. `base` = the closing line itself (de-vigged ML, -spread, total).
- 2021+ straight-up winner, games with a closing line (n=19880): **model 72.16%** vs market favourite 72.20% vs script alone 70.66%. Agreement model = market favourite on 99.3% of games.
- All 2021+ games incl. those without a line (n=35304): model 74.49%, script 73.55%.
- Log loss (lines games): model 0.5326 vs market 0.5330 vs script 0.5539.

## 1. Per-season learning curve (straight-up winner, log loss, score MAE)

Games with a closing line. `model` = final pick; `mkt` = de-vigged closing favourite; `script` = Kalman script alone. MAE margin/total: model vs closing spread/total vs script.

| season | n_line | acc_model | acc_mkt | acc_script | ll_model | ll_mkt | ll_script | mae_m_model | mae_m_mkt | mae_m_script | mae_t_model | mae_t_mkt | mae_t_script | n_all | acc_all_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 4073 | 73.78 | 73.56 | 72.28 | 0.52 | 0.52 | 0.54 | 8.70 | 8.70 | 9.07 | 13.51 | 13.51 | 13.65 | 6003 | 76.15 |
| 2019 | 5529 | 73.63 | 73.47 | 72.45 | 0.51 | 0.51 | 0.54 | 8.78 | 8.78 | 9.18 | 13.33 | 13.33 | 13.50 | 6049 | 75.42 |
| 2020 | 5327 | 73.38 | 73.34 | 72.05 | 0.52 | 0.52 | 0.54 | 8.77 | 8.77 | 9.23 | 13.15 | 13.15 | 13.34 | 5766 | 75.30 |
| 2021 | 3924 | 71.64 | 71.64 | 70.23 | 0.54 | 0.54 | 0.56 | 9.05 | 9.05 | 9.48 | 13.11 | 13.11 | 13.44 | 4285 | 73.68 |
| 2022 | 5476 | 73.19 | 73.28 | 71.75 | 0.52 | 0.52 | 0.54 | 8.64 | 8.64 | 9.12 | 12.95 | 12.95 | 13.15 | 5965 | 75.26 |
| 2023 | 5714 | 72.59 | 72.54 | 71.39 | 0.53 | 0.53 | 0.55 | 8.87 | 8.87 | 9.22 | 12.96 | 12.96 | 13.09 | 6221 | 74.68 |
| 2024 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 6242 | 73.81 |
| 2025 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 6291 | 74.58 |
| 2026 | 4766 | 70.90 | 71.00 | 68.88 | 0.54 | 0.54 | 0.56 | 8.86 | 8.86 | 9.28 | 13.31 | 13.31 | 13.60 | 6300 | 74.73 |

Refit cadence and candidate comparison on the selection window 2018-2020 (market variant = games with a line; no-market variant = all games):

| variant | refit | target | model | logloss | acc% |
|---|---|---|---|---|---|
| mkt | season | win | base | 0.5193 | 73.5481 |
| mkt | season | win | lr | 0.5201 | 73.5079 |
| mkt | season | win | gbm | 0.5234 | 73.1730 |
| mkt | season | win | stack | 0.5199 | 73.3204 |
| mkt | month | win | base | 0.5193 | 73.5481 |
| mkt | month | win | lr | 0.5203 | 73.4744 |
| mkt | month | win | gbm | 0.5229 | 72.9922 |
| mkt | month | win | stack | 0.5187 | 73.5816 |
| nomkt | season | win | base | 0.5057 | 74.5145 |
| nomkt | season | win | lr | 0.4984 | 74.7166 |
| nomkt | season | win | gbm | 0.4993 | 74.3798 |
| nomkt | season | win | stack | 0.4990 | 74.6043 |

| variant | refit | target | model | MAE |
|---|---|---|---|---|
| mkt | season | margin | base | 8.754 |
| mkt | season | total | base | 13.315 |
| mkt | season | margin | ridge | 8.765 |
| mkt | season | total | ridge | 13.343 |
| mkt | season | margin | gbm | 8.810 |
| mkt | season | total | gbm | 13.406 |
| mkt | season | margin | stack | 8.779 |
| mkt | season | total | stack | 13.343 |
| mkt | month | margin | base | 8.754 |
| mkt | month | total | base | 13.315 |
| mkt | month | margin | ridge | 8.765 |
| mkt | month | total | ridge | 13.335 |
| mkt | month | margin | gbm | 8.807 |
| mkt | month | total | gbm | 13.381 |
| mkt | month | margin | stack | 8.756 |
| mkt | month | total | stack | 13.330 |
| nomkt | season | margin | base | 9.685 |
| nomkt | season | total | base | 13.624 |
| nomkt | season | margin | ridge | 9.465 |
| nomkt | season | total | ridge | 13.614 |
| nomkt | season | margin | gbm | 9.375 |
| nomkt | season | total | gbm | 13.660 |
| nomkt | season | margin | stack | 9.350 |
| nomkt | season | total | stack | 13.618 |

Monthly vs season refits on 2021+ (market variant, informational, not used for selection):

| refit | model | acc% | logloss | MAE margin |
|---|---|---|---|---|
| season | base | 72.1932 | 0.5330 | 8.8376 |
| season | lr | 72.0825 | 0.5336 | 8.8352 |
| season | gbm | 72.0070 | 0.5369 | 8.8694 |
| season | stack | 72.2133 | 0.5328 | 8.8340 |
| month | base | 72.1932 | 0.5330 | 8.8376 |
| month | lr | 72.0825 | 0.5335 | 8.8346 |
| month | gbm | 71.8461 | 0.5355 | 8.8683 |
| month | stack | 72.1630 | 0.5326 | 8.8323 |

Calibration of P(home win), 2021+ all games:

| bin | n | mean_pred | observed |
|---|---|---|---|
| (0.0, 0.1] | 374 | 0.067 | 0.075 |
| (0.1, 0.2] | 1110 | 0.155 | 0.164 |
| (0.2, 0.3] | 1839 | 0.253 | 0.255 |
| (0.3, 0.4] | 2602 | 0.353 | 0.367 |
| (0.4, 0.5] | 3311 | 0.450 | 0.464 |
| (0.5, 0.6] | 4050 | 0.554 | 0.551 |
| (0.6, 0.7] | 4821 | 0.651 | 0.637 |
| (0.7, 0.8] | 4957 | 0.750 | 0.743 |
| (0.8, 0.9] | 4816 | 0.850 | 0.850 |
| (0.9, 1.0] | 7424 | 0.961 | 0.964 |

## 2. Final-score accuracy

| games | n | MAE home | MAE away | MAE margin | MAE total | MAE margin (final int score) | MAE total (final int score) | winner & margin within 3% | winner & margin within 6% | exact score% | mkt MAE margin | mkt MAE total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021+ with line | 19880 | 7.989 | 7.868 | 8.838 | 13.070 | 8.825 | 13.072 | 24.105 | 40.387 | 0.181 | 8.838 | 13.070 |
| 2021+ all games | 35304 | 8.270 | 7.977 | 9.308 | 13.184 | 9.301 | 13.192 | 23.249 | 38.998 | 0.173 |  |  |
| 2021+ no line (script variant) | 15424 | 8.632 | 8.117 | 9.915 | 13.330 | 9.913 | 13.346 | 22.147 | 37.208 | 0.162 |  |  |

Final predicted score = rounded predicted total and margin, with the margin forced to at least 1 point toward the predicted winner (so the score always agrees with the pick).

## 3. Win rate by confidence tier

Cells are `n / win%` per season. Pushes excluded. `break-even%` = moneyline: the average vigged implied probability of the picked side's closing price; spread/total: -110.

### (a) Moneyline winner (games with a closing line)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 368 / 53% | 486 / 50% | 588 / 50% | 0 | 0 | 595 / 51% | 2037 | 50.8 | 54.7 |
| 55-60 | 499 / 58% | 719 / 58% | 762 / 60% | 0 | 0 | 601 / 54% | 2581 | 57.8 | 58.6 |
| 60-65 | 503 / 56% | 638 / 60% | 682 / 62% | 0 | 0 | 556 / 60% | 2379 | 59.9 | 63.5 |
| 65-70 | 434 / 67% | 613 / 64% | 654 / 67% | 0 | 0 | 510 / 67% | 2211 | 66.2 | 68.3 |
| 70-75 | 441 / 71% | 578 / 75% | 623 / 72% | 0 | 0 | 518 / 73% | 2160 | 72.5 | 73.5 |
| 75-80 | 377 / 77% | 527 / 82% | 564 / 76% | 0 | 0 | 491 / 74% | 1959 | 77.2 | 78.2 |
| 80-85 | 355 / 78% | 484 / 83% | 526 / 85% | 0 | 0 | 411 / 82% | 1776 | 82.2 | 83.4 |
| 85+ | 947 / 93% | 1431 / 91% | 1315 / 92% | 0 | 0 | 1084 / 92% | 4777 | 92.0 | 92.9 |

**Best tier with >=100 games 2021+: 85+% -> 92.0% (n=4777).**

Cumulative (P(pick) at or above threshold): >=0.50: 72.2% (n=19880); >=0.55: 74.6% (n=17843); >=0.60: 77.5% (n=15262); >=0.65: 80.7% (n=12883); >=0.70: 83.7% (n=10672); >=0.75: 86.5% (n=8512); >=0.80: 89.3% (n=6553); >=0.85: 92.0% (n=4777); >=0.90: 94.7% (n=3080)

### (b) Spread pick (P(cover) from predicted margin distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 3832 / 51% | 5355 / 50% | 5559 / 50% | 0 | 0 | 4684 / 51% | 19430 | 50.6 | 52.4 |
| 55-60 | 17 / 65% | 16 / 56% | 36 / 67% | 0 | 0 | 81 / 56% | 150 | 59.3 | 52.4 |
| 60-65 | 0 | 0 | 8 / 88% | 0 | 0 | 1 / 100% | 9 | 88.9 | 52.4 |
| 65-70 | 0 | 0 | 3 / 100% | 0 | 0 | 0 | 3 | 100.0 | 52.4 |
| 70-75 | 0 | 0 | 3 / 100% | 0 | 0 | 0 | 3 | 100.0 | 52.4 |
| 75-80 | 1 / 0% | 0 | 2 / 100% | 0 | 0 | 0 | 3 | 66.7 | 52.4 |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 55-60% -> 59.3% (n=150).**

Cumulative (P(pick) at or above threshold): >=0.50: 50.7% (n=19598); >=0.55: 62.5% (n=168); >=0.60: 88.9% (n=18); >=0.65: 88.9% (n=9); >=0.70: 83.3% (n=6); >=0.75: 66.7% (n=3)

### (c) Total pick (P(over/under) from predicted total distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 3837 / 51% | 5408 / 50% | 5635 / 52% | 0 | 0 | 4765 / 50% | 19645 | 50.7 | 52.4 |
| 55-60 | 46 / 57% | 6 / 50% | 2 / 0% | 0 | 0 | 1 / 0% | 55 | 52.7 | 52.4 |
| 60-65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 50.7% (n=19645).**

Cumulative (P(pick) at or above threshold): >=0.50: 50.7% (n=19700); >=0.55: 52.7% (n=55)

## 4. Daily top-N confidence parlays (moneyline picks)

Each date with at least N games, the N most confident model picks parlayed; 2021+. `model-expected%` = mean product of the model's pick probabilities (a calibration check); the last column parlays the N biggest market favourites instead.

Games with a line:

| top-N | days | hit% | model-expected% | market-fav top-N hit% |
|---|---|---|---|---|
| 1 | 534 | 93.6 | 93.7 | 93.3 |
| 2 | 523 | 86.2 | 86.7 | 86.6 |
| 3 | 503 | 80.9 | 80.5 | 81.5 |
| 4 | 501 | 75.6 | 72.7 | 75.6 |
| 5 | 487 | 68.4 | 66.2 | 68.2 |

All games (incl. no-line games):

| top-N | days | hit% | model-expected% |
|---|---|---|---|
| 1 | 855 | 94.5 | 94.6 |
| 2 | 841 | 89.7 | 88.6 |
| 3 | 812 | 84.9 | 83.4 |
| 4 | 804 | 80.0 | 77.1 |
| 5 | 779 | 73.9 | 72.4 |

## 5. The formula in plain English

Compact version (OLS / logistic on all out-of-sample 2018+ games with a line; this is what the full model boils down to):

- **predicted home margin = -0.05 + 1.010 x (market margin = -closing spread) + 0.127 x (script margin - market margin)**
- **predicted total = +3.16 + 0.982 x closing total + 0.117 x (script total - closing total)**
- **logit P(home win) = +0.009 + 1.057 x logit(de-vigged market P(home)) + 0.105 x (script logit - market logit)**
- home score = (total + margin) / 2, away score = (total - margin) / 2.

Stack weights of the deployed model (fit on all out-of-sample seasons):

- logit P(home) = +0.012 + 0.343 x logit(logistic) + 0.065 x logit(LightGBM-from-market) + 0.619 x logit(market)
- margin = +0.04 + 0.582 x ridge + 0.058 x LightGBM + 0.355 x market margin
- total = +2.90 + 0.270 x ridge + 0.154 x LightGBM + 0.560 x market total

Top features, Logistic P(home win) (deployed fit; log-odds):

| feature | per 1 sd | per raw unit |
|---|---|---|
| mkt_logit | 0.365 | 0.269 |
| margin_b | 0.171 | 0.020 |
| script_vs_mkt | -0.160 | -0.043 |
| mkt_margin | 0.157 | 0.018 |
| margin_mx | 0.148 | 0.017 |
| script_logit | 0.118 | 0.103 |
| margin_core | 0.092 | 0.010 |
| margin | 0.089 | 0.010 |
| mu_away | -0.069 | -0.011 |
| mu_away_core | -0.063 | -0.009 |
| gs_home | 0.060 | 0.007 |
| mu_home_core | 0.057 | 0.008 |

Top features, Ridge margin (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| mkt_logit | 3.607 | 2.656 |
| margin_b | 1.413 | 0.163 |
| script_logit | 1.235 | 1.076 |
| gs_home | 1.132 | 0.123 |
| gs_away | -0.924 | -0.100 |
| script_vs_mkt | -0.885 | -0.239 |
| mkt_margin | 0.635 | 0.073 |
| total_mx | 0.500 | 0.051 |
| same_conf | -0.412 | -0.870 |
| conf_m_home | 0.404 | 0.714 |
| mult_home | 0.400 | 0.047 |
| margin_mx | 0.395 | 0.045 |

Top features, Ridge total (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| total_vs_line | -1.573 | -0.461 |
| total_line | 1.564 | 0.158 |
| total_b | 1.270 | 0.130 |
| total_mx | 1.268 | 0.130 |
| total | 1.056 | 0.111 |
| total_core | 1.017 | 0.099 |
| mu_away | 0.817 | 0.130 |
| mu_away_core | 0.811 | 0.120 |
| mu_home | 0.747 | 0.113 |
| mu_home_core | 0.727 | 0.106 |
| neutral | -0.530 | -1.586 |
| mult_away | 0.526 | 0.067 |

LightGBM (boosted from the market logit) top gain features: corr_away (1269), total_vs_line (1180), tempo_home (1171), tempo_away (1168), conf_m_home (1140), ewm_home (1087), conf_t_home (1060), mult_home (1032), sd_total (992), conf_t_away (974)

## 6. Ablation (what each feature group adds)

Logistic + ridge, season refits, games with a line, 2021+. Groups: market = mkt_logit, mkt_margin, total_line, spread_move, total_move, script_vs_mkt, total_vs_line, has_ml; blend = script-x-market blends (4 cols); script = 20 Kalman/enrichment cols; context = 15 cols (neutral, playoff, tempo_home, tempo_away, conf_m_home, conf_m_away, conf_t_home, conf_t_away, same_conf, rest_home, rest_away, gs_home...).

| features | n_feat | SU acc% | logloss | MAE margin | MAE total |
|---|---|---|---|---|---|
| full (all groups) | 47 | 72.0825 | 0.5336 | 8.8352 | 13.0725 |
| - market (no lines) | 35 | 71.0211 | 0.5484 | 9.0969 | 13.2834 |
| - blend | 43 | 72.0775 | 0.5335 | 8.8359 | 13.0729 |
| - script | 27 | 72.1278 | 0.5335 | 8.8346 | 13.0767 |
| - context | 32 | 72.0221 | 0.5333 | 8.8376 | 13.0745 |
| market only | 8 | 72.1881 | 0.5331 | 8.8381 | 13.0747 |
| market + script | 28 | 72.0171 | 0.5332 | 8.8379 | 13.0746 |

## 7. Does more history help?

Evaluation fixed at 2023+ seasons (games with a line). The chosen base models are retrained with only the last k seasons (that have closing lines) before each test season (or all prior seasons, optionally with exponential recency weights by season), then combined with that season's stack layer. Win model shown: **stack**; margin **stack**, total **stack** (when the deployed choice is the bare market line, the learned stack/ridge is shown instead, since the market line itself has no training window). The high-confidence column uses the best moneyline tier's lower bound from section 3.

| training window | SU win% | logloss | MAE margin | MAE total | P(pick)>=0.85 n | P(pick)>=0.85 win% |
|---|---|---|---|---|---|---|
| last 1 | 71.7080 | 0.5376 | 8.8659 | 13.1245 | 2417 | 92.4286 |
| last 2 | 71.7939 | 0.5376 | 8.8662 | 13.1236 | 2442 | 92.3014 |
| last 4 | 71.7844 | 0.5379 | 8.8667 | 13.1231 | 2471 | 92.0275 |
| last 6 | 71.8702 | 0.5373 | 8.8609 | 13.1184 | 2444 | 92.1031 |
| all prior | 71.8893 | 0.5369 | 8.8596 | 13.1190 | 2434 | 92.0707 |
| all, decay half-life 1 seasons | 71.7366 | 0.5378 | 8.8666 | 13.1199 | 2451 | 92.1257 |
| all, decay half-life 2 seasons | 71.7176 | 0.5375 | 8.8644 | 13.1184 | 2450 | 92.0000 |
| all, decay half-life 4 seasons | 71.7271 | 0.5373 | 8.8625 | 13.1185 | 2462 | 92.0390 |

Read: win% last-1 71.71 -> all 71.89; log loss 0.5376 -> 0.5369; margin MAE 8.866 -> 8.860. Win% moves only 0.18 pts across windows (noise level). Log loss is within 0.001 of its best by 'last 1' (last-1 0.5376 vs all 0.5369); margin MAE is within 0.01 of its best by 'last 1' (last-1 8.866 vs all 8.860). More history keeps helping a little. Recency weighting: best decay = all, decay half-life 4 seasons (log loss 0.5373, MAE margin 8.863) vs unweighted all-prior 0.5369 / 8.860: no gain. The model is dominated by the closing line, which already embodies the latest information, so the learned layer has little to gain from extra history.

## 8. Live use

`from engine.predict.ncaam import predict; predict(slate_df)` with rows shaped like `results/scripts/ncaam_games.parquet` (the script engine's pre-game columns for upcoming games) plus `spread_home`, `total_line`, `home_ml`, `away_ml` (and `spread_open_home`/`total_open` if known). Rows without `spread_home` use the script-only model. Model: `results/predict/ncaam_model.pkl` (trained through 2026-04-06). Required script columns: conf_m_away, conf_m_home, conf_t_away, conf_t_home, corr_away, corr_home, ewm_away, ewm_home, games_away, games_home, gs_away, gs_home, margin, margin_b, margin_core, margin_mx, mu_away, mu_away_core, mu_home, mu_home_core, mult_away, mult_home, neutral, playoff, rest_away, rest_home, rho, same_conf, script_logit, sd_away, sd_home, sd_margin, sd_total, tempo_away, tempo_home, total, total_b, total_core, total_mx.
