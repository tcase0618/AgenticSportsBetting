# NFL game predictor (win% first)

Code: `engine/predict/nfl.py` -> `engine/predict/common_team.py`. Inputs: `results/scripts/nfl_games.parquet`. Runtime 1.0 min. Strict walk-forward: for each season S, models are fit only on earlier games (season < S, or date < first of the month when monthly refits were chosen); the stack/calibration layer for season S is fit only on the out-of-sample predictions of seasons 2017..S-1. Candidates and refit cadence were picked on 2018-2020; **everything below is 2021+ unless stated**. Ties (NFL) are excluded from win% tables.

## 0. Bottom line

- Chosen: market-variant win model **base** (season refits), margin **base**, total **base**; no-line games use the script-only variant (win base, margin stack, total ridge). Spread / total picks use the best learned (non-market) margin / total model, **stack / stack**, because the bare market line gives P(cover) = P(over) = 50% by construction. `base` = the closing line itself (de-vigged ML, -spread, total).
- 2021+ straight-up winner, games with a closing line (n=1498): **model 66.22%** vs market favourite 66.22% vs script alone 63.62%. Agreement model = market favourite on 100.0% of games.
- All 2021+ games incl. those without a line (n=1498): model 66.22%, script 63.62%.
- Log loss (lines games): model 0.6122 vs market 0.6122 vs script 0.6372.

## 1. Per-season learning curve (straight-up winner, log loss, score MAE)

Games with a closing line. `model` = final pick; `mkt` = de-vigged closing favourite; `script` = Kalman script alone. MAE margin/total: model vs closing spread/total vs script.

| season | n_line | acc_model | acc_mkt | acc_script | ll_model | ll_mkt | ll_script | mae_m_model | mae_m_mkt | mae_m_script | mae_t_model | mae_t_mkt | mae_t_script | n_all | acc_all_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 265 | 65.66 | 65.66 | 64.15 | 0.61 | 0.61 | 0.62 | 9.87 | 9.87 | 9.94 | 10.72 | 10.72 | 11.28 | 265 | 65.66 |
| 2019 | 266 | 64.29 | 64.29 | 65.04 | 0.62 | 0.62 | 0.64 | 10.18 | 10.18 | 10.45 | 10.78 | 10.78 | 10.98 | 266 | 64.29 |
| 2020 | 268 | 67.54 | 67.54 | 64.55 | 0.59 | 0.59 | 0.63 | 9.79 | 9.79 | 10.10 | 10.30 | 10.30 | 10.73 | 268 | 67.54 |
| 2021 | 284 | 62.32 | 62.32 | 63.03 | 0.62 | 0.62 | 0.65 | 10.67 | 10.67 | 10.95 | 10.81 | 10.81 | 10.93 | 284 | 62.32 |
| 2022 | 282 | 65.60 | 65.60 | 63.12 | 0.60 | 0.60 | 0.64 | 8.78 | 8.78 | 9.21 | 10.40 | 10.40 | 10.78 | 282 | 65.60 |
| 2023 | 285 | 67.02 | 67.02 | 62.81 | 0.63 | 0.63 | 0.65 | 9.98 | 9.98 | 10.46 | 10.17 | 10.17 | 10.43 | 285 | 67.02 |
| 2024 | 285 | 70.88 | 70.88 | 67.37 | 0.59 | 0.59 | 0.62 | 9.70 | 9.70 | 10.14 | 9.77 | 9.77 | 10.07 | 285 | 70.88 |
| 2025 | 284 | 65.85 | 65.85 | 61.62 | 0.61 | 0.61 | 0.64 | 9.67 | 9.67 | 10.24 | 10.42 | 10.42 | 10.47 | 284 | 65.85 |
| 2026 | 78 | 64.10 | 64.10 | 64.10 | 0.65 | 0.65 | 0.63 | 9.78 | 9.78 | 9.73 | 10.63 | 10.63 | 11.27 | 78 | 64.10 |

Refit cadence and candidate comparison on the selection window 2018-2020 (market variant = games with a line; no-market variant = all games):

| variant | refit | target | model | logloss | acc% |
|---|---|---|---|---|---|
| mkt | season | win | base | 0.6066 | 66.0826 |
| mkt | season | win | lr | 0.6157 | 65.4568 |
| mkt | season | win | gbm | 0.6208 | 65.5820 |
| mkt | season | win | stack | 0.6091 | 65.9574 |
| mkt | month | win | base | 0.6066 | 66.0826 |
| mkt | month | win | lr | 0.6171 | 65.4568 |
| mkt | month | win | gbm | 0.6219 | 66.3329 |
| mkt | month | win | stack | 0.6080 | 66.2078 |
| nomkt | season | win | base | 0.6266 | 64.5807 |
| nomkt | season | win | lr | 0.6280 | 63.4543 |
| nomkt | season | win | gbm | 0.6430 | 64.2053 |
| nomkt | season | win | stack | 0.6268 | 64.5807 |

| variant | refit | target | model | MAE |
|---|---|---|---|---|
| mkt | season | margin | base | 9.946 |
| mkt | season | total | base | 10.601 |
| mkt | season | margin | ridge | 10.106 |
| mkt | season | total | ridge | 10.757 |
| mkt | season | margin | gbm | 10.119 |
| mkt | season | total | gbm | 10.913 |
| mkt | season | margin | stack | 9.983 |
| mkt | season | total | stack | 10.673 |
| mkt | month | margin | base | 9.946 |
| mkt | month | total | base | 10.601 |
| mkt | month | margin | ridge | 10.089 |
| mkt | month | total | ridge | 10.797 |
| mkt | month | margin | gbm | 10.269 |
| mkt | month | total | gbm | 10.995 |
| mkt | month | margin | stack | 10.035 |
| mkt | month | total | stack | 10.671 |
| nomkt | season | margin | base | 10.162 |
| nomkt | season | total | base | 10.992 |
| nomkt | season | margin | ridge | 10.208 |
| nomkt | season | total | ridge | 10.893 |
| nomkt | season | margin | gbm | 10.445 |
| nomkt | season | total | gbm | 11.146 |
| nomkt | season | margin | stack | 10.141 |
| nomkt | season | total | stack | 10.899 |

Monthly vs season refits on 2021+ (market variant, informational, not used for selection):

| refit | model | acc% | logloss | MAE margin |
|---|---|---|---|---|
| season | base | 66.2884 | 0.6122 | 9.7764 |
| season | lr | 65.8879 | 0.6179 | 9.8842 |
| season | gbm | 65.7543 | 0.6191 | 9.8458 |
| season | stack | 66.4887 | 0.6128 | 9.7784 |
| month | base | 66.2884 | 0.6122 | 9.7764 |
| month | lr | 66.3551 | 0.6178 | 9.8852 |
| month | gbm | 66.0214 | 0.6183 | 9.8123 |
| month | stack | 66.2216 | 0.6125 | 9.7816 |

Calibration of P(home win), 2021+ all games:

| bin | n | mean_pred | observed |
|---|---|---|---|
| (0.0, 0.1] | 2 | 0.083 | 0.500 |
| (0.1, 0.2] | 44 | 0.158 | 0.227 |
| (0.2, 0.3] | 109 | 0.250 | 0.229 |
| (0.3, 0.4] | 205 | 0.353 | 0.341 |
| (0.4, 0.5] | 222 | 0.446 | 0.437 |
| (0.5, 0.6] | 262 | 0.559 | 0.553 |
| (0.6, 0.7] | 276 | 0.646 | 0.627 |
| (0.7, 0.8] | 240 | 0.751 | 0.713 |
| (0.8, 0.9] | 121 | 0.847 | 0.893 |
| (0.9, 1.0] | 17 | 0.918 | 1.000 |

## 2. Final-score accuracy

| games | n | MAE home | MAE away | MAE margin | MAE total | MAE margin (final int score) | MAE total (final int score) | winner & margin within 3% | winner & margin within 7% | exact score% | mkt MAE margin | mkt MAE total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021+ with line | 1503 | 7.413 | 7.092 | 9.763 | 10.329 | 9.773 | 10.343 | 24.551 | 38.856 | 0.200 | 9.763 | 10.329 |
| 2021+ all games | 1503 | 7.413 | 7.092 | 9.763 | 10.329 | 9.773 | 10.343 | 24.551 | 38.856 | 0.200 | 9.763 | 10.329 |

Final predicted score = rounded predicted total and margin, with the margin forced to at least 1 point toward the predicted winner (so the score always agrees with the pick).

## 3. Win rate by confidence tier

Cells are `n / win%` per season. Pushes excluded. `break-even%` = moneyline: the average vigged implied probability of the picked side's closing price; spread/total: -110.

### (a) Moneyline winner (games with a closing line)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 27 / 44% | 42 / 40% | 36 / 56% | 37 / 54% | 37 / 49% | 10 / 60% | 189 | 49.2 | 54.5 |
| 55-60 | 48 / 44% | 51 / 71% | 72 / 61% | 59 / 59% | 51 / 63% | 14 / 57% | 295 | 59.7 | 59.4 |
| 60-65 | 56 / 55% | 57 / 54% | 50 / 78% | 48 / 77% | 44 / 57% | 17 / 59% | 272 | 63.6 | 64.2 |
| 65-70 | 29 / 59% | 40 / 65% | 41 / 59% | 44 / 80% | 43 / 58% | 12 / 67% | 209 | 64.6 | 69.4 |
| 70-75 | 36 / 61% | 33 / 70% | 18 / 67% | 37 / 76% | 26 / 81% | 10 / 90% | 160 | 71.9 | 74.2 |
| 75-80 | 41 / 85% | 32 / 88% | 32 / 69% | 36 / 69% | 39 / 67% | 9 / 44% | 189 | 74.1 | 79.2 |
| 80-85 | 27 / 81% | 13 / 85% | 17 / 76% | 13 / 85% | 18 / 83% | 4 / 75% | 92 | 81.5 | 83.9 |
| 85+ | 20 / 85% | 14 / 93% | 19 / 89% | 11 / 100% | 26 / 96% | 2 / 100% | 92 | 92.4 | 90.3 |

**Best tier with >=100 games 2021+: 75-80% -> 74.1% (n=189).**

Cumulative (P(pick) at or above threshold): >=0.50: 66.2% (n=1498); >=0.55: 68.7% (n=1309); >=0.60: 71.3% (n=1014); >=0.65: 74.1% (n=742); >=0.70: 77.9% (n=533); >=0.75: 80.4% (n=373); >=0.80: 87.0% (n=184); >=0.85: 92.4% (n=92); >=0.90: 94.7% (n=19)

### (b) Spread pick (P(cover) from predicted margin distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 281 / 52% | 274 / 51% | 271 / 51% | 281 / 49% | 284 / 53% | 73 / 44% | 1464 | 50.8 | 52.4 |
| 55-60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 60-65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 50.8% (n=1464).**

Cumulative (P(pick) at or above threshold): >=0.50: 50.8% (n=1464)

### (c) Total pick (P(over/under) from predicted total distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 282 / 46% | 281 / 47% | 281 / 50% | 282 / 47% | 285 / 54% | 79 / 51% | 1490 | 48.9 | 52.4 |
| 55-60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 60-65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 48.9% (n=1490).**

Cumulative (P(pick) at or above threshold): >=0.50: 48.9% (n=1490)

## 4. Daily top-N confidence parlays (moneyline picks)

Each date with at least N games, the N most confident model picks parlayed; 2021+. `model-expected%` = mean product of the model's pick probabilities (a calibration check); the last column parlays the N biggest market favourites instead.

Games with a line:

| top-N | days | hit% | model-expected% | market-fav top-N hit% |
|---|---|---|---|---|
| 1 | 331 | 74.0 | 73.1 | 74.0 |
| 2 | 157 | 61.8 | 59.8 | 61.8 |
| 3 | 110 | 46.4 | 49.3 | 46.4 |
| 4 | 95 | 41.1 | 37.7 | 41.1 |
| 5 | 95 | 32.6 | 26.6 | 32.6 |

All games (incl. no-line games):

| top-N | days | hit% | model-expected% |
|---|---|---|---|
| 1 | 331 | 74.0 | 73.1 |
| 2 | 157 | 61.8 | 59.8 |
| 3 | 110 | 46.4 | 49.3 |
| 4 | 95 | 41.1 | 37.7 |
| 5 | 95 | 32.6 | 26.6 |

## 5. The formula in plain English

Compact version (OLS / logistic on all out-of-sample 2018+ games with a line; this is what the full model boils down to):

- **predicted home margin = -0.13 + 1.038 x (market margin = -closing spread) + 0.070 x (script margin - market margin)**
- **predicted total = +0.82 + 0.996 x closing total + -0.146 x (script total - closing total)**
- **logit P(home win) = -0.050 + 0.947 x logit(de-vigged market P(home)) + -0.187 x (script logit - market logit)**
- home score = (total + margin) / 2, away score = (total - margin) / 2.

Stack weights of the deployed model (fit on all out-of-sample seasons):

- logit P(home) = -0.051 + -0.046 x logit(logistic) + 0.036 x logit(LightGBM-from-market) + 1.022 x logit(market)
- margin = -0.07 + -0.057 x ridge + 0.201 x LightGBM + 0.873 x market margin
- total = +0.24 + 0.247 x ridge + -0.064 x LightGBM + 0.823 x market total

Top features, Logistic P(home win) (deployed fit; log-odds):

| feature | per 1 sd | per raw unit |
|---|---|---|
| script_vs_mkt | -0.121 | -0.041 |
| margin_b | 0.098 | 0.016 |
| mkt_logit | 0.098 | 0.113 |
| p_home_blend | 0.098 | 0.614 |
| mkt_margin | 0.096 | 0.016 |
| margin_b_base | 0.096 | 0.016 |
| home_d_pres | 0.064 | 1.911 |
| home_o_prate | -0.052 | -1.311 |
| home_o_epa_pass | 0.052 | 0.515 |
| away_d_man | 0.046 | 0.651 |
| away_d_blitz | 0.044 | 1.135 |
| mu_away_base | -0.041 | -0.013 |

Top features, Ridge margin (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| mkt_logit | 0.595 | 0.682 |
| mkt_margin | 0.561 | 0.093 |
| margin_b_base | 0.559 | 0.093 |
| script_vs_mkt | -0.556 | -0.188 |
| margin_b | 0.556 | 0.093 |
| p_home_blend | 0.547 | 3.443 |
| margin | 0.301 | 0.052 |
| script_logit | 0.300 | 0.436 |
| p_home_script | 0.297 | 1.947 |
| mu_away_base | -0.258 | -0.081 |
| mu_away | -0.240 | -0.063 |
| home_o_epa_pass | 0.240 | 2.394 |

Top features, Ridge total (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| total_line | 0.935 | 0.211 |
| total_b_base | 0.926 | 0.210 |
| total_vs_line | -0.900 | -0.325 |
| total_b | 0.892 | 0.202 |
| wind | -0.690 | -0.166 |
| home_o_prate | -0.656 | -16.614 |
| home_o_heavy | -0.429 | -3.954 |
| roof_dome | 0.400 | 0.883 |
| away_d_blitz | -0.399 | -10.280 |
| away_plays_s | 0.387 | 0.174 |
| away_o_proe | 0.386 | 9.989 |
| home_d_box | 0.381 | 2.390 |

LightGBM (boosted from the market logit) top gain features: home_d_pres (303), away_o_shotgun (258), away_m_pass (207), home_d_box (196), home_o_shotgun (193), home_m_run (193), script_vs_mkt (167), away_d_pres (161), away_d_epa_rush (143), total_b (142)

## 6. Ablation (what each feature group adds)

Logistic + ridge, season refits, games with a line, 2021+. Groups: market = mkt_logit, mkt_margin, total_line, script_vs_mkt, total_vs_line, has_ml; blend = script-x-market blends (5 cols); script = 58 Kalman/enrichment cols; context = 13 cols (neutral, games_home, games_away, week, wind, temp, div_game, home_rest, away_rest, home_qb_delta, away_qb_delta, roof_dome...).

| features | n_feat | SU acc% | logloss | MAE margin | MAE total |
|---|---|---|---|---|---|
| full (all groups) | 82 | 65.8879 | 0.6179 | 9.8679 | 10.3832 |
| - market (no lines) | 71 | 63.8852 | 0.6343 | 10.0927 | 10.5851 |
| - blend | 77 | 65.8211 | 0.6197 | 9.8977 | 10.3999 |
| - script | 24 | 66.3551 | 0.6150 | 9.7951 | 10.3806 |
| - context | 69 | 66.6889 | 0.6161 | 9.8524 | 10.3863 |
| market only | 6 | 66.3551 | 0.6120 | 9.7778 | 10.3414 |
| market + script | 64 | 66.4887 | 0.6180 | 9.8582 | 10.4117 |

## 7. Does more history help?

Evaluation fixed at 2023+ seasons (games with a line). The chosen base models are retrained with only the last k seasons (that have closing lines) before each test season (or all prior seasons, optionally with exponential recency weights by season), then combined with that season's stack layer. Win model shown: **stack**; margin **stack**, total **stack** (when the deployed choice is the bare market line, the learned stack/ridge is shown instead, since the market line itself has no training window). The high-confidence column uses the best moneyline tier's lower bound from section 3.

| training window | SU win% | logloss | MAE margin | MAE total | P(pick)>=0.75 n | P(pick)>=0.75 win% |
|---|---|---|---|---|---|---|
| last 1 | 67.5966 | 0.6111 | 9.8263 | 10.2340 | 218 | 77.5229 |
| last 2 | 67.7039 | 0.6114 | 9.8127 | 10.1838 | 219 | 77.1689 |
| last 4 | 67.2747 | 0.6113 | 9.7825 | 10.1843 | 217 | 76.9585 |
| last 6 | 67.3820 | 0.6116 | 9.7911 | 10.1726 | 220 | 77.2727 |
| all prior | 67.4893 | 0.6116 | 9.7900 | 10.1640 | 217 | 77.4194 |
| all, decay half-life 1 seasons | 67.5966 | 0.6112 | 9.8050 | 10.1784 | 217 | 77.4194 |
| all, decay half-life 2 seasons | 67.3820 | 0.6114 | 9.7984 | 10.1732 | 219 | 77.1689 |
| all, decay half-life 4 seasons | 67.2747 | 0.6115 | 9.7935 | 10.1623 | 218 | 77.9817 |

Read: win% last-1 67.60 -> all 67.49; log loss 0.6111 -> 0.6116; margin MAE 9.826 -> 9.790. Win% moves only 0.43 pts across windows (noise level). Log loss is within 0.001 of its best by 'last 1' (last-1 0.6111 vs all 0.6116); margin MAE is within 0.01 of its best by 'last 4' (last-1 9.826 vs all 9.790). It plateaus: beyond that, more seasons add nothing measurable. Recency weighting: best decay = all, decay half-life 1 seasons (log loss 0.6112, MAE margin 9.805) vs unweighted all-prior 0.6116 / 9.790: no gain. The model is dominated by the closing line, which already embodies the latest information, so the learned layer has little to gain from extra history.

## 8. Live use

`from engine.predict.nfl import predict; predict(slate_df)` with rows shaped like `results/scripts/nfl_games.parquet` (the script engine's pre-game columns for upcoming games) plus `spread_home`, `total_line`, `home_ml`, `away_ml` (and `spread_open_home`/`total_open` if known). Rows without `spread_home` use the script-only model. Model: `results/predict/nfl_model.pkl` (trained through 2026-10-12). Required script columns: away_base_epa_pass, away_d_blitz, away_d_box, away_d_epa_pass, away_d_epa_rush, away_d_man, away_d_pres, away_m_pass, away_m_run, away_o_epa_pass, away_o_epa_rush, away_o_heavy, away_o_p11, away_o_prate, away_o_proe, away_o_shotgun, away_o_spp, away_pass_att_s, away_pass_yds_s, away_plays_s, away_qb_delta, away_rest, away_rush_att_s, div_game, games_away, games_home, home_base_epa_pass, home_d_blitz, home_d_box, home_d_epa_pass, home_d_epa_rush, home_d_man, home_d_pres, home_m_pass, home_m_run, home_o_epa_pass, home_o_epa_rush, home_o_heavy, home_o_p11, home_o_prate, home_o_proe, home_o_shotgun, home_o_spp, home_pass_att_s, home_pass_yds_s, home_plays_s, home_qb_delta, home_rest, home_rush_att_s, is_post, margin, margin_b, margin_b_base, margin_base, mu_away, mu_away_base, mu_home, mu_home_base, neutral, p_home_blend, p_home_script, rating_unc, rho, roof_dome, script_logit, sd_away, sd_home, sd_margin, sd_total, temp, total, total_b, total_b_base, total_base, week, wind.
