# NBA game predictor (win% first)

Code: `engine/predict/nba.py` -> `engine/predict/common_team.py`. Inputs: `results/scripts/nba_games.parquet`. Runtime 4.5 min. Strict walk-forward: for each season S, models are fit only on earlier games (season < S, or date < first of the month when monthly refits were chosen); the stack/calibration layer for season S is fit only on the out-of-sample predictions of seasons 2017..S-1. Candidates and refit cadence were picked on 2018-2020; **everything below is 2021+ unless stated**. Ties (NFL) are excluded from win% tables.

## 0. Bottom line

- Chosen: market-variant win model **stack** (month refits), margin **base**, total **base**; no-line games use the script-only variant (win base, margin base, total base). Spread / total picks use the best learned (non-market) margin / total model, **stack / stack**, because the bare market line gives P(cover) = P(over) = 50% by construction. `base` = the closing line itself (de-vigged ML, -spread, total).
- 2021+ straight-up winner, games with a closing line (n=7708): **model 67.55%** vs market favourite 67.75% vs script alone 65.81%. Agreement model = market favourite on 98.9% of games.
- All 2021+ games incl. those without a line (n=7776): model 67.50%, script 65.78%.
- Log loss (lines games): model 0.5997 vs market 0.5994 vs script 0.6205.

## 1. Per-season learning curve (straight-up winner, log loss, score MAE)

Games with a closing line. `model` = final pick; `mkt` = de-vigged closing favourite; `script` = Kalman script alone. MAE margin/total: model vs closing spread/total vs script.

| season | n_line | acc_model | acc_mkt | acc_script | ll_model | ll_mkt | ll_script | mae_m_model | mae_m_mkt | mae_m_script | mae_t_model | mae_t_mkt | mae_t_script | n_all | acc_all_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 1311 | 68.50 | 68.73 | 64.61 | 0.59 | 0.59 | 0.62 | 9.52 | 9.52 | 9.95 | 14.35 | 14.35 | 14.80 | 1311 | 68.50 |
| 2019 | 1311 | 66.90 | 67.05 | 65.83 | 0.60 | 0.60 | 0.61 | 9.94 | 9.94 | 10.20 | 14.50 | 14.50 | 15.00 | 1311 | 66.90 |
| 2020 | 1143 | 66.58 | 66.49 | 64.22 | 0.61 | 0.61 | 0.63 | 10.18 | 10.18 | 10.27 | 14.91 | 14.91 | 15.21 | 1143 | 66.58 |
| 2021 | 1171 | 65.24 | 65.58 | 63.36 | 0.62 | 0.62 | 0.64 | 10.75 | 10.75 | 11.15 | 14.46 | 14.46 | 14.74 | 1171 | 65.24 |
| 2022 | 1323 | 67.50 | 67.65 | 66.06 | 0.61 | 0.61 | 0.63 | 10.71 | 10.71 | 11.12 | 14.15 | 14.15 | 14.68 | 1323 | 67.50 |
| 2023 | 1320 | 65.91 | 66.21 | 64.47 | 0.63 | 0.63 | 0.65 | 9.81 | 9.81 | 10.22 | 14.03 | 14.03 | 14.55 | 1320 | 65.91 |
| 2024 | 1253 | 69.11 | 69.27 | 67.12 | 0.58 | 0.58 | 0.61 | 10.54 | 10.54 | 10.93 | 14.05 | 14.05 | 14.52 | 1319 | 68.69 |
| 2025 | 1319 | 69.29 | 69.14 | 65.88 | 0.59 | 0.59 | 0.60 | 10.65 | 10.65 | 11.01 | 14.11 | 14.11 | 14.65 | 1321 | 69.34 |
| 2026 | 1322 | 68.08 | 68.46 | 67.78 | 0.58 | 0.58 | 0.60 | 10.97 | 10.97 | 11.37 | 14.50 | 14.50 | 14.88 | 1322 | 68.08 |

Refit cadence and candidate comparison on the selection window 2018-2020 (market variant = games with a line; no-market variant = all games):

| variant | refit | target | model | logloss | acc% |
|---|---|---|---|---|---|
| mkt | season | win | base | 0.6003 | 67.4635 |
| mkt | season | win | lr | 0.6239 | 66.4807 |
| mkt | season | win | gbm | 0.6081 | 66.5073 |
| mkt | season | win | stack | 0.5995 | 67.4104 |
| mkt | month | win | base | 0.6003 | 67.4635 |
| mkt | month | win | lr | 0.6359 | 66.6932 |
| mkt | month | win | gbm | 0.6071 | 67.0120 |
| mkt | month | win | stack | 0.5995 | 67.3572 |
| nomkt | season | win | base | 0.6194 | 64.9137 |
| nomkt | season | win | lr | 0.6257 | 64.3028 |
| nomkt | season | win | gbm | 0.6283 | 64.4622 |
| nomkt | season | win | stack | 0.6202 | 65.3918 |

| variant | refit | target | model | MAE |
|---|---|---|---|---|
| mkt | season | margin | base | 9.866 |
| mkt | season | total | base | 14.569 |
| mkt | season | margin | ridge | 9.953 |
| mkt | season | total | ridge | 15.291 |
| mkt | season | margin | gbm | 9.979 |
| mkt | season | total | gbm | 14.585 |
| mkt | season | margin | stack | 9.861 |
| mkt | season | total | stack | 14.603 |
| mkt | month | margin | base | 9.866 |
| mkt | month | total | base | 14.569 |
| mkt | month | margin | ridge | 9.989 |
| mkt | month | total | ridge | 14.802 |
| mkt | month | margin | gbm | 9.968 |
| mkt | month | total | gbm | 14.620 |
| mkt | month | margin | stack | 9.872 |
| mkt | month | total | stack | 14.597 |
| nomkt | season | margin | base | 10.136 |
| nomkt | season | total | base | 14.992 |
| nomkt | season | margin | ridge | 10.221 |
| nomkt | season | total | ridge | 15.266 |
| nomkt | season | margin | gbm | 10.252 |
| nomkt | season | total | gbm | 15.025 |
| nomkt | season | margin | stack | 10.143 |
| nomkt | season | total | stack | 15.088 |

Monthly vs season refits on 2021+ (market variant, informational, not used for selection):

| refit | model | acc% | logloss | MAE margin |
|---|---|---|---|---|
| season | base | 67.7478 | 0.5994 | 10.5701 |
| season | lr | 67.1251 | 0.6051 | 10.6324 |
| season | gbm | 66.9434 | 0.6066 | 10.7459 |
| season | stack | 67.6959 | 0.5996 | 10.5776 |
| month | base | 67.7478 | 0.5994 | 10.5701 |
| month | lr | 67.5921 | 0.6015 | 10.6095 |
| month | gbm | 66.9694 | 0.6057 | 10.6843 |
| month | stack | 67.5532 | 0.5997 | 10.5776 |

Calibration of P(home win), 2021+ all games:

| bin | n | mean_pred | observed |
|---|---|---|---|
| (0.0, 0.1] | 33 | 0.083 | 0.030 |
| (0.1, 0.2] | 289 | 0.158 | 0.145 |
| (0.2, 0.3] | 549 | 0.255 | 0.244 |
| (0.3, 0.4] | 936 | 0.353 | 0.340 |
| (0.4, 0.5] | 1184 | 0.449 | 0.448 |
| (0.5, 0.6] | 1392 | 0.555 | 0.561 |
| (0.6, 0.7] | 1408 | 0.649 | 0.657 |
| (0.7, 0.8] | 1109 | 0.746 | 0.744 |
| (0.8, 0.9] | 755 | 0.844 | 0.845 |
| (0.9, 1.0] | 121 | 0.920 | 0.942 |

## 2. Final-score accuracy

| games | n | MAE home | MAE away | MAE margin | MAE total | MAE margin (final int score) | MAE total (final int score) | winner & margin within 3% | winner & margin within 6% | exact score% | mkt MAE margin | mkt MAE total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021+ with line | 7708 | 8.955 | 8.915 | 10.570 | 14.212 | 10.579 | 14.221 | 21.938 | 35.431 | 0.182 | 10.570 | 14.212 |
| 2021+ all games | 7776 | 8.960 | 8.928 | 10.585 | 14.213 | 10.595 | 14.222 | 21.914 | 35.378 | 0.180 |  |  |
| 2021+ no line (script variant) | 68 | 9.468 | 10.345 | 12.301 | 14.312 | 12.338 | 14.368 | 19.118 | 29.412 | 0.000 |  |  |

Final predicted score = rounded predicted total and margin, with the margin forced to at least 1 point toward the predicted winner (so the score always agrees with the pick).

## 3. Win rate by confidence tier

Cells are `n / win%` per season. Pushes excluded. `break-even%` = moneyline: the average vigged implied probability of the picked side's closing price; spread/total: -110.

### (a) Moneyline winner (games with a closing line)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 179 / 48% | 207 / 55% | 226 / 55% | 191 / 52% | 166 / 54% | 181 / 42% | 1150 | 51.2 | 55.1 |
| 55-60 | 217 / 62% | 258 / 58% | 268 / 58% | 234 / 59% | 194 / 61% | 224 / 59% | 1395 | 59.4 | 60.6 |
| 60-65 | 216 / 64% | 223 / 68% | 223 / 66% | 172 / 59% | 207 / 63% | 182 / 61% | 1223 | 63.9 | 66.0 |
| 65-70 | 170 / 67% | 182 / 68% | 178 / 67% | 161 / 70% | 225 / 66% | 184 / 72% | 1100 | 68.1 | 71.6 |
| 70-75 | 146 / 60% | 156 / 71% | 165 / 73% | 155 / 79% | 158 / 75% | 161 / 77% | 941 | 72.5 | 76.9 |
| 75-80 | 94 / 82% | 119 / 73% | 108 / 74% | 128 / 86% | 123 / 73% | 132 / 78% | 704 | 77.7 | 81.8 |
| 80-85 | 77 / 79% | 86 / 85% | 100 / 81% | 105 / 84% | 134 / 84% | 116 / 78% | 618 | 81.9 | 86.9 |
| 85+ | 72 / 92% | 92 / 90% | 52 / 81% | 107 / 88% | 112 / 95% | 142 / 93% | 577 | 90.6 | 92.7 |

**Best tier with >=100 games 2021+: 85+% -> 90.6% (n=577).**

Cumulative (P(pick) at or above threshold): >=0.50: 67.6% (n=7708); >=0.55: 70.4% (n=6558); >=0.60: 73.4% (n=5163); >=0.65: 76.3% (n=3940); >=0.70: 79.5% (n=2840); >=0.75: 83.0% (n=1899); >=0.80: 86.1% (n=1195); >=0.85: 90.6% (n=577); >=0.90: 94.8% (n=154)

### (b) Spread pick (P(cover) from predicted margin distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 1111 / 49% | 1299 / 52% | 1302 / 50% | 1237 / 50% | 1293 / 52% | 1317 / 51% | 7559 | 50.8 | 52.4 |
| 55-60 | 47 / 47% | 2 / 0% | 1 / 0% | 1 / 0% | 1 / 0% | 1 / 100% | 53 | 43.4 | 52.4 |
| 60-65 | 0 | 0 | 1 / 0% | 0 | 0 | 0 | 1 | 0.0 | 52.4 |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 50.8% (n=7559).**

Cumulative (P(pick) at or above threshold): >=0.50: 50.7% (n=7613); >=0.55: 42.6% (n=54); >=0.60: 0.0% (n=1)

### (c) Total pick (P(over/under) from predicted total distribution)

| tier | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | n | win% | break-even% |
|---|---|---|---|---|---|---|---|---|---|
| 50-55 | 1155 / 49% | 1300 / 52% | 1309 / 53% | 1240 / 49% | 1305 / 52% | 1322 / 50% | 7631 | 50.8 | 52.4 |
| 55-60 | 0 | 0 | 0 | 1 / 0% | 0 | 0 | 1 | 0.0 | 52.4 |
| 60-65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 65-70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 70-75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 75-80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 80-85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| 85+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

**Best tier with >=100 games 2021+: 50-55% -> 50.8% (n=7631).**

Cumulative (P(pick) at or above threshold): >=0.50: 50.8% (n=7632); >=0.55: 0.0% (n=1)

## 4. Daily top-N confidence parlays (moneyline picks)

Each date with at least N games, the N most confident model picks parlayed; 2021+. `model-expected%` = mean product of the model's pick probabilities (a calibration check); the last column parlays the N biggest market favourites instead.

Games with a line:

| top-N | days | hit% | model-expected% | market-fav top-N hit% |
|---|---|---|---|---|
| 1 | 1214 | 78.6 | 79.0 | 78.8 |
| 2 | 1094 | 62.2 | 60.2 | 62.0 |
| 3 | 973 | 48.9 | 44.7 | 48.9 |
| 4 | 898 | 34.0 | 31.3 | 34.1 |
| 5 | 813 | 24.1 | 21.4 | 24.4 |

All games (incl. no-line games):

| top-N | days | hit% | model-expected% |
|---|---|---|---|
| 1 | 1250 | 78.2 | 78.7 |
| 2 | 1110 | 61.9 | 59.9 |
| 3 | 977 | 48.8 | 44.6 |
| 4 | 899 | 33.9 | 31.3 |
| 5 | 814 | 24.1 | 21.4 |

## 5. The formula in plain English

Compact version (OLS / logistic on all out-of-sample 2018+ games with a line; this is what the full model boils down to):

- **predicted home margin = -0.16 + 1.008 x (market margin = -closing spread) + 0.090 x (script margin - market margin)**
- **predicted total = +4.15 + 0.983 x closing total + -0.103 x (script total - closing total)**
- **logit P(home win) = -0.008 + 0.926 x logit(de-vigged market P(home)) + 0.024 x (script logit - market logit)**
- home score = (total + margin) / 2, away score = (total - margin) / 2.

Stack weights of the deployed model (fit on all out-of-sample seasons):

- logit P(home) = -0.007 + -0.051 x logit(logistic) + -0.067 x logit(LightGBM-from-market) + 1.012 x logit(market)
- margin = -0.13 + 0.296 x ridge + -0.064 x LightGBM + 0.775 x market margin
- total = +2.10 + -0.025 x ridge + 0.143 x LightGBM + 0.874 x market total

Top features, Logistic P(home win) (deployed fit; log-odds):

| feature | per 1 sd | per raw unit |
|---|---|---|
| mkt_logit | 0.256 | 0.241 |
| script_vs_mkt | -0.134 | -0.039 |
| mkt_margin | 0.123 | 0.018 |
| margin_b | 0.114 | 0.017 |
| margin_mx | 0.099 | 0.014 |
| script_logit | 0.064 | 0.079 |
| margin | 0.061 | 0.009 |
| margin_core | 0.056 | 0.009 |
| drtg_home_e | -0.048 | -0.011 |
| pts_poss_away | -0.036 | -0.005 |
| mu_home | 0.035 | 0.006 |
| ortg_home_e | 0.035 | 0.007 |

Top features, Ridge margin (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| script_logit | -5.145 | -6.393 |
| margin_mx | -5.145 | -0.740 |
| mkt_logit | 4.350 | 4.086 |
| mkt_margin | 3.383 | 0.486 |
| margin | 2.794 | 0.423 |
| margin_core | 2.626 | 0.424 |
| missing(spread_move) | 2.445 | 5.308 |
| missing(total_move) | -2.247 | -4.874 |
| total_mx | 1.831 | 0.153 |
| mu_away | -1.538 | -0.231 |
| sd_total | -1.527 | -3.093 |
| script_vs_mkt | -1.488 | -0.437 |

Top features, Ridge total (deployed fit; points):

| feature | per 1 sd | per raw unit |
|---|---|---|
| total_vs_line | -2.350 | -0.572 |
| total_line | 1.675 | 0.142 |
| total_b | 1.669 | 0.141 |
| total_mx | 1.480 | 0.123 |
| total_core | 0.908 | 0.083 |
| total | 0.897 | 0.080 |
| mu_home_core | 0.808 | 0.130 |
| mu_away | 0.780 | 0.117 |
| mu_away_core | 0.774 | 0.121 |
| mu_home | 0.767 | 0.120 |
| poss_pred | 0.761 | 0.213 |
| playoff | -0.752 | -3.132 |

LightGBM (boosted from the market logit) top gain features: corr_away (1044), m_3pa_home (1012), m_3pct_away (923), f3r_away_e (889), tov_home_e (863), total_vs_line (855), pace_home_e (852), orb_home_e (835), m_3pct_home (789), m_3pa_away (754)

## 6. Ablation (what each feature group adds)

Logistic + ridge, season refits, games with a line, 2021+. Groups: market = mkt_logit, mkt_margin, total_line, spread_move, total_move, script_vs_mkt, total_vs_line, has_ml; blend = script-x-market blends (4 cols); script = 35 Kalman/enrichment cols; context = 16 cols (neutral, playoff, games_home, games_away, rest_home, b2b_home, travel_home_kkm, tz_home, miss_min_home, miss_st_home, rest_away, b2b_away...).

| features | n_feat | SU acc% | logloss | MAE margin | MAE total |
|---|---|---|---|---|---|
| full (all groups) | 63 | 67.1251 | 0.6051 | 10.6324 | 14.2617 |
| - market (no lines) | 51 | 65.3736 | 0.6210 | 10.9857 | 14.6912 |
| - blend | 59 | 67.1121 | 0.6049 | 10.6586 | 14.2644 |
| - script | 28 | 67.5013 | 0.6043 | 10.6417 | 14.2847 |
| - context | 47 | 67.2159 | 0.6044 | 10.6306 | 14.2439 |
| market only | 8 | 67.6440 | 0.6017 | 10.6206 | 14.2142 |
| market + script | 43 | 67.3456 | 0.6042 | 10.6462 | 14.2474 |

## 7. Does more history help?

Evaluation fixed at 2023+ seasons (games with a line). The chosen base models are retrained with only the last k seasons (that have closing lines) before each test season (or all prior seasons, optionally with exponential recency weights by season), then combined with that season's stack layer. Win model shown: **stack**; margin **stack**, total **stack** (when the deployed choice is the bare market line, the learned stack/ridge is shown instead, since the market line itself has no training window). The high-confidence column uses the best moneyline tier's lower bound from section 3.

| training window | SU win% | logloss | MAE margin | MAE total | P(pick)>=0.85 n | P(pick)>=0.85 win% |
|---|---|---|---|---|---|---|
| last 1 | 68.3353 | 0.5933 | 10.6114 | 14.1670 | 385 | 90.9091 |
| last 2 | 68.2394 | 0.5931 | 10.5519 | 14.1662 | 392 | 90.0510 |
| last 4 | 68.1243 | 0.5931 | 10.5034 | 14.1704 | 411 | 90.5109 |
| last 6 | 68.2585 | 0.5935 | 10.5042 | 14.1601 | 416 | 90.8654 |
| all prior | 68.1626 | 0.5934 | 10.5014 | 14.1639 | 417 | 90.4077 |
| all, decay half-life 1 seasons | 68.3161 | 0.5933 | 10.5202 | 14.1608 | 402 | 90.7960 |
| all, decay half-life 2 seasons | 68.1626 | 0.5933 | 10.5108 | 14.1660 | 414 | 90.0966 |
| all, decay half-life 4 seasons | 68.1243 | 0.5934 | 10.5021 | 14.1653 | 407 | 90.4177 |

Read: win% last-1 68.34 -> all 68.16; log loss 0.5933 -> 0.5934; margin MAE 10.611 -> 10.501. Win% moves only 0.21 pts across windows (noise level). Log loss is within 0.001 of its best by 'last 1' (last-1 0.5933 vs all 0.5934); margin MAE is within 0.01 of its best by 'last 4' (last-1 10.611 vs all 10.501). It plateaus: beyond that, more seasons add nothing measurable. Recency weighting: best decay = all, decay half-life 1 seasons (log loss 0.5933, MAE margin 10.520) vs unweighted all-prior 0.5934 / 10.501: no gain. The model is dominated by the closing line, which already embodies the latest information, so the learned layer has little to gain from extra history.

## 8. Live use

`from engine.predict.nba import predict; predict(slate_df)` with rows shaped like `results/scripts/nba_games.parquet` (the script engine's pre-game columns for upcoming games) plus `spread_home`, `total_line`, `home_ml`, `away_ml` (and `spread_open_home`/`total_open` if known). Rows without `spread_home` use the script-only model. Model: `results/predict/nba_model.pkl` (trained through 2026-06-13). Required script columns: b2b_away, b2b_home, corr_away, corr_home, drtg_away_e, drtg_home_e, f3r_away_e, f3r_home_e, ftr_home_e, games_away, games_home, m_3pa_away, m_3pa_home, m_3pct_away, m_3pct_home, margin, margin_b, margin_core, margin_mx, miss_min_away, miss_min_home, miss_st_away, miss_st_home, mu_away, mu_away_core, mu_home, mu_home_core, neutral, orb_home_e, ortg_away_e, ortg_home_e, pace_away_e, pace_clash, pace_home_e, playoff, poss_pred, pts_poss_away, pts_poss_home, rest_away, rest_home, rho, script_logit, sd_away, sd_home, sd_margin, sd_total, total, total_b, total_core, total_mx, tov_home_e, travel_away_kkm, travel_home_kkm, tz_away, tz_home.
