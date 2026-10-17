# NFL script-driven props + 3/5/7-leg parlays (walk-forward, report 2021+)
Code: `engine/props_script/nfl.py` (+ `common_hoops_football.py`). Runtime 16.9 min. Distributional LightGBM P(stat >= k) refit each season on the prior 4 seasons (script + tendencies), online calibrator per stat x threshold x probability bin, date by date. 2017-2020 fit/choose, 2021-2026 reported (2026 = weeks 1-5).

## (a) Real lines: hit rate by model confidence (side = higher calibrated prob)
One line per player-game-market (latest snapshot, sportsbook line closest to even odds, half-points). 2017-18 = completions only (propbets.football); 2026 = weeks 1-5, all five markets. Lines never used for training; the online calibrator (per market) learns from them date by date. **Small samples.**

**Season 2017** (n = 382)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 102 | 0.513 | 0.510 | 0.049 |
| 0.53-0.56 | 76 | 0.545 | 0.539 | 0.057 |
| 0.56-0.60 | 57 | 0.577 | 0.456 | 0.066 |
| 0.60-0.65 | 63 | 0.623 | 0.556 | 0.063 |
| 0.65-1.00 | 84 | 0.738 | 0.512 | 0.055 |
| >= 0.53 | 280 | 0.627 | 0.518 | 0.030 |
| >= 0.56 | 204 | 0.658 | 0.510 | 0.035 |
| >= 0.60 | 147 | 0.689 | 0.531 | 0.041 |
| >= 0.65 | 84 | 0.738 | 0.512 | 0.055 |

**Season 2018** (n = 322)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 92 | 0.514 | 0.554 | 0.052 |
| 0.53-0.56 | 53 | 0.545 | 0.509 | 0.069 |
| 0.56-0.60 | 50 | 0.576 | 0.520 | 0.071 |
| 0.60-0.65 | 69 | 0.623 | 0.580 | 0.059 |
| 0.65-1.00 | 58 | 0.727 | 0.638 | 0.063 |
| >= 0.53 | 230 | 0.621 | 0.565 | 0.033 |
| >= 0.56 | 177 | 0.644 | 0.582 | 0.037 |
| >= 0.60 | 127 | 0.670 | 0.606 | 0.043 |
| >= 0.65 | 58 | 0.727 | 0.638 | 0.063 |

**Season 2026** (n = 1,847)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 448 | 0.515 | 0.522 | 0.024 |
| 0.53-0.56 | 326 | 0.544 | 0.534 | 0.028 |
| 0.56-0.60 | 315 | 0.580 | 0.486 | 0.028 |
| 0.60-0.65 | 337 | 0.624 | 0.558 | 0.027 |
| 0.65-1.00 | 421 | 0.723 | 0.570 | 0.024 |
| >= 0.53 | 1399 | 0.625 | 0.540 | 0.013 |
| >= 0.56 | 1073 | 0.650 | 0.541 | 0.015 |
| >= 0.60 | 758 | 0.679 | 0.565 | 0.018 |
| >= 0.65 | 421 | 0.723 | 0.570 | 0.024 |

2026 by market (all picks; p >= 0.58):

| market | n | hit | n>=0.58 | hit>=0.58 |
|---|---|---|---|---|
| passing_yards | 87 | 0.471 | 36 | 0.500 |
| receiving_yards | 950 | 0.540 | 464 | 0.534 |
| receptions | 519 | 0.543 | 268 | 0.571 |
| rushing_yards | 291 | 0.526 | 148 | 0.547 |

Log loss on 2026 rows with both prices (n=1732): model-cal 0.7110, de-vigged market 0.6901.

## (b) Milestones, 2021-2026: realized hit rate by confidence tier vs naive base rates
`group_base` = hit rate of the same stat x threshold x position in prior seasons; `player_naive` = the player's own trailing (EWM, hl 6 games) frequency of reaching k.

| tier | n | pred | hit | group_base | player_naive | lift_vs_group | lift_vs_player | se |
|---|---|---|---|---|---|---|---|---|
| 0.00-0.70 | 227281 | 0.212 | 0.211 | 0.277 | 0.225 | -0.066 | -0.014 | 0.001 |
| 0.70-0.75 | 6139 | 0.724 | 0.724 | 0.477 | 0.752 | 0.247 | -0.028 | 0.006 |
| 0.75-0.80 | 6365 | 0.774 | 0.777 | 0.501 | 0.809 | 0.276 | -0.032 | 0.005 |
| 0.80-0.85 | 5101 | 0.824 | 0.828 | 0.519 | 0.850 | 0.309 | -0.021 | 0.005 |
| 0.85-0.90 | 5134 | 0.875 | 0.877 | 0.543 | 0.894 | 0.334 | -0.017 | 0.005 |
| 0.90-1.00 | 4990 | 0.933 | 0.929 | 0.578 | 0.947 | 0.352 | -0.017 | 0.004 |
| >= 0.70 | 27729 | 0.820 | 0.821 | 0.521 | 0.844 | 0.300 | -0.024 | 0.002 |
| >= 0.75 | 21590 | 0.847 | 0.848 | 0.533 | 0.871 | 0.315 | -0.023 | 0.002 |
| >= 0.80 | 15225 | 0.877 | 0.878 | 0.546 | 0.897 | 0.332 | -0.019 | 0.003 |
| >= 0.85 | 10124 | 0.904 | 0.903 | 0.560 | 0.920 | 0.343 | -0.017 | 0.003 |
| >= 0.90 | 4990 | 0.933 | 0.929 | 0.578 | 0.947 | 0.352 | -0.017 | 0.004 |

By stat x threshold, legs with p >= 0.75:

| stat | k | n | pred | hit | group_base | player_naive | all_legs_base(k) |
|---|---|---|---|---|---|---|---|
| passing_yards | 200 | 675 | 0.820 | 0.810 | 0.697 | 0.829 | 0.651 |
| passing_yards | 250 | 23 | 0.807 | 0.609 | 0.448 | 0.733 | 0.385 |
| receiving_yards | 25 | 3472 | 0.835 | 0.841 | 0.538 | 0.870 | 0.376 |
| receiving_yards | 40 | 793 | 0.808 | 0.796 | 0.390 | 0.843 | 0.243 |
| receiving_yards | 60 | 38 | 0.799 | 0.658 | 0.248 | 0.815 | 0.133 |
| receiving_yards | 80 | 5 | 0.791 | 0.400 | 0.147 | 0.707 | 0.071 |
| receptions | 2 | 8249 | 0.866 | 0.871 | 0.629 | 0.889 | 0.544 |
| receptions | 3 | 3616 | 0.828 | 0.832 | 0.477 | 0.862 | 0.376 |
| receptions | 4 | 1060 | 0.827 | 0.833 | 0.344 | 0.857 | 0.251 |
| receptions | 5 | 269 | 0.802 | 0.751 | 0.233 | 0.836 | 0.165 |
| rushing_yards | 25 | 2472 | 0.869 | 0.868 | 0.484 | 0.872 | 0.474 |
| rushing_yards | 40 | 795 | 0.826 | 0.815 | 0.348 | 0.833 | 0.341 |
| rushing_yards | 60 | 118 | 0.797 | 0.754 | 0.211 | 0.791 | 0.207 |
| rushing_yards | 80 | 5 | 0.780 | 1.000 | 0.117 | 0.675 | 0.117 |

## (c) Parlays, one ticket per game-day per construction, 2021-2026
Predicted = Gaussian-copula joint (correlations from prior seasons' residuals). `typical_price` = median ticket odds (real odds for real-line and ML legs; milestone legs priced at player naive rate + 4.5%).

| construction | legs | n_days | pred_joint | pred_indep | realized | se | typical_price |
|---|---|---|---|---|---|---|---|
| 1 top-N cross-game | 3 | 331 | 0.833 | 0.832 | 0.831 | 0.021 | -1454 |
| 1 top-N cross-game | 5 | 331 | 0.681 | 0.678 | 0.680 | 0.026 | -470 |
| 1 top-N cross-game | 7 | 331 | 0.526 | 0.520 | 0.535 | 0.027 | -196 |
| 1R top-N real lines | 3 | 16 | 0.505 | 0.500 | 0.250 | 0.108 | +559 |
| 1R top-N real lines | 5 | 16 | 0.288 | 0.281 | 0.062 | 0.061 | +2069 |
| 1R top-N real lines | 7 | 16 | 0.159 | 0.156 | 0.000 | 0.000 | +7645 |
| 2 same-game script stack | 3 | 331 | 0.820 | 0.818 | 0.825 | 0.021 | -1223 |
| 2 same-game script stack | 5 | 331 | 0.637 | 0.631 | 0.647 | 0.026 | -348 |
| 2 same-game script stack | 7 | 331 | 0.445 | 0.428 | 0.471 | 0.027 | -134 |
| 3 mixed 85%+ ML + props | 3 | 67 | 0.819 | 0.818 | 0.851 | 0.044 | -630 |
| 3 mixed 85%+ ML + props | 5 | 67 | 0.739 | 0.737 | 0.776 | 0.051 | -486 |
| 3 mixed 85%+ ML + props | 7 | 67 | 0.655 | 0.652 | 0.672 | 0.057 | -378 |
| 4 most-likely ticket | 3 | 331 | 0.833 | 0.832 | 0.837 | 0.020 | -1454 |
| 4 most-likely ticket | 5 | 331 | 0.683 | 0.678 | 0.689 | 0.025 | -484 |
| 4 most-likely ticket | 7 | 331 | 0.532 | 0.521 | 0.550 | 0.027 | -196 |

## (d) Which tendencies mattered (ablation; train 2017-2020, test all 2021-2026)

| config | ms_logloss | ms_ece | n>=0.8 | hit>=0.8 | lift_vs_player_naive>=0.8 | n>=0.9 | hit>=0.9 | top5_daily_hit |
|---|---|---|---|---|---|---|---|---|
| full model | 0.3959 | 0.0040 | 16580 | 0.8728 | -0.0174 | 6826 | 0.9183 | 0.6767 |
| full model, no online calibrator | 0.3966 | 0.0112 | 18904 | 0.8612 | -0.0183 | 7153 | 0.9154 | 0.6828 |
| drop game_script_blowout | 0.3969 | 0.0048 | 17018 | 0.8682 | -0.0198 | 6167 | 0.9181 | 0.6767 |
| drop teammates_out | 0.3983 | 0.0054 | 16587 | 0.8676 | -0.0202 | 6030 | 0.9207 | 0.7039 |
| drop opp_def_vs_pos | 0.3964 | 0.0054 | 16606 | 0.8700 | -0.0192 | 6374 | 0.9206 | 0.7130 |
| drop rest | 0.3961 | 0.0036 | 16508 | 0.8720 | -0.0185 | 6570 | 0.9196 | 0.6888 |
| drop home_away | 0.3968 | 0.0046 | 16617 | 0.8706 | -0.0190 | 6045 | 0.9219 | 0.6918 |
| drop weather | 0.3966 | 0.0039 | 16320 | 0.8728 | -0.0180 | 5843 | 0.9204 | 0.6888 |
| drop recent_form | 0.4001 | 0.0032 | 15889 | 0.8721 | -0.0240 | 5701 | 0.9221 | 0.6737 |
| drop player_bias_dispersion | 0.3956 | 0.0041 | 16569 | 0.8729 | -0.0176 | 6016 | 0.9205 | 0.6888 |
| script features only (LGBM) | 0.4025 | 0.0019 | 14638 | 0.8759 | -0.0247 | 5493 | 0.9214 | 0.6918 |
| script parametric (no LGBM) | 0.4020 | 0.0075 | 14127 | 0.8804 | -0.0255 | 4606 | 0.9292 | 0.6828 |

## (e) Learning curve by season

| season | n_legs | ll_script_param | ll_lgbm | ll_calibrated | ece_param | ece_lgbm | ece_cal | hit>=0.75 | pred>=0.75 | line_n | line_hit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2017 | 43472 | 0.4454 | 0.4381 | 0.4356 | 0.0410 | 0.0293 | 0.0170 | 0.8082 | 0.8563 | 382 | 0.5157 |
| 2018 | 42332 | 0.4497 | 0.4348 | 0.4363 | 0.0356 | 0.0156 | 0.0187 | 0.8658 | 0.8404 | 322 | 0.5621 |
| 2019 | 42184 | 0.4477 | 0.4297 | 0.4295 | 0.0384 | 0.0063 | 0.0068 | 0.8469 | 0.8458 | 0 |  |
| 2020 | 44050 | 0.4491 | 0.4317 | 0.4312 | 0.0368 | 0.0075 | 0.0048 | 0.8456 | 0.8421 | 0 |  |
| 2021 | 45848 | 0.4496 | 0.4318 | 0.4313 | 0.0387 | 0.0103 | 0.0058 | 0.8415 | 0.8480 | 0 |  |
| 2022 | 45620 | 0.4314 | 0.4151 | 0.4148 | 0.0382 | 0.0099 | 0.0072 | 0.8444 | 0.8458 | 0 |  |
| 2023 | 45572 | 0.4299 | 0.4134 | 0.4131 | 0.0353 | 0.0082 | 0.0065 | 0.8526 | 0.8462 | 0 |  |
| 2024 | 50900 | 0.3923 | 0.3763 | 0.3762 | 0.0286 | 0.0071 | 0.0085 | 0.8539 | 0.8475 | 0 |  |
| 2025 | 52590 | 0.3747 | 0.3579 | 0.3577 | 0.0330 | 0.0074 | 0.0046 | 0.8464 | 0.8466 | 0 |  |
| 2026 | 14480 | 0.3829 | 0.3650 | 0.3653 | 0.0322 | 0.0136 | 0.0145 | 0.8593 | 0.8462 | 1847 | 0.5355 |

Top features (gain share, 2026 model):

| feature | gain_share |
|---|---|
| zk | 0.6837 |
| rk | 0.0973 |
| snap_last | 0.0369 |
| snap3 | 0.0131 |
| share_of_team | 0.0099 |
| car3 | 0.0094 |
| tgt3 | 0.0080 |
| stat_c | 0.0064 |
| temp | 0.0062 |
| week | 0.0057 |
| opp_d_pres | 0.0057 |
| opp_d_epa_pass | 0.0056 |
| qb_delta | 0.0052 |
| f_z2 | 0.0051 |
| g_total | 0.0050 |
