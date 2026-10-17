# NBA script-driven props + 3/5/7-leg parlays (walk-forward, report 2021+)
Code: `engine/props_script/nba.py` (+ `common_hoops_football.py`). Runtime 12.3 min. Distributional LightGBM P(stat >= k) refit each season on the prior 4 seasons from the pre-game script plus tendencies; online per stat x threshold x probability-bin calibrator updated date by date. Seasons 2017-2020 = fit/choose, 2021-2026 = reported. NBA season label = year it ends.
## (a) Real points lines: hit rate by model confidence (side = higher calibrated prob)
One main line per player-game (latest snapshot, line closest to even odds, half-points only). Real lines were never used to train the model; only the online calibrator sees them, date by date.

**Season 2022** (n = 9,291)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 2114 | 0.518 | 0.503 | 0.011 |
| 0.53-0.56 | 3649 | 0.544 | 0.533 | 0.008 |
| 0.56-0.60 | 2401 | 0.576 | 0.552 | 0.010 |
| 0.60-0.65 | 693 | 0.619 | 0.540 | 0.019 |
| 0.65-1.00 | 434 | 0.696 | 0.535 | 0.024 |
| >= 0.53 | 7177 | 0.571 | 0.540 | 0.006 |
| >= 0.56 | 3528 | 0.599 | 0.547 | 0.008 |
| >= 0.60 | 1127 | 0.649 | 0.538 | 0.015 |
| >= 0.65 | 434 | 0.696 | 0.535 | 0.024 |

**Season 2025** (n = 11,838)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 7577 | 0.514 | 0.498 | 0.006 |
| 0.53-0.56 | 2726 | 0.541 | 0.512 | 0.010 |
| 0.56-0.60 | 1324 | 0.576 | 0.545 | 0.014 |
| 0.60-0.65 | 134 | 0.617 | 0.485 | 0.043 |
| 0.65-1.00 | 77 | 0.701 | 0.519 | 0.057 |
| >= 0.53 | 4261 | 0.557 | 0.522 | 0.008 |
| >= 0.56 | 1535 | 0.586 | 0.539 | 0.013 |
| >= 0.60 | 211 | 0.648 | 0.498 | 0.034 |
| >= 0.65 | 77 | 0.701 | 0.519 | 0.057 |

**Season 2026** (n = 14,340)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 9750 | 0.514 | 0.508 | 0.005 |
| 0.53-0.56 | 3423 | 0.541 | 0.534 | 0.009 |
| 0.56-0.60 | 977 | 0.571 | 0.544 | 0.016 |
| 0.60-0.65 | 134 | 0.616 | 0.575 | 0.043 |
| 0.65-1.00 | 56 | 0.700 | 0.679 | 0.062 |
| >= 0.53 | 4590 | 0.552 | 0.539 | 0.007 |
| >= 0.56 | 1167 | 0.582 | 0.554 | 0.015 |
| >= 0.60 | 190 | 0.641 | 0.605 | 0.035 |
| >= 0.65 | 56 | 0.700 | 0.679 | 0.062 |

**Variant: model + de-vigged market, walk-forward monthly logistic stack** (rows with both prices, n = 23,423; all seasons)

| tier | n | pred | hit | se |
|---|---|---|---|---|
| 0.50-0.53 | 20877 | 0.511 | 0.519 | 0.003 |
| 0.53-0.56 | 2354 | 0.539 | 0.538 | 0.010 |
| 0.56-0.60 | 175 | 0.573 | 0.583 | 0.037 |
| 0.60-0.65 | 14 | 0.611 | 0.643 | 0.128 |
| 0.65-1.00 | 3 | 0.667 | 0.667 | 0.272 |
| >= 0.53 | 2546 | 0.541 | 0.542 | 0.010 |
| >= 0.56 | 192 | 0.577 | 0.589 | 0.036 |
| >= 0.60 | 17 | 0.621 | 0.647 | 0.116 |
| >= 0.65 | 3 | 0.667 | 0.667 | 0.272 |

Log loss on these rows: model-cal 0.6929, market 0.6925, stack 0.6920.

## (b) Milestones (k+ ladders), 2021-2026: realized hit rate by confidence tier vs naive base rates
`group_base` = hit rate of the same stat x threshold x role tier in prior seasons; `player_naive` = the player's own trailing (EWM, hl 10 games) frequency of reaching k. Lift = hit - naive.

| tier | n | pred | hit | group_base | player_naive | lift_vs_group | lift_vs_player | se |
|---|---|---|---|---|---|---|---|---|
| 0.00-0.70 | 2145096 | 0.186 | 0.186 | 0.205 | 0.186 | -0.019 | -0.000 | 0.000 |
| 0.70-0.75 | 54369 | 0.725 | 0.725 | 0.518 | 0.722 | 0.207 | 0.003 | 0.002 |
| 0.75-0.80 | 54048 | 0.775 | 0.775 | 0.553 | 0.772 | 0.222 | 0.003 | 0.002 |
| 0.80-0.85 | 50210 | 0.825 | 0.826 | 0.583 | 0.824 | 0.243 | 0.002 | 0.002 |
| 0.85-0.90 | 51596 | 0.875 | 0.876 | 0.630 | 0.877 | 0.246 | -0.002 | 0.001 |
| 0.90-0.95 | 48651 | 0.926 | 0.927 | 0.672 | 0.928 | 0.255 | -0.001 | 0.001 |
| 0.95-1.00 | 47510 | 0.972 | 0.971 | 0.763 | 0.972 | 0.208 | -0.001 | 0.001 |
| >= 0.70 | 306384 | 0.846 | 0.846 | 0.616 | 0.845 | 0.230 | 0.001 | 0.001 |
| >= 0.75 | 252015 | 0.872 | 0.872 | 0.637 | 0.872 | 0.235 | 0.000 | 0.001 |
| >= 0.80 | 197967 | 0.898 | 0.899 | 0.660 | 0.899 | 0.238 | -0.000 | 0.001 |
| >= 0.85 | 147757 | 0.923 | 0.923 | 0.687 | 0.924 | 0.237 | -0.001 | 0.001 |
| >= 0.90 | 96161 | 0.949 | 0.949 | 0.717 | 0.950 | 0.232 | -0.001 | 0.001 |
| >= 0.95 | 47510 | 0.972 | 0.971 | 0.763 | 0.972 | 0.208 | -0.001 | 0.001 |

By stat, legs with p >= 0.80:

| stat | k | n | pred | hit | group_base | player_naive | all_legs_base(k) |
|---|---|---|---|---|---|---|---|
| ast | 2 | 40457 | 0.922 | 0.923 | 0.718 | 0.919 | 0.528 |
| ast | 4 | 12168 | 0.885 | 0.882 | 0.505 | 0.882 | 0.255 |
| ast | 6 | 2784 | 0.865 | 0.865 | 0.332 | 0.860 | 0.121 |
| ast | 8 | 255 | 0.850 | 0.847 | 0.185 | 0.837 | 0.055 |
| fg3m | 1 | 40936 | 0.876 | 0.878 | 0.675 | 0.881 | 0.560 |
| fg3m | 2 | 3504 | 0.845 | 0.842 | 0.491 | 0.854 | 0.325 |
| fg3m | 3 | 210 | 0.845 | 0.838 | 0.303 | 0.843 | 0.174 |
| pts | 10 | 35076 | 0.916 | 0.917 | 0.806 | 0.917 | 0.475 |
| pts | 15 | 15306 | 0.890 | 0.891 | 0.689 | 0.893 | 0.288 |
| pts | 20 | 3801 | 0.857 | 0.860 | 0.530 | 0.868 | 0.162 |
| pts | 25 | 304 | 0.839 | 0.822 | 0.335 | 0.837 | 0.083 |
| reb | 4 | 28210 | 0.908 | 0.910 | 0.652 | 0.908 | 0.490 |
| reb | 6 | 10741 | 0.891 | 0.888 | 0.439 | 0.891 | 0.280 |
| reb | 8 | 3751 | 0.864 | 0.856 | 0.276 | 0.861 | 0.151 |
| reb | 10 | 464 | 0.849 | 0.856 | 0.172 | 0.854 | 0.080 |

Selection skill at equal naive rate: legs where the player's naive rate is 0.80-0.90, split by model p:

| tier | n | pred | hit | player_naive | se |
|---|---|---|---|---|---|
| 0.00-0.70 | 8767 | 0.645 | 0.682 | 0.833 | 0.005 |
| 0.70-0.80 | 29015 | 0.759 | 0.776 | 0.841 | 0.002 |
| 0.80-0.90 | 48623 | 0.850 | 0.851 | 0.853 | 0.002 |
| 0.90-1.00 | 13587 | 0.928 | 0.920 | 0.868 | 0.002 |
| >= 0.70 | 91225 | 0.833 | 0.838 | 0.852 | 0.001 |
| >= 0.80 | 62210 | 0.867 | 0.866 | 0.857 | 0.001 |
| >= 0.90 | 13587 | 0.928 | 0.920 | 0.868 | 0.002 |

## (c) Parlays, one ticket per day per construction, 2021-2026
Predicted = Gaussian-copula joint of calibrated leg probabilities with correlations learned from prior seasons' residuals (teammate/opponent stat pairs, player vs team margin and game total). `typical_price` = median ticket odds: real odds for real-line and ML legs; milestone legs priced at the player's naive rate plus 4.5% margin; totals -110.

| construction | legs | n_days | pred_joint | pred_indep | realized | se | typical_price |
|---|---|---|---|---|---|---|---|
| 1 top-N cross-game | 3 | 1246 | 0.965 | 0.965 | 0.969 | 0.005 | -2156 |
| 1 top-N cross-game | 5 | 1246 | 0.930 | 0.929 | 0.925 | 0.007 | -1274 |
| 1 top-N cross-game | 7 | 1246 | 0.884 | 0.883 | 0.890 | 0.009 | -896 |
| 1R top-N real lines | 3 | 422 | 0.240 | 0.240 | 0.133 | 0.017 | +543 |
| 1R top-N real lines | 5 | 422 | 0.086 | 0.086 | 0.028 | 0.008 | +2136 |
| 1R top-N real lines | 7 | 422 | 0.030 | 0.030 | 0.002 | 0.002 | +7684 |
| 2 same-game script stack | 3 | 1246 | 0.954 | 0.954 | 0.943 | 0.007 | -2156 |
| 2 same-game script stack | 5 | 1246 | 0.883 | 0.882 | 0.864 | 0.010 | -1237 |
| 2 same-game script stack | 7 | 1246 | 0.758 | 0.756 | 0.742 | 0.012 | -540 |
| 3 mixed 85%+ ML + props | 3 | 393 | 0.873 | 0.872 | 0.885 | 0.016 | -858 |
| 3 mixed 85%+ ML + props | 5 | 393 | 0.852 | 0.852 | 0.860 | 0.018 | -663 |
| 3 mixed 85%+ ML + props | 7 | 393 | 0.828 | 0.827 | 0.850 | 0.018 | -537 |
| 4 most-likely ticket | 3 | 1246 | 0.965 | 0.965 | 0.969 | 0.005 | -2156 |
| 4 most-likely ticket | 5 | 1246 | 0.930 | 0.930 | 0.926 | 0.007 | -1274 |
| 4 most-likely ticket | 7 | 1246 | 0.884 | 0.883 | 0.890 | 0.009 | -896 |

By season (realized / predicted joint), construction 1 and 4:

| construction | legs | season | n_days | pred | realized |
|---|---|---|---|---|---|
| 1 top-N cross-game | 3 | 2021 | 187 | 0.969 | 0.957 |
| 1 top-N cross-game | 3 | 2022 | 213 | 0.965 | 0.981 |
| 1 top-N cross-game | 3 | 2023 | 212 | 0.965 | 0.962 |
| 1 top-N cross-game | 3 | 2024 | 208 | 0.961 | 0.971 |
| 1 top-N cross-game | 3 | 2025 | 214 | 0.966 | 0.958 |
| 1 top-N cross-game | 3 | 2026 | 212 | 0.964 | 0.981 |
| 1 top-N cross-game | 5 | 2021 | 187 | 0.937 | 0.925 |
| 1 top-N cross-game | 5 | 2022 | 213 | 0.930 | 0.944 |
| 1 top-N cross-game | 5 | 2023 | 212 | 0.933 | 0.929 |
| 1 top-N cross-game | 5 | 2024 | 208 | 0.926 | 0.923 |
| 1 top-N cross-game | 5 | 2025 | 214 | 0.928 | 0.902 |
| 1 top-N cross-game | 5 | 2026 | 212 | 0.926 | 0.929 |
| 1 top-N cross-game | 7 | 2021 | 187 | 0.890 | 0.882 |
| 1 top-N cross-game | 7 | 2022 | 213 | 0.885 | 0.901 |
| 1 top-N cross-game | 7 | 2023 | 212 | 0.890 | 0.896 |
| 1 top-N cross-game | 7 | 2024 | 208 | 0.881 | 0.909 |
| 1 top-N cross-game | 7 | 2025 | 214 | 0.879 | 0.855 |
| 1 top-N cross-game | 7 | 2026 | 212 | 0.876 | 0.896 |
| 1R top-N real lines | 3 | 2022 | 98 | 0.312 | 0.184 |
| 1R top-N real lines | 3 | 2025 | 162 | 0.223 | 0.111 |
| 1R top-N real lines | 3 | 2026 | 162 | 0.214 | 0.123 |
| 1R top-N real lines | 5 | 2022 | 98 | 0.132 | 0.031 |
| 1R top-N real lines | 5 | 2025 | 162 | 0.075 | 0.025 |
| 1R top-N real lines | 5 | 2026 | 162 | 0.070 | 0.031 |
| 1R top-N real lines | 7 | 2022 | 98 | 0.054 | 0.000 |
| 1R top-N real lines | 7 | 2025 | 162 | 0.025 | 0.006 |
| 1R top-N real lines | 7 | 2026 | 162 | 0.022 | 0.000 |
| 4 most-likely ticket | 3 | 2021 | 187 | 0.969 | 0.957 |
| 4 most-likely ticket | 3 | 2022 | 213 | 0.965 | 0.981 |
| 4 most-likely ticket | 3 | 2023 | 212 | 0.965 | 0.962 |
| 4 most-likely ticket | 3 | 2024 | 208 | 0.961 | 0.971 |
| 4 most-likely ticket | 3 | 2025 | 214 | 0.966 | 0.958 |
| 4 most-likely ticket | 3 | 2026 | 212 | 0.964 | 0.981 |
| 4 most-likely ticket | 5 | 2021 | 187 | 0.937 | 0.925 |
| 4 most-likely ticket | 5 | 2022 | 213 | 0.930 | 0.944 |
| 4 most-likely ticket | 5 | 2023 | 212 | 0.934 | 0.929 |
| 4 most-likely ticket | 5 | 2024 | 208 | 0.926 | 0.923 |
| 4 most-likely ticket | 5 | 2025 | 214 | 0.928 | 0.907 |
| 4 most-likely ticket | 5 | 2026 | 212 | 0.926 | 0.929 |
| 4 most-likely ticket | 7 | 2021 | 187 | 0.890 | 0.882 |
| 4 most-likely ticket | 7 | 2022 | 213 | 0.885 | 0.901 |
| 4 most-likely ticket | 7 | 2023 | 212 | 0.891 | 0.896 |
| 4 most-likely ticket | 7 | 2024 | 208 | 0.881 | 0.909 |
| 4 most-likely ticket | 7 | 2025 | 214 | 0.879 | 0.855 |
| 4 most-likely ticket | 7 | 2026 | 212 | 0.876 | 0.896 |

## (d) Which tendencies mattered (ablation)
Single split: train 2017-2020, test a fixed random 35% of 2021-2026 dates (same dates for every row). Each row refits the LightGBM without that feature group, then runs the same online calibrator. Lower log loss = better.

| config | ms_logloss | ms_ece | n>=0.8 | hit>=0.8 | lift_vs_player_naive>=0.8 | n>=0.9 | hit>=0.9 | top5_daily_hit | line_hit_all | line_n>=0.58 | line_hit>=0.58 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full model | 0.3521 | 0.0009 | 69529 | 0.8965 | -0.0005 | 32947 | 0.9475 | 0.9427 | 0.5155 | 1387 | 0.5242 |
| full model, no online calibrator | 0.3519 | 0.0022 | 67696 | 0.8994 | -0.0005 | 31581 | 0.9502 | 0.9381 | 0.5177 | 4816 | 0.5255 |
| drop minutes_blowout | 0.3525 | 0.0011 | 68433 | 0.8978 | -0.0014 | 33281 | 0.9476 | 0.9312 | 0.5135 | 1901 | 0.5502 |
| drop teammates_out | 0.3531 | 0.0012 | 69088 | 0.8954 | -0.0012 | 32716 | 0.9467 | 0.9106 | 0.5176 | 1913 | 0.5384 |
| drop opp_def_vs_pos | 0.3522 | 0.0008 | 68878 | 0.8976 | -0.0005 | 33126 | 0.9477 | 0.9220 | 0.5223 | 1703 | 0.5367 |
| drop rest_b2b | 0.3521 | 0.0009 | 69461 | 0.8968 | -0.0003 | 33011 | 0.9470 | 0.9381 | 0.5191 | 1450 | 0.5441 |
| drop home_away | 0.3520 | 0.0010 | 69160 | 0.8972 | -0.0006 | 32555 | 0.9484 | 0.9450 | 0.5200 | 1592 | 0.5383 |
| drop recent_form | 0.3528 | 0.0013 | 68185 | 0.8972 | -0.0026 | 33604 | 0.9463 | 0.9151 | 0.5170 | 1720 | 0.5407 |
| drop player_bias_dispersion | 0.3522 | 0.0009 | 69102 | 0.8972 | -0.0004 | 32888 | 0.9482 | 0.9289 | 0.5190 | 1515 | 0.5373 |
| script features only (LGBM) | 0.3561 | 0.0012 | 67772 | 0.8937 | -0.0047 | 32351 | 0.9456 | 0.9174 | 0.5134 | 1874 | 0.5406 |
| script parametric (no LGBM) | 0.3577 | 0.0034 | 67004 | 0.8947 | -0.0049 | 30210 | 0.9485 | 0.9312 | 0.5164 | 1930 | 0.5487 |

**Tendency evidence (2021-2026):** mean script residual z = (actual - script mu)/sd by bucket. Non-zero means the coupled script under/over-shoots there and the LightGBM layer has something to learn (the script already renormalises usage for teammates missing the previous game with exponent 0.5, so these are residual effects).

| tendency | bucket | n | ast | fg3m | pts | reb |
|---|---|---|---|---|---|---|
| teammates out prev game: vacated stat share | (-0.011, 0.0] | 179442 | -0.006 | -0.018 | -0.006 | 0.001 |
| teammates out prev game: vacated stat share | (0.0, 0.1] | 122680 | 0.014 | -0.005 | 0.005 | 0.009 |
| teammates out prev game: vacated stat share | (0.1, 0.25] | 197987 | 0.018 | 0.001 | 0.009 | 0.008 |
| teammates out prev game: vacated stat share | (0.25, 0.5] | 126735 | 0.020 | 0.001 | 0.015 | -0.006 |
| teammates out prev game: vacated stat share | (0.5, 5.0] | 26884 | -0.038 | -0.042 | -0.052 | -0.060 |
| teammates out prev game: vacated minutes | (-1.001, 0.0] | 176508 | -0.006 | -0.018 | -0.006 | 0.001 |
| teammates out prev game: vacated minutes | (0.0, 25.0] | 101612 | 0.014 | -0.012 | 0.003 | -0.007 |
| teammates out prev game: vacated minutes | (25.0, 50.0] | 181160 | 0.012 | -0.005 | 0.008 | 0.006 |
| teammates out prev game: vacated minutes | (50.0, 100.0] | 158548 | 0.017 | 0.004 | 0.011 | 0.006 |
| teammates out prev game: vacated minutes | (100.0, 500.0] | 35900 | -0.002 | -0.001 | -0.013 | -0.021 |
| player returning from absence | (-0.501, 0.5] | 567408 | 0.033 | 0.012 | 0.032 | 0.033 |
| player returning from absence | (0.5, 1.5] | 86320 | -0.157 | -0.132 | -0.186 | -0.204 |
| opponent rotation minutes out | (-1.001, 0.0] | 173888 | -0.014 | -0.018 | -0.019 | -0.006 |
| opponent rotation minutes out | (0.0, 30.0] | 145084 | -0.001 | -0.009 | -0.006 | -0.010 |
| opponent rotation minutes out | (30.0, 60.0] | 192548 | 0.018 | 0.002 | 0.014 | 0.004 |
| opponent rotation minutes out | (60.0, 500.0] | 142208 | 0.030 | -0.005 | 0.025 | 0.017 |
| team on back-to-back | (-0.501, 0.5] | 545104 | -0.003 | -0.015 | -0.014 | -0.010 |
| team on back-to-back | (0.5, 1.5] | 108624 | 0.065 | 0.035 | 0.092 | 0.060 |
| opponent on back-to-back | (-0.501, 0.5] | 544388 | 0.000 | -0.012 | -0.002 | -0.004 |
| opponent on back-to-back | (0.5, 1.5] | 109340 | 0.046 | 0.016 | 0.030 | 0.027 |
| home | (-0.501, 0.5] | 326052 | 0.000 | -0.016 | 0.013 | -0.002 |
| home | (0.5, 1.5] | 327676 | 0.016 | 0.002 | -0.007 | 0.005 |
| opp defence vs position (prior z allowed) | (-9.001, -0.15] | 29540 | -0.037 | -0.082 | -0.042 | -0.014 |
| opp defence vs position (prior z allowed) | (-0.15, -0.05] | 128023 | -0.018 | -0.028 | -0.025 | -0.016 |
| opp defence vs position (prior z allowed) | (-0.05, 0.05] | 249643 | 0.010 | -0.004 | -0.002 | -0.006 |
| opp defence vs position (prior z allowed) | (0.05, 0.15] | 173774 | 0.020 | 0.015 | 0.016 | 0.010 |
| opp defence vs position (prior z allowed) | (0.15, 9.0] | 72748 | 0.031 | 0.025 | 0.046 | 0.039 |
| recent form (EWM z, hl 10) | (-9.001, -0.3] | 35890 | -0.045 | -0.075 | -0.099 | -0.072 |
| recent form (EWM z, hl 10) | (-0.3, -0.1] | 165726 | 0.015 | -0.046 | -0.013 | 0.012 |
| recent form (EWM z, hl 10) | (-0.1, 0.1] | 249551 | 0.007 | 0.010 | 0.020 | 0.002 |
| recent form (EWM z, hl 10) | (0.1, 0.3] | 148006 | -0.002 | 0.022 | 0.007 | -0.009 |
| recent form (EWM z, hl 10) | (0.3, 30.0] | 52231 | 0.042 | 0.029 | 0.030 | 0.042 |
| player long-run bias (EWM z, hl 40) | (-9.001, -0.2] | 21268 | 0.029 | -0.049 | -0.050 | 0.004 |
| player long-run bias (EWM z, hl 40) | (-0.2, -0.05] | 162404 | -0.000 | -0.069 | -0.034 | -0.021 |
| player long-run bias (EWM z, hl 40) | (-0.05, 0.05] | 217851 | 0.002 | 0.015 | 0.014 | -0.000 |
| player long-run bias (EWM z, hl 40) | (0.05, 0.2] | 198769 | 0.007 | 0.027 | 0.016 | 0.006 |
| player long-run bias (EWM z, hl 40) | (0.2, 30.0] | 51112 | 0.042 | 0.025 | 0.035 | 0.046 |
| minutes vs expectation, last ~5 | (-0.001, 0.85] | 79396 | -0.137 | -0.138 | -0.125 | -0.150 |
| minutes vs expectation, last ~5 | (0.85, 0.95] | 104208 | -0.058 | -0.051 | -0.040 | -0.042 |
| minutes vs expectation, last ~5 | (0.95, 1.05] | 259864 | 0.019 | 0.006 | 0.017 | 0.010 |
| minutes vs expectation, last ~5 | (1.05, 1.15] | 122280 | 0.041 | 0.019 | 0.017 | 0.026 |
| minutes vs expectation, last ~5 | (1.15, 3.0] | 85656 | 0.133 | 0.086 | 0.109 | 0.126 |
| blowout probability | (-0.001, 0.28] | 34248 | -0.006 | -0.013 | -0.016 | 0.002 |
| blowout probability | (0.28, 0.33] | 306484 | -0.018 | -0.021 | -0.021 | -0.012 |
| blowout probability | (0.33, 0.4] | 189484 | 0.017 | -0.003 | 0.013 | 0.007 |
| blowout probability | (0.4, 1.0] | 123512 | 0.063 | 0.024 | 0.053 | 0.025 |
| script team margin | (-40.001, -8.0] | 79580 | 0.073 | 0.045 | 0.056 | 0.040 |
| script team margin | (-8.0, -3.0] | 134068 | 0.010 | -0.002 | 0.015 | 0.016 |
| script team margin | (-3.0, 3.0] | 219128 | -0.013 | -0.017 | -0.018 | -0.013 |
| script team margin | (3.0, 8.0] | 136212 | -0.005 | -0.023 | -0.010 | -0.007 |
| script team margin | (8.0, 40.0] | 84740 | 0.018 | -0.013 | 0.011 | -0.008 |

Top features (gain share, 2026 model; k/zk/rk = threshold terms):

| feature | gain_share |
|---|---|
| zk | 0.7686 |
| rk | 0.1219 |
| minr5 | 0.0099 |
| starter_prev | 0.0064 |
| gnum | 0.0063 |
| spread_team | 0.0059 |
| self_missed_prev | 0.0053 |
| min_e | 0.0046 |
| n_prev | 0.0046 |
| min_exp | 0.0039 |
| share_of_team | 0.0028 |
| t_margin | 0.0028 |
| zsd | 0.0028 |
| blowout_p | 0.0027 |
| f_z10 | 0.0027 |
| pace_o | 0.0027 |
| pace_t | 0.0027 |
| gap_games | 0.0024 |
| f_z3 | 0.0024 |
| miss_min_t | 0.0024 |

## (e) Learning curve by season (milestone legs; log loss / calibration error)

| season | n_legs | ll_script_param | ll_lgbm | ll_calibrated | ece_param | ece_lgbm | ece_cal | hit>=0.8 | pred>=0.8 | line_n | line_hit_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2017 | 412200 | 0.3328 | 0.3248 | 0.3250 | 0.0207 | 0.0023 | 0.0026 | 0.9001 | 0.8985 | 0 |  |
| 2018 | 410280 | 0.3438 | 0.3350 | 0.3351 | 0.0191 | 0.0032 | 0.0021 | 0.8934 | 0.8973 | 0 |  |
| 2019 | 411615 | 0.3603 | 0.3487 | 0.3486 | 0.0216 | 0.0023 | 0.0021 | 0.8950 | 0.8949 | 0 |  |
| 2020 | 354825 | 0.3645 | 0.3541 | 0.3540 | 0.0232 | 0.0023 | 0.0024 | 0.8954 | 0.8950 | 0 |  |
| 2021 | 367965 | 0.3549 | 0.3436 | 0.3436 | 0.0229 | 0.0021 | 0.0017 | 0.9015 | 0.8982 | 0 |  |
| 2022 | 413790 | 0.3620 | 0.3503 | 0.3502 | 0.0239 | 0.0021 | 0.0015 | 0.8966 | 0.8960 | 9291 | 0.5318 |
| 2023 | 410865 | 0.3614 | 0.3481 | 0.3480 | 0.0242 | 0.0026 | 0.0023 | 0.9010 | 0.8993 | 0 |  |
| 2024 | 416985 | 0.3572 | 0.3461 | 0.3460 | 0.0227 | 0.0025 | 0.0020 | 0.8972 | 0.9006 | 0 |  |
| 2025 | 417690 | 0.3687 | 0.3556 | 0.3555 | 0.0220 | 0.0043 | 0.0031 | 0.8980 | 0.9002 | 11838 | 0.5066 |
| 2026 | 424185 | 0.3693 | 0.3561 | 0.3560 | 0.0235 | 0.0021 | 0.0026 | 0.8971 | 0.8941 | 14340 | 0.5181 |

Real-line hit rate by month of the calibrator's life (all picks):

| date | n | hit | pred |
|---|---|---|---|
| 2021Q4 | 6866 | 0.529 | 0.564 |
| 2022Q1 | 2425 | 0.539 | 0.545 |
| 2024Q4 | 4627 | 0.514 | 0.534 |
| 2025Q1 | 6322 | 0.502 | 0.527 |
| 2025Q2 | 889 | 0.501 | 0.527 |
| 2025Q4 | 6126 | 0.517 | 0.527 |
| 2026Q1 | 7459 | 0.519 | 0.525 |
| 2026Q2 | 755 | 0.526 | 0.525 |
