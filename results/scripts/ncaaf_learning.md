# NCAAF script-then-learn: learning report

Games scripted 2016-2026: 8,819 (warm-up from 2008). Tuning window 2016-2020, holdout 2021-2026 (2026 = live season through 2026-10-14, PickCenter lines). Play-by-play from cfbfastR (raw.githubusercontent cfbfastR-data <=2021, sportsdataverse-data releases 2022+): 99% of games covered.

Core Kalman params chosen on 2016-2020: {'prior_var': 60.0, 'q_game': 0.3, 'carry': 0.85} ; RLS forgetting/delta: (0.9995, 0.1).

## Per-season learning curve (MAE, points)

(games with a closing line; ll = home-win log loss vs de-vigged closing moneyline)

|   season |   games |   w_mkt |   margin_core |   margin_script |   margin_blend |   margin_market |   total_core |   total_script |   total_blend |   total_market |   ll_script |   ll_blend |   ll_market |
|---------:|--------:|--------:|--------------:|----------------:|---------------:|----------------:|-------------:|---------------:|--------------:|---------------:|------------:|-----------:|------------:|
|     2016 |     832 |    0.93 |        13.671 |          13.212 |         12.734 |          12.725 |       13.182 |         13.373 |        13.057 |         13.075 |       0.504 |      0.485 |       0.486 |
|     2017 |     834 |    0.97 |        13.471 |          13.3   |         12.619 |          12.603 |       14.457 |         14.341 |        13.809 |         13.789 |       0.488 |      0.465 |       0.46  |
|     2018 |     845 |    1    |        13.228 |          12.822 |         12.179 |          12.176 |       13.706 |         13.675 |        12.913 |         12.921 |       0.467 |      0.449 |       0.44  |
|     2019 |     848 |    1    |        13.005 |          12.574 |         11.96  |          11.961 |       13.415 |         13.322 |        12.944 |         12.944 |       0.459 |      0.446 |       0.437 |
|     2020 |     542 |    0.97 |        13.258 |          13.062 |         12.768 |          12.768 |       14.173 |         13.926 |        13.845 |         13.85  |     nan     |    nan     |     nan     |
|     2021 |     849 |    0.94 |        13.559 |          13.249 |         12.614 |          12.589 |       13.061 |         12.606 |        12.36  |         12.367 |       0.556 |      0.539 |       0.541 |
|     2022 |     854 |    0.99 |        13.497 |          13.06  |         12.226 |          12.212 |       12.779 |         12.728 |        12.078 |         12.081 |       0.59  |      0.574 |       0.574 |
|     2023 |     910 |    1    |        12.992 |          12.763 |         12.087 |          12.085 |       13.148 |         13.077 |        12.639 |         12.64  |       0.543 |      0.518 |       0.511 |
|     2024 |     919 |    0.96 |        13.21  |          12.737 |         11.983 |          11.962 |       13.075 |         12.854 |        12.64  |         12.657 |       0.55  |      0.532 |       0.528 |
|     2025 |     934 |    0.85 |        12.611 |          12.235 |         11.9   |          11.895 |       12.992 |         12.708 |        12.417 |         12.412 |       0.52  |      0.509 |       0.505 |
|     2026 |     452 |    0.84 |        13.985 |          12.882 |         11.786 |          11.702 |       12.746 |         12.286 |        11.915 |         11.892 |       0.465 |      0.434 |       0.419 |

**2016-2020** margin MAE core 13.329 / script 12.987 / blend 12.429 / market 12.423;  total MAE core 13.760 / script 13.713 / blend 13.276 / market 13.278

**2021-2026 holdout** margin MAE core 13.237 / script 12.805 / blend 12.120 / market 12.100;  total MAE core 12.988 / script 12.750 / blend 12.384 / market 12.386

## Ablation (RLS correction re-run without each enrichment group)

Positive `hold_d_*` = dropping the group made the 2021+ holdout WORSE, i.e. the group helps.

| variant                   |   tune_mae_margin |   tune_mae_total |   hold_mae_margin |   hold_mae_total |   hold_d_margin_vs_all |   hold_d_total_vs_all |
|:--------------------------|------------------:|-----------------:|------------------:|-----------------:|-----------------------:|----------------------:|
| all enrichments           |            12.949 |           13.739 |            12.796 |           12.737 |                  0     |                 0     |
| core only (no correction) |            13.296 |           13.785 |            13.232 |           12.979 |                  0.435 |                 0.241 |
| drop tempo                |            12.945 |           13.716 |            12.793 |           12.758 |                 -0.003 |                 0.021 |
| drop eff                  |            12.981 |           13.738 |            12.826 |           12.739 |                  0.029 |                 0.002 |
| drop passrush             |            12.942 |           13.732 |            12.792 |           12.737 |                 -0.004 |                 0     |
| drop conf                 |            12.947 |           13.737 |            12.797 |           12.743 |                  0.001 |                 0.006 |
| drop rest                 |            12.943 |           13.729 |            12.797 |           12.736 |                  0     |                -0.001 |
| drop site                 |            12.963 |           13.742 |            12.774 |           12.735 |                 -0.022 |                -0.002 |
| drop p5g5                 |            13.011 |           13.74  |            12.844 |           12.734 |                  0.048 |                -0.003 |
| drop scale                |            13.003 |           13.801 |            12.83  |           12.95  |                  0.034 |                 0.213 |
| only tempo                |           nan     |          nan     |            13.249 |           12.986 |                  0.453 |                 0.249 |
| only eff                  |           nan     |          nan     |            13.132 |           12.926 |                  0.336 |                 0.189 |
| only passrush             |           nan     |          nan     |            13.202 |           12.936 |                  0.406 |                 0.199 |
| only conf                 |           nan     |          nan     |            12.895 |           12.961 |                  0.099 |                 0.224 |
| only rest                 |           nan     |          nan     |            13.233 |           12.987 |                  0.437 |                 0.25  |
| only site                 |           nan     |          nan     |            13.139 |           12.986 |                  0.343 |                 0.249 |
| only p5g5                 |           nan     |          nan     |            12.832 |           12.976 |                  0.036 |                 0.238 |
| only scale                |           nan     |          nan     |            13.209 |           12.769 |                  0.412 |                 0.032 |

Final RLS coefficients (standardised features; margin / total):

| feature        |   margin |   total |
|:---------------|---------:|--------:|
| intercept      |   -0.022 |  -1.373 |
| plays_exp      |   -0.183 |   1.364 |
| plays_diff     |   -0.158 |  -0.165 |
| epa_net        |   -1.275 |  -1.575 |
| epa_sum        |   -1.264 |   1.813 |
| succ_net       |    2.61  |   0.481 |
| pass_match     |    0.572 |   0.137 |
| rush_match     |    0.712 |   0.669 |
| pass_match_sum |    0.052 |   0.364 |
| rush_match_sum |    0.562 |  -0.494 |
| passrate_sum   |    0.177 |   0.872 |
| conf_diff      |    0.873 |   1.067 |
| rest_diff      |   -0.183 |  -0.339 |
| bye_h          |    0.456 |   0.241 |
| bye_a          |   -0.169 |  -0.471 |
| short_h        |    1.001 |   0.695 |
| short_a        |   -0.687 |  -0.499 |
| neutral_f      |    0.123 |  -0.345 |
| post           |   -0.348 |   0.725 |
| early          |    0.898 |  -0.65  |
| p5_vs_g5       |    1.964 |  -0.579 |
| fbs_vs_fcs     |    2.811 |  -0.476 |
| core_margin    |   -2.331 |   1.622 |
| core_total     |    0.206 |  -5.033 |

## What the engine got wrong most

Total-points error split (log scale, 2021+, 4,821 games with pbp): var(pace error) 0.0083, var(points-per-play error) 0.0986, 2cov -0.0031; total var 0.1039. -> efficiency (points per play) dominates the total error; the scripted play count misses by MAE 9.4 plays.

Margin MAE by matchup type, 2021+ (gap = script minus market; positive = market better):

| segment         |   games |   script_mae |   market_mae |   gap |   script_total |   market_total |
|:----------------|--------:|-------------:|-------------:|------:|---------------:|---------------:|
| all             |    4891 |       12.805 |       12.1   | 0.704 |         12.75  |         12.386 |
| conference game |    2866 |       12.499 |       12.118 | 0.382 |         12.858 |         12.462 |
| non-conference  |    2025 |       13.236 |       12.075 | 1.161 |         12.598 |         12.279 |
| P5 vs G5        |     612 |       12.707 |       11.68  | 1.028 |         12.345 |         12.038 |
| FBS vs FCS      |     705 |       13.86  |       12.377 | 1.483 |         12.255 |         11.857 |
| neutral site    |     252 |       13.724 |       11.986 | 1.738 |         13.514 |         13.028 |
| postseason/bowl |     132 |       14.395 |       11.875 | 2.52  |         13.599 |         13.491 |
| weeks 1-2       |    1102 |       13.833 |       12.429 | 1.404 |         12.801 |         12.471 |
| weeks 3-8       |    2121 |       12.492 |       11.863 | 0.629 |         12.659 |         12.311 |
| weeks 9+        |    1668 |       12.522 |       12.184 | 0.338 |         12.833 |         12.426 |
| |spread| >= 21  |    1161 |       13.256 |       12.151 | 1.104 |         11.856 |         11.557 |
| |spread| < 7    |    1813 |       12.467 |       11.797 | 0.67  |         13.223 |         12.824 |
| 2026 live       |     452 |       12.882 |       11.702 | 1.18  |         12.286 |         11.892 |

## Does anything beat the closing line out of sample?

Bet the side when the script disagrees with the close by more than tau points (tau picked on 2016-2020 by ROI, min 100 bets), flat -110 (break-even 52.4%):

| strategy       |   tau_chosen_2016_20 |   tune_bets |   tune_hit |   tune_roi |   hold_bets |   hold_hit |   hold_roi |
|:---------------|---------------------:|------------:|-----------:|-----------:|------------:|-----------:|-----------:|
| ATS via margin |                    8 |         344 |      0.523 |     -0.001 |         493 |      0.491 |     -0.063 |
| O/U via total  |                    4 |        1342 |      0.517 |     -0.013 |        1540 |      0.52  |     -0.007 |

Line-move test 2021+ (4,230 games with opening spreads): corr(script - open, close - open) = 0.211; script MAE vs result 12.72, open 12.26, close 12.12.

**Verdict:** No. The closing spread and total are more accurate than the script on 2021+; the learned blend weight sits at 0.94 on the market. Any ATS/O-U edge above must clear 52.4% on the holdout to matter; see the table (holdout numbers are the honest ones).
