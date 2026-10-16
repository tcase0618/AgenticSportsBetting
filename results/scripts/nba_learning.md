# NBA script-then-learn: learning report

Code: `engine/scripts/nba.py` (core = `engine/scripts/base.py`, unmodified). Games 2010-2026 are run sequentially (2010-15 warm-up). Each date's games are all scripted before any of them is learned. Tuning used 2016-2020 only, and 2021+ is the reported period. Market = closing spread/total; moneyline de-vigged with `devig(..., 'odds_ratio')`.

Tuned core Kalman: {'q_game': 0.08, 'prior_var': 10.0, 'carry': 0.6}. Online RLS correction: forgetting λ=0.99995, ridge=300.0. Player renormalisation exponent α=0.5.

## Bottom line

- **2021+ margin MAE:** core 11.043, enriched script 10.962, learned blend 10.582, market + learned residual 10.590, **closing line 10.570** (n=7708).
- **2021+ total MAE:** core 14.784, enriched 14.671, blend 14.211, market + residual 14.236, **closing line 14.212**.
- **Learned market weight** at the end of 2026: margin 0.99, total 0.91; mean over 2021+ was 0.89 / 0.99.
- **Information beyond the close (2021+):** corr(script − close, actual − close) is 0.010 for margin and -0.019 for total.
- **Line movement (CLV) on games with openers (3555):** corr(script − open, close − open) is 0.174 for spreads and 0.307 for totals. If positive, the script anticipates where the line moves, even though it does not beat the close.

**Betting the closing number at −110** (edge threshold picked on 2016-20 with n≥300, reported on 2021+):

| bet | thr | roi_2016_20 | n_2021p | roi_2021p |
|---|---|---|---|---|
| spread: script | 6.000 | 0.025 | 720 | -0.009 |
| spread: mkt+resid | 0.500 | -0.034 | 2864 | -0.042 |
| total: script | 6.000 | -0.032 | 1216 | -0.072 |
| total: mkt+resid | 1.000 | -0.053 | 1991 | -0.051 |

**Betting the OPENING number at −110** with the same rule. Openers exist for 2016-22 and 81% of 2026. `mean_clv_pts` is the average line move in the bet's favour from open to close:

| bet | thr | roi_2016_20 | n_2021p | roi_2021p | mean_clv_pts |
|---|---|---|---|---|---|
| spread @open: script | 4 | -0.020 | 695 | 0.020 | 0.509 |
| total @open: script | 6 | 0.017 | 295 | 0.010 | 2.592 |

## Learning curve (per season, MAE in points; ML log loss)

| season | games | margin_core | margin_script | margin_blend | margin_mkt_plus | margin_market | total_core | total_script | total_blend | total_mkt_plus | total_market | w_mkt_margin | w_mkt_total | ml_ll_script | ml_ll_blend | ml_ll_market |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 1316 | 9.512 | 9.506 | 9.244 | 9.255 | 9.238 | 13.809 | 13.761 | 13.594 | 13.694 | 13.590 | 0.953 | 0.938 | 0.587 | 0.572 | 0.569 |
| 2017 | 1308 | 10.151 | 10.115 | 9.809 | 9.810 | 9.801 | 13.999 | 13.963 | 13.602 | 13.598 | 13.601 | 0.994 | 1.000 | 0.624 | 0.607 | 0.610 |
| 2018 | 1311 | 10.001 | 9.955 | 9.525 | 9.544 | 9.519 | 14.865 | 14.795 | 14.355 | 14.380 | 14.350 | 1.000 | 1.000 | 0.619 | 0.594 | 0.591 |
| 2019 | 1311 | 10.272 | 10.198 | 9.937 | 9.944 | 9.937 | 15.060 | 15.000 | 14.496 | 14.512 | 14.496 | 1.000 | 1.000 | 0.613 | 0.598 | 0.597 |
| 2020 | 1143 | 10.249 | 10.273 | 10.134 | 10.147 | 10.183 | 15.302 | 15.207 | 14.905 | 14.906 | 14.905 | 0.560 | 1.000 | 0.627 | 0.625 | 0.615 |
| 2021 | 1171 | 11.219 | 11.150 | 10.804 | 10.784 | 10.754 | 14.852 | 14.741 | 14.462 | 14.511 | 14.460 | 0.750 | 0.954 | 0.641 | 0.623 | 0.620 |
| 2022 | 1323 | 11.204 | 11.117 | 10.719 | 10.730 | 10.709 | 14.713 | 14.683 | 14.149 | 14.174 | 14.148 | 1.000 | 1.000 | 0.630 | 0.608 | 0.607 |
| 2023 | 1320 | 10.249 | 10.217 | 9.806 | 9.828 | 9.807 | 14.664 | 14.546 | 14.027 | 14.043 | 14.027 | 0.913 | 1.000 | 0.646 | 0.625 | 0.627 |
| 2024 | 1253 | 11.050 | 10.926 | 10.552 | 10.557 | 10.545 | 14.588 | 14.522 | 14.048 | 14.068 | 14.048 | 0.969 | 1.000 | 0.607 | 0.584 | 0.580 |
| 2025 | 1319 | 11.126 | 11.006 | 10.657 | 10.669 | 10.653 | 14.798 | 14.653 | 14.110 | 14.133 | 14.111 | 0.946 | 1.000 | 0.602 | 0.588 | 0.586 |
| 2026 | 1322 | 11.429 | 11.372 | 10.979 | 10.990 | 10.972 | 15.087 | 14.881 | 14.489 | 14.511 | 14.495 | 0.994 | 0.913 | 0.599 | 0.581 | 0.578 |

**Within-season learning:** margin MAE by the fewer of the two teams' games played that season (2016+):

| gnum | n | script | core | market |
|---|---|---|---|---|
| 0-10 | 1881 | 10.396 | 10.386 | 10.039 |
| 11-25 | 2515 | 10.426 | 10.475 | 10.171 |
| 26-50 | 4116 | 10.462 | 10.521 | 10.146 |
| 51-82 | 4927 | 10.502 | 10.600 | 10.117 |
| playoffs/83+ | 658 | 11.877 | 11.854 | 11.442 |

## Error attribution

**Totals (2021+), split into pace and efficiency.** Total error = (poss_pred − poss_act)·ppp_pred + poss_act·(ppp_pred − ppp_act). Pace part: sd 11.46 pts. Efficiency part: sd 17.37 pts. The total-error sd is 18.53, and the covariance share is -0.26. Possession MAE is 3.81, and the mean possessions error (bias) is -1.02. Shooting-efficiency noise dominates. Overtime games are 8.2% of games with +8 or more possessions.

**By game type (2021+).** Where the script falls furthest behind the market:

| group | n | mae_margin_script | mae_margin_mkt | mae_total_script | mae_total_mkt | bias_total_script | bias_total_mkt |
|---|---|---|---|---|---|---|---|
| back-to-back involved=True | 2228 | 11.048 | 10.601 | 15.140 | 14.716 | -0.082 | -0.692 |
| back-to-back involved=False | 5548 | 10.944 | 10.557 | 14.478 | 14.007 | 0.226 | -0.226 |
| starter missed prev game=True | 5219 | 10.950 | 10.480 | 14.669 | 14.200 | 0.192 | -0.442 |
| starter missed prev game=False | 2557 | 11.022 | 10.755 | 14.665 | 14.237 | 0.027 | -0.194 |
| final margin ≥20=True | 1523 | 22.543 | 21.440 | 13.867 | 13.435 | 0.038 | -0.687 |
| final margin ≥20=False | 6253 | 8.156 | 7.931 | 14.863 | 14.401 | 0.162 | -0.282 |
| playoff=True | 507 | 12.114 | 11.812 | 14.500 | 14.473 | 2.613 | 1.246 |
| playoff=False | 7269 | 10.894 | 10.492 | 14.680 | 14.195 | -0.035 | -0.462 |
| slow-fast clash | 2592 | 10.783 | 10.365 | 14.569 | 14.110 | -0.197 | -0.523 |
| neutral | 2592 | 10.938 | 10.538 | 14.305 | 13.945 | 0.394 | -0.172 |
| same-tempo | 2592 | 11.199 | 10.808 | 15.129 | 14.581 | 0.218 | -0.386 |
| 3pt matchup low | 2592 | 11.007 | 10.677 | 14.563 | 14.093 | 0.480 | 0.082 |
| mid | 2592 | 10.793 | 10.397 | 15.017 | 14.471 | 0.006 | -0.560 |
| high | 2592 | 11.121 | 10.636 | 14.423 | 14.070 | -0.072 | -0.607 |

**Online RLS coefficients at the end of 2026** (points per 1 sd of feature, for team T's score; `d_poss` = possession-model points − core points):

| feature | coef |
|---|---|
| bias | 0.508 |
| is_home | -0.175 |
| playoff | -3.601 |
| d_poss | 0.588 |
| rest_T | 0.213 |
| rest_O | 0.071 |
| b2b_T | -0.350 |
| b2b_O | 0.431 |
| g3in4_T | -0.115 |
| g3in4_O | 0.034 |
| travel_T | -0.116 |
| travel_O | -0.018 |
| tz_T | -0.045 |
| tz_O | 0.134 |
| road_T | 0.061 |
| road_O | 0.059 |
| miss_min_T | 0.592 |
| miss_min_O | 0.261 |
| miss_st_T | -0.023 |
| miss_st_O | 0.114 |
| miss_pts_T | -0.854 |
| miss_pts_O | -0.138 |
| pace_clash | 0.173 |
| m_3pa | 0.064 |
| m_3pct | 0.023 |
| m_orb | 0.111 |
| m_tov | 0.053 |
| m_ft | 0.432 |

## Ablation (online correction re-run with groups dropped; MAE)

| config | margin 2016-20 | margin 2021+ | total 2016-20 | total 2021+ |
|---|---|---|---|---|
| base + inj only | 10.016 | 10.996 | 14.562 | 14.699 |
| base + matchup only | 10.034 | 11.046 | 14.560 | 14.700 |
| base + poss only | 10.038 | 11.039 | 14.538 | 14.696 |
| base + rest only | 10.005 | 11.007 | 14.563 | 14.711 |
| base + travel only | 10.031 | 11.047 | 14.569 | 14.711 |
| core only (no enrichment) | 10.031 | 11.043 | 14.588 | 14.784 |
| drop inj | 10.017 | 11.003 | 14.532 | 14.683 |
| drop matchup | 9.998 | 10.961 | 14.534 | 14.681 |
| drop poss | 9.995 | 10.962 | 14.561 | 14.689 |
| drop rest | 10.025 | 11.001 | 14.529 | 14.669 |
| drop travel | 10.001 | 10.960 | 14.525 | 14.672 |
| full | 10.002 | 10.962 | 14.528 | 14.671 |

Closing line on the same 2021+ games: margin 10.570, total 14.212.

Core tuning grid (2016-20):

| q_game | prior_var | carry | mae_sum_2016_20 |
|---|---|---|---|
| 0.030 | 10.000 | 0.600 | 24.647 |
| 0.030 | 10.000 | 0.800 | 24.659 |
| 0.030 | 25.000 | 0.600 | 24.694 |
| 0.030 | 25.000 | 0.800 | 24.654 |
| 0.080 | 10.000 | 0.600 | 24.619 |
| 0.080 | 10.000 | 0.800 | 24.624 |
| 0.080 | 25.000 | 0.600 | 24.676 |
| 0.080 | 25.000 | 0.800 | 24.640 |
| 0.200 | 10.000 | 0.600 | 24.634 |
| 0.200 | 10.000 | 0.800 | 24.645 |
| 0.200 | 25.000 | 0.600 | 24.691 |
| 0.200 | 25.000 | 0.800 | 24.667 |

RLS grid (2016-20):

| lam | ridge | mae_sum_2016_20 |
|---|---|---|
| 0.999 | 3.0 | 24.740 |
| 0.999 | 30.0 | 24.740 |
| 0.999 | 300.0 | 24.740 |
| 0.9997 | 3.0 | 24.581 |
| 0.9997 | 30.0 | 24.581 |
| 0.9997 | 300.0 | 24.581 |
| 0.99995 | 3.0 | 24.530 |
| 0.99995 | 30.0 | 24.530 |
| 0.99995 | 300.0 | 24.530 |

## Player script

MAE on 2021+, by stat. The oracle columns swap in one true quantity, which shows how much error each part of the chain carries: team-script error, then minutes error, and the rest is usage/efficiency.

| stat | n | mae_script | mae_no_market | mae_last10_avg | mae_if_team_stat_known | mae_if_minutes_known | coverage_1sd |
|---|---|---|---|---|---|---|---|
| ast | 163432 | 1.319 | 1.319 | 1.347 | 1.269 | 1.197 | 0.732 |
| fg3m | 163432 | 0.883 | 0.883 | 0.889 | 0.837 | 0.809 | 0.771 |
| pts | 163432 | 4.619 | 4.608 | 4.712 | 4.531 | 3.774 | 0.702 |
| reb | 163432 | 1.914 | 1.914 | 1.955 | 1.854 | 1.622 | 0.703 |

**Blowout minutes model** (pts rows, 2021+): predicted vs actual minutes relative to the player's EWMA, by role and by quartile of scripted P(|margin|≥15):

| role | bp_bin | n | pred_min_ratio | actual_min_ratio |
|---|---|---|---|---|
| bench | bp Q1 (low) | 21849.000 | 1.000 | 0.991 |
| bench | Q2 | 21899.000 | 1.000 | 0.996 |
| bench | Q3 | 22082.000 | 0.999 | 1.031 |
| bench | Q4 (high) | 22869.000 | 0.999 | 1.083 |
| other starter | bp Q1 (low) | 5687.000 | 1.000 | 1.031 |
| other starter | Q2 | 5816.000 | 0.998 | 1.033 |
| other starter | Q3 | 5968.000 | 0.996 | 1.029 |
| other starter | Q4 (high) | 6261.000 | 0.989 | 1.043 |
| starter 28+ min | bp Q1 (low) | 13339.000 | 0.999 | 1.008 |
| starter 28+ min | Q2 | 13141.000 | 0.996 | 1.005 |
| starter 28+ min | Q3 | 12798.000 | 0.992 | 0.995 |
| starter 28+ min | Q4 (high) | 11723.000 | 0.980 | 0.973 |

## Points props vs real lines (2024-25 = choose, 2025-26 = report)

Projection coverage of prop quotes: 99.9%. Main-line point MAE on unique player-days: {'mae_script': 5.254697598295466, 'mae_script_nm': 5.240886521724436, 'mae_props_model': 5.230706844106427, 'mae_mkt': 5.100090952306518, 'n': 26178}.

Log loss. `p_dv` is the quote's own odds-ratio de-vig. `p_mkt` is the leave-one-book-out consensus. `p_mod` is the existing HGB model. `p_blend` = .75 p_mkt + .25 p_mod. `p_script` / `p_script_nm` are this script, with and without the market-blended team script. `p_mkt_script` = .75 p_mkt + .25 p_script. `p_sel` uses weights (mod, script) = (np.float64(0.05), np.float64(0.05)) chosen on 2025 main-line log loss.

| prob | LL 2025 all | LL 2025 main | LL 2026 all | LL 2026 main |
|---|---|---|---|---|
| p_blend | 0.66517 | 0.69185 | 0.65594 | 0.69162 |
| p_dv | 0.66708 | 0.69185 | 0.65693 | 0.69203 |
| p_mkt | 0.66603 | 0.69156 | 0.65609 | 0.69189 |
| p_mkt_script | 0.66654 | 0.69198 | 0.65574 | 0.69166 |
| p_mod | 0.67805 | 0.70697 | 0.67160 | 0.70723 |
| p_script | 0.68410 | 0.70922 | 0.67150 | 0.70816 |
| p_script_nm | 0.68237 | 0.70721 | 0.66984 | 0.70716 |
| p_sel | 0.66562 | 0.69140 | 0.65565 | 0.69145 |

**ROI at the EV threshold chosen on 2024-25** (n≥300), reported on 2025-26 (`pick_bets`, best quote per player-day):

| prob | thr | roi_2025 | n_2025 | roi_2026 | n_2026 | z_2026 | under_2026 | over_2026 |
|---|---|---|---|---|---|---|---|---|
| p_mkt | 0.040 | 0.042 | 486 | -0.000 | 801 | -0.007 | 0.016 | -0.040 |
| p_mod | 0.100 | -0.012 | 5126 | -0.013 | 7298 | -0.881 | 0.033 | -0.042 |
| p_script | 0.040 | -0.022 | 7836 | -0.011 | 10263 | -0.926 | 0.020 | -0.042 |
| p_script_nm | 0.040 | -0.020 | 7798 | -0.016 | 10221 | -1.389 | 0.019 | -0.053 |
| p_blend | 0.080 | 0.041 | 507 | 0.045 | 1026 | 1.143 | 0.073 | 0.016 |
| p_mkt_script | 0.040 | 0.026 | 2032 | 0.050 | 3393 | 2.384 | 0.101 | -0.009 |
| p_sel | 0.020 | 0.039 | 1378 | 0.034 | 2123 | 1.338 | 0.035 | 0.034 |

Full ROI grid:

| prob | thr | n 2025 | n 2026 | roi 2025 | roi 2026 |
|---|---|---|---|---|---|
| p_blend | 0.020 | 3411.000 | 5060.000 | 0.018 | 0.017 |
| p_blend | 0.040 | 1840.000 | 3059.000 | 0.028 | 0.039 |
| p_blend | 0.060 | 953.000 | 1788.000 | 0.038 | 0.048 |
| p_blend | 0.080 | 507.000 | 1026.000 | 0.041 | 0.045 |
| p_blend | 0.100 | 281.000 | 663.000 | 0.045 | 0.030 |
| p_mkt | 0.020 | 1200.000 | 1875.000 | 0.027 | -0.022 |
| p_mkt | 0.040 | 486.000 | 801.000 | 0.042 | -0.000 |
| p_mkt | 0.060 | 207.000 | 349.000 | -0.019 | 0.000 |
| p_mkt | 0.080 | 93.000 | 144.000 | -0.035 | -0.095 |
| p_mkt | 0.100 | 47.000 | 59.000 | 0.014 | -0.102 |
| p_mkt_script | 0.020 | 3586.000 | 5430.000 | 0.001 | 0.019 |
| p_mkt_script | 0.040 | 2032.000 | 3393.000 | 0.026 | 0.050 |
| p_mkt_script | 0.060 | 1130.000 | 2056.000 | -0.004 | 0.063 |
| p_mkt_script | 0.080 | 579.000 | 1215.000 | -0.041 | 0.054 |
| p_mkt_script | 0.100 | 329.000 | 701.000 | -0.035 | 0.080 |
| p_mod | 0.020 | 8860.000 | 11269.000 | -0.026 | -0.011 |
| p_mod | 0.040 | 7907.000 | 10220.000 | -0.024 | -0.011 |
| p_mod | 0.060 | 6907.000 | 9212.000 | -0.016 | -0.014 |
| p_mod | 0.080 | 5978.000 | 8242.000 | -0.022 | -0.014 |
| p_mod | 0.100 | 5126.000 | 7298.000 | -0.012 | -0.013 |
| p_script | 0.020 | 8851.000 | 11327.000 | -0.026 | -0.015 |
| p_script | 0.040 | 7836.000 | 10263.000 | -0.022 | -0.011 |
| p_script | 0.060 | 6940.000 | 9285.000 | -0.022 | -0.013 |
| p_script | 0.080 | 6051.000 | 8367.000 | -0.029 | -0.010 |
| p_script | 0.100 | 5216.000 | 7516.000 | -0.025 | -0.008 |
| p_script_nm | 0.020 | 8789.000 | 11290.000 | -0.021 | -0.016 |
| p_script_nm | 0.040 | 7798.000 | 10221.000 | -0.020 | -0.016 |
| p_script_nm | 0.060 | 6808.000 | 9192.000 | -0.025 | -0.013 |
| p_script_nm | 0.080 | 5898.000 | 8264.000 | -0.029 | -0.007 |
| p_script_nm | 0.100 | 5103.000 | 7442.000 | -0.024 | -0.003 |
| p_sel | 0.020 | 1378.000 | 2123.000 | 0.039 | 0.034 |
| p_sel | 0.040 | 553.000 | 936.000 | 0.028 | 0.027 |
| p_sel | 0.060 | 214.000 | 403.000 | -0.070 | 0.037 |
| p_sel | 0.080 | 97.000 | 157.000 | 0.121 | 0.026 |
| p_sel | 0.100 | 43.000 | 80.000 | 0.231 | -0.056 |
## Conclusions (honest)

1. **The closing line wins.** On 2021+ the enriched script misses the margin by 10.96 points on average, against 10.57 for the close, and the total by 14.67 against 14.21. The learned Bates-Granger weight sits at about 0.9-1.0 on the market. So the enrichments do not change the smoke-test result.
2. **The enrichments help the script, a little.** Versus the bare core on 2021+: margin 11.04 → 10.96, total 14.78 → 14.67. Most of the margin gain comes from **injuries** (rotation players who missed the previous game) and **rest/b2b**. Dropping injuries costs +0.041 margin MAE, and dropping rest costs +0.039. Totals gain from the **possession model** (+0.018 if dropped), injuries (+0.012) and matchup terms (+0.010; the FT-rate-vs-fouling-D and pace-clash terms). Travel and time zone add nothing; dropping them is neutral. All gains are small next to the 0.4-pt margin gap to the market. The gap is mostly same-day lineup news, which the previous game's box cannot see.
3. **Nothing beats the close out of sample.** The residual correlation with the closing error is about 0. The market-residual learner (market + online correction) is slightly worse than the raw close: 10.59 vs 10.57. ATS and O/U bets at the close lose on 2021+ (−0.9% to −7%). The one exception was the 2020 bubble season, where the blend (10.13) beat the close (10.18) and the learned weight fell to 0.56. That is the only time the engine "learned" the market was beatable, and it reverted in 2021.
4. **The script does anticipate line movement.** corr(script − open, close − open) is +0.17 for spreads and +0.31 for totals. Betting the opener with the 2016-20 rule gave 2021+ ROI of +2.0% on spreads (n=695, CLV +0.5 pts) and +1.0% on totals (n=295, CLV +2.6 pts). Neither is significant, and openers are missing for 2023-25. This is the most promising direction (bet early, not at the close), but it is unproven.
5. **Players.** Share × scripted team stat × minutes beats a last-10 average on every stat. The share-based points projection (main-line MAE 5.25) is slightly worse than the existing HGB model (5.23) and the market (5.10). The oracle attribution says **minutes** carry most of the player error: points MAE drops 4.62 → 3.77 if minutes were known, but only → 4.53 if the team total were known. The blowout-minutes model gets the direction right for heavy-minute starters: predicted 0.980 vs actual 0.973 of normal minutes in the top blowout quartile. It misses the bench minutes bump.
6. **Props.** On log loss the script is far worse than the market. As a 25% blend partner it ties the existing blend: .75 p_mkt + .25 p_script scores 2026 main-line LL .69166 vs .69162 for p_blend. The best weight chosen on 2025 (90/5/5 mkt/mod/script) is best on 2026 (.69145 vs .69189 for p_mkt). On ROI, p_mkt_script at the 2025-chosen EV≥0.04 made +2.6% (n=2032) in 2025 and **+5.0% (n=3393, z=2.4) in 2026**. The existing p_blend at its chosen threshold made +4.1% / +4.5% (n=507 / 1026), and at the same 0.04 threshold +2.8% / +3.9%. The edge is again almost entirely **unders**: 2026 unders +10.1%, overs −0.9%. The script blend did not beat p_blend in the selection season, so it has not earned a switch. It does make a reasonable second opinion for under bets. Wait for more holdout data.

