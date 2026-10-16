# NCAAM script-then-learn: learning report

Code: `engine/scripts/ncaam.py`. The core is `base.ScriptEngine` maths through `FastEngine`, which has an identical update with O(1) window sums. Games run sequentially from 2013, which is warm-up; there is no 2012 file. Each date is scripted fully before it is learned. Tuning used 2016-20, and 2021+ is reported. **2024 and 2025 have no betting lines in `ncaam.csv`**, so market comparisons use 2021-23 + 2026. Lines cover about 68% of games in 2016-18 and 92% from 2019 on, mostly the D1-vs-D1 games.

Tuned core: {'q_game': 0.3, 'carry': 0.9}. RLS: λ=0.9999, ridge=100.0.

## Bottom line

- **2021+ games with lines (n=19864), margin MAE:** core 9.232, enriched 9.258, blend 8.834, market + learned residual 8.838, **closing line 8.838**.
- **Total MAE:** core 13.362, enriched 13.299, blend 13.076, market + residual 13.079, **closing line 13.072**.
- **Learned market weight, mean over 2021+:** margin 0.90, total 0.83.
- **Information beyond the close:** corr(script − close, actual − close) is 0.041 for margin and 0.013 for total.
- **Line movement on games with openers (4751):** corr(script − open, close − open) is 0.303 for spreads and 0.265 for totals.

**Betting the close at −110** (threshold chosen on 2016-20 with n≥300):

| bet | thr | roi_2016_20 | n_2021p | roi_2021p |
|---|---|---|---|---|
| spread: script | 8 | 0.002 | 1052 | 0.049 |
| spread: mkt+resid | 1 | -0.032 | 1818 | 0.012 |
| total: script | 3 | 0.010 | 6932 | -0.017 |
| total: mkt+resid | 1 | -0.011 | 6596 | -0.014 |

## Learning curve

All games (script only):

| season | games | margin_core | margin_script | total_core | total_script |
|---|---|---|---|---|---|
| 2016 | 5891 | 9.302 | 9.278 | 14.024 | 13.995 |
| 2017 | 5926 | 9.453 | 9.430 | 13.669 | 13.564 |
| 2018 | 6003 | 9.677 | 9.647 | 13.814 | 13.775 |
| 2019 | 6049 | 9.652 | 9.646 | 13.654 | 13.603 |
| 2020 | 5766 | 9.763 | 9.766 | 13.561 | 13.491 |
| 2021 | 4285 | 9.838 | 9.884 | 13.614 | 13.537 |
| 2022 | 5965 | 9.634 | 9.642 | 13.282 | 13.207 |
| 2023 | 6221 | 9.766 | 9.738 | 13.244 | 13.178 |
| 2024 | 6242 | 9.752 | 9.763 | 13.376 | 13.314 |
| 2025 | 6291 | 9.851 | 9.869 | 13.082 | 13.014 |
| 2026 | 6300 | 10.021 | 10.032 | 13.931 | 13.863 |

Games with closing lines (`_l` = script on the same games):

| season | games_w_line | margin_script_l | margin_blend_l | margin_market | total_script_l | total_blend_l | total_market | w_mkt_m | w_mkt_t | ml_ll_script | ml_ll_blend | ml_ll_market |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 3966 | 8.748 | 8.347 | 8.344 | 13.733 | 13.525 | 13.513 | 1.000 | 0.756 | nan | nan | nan |
| 2017 | 3973 | 9.102 | 8.704 | 8.702 | 13.328 | 13.173 | 13.189 | 1.000 | 0.617 | nan | nan | nan |
| 2018 | 4069 | 9.070 | 8.704 | 8.701 | 13.656 | 13.511 | 13.510 | 1.000 | 0.651 | 0.552 | 0.538 | 0.534 |
| 2019 | 5523 | 9.176 | 8.789 | 8.783 | 13.507 | 13.329 | 13.333 | 1.000 | 0.541 | 0.552 | 0.534 | 0.521 |
| 2020 | 5327 | 9.225 | 8.774 | 8.765 | 13.338 | 13.152 | 13.152 | 0.879 | 0.665 | 0.547 | 0.530 | 0.522 |
| 2021 | 3923 | 9.481 | 9.051 | 9.045 | 13.433 | 13.117 | 13.108 | 0.814 | 0.795 | 0.562 | 0.547 | 0.545 |
| 2022 | 5476 | 9.117 | 8.634 | 8.636 | 13.154 | 12.953 | 12.945 | 0.962 | 0.870 | 0.544 | 0.525 | 0.521 |
| 2023 | 5714 | 9.222 | 8.852 | 8.867 | 13.091 | 12.941 | 12.960 | 0.903 | 0.364 | 0.557 | 0.544 | 0.539 |
| 2026 | 4751 | 9.279 | 8.865 | 8.864 | 13.607 | 13.345 | 13.323 | 0.795 | 0.975 | 0.570 | 0.553 | 0.548 |

**Within-season learning:** margin MAE by the fewer of the two teams' games played that season (2016+, games with lines):

| gnum | n | core | script | market |
|---|---|---|---|---|
| 0-3 | 4651 | 10.141 | 10.217 | 9.404 |
| 4-8 | 6416 | 9.612 | 9.630 | 8.953 |
| 9-15 | 9501 | 9.299 | 9.312 | 8.773 |
| 16-25 | 13914 | 8.711 | 8.720 | 8.545 |
| 26+ | 8240 | 8.730 | 8.783 | 8.543 |

## Error attribution

Per-team errors split into a **shared** part ((e_h + e_a)/2; game environment, tempo, whistle, shooting variance common to both) with sd 8.48, and a **split** part ((e_h − e_a)/2; relative strength / efficiency) with sd 6.32. Their correlation is 0.05. The learned score correlation rho is 0.28.

**By game type (2021+, games with lines).** `gap` is script MAE − market MAE:

| group | n | mae_margin_script | mae_margin_mkt | gap_margin | mae_total_script | mae_total_mkt | bias_margin_script | bias_total_script |
|---|---|---|---|---|---|---|---|---|
| |spread|≤4 | 6936 | 9.106 | 8.756 | 0.350 | 13.471 | 13.297 | 1.087 | 0.125 |
| 4-9 | 6962 | 9.242 | 8.768 | 0.474 | 13.209 | 13.008 | 1.269 | 0.252 |
| 9-15 | 3773 | 9.081 | 8.601 | 0.480 | 13.190 | 12.895 | 0.685 | 0.307 |
| >15 | 2193 | 10.093 | 9.725 | 0.368 | 13.228 | 12.868 | -1.277 | -0.583 |
| first 8 games | 4535 | 10.078 | 9.327 | 0.751 | 13.752 | 13.305 | 1.190 | 0.699 |
| later | 15329 | 9.015 | 8.693 | 0.322 | 13.165 | 13.003 | 0.702 | -0.044 |
| conference | 13725 | 8.976 | 8.672 | 0.303 | 13.136 | 13.000 | 0.791 | -0.121 |
| non-conference | 5702 | 9.909 | 9.203 | 0.706 | 13.680 | 13.245 | 0.993 | 0.680 |
| postseason | 437 | 9.622 | 9.261 | 0.361 | 13.472 | 13.070 | -0.842 | 0.661 |
| home/away | 17369 | 9.239 | 8.797 | 0.442 | 13.293 | 13.073 | 1.015 | 0.119 |
| neutral | 2495 | 9.388 | 9.122 | 0.266 | 13.346 | 13.069 | -0.591 | 0.173 |

**Home-team conference, worst and best relative to market** (2021+, n≥150):

| home_conference | n | gap_margin | bias_margin |
|---|---|---|---|
| NEC | 436.000 | 0.719 | 2.260 |
| Am. East | 439.000 | 0.661 | 2.155 |
| WAC | 511.000 | 0.602 | 0.165 |
| MAAC | 584.000 | 0.585 | 1.975 |
| Big East | 739.000 | 0.555 | 0.313 |
| MAC | 622.000 | 0.286 | 2.060 |
| C-USA | 739.000 | 0.232 | 0.083 |
| SEC | 1048.000 | 0.220 | -0.553 |
| SWAC | 474.000 | 0.191 | 0.641 |
| Summit | 462.000 | 0.176 | 1.115 |

**Final RLS coefficients** (points per 1 sd feature, for team T's score):

| feature | coef |
|---|---|
| bias | 0.722 |
| is_home | 0.176 |
| neutral | -1.365 |
| playoff | 0.935 |
| d_mult | 0.511 |
| tempo_clash | 0.039 |
| tempo_sum | -0.481 |
| d_ewm | 0.433 |
| conf_m_T | 0.902 |
| conf_m_O | -0.732 |
| conf_t_T | 0.106 |
| conf_t_O | -0.111 |
| same_conf | -0.659 |
| conf_early | -0.363 |
| rest_T | -0.323 |
| rest_O | 0.477 |

## Ablation (MAE; online correction re-run without each group)

| config | margin all 2016-20 | margin all 2021+ | margin with lines 2016-20 | margin with lines 2021+ | total all 2016-20 | total all 2021+ | total with lines 2016-20 | total with lines 2021+ |
|---|---|---|---|---|---|---|---|---|
| base + conf only | 9.573 | 9.812 | 9.061 | 9.241 | 13.742 | 13.400 | 13.559 | 13.356 |
| base + form only | 9.568 | 9.823 | 9.056 | 9.237 | 13.721 | 13.381 | 13.529 | 13.324 |
| base + rest only | 9.553 | 9.800 | 9.087 | 9.254 | 13.735 | 13.398 | 13.553 | 13.352 |
| base + tempo only | 9.601 | 9.850 | 9.089 | 9.266 | 13.684 | 13.356 | 13.497 | 13.308 |
| core only | 9.570 | 9.811 | 9.053 | 9.232 | 13.745 | 13.412 | 13.563 | 13.362 |
| drop conf | 9.563 | 9.829 | 9.095 | 9.268 | 13.683 | 13.346 | 13.498 | 13.299 |
| drop form | 9.579 | 9.829 | 9.110 | 9.278 | 13.682 | 13.347 | 13.500 | 13.306 |
| drop rest | 9.576 | 9.837 | 9.060 | 9.247 | 13.690 | 13.349 | 13.503 | 13.304 |
| drop tempo | 9.535 | 9.797 | 9.062 | 9.237 | 13.715 | 13.370 | 13.530 | 13.320 |
| full | 9.553 | 9.820 | 9.081 | 9.258 | 13.686 | 13.344 | 13.502 | 13.299 |

Closing line on the 2021+ games with lines: margin 8.838, total 13.072.

Core grid (2016-20):

| q_game | carry | mae_sum_2016_20 |
|---|---|---|
| 0.050 | 0.600 | 23.693 |
| 0.150 | 0.800 | 23.383 |
| 0.300 | 0.900 | 23.315 |
| 0.600 | 0.900 | 23.356 |
| 1.000 | 0.900 | 23.434 |
| 0.300 | 0.970 | 23.364 |
| 0.600 | 0.970 | 23.359 |
| 1.000 | 0.970 | 23.417 |

RLS grid:

| lam | ridge | mae_sum_2016_20 |
|---|---|---|
| 1.000 | 10.000 | 2497.058 |
| 1.000 | 100.000 | 1256.750 |
| 1.000 | 10.000 | 23.239 |
| 1.000 | 100.000 | 23.239 |
## Conclusions (honest)

1. **The closing line wins.** On 2021+ games with lines the script's margin MAE is 9.26 against 8.84 for the close, and its total MAE is 13.30 against 13.07. The learned market weight is about 0.8-1.0, and the blend only ties the close (8.834 / 13.076).
2. **The score-derived enrichments help totals, not margins.** Versus the bare core, total MAE improves 13.36 → 13.30, from the multiplicative tempo model (−0.05 alone) and EWMA form. Margin gets slightly worse (9.23 → 9.26 on lined games). Per-team Kalman ratings already hold the efficiency signal, so the margin features mostly add noise. On ALL games, including unlined D1-vs-non-D1, rest and dropping tempo help a little (9.80 vs 9.81).
3. **Where the engine is wrong most:** early season and non-conference games. In the first 8 games the gap to the market is +0.75 pts, against +0.32 later. In non-conference games it is +0.71, against +0.30 in conference play. The market knows rosters, transfers and preseason ratings; carry-over Kalman ratings don't. There is also a persistent home-margin bias of about +1 pt in low-major conferences (NEC, America East, MAAC, MAC). Shared, game-environment error (sd 8.5) is bigger than the split, strength error (sd 6.3).
4. **Does anything beat the close?** The residual correlation is 0.04 for margins and 0.01 for totals. Spread bets at the close with |script − close| ≥ 8 (rule picked on 2016-20, where it was +0.2%) returned **+5.0% on 1,056 bets in 2021+ (z≈1.7)**. That is not significant, and it is a fade-the-market rule in mismatches. Totals lose. The script anticipates line moves strongly (corr(script − open, close − open) = +0.30 spreads, +0.27 totals). As in the NBA, the realistic angle is early lines, not the close.

