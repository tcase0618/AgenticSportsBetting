# NFL script-then-learn: learning report

Code: `engine/scripts/nfl.py` (base: `engine/scripts/base.py`, unmodified). Ratings warmed up from 2010; report covers 2016-2026 (2026 = weeks played so far). Every pre-game number uses only games on earlier dates. Kalman params, ridge lambda, sd exponents chosen on 2016-2020 (sd exponents on pre-2016); everything else is learned online from each game's error. Runtime 6.7 min.

Kalman params chosen on 2016-20: {'q_game': np.float64(0.15), 'carry': np.float64(0.5), 'prior_var': np.float64(8.0)}; correction ridge lambda=1000.0 (grid [(100.0, np.float64(14.908)), (300.0, np.float64(14.888)), (1000.0, np.float64(14.871)), (3000.0, np.float64(14.887)), (10000.0, np.float64(14.94))]).

## 1. Per-season learning curve (MAE, points)

`script` = Kalman + online NFL correction; `blend` = learned Bates-Granger blend with the close; `market` = closing spread / total.

| season | n | margin_script | margin_blend | margin_market | total_script | total_blend | total_market | w_mkt |
|---|---|---|---|---|---|---|---|---|
| 2016 | 267 | 9.237 | 9.048 | 9.043 | 10.131 | 9.821 | 9.794 | 0.777 |
| 2017 | 267 | 10.331 | 10.086 | 10.081 | 11.490 | 11.260 | 11.257 | 0.909 |
| 2018 | 267 | 9.938 | 9.866 | 9.875 | 11.276 | 10.744 | 10.719 | 0.960 |
| 2019 | 267 | 10.448 | 10.178 | 10.176 | 10.975 | 10.786 | 10.785 | 1.000 |
| 2020 | 269 | 10.102 | 9.790 | 9.790 | 10.725 | 10.303 | 10.303 | 0.989 |
| 2021 | 285 | 10.948 | 10.665 | 10.667 | 10.930 | 10.808 | 10.807 | 0.989 |
| 2022 | 284 | 9.213 | 8.783 | 8.783 | 10.782 | 10.398 | 10.398 | 1.000 |
| 2023 | 285 | 10.461 | 9.984 | 9.984 | 10.428 | 10.168 | 10.168 | 1.000 |
| 2024 | 285 | 10.137 | 9.705 | 9.704 | 10.073 | 9.770 | 9.768 | 0.994 |
| 2025 | 285 | 10.237 | 9.675 | 9.670 | 10.474 | 10.423 | 10.423 | 0.990 |
| 2026 | 79 | 9.729 | 9.780 | 9.778 | 11.272 | 10.631 | 10.627 | 1.000 |


Base Kalman script alone (no NFL enrichment):

| season | margin_script | total_script | margin_market | total_market |
|---|---|---|---|---|
| 2016 | 9.397 | 10.083 | 9.043 | 9.794 |
| 2017 | 10.515 | 11.698 | 10.081 | 11.257 |
| 2018 | 9.967 | 11.200 | 9.875 | 10.719 |
| 2019 | 10.464 | 10.955 | 10.176 | 10.785 |
| 2020 | 10.038 | 10.811 | 9.790 | 10.303 |
| 2021 | 11.100 | 11.146 | 10.667 | 10.807 |
| 2022 | 9.063 | 10.798 | 8.783 | 10.398 |
| 2023 | 10.541 | 10.592 | 9.984 | 10.168 |
| 2024 | 10.200 | 10.078 | 9.704 | 9.768 |
| 2025 | 10.192 | 10.641 | 9.670 | 10.423 |
| 2026 | 9.851 | 11.445 | 9.778 | 10.627 |


Winner log loss (script P(home win) = Phi(margin/sd_margin); market = odds_ratio de-vigged ML):

|  | n | script | blend | market |
|---|---|---|---|---|
| 2016-20 | 1330 | 0.6238 | 0.6093 | 0.6067 |
| 2021-26 | 1498 | 0.6372 | 0.6139 | 0.6122 |


## 2. Does anything beat the closing line?

| target | period | mae_script | mae_market | slope_beyond_close | t |
|---|---|---|---|---|---|
| margin (side) | 2016-20 | 10.011 | 9.793 | 0.219 | 1.860 |
| margin (side) | 2021-26 | 10.175 | 9.763 | -0.081 | -0.736 |
| total | 2016-20 | 10.919 | 10.571 | -0.237 | -1.821 |
| total | 2021-26 | 10.576 | 10.329 | 0.009 | 0.075 |


Betting |script - close| >= k points at -110 (k chosen on 2016-20):

- margin: k chosen on 2016-20 = 1 (n=908, ROI +0.9% at -110); **2021-26: n=1049, ROI -7.2%**. All k on 2021-26: k=1: -7.2% (n=1049), k=2: -8.5% (n=684), k=3: -6.0% (n=406), k=4: -9.2% (n=225), k=5: -3.0% (n=124), k=6: -6.1% (n=63)
- total: k chosen on 2016-20 = 4 (n=207, ROI -5.9% at -110); **2021-26: n=224, ROI -2.8%**. All k on 2021-26: k=1: -4.9% (n=1056), k=2: -2.5% (n=687), k=3: -1.3% (n=408), k=4: -2.8% (n=224), k=5: -4.5% (n=110), k=6: +1.3% (n=49)

## 3. Error attribution: what did the script get wrong?

Team-points error decomposed as pace (play-count miss x scripted pts/play) + efficiency (actual plays x pts/play miss), team-games 2016+: mean |pace part| = 2.42 pts, mean |efficiency part| = 7.48 pts; share of error variance: pace 11%, efficiency 101% (cov term -12%). Play-count MAE 6.6 plays/team.

Variance of team-points error explained by realised surprises (R^2, single variable): turnovers committed 8%, pass-EPA miss (script vs realised EPA/dropback) 44%, play-count miss 2%. Efficiency (EPA) and turnovers dominate; pace/volume is a small part.

Team-points bias (mean script - actual) and MAE by environment:

| wind mph (outdoor) | n | bias | mae |
|---|---|---|---|
| <10 | 4440 | -0.17 | 7.49 |
| 10-14 | 832 | 0.54 | 7.06 |
| 15-19 | 322 | 0.98 | 7.61 |
| 20+ | 84 | -0.61 | 8.29 |

| |closing spread| | n | bias | mae |
|---|---|---|---|
| 0-3 | 2120 | 0.49 | 7.62 |
| 3.5-7 | 2256 | -0.31 | 7.28 |
| 7.5-10 | 752 | -0.69 | 7.45 |
| 10.5+ | 552 | 0.24 | 7.38 |

| starter QB vs team pass EPA | n | bias | mae |
|---|---|---|---|
| QB worse than team | 1526 | 0.18 | 7.90 |
| same | 3158 | -0.04 | 7.36 |
| QB better | 996 | -0.19 | 7.01 |


Game totals by wind, script vs market:

| wind | n | total_bias_script | total_mae_script | total_bias_market | total_mae_market |
|---|---|---|---|---|---|
| <10 | 2220 | -0.35 | 10.89 | -0.95 | 10.61 |
| 10-14 | 416 | 1.09 | 9.76 | 1.19 | 9.56 |
| 15+ | 203 | 1.31 | 11.14 | 1.89 | 10.44 |


## 4. Ablation of NFL enrichments (2021-26, MAE; online loop re-run each time)

`pts` = team points MAE, `pass_att` = home pass attempts MAE (volume script). Lower is better.

| variant | pts | margin | total | margin_b | total_b | pass_att |
|---|---|---|---|---|---|---|
| base Kalman only | 7.488 | 10.201 | 10.692 | 9.763 | 10.329 | nan |
| all enrichments | 7.446 | 10.175 | 10.576 | 9.764 | 10.330 | 6.114 |
| drop eff | 7.456 | 10.190 | 10.604 | 9.763 | 10.330 | 6.120 |
| drop pace | 7.448 | 10.182 | 10.580 | 9.764 | 10.330 | 6.119 |
| drop proe | 7.444 | 10.171 | 10.574 | 9.764 | 10.330 | 6.116 |
| drop form | 7.433 | 10.148 | 10.570 | 9.765 | 10.329 | 6.101 |
| drop defn | 7.440 | 10.169 | 10.576 | 9.764 | 10.330 | 6.132 |
| drop matchup | 7.444 | 10.164 | 10.573 | 9.764 | 10.330 | 6.101 |
| drop qb | 7.478 | 10.219 | 10.626 | 9.763 | 10.330 | 6.117 |
| drop env | 7.462 | 10.163 | 10.627 | 9.763 | 10.329 | 6.112 |
| only eff | 7.476 | 10.179 | 10.672 | 9.763 | 10.329 | 6.122 |
| only pace | 7.484 | 10.196 | 10.699 | 9.763 | 10.329 | 6.169 |
| only proe | 7.488 | 10.197 | 10.696 | 9.763 | 10.329 | 6.157 |
| only form | 7.495 | 10.220 | 10.698 | 9.763 | 10.329 | 6.199 |
| only defn | 7.497 | 10.200 | 10.712 | 9.763 | 10.329 | 6.185 |
| only matchup | 7.487 | 10.215 | 10.698 | 9.763 | 10.329 | 6.147 |
| only qb | 7.455 | 10.147 | 10.640 | 9.763 | 10.329 | 6.182 |
| only env | 7.473 | 10.203 | 10.667 | 9.763 | 10.329 | 6.187 |


Team-points MAE change when each group is DROPPED (positive = the group helps): qb +0.032, env +0.016, eff +0.010, pace +0.002, matchup -0.002, proe -0.002, defn -0.006, form -0.013. All enrichments vs base Kalman: -0.043 pts/team, total -0.117, margin -0.026. None of this moves the market-blended script (margin_b/total_b), because the learned blend weight sits at ~1.0 on the close.


Final online-ridge weights on standardized features (points per 1 sd), largest |w|:

qb_exp +0.42, wind -0.36, qb_delta +0.36, o_sr +0.29, wind15 -0.24, rest_diff +0.22, o_epa_pass +0.21, d_epa_pass -0.21, o_spp -0.20, qb_new -0.18, dome +0.17, o_plays -0.17, m_adot_vs_man +0.15, div -0.15


## 5. Player script

mu = online usage share (targets blended with air-yards share; carries; QB attempt share) x scripted team pass/rush attempts (online volume script fed by the market-informed game script) x player efficiency (shrunk, decayed) x opponent allowed-vs-league factor x online calibration. sd: var = k*mu^p, p per stat chosen on pre-2016 ({'completions': 1.0, 'passing_yards': 1.0, 'receiving_yards': 1.0, 'receptions': 1.0, 'rushing_yards': 1.25}), k learned online. Only players with >=1 prior game; QB = listed starter.

| stat | n_2016-20 | n_2021-26 | mae_2016-20 | mae_2021-26 | bias_2016-20 | bias_2021-26 | corr_2016-20 | corr_2021-26 | z_sd_2016-20 | z_sd_2021-26 | cover90_2016-20 | cover90_2021-26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| completions | 2649 | 2937 | 5.26 | 5.22 | -0.10 | -0.11 | 0.28 | 0.31 | 1.01 | 0.98 | 0.95 | 0.94 |
| passing_yards | 2649 | 2937 | 64.33 | 63.31 | 0.26 | -1.04 | 0.31 | 0.31 | 0.98 | 0.98 | 0.95 | 0.93 |
| receiving_yards | 22596 | 27010 | 19.00 | 17.26 | 0.12 | 0.20 | 0.58 | 0.62 | 0.98 | 0.99 | 0.93 | 0.93 |
| receptions | 22596 | 27010 | 1.41 | 1.32 | 0.00 | 0.01 | 0.59 | 0.63 | 0.98 | 1.00 | 0.93 | 0.93 |
| rushing_yards | 7189 | 8264 | 20.49 | 19.66 | -0.33 | 0.23 | 0.61 | 0.64 | 0.99 | 0.99 | 0.94 | 0.94 |


Player ablation (2021-26 MAE by stat):

|  | completions | passing_yards | receiving_yards | receptions | rushing_yards |
|---|---|---|---|---|---|
| full script | 5.22 | 63.31 | 17.26 | 1.32 | 19.66 |
| no air-yards share | 5.22 | 63.31 | 17.23 | 1.32 | 19.66 |
| no opponent adj | 5.22 | 63.24 | 17.21 | 1.32 | 19.65 |
| no team script (player's own EWMA) | 5.39 | 65.47 | 17.28 | 1.32 | 19.68 |
| team volume = raw tendencies (no game-script/correction) | 5.21 | 63.10 | 17.32 | 1.33 | 19.70 |


## 6. Player props vs real lines

| Set / market | n | log loss model | log loss market (odds_ratio) | log loss 50/50 blend | Brier model / market |
|---|---|---|---|---|---|
| 2017-18 completions / completions | 935 | 0.7469 | 0.6939 | 0.6954 | 0.2600 / 0.2504 |
| 2026 wk1-5 / passing_yards | 87 | 0.7253 | 0.6931 | 0.7012 | 0.2656 / 0.2500 |
| 2026 wk1-5 / receiving_yards | 851 | 0.7273 | 0.6925 | 0.6954 | 0.2631 / 0.2497 |
| 2026 wk1-5 / receptions | 519 | 0.6970 | 0.6827 | 0.6779 | 0.2498 / 0.2448 |
| 2026 wk1-5 / rushing_yards | 279 | 0.7385 | 0.6934 | 0.6967 | 0.2670 / 0.2501 |

ROI at EV thresholds. Threshold and model weight w (p = w*model + (1-w)*market) chosen on the earliest data (2017 for completions; 2026 weeks 1-2 for 2026), then applied to the test window (2018; 2026 weeks 3-5). One bet per player-game-market at median book prices.

| Set | chosen w, EV thr | calib n / ROI | TEST n / ROI / z |
|---|---|---|---|
| 2017-18 completions | w=0.25, EV>=8% | 26 / +16.7% | 25 / +45.3% / +2.77 |
| 2026 wk1-5 | w=0.75, EV>=8% | 214 / -0.7% | 613 / +2.6% / +0.68 |

Pure model (w=1), EV>=5%, test windows, by market: 2017-18 completions/completions: n=203, ROI +3.1%; 2026 wk1-5/passing_yards: n=50, ROI -13.3%; 2026 wk1-5/receiving_yards: n=375, ROI +2.9%; 2026 wk1-5/receptions: n=323, ROI +8.9%; 2026 wk1-5/rushing_yards: n=135, ROI +10.0%

## 7. Caveats

- Player rows exist only for players who appear in that game's box score (and QB = listed starter); that is a mild selection on availability, similar to props only being offered on active players.
- Volume scripts (pass/rush attempts) use the market-informed blended margin/total as game state (the closing line is pre-game information); the points script itself never uses the market.
- 2017-18 completions test window selected only ~25 bets (min-n rule of 20 on the 2017 calibration set fixed in advance); treat its ROI as noise-level. 2026 is 5 weeks.
- Participation data is missing for 2026 (not yet published), so formation/personnel/box/pressure tendencies carry 2025 values into 2026 via decay; FTN (play action, motion, blitz) covers 2022+.
