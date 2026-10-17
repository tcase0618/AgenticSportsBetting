# Situational edge hunt (hunter: situational)

Code: `engine/edge_hunt/situational.py` · all tests: `results/edge_hunt/situational_candidates.csv` · discovery thresholds: `results/edge_hunt/situational_thresholds.json`.

## Design
* Data: `results/scripts/<sport>_games.parquet` (Kalman script margin/total/win prob, rest, b2b, travel, timezone, missing minutes, QB name/QB delta, SP & park effects, goalie effect & backup flag, pens, EPA, weather, conference) joined to `results/predict/<sport>_preds.parquet` (walk-forward model; available 2018+). Seasons 2016-2026.
* Discovery = seasons <= 2022, lockbox = 2023+. Continuous thresholds = discovery-only quantiles.
* Rules = 1 or 2 atoms (conjunction) x market (side ML, side ATS, game total) x direction. Side rules are written from the perspective of the team bet on; the atom set is mirror-symmetric so every fade is the 'back' of the mirrored atoms (family A lists fades explicitly).
* Family A (targeted, pre-registered): script-vs-close disagreement follow/fade at fixed thresholds per sport and market; NHL backup-goalie spots; NFL QB-change spots; NBA heavy-missing-minutes spots; NBA/NHL rest-travel-timezone stacks; MLB new SP / bullpen; NCAA rest-in-conference.
* Family B (broad): every single atom and every pair of atoms per sport.
* Screening: n >= {'nba': 200, 'nhl': 200, 'mlb': 300, 'ncaam': 300, 'nfl': 80, 'ncaaf': 100} (discovery), ROI > 0, Benjamini-Hochberg q < 0.10 within family on one-sided discovery p, and >= 60% of discovery seasons positive. Survivors get the lockbox exactly once.
* Prices: ML = real closing prices (quotes with |odds|>5000 or overround outside 1.00-1.15 dropped). **Spreads and totals for NBA/NFL/NCAAF/NCAAM carry no juice in the data -> -110 ASSUMED.** NHL totals: real over/under prices 2024+, -110 assumed before. MLB totals: real prices 2022+, -110 assumed before. MLB run line and NHL puck line not tested (no juice in data). **MLB 2021+ closing prices failed an integrity check (results/KEYS.md) — any MLB lockbox number is provisional.** NCAAM has no lines in 2024-25, so its lockbox is 2023 + 2026.

## How many rules were screened

**n_tests = 4354** rules logged (A_targeted: 198, B_broad: 4156); 2596 met the minimum-n requirement and entered BH.

| sport   |   A_targeted |   B_broad |
|:--------|-------------:|----------:|
| mlb     |           22 |       507 |
| nba     |           52 |      1060 |
| ncaaf   |           28 |       724 |
| ncaam   |           28 |       548 |
| nfl     |           38 |       816 |
| nhl     |           30 |       501 |

Screen survivors (BH q<0.10, ROI>0): 0; dropped for discovery-season instability: 0; final lockbox candidates: 0.

## Verdicts — final candidates (survived discovery screen)

None: no rule survived the discovery screen.

## Informational near-misses (top-5 discovery z per family that FAILED the screen; lockbox shown for context only — cannot be TRUE)

| family | sport | market | rule (bet) | disc n | disc ROI | disc z | disc q | lock n | lock ROI | lock z | lock p | lock BH q | units | max DD | lock -110 assumed | seasons + (full) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_targeted | nhl | back ML | follow script ML edge >= 0.05 | 2331 | +4.4% | 1.88 | 1.000 | 1244 | -2.1% | -0.69 | 0.7555 |  | -25.5 | 38.9 | 0% | 6/12 | FALSE |
| A_targeted | nhl | over | follow script total >= line+0.8 | 217 | +10.8% | 1.75 | 1.000 | 55 | +13.7% | 1.15 | 0.1246 |  | +7.5 | 6.1 | 9% | 10/12 | FALSE |
| A_targeted | nhl | back ML | follow script ML edge >= 0.10 | 297 | +10.4% | 1.54 | 1.000 | 185 | -6.8% | -0.89 | 0.8128 |  | -12.5 | 19.6 | 0% | 6/12 | FALSE |
| A_targeted | ncaaf | under | follow script total <= line-9 | 131 | +11.5% | 1.40 | 1.000 | 25 | +22.2% | 1.19 | 0.1178 |  | +5.5 | 3.0 | 100% | 7/11 | FALSE |
| A_targeted | nhl | over | follow script total >= line+0.5 | 1212 | +3.5% | 1.34 | 1.000 | 424 | -3.2% | -0.72 | 0.7634 |  | -13.4 | 24.6 | 7% | 8/12 | FALSE |
| B_broad | nhl | back ML | script>mkt top10% & model>mkt top10% | 539 | +11.7% | 2.62 | 1.000 | 507 | -8.3% | -2.07 | 0.9806 |  | -42.0 | 53.4 | 0% | 4/10 | FALSE |
| B_broad | nhl | over | script total>line top20% & line bot10% | 402 | +10.4% | 2.55 | 1.000 | 5 | -25.8% | -0.57 | 0.7150 |  | -1.3 | 2.0 | 0% | 6/8 | FALSE |
| B_broad | nfl | back ML | script<mkt bot20% & model>mkt top10% | 109 | +8.8% | 2.48 | 1.000 | 132 | -6.5% | -1.37 | 0.9139 |  | -8.6 | 10.4 | 0% | 5/9 | FALSE |
| B_broad | nfl | under | wind>=12 | 304 | +13.1% | 2.43 | 1.000 | 114 | +5.6% | 0.63 | 0.2641 |  | +6.4 | 8.8 | 100% | 7/11 | FALSE |
| B_broad | nhl | back ML | model>mkt top10% & o b2b | 249 | +11.6% | 2.26 | 1.000 | 300 | +1.9% | 0.43 | 0.3340 |  | +5.7 | 9.0 | 0% | 6/10 | FALSE |

## Family A (targeted) — discovery results for every pre-registered rule

| sport | market | bet | rule | disc n | disc ROI | disc z | BH q |
|---|---|---|---|---|---|---|---|
| mlb | ml | back | follow script ML edge >= 0.15 | 77 | +61.5% | 3.25 |  |
| mlb | tot | under | follow script total <= line-1.5 | 91 | +9.6% | 0.97 |  |
| mlb | tot | under | follow script total <= line-1.0 | 340 | +2.4% | 0.47 | 1.000 |
| mlb | ml | back | follow script ML edge >= 0.10 | 708 | +1.0% | 0.21 | 1.000 |
| mlb | ml | back | back team vs debut/new SP | 1556 | -0.1% | -0.05 | 1.000 |
| mlb | ml | back | back opp bullpen taxed (top10%) + own pen fresh | 1021 | -0.6% | -0.21 | 1.000 |
| mlb | tot | under | follow script total <= line-0.5 | 1312 | -0.6% | -0.22 | 1.000 |
| mlb | tot | over | follow script total >= line+1.5 | 1029 | -2.4% | -0.83 | 1.000 |
| mlb | ml | fade | fade script ML edge >= 0.10 | 708 | -2.9% | -0.94 | 1.000 |
| mlb | ml | back | follow script ML edge >= 0.05 | 4786 | -2.3% | -1.33 | 1.000 |
| mlb | ml | fade | fade script ML edge >= 0.05 | 4786 | -2.1% | -1.66 | 1.000 |
| mlb | tot | over | follow script total >= line+1.0 | 3158 | -2.9% | -1.77 | 1.000 |
| mlb | ml | fade | fade opp bullpen taxed (top10%) + own pen fresh | 1021 | -5.9% | -1.79 | 1.000 |
| mlb | ml | fade | fade team vs debut/new SP | 1556 | -5.4% | -1.93 | 1.000 |
| mlb | tot | over | fade script total <= line-1.5 | 91 | -19.4% | -1.97 |  |
| mlb | tot | under | fade script total >= line+1.5 | 1029 | -6.6% | -2.25 | 1.000 |
| mlb | tot | over | fade script total <= line-1.0 | 340 | -11.6% | -2.29 | 1.000 |
| mlb | tot | over | fade script total <= line-0.5 | 1312 | -8.5% | -3.27 | 1.000 |
| mlb | tot | under | fade script total >= line+0.5 | 7223 | -3.9% | -3.57 | 1.000 |
| mlb | ml | fade | fade script ML edge >= 0.15 | 77 | -31.2% | -3.60 |  |
| mlb | tot | under | fade script total >= line+1.0 | 3158 | -6.1% | -3.68 | 1.000 |
| mlb | tot | over | follow script total >= line+0.5 | 7223 | -5.1% | -4.66 | 1.000 |
| nba | ats | fade | fade team missing >=2 starters, opp <=0 | 854 | +3.3% | 1.01 | 1.000 |
| nba | ml | fade | fade b2b + |tz|>=2 (abs) | 3 | +65.0% | 0.56 |  |
| nba | ml | fade | fade rest disadvantage <=-2 + travel top20% | 132 | +5.1% | 0.50 |  |
| nba | ml | back | follow script ML edge >= 0.15 | 952 | +2.1% | 0.34 | 1.000 |
| nba | tot | over | fade script total <= line-6 | 431 | +1.5% | 0.33 | 1.000 |
| nba | tot | under | fade script total >= line+9 | 122 | +2.5% | 0.29 |  |
| nba | tot | under | follow script total <= line-9 | 101 | +2.2% | 0.23 |  |
| nba | tot | over | over: both teams b2b | 393 | +1.1% | 0.23 | 1.000 |
| nba | ats | back | follow script ATS edge >= 6 pts | 544 | +0.6% | 0.16 | 1.000 |
| nba | ml | back | back rest disadvantage <=-2 + travel top20% | 132 | +0.9% | 0.07 |  |
| nba | ats | back | back b2b + |tz|>=2 (abs) | 3 | -3.0% | -0.05 |  |
| nba | ats | fade | fade b2b + |tz|>=2 (abs) | 3 | -3.0% | -0.05 |  |
| nba | ml | fade | fade team missing >=2 starters, opp <=0 | 850 | -0.5% | -0.14 | 1.000 |
| nba | ml | back | follow script ML edge >= 0.05 | 4956 | -0.9% | -0.39 | 1.000 |
| nba | ml | back | follow script ML edge >= 0.10 | 2296 | -1.4% | -0.39 | 1.000 |
| nba | ml | fade | fade b2b + travel top20% | 221 | -2.7% | -0.41 | 1.000 |
| nba | ats | fade | fade rest disadvantage <=-2 + travel top20% | 133 | -3.8% | -0.46 |  |
| nba | ml | back | back b2b vs rested opp (rest>=2) | 2249 | -1.6% | -0.49 | 1.000 |
| nba | ats | fade | fade b2b + travel top20% | 224 | -3.6% | -0.57 | 1.000 |
| nba | ml | back | back b2b + |tz|>=2 (abs) | 3 | -36.4% | -0.57 |  |
| nba | ats | back | back rest disadvantage <=-2 + travel top20% | 133 | -5.2% | -0.63 |  |
| nba | ml | fade | fade team heavy missing minutes (top10%) | 1768 | -2.1% | -0.78 | 1.000 |
| nba | ml | back | back team heavy missing minutes (top10%) | 1768 | -2.7% | -0.81 | 1.000 |
| nba | ats | back | back b2b + travel top20% | 224 | -5.3% | -0.84 | 1.000 |
| nba | tot | over | fade script total <= line-3 | 1674 | -2.4% | -1.03 | 1.000 |
| nba | tot | under | fade script total >= line+6 | 485 | -4.5% | -1.04 | 1.000 |
| nba | tot | over | follow script total >= line+6 | 485 | -4.5% | -1.04 | 1.000 |
| nba | tot | over | fade script total <= line-9 | 101 | -11.1% | -1.17 |  |
| nba | tot | under | fade script total >= line+3 | 1786 | -2.7% | -1.22 | 1.000 |
| nba | tot | over | over: missing minutes sum top10% | 889 | -4.3% | -1.34 | 1.000 |
| nba | ats | fade | fade team heavy missing minutes (top10%) | 1777 | -3.0% | -1.34 | 1.000 |
| nba | tot | over | follow script total >= line+9 | 122 | -11.5% | -1.34 |  |
| nba | ml | back | back b2b + travel top20% | 221 | -12.2% | -1.35 | 1.000 |
| nba | ml | fade | fade script ML edge >= 0.15 | 952 | -3.1% | -1.39 | 1.000 |
| nba | tot | under | under: missing minutes sum top10% | 889 | -4.7% | -1.47 | 1.000 |
| nba | ats | back | follow script ATS edge >= 4 pts | 1645 | -3.5% | -1.52 | 1.000 |
| nba | ml | fade | fade b2b vs rested opp (rest>=2) | 2249 | -3.4% | -1.57 | 1.000 |
| nba | ml | back | back team missing >=2 starters, opp <=0 | 850 | -7.3% | -1.57 | 1.000 |
| nba | ats | back | back b2b vs rested opp (rest>=2) | 2266 | -3.8% | -1.89 | 1.000 |
| nba | ml | fade | fade script ML edge >= 0.10 | 2296 | -3.1% | -2.07 | 1.000 |
| nba | tot | under | under: both teams b2b | 393 | -10.1% | -2.11 | 1.000 |
| nba | tot | under | follow script total <= line-6 | 431 | -10.5% | -2.29 | 1.000 |
| nba | ats | fade | fade script ATS edge >= 4 pts | 1645 | -5.4% | -2.31 | 1.000 |
| nba | ats | fade | fade script ATS edge >= 6 pts | 544 | -9.5% | -2.36 | 1.000 |
| nba | ats | fade | fade b2b vs rested opp (rest>=2) | 2266 | -5.2% | -2.61 | 1.000 |
| nba | ats | back | back team heavy missing minutes (top10%) | 1777 | -5.9% | -2.63 | 1.000 |
| nba | tot | over | follow script total >= line+3 | 1786 | -6.3% | -2.79 | 1.000 |
| nba | tot | under | follow script total <= line-3 | 1674 | -6.6% | -2.85 | 1.000 |
| nba | ats | fade | fade script ATS edge >= 2 pts | 4387 | -4.3% | -3.00 | 1.000 |
| nba | ats | back | follow script ATS edge >= 2 pts | 4387 | -4.6% | -3.25 | 1.000 |
| nba | ml | fade | fade script ML edge >= 0.05 | 4956 | -3.9% | -3.39 | 1.000 |
| nba | ats | back | back team missing >=2 starters, opp <=0 | 854 | -12.2% | -3.77 | 1.000 |
| ncaaf | tot | under | follow script total <= line-9 | 131 | +11.5% | 1.40 | 1.000 |
| ncaaf | tot | under | follow script total <= line-6 | 416 | +2.8% | 0.61 | 1.000 |
| ncaaf | tot | under | fade script total >= line+9 | 117 | +2.0% | 0.23 | 1.000 |
| ncaaf | ml | back | back rest adv>=4 in conf game | 467 | +0.9% | 0.13 | 1.000 |
| ncaaf | ats | fade | fade script ATS edge >= 6 pts | 990 | -0.4% | -0.13 | 1.000 |
| ncaaf | tot | under | follow script total <= line-3 | 1162 | -0.4% | -0.14 | 1.000 |
| ncaaf | ats | back | back rest adv>=4 in conf game | 631 | -0.9% | -0.23 | 1.000 |
| ncaaf | ats | fade | fade script ATS edge >= 9 pts | 326 | -1.3% | -0.24 | 1.000 |
| ncaaf | ml | fade | fade script ML edge >= 0.10 | 959 | -1.1% | -0.51 | 1.000 |
| ncaaf | ml | fade | fade script ML edge >= 0.15 | 393 | -2.0% | -0.59 | 1.000 |
| ncaaf | tot | under | fade script total >= line+6 | 499 | -2.6% | -0.60 | 1.000 |
| ncaaf | tot | under | fade script total >= line+3 | 1490 | -1.5% | -0.61 | 1.000 |
| ncaaf | ats | fade | fade script ATS edge >= 3 pts | 2557 | -1.2% | -0.64 | 1.000 |
| ncaaf | ml | fade | fade rest adv>=4 in conf game | 467 | -6.6% | -1.16 | 1.000 |
| ncaaf | tot | over | follow script total >= line+9 | 117 | -11.0% | -1.25 | 1.000 |
| ncaaf | ats | back | follow script ATS edge >= 9 pts | 326 | -7.7% | -1.47 | 1.000 |
| ncaaf | tot | over | follow script total >= line+6 | 499 | -6.4% | -1.50 | 1.000 |
| ncaaf | ml | back | follow script ML edge >= 0.15 | 393 | -13.4% | -1.76 | 1.000 |
| ncaaf | ml | back | follow script ML edge >= 0.05 | 2192 | -7.2% | -2.01 | 1.000 |
| ncaaf | ml | fade | fade script ML edge >= 0.05 | 2192 | -3.3% | -2.05 | 1.000 |
| ncaaf | ats | fade | fade rest adv>=4 in conf game | 631 | -8.1% | -2.15 | 1.000 |
| ncaaf | ml | back | follow script ML edge >= 0.10 | 959 | -12.5% | -2.32 | 1.000 |
| ncaaf | tot | over | fade script total <= line-9 | 131 | -20.5% | -2.50 | 1.000 |
| ncaaf | tot | over | fade script total <= line-6 | 416 | -11.8% | -2.55 | 1.000 |
| ncaaf | ats | back | follow script ATS edge >= 6 pts | 990 | -8.5% | -2.83 | 1.000 |
| ncaaf | tot | over | follow script total >= line+3 | 1490 | -7.5% | -3.05 | 1.000 |
| ncaaf | tot | over | fade script total <= line-3 | 1162 | -8.6% | -3.09 | 1.000 |
| ncaaf | ats | back | follow script ATS edge >= 3 pts | 2557 | -7.7% | -4.12 | 1.000 |
| ncaam | tot | under | follow script total <= line-3 | 3618 | +1.2% | 0.77 | 1.000 |
| ncaam | tot | over | follow script total >= line+9 | 396 | +1.3% | 0.28 | 1.000 |
| ncaam | tot | over | follow script total >= line+6 | 1963 | +0.3% | 0.13 | 1.000 |
| ncaam | tot | under | follow script total <= line-6 | 801 | +0.4% | 0.12 | 1.000 |
| ncaam | ml | back | follow script ML edge >= 0.15 | 2017 | +0.4% | 0.10 | 1.000 |
| ncaam | tot | under | follow script total <= line-9 | 187 | +0.2% | 0.03 |  |
| ncaam | ats | back | follow script ATS edge >= 6 pts | 3166 | -0.3% | -0.20 | 1.000 |
| ncaam | tot | over | follow script total >= line+3 | 7804 | -1.0% | -0.93 | 1.000 |
| ncaam | ats | fade | fade rest adv>=2 in conf game | 3993 | -1.8% | -1.18 | 1.000 |
| ncaam | tot | over | fade script total <= line-9 | 187 | -9.0% | -1.31 |  |
| ncaam | ats | back | follow script ATS edge >= 4 pts | 7235 | -1.9% | -1.73 | 1.000 |
| ncaam | tot | under | fade script total >= line+9 | 396 | -10.2% | -2.16 | 1.000 |
| ncaam | ml | back | follow script ML edge >= 0.10 | 4587 | -5.5% | -2.58 | 1.000 |
| ncaam | tot | over | fade script total <= line-6 | 801 | -9.4% | -2.80 | 1.000 |
| ncaam | ml | fade | fade rest adv>=2 in conf game | 2619 | -7.7% | -3.07 | 1.000 |
| ncaam | ml | back | back rest adv>=2 in conf game | 2619 | -9.1% | -4.07 | 1.000 |
| ncaam | ats | back | follow script ATS edge >= 2 pts | 16458 | -3.2% | -4.30 | 1.000 |
| ncaam | tot | under | fade script total >= line+6 | 1963 | -9.2% | -4.33 | 1.000 |
| ncaam | ml | fade | fade script ML edge >= 0.15 | 2017 | -8.3% | -4.67 | 1.000 |
| ncaam | ats | back | back rest adv>=2 in conf game | 3993 | -7.2% | -4.79 | 1.000 |
| ncaam | ml | fade | fade script ML edge >= 0.10 | 4587 | -6.1% | -4.84 | 1.000 |
| ncaam | ats | fade | fade script ATS edge >= 6 pts | 3166 | -8.6% | -5.12 | 1.000 |
| ncaam | ml | fade | fade script ML edge >= 0.05 | 9926 | -5.3% | -5.56 | 1.000 |
| ncaam | ats | fade | fade script ATS edge >= 4 pts | 7235 | -7.0% | -6.29 | 1.000 |
| ncaam | tot | over | fade script total <= line-3 | 3618 | -10.2% | -6.47 | 1.000 |
| ncaam | ml | back | follow script ML edge >= 0.05 | 9926 | -9.6% | -7.00 | 1.000 |
| ncaam | tot | under | fade script total >= line+3 | 7804 | -8.0% | -7.44 | 1.000 |
| ncaam | ats | fade | fade script ATS edge >= 2 pts | 16458 | -5.7% | -7.80 | 1.000 |
| nfl | ml | fade | fade QB change, opp no change | 361 | +7.1% | 1.34 | 1.000 |
| nfl | ml | fade | fade team with QB change | 437 | +5.2% | 1.06 | 1.000 |
| nfl | ml | back | back QB much worse than base (qb_delta<=-0.08) | 541 | +4.0% | 0.82 | 1.000 |
| nfl | tot | under | under: any QB change | 400 | +3.8% | 0.80 | 1.000 |
| nfl | tot | under | fade script total >= line+6 | 39 | +2.8% | 0.18 |  |
| nfl | tot | under | follow script total <= line-6 | 24 | +3.4% | 0.17 |  |
| nfl | ml | fade | fade script ML edge >= 0.10 | 453 | -0.3% | -0.08 | 1.000 |
| nfl | tot | under | fade script total >= line+2 | 471 | -1.3% | -0.29 | 1.000 |
| nfl | tot | under | fade script total >= line+4 | 164 | -2.2% | -0.30 | 1.000 |
| nfl | ats | back | follow script ATS edge >= 6 pts | 85 | -3.4% | -0.33 | 1.000 |
| nfl | tot | over | fade script total <= line-4 | 136 | -3.1% | -0.38 | 1.000 |
| nfl | ats | back | back QB much worse than base (qb_delta<=-0.08) | 541 | -1.8% | -0.45 | 1.000 |
| nfl | tot | over | fade script total <= line-2 | 427 | -2.5% | -0.54 | 1.000 |
| nfl | ml | back | follow script ML edge >= 0.15 | 163 | -6.3% | -0.54 | 1.000 |
| nfl | ats | fade | fade script ATS edge >= 6 pts | 85 | -5.7% | -0.54 | 1.000 |
| nfl | ats | fade | fade QB change, opp no change | 362 | -2.9% | -0.58 | 1.000 |
| nfl | ats | fade | fade script ATS edge >= 4 pts | 292 | -3.5% | -0.63 | 1.000 |
| nfl | tot | over | fade script total <= line-6 | 24 | -12.5% | -0.63 |  |
| nfl | ats | fade | fade team with QB change | 438 | -3.1% | -0.69 | 1.000 |
| nfl | tot | under | follow script total <= line-4 | 136 | -5.9% | -0.72 | 1.000 |
| nfl | ml | fade | fade script ML edge >= 0.15 | 163 | -3.9% | -0.72 | 1.000 |
| nfl | ml | fade | fade script ML edge >= 0.05 | 1043 | -1.8% | -0.75 | 1.000 |
| nfl | tot | over | follow script total >= line+6 | 39 | -11.9% | -0.77 |  |
| nfl | ats | back | follow script ATS edge >= 2 pts | 875 | -2.8% | -0.88 | 1.000 |
| nfl | tot | over | follow script total >= line+4 | 164 | -6.9% | -0.92 | 1.000 |
| nfl | ats | back | follow script ATS edge >= 4 pts | 292 | -5.4% | -0.98 | 1.000 |
| nfl | ats | back | back QB change, opp no change | 362 | -6.1% | -1.21 | 1.000 |
| nfl | ats | back | back team with QB change | 438 | -5.7% | -1.27 | 1.000 |
| nfl | tot | under | follow script total <= line-2 | 427 | -6.5% | -1.42 | 1.000 |
| nfl | ml | back | follow script ML edge >= 0.10 | 453 | -10.5% | -1.52 | 1.000 |
| nfl | ats | fade | fade QB much worse than base (qb_delta<=-0.08) | 541 | -7.1% | -1.75 | 1.000 |
| nfl | tot | over | follow script total >= line+2 | 471 | -7.8% | -1.77 | 1.000 |
| nfl | ml | back | back team with QB change | 437 | -11.0% | -1.78 | 1.000 |
| nfl | ml | back | back QB change, opp no change | 361 | -12.5% | -1.79 | 1.000 |
| nfl | ats | fade | fade script ATS edge >= 2 pts | 875 | -6.1% | -1.91 | 1.000 |
| nfl | ml | fade | fade QB much worse than base (qb_delta<=-0.08) | 541 | -9.8% | -1.92 | 1.000 |
| nfl | ml | back | follow script ML edge >= 0.05 | 1043 | -8.3% | -1.93 | 1.000 |
| nfl | tot | over | over: any QB change | 400 | -12.9% | -2.71 | 1.000 |
| nhl | ml | back | follow script ML edge >= 0.05 | 2331 | +4.4% | 1.88 | 1.000 |
| nhl | tot | over | follow script total >= line+0.8 | 217 | +10.8% | 1.75 | 1.000 |
| nhl | ml | back | follow script ML edge >= 0.10 | 297 | +10.4% | 1.54 | 1.000 |
| nhl | tot | over | follow script total >= line+0.5 | 1212 | +3.5% | 1.34 | 1.000 |
| nhl | ml | fade | fade script ML edge >= 0.15 | 33 | +16.4% | 1.14 |  |
| nhl | tot | over | follow script total >= line+0.3 | 2932 | +1.7% | 0.99 | 1.000 |
| nhl | ml | fade | fade backup G on b2b | 1296 | +1.8% | 0.66 | 1.000 |
| nhl | ml | fade | fade team on b2b vs rested | 1846 | +1.2% | 0.59 | 1.000 |
| nhl | tot | under | follow script total <= line-0.5 | 147 | +3.4% | 0.44 |  |
| nhl | tot | under | follow script total <= line-0.3 | 614 | +1.0% | 0.26 | 1.000 |
| nhl | tot | under | follow script total <= line-0.8 | 7 | +9.1% | 0.24 |  |
| nhl | tot | over | fade script total <= line-0.8 | 7 | -18.2% | -0.47 |  |
| nhl | tot | under | under: both backup G | 596 | -2.1% | -0.54 | 1.000 |
| nhl | ml | back | follow script ML edge >= 0.15 | 33 | -14.9% | -0.63 |  |
| nhl | ml | back | back team vs backup G (own starter) | 3172 | -1.8% | -1.05 | 1.000 |
| nhl | ml | fade | fade team starting backup G | 4364 | -2.3% | -1.54 | 1.000 |
| nhl | tot | over | fade script total <= line-0.5 | 147 | -12.2% | -1.57 |  |
| nhl | tot | over | over: both backup G | 596 | -6.5% | -1.72 | 1.000 |
| nhl | ml | fade | fade team vs backup G (own starter) | 3172 | -4.8% | -2.52 | 1.000 |
| nhl | tot | over | fade script total <= line-0.3 | 614 | -9.6% | -2.56 | 1.000 |
| nhl | ml | fade | fade script ML edge >= 0.10 | 297 | -13.5% | -2.60 | 1.000 |
| nhl | ml | back | back team starting backup G | 4364 | -4.4% | -2.78 | 1.000 |
| nhl | tot | under | under: any backup G | 3768 | -4.3% | -2.82 | 1.000 |
| nhl | tot | over | over: any backup G | 3768 | -4.3% | -2.86 | 1.000 |
| nhl | tot | under | fade script total >= line+0.8 | 217 | -19.1% | -3.11 | 1.000 |
| nhl | ml | back | back backup G on b2b | 1296 | -9.5% | -3.18 | 1.000 |
| nhl | ml | back | back team on b2b vs rested | 1846 | -9.1% | -3.53 | 1.000 |
| nhl | ml | fade | fade script ML edge >= 0.05 | 2331 | -7.8% | -4.03 | 1.000 |
| nhl | tot | under | fade script total >= line+0.5 | 1212 | -11.9% | -4.53 | 1.000 |
| nhl | tot | under | fade script total >= line+0.3 | 2932 | -10.1% | -5.96 | 1.000 |

## Family B (broad) — top 15 by discovery z

| sport | market | bet | rule | disc n | disc ROI | disc z | BH q |
|---|---|---|---|---|---|---|---|
| nhl | ml | back | script>mkt top10% & model>mkt top10% | 539 | +11.7% | 2.62 | 1.000 |
| nhl | tot | over | script total>line top20% & line bot10% | 402 | +10.4% | 2.55 | 1.000 |
| nfl | ml | back | script<mkt bot20% & model>mkt top10% | 109 | +8.8% | 2.48 | 1.000 |
| nfl | tot | under | wind>=12 | 304 | +13.1% | 2.43 | 1.000 |
| nhl | ml | back | model>mkt top10% & o b2b | 249 | +11.6% | 2.26 | 1.000 |
| nhl | tot | over | line bot10% | 1615 | +4.5% | 2.21 | 1.000 |
| nhl | ml | back | dog & model>mkt top10% | 431 | +12.1% | 2.19 | 1.000 |
| nba | ats | back | script<mkt bot20% & playoff | 343 | +11.1% | 2.19 | 1.000 |
| nba | ats | back | fav & playoff | 589 | +8.3% | 2.14 | 1.000 |
| ncaaf | tot | under | line top10% & plays top10% | 168 | +15.4% | 2.13 | 1.000 |
| nhl | ml | back | bigdog & script>mkt top20% | 319 | +19.2% | 2.13 | 1.000 |
| nhl | ml | back | script>mkt top20% & model>mkt top10% | 808 | +7.4% | 2.12 | 1.000 |
| ncaaf | tot | under | line top10% | 555 | +8.4% | 2.10 | 1.000 |
| nhl | ml | back | away & script>mkt top10% | 1025 | +7.8% | 2.07 | 1.000 |
| ncaaf | tot | under | script total<line bot10% & line top10% | 214 | +13.3% | 2.07 | 1.000 |

## Multiple-testing sanity check

Share of eligible tests with one-sided discovery p below each cut (pure noise would give ~ the cut itself; side rules in family B are correlated, so this is only indicative):

* A_targeted: n=170; p<0.05: 1.2%; p<0.01: 0.0%; p<0.001: 0.00%; min p = 3.02e-02
* B_broad: n=2426; p<0.05: 1.8%; p<0.01: 0.2%; p<0.001: 0.00%; min p = 4.41e-03

## Post-hoc look (disclosed; lockbox burned for this rule)

MLB 'follow script ML edge >= 0.15' shows +61.5% on 77 discovery bets (z 3.25) but is below the pre-set MLB minimum n (300), so it never entered screening. After the run I inspected it on all seasons (2016-25: 136 bets, avg price +275, max +1500, market p as low as 0.06 where the script says ~0.5): it is driven by stale/broken MLB quotes (the known integrity problem), not a situation. Because I looked at its 2023+ seasons, its lockbox is burned and it is NOT a candidate.

## Mechanism notes

No rule earned a mechanism claim. The likely reason no situational edge survives: every feature tested here is public before the close (rest/b2b/travel from the schedule, confirmed goalies/SPs/QBs, injury-driven missing minutes, weather), and the closing line already prices it — the script reports show the learned market weight sits at 0.9-1.0 and corr(script - close, actual - close) is ~0 (NBA 0.010 margin / -0.019 total). Script-vs-close disagreement therefore measures script error, not market error, in every sport (follow AND fade both lose roughly the vig). Only 1.2-1.8% of eligible tests reached one-sided p<0.05 in discovery (noise alone would give ~5%; the vig pushes the whole distribution below zero), so the search found less signal than chance, not more. The best NFL total rule (wind>=12 unders) is the existing 'Wind Wall Unders' key re-discovered, not a new edge.

## Live use

`python -m engine.edge_hunt.situational --scan [--sport nba]` pulls today's slate (`nightly.live.today_slate`, ESPN + odds board), rebuilds the schedule features (rest, b2b) from the game history, de-vigs the ML, and checks the TRUE rules in `results/edge_hunt/situational_rules.json` (discovery thresholds from `situational_thresholds.json`). Atoms that need a script-engine column (script edge, goalie/SP effect, missing minutes, travel) are filled from today's row of `results/scripts/<sport>_games.parquet` when the nightly script run has already appended it; otherwise the game is reported as `needs_script_refresh` instead of being silently skipped.
