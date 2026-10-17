# Edge hunt: clv_openers — predict the open→close move, bet the OPENER

Code: `engine/edge_hunt/clv_openers.py` (`python -m engine.edge_hunt.clv_openers`; live: `scan(sport)`).
Every test is in `results/edge_hunt/clv_openers_candidates.csv` (395 rows: 280 discovery + 30 lockbox in the
pre-registered pass, 75 + 10 in the post-hoc pass). Final rules are in `clv_openers_finals.csv`, and the
correlation diagnostics are in `clv_openers_diag.csv`.

## Verdict: no TRUE edge

None of the 30 pre-registered lockbox candidates passes. Bonferroni α = 0.05/30 = 0.0017, and the smallest
BH q is 0.05. That one is NHL ML raw (q = 0.051, p = 0.0017). It is FALSE because of leakage, explained
below. The models do predict line movement, and they do capture closing-line value at the opener. That CLV
does not turn into significant ROI after the vig, and in several cases the juice had to be assumed.

## Data: which openers exist (all sports checked for `*_open`)

| sport | market | opener rows | opening juice | split used |
|---|---|---|---|---|
| NCAAF | spread, total | 2016-19, 2021-25 (CFBD consensus; none in 2020) | **not in data → -110 assumed** (CFBD opening juice exists only 2016-19, per book) | ≤2022 / 2023-25 |
| NCAAF | ML | 2016-19 only | – | **not testable** (no lockbox openers) |
| NBA | spread, total | 2016-22 (SBR, no juice → -110 assumed) + 2025-26 (ESPN PickCenter DraftKings, **real** open/close juice) | lockbox real | ≤2022 / 2026 |
| NBA | ML | 2025-26 only (DK real) | real | 60/40 dates |
| NCAAM | spread, total, ML | 2025-26 only (DK real) | real | 60/40 dates |
| MLB | ML | 2022-25 (SBR DraftKings open/close, real) | real | 2022 / 2023-25 |
| NHL | ML | 2015-16 … 2021-22 (SBRO, real) | real | **60/40 dates**: no 2023+ openers (2024+ "lines" are noon snapshots) |
| NFL, MLB/NHL totals, NHL/MLB run/puck lines | – | no openers | – | not testable |

Rows with |close−open| > 10 pts (spread) or > 15 (total) were dropped as bad records; CFBD has 299-point typos.

## Method

* **Target:** close − open. That is the home spread, the total, or the de-vigged home ML probability.
* **"model" family:** walk-forward OLS, refit at each month start on all earlier months, on
  (script − opener), (core − opener) and the opener level. The feature set and threshold T on |predicted move|
  were chosen on discovery ROI with n ≥ min(150, max(60, 5% of discovery)).
* **"raw" family:** bet when |script − opener| ≥ T. No regression.
* **"move" family (public rule):** the line moved ≥ k from the opener toward X. FOLLOW or FADE it at the
  **closing** price. Live, that means the current FanDuel/DK line vs the ESPN opener.
  True reverse-line-movement needs bet-% data, which we do not have.
* "model" and "raw" bets are settled at the **opening** line and price against the final score. "move" bets
  are settled at the closing line and price. CLV is the line move in the bet's favour, in points or in
  de-vigged probability.

## Can the line move be predicted? Yes (corr of the signal with close−open)

| sport   | market   | part   |    n |   move_sd |   corr_raw_script |   corr_raw_core |   corr_script |   corr_core |   corr_script+core+level |
|:--------|:---------|:-------|-----:|----------:|------------------:|----------------:|--------------:|------------:|-------------------------:|
| ncaaf   | spread   | disc   | 4679 |     2.154 |             0.195 |           0.012 |         0.194 |      -0.003 |                    0.271 |
| ncaaf   | spread   | lock   | 2617 |     2.066 |             0.17  |           0.008 |         0.166 |      -0.051 |                    0.209 |
| ncaaf   | total    | disc   | 4798 |     2.48  |             0.115 |           0.071 |         0.106 |       0.051 |                    0.111 |
| ncaaf   | total    | lock   | 2645 |     2.095 |             0.195 |           0.18  |         0.182 |       0.166 |                    0.181 |
| nba     | spread   | disc   | 8716 |     1.598 |             0.151 |           0.068 |         0.143 |       0.052 |                    0.196 |
| nba     | spread   | lock   | 1065 |     1.976 |             0.068 |          -0.003 |         0.068 |      -0.004 |                    0.136 |
| nba     | total    | disc   | 8707 |     2.824 |             0.259 |           0.226 |         0.256 |       0.219 |                    0.256 |
| nba     | total    | lock   | 1068 |     3.294 |             0.223 |           0.158 |         0.222 |       0.158 |                    0.228 |
| nba     | ml       | disc   |  768 |     0.047 |             0.052 |          -0.003 |        -0.023 |      -0.08  |                    0.058 |
| nba     | ml       | lock   |  291 |     0.049 |             0.072 |           0.034 |         0.06  |      -0.046 |                    0.114 |
| ncaam   | spread   | disc   | 3144 |     1.603 |             0.168 |           0.181 |         0.209 |       0.241 |                    0.182 |
| ncaam   | spread   | lock   | 1603 |     1.43  |             0.069 |           0.136 |         0.068 |       0.134 |                    0.125 |
| ncaam   | total    | disc   | 3184 |     2.705 |             0.211 |           0.158 |         0.309 |       0.239 |                    0.329 |
| ncaam   | total    | lock   | 1605 |     2.514 |             0.412 |           0.263 |         0.411 |       0.251 |                    0.433 |
| ncaam   | ml       | disc   | 3046 |     0.037 |             0.113 |           0.118 |         0.123 |       0.137 |                    0.218 |
| ncaam   | ml       | lock   | 1600 |     0.034 |             0.017 |           0.113 |         0.014 |       0.11  |                    0.153 |
| mlb     | ml       | disc   | 2421 |     0.037 |             0.379 |           0.19  |         0.368 |       0.174 |                    0.37  |
| mlb     | ml       | lock   | 6720 |     0.037 |             0.309 |           0.199 |         0.309 |       0.197 |                    0.309 |
| nhl     | ml       | disc   | 5052 |     0.028 |             0.304 |           0.067 |         0.287 |       0.058 |                    0.316 |
| nhl     | ml       | lock   | 3159 |     0.029 |             0.354 |          -0.018 |         0.354 |      -0.032 |                    0.373 |

The script anticipates the move everywhere: corr 0.15-0.43, and in MLB/NHL ML 0.30-0.38. The core-only
(team-rating) signal is weaker in NBA and NHL, and in NHL it is about zero. That gap is the leakage
described below.

## Verdicts (one frozen rule per sport × market × family; lockbox evaluated once)

| pass | candidate | lockbox price | discovery n / ROI / CLV | **lockbox n / ROI / z / p** | units / maxDD | lockbox CLV (% beat close) | seasons(months) + | BH q | verdict | why |
|---|---|---|---|---|---|---|---|---|---|---|
| main | ncaaf spread model [script+core+level ≥0.5] @open | assumed -110 | 1912 / +0.6% / +0.687 | 908 / +1.0% / +0.32 / 0.375 | +9.1u / 18.3 | +0.714 pts (55%) | 4/9 | 0.72 | **FALSE** | juice assumed -110 (not in data); not significant; unstable (<60% seasons/months positive) |
| main | ncaaf spread raw [- ≥2] @open | assumed -110 | 2871 / -3.8% / +0.495 | 1633 / +0.1% / +0.06 / 0.476 | +2.3u / 32.5 | +0.395 pts (50%) | 1/9 | 0.72 | **FALSE** | juice assumed -110 (not in data); not significant; unstable (<60% seasons/months positive) |
| main | ncaaf spread move [- fade ≥2] @close | assumed -110 | 1636 / -3.5% / -3.188 | 779 / -5.3% / -1.57 / 0.942 | -41.5u / 53.1 | -3.228 pts (0%) | 2/9 | 0.96 | **FALSE** | juice assumed -110 (not in data); not significant; unstable (<60% seasons/months positive) |
| main | ncaaf total model [script+core+level ≥0.5] @open | assumed -110 | 649 / +9.9% / +0.549 | 520 / +0.3% / +0.06 / 0.475 | +1.4u / 24.5 | +1.028 pts (66%) | 7/9 | 0.72 | **FALSE** | juice assumed -110 (not in data); not significant |
| main | ncaaf total raw [- ≥8] @open | assumed -110 | 248 / +14.0% / +0.901 | 88 / +0.9% / +0.09 / 0.464 | +0.8u / 8.5 | +1.088 pts (62%) | 6/9 | 0.72 | **FALSE** | juice assumed -110 (not in data); not significant |
| main | ncaaf total move [- follow ≥3] @close | assumed -110 | 1185 / +2.1% / +4.172 | 460 / -4.2% / -0.96 / 0.832 | -19.5u / 28.2 | +4.035 pts (100%) | 4/9 | 0.96 | **FALSE** | juice assumed -110 (not in data); not significant; unstable (<60% seasons/months positive) |
| main | nba spread model [script ≥0.5] @open | real | 267 / +1.3% / +0.725 | 64 / +7.3% / +0.61 / 0.271 | +4.6u / 5.2 | -0.125 pts (36%) | 5/8 | 0.72 | **FALSE** | LEAK: NBA script uses actual missing minutes (post-opener lineup info); not significant |
| main | nba spread raw [- ≥6] @open | real | 416 / -0.6% / +0.633 | 64 / +10.1% / +0.85 / 0.198 | +6.4u / 5.7 | -0.250 pts (34%) | 5/8 | 0.72 | **FALSE** | LEAK: NBA script uses actual missing minutes (post-opener lineup info); not significant |
| main | nba spread move [- follow ≥3] @close | real | 675 / -0.4% / +3.921 | 182 / +7.7% / +1.09 / 0.138 | +14.1u / 10.9 | +3.868 pts (100%) | 5/8 | 0.72 | **FALSE** | not significant |
| main | nba total model [core ≥1.5] @open | real | 217 / +3.9% / +2.085 | 82 / +6.9% / +0.66 / 0.255 | +5.7u / 9.6 | +1.073 pts (60%) | 6/8 | 0.72 | **FALSE** | not significant |
| main | nba total raw [- ≥8] @open | real | 220 / +4.2% / +2.016 | 40 / +9.6% / +0.63 / 0.263 | +3.8u / 4.4 | +1.775 pts (60%) | 7/8 | 0.72 | **FALSE** | LEAK: NBA script uses actual missing minutes (post-opener lineup info); not significant |
| main | nba total move [- follow ≥2] @close | real | 4499 / -0.6% / +3.492 | 684 / -4.1% / -1.12 / 0.868 | -27.9u / 49.9 | +3.671 pts (100%) | 4/8 | 0.96 | **FALSE** | not significant; unstable (<60% seasons/months positive) |
| main | nba ml model [script+core+level ≥0.005] @open | real | 162 / +4.2% / +0.001 | 111 / +2.5% / +0.28 / 0.388 | +2.8u / 9.4 | +0.010 prob (58%) | 4/5 | 0.72 | **FALSE** | LEAK: NBA script uses actual missing minutes (post-opener lineup info); not significant; single-season 60/40 split |
| main | nba ml raw [- ≥0.1] @open | real | 149 / +14.2% / +0.002 | 87 / -7.3% / -0.49 / 0.688 | -6.3u / 24.9 | +0.000 prob (39%) | 6/8 | 0.94 | **FALSE** | LEAK: NBA script uses actual missing minutes (post-opener lineup info); not significant; single-season 60/40 split |
| main | nba ml move [- follow ≥0.08] @close | real | 65 / +14.4% / +0.114 | 31 / +4.4% / +0.23 / 0.409 | +1.4u / 4.9 | +0.115 prob (100%) | 4/6 | 0.72 | **FALSE** | not significant; single-season 60/40 split |
| main | ncaam spread model [script ≥0.25] @open | real | 244 / -6.6% / +0.553 | 250 / +0.3% / +0.06 / 0.477 | +0.9u / 15.9 | +0.212 pts (38%) | 1/4 | 0.72 | **FALSE** | not significant; unstable (<60% seasons/months positive); single-season 60/40 split |
| main | ncaam spread raw [- ≥2] @open | real | 1802 / +0.0% / +0.331 | 604 / -1.2% / -0.30 / 0.617 | -7.0u / 27.5 | +0.116 pts (34%) | 3/6 | 0.88 | **FALSE** | not significant; unstable (<60% seasons/months positive); single-season 60/40 split |
| main | ncaam spread move [- fade ≥3] @close | real | 345 / -1.8% / -3.539 | 142 / +3.0% / +0.38 / 0.353 | +4.2u / 8.6 | -3.423 pts (0%) | 4/6 | 0.72 | **FALSE** | not significant; single-season 60/40 split |
| main | ncaam total model [script ≥0.5] @open | real | 238 / +10.3% / +1.660 | 402 / +7.2% / +1.52 / 0.064 | +28.8u / 15.0 | +1.550 pts (64%) | 3/4 | 0.64 | **FALSE** | not significant; single-season 60/40 split |
| main | ncaam total raw [- ≥6] @open | real | 297 / +9.8% / +1.145 | 43 / +23.5% / +1.68 / 0.046 | +10.1u / 3.1 | +3.163 pts (81%) | 5/5 | 0.64 | **FALSE** | not significant; single-season 60/40 split |
| main | ncaam total move [- follow ≥4] @close | real | 591 / +2.7% / +4.966 | 243 / -10.0% / -1.62 / 0.947 | -24.2u / 36.7 | +4.877 pts (100%) | 3/6 | 0.96 | **FALSE** | not significant; unstable (<60% seasons/months positive); single-season 60/40 split |
| main | ncaam ml model [script+core+level ≥0.005] @open | real | 753 / -2.2% / +0.010 | 807 / -2.6% / -0.97 / 0.833 | -21.2u / 42.2 | +0.007 prob (53%) | 1/4 | 0.96 | **FALSE** | not significant; unstable (<60% seasons/months positive); single-season 60/40 split |
| main | ncaam ml raw [- ≥0.08] @open | real | 976 / -6.8% / +0.007 | 331 / +1.9% / +0.24 / 0.407 | +6.2u / 20.8 | +0.001 prob (47%) | 2/6 | 0.72 | **FALSE** | not significant; unstable (<60% seasons/months positive); single-season 60/40 split |
| main | ncaam ml move [- follow ≥0.02] @close | real | 1555 / -2.9% / +0.045 | 836 / -5.8% / -1.70 / 0.956 | -48.6u / 88.1 | +0.042 prob (100%) | 2/6 | 0.96 | **FALSE** | not significant; unstable (<60% seasons/months positive); single-season 60/40 split |
| main | mlb ml model [core ≥0.015] @open | real | 275 / +3.4% / +0.016 | 719 / -5.4% / -1.43 / 0.923 | -39.1u / 53.3 | +0.019 prob (70%) | 1/4 | 0.96 | **FALSE** | not significant; unstable (<60% seasons/months positive) |
| main | mlb ml raw [- ≥0.08] @open | real | 221 / -1.9% / +0.031 | 477 / +4.5% / +0.84 / 0.200 | +21.3u / 22.1 | +0.025 prob (74%) | 2/4 | 0.72 | **FALSE** | not significant; unstable (<60% seasons/months positive) |
| main | mlb ml move [- fade ≥0.05] @close | real | 395 / +6.4% / -0.072 | 949 / +2.6% / +0.70 / 0.242 | +25.0u / 24.5 | -0.075 prob (0%) | 3/4 | 0.72 | **FALSE** | not significant |
| main | nhl ml model [script ≥0.02] @open | real | 160 / +8.1% / +0.022 | 113 / +12.0% / +1.21 / 0.113 | +13.6u / 9.8 | +0.034 prob (89%) | 4/7 | 0.72 | **FALSE** | LEAK: NHL script uses the confirmed starting goalie (post-opener info); not significant; unstable (<60% seasons/months positive); 60/40 date split (no 2023+ openers) |
| main | nhl ml raw [- ≥0.06] @open | real | 817 / -0.0% / +0.016 | 670 / +12.1% / +2.93 / 0.002 | +81.2u / 27.5 | +0.017 prob (74%) | 5/7 | 0.05 | **FALSE** | LEAK: NHL script uses the confirmed starting goalie (post-opener info); 60/40 date split (no 2023+ openers) |
| main | nhl ml move [- follow ≥0.05] @close | real | 339 / -0.4% / +0.067 | 242 / -4.3% / -0.79 / 0.786 | -10.4u / 16.1 | +0.068 prob (100%) | 3/7 | 0.96 | **FALSE** | not significant; unstable (<60% seasons/months positive); 60/40 date split (no 2023+ openers) |
| POSTHOC-clean | nba spread model [core+level ≥0.25] @open | real | 818 / -8.6% / +0.234 | 127 / +0.8% / +0.09 / 0.465 | +1.0u / 9.2 | -0.102 pts (35%) | 3/8 | 0.83 | **FALSE** | post-hoc (lockbox burned); not significant; unstable (<60% seasons/months positive) |
| POSTHOC-clean | nba spread raw [core ≥6] @open | real | 429 / -1.8% / +0.449 | 63 / +15.0% / +1.27 / 0.102 | +9.5u / 4.3 | -0.683 pts (21%) | 4/8 | 0.83 | **FALSE** | post-hoc (lockbox burned); not significant; unstable (<60% seasons/months positive) |
| POSTHOC-clean | nba total model [core+level ≥1.5] @open | real | 264 / +4.2% / +1.722 | 106 / +2.5% / +0.27 / 0.393 | +2.7u / 10.7 | +0.877 pts (58%) | 7/8 | 0.83 | **FALSE** | post-hoc (lockbox burned); not significant |
| POSTHOC-clean | nba total raw [core ≥8] @open | real | 301 / +4.7% / +2.055 | 100 / +2.9% / +0.30 / 0.382 | +2.9u / 11.0 | +1.130 pts (60%) | 6/8 | 0.83 | **FALSE** | post-hoc (lockbox burned); not significant |
| POSTHOC-clean | nba ml model [core+level ≥0.005] @open | real | 93 / +11.7% / +0.003 | 22 / +0.1% / +0.01 / 0.498 | +0.0u / 2.0 | -0.000 prob (41%) | 3/4 | 0.83 | **FALSE** | post-hoc (lockbox burned); not significant; single-season 60/40 split |
| POSTHOC-clean | nba ml raw [core ≥0.1] @open | real | 151 / +22.7% / +0.001 | 101 / -22.3% / -1.56 / 0.941 | -22.5u / 39.9 | -0.002 prob (34%) | 6/8 | 1.00 | **FALSE** | post-hoc (lockbox burned); not significant; single-season 60/40 split |
| POSTHOC-clean | nhl ml model [core+level ≥0.01] @open | real | 253 / -3.2% / +0.004 | 0 / – / – / – | – / – | – | 2/3 | 1.00 | **FALSE** | post-hoc (lockbox burned); not significant; 60/40 date split (no 2023+ openers) |
| POSTHOC-clean | nhl ml raw [core ≥0.1] @open | real | 181 / +8.6% / +0.009 | 314 / -10.5% / -1.32 / 0.907 | -33.0u / 49.7 | +0.003 prob (52%) | 4/7 | 1.00 | **FALSE** | post-hoc (lockbox burned); not significant; unstable (<60% seasons/months positive); 60/40 date split (no 2023+ openers) |
| POSTHOC-clean | mlb ml model [core ≥0.015] @open | real | 275 / +3.4% / +0.016 | 719 / -5.4% / -1.43 / 0.923 | -39.1u / 53.3 | +0.019 prob (70%) | 1/4 | 1.00 | **FALSE** | post-hoc (lockbox burned); not significant; unstable (<60% seasons/months positive) |
| POSTHOC-clean | mlb ml raw [core ≥0.06] @open | real | 779 / -0.9% / +0.006 | 2445 / +1.0% / +0.47 / 0.320 | +25.2u / 37.2 | +0.007 prob (54%) | 2/4 | 0.83 | **FALSE** | post-hoc (lockbox burned); not significant; unstable (<60% seasons/months positive) |

The seasons column counts seasons for multi-season samples and months for single-season ones, over the
full sample. The POSTHOC-clean rows were added after the lockbox had been seen, so their lockbox is burned
and they are diagnostics only. Superseded run: the first execution allowed only the "script" feature set
for MLB/NHL ML. Its frozen rules gave a lockbox of MLB ML model n=3206, +1.5%, z 0.81 and NHL ML model
n=113, +12.0%, z 1.21. Neither was significant; both are logged here for completeness.

## Key findings

1. **The CLV is real, but it is small next to the vig.** Pre-registered opener bets beat the close on
   average in every spread and total cell:
   * NCAAF spread model: +0.71 pts CLV, n=908.
   * NCAAF total model: +1.03 pts, 66% beat the close.
   * NBA total model: +1.07 pts.
   * NCAAM total model: +1.55 pts, 64% beat the close.
   * MLB ML: about +1.5 to 2.5 pp of probability.

   Lockbox ROI is mostly positive (NCAAF spread +1.0%, NCAAM total +7.2% with p=0.064, NBA total +6.9% on
   n=82), but none comes close to the multiple-testing bar. Many cells also fail stability, or use assumed
   -110 opening juice. Real openers usually carry higher juice and lower limits.
2. **NHL ML raw at the opener (lockbox +12.1%, n=670, z 2.93) is leakage, not an edge.** The NHL script
   includes the **confirmed starting goalie**, which is not known when the opener is posted. With the
   goalie-free core rating the same rule loses: −5.8% on the lockbox, with corr(core−open, move) ≈ 0. The
   profit is concentrated where the *opponent started a backup* (+25% ROI, n=199 vs +6.6% otherwise). In
   other words, the market moves on goalie news. That would be a news-speed strategy (bet the instant a
   backup is confirmed, before the book moves), and open/close data cannot validate it. Its discovery ROI
   was −0.0%, and the split is a 60/40 fallback. **FALSE.**
3. NBA "script" likewise uses actual missing minutes, i.e. who played. Its spread/total opener edge is partly
   post-opener lineup info. The clean core-only rules gave lockbox +0.8% / +2.5% / +2.9% (not significant).
4. **Public line-move rules lose.** Following a ≥2-3 point move at the close loses 4-10% in the lockbox
   (NBA/NCAAF/NCAAM totals). Fading moves is about breakeven or negative. The close has already absorbed
   the move, and these rules carry no information beyond it.
5. **Thin data is the binding constraint.** Real opening juice exists only for NBA/NCAAM 2025-26 (DK via
   ESPN) and MLB 2022-25 / NHL ≤2022 moneylines. NCAAF/NBA 2023-25 spread openers are missing or have no
   juice.

## Mechanism (why it could exist, and why it isn't proven)

Openers are set by a few books on thin information, at low limits. Sharp money and news (injuries, goalies,
lineups) then move the line, and a good projection front-runs part of that. The other side is the
opener-setting book, which accepts the risk at low limits *because* it uses early sharp action for price
discovery. That limit cap, plus the vig, is why measurable CLV (0.5-1.5 pts) does not show up as
significant ROI here.

## Live scan

`scan(sport, projections=None)` does three things:
* pulls the ESPN scoreboard DraftKings **opener** (line + juice) for pre-game events;
* pulls the current FanDuel board (`nightly.providers.fanduel_board`) and falls back to DK current;
* fires the frozen rules: "move" rules from current vs opener; "raw" opener rules when you pass script
  projections (`home, away, s_margin, s_total, p_script`).

Each row carries the rule's lockbox verdict. All rules are FALSE, so `scan(..., only_true=True)` returns
nothing. Tested live 2026-10-17: NCAAF 8 signals (e.g. Nebraska@Oregon opened −21.5, now −6.5), NBA 3,
NHL 0.

## Follow-ups worth doing (not edges yet)

* Log ESPN openers + FanDuel snapshots nightly with timestamps. That builds a real-juice opener lockbox for
  NCAAF/NBA spreads and totals (2026+), where the CLV signal is strongest.
* NHL backup-goalie news-speed test: needs timestamped quotes around goalie confirmations.
