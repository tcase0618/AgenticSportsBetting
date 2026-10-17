# Edge hunt: props_market — calibrated prop probabilities vs REAL prop prices

Protocol: engine/edge_hunt/PROTOCOL.md. Code: engine/edge_hunt/props_market.py. Every knob tried: props_market_candidates.csv (n_tests = 546). Discovery = first 60% of dates (UFC <= 2022); lockbox = last 40% (UFC 2023+), evaluated ONCE for the single discovery-chosen config per family (max discovery z, n >= nmin). Significance: one-sided p < 0.05 / 12 (families with lockbox bets) or BH q < 0.10, using the LARGER of the naive and the cluster-robust (by game / date) p; TRUE also needs disc ROI > 0, lockbox n >= 100 and >= 60% positive blocks (months NBA, weeks NFL/NHL/MLB, years UFC) over the full sample.

## Verdicts

| test | candidate (discovery-chosen config) | disc n / ROI | lockbox n / ROI / z / p | clustered z / p | units / maxDD | blocks + | BH q | verdict |
|---|---|---|---|---|---|---|---|---|
| T1 | NFL alt+main ladder vs model: `side=both ev>=0.05 odds<=+150` | 585 / +0.052 | 532 / +0.044 / +1.03 / 0.153 | +1.03 / 0.152 (game_id) | +23.4 / 24.9 | 4/6 | 0.217 | **FALSE** |
| T1 | NFL alt-ladder only vs model: `side=both ev>=0.08 odds<=+150` | 208 / +0.080 | 350 / +0.108 / +2.00 / 0.023 | +1.80 / 0.036 (game_id) | +38.0 / 15.7 | 6/6 | 0.144 | **FALSE** |
| T2 | NFL LOO consensus line-shop (main+alt): `w_model=0.0 ev>=0.04 scope=all` | 64 / +0.291 | 18 / +0.625 / +1.93 / 0.027 | +1.43 / 0.076 (game_id) | +11.3 / 3.0 | 4/5 | 0.183 | **FALSE** |
| T3 | NFL DFS pick'em vs book consensus/model: `prob=blend ev>=0.02 @-122` | 90 / -0.009 | 31 / +0.174 / +1.09 / 0.137 | +1.09 / 0.138 (game_id) | +5.4 / 2.2 | 4/6 | 0.217 | **FALSE** |
| T1 | NBA pts lines vs props_script legs model: `side=both ev>=0.2 odds<=+300` | 3371 / +0.014 | 3270 / +0.033 / +1.46 / 0.072 | +1.51 / 0.065 (date) | +106.6 / 67.1 | 8/14 | 0.183 | **FALSE** |
| T1 | NBA pts lines vs GBM projection: `side=under ev>=0.12 odds<=+150` | 2588 / +0.012 | 1659 / +0.055 / +2.05 / 0.020 | +2.10 / 0.018 (date) | +90.8 / 39.5 | 10/14 | 0.122 | **FALSE** |
| T2 | NBA pts LOO consensus line-shop: `w_model=0.25 ev>=0.04 scope=all odds<=+200` | 2618 / +0.037 | 2281 / +0.032 / +1.30 / 0.097 | +1.41 / 0.080 (date) | +73.7 / 43.9 | 7/14 | 0.195 | **FALSE** |
| T1 | NHL SOG/points vs model (best price across books): `side=under ev>=0.0 odds<=+300` | 295 / +0.057 | 131 / -0.086 / -1.05 / 0.853 | -1.08 / 0.859 (game_id) | -11.2 / 15.1 | 2/3 | 0.859 | **FALSE** |
| T1 | MLB Ks vs model (DK/FD, am+late snaps): `side=under ev>=0.08 odds<=+150` | 169 / +0.100 | 65 / -0.115 / -0.95 / 0.830 | -0.98 / 0.836 (date) | -7.5 / 13.4 | 5/11 | 0.859 | **FALSE** |
| T2 | MLB Ks DK-vs-FD stale book: `w_model=0.25 ev>=0.02 (other-book)` | 163 / +0.023 | 84 / -0.099 / -0.92 / 0.822 | -1.01 / 0.844 (date) | -8.3 / 12.6 | 5/12 | 0.859 | **FALSE** |
| T4 | UFC method props model vs market: `model>=mkt+0.03 meth=DEC odds<=+400` | 1457 / +0.007 | 524 / +0.067 / +0.98 / 0.163 | +1.03 / 0.151 (date) | +34.9 / 46.7 | 7/9 | 0.217 | **FALSE** |
| T4 | UFC prior lead (fixed, pre-registered): `model>=mkt+0.05 all methods any price` | 1132 / -0.074 | 509 / +0.286 / +2.93 / 0.002 | +3.08 / 0.001 (date) | +145.3 / 27.4 | 5/9 | 0.020 | **FALSE** |

NFL DFS pick'em vs book consensus/model sensitivity: at -119/leg: ROI +0.187; hit 0.645 (n=31)

## Mechanism (one line each)

* **NFL alt+main ladder vs model** — Alt ladders are priced off the book's own main line with a generic distribution; a player-specific script model could know the shape (snap share, game script) better than the book's ladder.
* **NFL alt-ladder only vs model** — Alt ladders are priced off the book's own main line with a generic distribution; a player-specific script model could know the shape (snap share, game script) better than the book's ladder.
* **NFL LOO consensus line-shop (main+alt)** — Books copy each other's main line; when one book's quote (often an auto-generated alt rung) lags the others after news, the other books' consensus is the sharper price and the laggard is beatable.
* **NFL DFS pick'em vs book consensus/model** — DFS pick'em posts one line per player and moves it slowly; when sportsbooks move after news the DFS line is stale and the side the books moved toward wins more than the ~55% flex breakeven.
* **NBA pts lines vs props_script legs model** — Model sees usage redistribution / minutes context the line may not fully price; alt rungs are set off the main line with a generic spread.
* **NBA pts lines vs GBM projection** — Model sees usage redistribution / minutes context the line may not fully price; alt rungs are set off the main line with a generic spread.
* **NBA pts LOO consensus line-shop** — Soft books (Bovada/BetRivers alt rungs, slow movers) hang lines that lag the other books' consensus ~12h before tip; the consensus of the other books is the sharper estimate.
* **NHL SOG/points vs model (best price across books)** — Model TOI/PP role vs books' generic SOG ladders.
* **MLB Ks vs model (DK/FD, am+late snaps)** — Pitcher-leash model vs books' K lines.
* **MLB Ks DK-vs-FD stale book** — Two books disagreeing within a scrape: the other book's price as fair value.
* **UFC method props model vs market** — Method-of-victory props are thin, high-hold markets priced mostly off the moneyline; fighter finish/durability tendencies are under-weighted.
* **UFC prior lead (fixed, pre-registered)** — same as UFC family

## Re-verification of the prior NBA rule (25% GBM model / 75% LOO consensus, EV >= 4%, odds <= +200)

Prior claim (results/SUPER_ENGINE.md): +3.9% on 3,059 bets in 2025-26 (season split, discovery = 2024-25). Under the protocol's date split most of Oct-Dec 2025 moves into discovery. Fixed rule, no re-tuning:

|                                   |    n |    roi |      z |      p |    units |   max_dd | months_pos   |
|:----------------------------------|-----:|-------:|-------:|-------:|---------:|---------:|:-------------|
| ('disc(first 60% dates)', 'all')  | 2618 | 0.0369 | 1.7211 | 0.0426 |  96.5357 |  36.7424 | 4/9          |
| ('disc(first 60% dates)', 'main') | 2080 | 0.0100 | 0.4749 | 0.3174 |  20.7014 |  37.9279 | 4/9          |
| ('disc(first 60% dates)', 'alt')  |  954 | 0.0669 | 1.5598 | 0.0594 |  63.8501 |  29.3900 | 6/8          |
| ('lockbox(last 40%)', 'all')      | 2281 | 0.0323 | 1.2970 | 0.0973 |  73.6816 |  43.8819 | 4/6          |
| ('lockbox(last 40%)', 'main')     | 1460 | 0.0341 | 1.3663 | 0.0859 |  49.7600 |  18.7375 | 4/6          |
| ('lockbox(last 40%)', 'alt')      | 1457 | 0.0265 | 0.7613 | 0.2232 |  38.6588 |  48.8500 | 4/6          |
| ('2024-25', 'all')                | 1840 | 0.0276 | 1.1304 | 0.1292 |  50.7678 |  26.9692 | 3/7          |
| ('2024-25', 'main')               | 1601 | 0.0149 | 0.6230 | 0.2666 |  23.8834 |  34.4108 | 3/7          |
| ('2024-25', 'alt')                |  429 | 0.0779 | 1.2219 | 0.1109 |  33.4004 |  25.0100 | 4/6          |
| ('2025-26', 'all')                | 3059 | 0.0390 | 1.8098 | 0.0352 | 119.4495 |  45.1007 | 4/7          |
| ('2025-26', 'main')               | 1939 | 0.0240 | 1.1110 | 0.1333 |  46.5779 |  30.8285 | 5/7          |
| ('2025-26', 'alt')                | 1982 | 0.0349 | 1.1669 | 0.1216 |  69.1086 |  51.6000 | 5/7          |

## UFC prior lead re-derived ('model >= market + 5 pts', all methods, posted price)

| window               |    n |     roi |       z |      p |    hit |   implied |
|:---------------------|-----:|--------:|--------:|-------:|-------:|----------:|
| <=2022 (disc)        | 1132 | -0.0738 | -1.5686 | 0.9416 | 0.2951 |    0.2698 |
| 2023+ (lockbox)      |  509 |  0.2855 |  2.9318 | 0.0017 | 0.3183 |    0.2238 |
| 2021+ (prior report) |  910 |  0.1691 |  2.5511 | 0.0054 | 0.3110 |    0.2371 |

By season:

|   season |   size |   mean |     sum |
|---------:|-------:|-------:|--------:|
|     2018 |    273 | -0.118 | -32.158 |
|     2019 |    229 | -0.141 | -32.184 |
|     2020 |    229 | -0.121 | -27.822 |
|     2021 |    211 |  0.115 |  24.346 |
|     2022 |    190 | -0.083 | -15.749 |
|     2023 |    155 |  0.257 |  39.905 |
|     2024 |    169 |  0.095 |  16.117 |
|     2025 |    152 |  0.232 |  35.333 |
|     2026 |     33 |  1.635 |  53.967 |

## Bottom line and caveats


**No TRUE edge.** Nothing clears the protocol (multiple-testing significance on the lockbox + discovery-positive +
stable + n >= 100). Closest calls, all to be treated as *forward-test only*, not bets:

* **NFL alt ladders vs props_script model** (+10.8% on 350 lockbox bets, clustered z 1.80, BH q 0.14): the lockbox is
  only 2 NFL weeks (33 games), discovery z was 1.05 and the config is the best of 36 tried. On ALL alt-ladder over
  quotes the model is *under*-confident by ~2-4 pts in every probability bin and the book's vigged price has the
  better Brier (0.1911 vs 0.1939) -- the model is not sharper than the ladder; selection picks up noise.
* **NBA GBM model, unders with EV >= 12%** (+5.5%, n 1,659, clustered p 0.018, BH q 0.12): fails BH/Bonferroni.
* **Prior NBA rule 25% model / 75% LOO consensus, EV >= 4%**: re-verified on the protocol date split it is +3.7%
  (disc, n 2,618) and +3.2% (lockbox, n 2,281, z 1.30-1.41, p 0.08-0.10). The previous "+3.9% on 3,059 in 2025-26"
  is reproduced exactly with the season split, but it is not significant at the protocol's bar, and only 7/14
  months are positive. The discovery search over 84 configs independently picked the same rule. Live it needs
  multi-book quotes (ODDS_API_KEY / The Odds API); FanDuel alone cannot form a leave-one-out consensus.
* **UFC 'model >= market + 5 pts'** (prior untested lead): discovery (<=2022) is NEGATIVE (-7.4%, n 1,132);
  2023+ is +28.6% (n 509, p 0.002) but 2026 alone contributes +54u on 33 bets, only 5/9 seasons are positive, the
  rule was found on a 2021+ report (not blind), and the method-prop overround in the source drops from ~1.22 to
  ~1.14 in 2025-26 (odds source/book mix changed), so the late-sample profit is partly a price-source artefact.
  Re-deriving the threshold on <=2022 only picks `DEC, +3 pts` -> lockbox +6.7% (z 0.98). FALSE.
* **DFS pick'em** (PrizePicks/Underdog/Pick6 at -122/leg): ~2.9k DFS legs have a same-scrape sportsbook
  consensus but only 121 clear the chosen EV bar; disc -0.9% (n 90), lockbox +17% on 31 (noise). The dataset's dfs_*_odds column is a flat -137 placeholder.
* **NHL / MLB** 2026 real lines: model-vs-price and DK-vs-FD stale-book both negative on the lockbox.

Data notes: (1) In nfl_props_alt the FanDuel integer, over-only lines are milestones "L+" (their price equals
DraftKings' L-0.5 rung); the file labels actual == L as "push" -- here they are settled as wins. (2) NFL quotes are
the last pre-kick scrape per book/line; LOO consensus uses only quotes from the SAME scrape (no look-ahead).
(3) NBA quotes are one scrape ~12 h before tip. (4) NHL prices are "best price across books" (not one book).
(5) NFL/NBA ladder probabilities are interpolated from the 4-point milestone grid (probit-linear in k), lines
outside the grid +/-25% (NFL) / +/-3 pts (NBA) are dropped.

