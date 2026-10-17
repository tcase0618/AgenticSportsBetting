# Script-fit same-game parlays: walk-forward backtest, 2016-2026

Code: `engine/scripts/sgp_backtest.py`, which runs in about 14 min (`--eval` re-scores the saved candidates in 15 s), and `engine/scripts/sgp_builder.py`, the nightly API. Data: `sgp_candidates.parquet` (407,880 candidate SGPs, 19 MB), `sgp_learning_curves.csv`, `sgp_edge.csv`, `sgp_corr_vs_indep.csv` and `sgp_state.json` (learned correlations, calibrators and coupling betas).

**There are no historical SGP prices anywhere.** Every ROI below is computed against an *assumed* SGP price. That price is the market-fair joint, cut by a hold of 0, 10, 20 or 30%. The exceptions are real leg prices, which exist only for NBA points props (2024-25 and 2025-26), UFC method props (a real price for "X wins and goes the distance / inside the distance") and the small 2026 NFL, NHL and MLB prop sets.

## What was tested
- **Games covered:** every game from 2016 on with a closing line, in all sports. NCAAM includes only games with lines (42.7k of 64.9k); UFC includes only fights that pass the odds-quality gate (most of 2025-26 fails it).
- **Legs:** taken in the script's direction against the line. These are spread (or ML for MLB/NHL, plus ML as an alternative), game total and player props. Each game gets up to 4 props; each player gets 1.
- **Candidates per game:** side+total, ML+total, side+total+P1, side+total+P1+P2, side+P1, total+P1 and P1+P2.
- **Prop lines:** real lines where they exist. Otherwise a **proxy line**: the player's trailing 10-game average from prior games only, rounded to x.5. Proxy lines are priced at the over-rate such lines had on prior dates.
- **Joint probability:**
  - **Script:** Monte Carlo. `simulate_scores` gives the team scores, and each player's stat is coupled to his team's simulated score. The coupling is structural (share × sd_team / sd_player) for NBA points and NHL points. For the other stats it is learned from prior seasons' residuals.
  - **Market-fair:** de-vigged marginals plus a Gaussian copula whose correlations are learned online from settled games only.
  - **Independent:** the product of the de-vigged marginals.
  - **Hybrid:** script marginals combined with the learned correlations.
- **Assumed leg prices:** spreads and totals with no juice in the data are assumed to be -110, i.e. a fair price of 0.5.
- **Learning loop, run date by date:** after each date settles, three things update: the correlation table (2x2 counts → tetrachoric correlation, shrunk by n/(n+300)), an online logistic calibration of the script joint (`p_script_cal`) and a script+market combiner. Nothing uses the current date's outcome.

## Learning curves (log loss of the joint; game-line and real-prop SGPs; full per-season curves in `sgp_learning_curves.csv`)
| sport | period | n | hit | script raw | script calibrated | market-fair | independent | ECE raw → cal | ECE market |
|---|---|---|---|---|---|---|---|---|---|
| NBA | 16-20 / 21+ | 12.5k / 26.9k | .220/.208 | .524/.520 | .513/.499 | **.510/.490** | .510/.491 | .065→.022 / .092→.027 | .011/.007 |
| NFL | 16-20 / 21+ | 2.6k / 3.3k | .215/.227 | .529/.542 | .517/.533 | **.510/.523** | .509/.523 | .073→.033 / .068→.033 | .022/.021 |
| NCAAF | 16-20 / 21+ | 6.6k / 8.9k | .194/.219 | .477/.513 | .461/.500 | **.456/.493** | .456/.494 | .078→.024 / .071→.025 | .008/.012 |
| NCAAM | 16-20 / 21+ | 32.9k / 38.6k | .243/.238 | .547/.535 | .538/.526 | **.534/.521** | .534/.521 | .063→.022 / .061→.018 | .010/.011 |
| NHL | 16-20 / 21+ | 6.0k / 6.1k | .239/.231 | .549/.535 | .548/.534 | **.546/.531** | .545/.532 | .035→.027 / .034→.023 | .010/.011 |
| MLB | 16-20 / 21+ | 10.2k / 11.9k | .251/.248 | .565/.566 | .559/.558 | **.557/.553** | .557/.553 | .057→.024 / .063→.024 | .013/.009 |
| UFC | 16-20 / 21+ | 3.7k / 3.3k | .142/.147 | .399/.391 | .381/.378 | **.373/.374** | .376/.376 | .074→.018 / .060→.011 | .014/.017 |

Online calibration learns quickly. Within the first season it cuts the script's calibration error from about 6-9% to about 2-3%. The raw script joint is **overconfident**, because selecting the script's favourite side, total and props is a winner's curse. Even after calibration, the **market-fair joint has lower log loss in every sport and every season block**. Calibration narrows the gap to the market but never closes it.

## Do the learned correlations beat independence? (2021+ holdout, same marginals)
| set | n | Δ log loss (copula − independent) | z (one SGP per game) |
|---|---|---|---|
| NBA real points props | 11,704 | **−0.0026** | **−4.0** |
| NHL props (mostly proxy) | 28,639 | −0.0111 | −2.8 |
| NBA proxy props | 26,279 | −0.0021 | −2.2 |
| NCAAF game lines | 8,918 | −0.0009 | −2.1 |
| other game-line sets (NBA, NFL, NCAAM, NHL, MLB) | | about 0 (±0.0001) | within ±1.4 |
| all | 209,705 | **−0.0022** | **−3.0** |

**Yes, where the correlation is real.** The gains come from player-vs-game legs and from NCAAF big spreads. Learned correlations at the end of the run:

| pair | correlation |
|---|---|
| NBA points over ~ game over | +0.17 |
| NBA points over ~ team cover | +0.13 |
| NBA points, teammate pairs | −0.02 |
| NHL points ~ game over | +0.37 |
| NHL points ~ team win | +0.40 |
| MLB Ks ~ game over | −0.19 |
| MLB Ks ~ team win | +0.19 |
| NFL RB rush yards ~ team cover | +0.23 |
| NCAAF favourite cover ~ over (14.5+) | +0.29 |

These match the earlier `engine/correlation.py` validation. For side+total pairs in the NFL, NBA, NHL and MLB the correlations are about 0, so independence is just as good.

## Edge test (rule chosen on 2016-20, reported on 2021+; one SGP per game; real-market legs only)
**Rule:** bet when `p_script_cal / p_market_fair ≥ k`, with k chosen per sport on 2016-20. Choosing k on the 0% hold column gives the same k at every hold. ROI is computed at price (1 − hold) / p_market_fair. "Real" uses the product of real leg prices × (independent product / market-fair joint); for UFC it is the actual method-prop price.

| sport | k | n 21+ | hit | p_cal / p_mkt | ROI 0% (z) | 10% | 20% | 30% | max DD at 20% hold (u) | real-price ROI (n, z) |
|---|---|---|---|---|---|---|---|---|---|---|
| NBA | 1.5 | 1,586 | .136 | .212/.116 | +21.6% (2.2) | +9.4% | −2.7% | −14.9% | 141 | −5.7% (801, −0.6) |
| NHL | 1.3 | 567 | .187 | .226/.161 | +17.8% (1.6) | +6.0% | −5.8% | −17.5% | 48 | −42.6% (30) |
| MLB | 1.3 | 1,236 | .244 | .333/.226 | +10.9% (1.6) | −0.1% | −11.2% | −22.3% | 154 | −27.0% (43) |
| UFC | 1.15 | 952 | .111 | .160/.105 | +9.4% (0.7) | −1.6% | −12.5% | −23.4% | 169 | **−20.7% (952, −2.3)** |
| NCAAM | 1.5 | 2,123 | .115 | .190/.105 | +6.5% (0.7) | −4.1% | −14.8% | −25.4% | 351 | n/a |
| NFL | 1.3 | 173 | .173 | .297/.173 | −17.7% (−1.2) | −25.9% | −34.2% | −42.4% | 80 | +3.2% (70, 0.2) |
| NCAAF | 2.0 | 223 | .045 | .138/.057 | −43.3% (−2.0) | −48.9% | −54.6% | −60.3% | 122 | n/a |

**NBA real points props** (k chosen on 2024-25, reported on 2025-26): n = 449, hit 16.0%. Against a calibrated prediction of 25.9%, the market-fair joint predicted 14.6%. ROI is +2.0% at 0% hold, −8.2% at 10% hold and **−13.9% at real leg prices** (z −1.2).

The hybrid and raw-script variants tell the same story. Their best 2021+ result at 0% hold is NHL hybrid +18.9% (z 2.3). Every variant is negative at a 20% hold, and NBA and UFC are significantly negative at real prices (−21%, z −2.3 and −20%, z −3.1).

**Proxy-line SGPs** show +100-280% "ROI" (z 4-8). This is an artefact, not an edge. The proxy is a naive trailing average, so it is far weaker than a book line, and its pooled over-rate is miscalibrated for the extreme legs the script selects. For example, NHL selections hit 7.0% where the proxy price implied 2.6%. Real books do not post these lines, and on real lines the same script loses (see the NBA, NFL, NHL and MLB prop learning reports).

**Multiple testing:** the selection grid covered 3 probability variants × 7 sports × 8 values of k. The best holdout z at 0% hold is 2.2, which is below a Bonferroni bar of about 3.0. Each 2-leg SGP at -110 already carries about 9% hold, and real SGP holds are usually 15-30%.

## Nightly API
```
from engine.scripts.sgp_builder import build_sgps
build_sgps(script_row, player_rows, lines, sport="nba", edge=0.05)
```
- `script_row` is a row from `<sport>_games.parquet`.
- `player_rows` has the columns player, stat, mu, sd, [nb_r], team_is_home, [beta], [trail_avg].
- `lines` = {spread_home, total_line, home_ml, away_ml, over_odds, under_odds, props: {(player, stat): {line, over_odds, under_odds}}}.

It returns ranked SGPs, each with: legs, `p_script`, `p_script_cal` (the learned calibration from `sgp_state.json`), `p_market_fair`, `p_independent`, `fair_american` (e.g. "+285"), `market_fair_american` and `bet_if_fanduel_at_least` (fair decimal × (1 + edge), in American format). Smoke test: `python -m engine.scripts.sgp_builder`. Given the results above, use `market_fair_american` as the honest fair price and treat `fair_american` (calibrated script) as an opinion.

## Verdict
**No demonstrated SGP edge in any sport.** The online learning works: calibration error drops by about two thirds, and the learned correlations beat independence on the 2021+ holdout (z −3.0 overall, −4.0 on NBA real props). That makes the market-fair price a good tool for spotting a mispriced SGP at a book. The script itself, however, is less accurate than the market's marginals in every sport. The apparent 0%-hold "edges" (NBA +22%, NHL +18%) are not significant after multiple testing. They vanish at a 20% hold and turn negative at real NBA and UFC prices.

The only way to profit here is shopping: bet an SGP only when FanDuel's quoted price beats `market_fair_american` by more than the required edge. That cannot be backtested, because historical SGP quotes do not exist. Log live quotes from now on.
