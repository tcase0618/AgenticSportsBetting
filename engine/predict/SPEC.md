# Game Predictor Spec — win% first

Goal (user): a formula per sport that predicts the WINNER and the FINAL SCORE of every game, learned from every
game of every prior season (2016 → now). The scoreboard is WIN RATE, not ROI.

## Inputs
`results/scripts/<sport>_games.parquet` (script-then-learn output: Kalman ratings, pace/efficiency, pitchers,
goalies, QB, injuries, rest, travel, weather, market lines) + anything in data/processed. Market lines
(de-vigged ML via engine.devig odds_ratio, spread, total) ARE allowed as features: the goal is accuracy.

## Method (strict walk-forward, no leakage)
* For each season S from 2018 on: fit on all games with season < S (plus, optionally, earlier games of S via
  monthly refits), predict season S. Every feature must be known before kickoff (script columns are already
  pre-game; anything else must be shifted).
* Targets: P(home win) [classifier, calibrated], home_score, away_score [regressors or joint Poisson/NB/normal].
  Final predicted score = rounded joint mode/median consistent with the predicted winner. Also derive P(cover)
  and P(over) from the predicted score distribution.
* Try: logistic/ridge on stacked features, gradient boosting (LightGBM/HistGB), and a stack of both with the
  market. Pick the model per sport on seasons 2018–2020, REPORT 2021+.
* Keep learning: the Kalman script features update after every game; the model refits each season (or monthly).

## Report `results/predict/<sport>_predict.md` + `<sport>_preds.parquet`
1. Straight-up winner accuracy per season: model vs market favorite vs script alone.
2. Score accuracy: MAE home/away/margin/total, % of games with exact winner AND margin within 3/7 (sport-scaled),
   exact-score hit rate where meaningful (NHL/MLB/soccer-like).
3. WIN RATE BY CONFIDENCE TIER (the user's main ask): bins of model P(pick) e.g. 50-55,55-60,...,85+, with count
   per season and hit rate, for (a) moneyline winner, (b) spread pick, (c) total pick. Show the tier where win%
   is highest with ≥100 games 2021+. Include a "break-even at typical price" column (one column, no commentary).
4. High-confidence parlay hit rates: per day, the top-1/2/3/4/5 most confident ML picks parlayed: hit rate 2021+.
5. The formula: top features / coefficients in plain English ("predicted margin = 0.8×market spread + ...").
6. Ablation: what each feature group adds to accuracy.

## Live function in `engine/predict/<sport>.py`
`predict(slate_df) -> DataFrame[home, away, p_home, pick, pick_prob, pred_home_score, pred_away_score,
pred_margin, pred_total, spread_pick, spread_prob, total_pick, total_prob, tier]` using the latest fitted model
(save model to results/predict/<sport>_model.pkl, small). Document what slate columns it needs.

Rules: only edit engine/predict/<your sports>.py and results/predict/<your sports>*; don't git commit; keep
each sport run < 20 min; honest numbers.
