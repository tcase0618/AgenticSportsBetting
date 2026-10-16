# Script-then-Learn: Spec for Sport Adapters

Goal: for every game of the last 10 seasons (2016 → present, every sport), write a pre-game SCRIPT of how the
game will go, then reveal the result, measure what was wrong, and learn before the next game. Then turn scripts
into player props and same-game parlays and test for an edge.

## Shared core (do not modify): `engine/scripts/base.py`
`ScriptEngine(Params).run(games)` runs the script→reveal→learn loop with Kalman-updated team offense/defense
ratings, learned league level, home advantage, noise, score correlation and a learned market-blend weight.
`simulate_scores(row)` draws joint (home, away) scores from a script. `learning_report(df)` gives per-season
MAE for own script vs market vs blend. Extend by wrapping or subclassing in your own file.

## Each sport adapter (`engine/scripts/<sport>.py`) must produce

1. `results/scripts/<sport>_games.parquet` (one row per game, 2016+), at minimum:
   `game_id, date, season, home, away, mu_home, mu_away, sd_home, sd_away, rho, margin, total, margin_b, total_b,
   w_mkt, spread_home, total_line, home_ml, away_ml, home_score, away_score, err_*`, plus sport-specific script
   fields (pace/plays/possessions, pass-rate, shots, pitcher, etc.) computed ONLY from information before the game.
2. `results/scripts/<sport>_players.parquet` where player data exists: one row per player-game with
   `game_id, date, player, team, stat, mu, sd (or dist params), share_of_team (if share-based), actual`,
   for the stats that have prop markets (NFL pass/rush/rec yds, receptions; NBA pts/reb/ast/3pm; NHL SOG/points;
   MLB pitcher Ks; UFC: method/round instead of players). Player means must be COUPLED to the team script
   (e.g. player points = learned usage share × scripted team points) so correlations come out of the script.
3. `results/scripts/<sport>_learning.md`: per-season learning curve (script vs market vs blend MAE / log loss),
   what the engine got wrong most (error attribution: pace vs efficiency vs usage vs specific matchup types),
   which script components actually improved accuracy (ablation: drop each enrichment, re-run, compare), and
   whether anything beats the closing line out of sample.

## Rules (non-negotiable)
* Strictly sequential: a game's script uses only games before its date. No season-level aggregates that include
  the game itself. All tuning (Kalman params, feature weights) chosen on 2016–2020, reported on 2021+.
* "Learning from mistakes" = online updates driven by each game's errors (Kalman/online gradient/Bayesian),
  not re-fitting on the future.
* Honest reporting: if the market is more accurate, say so with numbers.
* Don't edit files outside your sport's files; don't git commit. Network: GitHub release downloads and the
  sports APIs (ESPN, NHL api-web.nhle.com, MLB statsapi, Baseball Savant) are reachable.
* Keep each full run under ~20 minutes; cache downloads under data/raw/<source>/ (gitignored).
