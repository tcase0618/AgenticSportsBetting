# Script-driven Player Props + 3/5/7-leg Parlays — win% first

User goal: script each game BEFORE it starts (pace, game flow, blowout risk, matchups, usage), derive each
player's prop projection from that script plus the player's own tendencies, pick the props most likely to hit,
build 3/5/7-leg parlays, and learn from every hit/miss across 10 seasons. Scoreboard = HIT RATE.

## Inputs
* Game scripts: results/scripts/<sport>_games.parquet; game predictors: results/predict/<sport>_preds.parquet.
* Player scripts (already coupled to the team script): results/scripts/<sport>_players.parquet
  (NBA pts/reb/ast/3pm 2015-26; NFL pass/rush/rec yds, receptions, completions 2016-26; NHL SOG/points 2015-26;
  MLB pitcher Ks 2016-26). UFC: results/scripts/ufc_fights.parquet method/round probabilities.
* Real historical lines: data/processed/nba_props.csv.gz (NBA points 2021-26, 238k), nfl_props.csv (2017-26, 12.6k),
  nfl_props_alt.csv.gz (2026 alt ladders), nhl_props.csv, mlb_props.csv (2026 only).

## Two prop universes (report both, separately)
A. REAL LINES: over/under the posted line (where lines exist). Pick side = higher model probability.
B. MILESTONES ("X+" ladders, FanDuel style, defined for every player-game 2016+, no line needed):
   NBA 10+/15+/20+/25+ pts, 4+/6+/8+/10+ reb, 2+/4+/6+/8+ ast, 1+/2+/3+ 3PM; NFL 25+/40+/60+/80+ rec/rush yds,
   2+/3+/4+/5+ rec, 200+/250+ pass yds; NHL 1+/2+/3+/4+ SOG, 1+ point; MLB 3+/4+/5+/6+/7+ Ks;
   UFC: fight goes distance / doesn't, winner by KO/SUB/DEC, fight ends under/over 1.5/2.5 rounds.
   Universe B tests PREDICTION SKILL: only pick legs the model says ≥ 70/75/80/85/90% — report realized hit rate.
   (Every 1+ rebound leg is 99% — so also report "hit rate above naive base rate" for the threshold/position.)

## Model = script + tendencies + learning (strict walk-forward)
Player probability for each prop from: script-coupled mean (usage share × scripted team volume), minutes /
snaps / TOI expectation, blowout risk from the game script (starters sit), pace, opponent defense vs position,
home/away, rest/b2b, recent form (shifted), player-specific variance (dispersion), injuries/teammate absences
when available (usage redistribution), and an online calibrator that learns from every past hit/miss
(per stat × threshold × confidence bin), updated in date order. Fit / choose on seasons ≤ 2020 (or first half
of available real-line years), REPORT later seasons only. Show learning curve: calibration error by season.

## Parlays (3, 5, 7 legs) — report hit rate per day for each construction, out-of-sample
1. Top-N most-likely legs across the whole day (cross-game, max 1 leg per player).
2. Script-correlated same-game stacks: legs that the script says move together (e.g. favorite's star over +
   game over + opponent under) — use learned correlations (engine/scripts/sgp_backtest.py has them).
3. Mixed: 1 game ML from results/predict (85%+ tier) + props.
For each: predicted joint hit prob (with correlation) vs realized hit rate, n days, and a "typical price" column
(one column, no commentary). Also the most likely 3/5/7-leg ticket each day and its realized hit rate.

## Output
results/props_script/<sport>_props.md, <sport>_legs.parquet (<40MB; legs with p, hit, side, thresholds),
<sport>_parlays.csv; engine/props_script/<sport>.py with `build_tickets(date)` returning tonight's 3/5/7-leg
tickets with each leg's probability and the joint probability. Don't git commit. Each sport run < 25 min.
Honest numbers; flag small samples.
