"""
NBA game predictor (win% first). See engine/predict/SPEC.md; all logic in engine/predict/common_team.py.

    python -m engine.predict.nba      -> results/predict/nba_predict.md, nba_preds.parquet, nba_model.pkl

Live: predict(slate_df) -> DataFrame[home, away, p_home, pick, pick_prob, pred_home_score, pred_away_score,
pred_margin, pred_total, spread_pick, spread_prob, total_pick, total_prob, tier, model_variant].
slate_df rows look like results/scripts/nba_games.parquet rows for upcoming games: the script engine's pre-game
columns (at minimum home, away, margin, total, sd_margin, mu_home, mu_away; every other script column it has
improves the prediction, missing ones are median-imputed) plus closing/current spread_home (negative = home
favoured), total_line, home_ml, away_ml (American). Rows without spread_home use the script-only model.
"""
from __future__ import annotations

from engine.predict import common_team as C

SPORT = "nba"


def predict(slate_df, model=None):
    return C.predict(SPORT, slate_df, model)


def main():
    print(C.run(SPORT))


if __name__ == "__main__":
    main()
