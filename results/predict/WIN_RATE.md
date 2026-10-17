# Win-Rate Scoreboard — every sport, 2021+ out-of-sample (models picked on 2018–2020)

| Sport | SU winner % (model / market fav) | Best ML tier (n) → win% | Daily top-3 ML parlay hit | ATS/RL/PL best tier | O/U best tier | Score accuracy |
|---|---|---|---|---|---|---|
| NCAAF | 76.2 / 76.2 | 85%+ (1716) → 94.5% | 63.3% | 51.5% | 51.8% | margin MAE 12.1 |
| NCAAM | 72.2 / 72.2 | 85%+ (4777) → 92.0% | 80.9% | 59.3% (n=150) | 50.7% | margin MAE 8.8, total 13.1 |
| NBA | 67.6 / 67.8 | 85%+ (577) → 90.6% | 48.9% | 50.8% | 50.8% | margin MAE 10.6, total 14.2 |
| UFC | 68.3 / 68.4 | 85%+ (196) → 89.8% | 47.8% / event | – | – | winner+method 35.6%, +round 29.9% |
| NHL | 60.6 / 61.0 | 75–80% (298) → 77.5% | 29.5% | 67.7% (+1.5) | 54.3% (n=208) | exact score 7.5% |
| NFL | 66.2 / 66.2 | 75–80% (189) → 74.1%; 85%+ → 92.4% | 46.4% | 50.8% | 48.9% | margin MAE 9.8, total 10.3 |
| MLB | 58.2 / 58.2 | 70–75% (573) → 71.7% | 25.4% | 65.1% (+1.5) | 51.1% | exact score 3.1% |

Formula, every sport: predicted margin ≈ 1.0 × closing-line margin + small script term; win prob ≈ de-vigged
closing ML (NHL/UFC: 0.84 × market log-odds + 0.20 × script log-odds). Scripts/matchups/injuries/pace add ≤0.4 pts.
More history: plateaus after ~1–4 seasons (MLB keeps creeping +0.9 pts); recency weighting never helps.
Details: results/predict/<sport>_predict.md. Live: engine/predict/<sport>.py predict(slate_df).
