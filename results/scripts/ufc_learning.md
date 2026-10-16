# UFC fight scripts: learning report

Fights scripted 2016-2026: 5,414 (5,305 with a KO/SUB/DEC result; 4,433 also have all six method props + moneyline). Fighter state warmed up on every UFC fight since 1993 (ufcstats round stats, greco dump); the method learner trains online from 2001. Learner AdaGrad lr/l2 chosen on 2016-2020: (0.02, 0.0001).

## De-vigging the method props

Each fighter's KO/TKO, SUB and DEC props are separate yes-props, but the six (A/B x 3) partition every non-draw outcome, so their implied probabilities sum to 1 + overround (median overround 21.8%). Candidates (6-way log loss on 2016-2020 fights with all six props):

* `mult6` - divide the six implied probabilities by their sum
* `power6` - p_k = q_k^k, k solved so sum = 1 (removes more vig from long shots)
* `ml_anchor_mult` / `ml_anchor_power` - fighter win probability from the de-vigged moneyline (more liquid, lower margin), split across that fighter's three methods in proportion to their (raw / power-de-vigged) prop prices

|                 |   ll6_2016_20 |
|:----------------|--------------:|
| mult6           |        1.5312 |
| power6          |        1.5242 |
| ml_anchor_mult  |        1.5278 |
| ml_anchor_power |        1.5237 |

Chosen market benchmark: **ml_anchor_power**. Draws (~1%) are not priced by the six props and are excluded from log loss; NC/overturned/DQ/CNC are void.

## Per-season learning curve (log loss; lower is better)

|   season |   fights |   ll6_script |   ll6_market |   ll6_blend |   win_script |   win_market_ml |   win_blend |   dist_script |   dist_market |   dist_blend |
|---------:|---------:|-------------:|-------------:|------------:|-------------:|----------------:|------------:|--------------:|--------------:|-------------:|
|     2016 |      458 |       1.6118 |       1.5399 |      1.5375 |       0.6648 |          0.6282 |      0.6282 |        0.664  |        0.6452 |       0.6452 |
|     2017 |      422 |       1.6143 |       1.5454 |      1.5409 |       0.6651 |          0.6111 |      0.6111 |        0.6716 |        0.6656 |       0.6633 |
|     2018 |      442 |       1.5839 |       1.5087 |      1.5058 |       0.654  |          0.5963 |      0.5963 |        0.65   |        0.6429 |       0.641  |
|     2019 |      479 |       1.5766 |       1.5052 |      1.5034 |       0.6676 |          0.632  |      0.632  |        0.6648 |        0.6434 |       0.6434 |
|     2020 |      428 |       1.6112 |       1.5211 |      1.5206 |       0.6599 |          0.6194 |      0.6194 |        0.6647 |        0.6534 |       0.6518 |
|     2021 |      465 |       1.5975 |       1.5599 |      1.5542 |       0.6598 |          0.6248 |      0.6248 |        0.68   |        0.6794 |       0.6769 |
|     2022 |      441 |       1.6284 |       1.5269 |      1.5298 |       0.6471 |          0.5902 |      0.5902 |        0.6717 |        0.6596 |       0.6587 |
|     2023 |      375 |       1.6162 |       1.5684 |      1.5623 |       0.6354 |          0.582  |      0.582  |        0.6605 |        0.6432 |       0.6421 |
|     2024 |      333 |       1.5451 |       1.4837 |      1.4781 |       0.6351 |          0.5915 |      0.5915 |        0.6633 |        0.669  |       0.6631 |
|     2025 |      477 |       1.5579 |       1.5757 |      1.5519 |       0.6284 |          0.5915 |      0.5915 |        0.6646 |        0.7057 |       0.6911 |
|     2026 |      113 |       1.5224 |       1.4911 |      1.4744 |       0.6071 |          0.5244 |      0.5244 |        0.7146 |        0.7591 |       0.7403 |

**2016-2020 (tuning)** (2,229 fights): 6-way winner x method log loss script 1.5991 / market 1.5237 / blend 1.5213;  3-way method (KO/SUB/DEC) script 0.9578 / market 0.9411 / blend 0.9382;  winner script 0.6624 / ML 0.6178 / blend 0.6178

**2021-2026 holdout** (2,204 fights): 6-way winner x method log loss script 1.5865 / market 1.5431 / blend 1.5346;  3-way method (KO/SUB/DEC) script 0.9666 / market 0.9707 / blend 0.9626;  winner script 0.6399 / ML 0.5932 / blend 0.5932

Blend weights chosen on 2016-2020: method log-linear pool w_market = **0.85**, winner w_market = **1.0**.

Holdout calibration-in-the-large (2021+):

| outcome   |   mean_script |   mean_market |   actual |
|:----------|--------------:|--------------:|---------:|
| A_KO      |         0.168 |         0.186 |    0.183 |
| A_SUB     |         0.098 |         0.103 |    0.103 |
| A_DEC     |         0.249 |         0.254 |    0.275 |
| B_KO      |         0.162 |         0.155 |    0.152 |
| B_SUB     |         0.087 |         0.084 |    0.074 |
| B_DEC     |         0.236 |         0.218 |    0.214 |
| distance  |         0.486 |         0.472 |    0.489 |

Finish-round distribution given a finish (2021+, 1,515 finishes): script log loss 1.0405 vs static 2016-20 base rates 1.1253; expected finish round MAE 0.684.

## Ablation (method learner re-run without each feature group)

ll6 = 6-way winner x method log loss; `hold_d_ll6_vs_all` > 0 means dropping the group HURT the 2021+ holdout (the group helps).

| variant                                 |   tune_ll6 |   hold_ll6 |   hold_ll_win |   hold_d_ll6_vs_all |
|:----------------------------------------|-----------:|-----------:|--------------:|--------------------:|
| all features                            |     1.5984 |     1.5976 |        0.6453 |              0      |
| drop skill rating                       |     1.6021 |     1.6023 |        0.65   |              0.0047 |
| drop finish tendencies (power/chin/sub) |     1.6173 |     1.6166 |        0.6474 |              0.0191 |
| drop striking                           |     1.6083 |     1.6045 |        0.6497 |              0.007  |
| drop grappling                          |     1.6053 |     1.6091 |        0.6484 |              0.0115 |
| drop physical (age/reach/height)        |     1.6086 |     1.6125 |        0.6587 |              0.015  |
| drop layoff/experience                  |     1.5999 |     1.5979 |        0.6454 |              0.0003 |
| drop class base rates & 5 rounds        |     1.6079 |     1.6015 |        0.6453 |              0.0039 |
| base rates only (intercepts)            |     1.7156 |     1.7169 |        0.6931 |              0.1194 |

Final learned weights (standardised features) per method:

|              |     KO |    SUB |    DEC |
|:-------------|-------:|-------:|-------:|
| intercept    | -1.098 | -1.589 | -0.897 |
| skill        |  0.068 |  0.116 |  0.093 |
| ko_win       |  0.161 | -0.044 |  0.008 |
| opp_ko_loss  |  0.1   | -0.019 | -0.09  |
| sub_win      | -0.038 |  0.177 | -0.043 |
| opp_sub_loss |  0.021 |  0.214 | -0.033 |
| dec_rate     | -0.021 | -0.068 |  0.04  |
| opp_dec_rate | -0.072 | -0.081 |  0.068 |
| own_ko_loss  | -0.05  | -0.06  | -0.079 |
| slpm         |  0.028 | -0.067 |  0.002 |
| opp_sapm     |  0.063 |  0.063 |  0.048 |
| strike_diff  |  0.064 | -0.042 |  0.146 |
| sig_acc      |  0.022 |  0.071 | -0.018 |
| kd_rate      |  0.071 | -0.073 | -0.047 |
| td_eff       | -0.028 |  0.095 |  0.049 |
| sub_att      | -0.043 |  0.172 | -0.083 |
| ctrl_diff    | -0.059 |  0.064 |  0.051 |
| opp_td_eff   | -0.099 |  0     | -0.117 |
| age_diff     | -0.167 | -0.137 | -0.099 |
| opp_age      |  0.043 |  0.005 |  0.012 |
| reach_diff   |  0.05  |  0.059 | -0.01  |
| height_diff  |  0.019 |  0.066 |  0.037 |
| layoff       | -0.064 | -0.029 | -0.058 |
| exp          | -0.01  |  0.055 |  0.07  |
| opp_exp      |  0.003 | -0.045 |  0.016 |
| class_ko     |  0.131 | -0.051 | -0.1   |
| class_sub    |  0.019 |  0.022 | -0.04  |
| five_rd      |  0.035 |  0.009 | -0.063 |
| female       | -0.075 | -0.016 |  0.009 |
| title        | -0.022 |  0.001 |  0.001 |
| corner       |  0     |  0     |  0     |

## What the script got wrong most (2021+)

| segment            |   fights |   ll6_script |   ll6_market |     gap |
|:-------------------|---------:|-------------:|-------------:|--------:|
| all                |     2204 |       1.5865 |       1.5431 |  0.0434 |
| 5-round            |      233 |       1.5725 |       1.5795 | -0.007  |
| women              |      356 |       1.523  |       1.4657 |  0.0573 |
| heavyweight        |      155 |       1.5511 |       1.4811 |  0.0701 |
| flyweight/bantam   |      403 |       1.5412 |       1.5272 |  0.0139 |
| A or B UFC debut   |      374 |       1.657  |       1.5726 |  0.0845 |
| both 5+ UFC fights |      867 |       1.5378 |       1.5071 |  0.0307 |
| result KO          |      722 |       1.6804 |       1.6225 |  0.0579 |
| result SUB         |      371 |       2.1454 |       2.004  |  0.1415 |
| result DEC         |     1111 |       1.3389 |       1.3376 |  0.0012 |
| underdog won (ML)  |      680 |       1.7679 |       1.9557 | -0.1877 |

Gap decomposition 2021+: winner part +0.0467, method-given-winner part -0.0033 nats/fight (positive = market better).

## Betting method props where the script disagrees with the market

Rule chosen on 2016-2020 (max ROI, >=100 bets): source=`blend`, EV threshold tau=0.0: tune 164 bets ROI +75.8%; **holdout 2021+ 1379 bets ROI +102.9%** (hit 16.3%, median odds +1800). Flat 1u; NC/DQ/overturned void, draws lose.

Holdout ROI bootstrap 90% interval: [+75.0%, +133.5%]. By leg:

| leg   |   bets |    roi |   hit |
|:------|-------:|-------:|------:|
| A_DEC |    240 | -0.102 | 0.262 |
| A_KO  |    230 |  2.385 | 0.187 |
| A_SUB |    219 |  0.897 | 0.082 |
| B_DEC |    245 |  0.021 | 0.204 |
| B_KO  |    231 |  1.924 | 0.147 |
| B_SUB |    214 |  1.163 | 0.079 |

Full grid:

| src   |   tau | split   |   bets |    roi |   hit |   avg_odds |
|:------|------:|:--------|-------:|-------:|------:|-----------:|
| p     |  0    | tune    |   4493 | -0.195 | 0.128 |      675   |
| p     |  0    | hold    |   5248 |  0.139 | 0.128 |      900   |
| p     |  0.05 | tune    |   3956 | -0.187 | 0.126 |      700   |
| p     |  0.05 | hold    |   4750 |  0.167 | 0.123 |      900   |
| p     |  0.1  | tune    |   3485 | -0.205 | 0.119 |      725   |
| p     |  0.1  | hold    |   4280 |  0.217 | 0.119 |      950   |
| p     |  0.2  | tune    |   2672 | -0.227 | 0.105 |      800   |
| p     |  0.2  | hold    |   3496 |  0.312 | 0.114 |     1100   |
| p     |  0.3  | tune    |   2035 | -0.261 | 0.092 |      850   |
| p     |  0.3  | hold    |   2898 |  0.381 | 0.11  |     1200   |
| p     |  0.5  | tune    |   1195 | -0.181 | 0.09  |      950   |
| p     |  0.5  | hold    |   2045 |  0.616 | 0.107 |     1500   |
| blend |  0    | tune    |    164 |  0.758 | 0.25  |      517.5 |
| blend |  0    | hold    |   1379 |  1.029 | 0.163 |     1800   |
| blend |  0.05 | tune    |     93 |  1.439 | 0.258 |      750   |
| blend |  0.05 | hold    |   1274 |  1.116 | 0.162 |     2000   |
| blend |  0.1  | tune    |     77 |  1.5   | 0.247 |      811   |
| blend |  0.1  | hold    |   1219 |  1.17  | 0.159 |     2000   |
| blend |  0.2  | tune    |     58 |  1.988 | 0.259 |      805.5 |
| blend |  0.2  | hold    |   1086 |  1.345 | 0.154 |     2200   |
| blend |  0.3  | tune    |     43 |  2.833 | 0.326 |     1050   |
| blend |  0.3  | hold    |    940 |  1.537 | 0.15  |     2200   |
| blend |  0.5  | tune    |     26 |  3.529 | 0.269 |     1117.5 |
| blend |  0.5  | hold    |    656 |  1.602 | 0.154 |     2200   |

## SGP fight legs: joint probabilities from the script (2021+)

`script_joint` = the script's joint probability; `independent_product` = P(winner) x P(method) as if the legs were independent (how a naive parlay is priced); `ll_market` = the market's own priced joint (winner-by-method props), on fights where it exists.

| leg                                 |    n |   actual |   script_joint |   independent_product |   ll_joint |   ll_independent |   ll_market |   ll_joint_same_fights |
|:------------------------------------|-----:|---------:|---------------:|----------------------:|-----------:|-----------------:|------------:|-----------------------:|
| A wins & inside distance            | 2962 |   0.286  |         0.2657 |                0.2658 |     0.5641 |           0.5694 |      0.5438 |                 0.5582 |
| A wins & goes distance              | 2962 |   0.2748 |         0.2492 |                0.2491 |     0.5685 |           0.5693 |      0.5717 |                 0.5816 |
| B wins & inside distance            | 2962 |   0.2255 |         0.2487 |                0.2487 |     0.5084 |           0.5128 |      0.4879 |                 0.4981 |
| B wins & goes distance              | 2962 |   0.2137 |         0.2363 |                0.2364 |     0.4989 |           0.4983 |      0.4878 |                 0.4969 |
| A wins by KO                        | 2962 |   0.183  |         0.1677 |                0.1698 |     0.4482 |           0.454  |      0.4498 |                 0.4509 |
| favourite (ML) wins & goes distance | 2962 |   0.316  |         0.2707 |                0.2723 |     0.6128 |           0.6125 |      0.6354 |                 0.6375 |
| favourite (ML) wins inside distance | 2962 |   0.3258 |         0.2903 |                0.2887 |     0.6102 |           0.6148 |      0.6178 |                 0.6286 |

Round-based SGP legs are in the parquet (`p_A_win_over_1_5`, `p_A_win_under_2_5`, `p_under_1_5`, ...): no round-prop prices exist in the data, so they are evaluated only for calibration: mean P(fight ends under 2.5 rounds) 0.461.

## Verdict

On 2021+ the method market (6-way ll 1.5431) is more accurate than the stand-alone script (1.5865); the 2016-20-chosen blend scores 1.5346. See the betting section for whether disagreement is profitable after the vig; treat any holdout ROI with the bootstrap interval in mind.
