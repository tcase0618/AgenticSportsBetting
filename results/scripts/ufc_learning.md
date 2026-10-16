# UFC fight scripts: learning report

Fights scripted 2016-2026: 5,414 (5,305 with a KO/SUB/DEC result; 4,109 also have all six method props + moneyline). Fighter state warmed up on every UFC fight since 1993 (ufcstats round stats, greco dump); the method learner trains online from 2001. Learner AdaGrad lr/l2 chosen on 2016-2020: (0.02, 0.0001).

## De-vigging the method props

Each fighter's KO/TKO, SUB and DEC props are separate yes-props, but the six (A/B x 3) partition every non-draw outcome, so their implied probabilities sum to 1 + overround (median overround 21.9%). Candidates (6-way log loss on 2016-2020 fights with all six props):

* `mult6` - divide the six implied probabilities by their sum
* `power6` - p_k = q_k^k, k solved so sum = 1 (removes more vig from long shots)
* `ml_anchor_mult` / `ml_anchor_power` - fighter win probability from the de-vigged moneyline (more liquid, lower margin), split across that fighter's three methods in proportion to their (raw / power-de-vigged) prop prices

|                 |   ll6_2016_20 |
|:----------------|--------------:|
| mult6           |        1.5243 |
| power6          |        1.5166 |
| ml_anchor_mult  |        1.523  |
| ml_anchor_power |        1.5182 |

**Data-quality gate** (prices only, no outcomes): keep a fight's props only if the six implied probabilities sum to 1.08-1.45 and their implied winner probability is within 0.08 of the de-vigged moneyline. In 2016-2024 this removes ~1-2% of fights; in 2025-26 the source odds are badly broken (a third of fights have the six props summing to < 1, e.g. a -270 favourite at +1700 to win by KO but -110 by decision; blind-betting every KO prop in 2025 'returns' +50-80%), so most 2025-26 fights fail. Fights with all six props / kept after the gate:

|      |   2016 |   2017 |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |   2026 |
|:-----|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| raw  |    458 |    423 |    445 |    481 |    436 |    469 |    442 |    376 |    333 |    484 |    114 |
| kept |    446 |    406 |    432 |    461 |    426 |    454 |    436 |    369 |    329 |    314 |     60 |

Chosen market benchmark: **power6**. Draws (~1%) are not priced by the six props and are excluded from log loss; NC/overturned/DQ/CNC are void.

## Per-season learning curve (log loss; lower is better)

|   season |   fights |   ll6_script |   ll6_market |   ll6_blend |   win_script |   win_market_ml |   win_blend |   dist_script |   dist_market |   dist_blend |
|---------:|---------:|-------------:|-------------:|------------:|-------------:|----------------:|------------:|--------------:|--------------:|-------------:|
|     2016 |      446 |       1.6147 |       1.5427 |      1.54   |       0.6677 |          0.6309 |      0.6309 |        0.6678 |        0.6483 |       0.6484 |
|     2017 |      405 |       1.6111 |       1.5275 |      1.5251 |       0.6642 |          0.6037 |      0.6037 |        0.6705 |        0.6609 |       0.659  |
|     2018 |      429 |       1.585  |       1.4989 |      1.4973 |       0.6532 |          0.5948 |      0.5948 |        0.6534 |        0.6452 |       0.6433 |
|     2019 |      459 |       1.57   |       1.4889 |      1.4886 |       0.6641 |          0.6308 |      0.6308 |        0.6641 |        0.6418 |       0.6419 |
|     2020 |      419 |       1.6102 |       1.5266 |      1.5249 |       0.6607 |          0.6233 |      0.6233 |        0.6608 |        0.6511 |       0.6493 |
|     2021 |      450 |       1.5959 |       1.5463 |      1.542  |       0.6603 |          0.6268 |      0.6268 |        0.6791 |        0.6742 |       0.6722 |
|     2022 |      435 |       1.6286 |       1.5254 |      1.5277 |       0.6476 |          0.5922 |      0.5922 |        0.6718 |        0.6597 |       0.6588 |
|     2023 |      368 |       1.6177 |       1.5618 |      1.5563 |       0.6342 |          0.5776 |      0.5776 |        0.6615 |        0.6446 |       0.6432 |
|     2024 |      329 |       1.5408 |       1.471  |      1.4657 |       0.6329 |          0.5915 |      0.5915 |        0.6612 |        0.6636 |       0.6583 |
|     2025 |      309 |       1.5624 |       1.5243 |      1.5146 |       0.6399 |          0.6149 |      0.6149 |        0.6667 |        0.6574 |       0.6544 |
|     2026 |       60 |       1.4845 |       1.4212 |      1.4152 |       0.581  |          0.4651 |      0.4651 |        0.7103 |        0.7766 |       0.76   |

**2016-2020 (tuning)** (2,158 fights): 6-way winner x method log loss script 1.5977 / market 1.5166 / blend 1.5149;  3-way method (KO/SUB/DEC) script 0.9570 / market 0.9385 / blend 0.9357;  winner script 0.6620 / ML 0.6171 / blend 0.6171

**2021-2026 holdout** (1,951 fights): 6-way winner x method log loss script 1.5893 / market 1.5245 / blend 1.5204;  3-way method (KO/SUB/DEC) script 0.9631 / market 0.9513 / blend 0.9473;  winner script 0.6423 / ML 0.5970 / blend 0.5970

Blend weights chosen on 2016-2020: method log-linear pool w_market = **0.85**, winner w_market = **1.0**.

Holdout calibration-in-the-large (2021+):

| outcome   |   mean_script |   mean_market |   actual |
|:----------|--------------:|--------------:|---------:|
| A_KO      |         0.168 |         0.198 |    0.183 |
| A_SUB     |         0.098 |         0.107 |    0.103 |
| A_DEC     |         0.249 |         0.239 |    0.275 |
| B_KO      |         0.162 |         0.165 |    0.152 |
| B_SUB     |         0.087 |         0.086 |    0.074 |
| B_DEC     |         0.236 |         0.205 |    0.214 |
| distance  |         0.486 |         0.444 |    0.489 |

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
| all                |     1951 |       1.5893 |       1.5245 |  0.0647 |
| 5-round            |      184 |       1.574  |       1.5361 |  0.0379 |
| women              |      315 |       1.5363 |       1.4784 |  0.0579 |
| heavyweight        |      138 |       1.5583 |       1.4667 |  0.0916 |
| flyweight/bantam   |      348 |       1.5421 |       1.5041 |  0.038  |
| A or B UFC debut   |      323 |       1.6619 |       1.5361 |  0.1258 |
| both 5+ UFC fights |      781 |       1.5386 |       1.4854 |  0.0533 |
| result KO          |      635 |       1.6884 |       1.5351 |  0.1533 |
| result SUB         |      327 |       2.1459 |       1.9423 |  0.2036 |
| result DEC         |      989 |       1.3416 |       1.3796 | -0.0381 |
| underdog won (ML)  |      606 |       1.7721 |       1.9543 | -0.1822 |

Gap decomposition 2021+: winner part +0.0486, method-given-winner part +0.0161 nats/fight (positive = market better).

## Betting method props where the script disagrees with the market

Rule chosen on 2016-2020 (max ROI, >=100 bets): source=`p`, EV threshold tau=0.05: tune 3826 bets ROI -22.5%; **holdout 2021+ 3703 bets ROI -17.7%** (hit 11.9%, median odds +800); of which 2021-2024 2872 bets ROI -16.2%, 2025-26 (suspect odds even after the gate) 831 bets ROI -23.0%. Flat 1u; NC/DQ/overturned void, draws lose.

2021-2024 holdout ROI bootstrap 90% interval: [-25.6%, -7.8%]. By leg:

| leg   |   bets |    roi |   hit |
|:------|-------:|-------:|------:|
| A_DEC |    509 |  0.145 | 0.202 |
| A_KO  |    351 |  0.029 | 0.114 |
| A_SUB |    365 | -0.315 | 0.06  |
| B_DEC |    706 | -0.268 | 0.137 |
| B_KO  |    530 | -0.351 | 0.085 |
| B_SUB |    411 | -0.146 | 0.075 |

Full grid:

| src   |   tau | split     |   bets |     roi |     hit |   avg_odds |
|:------|------:|:----------|-------:|--------:|--------:|-----------:|
| p     |  0    | tune      |   4351 |  -0.227 |   0.126 |        675 |
| p     |  0    | hold      |   4167 |  -0.178 |   0.124 |        750 |
| p     |  0    | hold21_24 |   3240 |  -0.177 |   0.121 |        700 |
| p     |  0    | hold25_26 |    927 |  -0.179 |   0.134 |        800 |
| p     |  0.05 | tune      |   3826 |  -0.225 |   0.123 |        700 |
| p     |  0.05 | hold      |   3703 |  -0.177 |   0.119 |        800 |
| p     |  0.05 | hold21_24 |   2872 |  -0.162 |   0.118 |        750 |
| p     |  0.05 | hold25_26 |    831 |  -0.23  |   0.122 |        850 |
| p     |  0.1  | tune      |   3368 |  -0.242 |   0.116 |        725 |
| p     |  0.1  | hold      |   3261 |  -0.162 |   0.113 |        800 |
| p     |  0.1  | hold21_24 |   2517 |  -0.127 |   0.115 |        800 |
| p     |  0.1  | hold25_26 |    744 |  -0.28  |   0.108 |        900 |
| p     |  0.2  | tune      |   2575 |  -0.275 |   0.102 |        800 |
| p     |  0.2  | hold      |   2541 |  -0.141 |   0.106 |        900 |
| p     |  0.2  | hold21_24 |   1945 |  -0.111 |   0.106 |        850 |
| p     |  0.2  | hold25_26 |    596 |  -0.241 |   0.106 |        950 |
| p     |  0.3  | tune      |   1958 |  -0.323 |   0.087 |        850 |
| p     |  0.3  | hold      |   1986 |  -0.159 |   0.101 |        900 |
| p     |  0.3  | hold21_24 |   1483 |  -0.15  |   0.098 |        900 |
| p     |  0.3  | hold25_26 |    503 |  -0.187 |   0.109 |       1000 |
| p     |  0.5  | tune      |   1138 |  -0.279 |   0.083 |        950 |
| p     |  0.5  | hold      |   1201 |  -0.147 |   0.091 |       1100 |
| p     |  0.5  | hold21_24 |    887 |  -0.112 |   0.089 |       1100 |
| p     |  0.5  | hold25_26 |    314 |  -0.246 |   0.096 |       1100 |
| blend |  0    | tune      |      4 |  -0.353 |   0.25  |       1250 |
| blend |  0    | hold      |     21 |   0.41  |   0.19  |        750 |
| blend |  0    | hold21_24 |      1 |  -1     |   0     |        600 |
| blend |  0    | hold25_26 |     20 |   0.48  |   0.2   |        775 |
| blend |  0.05 | tune      |      0 | nan     | nan     |        nan |
| blend |  0.05 | hold      |      1 |  -1     |   0     |        750 |
| blend |  0.05 | hold21_24 |      0 | nan     | nan     |        nan |
| blend |  0.05 | hold25_26 |      1 |  -1     |   0     |        750 |
| blend |  0.1  | tune      |      0 | nan     | nan     |        nan |
| blend |  0.1  | hold      |      0 | nan     | nan     |        nan |
| blend |  0.1  | hold21_24 |      0 | nan     | nan     |        nan |
| blend |  0.1  | hold25_26 |      0 | nan     | nan     |        nan |
| blend |  0.2  | tune      |      0 | nan     | nan     |        nan |
| blend |  0.2  | hold      |      0 | nan     | nan     |        nan |
| blend |  0.2  | hold21_24 |      0 | nan     | nan     |        nan |
| blend |  0.2  | hold25_26 |      0 | nan     | nan     |        nan |
| blend |  0.3  | tune      |      0 | nan     | nan     |        nan |
| blend |  0.3  | hold      |      0 | nan     | nan     |        nan |
| blend |  0.3  | hold21_24 |      0 | nan     | nan     |        nan |
| blend |  0.3  | hold25_26 |      0 | nan     | nan     |        nan |
| blend |  0.5  | tune      |      0 | nan     | nan     |        nan |
| blend |  0.5  | hold      |      0 | nan     | nan     |        nan |
| blend |  0.5  | hold21_24 |      0 | nan     | nan     |        nan |
| blend |  0.5  | hold25_26 |      0 | nan     | nan     |        nan |

## SGP fight legs: joint probabilities from the script (2021+)

`script_joint` = the script's joint probability; `independent_product` = P(winner) x P(method) as if the legs were independent (how a naive parlay is priced); `ll_market` = the market's own priced joint (winner-by-method props), on fights where it exists.

| leg                                 |    n |   actual |   script_joint |   independent_product |   ll_joint |   ll_independent |   ll_market |   ll_joint_same_fights |
|:------------------------------------|-----:|---------:|---------------:|----------------------:|-----------:|-----------------:|------------:|-----------------------:|
| A wins & inside distance            | 2962 |   0.286  |         0.2657 |                0.2658 |     0.5641 |           0.5694 |      0.537  |                 0.5576 |
| A wins & goes distance              | 2962 |   0.2748 |         0.2492 |                0.2491 |     0.5685 |           0.5693 |      0.5729 |                 0.5852 |
| B wins & inside distance            | 2962 |   0.2255 |         0.2487 |                0.2487 |     0.5084 |           0.5128 |      0.4814 |                 0.498  |
| B wins & goes distance              | 2962 |   0.2137 |         0.2363 |                0.2364 |     0.4989 |           0.4983 |      0.4794 |                 0.4974 |
| A wins by KO                        | 2962 |   0.183  |         0.1677 |                0.1698 |     0.4482 |           0.454  |      0.4443 |                 0.452  |
| favourite (ML) wins & goes distance | 2962 |   0.316  |         0.2707 |                0.2723 |     0.6128 |           0.6125 |      0.6308 |                 0.6418 |
| favourite (ML) wins inside distance | 2962 |   0.3258 |         0.2903 |                0.2887 |     0.6102 |           0.6148 |      0.6024 |                 0.6239 |

Round-based SGP legs are in the parquet (`p_A_win_over_1_5`, `p_A_win_under_2_5`, `p_under_1_5`, ...): no round-prop prices exist in the data, so they are evaluated only for calibration: mean P(fight ends under 2.5 rounds) 0.461.

## Verdict

On 2021+ the method market (6-way ll 1.5245) is more accurate than the stand-alone script (1.5893); the 2016-20-chosen blend scores 1.5204. See the betting section for whether disagreement is profitable after the vig; treat any holdout ROI with the bootstrap interval in mind.
