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
|     2016 |      446 |       1.61   |       1.5427 |      1.5398 |       0.6628 |          0.6309 |      0.6306 |        0.6676 |        0.6483 |       0.6483 |
|     2017 |      405 |       1.605  |       1.5275 |      1.5245 |       0.6586 |          0.6037 |      0.6039 |        0.6704 |        0.6609 |       0.659  |
|     2018 |      429 |       1.5802 |       1.4989 |      1.4968 |       0.6491 |          0.5948 |      0.5954 |        0.6537 |        0.6452 |       0.6433 |
|     2019 |      459 |       1.567  |       1.4889 |      1.4884 |       0.6599 |          0.6308 |      0.6306 |        0.6646 |        0.6418 |       0.6419 |
|     2020 |      419 |       1.599  |       1.5266 |      1.5236 |       0.6494 |          0.6233 |      0.623  |        0.66   |        0.6511 |       0.6492 |
|     2021 |      450 |       1.5915 |       1.5463 |      1.5415 |       0.6564 |          0.6268 |      0.6269 |        0.6794 |        0.6742 |       0.6724 |
|     2022 |      435 |       1.6186 |       1.5254 |      1.5266 |       0.6362 |          0.5922 |      0.5926 |        0.6718 |        0.6597 |       0.6589 |
|     2023 |      368 |       1.6094 |       1.5618 |      1.5554 |       0.6266 |          0.5776 |      0.5784 |        0.6608 |        0.6446 |       0.6431 |
|     2024 |      329 |       1.5414 |       1.471  |      1.4659 |       0.6327 |          0.5915 |      0.5916 |        0.6614 |        0.6636 |       0.6583 |
|     2025 |      309 |       1.555  |       1.5243 |      1.5135 |       0.6327 |          0.6149 |      0.6134 |        0.6668 |        0.6574 |       0.6542 |
|     2026 |       60 |       1.4784 |       1.4212 |      1.4143 |       0.575  |          0.4651 |      0.4684 |        0.7098 |        0.7766 |       0.7601 |

**2016-2020 (tuning)** (2,158 fights): 6-way winner x method log loss script 1.5918 / market 1.5166 / blend 1.5143;  3-way method (KO/SUB/DEC) script 0.9567 / market 0.9385 / blend 0.9357;  winner script 0.6560 / ML 0.6171 / blend 0.6171

**2021-2026 holdout** (1,951 fights): 6-way winner x method log loss script 1.5832 / market 1.5245 / blend 1.5197;  3-way method (KO/SUB/DEC) script 0.9633 / market 0.9513 / blend 0.9473;  winner script 0.6360 / ML 0.5970 / blend 0.5971

Blend weights chosen on 2016-2020: method log-linear pool w_market = **0.85**, winner w_market = **0.95**.

Holdout calibration-in-the-large (2021+):

| outcome   |   mean_script |   mean_market |   actual |
|:----------|--------------:|--------------:|---------:|
| A_KO      |         0.176 |         0.198 |    0.183 |
| A_SUB     |         0.106 |         0.107 |    0.103 |
| A_DEC     |         0.266 |         0.239 |    0.275 |
| B_KO      |         0.154 |         0.165 |    0.152 |
| B_SUB     |         0.079 |         0.086 |    0.074 |
| B_DEC     |         0.219 |         0.205 |    0.214 |
| distance  |         0.486 |         0.444 |    0.489 |

Finish-round distribution given a finish (2021+, 1,515 finishes): script log loss 1.0404 vs static 2016-20 base rates 1.1253; expected finish round MAE 0.684.

## Ablation (method learner re-run without each feature group)

ll6 = 6-way winner x method log loss; `hold_d_ll6_vs_all` > 0 means dropping the group HURT the 2021+ holdout (the group helps).

| variant                                 |   tune_ll6 |   hold_ll6 |   hold_ll_win |   hold_d_ll6_vs_all |
|:----------------------------------------|-----------:|-----------:|--------------:|--------------------:|
| all features                            |     1.5921 |     1.5922 |        0.6398 |              0      |
| drop skill rating                       |     1.594  |     1.5947 |        0.6423 |              0.0025 |
| drop finish tendencies (power/chin/sub) |     1.6111 |     1.6112 |        0.6418 |              0.019  |
| drop striking                           |     1.6024 |     1.5998 |        0.6448 |              0.0075 |
| drop grappling                          |     1.5992 |     1.604  |        0.6431 |              0.0117 |
| drop physical (age/reach/height)        |     1.6033 |     1.6076 |        0.6536 |              0.0154 |
| drop layoff/experience                  |     1.5927 |     1.5926 |        0.64   |              0.0003 |
| drop class base rates & 5 rounds        |     1.6017 |     1.5962 |        0.6398 |              0.0039 |
| base rates only (intercepts)            |     1.7156 |     1.7169 |        0.6931 |              0.1247 |

Final learned weights (standardised features) per method:

|              |     KO |    SUB |    DEC |
|:-------------|-------:|-------:|-------:|
| intercept    | -1.097 | -1.592 | -0.896 |
| skill        |  0.054 |  0.096 |  0.076 |
| ko_win       |  0.16  | -0.044 |  0.009 |
| opp_ko_loss  |  0.099 | -0.023 | -0.091 |
| sub_win      | -0.038 |  0.176 | -0.043 |
| opp_sub_loss |  0.018 |  0.21  | -0.036 |
| dec_rate     | -0.021 | -0.068 |  0.04  |
| opp_dec_rate | -0.072 | -0.081 |  0.069 |
| own_ko_loss  | -0.049 | -0.057 | -0.077 |
| slpm         |  0.028 | -0.067 |  0.001 |
| opp_sapm     |  0.064 |  0.063 |  0.049 |
| strike_diff  |  0.065 | -0.042 |  0.147 |
| sig_acc      |  0.022 |  0.072 | -0.017 |
| kd_rate      |  0.071 | -0.075 | -0.048 |
| td_eff       | -0.028 |  0.095 |  0.049 |
| sub_att      | -0.043 |  0.172 | -0.083 |
| ctrl_diff    | -0.058 |  0.064 |  0.052 |
| opp_td_eff   | -0.099 |  0     | -0.117 |
| age_diff     | -0.17  | -0.141 | -0.103 |
| opp_age      |  0.045 |  0.007 |  0.014 |
| reach_diff   |  0.049 |  0.057 | -0.012 |
| height_diff  |  0.019 |  0.065 |  0.036 |
| layoff       | -0.064 | -0.027 | -0.058 |
| exp          | -0.019 |  0.041 |  0.058 |
| opp_exp      |  0.012 | -0.03  |  0.028 |
| class_ko     |  0.131 | -0.051 | -0.1   |
| class_sub    |  0.018 |  0.022 | -0.039 |
| five_rd      |  0.035 |  0.009 | -0.062 |
| female       | -0.075 | -0.016 |  0.009 |
| title        | -0.022 |  0.001 |  0.002 |
| corner       |  0.072 |  0.108 |  0.089 |

## What the script got wrong most (2021+)

| segment            |   fights |   ll6_script |   ll6_market |     gap |
|:-------------------|---------:|-------------:|-------------:|--------:|
| all                |     1951 |       1.5832 |       1.5245 |  0.0587 |
| 5-round            |      184 |       1.572  |       1.5361 |  0.0359 |
| women              |      315 |       1.5381 |       1.4784 |  0.0596 |
| heavyweight        |      138 |       1.544  |       1.4667 |  0.0774 |
| flyweight/bantam   |      348 |       1.5333 |       1.5041 |  0.0292 |
| A or B UFC debut   |      323 |       1.6496 |       1.5361 |  0.1135 |
| both 5+ UFC fights |      781 |       1.5344 |       1.4854 |  0.049  |
| result KO          |      635 |       1.6835 |       1.5351 |  0.1484 |
| result SUB         |      327 |       2.1418 |       1.9423 |  0.1996 |
| result DEC         |      989 |       1.3341 |       1.3796 | -0.0455 |
| underdog won (ML)  |      606 |       1.7788 |       1.9543 | -0.1755 |

Gap decomposition 2021+: winner part +0.0423, method-given-winner part +0.0163 nats/fight (positive = market better).

## Betting method props where the script disagrees with the market

Rule chosen on 2016-2020 (max ROI, >=100 bets): source=`p`, EV threshold tau=0.05: tune 3751 bets ROI -23.2%; **holdout 2021+ 3658 bets ROI -14.6%** (hit 12.5%, median odds +800); of which 2021-2024 2819 bets ROI -15.1%, 2025-26 (suspect odds even after the gate) 839 bets ROI -12.9%. Flat 1u; NC/DQ/overturned void, draws lose.

2021-2024 holdout ROI bootstrap 90% interval: [-23.9%, -7.2%]. By leg:

| leg   |   bets |    roi |   hit |
|:------|-------:|-------:|------:|
| A_DEC |    631 |  0.179 | 0.225 |
| A_KO  |    420 | -0.015 | 0.11  |
| A_SUB |    456 | -0.418 | 0.055 |
| B_DEC |    551 | -0.239 | 0.131 |
| B_KO  |    446 | -0.403 | 0.078 |
| B_SUB |    315 | -0.098 | 0.073 |

Full grid:

| src   |   tau | split     |   bets |     roi |     hit |   avg_odds |
|:------|------:|:----------|-------:|--------:|--------:|-----------:|
| p     |  0    | tune      |   4256 |  -0.238 |   0.127 |      674   |
| p     |  0    | hold      |   4097 |  -0.158 |   0.128 |      750   |
| p     |  0    | hold21_24 |   3173 |  -0.162 |   0.125 |      700   |
| p     |  0    | hold25_26 |    924 |  -0.147 |   0.137 |      800   |
| p     |  0.05 | tune      |   3751 |  -0.232 |   0.124 |      700   |
| p     |  0.05 | hold      |   3658 |  -0.146 |   0.125 |      800   |
| p     |  0.05 | hold21_24 |   2819 |  -0.151 |   0.122 |      750   |
| p     |  0.05 | hold25_26 |    839 |  -0.129 |   0.135 |      850   |
| p     |  0.1  | tune      |   3307 |  -0.237 |   0.117 |      711   |
| p     |  0.1  | hold      |   3240 |  -0.149 |   0.118 |      800   |
| p     |  0.1  | hold21_24 |   2489 |  -0.15  |   0.114 |      800   |
| p     |  0.1  | hold25_26 |    751 |  -0.148 |   0.13  |      850   |
| p     |  0.2  | tune      |   2514 |  -0.258 |   0.104 |      800   |
| p     |  0.2  | hold      |   2499 |  -0.134 |   0.109 |      900   |
| p     |  0.2  | hold21_24 |   1894 |  -0.113 |   0.108 |      850   |
| p     |  0.2  | hold25_26 |    605 |  -0.201 |   0.112 |      900   |
| p     |  0.3  | tune      |   1903 |  -0.263 |   0.096 |      845   |
| p     |  0.3  | hold      |   1968 |  -0.099 |   0.106 |      900   |
| p     |  0.3  | hold21_24 |   1484 |  -0.07  |   0.106 |      900   |
| p     |  0.3  | hold25_26 |    484 |  -0.187 |   0.107 |      975   |
| p     |  0.5  | tune      |   1072 |  -0.286 |   0.081 |      943.5 |
| p     |  0.5  | hold      |   1153 |  -0.095 |   0.095 |     1000   |
| p     |  0.5  | hold21_24 |    831 |  -0.043 |   0.095 |     1000   |
| p     |  0.5  | hold25_26 |    322 |  -0.23  |   0.096 |     1100   |
| blend |  0    | tune      |      2 |  -1     |   0     |      650   |
| blend |  0    | hold      |     25 |   0.184 |   0.16  |      750   |
| blend |  0    | hold21_24 |      2 |  -1     |   0     |      490   |
| blend |  0    | hold25_26 |     23 |   0.287 |   0.174 |      750   |
| blend |  0.05 | tune      |      0 | nan     | nan     |      nan   |
| blend |  0.05 | hold      |      4 |  -1     |   0     |      925   |
| blend |  0.05 | hold21_24 |      0 | nan     | nan     |      nan   |
| blend |  0.05 | hold25_26 |      4 |  -1     |   0     |      925   |
| blend |  0.1  | tune      |      0 | nan     | nan     |      nan   |
| blend |  0.1  | hold      |      0 | nan     | nan     |      nan   |
| blend |  0.1  | hold21_24 |      0 | nan     | nan     |      nan   |
| blend |  0.1  | hold25_26 |      0 | nan     | nan     |      nan   |
| blend |  0.2  | tune      |      0 | nan     | nan     |      nan   |
| blend |  0.2  | hold      |      0 | nan     | nan     |      nan   |
| blend |  0.2  | hold21_24 |      0 | nan     | nan     |      nan   |
| blend |  0.2  | hold25_26 |      0 | nan     | nan     |      nan   |
| blend |  0.3  | tune      |      0 | nan     | nan     |      nan   |
| blend |  0.3  | hold      |      0 | nan     | nan     |      nan   |
| blend |  0.3  | hold21_24 |      0 | nan     | nan     |      nan   |
| blend |  0.3  | hold25_26 |      0 | nan     | nan     |      nan   |
| blend |  0.5  | tune      |      0 | nan     | nan     |      nan   |
| blend |  0.5  | hold      |      0 | nan     | nan     |      nan   |
| blend |  0.5  | hold21_24 |      0 | nan     | nan     |      nan   |
| blend |  0.5  | hold25_26 |      0 | nan     | nan     |      nan   |

## SGP fight legs: joint probabilities from the script (2021+)

`script_joint` = the script's joint probability; `independent_product` = P(winner) x P(method) as if the legs were independent (how a naive parlay is priced); `ll_market` = the market's own priced joint (winner-by-method props), on fights where it exists.

| leg                                 |    n |   actual |   script_joint |   independent_product |   ll_joint |   ll_independent |   ll_market |   ll_joint_same_fights |
|:------------------------------------|-----:|---------:|---------------:|----------------------:|-----------:|-----------------:|------------:|-----------------------:|
| A wins & inside distance            | 2962 |   0.286  |         0.2822 |                0.2828 |     0.5633 |           0.568  |      0.537  |                 0.5568 |
| A wins & goes distance              | 2962 |   0.2748 |         0.2663 |                0.2657 |     0.5662 |           0.5669 |      0.5729 |                 0.5823 |
| B wins & inside distance            | 2962 |   0.2255 |         0.2321 |                0.2315 |     0.506  |           0.5106 |      0.4814 |                 0.4952 |
| B wins & goes distance              | 2962 |   0.2137 |         0.2194 |                0.22   |     0.497  |           0.4966 |      0.4794 |                 0.4957 |
| A wins by KO                        | 2962 |   0.183  |         0.1763 |                0.1805 |     0.4476 |           0.4528 |      0.4443 |                 0.4511 |
| favourite (ML) wins & goes distance | 2962 |   0.316  |         0.272  |                0.2735 |     0.6108 |           0.6105 |      0.6308 |                 0.64   |
| favourite (ML) wins inside distance | 2962 |   0.3258 |         0.2914 |                0.2899 |     0.609  |           0.6133 |      0.6024 |                 0.6223 |

Round-based SGP legs are in the parquet (`p_A_win_over_1_5`, `p_A_win_under_2_5`, `p_under_1_5`, ...): no round-prop prices exist in the data, so they are evaluated only for calibration: mean P(fight ends under 2.5 rounds) 0.461.

## Verdict

On 2021+ the method market (6-way ll 1.5245) is more accurate than the stand-alone script (1.5832); the 2016-20-chosen blend scores 1.5197. See the betting section for whether disagreement is profitable after the vig; treat any holdout ROI with the bootstrap interval in mind.
