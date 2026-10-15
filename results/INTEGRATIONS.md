# GitHub Integrations: What Got Added and What the Data Said

Every idea went through the same gate as the strategies: choose on discovery data, judge on unseen holdout data.

| Source repo | What we built | Result | In the engine? |
|---|---|---|---|
| martineastwood/penaltyblog | `engine/devig.py`: multiplicative / additive (Shin) / power / odds-ratio vig removal | Multiplicative (old) overrated +250-or-longer dogs: 17.5% predicted vs 15.8% actual. **Odds-ratio** won on discovery and holdout (log loss 0.59283 vs 0.59340) | **Yes**: all live fair prices |
| nflverse/nflreadpy | `nightly/injuries.py`: weekly injury report on the card | A starting-QB change did **not** hurt the NFL strategies (wind unders +12.4% with a change vs +9.5% without) | **Yes**, as context only, never a veto |
| jldbc/pybaseball | `scripts/data/mlb_statcast_build.py` (62,361 pitcher-games 2024–26) + `engine/mlb_k_model.py` (negative-binomial K projection) | Market beats the model (log loss 0.683 vs 0.701); the discovery pick +4.4% went **−10.5%** on holdout | **No**: K props stay off the card |
| vivekjoshy/openskill.py | `engine/ufc_openskill.py`: fighter ratings with uncertainty | Alone it loses badly (disagreement bets −11% to −19%). As a **confirmation** of Youth Over Experience: **+17.8% / +21.2%** when it agrees vs +5% when it disagrees | **Yes**: confirmed youth legs get priority (🔥) |
| (idea from the above) | `engine/confirmation_test.py`: Elo agreement for every strategy | Only **NHL Bounce-Back** improves in both periods (agree +7.1% / +8.0% vs disagree +5.5% / +0.2%) | **Yes**: unconfirmed NHL legs are no longer bet |
| nfelo (concept) | — | Its own numbers only match the Vegas close; our NFL and NBA rating models lost to the close | No |

**Pattern:** prediction models on their own don't beat closing lines; the market already knows what they know.
Their value is as a **second opinion on spots the market misprices**, where agreement separates the strongest bets
from the rest.
