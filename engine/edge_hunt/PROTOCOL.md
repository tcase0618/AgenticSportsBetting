# Edge Hunt Protocol — what counts as a TRUE edge

An edge is real only if ALL of these hold. No exceptions, no "promising" upgrades.

1. **Real prices.** Settled at a price you could actually have bet (closing line, or a timestamped quote).
   No assumed -110 where real juice exists; no proxy lines; no fair-price parlays.
2. **Found blind.** Rule/model/threshold chosen ONLY on discovery data (≤ 2022 unless the data starts later;
   then the first 60% of dates). Every knob you tried is logged in your candidates file (count = n_tests).
3. **Lockbox.** Evaluate on the lockbox (2023+ / last 40% of dates) ONCE per final candidate. If you change
   the rule after looking, the lockbox is burned for that rule — say so.
4. **Significance after multiple testing.** Lockbox ROI > 0 with one-sided p < 0.05 / n_final_candidates,
   OR Benjamini-Hochberg q < 0.10 across all your lockbox tests. Report z, n, ROI, units, max drawdown.
5. **Stable.** Positive in ≥ 60% of seasons (or monthly blocks) in the full sample; not driven by 1 season.
6. **Mechanism.** One sentence on WHY the market is wrong here (who's on the other side).
7. **Live-able.** It must be detectable before the bet with data available live (ESPN, FanDuel sbapi,
   Polymarket Gamma/CLOB, Open-Meteo, NHL/MLB APIs, nflreadpy). Write a `scan()` that finds today's instances.

Pure arbitrage (guaranteed profit across books) and middles are exempt from 2–5 but must be verified on
live quotes with fees (Polymarket taker fee: fee/share = p·rate·(p(1−p))^exp from feeSchedule).

Output per hunter: `results/edge_hunt/<hunter>.md` (verdicts table: candidate | discovery n/ROI | lockbox
n/ROI/z/p | seasons positive | verdict TRUE/FALSE) and `results/edge_hunt/<hunter>_candidates.csv` (every test
tried). Code in `engine/edge_hunt/<hunter>.py`. Don't git commit. Be honest: "no edge found" is a valid result.

Existing assets: data/processed/* (games with open/close lines, props, UFC), results/ledger.csv (validated
strategies), results/scripts/* (game & player scripts), results/predict/* (walk-forward predictors),
results/props_script/* (calibrated prop probabilities), engine/devig.py, nightly/providers.py (FanDuel +
Polymarket live boards), backtest/core.py (perf, settle, american_to_decimal).
