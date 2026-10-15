# AgenticSportsBetting: Seven-Sport Strategy Backtest

A leak-free, out-of-sample backtest that searches thousands of betting systems across
**NFL, NCAAF, NBA, NCAAM, MLB, NHL and UFC**, settles every bet at the real closing
price, and keeps only systems that survive validation on seasons they never saw.

**→ The seven chosen strategies with full stats: [`results/KEYS.md`](results/KEYS.md)**

## Data (≈280,000 games and fights, all real closing lines)

| Sport | Games | Seasons with lines | Source (all via GitHub mirrors) |
|---|---|---|---|
| NFL | 7,355 | 1999–2026 | nflverse `games.csv` (spread/total/ML + juice, weather, rest) |
| NCAAF | 16,263 | 2006–2025 | cfbfastR-data betting lines (median of books), cross-checked vs SBR (98% within 1 pt) |
| NBA | 24,440 | 2012–2026 (ML 2008+) | SBR archive, kyleskom OddsData, ESPN PickCenter |
| NCAAM | 105,286 | 2013–2023, 2026 | ESPN PickCenter (TeamRankings closing lines) |
| MLB | 37,960 | 2010–2025 | SBR via Retrosheet-matched repo; DraftKings 2021–25; starting pitchers |
| NHL | 24,542 | 2008–2026 | SBR archive + Odds-API snapshots; starting goalies |
| UFC | 7,062 w/ odds | 2010–Mar 2026 | ufcstats scrape + Ultimate UFC Dataset (BestFightOdds) |

ESPN/Google/Kaggle were blocked from the build container, so every dataset comes from public GitHub
mirrors of those sources. `scripts/data/*_build.py` rebuilds each file from its raw downloads.

## Method

1. **Leak-free features** (`backtest/features.py`): every rolling stat (streaks, ATS form, rest,
   fatigue, Elo, starter form) is shifted so a game only sees prior games.
2. **Strategy grid** (`backtest/strategies.py`, `run_ufc.py`): team condition × role (fav/dog/price
   bucket/home/away) × bet on/against × phase, plus totals filters and their pairwise combos. That's
   ~2,000–9,000 candidates per sport, plus walk-forward logistic models refit each season.
3. **Settlement**: actual closing American odds; −110 is assumed only when juice is missing (never for
   MLB run lines or NHL puck lines, whose prices vary too much).
4. **Discovery → holdout** (`backtest/evaluate.py`): systems must be profitable with p<0.05 on the
   first ~62% of seasons, then they're judged on the untouched later seasons. Benjamini-Hochberg q-values
   control false discoveries.
5. **Key selection** (`backtest/keys.py`): among survivors, prefer out-of-sample strength, sample size,
   season-to-season consistency and a real causal mechanism over raw ROI.

## Run it

```bash
pip install pandas numpy scikit-learn scipy openpyxl
python -m backtest.run_team nfl      # nfl | ncaaf | nba | ncaam | mlb | nhl
python -m backtest.run_ufc
python -m backtest.report            # rebuilds results/KEYS_body.md + keys.json
python scripts/inspect_key.py nfl 'tot|wind15|game|under|all'
```

Outputs: `results/<sport>_strategies.csv` (every evaluated system with train/holdout stats),
`results/<sport>_meta.json` (baseline ROI per market side), `results/keys.json`.

## Honest limitations

- Backtested ROI ≠ future ROI. Edges decay once they're known, and closing lines are hard to get in size.
  Expect live results below these numbers. Flat-stake 1u (1–2% of bankroll); never chase.
- NFL wind uses recorded game-time wind. Bet it off the forecast (usually reliable 24h out).
- UFC odds are "near-closing" (BestFightOdds). Height and reach are current profile values; age is exact.
- MLB 2021+ and NBA 2023–25 lines are scraper snapshots of unconfirmed timing. MLB line-movement
  systems failed an integrity check and were excluded.
- NCAAM has no lines for 2024–25; NHL 2022-23 is only 24% covered.
- Sports betting is legal only where you are and only at the legal age (21 in most US states).
