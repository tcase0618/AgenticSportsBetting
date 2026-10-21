# Hunter: arb_scanner: live cross-venue arbs, middles and +EV, plus a backtest of Polymarket vs the sportsbook close

Code: `engine/edge_hunt/arb_scanner.py` · Run window: 2026-10-17 23:47Z to 2026-10-18 00:26Z (8 scans, about 5.5 min apart)

```
python -m engine.edge_hunt.arb_scanner scan                     # one scan -> arb_log.csv, arb_scans.csv, flagged_quotes.csv, alerts.jsonl
python -m engine.edge_hunt.arb_scanner scan light               # same, skipping the Kalshi player-prop ladders
python -m engine.edge_hunt.arb_scanner loop 6 300               # 6 scans, 300 s apart
python -m engine.edge_hunt.arb_scanner watch --minutes 120 --every 75 [--full-every 4]   # continuous + alerts
python -m engine.edge_hunt.arb_scanner log-fill --venue Kalshi --price +118 --stake 50 [--filled 40] \
        [--alert-key <key or prefix from alerts.jsonl>] [--scanned +122] [--event ..] [--market ..] [--selection ..] [--line ..] [--note ..]
python -m engine.edge_hunt.arb_scanner settle                   # grade logged +EV signals from ESPN finals -> arb_settled.csv
python -m engine.edge_hunt.arb_scanner stale-backtest 20261019  # replay saved quote snapshots through the stale filter
python -m engine.edge_hunt.arb_scanner pmhist [refresh]         # Polymarket-history backtest
```
Placeable venues default to `Kalshi,Polymarket,FanDuel`; override with `ARB_PLACEABLE="Kalshi,Polymarket"`.
**Note:** Tennessee sportsbooks, FanDuel included, require bettors to be 21. A 19-year-old cannot legally hold a TN
FanDuel account, so for this bettor FanDuel alerts are not actionable. Running with
`ARB_PLACEABLE=Kalshi,Polymarket` is the honest setting. Check the exchanges' own age and state terms too.

## v2 (2026-10-21): stale filter, sharp fair price, watch/alerts, ladder-consistency finder, fill log

New files:
- `engine/edge_hunt/quote_history.py`
- `engine/edge_hunt/consistency.py`
- `results/edge_hunt/quote_history.sqlite` (48h)
- `flagged_quotes.csv`, `alerts.jsonl` (+ `alerts_state.json`), `fills.csv`

`arb_log.csv` gained these columns: `source`, `check`, `flag`, `note`, `mid_ev`, `p_window`, `w_cons`.

**Feeds added**
- Pinnacle team totals and 1H spread/total (basketball and football).
- Action Network team totals and `line_status`.
- Polymarket 1H spreads and totals, plus `POST /books` depth on every Polymarket quote.
- Kalshi 1H spread/total and team totals (`KXNBA1HSPREAD`, `KXNBA1HTOTAL`, `KXNBATEAMTOTAL`).
- Kalshi NBA prop ladders (PTS/REB/AST/3PT) and NHL prop ladders (GOAL/AST/PTS).
- Kalshi depth on both sides: YES uses ask size; NO uses YES-bid size.

**1. Stale and suspect quotes.** Every scan records each (venue, event, market, line, side) price and the
cross-venue median for each outcome. A flagged quote stays in the quote table and in `flagged_quotes.csv`, but it
is excluded from arbs, +EV and alerts. Any signal it would have produced is logged with `flag` set.

STALE fires on any of three conditions:
- **Per-outcome freeze.** A non-exchange price has not changed for at least 60 min while the median for that
  outcome moved at least 1.5 pts.
- **Frozen aggregator feed.** An Action Network or ESPN-DraftKings book has not changed any main ml/spread/total quote
  on the event for at least 60 min, while at least 4 other sportsbooks (and at least 2/3 of them) moved.
- **Feed age.** Action Network reports `line_status=opener`. Action Network exposes no per-quote timestamp, so this
  flag is the only feed-age signal available.

SUSPECT fires when a quote is more than 4 pts better than every other venue. It counts only on Kalshi or
Polymarket, where the order book is re-checked.

The event-level off-market filter is also tighter for Action Network books: 4.5 pts of de-vigged ML, or 2 points of
line. It now also covers team totals and 1H lines. It skips puck lines and run lines, which are always ±1.5.

**2. Fair price.**
- Each venue is de-vigged with `odds_ratio`. Two-sided quotes with an overround above 1.15 are dropped; these are
  empty exchange books with asks around 0.98/0.99.
- The fair price is a log-odds weighted mean. Weights: Pinnacle 3; Kalshi and Polymarket 1.5 when both sides have
  at least $200 at the quoted price, otherwise 0.5; BetRivers, FanDuel, DraftKings and Circa 1; others 0.5.
- +EV means the venue's price is at least 2% better than the fair price of the *other* venues. Those venues need
  Pinnacle among them, or at least 2 venues with weight of at least 2.

**3. Watch and alerts.**
- `watch` rescans every `--every` seconds, measured start to start. It fetches the Kalshi prop ladders and saves a
  snapshot on every 4th pass only.
- Light passes do not record or use prop medians, so a missing Kalshi rung cannot fake a median move.
- Opportunities whose legs are all on PLACEABLE and not flagged are appended to `alerts.jsonl`:
  - arb: book-verified margin above 0
  - middle: EV of at least 1% against the fair price, window included
  - +EV: book-verified edge of at least 2%
- Each alert carries a ready-to-send `message`: event and start time ET, each leg (venue, side, all-in American
  price after re-pricing from the book), the stake split for $100, max size (exchange top-of-book or Pinnacle limit),
  and the time first seen.
- An opportunity alerts once. It re-alerts only if its edge grows by at least 1 pt (state in `alerts_state.json`).

**4. Consistency finder (`consistency.py`).** Every quote is mapped onto a threshold `c` on one integer outcome:
- Spread family: the home margin. ML is a rung of this ladder: home ML is the same bet as home -0.5, and NFL ML is
  the same as home pick'em.
- Totals family: the game total, 1H total, a team's total, or a player's stat.

The checks:
- **Pairs.** Up at c_u plus down at c_d, with c_u ≤ c_d, always has at least one winning leg. Cost below 1 is a
  guaranteed arb, with a middle bonus. Cost up to 1.03 with an integer inside the window is a middle, with
  EV = (p_up + p_down)/cost − 1 using the fair price at each leg's own line. Same-venue pairs are included.
- **Monotonicity.**
  - Per-venue de-vigged ladders: violations above 2 pts are reported, for rungs with an overround of 1.08 or less.
  - The fair ladder: violations above 3 pts are reported, for rungs with at least 1.5 of venue weight.
  - Together these cover (a) alt spreads, (b) alt totals and (d) P(win) between P(−1.5) and P(+1.5).
- **(e) Team totals vs game total.** Compares the sum of the two team-total medians with the game-total median.
  Single lines are shifted to their 50/50 point with a per-sport slope; ladders are interpolated. A gap of at least
  1.5 is reported.
- Output kinds are `arb`, `middle` or `violation` (informational), tagged `check` = a/b/c/d/e.

**5. Fill log.** `fills.csv` has these columns: logged_at, alert_key, kind, sport, event, market, selection, line,
venue, scanned_american, realized_american, stake_usd, filled_usd, slippage_prob_pts, note. With `--alert-key`,
`log-fill` copies the event, leg and scanned price from `alerts.jsonl`. Slippage is measured in implied-probability
points (positive means a worse fill than scanned).

**Runtime.** About 23–32 s per scan (fetch 8–17 s). The first v2 build took 200 s+ because pandas deep-copies
`DataFrame.attrs` on every row operation. The fix was to stop using attrs and switch the hot loops to plain dicts.

### Backtest of the stale filter on Monday 10/19 snapshots (3 scans: 21:31, 22:51, 23:51Z)

This replays `arb_quotes/quotes_20261019*` through the new filters (`stale-backtest`, plus a replay of the signals).
Results:

| Monday case | What the data shows | Caught by |
|---|---|---|
| HardRock Celtics +105 / +1.5 −105 (ML 1–1.6% "arbs", +5–6.5% "+EV") | HardRock had the **sides swapped**: Celtics +105 / Pistons −125 and Celtics +1.5, while every other venue had the Celtics −112 to −120 and −1 to −1.5. Nobody else moved either (the game was the next day), so a "frozen while others moved" rule cannot catch it | the tighter AN off-market filter (spread −1.5 vs median +1.25), from scan 1; the HardRock +1.5 quote is also SUSPECT (4.5 pts better than the next venue) |
| theScore Rays −1.5 +165 (0.3–1.2% arbs, +3.7% EV) | unchanged for 140 min. At 80 min only 7/12 venues had moved (not enough to flag); at 140 min 9/10 had | the frozen-feed rule at the 23:51 scan. The single-outcome rule alone never fires: the −1.5 median moved only about 0.5 pts |

Signals removed across the replay, all on HardRock, theScore or HardRock-NHL:
- 21:31: 6 of 7 arbs and 2 of 4 +EV
- 22:51: 10 of 11 arbs and 3 of 4 +EV
- 23:51: 11 of 11 arbs and 4 of 5 +EV

The first scan of any session has no history, so only the off-market and outlier rules work in it. This is why
`quote_history.sqlite` persists across runs.

### Live test 2026-10-21

Runs: scans at 12:37Z, 13:04Z and 13:31Z; an 8-pass `watch` from 13:15 to 13:25Z (2 full passes, 6 light). The slate
was 11 NBA, NHL and NCAAF/NFL lookahead games, with 32 events in all. There were no MLB games today.

**Stale filter.**
- Before 60 min of history: 14 flags per scan, all `AN line_status=opener` on theScore NHL and NBA lines and team
  totals.
- After 60 min (13:21Z onward): 19 flags, then 39 at 13:31Z. Detail:
  - **frozen feed:** DraftKings on Pacers@Pelicans and BallyBet on Warriors@Lakers, 71 min with 7/10 books moving
  - **per-outcome freeze:** FanDuel Bulls +9.5 −105 (median −2.8 pts) and FanDuel Jazz/Grizzlies O242.5,
    Caesars/BallyBet team totals and spreads (1.6–2.3 pts)
- The off-market filter removed:
  - HardRock NFL Patriots@Bears: team totals posted at 0.5, which produced fake 4–58% "team-total ladder arbs" before
    the filter was extended
  - Caesars Nuggets@Thunder (total 232 vs 236)
  - Fanatics Hornets@Nets (spread on the wrong side of pick'em)
- No arb or +EV signal was lost to a stale flag today: `flag` was empty for every row.
- Caveat: FanDuel is a direct feed, not an aggregator. A frozen FanDuel price may be live and stale *in the bettor's
  favour*. The spec excludes it anyway; recheck by hand if FanDuel shows in `flagged_quotes.csv`.

**Consistency findings.** There were no guaranteed cross-rung arbs and no monotonicity (a/b/d) violations above the
tolerances in the final code. Findings:
- **Middles with positive EV (cost 1.01–1.03).** Examples:
  - Hornets@Nets: Nets +1.5 −104 Polymarket with Hornets +1 −105 Bovada (EV +1.5%, window P≈3.8%)
  - ML vs spread (d): Nets ML +104 Polymarket with Hornets +1.5 −115 DraftKings/Fanatics (+0.5%); Grizzlies ML −108
    Polymarket with Jazz +1.5 +100/+102 (+0.0–0.5%)
  - Polymarket rungs vs book lines (c): Heat/Wolves O240.5 +104 Polymarket with U241.5 −110 Fanatics (+1.0%)
- **(e) team totals vs total.** The consensus gap was ≤0.9 everywhere. The per-venue outliers were theScore (+1.5 to
  +1.9) and a few books at ±1.5.
- Early versions flagged Kalshi prop ladders as non-monotone. This was de-vig noise on empty books, now filtered by
  overround.

**Placeable alerts (5, all +EV; no placeable arb or middle cleared its threshold).** None has been placed or graded
yet. All are in `alerts.jsonl`:

| First seen (ET) | Event | Bet | Edge vs fair (book-verified) | Max $ at price |
|---|---|---|---|---|
| 9:04 | NHL Flyers@Sabres | Flyers ML +175 Polymarket | +2.0% | $264 |
| 9:04 | NBA Hornets@Nets | OVER 221.5 +117 Polymarket | +2.2% | $853 |
| 9:04 | NBA Hornets@Nets | Brandon Miller O24.5 pts +101 Kalshi (fair = Pinnacle only) | +2.3% | $1,097 |
| 9:15 | NBA Hornets@Nets | Nets −1.5 +122 Polymarket | +3.1% | $120 |
| 9:31 | NBA Pacers@Pelicans | Pelicans +2.5 +113 Polymarket | +2.6% | $7,805 |

The arbs found today (Polymarket vs BallyBet/DraftKings/Fanatics/Pinnacle/Caesars, 0.1–1.5%; Kalshi vs theScore
team total 4.1%) all have a non-placeable leg. The Kalshi/theScore team total is a theScore line that was steady
all morning, so it is probably stale but not yet flagged.

## Venues (reachability tested from this sandbox, not assumed)

| Venue | Source | Status | Markets used |
|---|---|---|---|
| Pinnacle | guest.api.arcadia.pinnacle.com (no key) | OK (one transient 401 on NHL) | ML, main + alt spreads/totals, NFL player props, max-risk limits |
| FanDuel | sbapi.tn (existing `fanduel_board`) | OK | main ML/spread/total, UFC |
| BetRivers | Kambi `rsiuspa` listView | OK | main ML/spread/total |
| Bovada | bovada.lv coupon v2 | OK | main + alt lines, UFC |
| DraftKings, BetMGM, bet365, Caesars, Fanatics, HardRock, theScore, Bally Bet | Action Network `web/v2/scoreboard` | OK | main ML/spread/total (an aggregator, so it can lag the book) |
| Polymarket | Gamma (all lines) + CLOB `/book` | OK | ML/spreads/totals at best ask + feeSchedule taker fee; legs re-priced from the book at $100. Player-prop books are too thin (spread above 5c), so they are skipped |
| Kalshi | trade-api/v2 (GAME/SPREAD/TOTAL + 7 NFL prop ladders) | OK | yes/no ask + taker fee ceil(0.07·C·P(1−P)); legs re-priced from the orderbook |
| ESPN | scoreboard / summary | OK | master event list, DraftKings fallback, finals and box scores for settlement |
| SX.bet | api.sx.bet | markets OK, order book returns "OrderBook V2 no longer supported" | not used |
| DraftKings direct, BetMGM CDS, Caesars, BetOnline, HardRock, Fanatics, MyBookie | direct APIs | 403 | not used |
| ProphetX, Novig, BetDEX | | unreachable | not used |
| The Odds API | | 401 (no key) | not used |

Matching works like this:
- The ESPN events for today and tomorrow (ET) are the master list.
- Each venue's teams are resolved to ESPN events by alias: exact first, then a fuzzy match that must be unique. Matching uses start time within 4h, or the ET date for Kalshi.
- Sides are assigned by team identity, never by listing order.
- Lines are compared only on the exact same home-perspective line.
- A stale-feed filter drops any (venue, event) whose de-vigged ML is more than 8 points off the cross-venue median, or whose main line is more than 3 points off. It fired once: HardRock had Nebraska at +800 and +22.5 when the market was about +200 and +7. Without the filter that produced a fake 18% "arb".

Today's slate in the window: 11 NCAAF (late), 6 NHL, 1 MLB (TB@CLE, an ALCS game), 12 NFL (Sunday), 6 UFC fights. NBA and NCAAM have no games yet. The NBA lines on FanDuel are virtual games and were left unmatched. The number of pre-game events fell from 36 to 25 as games started.

## Live results (8 scans; each signal counted once, with how many scans it lasted)

Distinct signals: **13 arbs, 11 middles, 14 +EV**. Every arb leg on Polymarket or Kalshi was re-priced from the live book; all 13 arbs survived that check.

### Arbs

| Event | Legs | Margin | Lasted | Max $ at quoted price | Verdict |
|---|---|---|---|---|---|
| NFL DAL@GB (prop) | Bo Melton receptions O1.5 **+240 Kalshi** / U1.5 **−200 Pinnacle** | **3.9%** | 6/6 prop scans | $500 (Pinnacle limit), Kalshi $2.6k | Best real candidate. Rule risk: if Melton does not play, Pinnacle voids but Kalshi likely settles No |
| NCAAF Ohio@Sacramento St | Sac St +15.5 +116 Polymarket / Ohio −15.5 −108 Bally Bet | 1.7% | 8/8 | $1.6k (PM book) | Lasted 40 min. The Bally leg comes via Action Network and is unverified |
| NCAAF Ohio@Sacramento St | Sac St +15.5 +113 Pinnacle / Ohio −15.5 −108 Bally Bet | 1.1% | 7/8 | $5k Pinnacle | Same Bally quote |
| NFL CAR@PHI | Eagles +3.5 −115 Bally Bet or theScore / Panthers −3.5 +116 Polymarket | 0.2% | 4/8 | $1.4k | Thin |
| NCAAF Ohio@Sacramento St ML | Sac St +700 BetRivers or Bally / Ohio −687 Polymarket | 0.2% | 8/8 | about $19k PM | Thin |
| NHL CAR@CGY ML | Flames +191 Polymarket / Hurricanes −190 HardRock | 0.1% | 8/8 | $689 | Thin |
| NHL STL@NSH | Predators −1.5 +254 Polymarket / Blues +1.5 −245 bet365 | 0.3–0.7% | 2 | $2k | Gone by scan 3 |
| MLB TB@CLE (ML and run line) | CLE −105 bet365 / TB +115 to +147 Polymarket | 2–8% (book-verified 1–4.7%, $48–128) | 1 each | under $130 | **Suspect.** Appeared at first pitch; the bet365 quote via Action Network was stale |

### Middles (all priced between −300 and +300 by construction)

- **Rhamondre Stevenson rushing yards: O59.5 +124/+129 Kalshi / U62.5 −114 Pinnacle.** This one costs nothing: the worst case is +2.0% to +3.0% (Kalshi size $36–268), and a 60–62 yard game wins both legs.
- **Ohio@Sacramento St: Sac St +16.5 +107 Polymarket / Ohio −15.5 −108 Bally.** It lasted all 8 scans, costs 0.3%, and wins both if Ohio wins by exactly 16.
- The rest cost 2.5–3.0% for a 1-point window:
  - HOU@JAX ±3.5/4.5 (Kalshi vs Polymarket)
  - PIT@TB 8.5/9.5 (Kalshi vs Polymarket) and 3.5/4.5 (Pinnacle vs Polymarket)
  - CHI@ATL total 48.5/49.5
  - NM@HAW 11.5/12.5
  - Navy@UTSA total 53.5/54.5
  - La Tech@ULM 15.5/16.5

### +EV (beats the de-vigged odds_ratio consensus of other venues by at least 2%)

Game lines need at least 2 other venues in the consensus. Props need Pinnacle in it (n=1, so they are weaker).

| Event | Bet | Edge | n consensus | Lasted |
|---|---|---|---|---|
| NCAAF Ohio@Sac St | Ohio −15.5 −108 Bally Bet | +3.7% | 3 | 8 |
| NFL CHI@ATL | UNDER 52 +106 Pinnacle (alt line) | +3.0% | 3 | 8 |
| NFL DAL@GB | Bo Melton O1.5 rec +240 Kalshi | +22.9% (Pinnacle only) | 1 | 6 |
| NFL NYJ@NE | Drake Maye U19.5 completions +114 Kalshi | +5.5% (book +4.8%, $113) | 1 | 6 |
| NFL NYJ@NE | Hunter Henry O3.5 rec +198 Kalshi | +5.2% (book +3.5%, $19) | 1 | 6 |
| NFL NO@NYG | Cam Skattebo O1.5 rec −116 Kalshi | +2.7% | 1 | 6 |
| NFL HOU@JAX | Brenton Strange O3.5 rec −137 Kalshi | +2.1% | 1 | 6 |
| NFL CAR@PHI | Panthers −3.5 +116 Polymarket | +2.9% | ≥2 | 4 |
| NFL CHI@ATL | OVER 53.5 +116 Polymarket | +3.1% | 2 | 3 |
| NHL STL@NSH | Predators −1.5 +254 Polymarket | +3.1% | 8 | 2 |
| NCAAF Navy@UTSA | OVER 53.5 +107 Polymarket | +3.7% | 4 | 1 |
| NFL BUF@LV | Raiders −3.5 +424 Polymarket | +2.7% | ≥2 | 1 |
| NCAAF Ohio@Sac St | Sac St +700 BetRivers / Bally | +2.5% | 11 | 1 |

All of these are in `arb_log.csv` with timestamps, fair_p, consensus venues and book-verified edge. `settle()` grades them, first appearance only, from ESPN finals; props are graded from ESPN box scores. A pattern: the exchanges, Kalshi props in particular, are often the soft side against Pinnacle.

## Historical test: Polymarket price vs sportsbook close

**Data.** Gamma closed game events have a gameId and moneyline token, and CLOB `prices-history` gives the last price in the 3h before start at 5-minute fidelity. I fetched 7,150 games: NBA, NHL, NCAAM (to mid-Dec 2025 because of paging), NFL and NCAAF. 3,122 of them joined to `data/processed` closing moneylines from 2025-10-07 to 2026-10-14: NBA 1,370, NHL 819, NCAAM 534, NCAAF 321, NFL 78. Orientation checks out: the final Polymarket price agreed with the actual winner 100% of the time, and the correlation with the de-vigged close is 0.98–0.999.

**Signal.** d = Polymarket pre-game probability − book close de-vigged (odds_ratio).
- Venue "book": bet the side Polymarket rates higher, at the real sportsbook close.
- Venue "pm": buy on Polymarket the side Polymarket rates lower, at the history price + 1c + taker fee. This is approximate, because the history is a last/mark price, not the ask.

Grid: thresholds {2, 3, 5, 8}% × venue {book, pm} × sport {ALL, nba, nhl, ncaam, nfl, ncaaf}, which is 48 tests. Every test is in `arb_scanner_candidates.csv`. Discovery is the first 60% of dates (to 2026-01-18); the lockbox is the rest (1,248 games).

**Pre-declared final-candidate rule:** discovery p < 0.05 with n ≥ 30.
- The best discovery result was ALL/5%/book: n=112, ROI +26%, z=1.62, p=0.053. That is above 0.05 even before any multiple-testing correction, so **no candidate qualified and the lockbox was not used to pick anything.**
- For reference, that cell's lockbox was n=5, −39%. Every pooled ALL lockbox cell had n ≤ 56. None is significant, and BH q > 0.10 for all of them.

**The main finding is structural.** Divergence has disappeared over time:

| Period | Share of games with \|d\| ≥ 2% | Mean \|d\| |
|---|---|---|
| Oct 2025 – Jan 2026 | 15–34% | 1.5–2.6 pts |
| Feb 2026 onward | 2–7% | 0.7 pts |

Polymarket's closing prices now track the sportsbook close closely. There is no persistent information edge to forward-test from the history.

One note on process: an earlier run had an incomplete team-name join (NFL abbreviations, NCAAM mascots). I fixed the join only; the rule grid is unchanged.

| Candidate | Discovery n / ROI | Lockbox n / ROI / z / p | Months positive | Verdict |
|---|---|---|---|---|
| PM-vs-close divergence ≥5%, bet the book (best of 48) | 112 / +26.0% (z 1.62, p 0.053) | 5 / −39% / −1.05 / 0.85 | 100% disc | **FALSE** (fails the discovery bar; divergence has vanished since Feb 2026) |
| Same, ≥2% | 385 / +10.8% (z 1.38) | 56 / −19.7% / −1.46 / 0.93 | 75% | FALSE |
| All other 46 cells | see csv | none significant | | FALSE |
| Live arbs (exempt from 2–5) | n/a | n/a | n/a | **Real but small.** Several persisted 30–40 min, verified on exchange books with fees. Best: a 3.9% Kalshi/Pinnacle prop arb with $500 executable and a rule (DNP) risk. Sportsbook legs sourced through Action Network can lag |
| Live +EV signals | n/a | forward test pending (`settle`) | n/a | UNPROVEN; logged for forward testing |

Mechanism: the exchanges (Kalshi and Polymarket alt and prop lines) are priced by thin market makers who update more slowly than Pinnacle. Soft books that copy feeds (Kambi and Bally) occasionally lag on small-conference college lines. The other side of these trades is a slow quoter, not an informed bettor.

Caveats:
- Sportsbook limits, account restrictions and the delay on Action Network quotes are not modeled.
- NFL 2-way moneylines push on a tie, while Polymarket and Kalshi settle differently.
- Player-prop void rules differ between Kalshi and the books.
- NCAAM backfill stopped at mid-December 2025 because of the paging limit.
