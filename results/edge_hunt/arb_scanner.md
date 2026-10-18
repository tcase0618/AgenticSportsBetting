# Hunter: arb_scanner: live cross-venue arbs, middles and +EV, plus a backtest of Polymarket vs the sportsbook close

Code: `engine/edge_hunt/arb_scanner.py` · Run window: 2026-10-17 23:47Z to 2026-10-18 00:26Z (8 scans, about 5.5 min apart)

```
python -m engine.edge_hunt.arb_scanner scan            # one scan, appends results/edge_hunt/arb_log.csv
python -m engine.edge_hunt.arb_scanner loop 6 300      # 6 scans, 300 s apart
python -m engine.edge_hunt.arb_scanner settle          # grade logged +EV signals from ESPN finals -> arb_settled.csv
python -m engine.edge_hunt.arb_scanner pmhist [refresh]  # Polymarket-history backtest
```

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
