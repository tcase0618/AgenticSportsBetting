# arb_scanner +EV audit (2026-10-22)

Scope: every first-appearance `kind=ev` signal in `arb_log.csv` from 2026-10-17 to 2026-10-21. That is 60 rows; 2
theScore rows were flagged as stale and are excluded, leaving 58. Sources:
- the 19 quote snapshots in `arb_quotes/`
- `quote_history.sqlite`
- Kalshi `GET /markets/{ticker}` and 1-minute candlesticks (refetched; every market is still served)
- Polymarket Gamma markets and CLOB `prices-history` (refetched)
- ESPN finals and box scores

Per-signal data: `ev_audit_clv.csv`.

## Verdict

- **No mapping or side bug in the signals.** Every placeable Kalshi and Polymarket signal maps to the right outcome,
  line, player and price, and the price is the ask we would pay, fee included. The evidence is below.
- **The settlement bugs are fixed and tested.** These bugs made the running record wrong: it read 41 settled at
  −7.5%. The corrected record is **50 settled, −12.2 u, ROI −24.4%** against +3.2% expected. The placeable
  (Kalshi/Polymarket) subset is 36 bets at −23.8%.
- **What actually went wrong is the fair price, not the plumbing.**
  - Kalshi player props were priced against a single venue: Pinnacle props, with 6.4% hold and $250 limits. Kalshi
    never moved toward that price. On 358 settled lines, Kalshi's own price was at least as well calibrated as
    Pinnacle's.
  - Game-line signals that Pinnacle alone did not support lost to the close every time it could be measured.
- The 10/21 evening (1 win in 12 placeable) was mostly bad luck on top of a real but smaller overstatement:
  - P(≤1 win) = 0.44% at the logged fair prices.
  - P(≤1 win) = 0.84% even at Kalshi/Polymarket's own closing mids.
  - Expected wins were 5.6 at logged fair and 5.2 at the closing mids.

## 1. Kalshi and Polymarket mapping audit

**Kalshi.** The scanner's line is `floor_strike`, and every market has `strike_type=greater`. So "N+" means YES =
over N−0.5. Over = buy YES at `yes_ask`; under = buy NO at `no_ask` = 1 − yes_bid. For the logged prices:
- Every logged price equals the ask plus `kx_fee` to 4 decimals, checked against Kalshi's own 1-minute candle at the
  scan minute (14 of 14 that have candles). It is not the bid, and not the last trade.
- The Pinnacle line is the same number in every case, and the player is the same.

| Signal | Kalshi market (title, floor) | Our side → contract | Logged cost = candle ask + fee | Pinnacle at 22:51 (same line) | Fair (logged) | Kalshi mid at entry → close | Result (ESPN) |
|---|---|---|---|---|---|---|---|
| McLeod goal +529 | `KXNHLGOAL-26OCT21PHIBUF-BUFRMCLEOD71-1` "1+ goals", 0.5 | over → YES | .159 = .15 + fee | O0.5 +360 / U −600 | .177 | .145 → .145 | 0 G |
| Dvorak goal +398 | `…PHICDVORAK22-1` "1+ goals", 0.5 | over → YES | .201 = .19 + fee | O0.5 +280 / U −417 | .226 | .185 → .175 | 0 G |
| Bennett ast O0.5 +198 | `KXNHLAST-…FLASBENNETT9-1` "1+ assists", 0.5 | over → YES | .335 | O0.5 +164 / U −224 | .343 | n/a → .320 | 0 A |
| Batherson goal U0.5 −228 | `KXNHLGOAL-…OTTDBATHERSON19-1` "1+ goals", 0.5 | under → NO | .695 = .68 + fee | O +210 / U −310 | .718 | .675 → .675 | 0 G (won) |
| Randle U19.5 −121 | `KXNBAPTS-26OCT21CHABKN-BKNJRANDLE30-20` "20+ points", 19.5 | under → NO | .5475 = .53 + fee | O19.5 +111 / U −147 | .561 | .525 → .525 | 33 pts¹ |
| Demin O3.5 ast −107 | `KXNBAAST-…BKNEDEMIN8-4` "4+ assists", 3.5 | over → YES | .5175 = .50 + fee | O3.5 −133 / U +100 | .536 | .495 → .515 | 2 ast |
| JJJ O19.5 +114 | `KXNBAPTS-…UTAJJACKSON20-20` "20+", 19.5 | over → YES | .4674 = .45 + fee | O19.5 −105 / U −126 | .477 | .445 → .445 | 15 pts |
| Zubac U14.5 +119 | `KXNBAPTS-…INDIZUBAC40-15` "15+", 14.5 | under → NO | .4573 = .44 + fee | O14.5 −129 / U −103 | .472 | .435 → .435 | 16 pts |
| Luka O7.5 ast −137 | `KXNBAAST-…LALLDONCIC77-8` "8+", 7.5 | over → YES | .5773 = .56 + fee | O7.5 −166 / U +124 | .589 | .555 → .545 | 7 ast |
| Booker O5.5 ast −116 | `KXNBAAST-…PHXDBOOKER15-6` "6+", 5.5 | over → YES | .5375 = .52 + fee | O5.5 −141 / U +107 | .551 | .515 → .515 | not alerted² |

¹ ESPN's box score has Randle at 33. Kalshi's `expiration_value` field says 24, but the market resolved YES (20+)
either way.

² Booker's top of book was only $3.64. The book-verified edge was 0.75% (below 2%), so no alert fired; it is in the
log only.

The sides agree with Pinnacle every time. Kalshi and Pinnacle favour the same side of every line, so the over/under
labels are not swapped on either venue. A fresh fetch of today's Pinnacle prop markets confirms that the
`participantId`s are named "Over" and "Under".

**Polymarket.** The `line` field applies to `outcomes[0]`, and sides are assigned by team name, not by listing order.

| Alert | Gamma market | Outcomes, token used | Our line | Check |
|---|---|---|---|---|
| Flyers ML +175 | "Flyers vs. Sabres" (moneyline) | [Flyers, Sabres], idx 0 | ML | ok (won) |
| Hornets/Nets O221.5 +117 | "Hornets vs. Nets: O/U 221.5" | [Over, Under], idx 0 | O221.5 | ok |
| Nets −1.5 +122 | "Spread: Nets (−1.5)", line −1.5 | [Nets, Hornets], idx 0 | Nets −1.5 | ok |
| Pelicans +2.5 +113 | "Spread: Pacers (−2.5)", line −2.5 | [Pacers, Pelicans], idx 1 | Pelicans +2.5 | ok |
| Jazz ML +122 | "Jazz vs. Grizzlies" | [Jazz, Grizzlies], idx 0 | ML | ok |
| Jazz +1.5 +108 | "Spread: Grizzlies (−1.5)", line −1.5 | [Grizzlies, Jazz], idx 1 | Jazz +1.5 | ok |

- Prices are the best ask plus the feeSchedule taker fee. The ask for token 1 is 1 − token 0's bid.
- Alerts were re-priced from `/book`, and every alert had `book_american == american`.
- `grade()` applies the home/away spread signs correctly. The final `outcomePrices` agree with the logged results.

### Bugs found (all in settlement and parsing, none in signal mapping)

1. **`_player_box` crashed on any player who did not play.** ESPN sends `stats: []` for a DNP, which raised an
   `IndexError`. `settle` caught it and stored an empty box, so **every prop in that game stayed ungraded**. This hit
   all the 10/21 NBA props except Booker's. It is the main reason the running record showed 41 settled at −7.5%.
2. **NHL props were never graded.** `PROP_BOX` had no hockey labels: NHL uses `G` and `A`, not `AST`. All Kalshi NHL
   props (McLeod, Dvorak, Bennett, Batherson) stayed pending forever.
3. **`settle` graded stale-flagged rows**, such as the two theScore NM State rows flagged "price unchanged 631m". These
   were never actionable.
4. **Pinnacle "Total Threes Made" props were keyed as player "cameronjohnsontotal".** The `PROP_STATS` key was
   "Threes Made", so threes props never matched Kalshi. This did no harm (it only lost signals), but it is wrong.

All four are fixed in `engine/edge_hunt/arb_scanner.py`:
- `_parse_box`, plus `PROP_BOX_NHL` with `hockey_points = G + A` and `threes` from `3PT`
- the flag filter in `settle`, plus a line for the placeable subset
- the `"Total Threes Made"` key

Tests are in `tests/test_arb_scanner_settle.py`, offline with mocked ESPN payloads; all pass. `settle` was re-run, so
`arb_settled.csv` is regraded. The pre-audit copy is in the session scratchpad.

Still open, by design (not changed): `settle` grades at the scanned price (`dec`), not the book-verified $100 price.
- Gobert was logged at +2.1% edge but was −1.8% after book verification, and it is still graded.

## 2. CLV

Definitions:
- **Sharp close:** Pinnacle's last pre-start quote, de-vigged with odds_ratio. It counts only if a snapshot exists at
  least 15 min after the signal and at most 120 min before the start. CLV_pin = dec × p_close − 1.
- **Venue close:**
  - Kalshi: the true close from 1-minute candles at the start minute.
  - Polymarket: the last `prices-history` point before the start. This is a mid/last price, not the ask.
- **Move:** the change in the venue mid from entry to close, toward our side, in points.

**No snapshot near close exists for 34 of 58 signals:**
- all 10/17 NFL games, 17–41 h after the last scan
- the 10/19 Celtics game and Hurricanes@Canucks
- every 10/21 evening signal. The last scan was 22:52Z, one minute after those signals, so "Pinnacle at close" would
  just be the entry price.

For exchange signals, the venue close is available for 38 of 41.

| Cut | n | Mean edge at signal | n with sharp close | CLV vs Pinnacle close | % positive | Pinnacle move toward us (pts) | n with venue close | CLV vs venue closing mid | Venue mid move (pts) | Settled | ROI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **All** | 58 | 3.6% | 24 | **−0.5%** | 38% | −1.5 | 38 | **−3.3%** | +0.2 | 50 | −24.4% |
| Kalshi | 20 | 4.8% | 3 | −3.3% | 33% | −2.6 | 18 | −5.9% | +0.1 | 16 | −21.1% |
| Polymarket | 21 | 2.7% | 12 | −1.4% | 25% | −1.8 | 20 | −0.9% | +0.2 | 20 | −26.0% |
| Sportsbooks (not placeable) | 17 | 3.2% | 9 | +1.8% | 56% | −0.8 | — | — | — | 14 | −25.8% |
| Game lines | 39 | 2.9% | 22 | −0.5% | 36% | −1.5 | 21 | −1.3% | +0.1 | 35 | −21.5% |
| Props (all Kalshi) | 19 | 4.9% | 2 | −0.5% | 50% | −1.4 | 17 | **−5.7%** | +0.3 | 15 | −31.0% |
| Edge 2–3% | 40 | 2.4% | 22 | −0.7% | 36% | −1.6 | 26 | −2.7% | +0.2 | 35 | −38.5% |
| Edge 3–5% | 9 | 3.4% | 1 | −1.8% | 0% | −2.2 | 7 | −2.6% | +0.1 | 8 | +15.0% |
| Edge 5–10% | 6 | 5.5% | 1 | +6.4% | 100% | +0.4 | 2 | −5.1% | — | 5 | +41.5% |
| Edge 10%+ (NFL Melton, NHL goals) | 3 | 15.7% | 0 | none near close | — | — | 3 | −8.3% | 0.0 | 2 | −100% |

How to read it:
- **Kalshi props.** The Kalshi mid did not move at all toward the Pinnacle-implied fair: +0.3 pts on average, and exactly
  0.0 on 6 of the 9 evening props that have an entry candle. We paid the spread plus the fee against a market that stayed where it was.
  - Measured against Kalshi's own close, every Kalshi prop bet was −EV, by 0.5% to 19%.
  - There is no evidence that the "edge" was Kalshi being stale. It was two venues disagreeing.
- **Game lines.** With a measurable close (all 10/21 morning or 10/19), the sharp close moved against us by 1.5 pts on
  average. CLV was −0.5%, against the +2.9% claimed.

The key split is on game lines only, using Pinnacle-only edge at signal time:

| Game-line group | n | n with sharp close | CLV vs Pinnacle close | % positive | Pinnacle move toward us (pts) |
|---|---|---|---|---|---|
| Pinnacle-only edge ≥ 2% (Pinnacle agrees) | 21 | 13 | **+1.4%** | 62% | −0.8 |
| Pinnacle-only edge < 2% (edge came from soft books) | 10 | 5 | **−3.5%** | 0% | −2.5 |
| Signal on Pinnacle itself vs softer books | 5 | 3 | −3.5% | 0% | −3.0 |

The sharp-weighted fair still lets about 10 soft books (weight 0.5–1 each, about 8 in total) outvote Pinnacle (3).
Examples from 10/21 evening:
- Jazz ML: +2.5% under the sharp-weighted fair, but only +0.6% against Pinnacle alone.
- Jazz +1.5: +2.7%, against +0.7% vs Pinnacle.
- Nets −1.5 (morning): +3.1%, against −0.7% vs Pinnacle.

## 3. Fair-price validation

**Data.** Every quote key (event, market, line) in the 19 snapshots for which ESPN has an outcome. Pushes are dropped.
- Game lines are graded from finals: ML, spread, total and team total.
- Props are graded from the fixed box parser: NBA, NFL and NHL.
- The main set uses the last pre-start snapshot for each key. Keys need Pinnacle plus at least one other venue that
  is two-sided and passes the overround filter (≤ 1.15).
- Uncertainty is a bootstrap standard error clustered by event.

Models compared:
- **Pinnacle-only:** odds_ratio de-vig.
- **Equal-weight:** mean logit across venues.
- **Sharp-weighted:** the live weights (Pinnacle 3, exchanges 1.5 with depth, books 1 or 0.5).

| Set | n keys / events | Pinnacle-only LL / Brier | Equal-weight LL / Brier | Sharp-weighted LL / Brier | Kalshi own LL |
|---|---|---|---|---|---|
| Game lines, last pre-start | 762 / 59 | 0.6613 / 0.2346 | 0.6608 / 0.2344 | 0.6611 / 0.2345 | — |
| Props, last pre-start (Pinnacle and Kalshi both two-sided) | 358 / 25 | 0.6757 / 0.2413 | 0.6747 / 0.2409 | 0.6752 / 0.2411 | **0.6740** / 0.2406 |
| Game lines, all snapshots | 4,888 / 59 | 0.6693 | 0.6688 | 0.6691 | — |
| Props, all snapshots | 2,236 / 25 | 0.6791 | 0.6779 | 0.6785 | 0.6769 |

Differences in log loss (negative = better than Pinnacle-only), last snapshot:

| Comparison | Game lines | Props |
|---|---|---|
| Equal-weight − Pinnacle | −0.0006 ± 0.0007 | −0.0010 ± 0.0008 |
| Sharp-weighted − Pinnacle | −0.0002 ± 0.0003 | −0.0006 ± 0.0003 |
| Kalshi − Pinnacle | −0.0015 ± 0.0013 (n=555) | −0.0017 ± 0.0016 |
| Polymarket − Pinnacle | +0.0001 ± 0.0010 (n=487) | — |

- **The three fair prices are statistically indistinguishable on outcomes.** About 60 events cannot separate models
  that differ by 1–2 pts. Calibration by decile is fine for all three; the biggest gap is n=22 in the 0.8–1.0 bin.
  - The outcome test cannot prove which fair price is right; CLV is the more sensitive instrument, and section 2
    favours Pinnacle on game lines.
- **Props: Pinnacle is not sharper than Kalshi.** Kalshi's own de-vigged price had the lowest log loss in every cut.
  - Where they disagreed by ≥3 pts (n=16), the side Pinnacle preferred won 25%. Pinnacle had it at 51%, Kalshi at 47%.
  - At ≥2 pts (n=65): 43% won, against 51% from Pinnacle and 48.5% from Kalshi.
  - The sample is small, but the direction is clear: nothing supports using Pinnacle props as the single source of
    truth against Kalshi.
- **De-vig on one-sided props.** `venue_fairs` needs both sides at the same line, so a one-sided price is never used.
  - Only 7 of 7,368 Pinnacle prop keys in the snapshots were one-sided, and Pinnacle posts one line per player.
  - Every Kalshi prop fair was therefore Pinnacle's own two-sided price (`n_cons=1`, `Pinnacle:3`): a 6.4% average
    hold (4.6–8.3%) and a $250 max-risk limit.
  - On the NHL 0.5-goal longshots, the 11–13% "edges" come from de-vigging a 7% hold:
    - McLeod: +360/−600 de-vigs to .177 (odds ratio) or .202 (multiplicative), against a Kalshi mid of .145.
    - The edge exists only because two venues disagree.

## 4. Recommendations

**Fixes.**
- Done: the four settlement and parsing bugs above.
- Not done (a policy change, not a bug): for game lines, require **both** the sharp-weighted edge and the
  Pinnacle-only edge to clear the threshold (`min(edge_sw, edge_pin)`). Where Pinnacle has no quote at the line, alert
  only if at least 2 other venues with weight ≥1 agree.
- Not done: drop +EV rows whose venue is Pinnacle; they are not placeable and are the wrong direction.

**Threshold.**
- Use **3% on `min(edge_sw, edge_pin)`, book-verified**, for Polymarket and Kalshi game lines.
- Do not simply raise the sharp-weighted threshold to 4%:
  - Replaying every snapshot quote, placeable signals at 2/3/4/6% settled at −31% (36), −10% (12), +2% (5) and
    −100% (3). Those samples are too small to choose between thresholds.
  - The one clear separator in the CLV data is Pinnacle agreement: +1.4% against −3.5% CLV.
  - Pinnacle-agreeing signals kept about 1 pt of their 2.4% claimed edge. A 3% bar on the Pinnacle-gated edge leaves
    roughly +1.5–2% expected after decay.

**Keep paused.**
- **Kalshi player props**, all sports. Keep them logged and CLV-tracked, but send no alerts:
  - single-venue fair (Pinnacle props: 6.4% hold, $250 limit)
  - zero movement toward Pinnacle (closing-mid CLV −5.7%)
  - Kalshi was at least as well calibrated on 358 settled lines
  - 5 wins in 15 settled, at −31%
  - **NHL 0.5 goal/assist longshots especially.**
  - Re-enable only if forward logging shows Kalshi mids moving toward Pinnacle (positive move) over 50 or more props,
    or if a second sharp prop source is added.
- **Kalshi team totals.** One signal, CLV −8.9%; the edge came from Pinnacle alone at weight 3 against an empty
  consensus.
- **Game-line +EV where Pinnacle disagrees**, until the gate above is implemented.
- Continue Polymarket game lines only with the Pinnacle gate. Run scans within about 90 min of start, so that real
  closes can be captured. Today no snapshot after 22:52Z means the evening card has no sharp close.

**Instrumentation gap.** CLV against a sharp close could be measured for only 24 of 58 signals.
- Add one light scan at T−10 min for every event that has a logged signal.
- Record Pinnacle's price on that event in the snapshot.
- Without this, the next audit will have the same blind spot.
