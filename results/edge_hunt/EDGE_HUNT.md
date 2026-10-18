# Edge Hunt — verdict (2026-10-18)

| Hunter | Tests | Lockbox candidates | TRUE edges |
|---|---|---|---|
| Opening-line CLV (beat the move) | 395 | 30 | 0 — CLV is real (+0.7–1.5 pts) but doesn't clear vig; NHL near-pass was goalie leakage |
| Situational (new script features) | 4,354 | 0 survived discovery | 0 |
| Props vs real prices / line shopping / DFS / UFC methods | 546 | 12 | 0 — NBA line-shop +3.2% (p≈0.09), NFL alt ladders +10.8% (2 wks), UFC lead fails discovery |
| Polymarket vs book close (3,122 games) | 48 | 1 | 0 — divergence collapsed after Feb 2026 |
| **Live cross-venue arbitrage** | live | — | **YES (guaranteed, not statistical)** — 0.1–3.9% per arb, size-capped |

Statistical edges still standing: only the original validated keys (results/KEYS.md).
Live arb scanner: `python -m engine.edge_hunt.arb_scanner scan | loop 6 300 | settle`.
Venues: Pinnacle, FanDuel, BetRivers, Bovada, Kalshi, Polymarket, + DK/MGM/Caesars/bet365/Fanatics/HardRock/theScore/Bally via Action Network (can lag — verify before betting).
