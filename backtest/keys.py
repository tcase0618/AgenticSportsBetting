"""
The seven KEYS: one chosen strategy per sport.

Choice rule: from strategies that (1) were discovered on train seasons with p<0.05 and
(2) stayed profitable on untouched holdout seasons, pick the one with the strongest
out-of-sample evidence, weighting sample size, season-to-season consistency and a
real causal mechanism above raw ROI. Narrow parameter-fit variants are listed as
runners-up instead of being promoted.
Key tuple = (market, condition, role, direction, phase) as produced by strategies.py.
"""

KEYS = {
    "nfl": dict(
        name="Wind Wall Unders",
        key=("tot", "wind15", "game", "under", "all"),
        rule="Bet the UNDER at the closing total in any NFL game played in sustained wind of 15+ mph.",
        why="Wind degrades deep passing and field goals far more than totals adjust for. The edge survives "
            "even at the closing number, after the forecast was already public.",
        runners_up=[
            "Bounce-back dogs ATS: bet a +3.5 to +9 type dog (30-45% win prob) whose last game was an ATS "
            "miss by 13+ pts, regular season (337 bets, +14.3% ROI, holdout +12.9%)",
            "Divisional wind unders (wind 15+ & division game: +15.0% ROI on 275 bets)",
        ],
    ),
    "ufc": dict(
        name="Youth Over Experience",
        key=("ml", "younger5", "fav", "on", "lightweight_down"),
        rule="At 155 lbs and below (incl. all women's divisions), bet the moneyline FAVORITE when that "
             "fighter is 5+ years younger than the opponent.",
        why="Speed and reaction time decline fastest in the lighter divisions, and the market prices "
            "record and name value more heavily than age. Younger favorites win more often than their "
            "price implies. Young underdogs show no edge, so this is mispricing at the top of the market, "
            "not a general 'youth' effect.",
        runners_up=[
            "Same rule with an 8+ year age gap (359 bets, +11.8% ROI, holdout z 2.71)",
            "Fade 35+ year-old underdogs at 155 lbs and below (528 bets, +7.6% ROI)",
        ],
    ),
    "ncaaf": dict(
        name="Rested Conference Dogs",
        key=("ats", "rest_adv2", "big_dog", "on", "conf"),
        rule="In CONFERENCE games, bet the ATS underdog of roughly 7+ points (market win prob <= 30%) "
             "when it has 2+ more days of rest than its opponent (e.g. off a bye or Thursday game).",
        why="Extra preparation is worth more against a familiar conference opponent, and big spreads "
            "price talent gaps rather than prep and scheme. The same spot outside conference play (cupcake "
            "mismatches) shows no edge, which is why the conference filter is part of the rule.",
        runners_up=[
            "Off-bye conference big dogs vs. a non-bye opponent (412 bets, +9.2% ROI, holdout +6.1%)",
        ],
    ),
    "nhl": dict(
        name="Blowout Bounce-Back",
        key=("ml", "prev_big_ats_fail", "small_fav", "on", "all"),
        rule="Bet the moneyline on a SMALL favorite (about -101 to -150) whose previous game was a puck-line "
             "disaster: lost as the puck-line favorite, or lost by 4+ goals as an underdog.",
        why="NHL results are very noisy (goaltending, shooting luck), but the market overreacts to the most "
            "recent ugly result. Good teams coming off a bad loss are priced too cheaply. NFL has the same "
            "overreaction pattern (see the NFL runner-up).",
        runners_up=[
            "Small favorites on a 3+ game puck-line losing streak, puck line (878 bets, +9.4% ROI, volatile)",
            "Small favorites after any 3+ goal loss (1,754 bets since 2008, +3.1% ROI)",
        ],
    ),
    "mlb": dict(
        name="Series Carryover Unders",
        key=("tot", "prev_both_under_big", "game", "under", "all"),
        rule="Bet the UNDER when BOTH teams' previous game finished 3.5+ runs under its closing total "
             "(usually yesterday's game in the same series and park).",
        why="Park, weather, umpire-zone and lineup conditions persist day to day within a series, and the "
            "next total under-adjusts. LOW CONFIDENCE: MLB moneylines and totals proved the most efficient "
            "market tested, and nothing passed the strict discovery+holdout gate. This is the best lean, "
            "not a proven edge.",
        runners_up=[
            "Fade a 5+ point moneyline steam move (2021-25 only, +4.8% on 1,352 bets). Excluded because "
            "the 2021+ 'closing' prices failed an integrity check",
        ],
    ),
    "nba": dict(
        name="Load-Management Overs",
        key=("tot", "late_season+total_down", "game", "over", "all"),
        rule="Final quarter of the NBA season (incl. playoffs): bet the OVER when the closing total has dropped "
             "1.5+ points from the opening total.",
        why="Late-season totals get bet down on rest/injury news (stars sitting, tanking, slowed playoff pace), "
            "but replacement-level rotations defend worse and the drop overshoots. Moderate-low confidence: "
            "the market moves are only recorded through 2022 and again in 2026, so 2023-25 can't be tested.",
        runners_up=[
            "Overs when both teams' last game went way over AND the total is rising (+13.8% on 471 bets, "
            "holdout +19.5%). It narrowly missed the pre-registered discovery cutoff (p=0.064), so it stays a "
            "runner-up rather than being promoted after seeing holdout",
            "Road favorite after an upset loss as a favorite, ATS (544 bets, +9.2%)",
        ],
    ),
}
