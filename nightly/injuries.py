"""NFL injury report via nflreadpy (github.com/nflverse/nflreadpy): latest-week Out/Doubtful/Questionable
players for skill positions, attached to NFL legs on the card.

Backtest note (2026-10): a starting-QB change in the game did NOT hurt either NFL strategy
(bounce-back dogs +14.0% with a QB change vs +16.1% without; wind unders +12.4% vs +9.5%), so injuries are
shown as context and never used to veto a bet."""
from __future__ import annotations

import pandas as pd

KEY_POS = {"QB", "RB", "WR", "TE"}
STATUS = {"Out", "Doubtful", "Questionable"}
NAMES = {  # nflverse abbreviations -> words that appear in FanDuel / ESPN team names
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers",
    "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}


def current_report(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    inj = nfl.load_injuries([season]).to_pandas()
    if inj.empty:
        return inj
    inj = inj[inj.week == inj.week.max()]
    inj = inj[inj.position.isin(KEY_POS) & inj.report_status.isin(STATUS)]
    inj["nickname"] = inj.team.map(NAMES)
    return inj[["team", "nickname", "week", "position", "full_name", "report_status", "report_primary_injury"]]


def notes_for_game(report: pd.DataFrame, event: str) -> str:
    """'QB Joe Burrow OUT; WR ... Q' for both teams in 'Away Team @ Home Team'."""
    if report is None or report.empty:
        return ""
    out = []
    for _, r in report.iterrows():
        if isinstance(r.nickname, str) and r.nickname in event:
            tag = {"Out": "OUT", "Doubtful": "D", "Questionable": "Q"}[r.report_status]
            out.append((0 if r.position == "QB" else 1, f"{r.nickname} {r.position} {r.full_name} {tag}"))
    return "; ".join(x for _, x in sorted(out)[:6])
