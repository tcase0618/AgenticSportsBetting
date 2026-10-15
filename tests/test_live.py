"""Offline tests of the live layer with mocked ESPN / Odds API payloads (documented schemas)."""
import sys, json, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import nightly.live as live

TODAY = datetime.now()

def espn_payload():
    def comp(h, a, hs=None, as_=None, done=False):
        t = lambda ab, loc, disp, ha, sc: {"homeAway": ha, "score": str(sc or 0),
                                           "team": {"abbreviation": ab, "location": loc, "displayName": disp}}
        return {"id": f"{h[0]}{a[0]}", "date": TODAY.strftime("%Y-%m-%dT23:00Z"),
                "status": {"type": {"completed": done}}, "season": {"type": 2},
                "competitions": [{"competitors": [t(*h, "home", hs), t(*a, "away", as_)], "neutralSite": False,
                                  "conferenceCompetition": False,
                                  "odds": [{"spread": -1.5, "overUnder": 6.5,
                                            "homeTeamOdds": {"moneyLine": -135, "favorite": True},
                                            "awayTeamOdds": {"moneyLine": 115}}],
                                  "venue": {"indoor": True, "address": {"city": "Tampa", "state": "FL"}}}]}
    return {"events": [comp(("TB", "Tampa Bay", "Tampa Bay Lightning"), ("NJ", "New Jersey", "New Jersey Devils"))]}

def odds_payload():
    o = lambda n, p, pt=None: {"name": n, "price": p, **({"point": pt} if pt is not None else {})}
    bk = lambda title, hp, ap: {"key": title.lower(), "title": title, "markets": [
        {"key": "h2h", "outcomes": [o("Tampa Bay Lightning", hp), o("New Jersey Devils", ap)]},
        {"key": "spreads", "outcomes": [o("Tampa Bay Lightning", 160, -1.5), o("New Jersey Devils", -190, 1.5)]},
        {"key": "totals", "outcomes": [o("Over", -110, 6.5), o("Under", -110, 6.5)]}]}
    return [{"id": "x1", "home_team": "Tampa Bay Lightning", "away_team": "New Jersey Devils",
             "commence_time": TODAY.strftime("%Y-%m-%dT23:00:00Z"),
             "bookmakers": [bk("DraftKings", -130, 110), bk("FanDuel", -126, 108), bk("BetMGM", -135, 114)]}]

def fake_get(url, params=None, tries=3):
    if "espn" in url:
        return espn_payload()
    if "the-odds-api" in url:
        return odds_payload()
    raise RuntimeError(url)

def test_slate_and_card():
    live._get = fake_get
    import os; os.environ["ODDS_API_KEY"] = "test"
    g = live.espn_games("nhl", TODAY)
    assert list(g.home) == ["TBL"] and list(g.away) == ["NJD"], g[["home", "away"]]
    assert g.espn_spread_home.iloc[0] == -1.5
    b = live.odds_board("nhl")
    assert b.home_ml.iloc[0] == -130 and json.loads(b.best.iloc[0])["home_ml"] == [-126, "FanDuel"]
    s = live.attach_odds(g, b)
    assert s.home_ml.iloc[0] == -130 and s.total.iloc[0] == 6.5
    s["season"] = live.season_of("nhl", TODAY)
    H = live.history_with_today("nhl", s)
    from nightly.card import team_legs
    legs = team_legs("nhl", pd.Timestamp(TODAY.date()), history=H)  # must run end-to-end without error
    print("slate ok; tonight's NHL legs on mock slate:", [l["pick"] for l in legs])

if __name__ == "__main__":
    test_slate_and_card()
    print("PASS")
