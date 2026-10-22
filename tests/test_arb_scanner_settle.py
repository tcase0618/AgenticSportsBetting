"""Offline tests of the arb_scanner settlement fixes from the 2026-10-22 +EV audit (mocked ESPN payloads).
Run: python tests/test_arb_scanner_settle.py   (or pytest)"""
import sys, tempfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import engine.edge_hunt.arb_scanner as A

NBA_LABELS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
NHL_LABELS = ["BS", "HT", "TK", "+/-", "TOI", "PPTOI", "SHTOI", "ESTOI", "SHFT", "G", "YTDG", "A", "S", "SM", "SOG"]


def _ath(name, stats):
    return {"athlete": {"displayName": name}, "stats": stats}


def nba_box():
    return {"boxscore": {"players": [{"statistics": [{"name": None, "labels": NBA_LABELS, "athletes": [
        _ath("Julius Randle", ["38", "33", "11-18", "3-5", "8-11", "8", "7", "4", "0", "1", "0", "8", "4", "0"]),
        _ath("Egor Demin", ["27", "4", "2-10", "0-5", "0-0", "4", "2", "3", "1", "0", "2", "2", "1", "+3"]),
        _ath("Josh Minott", []),                        # DNP: empty stats used to crash the whole game's grading
    ]}]}]}}


def nhl_box():
    row = lambda g, a: ["0", "1", "0", "0", "14:04", "0:00", "0:00", "14:04", "18", g, "5", a, "2", "1", "0"]
    return {"boxscore": {"players": [{"statistics": [{"name": "forwards", "labels": NHL_LABELS, "athletes": [
        _ath("Ryan McLeod", row("0", "0")), _ath("Drake Batherson", row("0", "1")), _ath("Healthy Scratch", [])]}]}]}}


def test_box_dnp_and_nba_stats():
    b = A._parse_box(nba_box(), "nba")
    assert b[("juliusrandle", "points")] == 33 and b[("egordemin", "assists")] == 2
    assert b[("juliusrandle", "threes")] == 3                      # '3-5' -> made threes
    assert not any(k[0] == "joshminott" for k in b)


def test_box_nhl_goals_assists_points():
    b = A._parse_box(nhl_box(), "nhl")
    assert b[("ryanmcleod", "goals")] == 0 and b[("drakebatherson", "assists")] == 1
    assert b[("drakebatherson", "hockey_points")] == 1
    assert ("drakebatherson", "goals") in b and b[("drakebatherson", "goals")] == 0   # 'G', not 'YTDG'


def test_pinnacle_threes_description_maps_to_player():
    d = "Cameron Johnson Total Threes Made"
    key = next(k for k in A.PROP_STATS if d.endswith(" " + k))
    assert A.PROP_STATS[key] == "threes" and A.norm(d[: -len(key)].strip()) == "cameronjohnson"


def test_settle_grades_props_and_skips_flagged():
    cols = ["scan_ts", "kind", "sport", "espn_id", "event", "commence", "market", "hline", "side", "venue", "edge",
            "edge_book", "dec", "fair_p", "flag"]
    rows = [["2026-10-21T22:51:22Z", "ev", "nba", "1", "x", "c", "prop:points:juliusrandle", 19.5, "under", "Kalshi",
             .024, .024, 1.826, .56, ""],
            ["2026-10-21T22:51:22Z", "ev", "nba", "1", "x", "c", "prop:assists:egordemin", 3.5, "over", "Kalshi",
             .035, .035, 1.932, .53, ""],
            ["2026-10-21T22:51:22Z", "ev", "nba", "1", "x", "c", "ml", None, "away", "theScore", .03, None, 1.74, .59,
             "price unchanged 631m while median moved +3.3pt"]]
    with tempfile.TemporaryDirectory() as td:
        log, out = Path(td) / "log.csv", Path(td) / "settled.csv"
        pd.DataFrame(rows, columns=cols).to_csv(log, index=False)
        saved = A.LOG, A.SETTLED, A._finals, A._player_box
        try:
            A.LOG, A.SETTLED = log, out
            A._finals = lambda sport, ids: {"1": (110.0, 100.0)}
            A._player_box = lambda sport, eid: A._parse_box(nba_box(), sport)
            L = A.settle()
        finally:
            A.LOG, A.SETTLED, A._finals, A._player_box = saved
    assert len(L) == 2                                     # the flagged theScore row is not a signal
    assert list(L.result) == [-1, -1]                      # Randle 33 > 19.5 (under loses); Demin 2 < 3.5


if __name__ == "__main__":
    for f in [test_box_dnp_and_nba_stats, test_box_nhl_goals_assists_points,
              test_pinnacle_threes_description_maps_to_player, test_settle_grades_props_and_skips_flagged]:
        f()
        print("ok", f.__name__)
    print("PASS")
