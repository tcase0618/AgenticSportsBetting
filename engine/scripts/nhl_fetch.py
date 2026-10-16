"""NHL API (api-web.nhle.com) boxscore cache for the NHL script adapter.

Pulls gamecenter boxscores for regular-season games, keeps only the compact per-player lines
(skaters: goals, assists, points, sog, toi, pp goals; goalies: starter, shots against, goals against)
and caches each game as data/raw/nhl_api/box/<game_id>.json. Resumable; threaded.

  python -m engine.scripts.nhl_fetch 2015 2026      # seasons by start year (2015 = 2015-16)
"""
from __future__ import annotations

import json
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "nhl_api" / "box"
OUT = ROOT / "data" / "raw" / "nhl_api" / "nhl_box_players.parquet"
URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore"


def _toi(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return None


def fetch_one(gid: int) -> str:
    f = CACHE / f"{gid}.json"
    if f.exists():
        return "cached"
    for attempt in range(4):
        try:
            r = requests.get(URL.format(gid=gid), timeout=20)
            if r.status_code == 404:
                f.write_text(json.dumps({"missing": True}))
                return "404"
            r.raise_for_status()
            d = r.json()
            break
        except Exception:
            time.sleep(1 + attempt)
    else:
        return "fail"
    if d.get("gameState") not in ("OFF", "FINAL"):
        return "notfinal"
    rows = []
    for side, opp_side, home in (("homeTeam", "awayTeam", 1), ("awayTeam", "homeTeam", 0)):
        team, opp = d[side]["abbrev"], d[opp_side]["abbrev"]
        pg = d.get("playerByGameStats", {}).get(side, {})
        for pos_grp in ("forwards", "defense"):
            for p in pg.get(pos_grp, []):
                rows.append(dict(team=team, opp=opp, home=home, player_id=p["playerId"], player=p["name"]["default"],
                                 pos=p.get("position"), goals=p.get("goals"), assists=p.get("assists"),
                                 points=p.get("points"), sog=p.get("sog"), ppg=p.get("powerPlayGoals"),
                                 toi_s=_toi(p.get("toi"))))
        for p in pg.get("goalies", []):
            rows.append(dict(team=team, opp=opp, home=home, player_id=p["playerId"], player=p["name"]["default"],
                             pos="G", starter=int(bool(p.get("starter"))), shots_against=p.get("shotsAgainst"),
                             goals_against=p.get("goalsAgainst"), toi_s=_toi(p.get("toi")),
                             pp_ga=p.get("powerPlayGoalsAgainst")))
    meta = dict(game_id=gid, date=d.get("gameDate"), home=d["homeTeam"]["abbrev"], away=d["awayTeam"]["abbrev"],
                home_score=d["homeTeam"].get("score"), away_score=d["awayTeam"].get("score"),
                home_sog=d["homeTeam"].get("sog"), away_sog=d["awayTeam"].get("sog"),
                period_type=d.get("periodDescriptor", {}).get("periodType"), rows=rows)
    f.write_text(json.dumps(meta))
    return "ok"


def fetch_seasons(y0: int, y1: int, workers: int = 8):
    CACHE.mkdir(parents=True, exist_ok=True)
    for y in range(y0, y1 + 1):
        n = 1353 if y >= 2021 else 1271
        gids = [int(f"{y}02{i:04d}") for i in range(1, n + 1)]
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(fetch_one, gids))
        print(y, {k: res.count(k) for k in set(res)}, flush=True)


def load_box() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (games, players) from the cache."""
    games, players = [], []
    for f in sorted(CACHE.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("missing"):
            continue
        rows = d.pop("rows")
        games.append(d)
        for r in rows:
            r["game_id"] = d["game_id"]; r["date"] = d["date"]
            players.append(r)
    return pd.DataFrame(games), pd.DataFrame(players)


if __name__ == "__main__":
    fetch_seasons(int(sys.argv[1]), int(sys.argv[2]))
    g, p = load_box()
    p.to_parquet(OUT, index=False)
    g.drop(columns=[], errors="ignore").to_parquet(OUT.with_name("nhl_box_games.parquet"), index=False)
    print("games", len(g), "player rows", len(p))
