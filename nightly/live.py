"""
Live data layer for the nightly card. Run on a machine with normal internet (your laptop or the
GitHub Actions workflow); the research sandbox could not reach these hosts.

Sources (all free tiers):
  * ESPN public scoreboard JSON: schedule, final scores, venue/indoor flag, ESPN odds fallback
  * The Odds API v4 (env ODDS_API_KEY, free key at the-odds-api.com): best price across US books
    for moneylines/spreads/totals, plus NBA player_points props
  * Open-Meteo (no key): NFL kickoff wind forecast

Each night:
  1. update_results(): append yesterday's finals (with the lines we captured) to data/live/<sport>.csv
  2. today_slate(): today's games with consensus lines (for rule qualification) and best price/book
  3. card: the same rules as the backtest, via nightly.card.team_legs(history=processed + live + today)
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from backtest.core import decimal_to_american, fmt_american
import pandas as pd
import requests

from backtest.core import no_vig_probs

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
ODDS = "https://api.the-odds-api.com/v4/sports/{key}/odds"
SPORTS = {  # our key -> (espn path, odds-api key, extra espn params)
    "nfl": ("football/nfl", "americanfootball_nfl", {}),
    "ncaaf": ("football/college-football", "americanfootball_ncaaf", {"groups": 80, "limit": 400}),
    "nba": ("basketball/nba", "basketball_nba", {}),
    "ncaam": ("basketball/mens-college-basketball", "basketball_ncaab", {"groups": 50, "limit": 400}),
    "mlb": ("baseball/mlb", "baseball_mlb", {}),
    "nhl": ("hockey/nhl", "icehockey_nhl", {}),
    "ufc": ("mma/ufc", "mma_mixed_martial_arts", {}),
}
# ESPN abbreviation -> dataset abbreviation (only differences listed)
ABBR = {
    "nfl": {"WSH": "WAS", "LAR": "LA"},
    "nhl": {"NJ": "NJD", "LA": "LAK", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA"},
    "mlb": {"AZ": "ARI", "OAK": "ATH", "CWS": "CHW", "WAS": "WSH"},
}
NAME_KEYED = {"nba": "displayName", "ncaaf": "location", "ncaam": "location"}
ODDS_ALIASES = {"Los Angeles Clippers": "LA Clippers"}


def _get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 422):
                raise RuntimeError(f"{url}: HTTP {r.status_code} {r.text[:200]}")
        except requests.RequestException:
            pass
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"failed to fetch {url}")


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ------------------------------------------------------------------ ESPN
def team_id(sport: str, team: dict) -> str:
    if sport in NAME_KEYED:
        return team.get(NAME_KEYED[sport]) or team.get("displayName")
    a = team.get("abbreviation", "")
    return ABBR.get(sport, {}).get(a, a)


def espn_games(sport: str, day: datetime) -> pd.DataFrame:
    path, _, extra = SPORTS[sport]
    js = _get(ESPN.format(path=path), {"dates": day.strftime("%Y%m%d"), **extra})
    rows = []
    for ev in js.get("events", []):
        c = ev["competitions"][0]
        teams = {t["homeAway"]: t for t in c["competitors"]}
        if set(teams) != {"home", "away"}:
            continue
        h, a = teams["home"], teams["away"]
        st = ev["status"]["type"]
        o = (c.get("odds") or [{}])[0]
        venue = c.get("venue", {})
        rows.append(dict(
            date=pd.Timestamp(day.date()), espn_id=ev["id"], start=ev.get("date"),
            home=team_id(sport, h["team"]), away=team_id(sport, a["team"]),
            home_name=h["team"].get("displayName"), away_name=a["team"].get("displayName"),
            home_score=float(h["score"]) if st.get("completed") else np.nan,
            away_score=float(a["score"]) if st.get("completed") else np.nan,
            completed=bool(st.get("completed")), neutral=int(bool(c.get("neutralSite"))),
            conference_game=int(bool(c.get("conferenceCompetition"))),
            playoff=int(ev.get("season", {}).get("type", 2) == 3),
            espn_spread_home=_espn_spread(o, h), espn_total=o.get("overUnder"),
            espn_home_ml=(o.get("homeTeamOdds") or {}).get("moneyLine"),
            espn_away_ml=(o.get("awayTeamOdds") or {}).get("moneyLine"),
            indoor=venue.get("indoor"), city=(venue.get("address") or {}).get("city"),
            state=(venue.get("address") or {}).get("state"),
        ))
    return pd.DataFrame(rows)


def _espn_spread(o: dict, home: dict):
    """ESPN 'spread' is from the favorite's view in some feeds; use homeTeamOdds.favorite to sign it."""
    if o.get("spread") is None:
        return np.nan
    sp = abs(float(o["spread"]))
    fav_home = (o.get("homeTeamOdds") or {}).get("favorite")
    if fav_home is None:
        return float(o["spread"])
    return -sp if fav_home else sp


# ------------------------------------------------------------------ The Odds API
def odds_board(sport: str) -> pd.DataFrame:
    """One row per event: consensus (median) lines across books + best price & book per side."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return pd.DataFrame()
    js = _get(ODDS.format(key=SPORTS[sport][1]),
              {"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"})
    rows = []
    for ev in js:
        H = ODDS_ALIASES.get(ev["home_team"], ev["home_team"])
        A = ODDS_ALIASES.get(ev["away_team"], ev["away_team"])
        q = {k: [] for k in ("hml", "aml", "hsp", "hsp_p", "asp_p", "tot", "ov", "un")}
        best = {}
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                oc = {o["name"]: o for o in m["outcomes"]}
                if m["key"] == "h2h" and ev["home_team"] in oc and ev["away_team"] in oc:
                    hp, ap = oc[ev["home_team"]]["price"], oc[ev["away_team"]]["price"]
                    q["hml"].append(hp), q["aml"].append(ap)
                    _best(best, "home_ml", hp, b["title"]), _best(best, "away_ml", ap, b["title"])
                elif m["key"] == "spreads" and ev["home_team"] in oc:
                    q["hsp"].append(oc[ev["home_team"]]["point"])
                    q["hsp_p"].append(oc[ev["home_team"]]["price"]), q["asp_p"].append(oc[ev["away_team"]]["price"])
                elif m["key"] == "totals" and "Over" in oc:
                    q["tot"].append(oc["Over"]["point"])
                    q["ov"].append(oc["Over"]["price"]), q["un"].append(oc["Under"]["price"])
        med = {k: (float(np.median(v)) if v else np.nan) for k, v in q.items()}
        rows.append(dict(odds_id=ev["id"], home_name=H, away_name=A, commence=ev["commence_time"],
                         home_ml=med["hml"], away_ml=med["aml"], spread_home=med["hsp"],
                         spread_home_odds=med["hsp_p"], spread_away_odds=med["asp_p"],
                         total=med["tot"], over_odds=med["ov"], under_odds=med["un"],
                         best=json.dumps(best), n_books=len(ev.get("bookmakers", []))))
    return pd.DataFrame(rows)


def _best(best, k, price, book):
    if k not in best or price > best[k][0]:
        best[k] = (price, book)


def attach_odds(games: pd.DataFrame, board: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    for c in ["home_ml", "away_ml", "spread_home", "spread_home_odds", "spread_away_odds", "total", "over_odds",
              "under_odds", "best"]:
        g[c] = np.nan
    if not board.empty:
        b = board.assign(k=board.home_name.map(norm_name) + "|" + board.away_name.map(norm_name))
        g["k"] = g.home_name.map(norm_name) + "|" + g.away_name.map(norm_name)
        g = g.drop(columns=[c for c in b.columns if c in g.columns and c not in ("k",)], errors="ignore")
        g = g.merge(b.drop(columns=["home_name", "away_name"]), on="k", how="left").drop(columns="k")
    # ESPN fallback where the odds feed had nothing
    g["home_ml"] = g["home_ml"].fillna(g.espn_home_ml)
    g["away_ml"] = g["away_ml"].fillna(g.espn_away_ml)
    g["spread_home"] = g["spread_home"].fillna(g.espn_spread_home)
    g["total"] = g["total"].fillna(g.espn_total)
    return g


# ------------------------------------------------------------------ history maintenance
def update_results(sport: str, days_back=3) -> pd.DataFrame:
    """Append recent finals to data/live/<sport>.csv using the lines we captured pre-game
    (data/live/slate_<sport>_<date>.csv) or ESPN's odds as a fallback."""
    LIVE.mkdir(parents=True, exist_ok=True)
    f = LIVE / f"{sport}.csv"
    hist = pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame()
    today = datetime.now()
    new = []
    for d in range(1, days_back + 1):
        day = today - timedelta(days=d)
        g = espn_games(sport, day)
        g = g[g.completed]
        snap = LIVE / f"slate_{sport}_{day:%Y-%m-%d}.csv"
        if snap.exists():
            s = pd.read_csv(snap)
            keep = ["home", "away", "home_ml", "away_ml", "spread_home", "spread_home_odds", "spread_away_odds",
                    "total", "over_odds", "under_odds"]
            g = g.merge(s[keep], on=["home", "away"], how="left")
        else:
            g = attach_odds(g, pd.DataFrame())
        new.append(g)
    if new:
        hist = pd.concat([hist] + new, ignore_index=True)
        hist = hist.drop_duplicates(["date", "home", "away"], keep="last")
        hist.to_csv(f, index=False)
    return hist


def season_of(sport: str, day: datetime) -> int:
    # dataset convention: NBA/NHL/NCAAM seasons labelled by END year; NFL/NCAAF/MLB by start year
    if sport in ("nba", "nhl", "ncaam"):
        return day.year + (1 if day.month >= 8 else 0)
    if sport in ("nfl", "ncaaf"):
        return day.year - (1 if day.month <= 2 else 0)
    return day.year


def history_with_today(sport: str, slate: pd.DataFrame) -> pd.DataFrame:
    from backtest import run_team as rt
    base = rt.load(sport)
    base["date"] = pd.to_datetime(base["date"])
    live = LIVE / f"{sport}.csv"
    parts = [base]
    if live.exists():
        L = pd.read_csv(live, parse_dates=["date"])
        parts.append(L[L.date > base.date.max()])
    parts.append(slate)
    H = pd.concat(parts, ignore_index=True)
    H["season"] = H["season"].fillna(H["date"].map(lambda d: season_of(sport, d)))
    return H.drop_duplicates(["date", "home", "away"], keep="last")


# ------------------------------------------------------------------ NFL wind
def nfl_wind(slate: pd.DataFrame) -> pd.Series:
    out = []
    for _, r in slate.iterrows():
        if r.get("indoor") is True or not r.get("city"):
            out.append(0.0 if r.get("indoor") else np.nan)
            continue
        try:
            geo = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": r["city"], "count": 5, "country": "US"})
            res = [x for x in geo.get("results", []) if not r.get("state") or x.get("admin1", "") == r["state"]] \
                or geo.get("results", [])
            lat, lon = res[0]["latitude"], res[0]["longitude"]
            kick = pd.Timestamp(r["start"]).tz_convert("UTC")
            fc = _get("https://api.open-meteo.com/v1/forecast",
                      {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m", "wind_speed_unit": "mph",
                       "timezone": "UTC", "start_date": kick.strftime("%Y-%m-%d"), "end_date": kick.strftime("%Y-%m-%d")})
            hrs = pd.to_datetime(fc["hourly"]["time"]).tz_localize("UTC")
            w = pd.Series(fc["hourly"]["wind_speed_10m"], index=hrs)
            out.append(float(w.iloc[np.argmin(np.abs((hrs - kick).total_seconds()))]))
        except Exception:
            out.append(np.nan)
    return pd.Series(out, index=slate.index)


# ------------------------------------------------------------------ live card
def today_slate(sport: str) -> pd.DataFrame:
    now = datetime.now()
    g = espn_games(sport, now)
    g = g[~g.completed]
    if g.empty:
        return g
    g = attach_odds(g, odds_board(sport))
    g["season"] = season_of(sport, now)
    if sport == "nfl":
        g["wind"] = nfl_wind(g)
        g["roof"] = np.where(g.indoor == True, "dome", "outdoors")  # noqa: E712
    LIVE.mkdir(parents=True, exist_ok=True)
    g.to_csv(LIVE / f"slate_{sport}_{now:%Y-%m-%d}.csv", index=False)   # becomes tomorrow's line history
    return g


def live_card() -> str:
    from nightly.card import team_legs, parlay_tickets, STAKE, fmt_price
    today = pd.Timestamp(datetime.now().date())
    legs, notes = [], []
    for sport in ["nfl", "ncaaf", "nba", "ncaam", "mlb", "nhl"]:
        try:
            update_results(sport)
            slate = today_slate(sport)
            if slate.empty:
                continue
            H = history_with_today(sport, slate)
            L = team_legs(sport, today, history=H)
            for x in L:  # attach best available price + book for the bet
                row = slate[(slate.home == x["_home"]) & (slate.away == x["_away"])]
                if len(row) and isinstance(row.iloc[0].get("best"), str) and x["market"] == "ml":
                    b = json.loads(row.iloc[0]["best"]).get("home_ml" if x["_home_side"] else "away_ml")
                    if b:
                        x["price"], x["book"] = float(b[0]), b[1]
            legs += L
        except Exception as e:  # one feed failing must not kill the card
            notes.append(f"{sport}: {e}")
    legs += live_ufc_legs(today, notes)
    tickets = parlay_tickets(legs)
    lines = [f"# Bet Card: {today:%A %b %d, %Y} (LIVE)", "", "1u = 1% of bankroll", "",
             "| Tier | Sport | Strategy | Game | Bet | Price | Book | Stake |", "|---|---|---|---|---|---|---|---|"]
    for x in sorted(legs, key=lambda z: (z["tier"], z["sport"])):
        lines.append(f"| {x['tier']} | {x['sport'].upper()} | {x['strategy']} | {x['event']} | **{x['pick']}** | "
                     f"{fmt_price(x['price'])} | {x.get('book', 'best available')} | {STAKE[x['tier']]}u |")
    if tickets:
        lines += ["", "## Parlays (one book, different games)", "", "| Legs | Payout | Stake |", "|---|---|---|"]
        for t in tickets:
            lines.append(f"| {' + '.join(L['pick'] for L in t['legs'])} | {fmt_american(decimal_to_american(t['dec']))} | {t['stake']}u |")
    try:
        props = live_nba_props(today)
        if len(props):
            lines += ["", "## NBA points props (EV ≥ 4%)", "", "| Bet | Price | Book | EV | Stake |", "|---|---|---|---|---|"]
            for _, r in props.iterrows():
                lines.append(f"| {r.player} {r.side.upper()} {r.line:g} | {fmt_price(r.price)} | {r.book} | "
                             f"{r.ev*100:+.1f}% | {STAKE['prop']}u |")
    except Exception as e:
        notes.append(f"nba props: {e}")
    if not legs:
        lines += ["", "_No qualifying strategy bets tonight. Passing is a position._"]
    if notes:
        lines += ["", "**Feed warnings:** " + "; ".join(notes)]
    return "\n".join(lines)


# ------------------------------------------------------------------ UFC live
def live_ufc_legs(today: pd.Timestamp, notes: list) -> list[dict]:
    """Youth Over Experience: <=155 lbs, favorite (de-vigged) who is 5+ years younger."""
    try:
        board = odds_board("ufc")
    except Exception as e:
        notes.append(f"ufc odds: {e}")
        return []
    if board.empty:
        return []
    tott = pd.read_csv(ROOT / "data" / "reference" / "ufc_fighter_tott.csv")
    tott["k"] = tott.FIGHTER.map(norm_name)
    tott["dob"] = pd.to_datetime(tott.DOB, errors="coerce")
    tott["lbs"] = pd.to_numeric(tott.WEIGHT.str.extract(r"(\d+)")[0], errors="coerce")
    T = tott.drop_duplicates("k").set_index("k")
    legs = []
    for _, r in board.iterrows():
        if pd.Timestamp(r.commence).tz_convert("America/New_York").date() > (today + pd.Timedelta(days=1)).date():
            continue
        a, b = norm_name(r.home_name), norm_name(r.away_name)
        if a not in T.index or b not in T.index or pd.isna(r.home_ml) or pd.isna(r.away_ml):
            continue
        pa, pb, _ = no_vig_probs(r.home_ml, r.away_ml)
        wa, wb = T.loc[a, "lbs"], T.loc[b, "lbs"]
        if max(wa, wb) > 155:
            continue
        age = lambda k: (today - T.loc[k, "dob"]).days / 365.25
        for me, op, p, price, name in ((a, b, pa, r.home_ml, r.home_name), (b, a, pb, r.away_ml, r.away_name)):
            if p > 0.5 and age(op) - age(me) >= 5:
                legs.append(dict(sport="ufc", strategy="Youth Over Experience", tier="A",
                                 event=f"{r.home_name} vs {r.away_name}", market="ml", pick=f"{name} ML",
                                 price=float(price), _home="", _away="", _home_side=True))
    return legs


# ------------------------------------------------------------------ NBA props live
def live_nba_props(today: pd.Timestamp) -> pd.DataFrame:
    """Needs a refreshed data/processed/nba_player_box (run scripts/data/nba_build.py / props build first)."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return pd.DataFrame()
    from backtest import props_nba as P
    events = _get("https://api.the-odds-api.com/v4/sports/basketball_nba/events", {"apiKey": key})
    rows = []
    for ev in events:
        if pd.Timestamp(ev["commence_time"]).tz_convert("America/New_York").date() != today.date():
            continue
        js = _get(f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{ev['id']}/odds",
                  {"apiKey": key, "regions": "us", "markets": "player_points", "oddsFormat": "american"})
        for b in js.get("bookmakers", []):
            for m in b.get("markets", []):
                byp = {}
                for o in m["outcomes"]:
                    byp.setdefault((o["description"], o["point"]), {})[o["name"]] = o["price"]
                for (player, line), d in byp.items():
                    if "Over" in d and "Under" in d:
                        rows.append(dict(date=today, player=player, line=line, over_odds=d["Over"],
                                         under_odds=d["Under"], book=b["key"], alt_line=0, market="pts",
                                         home_team=ev["home_team"], away_team=ev["away_team"]))
    q = pd.DataFrame(rows)
    if q.empty:
        return q
    box = pd.read_csv(P.proc_path("nba_player_box"), low_memory=False,
                      usecols=["date", "player_id", "player", "team", "team_name"])
    ids = box.sort_values("date").drop_duplicates("player_id", keep="last")
    ids["k"] = ids.player.map(norm_name)
    q["k"] = q.player.map(norm_name)
    q = q.merge(ids[["k", "player_id", "team", "team_name"]], on="k", how="left").dropna(subset=["player_id"])
    q["pid"] = q.player_id.astype("Int64")
    names = ids.drop_duplicates("team").set_index("team_name")["team"].to_dict()
    names.update({ODDS_ALIASES.get(k, k): v for k, v in list(names.items())})
    model_season = season_of("nba", datetime.now())
    up = q.drop_duplicates("pid")[["pid", "player", "team", "team_name", "home_team", "away_team"]].copy()
    up["home"] = (up.team_name.map(norm_name) == up.home_team.map(lambda n: norm_name(ODDS_ALIASES.get(n, n)))).astype(int)
    up["opp"] = np.where(up.home == 1, up.away_team.map(lambda n: names.get(ODDS_ALIASES.get(n, n))),
                         up.home_team.map(lambda n: names.get(ODDS_ALIASES.get(n, n))))
    up["game_id"] = -(up.home_team + up.away_team).map(lambda x: abs(hash(x)) % 10**9)
    up = up.rename(columns={"pid": "player_id"}).assign(date=today, season=model_season)
    F = P.box_features(upcoming=up[["date", "season", "game_id", "team", "opp", "home", "player_id", "player"]])
    tonight = F[F.upcoming == True]  # noqa: E712
    proj = P.projections_for(F, tonight, model_season).drop_duplicates("player_id")
    q = q.merge(proj.rename(columns={"player_id": "pid"}).assign(pid=lambda d: d.pid.astype("Int64")), on="pid", how="left")
    q["sd_c"] = q.proj_sd.fillna(P.sd_from_line(q.line))
    q["season"], q["result"], q["y"] = model_season, "", 0
    q = P.consensus(q)
    q["dec_o"], q["dec_u"] = P.american_to_decimal(q.over_odds), P.american_to_decimal(q.under_odds)
    q["p_mkt"] = 1 - P.norm.cdf((q.line - q.mu_mkt) / q.sd_c)
    q["p_mod"] = 1 - P.norm.cdf((q.line - q.proj) / q.sd_c)
    q["p_blend"] = 0.75 * q.p_mkt + 0.25 * q.p_mod
    B = P.pick_bets(q, "p_blend", 0.04)
    return B.sort_values("ev", ascending=False)
