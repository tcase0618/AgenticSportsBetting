"""
Live cross-venue arbitrage / middle / +EV scanner (hunter "arb_scanner").

Venues (all free, key-less, pre-game quotes only; every venue is optional - a failing feed is logged and skipped):
  Pinnacle      guest.api.arcadia.pinnacle.com   sharp book, main + alternate spreads/totals, ML
  FanDuel       sbapi.<state>.sportsbook.fanduel.com (nightly.providers.fanduel_board)
  BetRivers     Kambi offering API (rsiuspa)      main spread/ML/total
  Bovada        bovada.lv coupon API              main + alternate lines
  ActionNetwork api.actionnetwork.com/web/v2 scoreboard -> DraftKings, BetMGM, bet365, Caesars, Fanatics,
                HardRock, theScore, Bally (each its own venue; quotes may lag the book by minutes)
  Polymarket    Gamma (all ML/spread/total lines) priced at best ask + taker fee from feeSchedule;
                CLOB /book used to verify depth on every flagged leg
  Kalshi        trade-api/v2 GAME/SPREAD/TOTAL series priced at yes/no ask + taker fee
                0.07*mult*P*(1-P) per contract (rounded up per order); orderbook used to verify depth
  ESPN          scoreboard = master event list (today+tomorrow, ET) + DraftKings fallback + settlement

Signals (per event / market / exact line; sides mapped by team identity, never by listing position):
  arb     best price per side from two different venues, sum(1/dec) < 1 after fees (exchange legs re-priced
          from the live order book at the stake size)
  middle  spread/total gap between venues covering >= 1 integer outcome, sum(1/dec) <= MIDDLE_MAX_COST
  ev      a venue's price beats the de-vigged (engine.devig odds_ratio) consensus of >= 2 OTHER venues
          quoting both sides of the same line by >= EV_MIN

  python -m engine.edge_hunt.arb_scanner scan            # one scan, appends results/edge_hunt/arb_log.csv
  python -m engine.edge_hunt.arb_scanner loop 6 300      # 6 scans, 300 s apart
  python -m engine.edge_hunt.arb_scanner settle          # grade logged signals from ESPN finals
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.devig import devig  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = ROOT / "results" / "edge_hunt"
LOG = OUT / "arb_log.csv"
SCANS = OUT / "arb_scans.csv"
QUOTES = OUT / "arb_quotes"
SETTLED = OUT / "arb_settled.csv"

SPORTS = ["nfl", "ncaaf", "nba", "ncaam", "mlb", "nhl", "ufc"]
EV_MIN = 0.02
MIN_CONSENSUS = 2
MIDDLE_MAX_COST = 1.03
STAKE_USD = 100.0          # stake used to re-price exchange legs from the book
PM_MAX_SPREAD = 0.05
HEAD = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
ET = "America/New_York"

ESPN_PATH = {"nfl": "football/nfl", "ncaaf": "football/college-football", "nba": "basketball/nba",
             "ncaam": "basketball/mens-college-basketball", "mlb": "baseball/mlb", "nhl": "hockey/nhl",
             "ufc": "mma/ufc"}
ESPN_GROUPS = {"ncaaf": [80, 81], "ncaam": [50]}
PIN_LEAGUE = {"nfl": 889, "ncaaf": 880, "nba": 487, "ncaam": 493, "mlb": 246, "nhl": 1456, "ufc": 1624}
KAMBI_PATH = {"nfl": "american_football/nfl", "ncaaf": "american_football/ncaaf", "nba": "basketball/nba",
              "ncaam": "basketball/ncaab", "mlb": "baseball/mlb", "nhl": "ice_hockey/nhl", "ufc": "ufc_mma/ufc"}
BOV_PATH = {"nfl": "football/nfl", "ncaaf": "football/college-football", "nba": "basketball/nba",
            "ncaam": "basketball/college-basketball", "mlb": "baseball/mlb", "nhl": "hockey/nhl", "ufc": "ufc-mma"}
AN_LEAGUE = {"nfl": "nfl", "ncaaf": "ncaaf", "nba": "nba", "ncaam": "ncaab", "mlb": "mlb", "nhl": "nhl"}
AN_BOOKS = {68: "DraftKings", 75: "BetMGM", 79: "bet365", 123: "Caesars", 2988: "Fanatics", 2724: "HardRock",
            4620: "theScore", 4693: "BallyBet"}
PM_SERIES = {"nba": 10345, "nfl": 12185, "ncaaf": 12756, "nhl": 10346, "mlb": 3, "ufc": 38, "ncaam": 10470}
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
KX_PROPS = {"nfl": {"KXNFLRECYDS": "receiving_yards", "KXNFLRSHYDS": "rushing_yards", "KXNFLPASSYDS": "passing_yards",
                    "KXNFLREC": "receptions", "KXNFLPASSTDS": "passing_touchdowns",
                    "KXNFLPASSCOMP": "passing_completions", "KXNFLPASSATT": "passing_attempts"},
            "nba": {"KXNBAPTS": "points", "KXNBAREB": "rebounds", "KXNBAAST": "assists", "KXNBA3PT": "threes"},
            "nhl": {"KXNHLGOAL": "goals", "KXNHLAST": "assists", "KXNHLPTS": "hockey_points"}}
# extra Kalshi game-level series (suffix after KX_SERIES prefix) -> our market name
KX_EXTRA = {"1HSPREAD": "spread:1h", "1HTOTAL": "total:1h", "TEAMTOTAL": "team_total"}
KX_SERIES = {"nfl": "KXNFL", "ncaaf": "KXNCAAF", "nba": "KXNBA", "ncaam": "KXNCAAMB", "mlb": "KXMLB", "nhl": "KXNHL"}
EXCHANGES = {"Polymarket", "Kalshi"}
# Pinnacle special description suffix -> Polymarket sportsMarketType (player props, O/U)
PROP_STATS = {"Total Receptions": "receptions", "Total Receiving Yards": "receiving_yards",
              "Total Rushing Yards": "rushing_yards", "Total Passing Yards": "passing_yards",
              "Total Touchdown Passes": "passing_touchdowns", "Total Pass Attempts": "passing_attempts",
              "Total Pass Completions": "passing_completions", "Total Points": "points",
              "Total Rebounds": "rebounds", "Total Assists": "assists", "Threes Made": "threes",
              "Total Goals": "goals"}
# Pinnacle period 1 is the first half only in these sports (hockey: 1st period, baseball: first 5 innings)
PIN_HALF = {"nba", "ncaam", "nfl", "ncaaf"}
PM_HALF = {"first_half_spreads": "spread:1h", "first_half_totals": "total:1h"}

# ---- bettor configuration (19, Tennessee): venues the bettor can actually place on. Override with
#      ARB_PLACEABLE="Kalshi,Polymarket,FanDuel" (comma list).
PLACEABLE = [v.strip() for v in os.environ.get("ARB_PLACEABLE", "Kalshi,Polymarket,FanDuel").split(",") if v.strip()]
# sharp weights for the fair price (log-odds weighted average of per-venue odds_ratio de-vigged probs)
SHARP_W = {"Pinnacle": 3.0, "Circa": 1.0, "BetRivers": 1.0, "FanDuel": 1.0, "DraftKings": 1.0,
           "DraftKings(ESPN)": 1.0}
EXCH_W, EXCH_MIN_DEPTH, OTHER_W = 1.5, 200.0, 0.5
AN_VENUES = {"DraftKings", "BetMGM", "bet365", "Caesars", "Fanatics", "HardRock", "theScore", "BallyBet",
             "DraftKings(ESPN)"}            # quotes that arrive through an aggregator (can lag / freeze)
ALERTS = OUT / "alerts.jsonl"
ALERT_STATE = OUT / "alerts_state.json"
FILLS = OUT / "fills.csv"
FLAGGED = OUT / "flagged_quotes.csv"


def mbase(mkt):
    """'spread:1h' -> 'spread', 'team_total:home' -> 'team_total', 'prop:x:y' -> 'prop'."""
    return str(mkt).split(":", 1)[0]


# ------------------------------------------------------------------ odds helpers
def am2dec(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def dec2am(d):
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def pm_fee(p, fs):
    fs = fs or {}
    rate, ex = float(fs.get("rate", 0) or 0), float(fs.get("exponent", 1) or 1)
    return p * rate * (p * (1 - p)) ** ex


def kx_fee(p, mult=1.0, contracts=100):
    """Kalshi taker fee per contract: ceil(0.07*mult*C*P*(1-P)) cents per order / C."""
    return math.ceil(0.07 * mult * contracts * p * (1 - p) * 100 - 1e-9) / 100 / contracts


def _get(url, params=None, tries=2, timeout=25):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 401, 403, 404, 451):
                break
        except (requests.RequestException, ValueError) as e:
            last = str(e)[:120]
        time.sleep(1 + i)
    raise RuntimeError(f"{url.split('?')[0]}: {last}")


def _q(venue, sport, start, teams, market, sel, line, am, size=np.nan, ref=None, alt=False, **extra):
    """One quote. sel: team name (ml/spread) or 'over'/'under'; line: points for sel (spread) or total.
    size = $ available at the quoted price (exchange top-of-book depth or sportsbook max risk), NaN if unknown.
    extra: tt_team (team of a team total), line_status (Action Network), ..."""
    if am is None or (isinstance(am, float) and np.isnan(am)):
        return None
    d = dict(venue=venue, sport=sport, start=pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo
             else pd.Timestamp(start).tz_localize("UTC"), t1=teams[0], t2=teams[1], market=market, sel=sel,
             line=None if line is None else float(line), am=float(am), dec=am2dec(am), size=size, ref=ref, alt=alt)
    d.update(extra)
    return d


# ------------------------------------------------------------------ name matching
NAME_FIX = {"louisianamonroe": "ulmonroe", "ulm": "ulmonroe", "louisianalafayette": "louisiana",
            "miamifl": "miami", "miamiflorida": "miami", "southernmiss": "southernmississippi",
            "appstate": "appalachianstate", "uconn": "connecticut", "umass": "massachusetts"}


def norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\(#?\d+\)", "", s)                       # '(11) Utah', 'Utah (#11)'
    s = s.lower().replace("&", "and").replace("'", "")
    s = re.sub(r"\bst\.?$", "state", s.strip())
    s = re.sub(r"\buniv(ersity)?\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return NAME_FIX.get(s, s)


class Teams:
    """Alias index of the ESPN teams playing in the scan window, per sport."""

    def __init__(self, events: pd.DataFrame):
        self.events = events
        self.alias = {}
        for _, e in events.iterrows():
            for side in ("home", "away"):
                for a in e[f"{side}_aliases"]:
                    self.alias.setdefault((e.sport, norm(a)), set()).add((e.espn_id, side))

    def resolve(self, sport, name, cand_ids):
        n = norm(name)
        hit = {x for x in self.alias.get((sport, n), set()) if x[0] in cand_ids}
        if hit:
            return hit
        # fuzzy: venue name is a suffix/substring of the ESPN display name ('Texans'), or ends with the
        # ESPN nickname ('JAC Jaguars'); accepted only when exactly one team qualifies
        out = set()
        if len(n) < 4:
            return out
        ev = self.events[self.events.espn_id.isin(cand_ids)]
        last = lambda x: norm(str(x).split()[-1]) if str(x).split() else ""
        for _, e in ev.iterrows():
            for side in ("home", "away"):
                full, nick, loc = norm(e[f"{side}_name"]), norm(e[f"{side}_nick"]), norm(e[f"{side}_loc"])
                if sport == "ufc":   # fighters: same surname (first names vary: Timmy/Timothy)
                    if last(name) and last(name) == last(e[f"{side}_name"]):
                        out.add((e.espn_id, side))
                    continue
                if full.endswith(n) or full.startswith(n) or n in full or (len(nick) >= 4 and n.endswith(nick)) \
                        or (len(loc) >= 4 and n.startswith(loc) and n.endswith(last(e[f"{side}_name"]))):
                    out.add((e.espn_id, side))
        return out if len(out) == 1 else set()

    def match(self, sport, t1, t2, start, tol_h=4.0, date_only=False):
        ev = self.events[self.events.sport == sport]
        if date_only:
            d = pd.Timestamp(start).date()
            ev = ev[ev.date_et == d]
        else:
            ev = ev[(ev.start - pd.Timestamp(start)).abs() <= pd.Timedelta(hours=tol_h)]
        ids = set(ev.espn_id)
        if not ids:
            return None
        a, b = self.resolve(sport, t1, ids), self.resolve(sport, t2, ids)
        for (ia, sa) in a:
            for (ib, sb) in b:
                if ia == ib and sa != sb:
                    return ia, {t1: sa, t2: sb}
        return None


# ------------------------------------------------------------------ ESPN master list
def espn_events(sport, days) -> list:
    rows = []
    for day in days:
        for grp in ESPN_GROUPS.get(sport, [None]):
            p = {"dates": day.strftime("%Y%m%d"), "limit": 500}
            if grp:
                p["groups"] = grp
            try:
                js = _get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/scoreboard", p)
            except Exception:
                continue
            for ev in js.get("events", []):
                comps = ev.get("competitions", [])
                for c in comps:
                    cs = c.get("competitors", [])
                    if len(cs) != 2:
                        continue
                    if sport == "ufc":
                        h, a = cs[0], cs[1]
                        ent = lambda t: t.get("athlete", {})
                    else:
                        d = {t.get("homeAway"): t for t in cs}
                        if set(d) != {"home", "away"}:
                            continue
                        h, a = d["home"], d["away"]
                        ent = lambda t: t.get("team", {})
                    st = (c.get("status") or ev.get("status") or {}).get("type", {})
                    row = dict(sport=sport, espn_id=str(c.get("id") if sport == "ufc" else ev["id"]),
                               card_id=str(ev["id"]), start=pd.Timestamp(c.get("date") or ev["date"]),
                               state=st.get("state"), completed=bool(st.get("completed")))
                    for side, t in (("home", h), ("away", a)):
                        e = ent(t)
                        disp = e.get("displayName") or e.get("fullName") or ""
                        nick = e.get("name") or (e.get("lastName") if sport == "ufc" else "") or ""
                        loc = e.get("location") or ""
                        abbr = e.get("abbreviation") or ""
                        al = {disp, e.get("shortDisplayName") or "", loc, nick if sport != "ufc" else disp, abbr,
                              f"{loc} {nick}", f"{abbr} {nick}"}
                        if sport == "ufc":
                            al |= {e.get("fullName", ""), e.get("shortName", "")}
                        row.update({f"{side}_name": disp, f"{side}_nick": nick, f"{side}_loc": loc,
                                    f"{side}_abbr": abbr, f"{side}_aliases": [x for x in al if x and x.strip()]})
                    # DraftKings via ESPN (fallback venue)
                    o = (c.get("odds") or [{}])[0] if c.get("odds") else {}
                    row["espn_odds"] = o
                    rows.append(row)
    return rows


def espn_dk_quotes(E: pd.DataFrame) -> list:
    out = []
    for _, e in E.iterrows():
        o = e.espn_odds or {}
        if (o.get("provider") or {}).get("name") != "DraftKings":
            continue
        teams = (e.home_name, e.away_name)
        try:
            ml = o.get("moneyline") or {}
            for side, nm in (("home", e.home_name), ("away", e.away_name)):
                v = ((ml.get(side) or {}).get("close") or {}).get("odds")
                if v not in (None, "", "OFF"):
                    out.append(_q("DraftKings(ESPN)", e.sport, e.start, teams, "ml", nm, None, float(str(v).replace("EVEN", "100"))))
                ps = ((o.get("pointSpread") or {}).get(side) or {}).get("close") or {}
                if ps.get("line") and ps.get("odds"):
                    out.append(_q("DraftKings(ESPN)", e.sport, e.start, teams, "spread", nm, float(ps["line"]),
                                  float(str(ps["odds"]).replace("EVEN", "100"))))
            for side in ("over", "under"):
                t = ((o.get("total") or {}).get(side) or {}).get("close") or {}
                if t.get("line") and t.get("odds"):
                    out.append(_q("DraftKings(ESPN)", e.sport, e.start, teams, "total", side, float(t["line"][1:]),
                                  float(str(t["odds"]).replace("EVEN", "100"))))
        except Exception:
            continue
    return [x for x in out if x]


# ------------------------------------------------------------------ venues
def pinnacle(sport):
    L = PIN_LEAGUE[sport]
    base = f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{L}"
    mu, mk = _get(f"{base}/matchups"), _get(f"{base}/markets/straight")
    M = {}
    for m in mu:
        if m.get("type") != "matchup" or m.get("isLive") or m.get("parentId") or m.get("units", "Regular") != "Regular":
            continue
        P = {p["alignment"]: p["name"] for p in m.get("participants", [])}
        if set(P) >= {"home", "away"}:
            M[m["id"]] = (P["home"], P["away"], m["startTime"])
        elif sport == "ufc" and len(m.get("participants", [])) == 2:
            ps = m["participants"]
            M[m["id"]] = (ps[0]["name"], ps[1]["name"], m["startTime"])
    out = []
    # player props: 'special' matchups whose parent is a game matchup
    PP = {}
    for m in mu:
        sp = m.get("special") or {}
        if m.get("type") == "special" and sp.get("category") == "Player Props" and m.get("parentId") in M:
            d = sp.get("description", "")
            key = next((k_ for k_ in PROP_STATS if d.endswith(" " + k_)), None)
            if key:
                ids = {p["id"]: p["name"].lower() for p in m.get("participants", [])}
                PP[m["id"]] = (M[m["parentId"]], PROP_STATS[key], d[: -len(key)].strip(), ids)
    for k in mk:
        if k.get("matchupId") in PP and k.get("type") == "total" and k.get("period") == 0 and k.get("status", "open") == "open":
            (h, a, st), stat, pl, ids = PP[k["matchupId"]]
            lim = next((x["amount"] for x in k.get("limits", []) if x.get("type") == "maxRiskStake"), np.nan)
            for p in k.get("prices", []):
                ou = ids.get(p.get("participantId"))
                if ou in ("over", "under"):
                    q = _q("Pinnacle", sport, st, (h, a), f"prop:{stat}:{norm(pl)}", ou, p["points"], p["price"], lim)
                    if q:
                        q["player"] = pl
                        out.append(q)
    for k in mk:
        per = k.get("period")
        if k.get("matchupId") not in M or k.get("status") != "open":
            continue
        if per == 0:
            sfx = ""
        elif per == 1 and sport in PIN_HALF and k["type"] in ("spread", "total"):
            sfx = ":1h"
        else:
            continue
        h, a, st = M[k["matchupId"]]
        lim = next((x["amount"] for x in k.get("limits", []) if x.get("type") == "maxRiskStake"), np.nan)
        alt = bool(k.get("isAlternate"))
        for p in k.get("prices", []):
            d = p.get("designation")
            nm = h if d == "home" else a if d == "away" else d
            if k["type"] == "moneyline" and d in ("home", "away") and not sfx:
                out.append(_q("Pinnacle", sport, st, (h, a), "ml", nm, None, p["price"], lim))
            elif k["type"] == "spread" and d in ("home", "away"):
                out.append(_q("Pinnacle", sport, st, (h, a), "spread" + sfx, nm, p["points"], p["price"], lim, alt=alt))
            elif k["type"] == "total" and d in ("over", "under"):
                out.append(_q("Pinnacle", sport, st, (h, a), "total" + sfx, d, p["points"], p["price"], lim, alt=alt))
            elif k["type"] == "team_total" and d in ("over", "under") and k.get("side") in ("home", "away") and not sfx:
                out.append(_q("Pinnacle", sport, st, (h, a), "team_total", d, p["points"], p["price"], lim, alt=alt,
                              tt_team=h if k["side"] == "home" else a))
    return [x for x in out if x]


def fanduel(sport):
    from nightly.providers import fanduel_board
    b = fanduel_board(sport)
    out = []
    for _, r in b.iterrows():
        t = (r.home_name, r.away_name)
        g = lambda c: r.get(c) if c in r and pd.notna(r.get(c)) else None
        out += [_q("FanDuel", sport, r.commence, t, "ml", r.home_name, None, g("home_ml")),
                _q("FanDuel", sport, r.commence, t, "ml", r.away_name, None, g("away_ml"))]
        if g("spread_home") is not None:
            out += [_q("FanDuel", sport, r.commence, t, "spread", r.home_name, g("spread_home"), g("spread_home_odds")),
                    _q("FanDuel", sport, r.commence, t, "spread", r.away_name, -g("spread_home"), g("spread_away_odds"))]
        if g("total") is not None:
            out += [_q("FanDuel", sport, r.commence, t, "total", "over", g("total"), g("over_odds")),
                    _q("FanDuel", sport, r.commence, t, "total", "under", g("total"), g("under_odds"))]
    return [x for x in out if x]


def kambi(sport):
    js = _get(f"https://eu-offering-api.kambicdn.com/offering/v2018/rsiuspa/listView/{KAMBI_PATH[sport]}/all/all/matches.json",
              {"lang": "en_US", "market": "US", "useCombined": "true"})
    out = []
    for e in js.get("events", []):
        ev = e["event"]
        if ev.get("state") != "NOT_STARTED":
            continue
        h, a = ev.get("homeName"), ev.get("awayName")
        t = (h, a)
        for b in e.get("betOffers", []):
            lab = b.get("criterion", {}).get("label", "")
            typ = b.get("betOfferType", {}).get("name", "")
            if "Overtime" not in lab and lab not in ("Moneyline", "Point Spread", "Total Points", "Total Goals",
                                                      "Total Runs", "Puck Line", "Run Line", "Handicap"):
                pass
            for o in b.get("outcomes", []):
                if o.get("status") != "OPEN" or not o.get("oddsAmerican"):
                    continue
                am = float(o["oddsAmerican"])
                if typ == "Match" and "Moneyline" in lab:
                    out.append(_q("BetRivers", sport, ev["start"], t, "ml", o.get("participant"), None, am))
                elif typ == "Handicap":
                    out.append(_q("BetRivers", sport, ev["start"], t, "spread", o.get("participant"), o["line"] / 1000, am))
                elif typ == "Over/Under":
                    out.append(_q("BetRivers", sport, ev["start"], t, "total", o.get("label", "").lower(), o["line"] / 1000, am))
    return [x for x in out if x]


def bovada(sport):
    js = _get(f"https://www.bovada.lv/services/sports/event/v2/events/A/description/{BOV_PATH[sport]}",
              {"marketFilterId": "def", "preMatchOnly": "true", "lang": "en"})
    out, now = [], pd.Timestamp.now(tz="UTC")
    for grp in js:
        for e in grp.get("events", []):
            st = pd.Timestamp(e["startTime"], unit="ms", tz="UTC")
            if e.get("live") or st <= now:
                continue
            C = e.get("competitors", [])
            if len(C) != 2:
                continue
            h = next((c["name"] for c in C if c.get("home")), C[0]["name"])
            a = next((c["name"] for c in C if c["name"] != h), C[1]["name"])
            for dg in e.get("displayGroups", []):
                for m in dg.get("markets", []):
                    per = m.get("period", {})
                    if not per.get("main") or per.get("live"):
                        continue
                    d = m.get("description", "")
                    for o in m.get("outcomes", []):
                        pr = o.get("price", {})
                        if o.get("status") not in (None, "O") or not pr.get("american"):
                            continue
                        am = 100.0 if pr["american"] == "EVEN" else float(pr["american"])
                        nm = o.get("description", "")
                        if d in ("Moneyline", "Fight Winner") and "Draw" not in nm:
                            out.append(_q("Bovada", sport, st, (h, a), "ml", nm, None, am))
                        elif d in ("Point Spread", "Spread", "Runline", "Puck Line", "Run Line") and pr.get("handicap"):
                            out.append(_q("Bovada", sport, st, (h, a), "spread", nm, float(pr["handicap"]), am,
                                          alt=dg.get("description") != "Game Lines"))
                        elif d in ("Total",) and pr.get("handicap") and nm.lower() in ("over", "under"):
                            out.append(_q("Bovada", sport, st, (h, a), "total", nm.lower(), float(pr["handicap"]), am,
                                          alt=dg.get("description") != "Game Lines"))
    return [x for x in out if x]


def actionnetwork(sport, days):
    if sport not in AN_LEAGUE:
        return []
    out = []
    for day in days:
        js = _get(f"https://api.actionnetwork.com/web/v2/scoreboard/{AN_LEAGUE[sport]}",
                  {"period": "game", "date": day.strftime("%Y%m%d"), "bookIds": ",".join(map(str, AN_BOOKS))})
        for g in js.get("games", []):
            if g.get("status") not in ("scheduled",):
                continue
            T = {t["id"]: t["full_name"] for t in g.get("teams", [])}
            h, a = T.get(g["home_team_id"]), T.get(g["away_team_id"])
            if not h or not a:
                continue
            for bid, mk in (g.get("markets") or {}).items():
                venue = AN_BOOKS.get(int(bid))
                if not venue:
                    continue
                ev = mk.get("event", {})
                ls = lambda o: o.get("line_status") or ""
                for o in ev.get("moneyline", []):
                    if not o.get("is_live") and o.get("odds"):
                        out.append(_q(venue, sport, g["start_time"], (h, a), "ml", h if o["side"] == "home" else a, None,
                                      o["odds"], line_status=ls(o)))
                for o in ev.get("spread", []):
                    if not o.get("is_live") and o.get("odds") and o.get("value") is not None:
                        out.append(_q(venue, sport, g["start_time"], (h, a), "spread", h if o["side"] == "home" else a,
                                      o["value"], o["odds"], line_status=ls(o)))
                for o in ev.get("total", []):
                    if not o.get("is_live") and o.get("odds") and o.get("value") is not None:
                        out.append(_q(venue, sport, g["start_time"], (h, a), "total", o["side"], o["value"], o["odds"],
                                      line_status=ls(o)))
                for o in ev.get("core_bet_type_6_team_score", []):       # team totals
                    tm = T.get(o.get("team_id"))
                    if not o.get("is_live") and o.get("odds") and o.get("value") is not None and tm \
                            and o.get("side") in ("over", "under") and not o.get("is_alt_market"):
                        out.append(_q(venue, sport, g["start_time"], (h, a), "team_total", o["side"], o["value"],
                                      o["odds"], tt_team=tm, line_status=ls(o)))
    return [x for x in out if x]


def _pm_side_prices(m):
    """(ask token0, ask token1) from Gamma's top of book; NaN if the book is thin."""
    ask, bid = m.get("bestAsk"), m.get("bestBid")
    if not ask or not bid or float(ask) - float(bid) > PM_MAX_SPREAD:
        return np.nan, np.nan
    return float(ask), 1 - float(bid)


def polymarket(sport):
    sid = PM_SERIES.get(sport)
    now = pd.Timestamp.now(tz="UTC")
    evs, off = [], 0
    while off < 1000:
        page = _get("https://gamma-api.polymarket.com/events",
                    {"series_id": sid, "active": "true", "closed": "false", "limit": 200, "offset": off,
                     "end_date_min": (now - pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")})
        evs += page
        if len(page) < 200:
            break
        off += 200
    out = []
    for e in evs:
        if not e.get("gameId") and sport != "ufc":
            continue
        mk = [m for m in e.get("markets", []) if m.get("active") and not m.get("closed") and m.get("acceptingOrders")]
        mls = [m for m in mk if m.get("sportsMarketType") == "moneyline"]
        if not mls:
            continue
        st = mls[0].get("gameStartTime") or e.get("startTime") or e.get("endDate")
        try:
            st = pd.Timestamp(st)
            st = st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC")
        except Exception:
            continue
        if st <= now + pd.Timedelta(minutes=2) or st > now + pd.Timedelta(days=3):
            continue
        teams = json.loads(mls[0]["outcomes"])
        if len(teams) != 2:
            continue
        T = (teams[1], teams[0])                     # Polymarket titles are "Away vs. Home"
        for m in mk:
            smt = m.get("sportsMarketType")
            outs = json.loads(m["outcomes"])
            toks = json.loads(m.get("clobTokenIds") or "[]")
            if len(outs) != 2 or len(toks) != 2:
                continue
            p0, p1 = _pm_side_prices(m)
            fs = m.get("feeSchedule")
            for p, o, tok in ((p0, outs[0], toks[0]), (p1, outs[1], toks[1])):
                if not (0 < p < 1):
                    continue
                c = p + pm_fee(p, fs)
                if c >= 1:
                    continue
                am = dec2am(1 / c)
                ref = json.dumps(dict(token=tok, fs=fs, p=p))
                if smt == "moneyline":
                    out.append(_q("Polymarket", sport, st, T, "ml", o, None, am, ref=ref))
                elif smt == "spreads" and m.get("line") is not None:
                    ln = float(m["line"])                # line applies to outcome[0]
                    out.append(_q("Polymarket", sport, st, T, "spread", o, ln if o == outs[0] else -ln, am, ref=ref))
                elif smt == "totals" and m.get("line") is not None and o.lower() in ("over", "under"):
                    out.append(_q("Polymarket", sport, st, T, "total", o.lower(), float(m["line"]), am, ref=ref))
                elif smt == "first_half_spreads" and m.get("line") is not None:
                    ln = float(m["line"])
                    out.append(_q("Polymarket", sport, st, T, "spread:1h", o, ln if o == outs[0] else -ln, am, ref=ref))
                elif smt == "first_half_totals" and m.get("line") is not None and o.lower() in ("over", "under"):
                    out.append(_q("Polymarket", sport, st, T, "total:1h", o.lower(), float(m["line"]), am, ref=ref))
                elif smt in PROP_STATS.values() and m.get("line") is not None and o.lower() in ("over", "under"):
                    pl = (m.get("question") or "").split(":")[0].strip()
                    q = _q("Polymarket", sport, st, T, f"prop:{smt}:{norm(pl)}", o.lower(), float(m["line"]), am, ref=ref)
                    if q:
                        q["player"] = pl
                        out.append(q)
    return [x for x in out if x]


def _kx_pages(series, max_pages=8):
    ms, cur = [], None
    for _ in range(max_pages):
        p = {"series_ticker": series, "status": "open", "limit": 1000}
        if cur:
            p["cursor"] = cur
        js = _get(f"{KALSHI}/markets", p)
        ms += js.get("markets", [])
        cur = js.get("cursor")
        if not cur or not js.get("markets"):
            break
    return ms


def _kx_date(ev_ticker):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ev_ticker)
    if not m:
        return None
    return datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%b%d").date()


def kalshi(sport, days, series_meta, props=True):
    """Kalshi GAME/SPREAD/TOTAL + 1H spread/total + team totals (+ player-prop ladders when props=True).
    size on each quote = $ at the top of book for that side: YES ask size*price, NO = YES-bid size*NO price."""
    pre = KX_SERIES.get(sport)
    if not pre:
        return []
    dset = {d.date() for d in days}
    titles = {}
    try:
        cur = None
        for _ in range(5):
            p = {"series_ticker": pre + "GAME", "status": "open", "limit": 200}
            if cur:
                p["cursor"] = cur
            js = _get(f"{KALSHI}/events", p)
            for e in js.get("events", []):
                titles[e["event_ticker"].split("-", 1)[1]] = e.get("title", "")
            cur = js.get("cursor")
            if not cur:
                break
    except Exception:
        pass
    out = []
    games = {}

    def depth(m):
        ya, na = float(m.get("yes_ask_dollars") or 0), float(m.get("no_ask_dollars") or 0)
        return ya, na, float(m.get("yes_ask_size_fp") or 0) * ya, float(m.get("yes_bid_size_fp") or 0) * na

    kinds = [("GAME", "ml"), ("SPREAD", "spread"), ("TOTAL", "total")] + [(k, v) for k, v in KX_EXTRA.items()]
    for kind, mname in kinds:
        mult = float(series_meta.get(pre + kind, {}).get("fee_multiplier") or 1)
        try:
            pages = _kx_pages(pre + kind)
        except Exception:
            if kind in ("GAME", "SPREAD", "TOTAL"):
                raise
            continue
        for m in pages:
            d = _kx_date(m["event_ticker"])
            if d not in dset or m.get("status") != "active":
                continue
            key = m["event_ticker"].split("-", 1)[1]
            games.setdefault(key, {"date": d})
            ya, na, ys, ns = depth(m)
            meta = dict(ticker=m["ticker"], mult=mult)
            sub = m.get("yes_sub_title") or m.get("title") or ""
            if kind == "GAME":
                games[key].setdefault("teams", []).append((m["ticker"].rsplit("-", 1)[1], m.get("yes_sub_title"), ya, na,
                                                           ys, ns, meta))
            elif mname.startswith("spread"):
                tm = re.match(r"(.+?) wins (?:the 1st half )?by over ([\d.]+)", sub) or \
                    re.match(r"(.+?) wins (?:the 1st half )?by over ([\d.]+)", m.get("title") or "")
                if tm:
                    games[key].setdefault("spreads", []).append((mname, tm.group(1), float(tm.group(2)), ya, na, ys, ns, meta))
            elif mname == "team_total":
                tm = re.match(r"(.+?) over ([\d.]+)", sub)
                if tm and m.get("floor_strike") is not None:
                    games[key].setdefault("tt", []).append((tm.group(1).strip(), float(m["floor_strike"]), ya, na, ys, ns, meta))
            else:
                if m.get("floor_strike") is not None and "over" in (m.get("title") or "").lower():
                    games[key].setdefault("totals", []).append((mname, float(m["floor_strike"]), ya, na, ys, ns, meta))
    for ser, stat in (KX_PROPS.get(sport, {}).items() if props else []):
        mult = float(series_meta.get(ser, {}).get("fee_multiplier") or 1)
        try:
            pm = _kx_pages(ser)
        except Exception:
            continue
        for m in pm:
            d = _kx_date(m["event_ticker"])
            if d not in dset or m.get("status") != "active" or m.get("floor_strike") is None:
                continue
            key = m["event_ticker"].split("-", 1)[1]
            pl = (m.get("title") or "").split(":")[0].strip()
            ya, na, ys, ns = depth(m)
            games.setdefault(key, {"date": d}).setdefault("props", []).append(
                (stat, pl, float(m["floor_strike"]), ya, na, ys, ns, dict(ticker=m["ticker"], mult=mult)))
    cost = lambda p, mult: p + kx_fee(p, mult) if 0 < p < 1 else np.nan
    for key, g in games.items():
        tm = g.get("teams", [])
        if len(tm) != 2:
            continue
        title = titles.get(key, "")
        names = {}
        for code, sub, *_ in tm:
            seg = next((s_.strip() for s_ in re.split(r"\s+(?:vs\.?|at|@)\s+", title) if s_.strip().startswith(code + " ")), None)
            names[code] = seg or sub
        t = (names[tm[0][0]], names[tm[1][0]])
        sub2name = {sub: names[code] for code, sub, *_ in tm}
        start = pd.Timestamp(datetime.combine(g["date"], datetime.min.time())).tz_localize(ET)

        def emit(market, sel, line, p, side, sz, meta, **extra):
            c = cost(p, meta["mult"])
            if c == c and c < 1:
                q = _q("Kalshi", sport, start, t, market, sel, line, dec2am(1 / c), sz,
                       json.dumps(dict(ticker=meta["ticker"], side=side, mult=meta["mult"], p=p)), **extra)
                if q:
                    out.append(q)

        for i, (code, sub, ya, na, ys, ns, meta) in enumerate(tm):
            other = tm[1 - i]
            emit("ml", names[code], None, ya, "yes", ys, meta)               # YES on its own market
            emit("ml", names[code], None, other[3], "no", other[5], other[6])  # NO on the opponent's market
        for (mname, who, ln, ya, na, ys, ns, meta) in g.get("spreads", []):
            nm = sub2name.get(who)
            if nm is None:
                continue
            opp = t[1] if nm == t[0] else t[0]
            emit(mname, nm, -ln, ya, "yes", ys, meta)
            emit(mname, opp, ln, na, "no", ns, meta)
        for (mname, ln, ya, na, ys, ns, meta) in g.get("totals", []):
            emit(mname, "over", ln, ya, "yes", ys, meta)
            emit(mname, "under", ln, na, "no", ns, meta)
        for (who, ln, ya, na, ys, ns, meta) in g.get("tt", []):
            nm = sub2name.get(who)
            if nm is None:
                continue
            emit("team_total", "over", ln, ya, "yes", ys, meta, tt_team=nm)
            emit("team_total", "under", ln, na, "no", ns, meta, tt_team=nm)
        for (stat, pl, ln, ya, na, ys, ns, meta) in g.get("props", []):
            emit(f"prop:{stat}:{norm(pl)}", "over", ln, ya, "yes", ys, meta, player=pl)
            emit(f"prop:{stat}:{norm(pl)}", "under", ln, na, "no", ns, meta, player=pl)
    return [x for x in out if x]


# ------------------------------------------------------------------ depth checks for exchange legs
def exchange_fill(venue, ref, stake=STAKE_USD):
    """Re-price an exchange leg from the live book. Returns (all-in decimal odds for `stake`, $ fillable at
    the quoted top price incl. fee, top price) or (nan, 0, nan)."""
    try:
        r = json.loads(ref)
        if venue == "Polymarket":
            bk = _get("https://clob.polymarket.com/book", {"token_id": r["token"]})
            asks = sorted(((float(a["price"]), float(a["size"])) for a in bk.get("asks", [])))
            fee = lambda p: p + pm_fee(p, r.get("fs"))
        else:
            bk = _get(f"{KALSHI}/markets/{r['ticker']}/orderbook")["orderbook_fp"]
            opp = bk.get("no_dollars" if r["side"] == "yes" else "yes_dollars") or []
            asks = sorted(((round(1 - float(p), 4), float(s)) for p, s in opp))
            fee = lambda p: p + kx_fee(p, r.get("mult", 1))
        if not asks:
            return np.nan, 0.0, np.nan
        top = asks[0][0]
        top_usd = sum(p * s for p, s in asks if p <= top + 1e-9)
        spent, shares = 0.0, 0.0
        for p, s in asks:
            take = min(s, (stake - spent) / fee(p))
            spent += take * fee(p)
            shares += take
            if spent >= stake - 1e-6:
                break
        if spent < stake - 1e-6:
            return np.nan, top_usd, top
        return shares / spent, top_usd, top
    except Exception:
        return np.nan, 0.0, np.nan


# ------------------------------------------------------------------ scan
def gather(sports=SPORTS, now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    d0 = now.tz_convert(ET).normalize().tz_localize(None)
    days = [d0, d0 + pd.Timedelta(days=1)]
    errors, status = [], {}
    with ThreadPoolExecutor(16) as ex:
        ev_f = {s: ex.submit(espn_events, s, days) for s in sports}
        try:
            smeta = {x["ticker"]: x for x in _get(f"{KALSHI}/series", {"category": "Sports"}).get("series", [])}
        except Exception as e:
            smeta = {}
            errors.append(f"kalshi series: {e}")
        jobs = {}
        for s in sports:
            jobs[("Pinnacle", s)] = ex.submit(pinnacle, s)
            jobs[("FanDuel", s)] = ex.submit(fanduel, s) if s != "ncaam" or True else None
            jobs[("BetRivers", s)] = ex.submit(kambi, s)
            jobs[("Bovada", s)] = ex.submit(bovada, s)
            jobs[("ActionNetwork", s)] = ex.submit(actionnetwork, s, days)
            jobs[("Polymarket", s)] = ex.submit(polymarket, s)
            jobs[("Kalshi", s)] = ex.submit(kalshi, s, days, smeta)
        E = []
        for s, f in ev_f.items():
            try:
                E += f.result()
            except Exception as e:
                errors.append(f"espn {s}: {e}")
        Q = []
        for (v, s), f in jobs.items():
            try:
                r = f.result()
                Q += r
                status[f"{v}:{s}"] = len(r)
            except Exception as e:
                status[f"{v}:{s}"] = -1
                errors.append(f"{v} {s}: {str(e)[:100]}")
    E = pd.DataFrame(E)
    if E.empty:
        return E, pd.DataFrame(), status, errors
    E["start"] = pd.to_datetime(E.start, utc=True)
    E["date_et"] = E.start.dt.tz_convert(ET).dt.date
    E = E.drop_duplicates("espn_id")
    E = E[(E.state == "pre") & (E.start > now + pd.Timedelta(minutes=5))].reset_index(drop=True)
    Q += espn_dk_quotes(E)
    return E, pd.DataFrame(Q), status, errors


def normalize(E, Q):
    """Attach espn_id / side / home-perspective line to each quote; drop anything unmatched."""
    if E.empty or Q.empty:
        return pd.DataFrame()
    T = Teams(E)
    cache, rows = {}, []
    for r in Q.itertuples():
        k = (r.sport, r.t1, r.t2, r.start, r.venue == "Kalshi")
        if k not in cache:
            cache[k] = T.match(r.sport, r.t1, r.t2, r.start.tz_convert(ET) if r.venue == "Kalshi" else r.start,
                               tol_h=8.0 if r.sport == "ufc" else 4.0, date_only=(r.venue == "Kalshi"))
        m = cache[k]
        if not m:
            continue
        eid, sides = m
        d = r._asdict()
        d.pop("Index")
        if r.market == "total" or r.market.startswith("prop:"):
            side, hline = r.sel, r.line
        else:
            side = sides.get(r.sel)
            if side is None:   # sel spelled differently from t1/t2 (e.g. Kambi participant)
                cand = T.resolve(r.sport, r.sel, {eid})
                side = next(iter(cand))[1] if len(cand) == 1 else None
            if side is None:
                continue
            hline = None if r.market == "ml" else (r.line if side == "home" else -r.line)
        d.update(espn_id=eid, side=side, hline=hline)
        rows.append(d)
    N = pd.DataFrame(rows)
    if N.empty:
        return N
    # DraftKings via ESPN only when ActionNetwork DraftKings is absent for that event
    has_dk = set(N.loc[N.venue == "DraftKings", "espn_id"])
    N = N[~((N.venue == "DraftKings(ESPN)") & N.espn_id.isin(has_dk))]
    # keep best price per venue/event/market/line/side (Kalshi YES vs NO-of-opponent, dup listings)
    N = N.sort_values("dec", ascending=False).drop_duplicates(["venue", "espn_id", "market", "hline", "side"])
    ev = E.set_index("espn_id")
    N["event"] = N.espn_id.map(lambda i: f"{ev.at[i, 'away_name']} @ {ev.at[i, 'home_name']}")
    N["commence"] = N.espn_id.map(lambda i: ev.at[i, "start"])
    N["hkey"] = N.hline.fillna(-9999.0)
    N, bad = stale_filter(N.reset_index(drop=True))
    N.attrs["stale"] = bad
    return N


SIDES = {"ml": ("home", "away"), "spread": ("home", "away"), "total": ("over", "under")}


def sides(mkt):
    return SIDES.get(mkt, ("over", "under"))


def stale_filter(N, ml_tol=0.08, line_tol=3.0):
    """Drop every quote of a (venue, event) whose main price/line is off the cross-venue median by more than
    ml_tol de-vigged probability or line_tol points: such gaps are stale/mis-posted feeds (e.g. a book that
    has not moved since the open), not bettable arbs."""
    bad = {}
    for mkt in ("ml", "spread", "total"):
        a, b = sides(mkt)
        rows = []
        X = N[(N.market == mkt) & (~N.alt.astype(bool))]
        for (eid, v), g in X.groupby(["espn_id", "venue"]):
            P = g.pivot_table(index="hkey", columns="side", values="dec", aggfunc="max")
            if a not in P or b not in P:
                continue
            P = P.dropna()
            if P.empty:
                continue
            if mkt == "ml":
                rows.append((eid, v, float(1 / P[a].iloc[0] / (1 / P[a].iloc[0] + 1 / P[b].iloc[0]))))
            else:
                k = (P[a] - P[b]).abs().idxmin()          # most balanced line = venue's main line
                rows.append((eid, v, float(k)))
        R = pd.DataFrame(rows, columns=["espn_id", "venue", "x"])
        if R.empty:
            continue
        R["med"] = R.groupby("espn_id").x.transform("median")
        R["n"] = R.groupby("espn_id").x.transform("size")
        tol = ml_tol if mkt == "ml" else line_tol
        for r in R[(R.n >= 3) & ((R.x - R.med).abs() > tol)].itertuples():
            bad[(r.espn_id, r.venue)] = f"{mkt} {r.x:.3g} vs median {r.med:.3g}"
    if bad:
        keep = [(e, v) not in bad for e, v in zip(N.espn_id, N.venue)]
        N = N[keep].reset_index(drop=True)
    return N, bad


def consensus_fair(N):
    """Per (event, market, hkey, venue) with both sides: odds_ratio de-vigged prob of side A."""
    rows = []
    for (eid, mkt, hk, v), g in N.groupby(["espn_id", "market", "hkey", "venue"]):
        a, b = sides(mkt)
        s = g.set_index("side")
        if a in s.index and b in s.index:
            oa, ob = s.at[a, "am"], s.at[b, "am"]
            if 1 / am2dec(oa) + 1 / am2dec(ob) < 0.99:     # crossed/broken two-sided quote
                continue
            pa, _ = devig(oa, ob, "odds_ratio")
            rows.append(dict(espn_id=eid, market=mkt, hkey=hk, venue=v, pa=float(pa)))
    return pd.DataFrame(rows)


def find_signals(N, verify=True):
    sig = []
    if N.empty:
        return sig
    F = consensus_fair(N)
    # ---- arbs (same line, two different venues)
    for (eid, mkt, hk), g in N.groupby(["espn_id", "market", "hkey"]):
        a, b = sides(mkt)
        A, B = g[g.side == a].sort_values("dec", ascending=False), g[g.side == b].sort_values("dec", ascending=False)
        if A.empty or B.empty:
            continue
        for _, x in A.head(3).iterrows():
            for _, y in B.head(3).iterrows():
                if x.venue == y.venue:
                    continue
                s = 1 / x.dec + 1 / y.dec
                if s < 1:
                    sig.append(dict(kind="arb", legs=[x, y], cost=s, margin=1 - s))
    # ---- middles (different lines, different venues)
    for (eid, mkt), g in N[N.market != "ml"].groupby(["espn_id", "market"]):
        a, b = sides(mkt)
        A = g[g.side == a].sort_values("dec", ascending=False).drop_duplicates("hkey")
        B = g[g.side == b].sort_values("dec", ascending=False).drop_duplicates("hkey")
        for _, x in A.iterrows():
            for _, y in B.iterrows():
                if x.venue == y.venue:
                    continue
                if mkt == "spread":       # home at hx, away at hy: both win if -hx < margin < -hy
                    lo, hi = -x.hkey, -y.hkey
                else:                     # over at tx, under at ty: both win if tx < total < ty
                    lo, hi = x.hkey, y.hkey
                if hi <= lo:
                    continue
                ints = [k for k in range(math.floor(lo) + 1, math.ceil(hi)) if lo < k < hi]
                s = 1 / x.dec + 1 / y.dec
                if not (1.3 <= x.dec <= 4.0 and 1.3 <= y.dec <= 4.0):   # skip lottery-ticket alt lines
                    continue
                if ints and s <= MIDDLE_MAX_COST:
                    sig.append(dict(kind="middle", legs=[x, y], cost=s, margin=1 - s, gap=hi - lo, ints=ints))
    # ---- +EV vs de-vigged consensus of the OTHER venues at the exact same line
    if not F.empty:
        FI = {k: g for k, g in F.groupby(["espn_id", "market", "hkey"])}
        for r in N.itertuples():
            f = FI.get((r.espn_id, r.market, r.hkey))
            if f is None:
                continue
            f = f[f.venue != r.venue]
            if r.venue.startswith("DraftKings"):
                f = f[~f.venue.str.startswith("DraftKings")]
            need = 1 if r.market.startswith("prop:") and (f.venue == "Pinnacle").any() else MIN_CONSENSUS
            if len(f) < need:
                continue
            pa = float(np.mean(f.pa))
            p = pa if r.side == sides(r.market)[0] else 1 - pa
            ev = p * r.dec - 1
            if ev >= EV_MIN:
                sig.append(dict(kind="ev", legs=[pd.Series(r._asdict())], fair_p=p, ev=ev, n_cons=len(f),
                                cons_venues=",".join(sorted(f.venue)), pin=bool((f.venue == "Pinnacle").any())))
    # ---- verify exchange legs on the live book
    if verify:
        cache = {}
        for s in sig:
            for L in s["legs"]:
                if L["venue"] in EXCHANGES and L.get("ref"):
                    if L["ref"] not in cache:
                        cache[L["ref"]] = exchange_fill(L["venue"], L["ref"])
                    d, usd, top = cache[L["ref"]]
                    L["book_dec"], L["book_usd"] = d, usd
            decs = [L.get("book_dec", L["dec"]) if L["venue"] in EXCHANGES else L["dec"] for L in s["legs"]]
            s["verified"] = all(d == d for d in decs)
            if s["kind"] in ("arb", "middle"):
                s["cost_book"] = sum(1 / d for d in decs) if s["verified"] else np.nan
            else:
                s["ev_book"] = s["fair_p"] * decs[0] - 1 if s["verified"] else np.nan
    return sig


def _fmt_leg(L):
    if str(L["market"]).startswith("prop:"):
        _, stat, pl = L["market"].split(":", 2)
        name = L.get("player") if isinstance(L.get("player"), str) else pl
        return f"{name} {stat} {L['sel'].upper()} {L['line']:g} {L['am']:+.0f} @{L['venue']}"
    ln = "" if L["market"] == "ml" else (f" {L['line']:+g}" if L["market"] == "spread" else f" {L['line']:g}")
    who = L["sel"] if L["market"] != "total" else L["sel"].upper()
    return f"{who}{ln} {L['am']:+.0f} @{L['venue']}"


def _min_size(legs):
    """$ executable at the quoted price = smallest known depth/limit across legs (sportsbook limits unknown)."""
    v = [L.get("book_usd", L.get("size")) for L in legs]
    v = [float(x) for x in v if x is not None and x == x]
    return min(v) if v else np.nan


def to_rows(sig, scan_ts):
    rows = []
    for s in sig:
        L0 = s["legs"][0]
        base = dict(scan_ts=scan_ts, kind=s["kind"], sport=L0["sport"], espn_id=L0["espn_id"], event=L0["event"],
                    commence=pd.Timestamp(L0["commence"]).isoformat(), market=L0["market"])
        if s["kind"] == "ev":
            rows.append(dict(base, hline=L0["hline"], side=L0["side"], sel=L0["sel"], line=L0["line"], venue=L0["venue"],
                             odds=round(L0["am"], 1), dec=round(L0["dec"], 4), fair_p=round(s["fair_p"], 4),
                             edge=round(s["ev"], 4), edge_book=round(s.get("ev_book", np.nan), 4),
                             n_cons=s["n_cons"], cons_venues=s["cons_venues"], pin_in_cons=s["pin"], alt=L0["alt"],
                             size_usd=L0.get("book_usd", L0.get("size")), verified=s.get("verified"),
                             legs=_fmt_leg(L0)))
        else:
            rows.append(dict(base, hline=L0["hline"], side=f"{s['legs'][0]['side']}/{s['legs'][1]['side']}",
                             venue="+".join(L["venue"] for L in s["legs"]), edge=round(s["margin"], 4),
                             edge_book=round(1 - s.get("cost_book", np.nan), 4), verified=s.get("verified"),
                             gap=s.get("gap"), size_usd=_min_size(s["legs"]),
                             legs=" | ".join(_fmt_leg(L) for L in s["legs"])))
    return pd.DataFrame(rows)


def scan(sports=SPORTS, save=True, quiet=False):
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC")
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    E, Q, status, errors = gather(sports, now)
    N = normalize(E, Q)
    sig = find_signals(N)
    R = to_rows(sig, ts)
    venues = sorted(N.venue.unique()) if len(N) else []
    summ = dict(scan_ts=ts, n_events=len(E), n_quotes_raw=len(Q), n_quotes_matched=len(N),
                n_events_2plus_venues=int((N.groupby("espn_id").venue.nunique() >= 2).sum()) if len(N) else 0,
                venues="|".join(f"{v}:{(N.venue == v).sum()}" for v in venues),
                n_arb=int((R.kind == "arb").sum()) if len(R) else 0,
                n_arb_verified=int(((R.kind == "arb") & (R.edge_book > 0)).sum()) if len(R) else 0,
                n_middle=int((R.kind == "middle").sum()) if len(R) else 0,
                n_ev=int((R.kind == "ev").sum()) if len(R) else 0, secs=round(time.time() - t0, 1),
                errors=" ; ".join(errors)[:2000], feed_status=json.dumps(status),
                stale_dropped=json.dumps({f"{v}|{e}": why for (e, v), why in N.attrs.get("stale", {}).items()})
                if len(N) else "{}")
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        QUOTES.mkdir(parents=True, exist_ok=True)
        if len(R):
            R.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        pd.DataFrame([summ]).to_csv(SCANS, mode="a", header=not SCANS.exists(), index=False)
        if len(N):
            N.drop(columns=["ref"]).to_csv(QUOTES / f"quotes_{now:%Y%m%dT%H%M%S}.csv.gz", index=False)
    if not quiet:
        print(f"[{ts}] events={len(E)} quotes={len(Q)} matched={len(N)} venues: {summ['venues']}")
        if errors:
            print("  feed errors:", "; ".join(errors)[:600])
        for k in ("arb", "middle", "ev"):
            if len(R) and (R.kind == k).any():
                X = R[R.kind == k].sort_values("edge", ascending=False)
                print(f"  {k.upper()} x{len(X)}")
                for _, x in X.head(25).iterrows():
                    extra = f" fair={x.fair_p:.3f} n={x.n_cons} book_edge={x.edge_book}" if k == "ev" else \
                        f" book_margin={x.edge_book} size=${x.size_usd}"
                    print(f"    {x.sport} {x.event} {x.market}: {x.legs}  edge={x.edge:+.3f}{extra}")
    return E, N, R, summ


# ------------------------------------------------------------------ settlement
def _finals(sport, espn_ids):
    out = {}
    for eid in espn_ids:
        try:
            js = _get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/summary", {"event": eid})
            c = js["header"]["competitions"][0]
            if not c.get("status", {}).get("type", {}).get("completed"):
                continue
            d = {t["homeAway"]: t for t in c["competitors"]}
            if sport == "ufc":
                continue
            out[eid] = (float(d["home"]["score"]), float(d["away"]["score"]))
        except Exception:
            continue
    return out


PROP_BOX = {"passing_yards": ("passing", "YDS"), "passing_touchdowns": ("passing", "TD"),
            "passing_completions": ("passing", "C/ATT", 0), "passing_attempts": ("passing", "C/ATT", 1),
            "rushing_yards": ("rushing", "YDS"), "receptions": ("receiving", "REC"),
            "receiving_yards": ("receiving", "YDS"), "points": ("", "PTS"), "rebounds": ("", "REB"),
            "assists": ("", "AST")}


def _player_box(sport, eid):
    """{(norm(player), stat): value} from the ESPN summary box score of a completed game."""
    js = _get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/summary", {"event": eid})
    out = {}
    for team in (js.get("boxscore") or {}).get("players", []):
        for cat in team.get("statistics", []):
            labels = cat.get("labels") or cat.get("keys") or []
            for a in cat.get("athletes", []):
                nm = norm((a.get("athlete") or {}).get("displayName", ""))
                for stat, spec in PROP_BOX.items():
                    if spec[0] and spec[0] != cat.get("name"):
                        continue
                    if spec[1] not in labels:
                        continue
                    v = a.get("stats", [])[labels.index(spec[1])]
                    try:
                        v = float(str(v).split("/")[spec[2]]) if len(spec) == 3 else float(v)
                    except Exception:
                        continue
                    out[(nm, stat)] = v
    return out


def grade(market, side, hline, hs, as_):
    """+1 win, -1 loss, 0 push for one leg."""
    if market == "ml":
        if hs == as_:
            return 0
        return 1 if (hs > as_) == (side == "home") else -1
    if market == "spread":
        m = (hs - as_) + hline            # home margin with home line
        v = m if side == "home" else -m
    else:
        v = (hs + as_ - hline) if side == "over" else (hline - hs - as_)
    return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)


def settle(first_only=True):
    """Grade every logged +EV signal from ESPN finals; 1u flat stake at the logged (and book-verified) price.
    first_only: keep the first time each (event, market, line, side, venue) signal appeared (forward test)."""
    if not LOG.exists():
        print("no log")
        return pd.DataFrame()
    L = pd.read_csv(LOG, dtype={"espn_id": str})
    L = L[L.kind == "ev"].copy()
    if first_only:
        L = L.sort_values("scan_ts").drop_duplicates(["espn_id", "market", "hline", "side", "venue"])
    fin = {}
    for sp, g in L.groupby("sport"):
        fin.update(_finals(sp, g.espn_id.unique()))
    res, boxes = [], {}
    for r in L.itertuples():
        if r.espn_id not in fin:
            res.append(np.nan)
            continue
        hs, as_ = fin[r.espn_id]
        if str(r.market).startswith("prop:"):
            _, stat, pl = r.market.split(":", 2)
            if r.espn_id not in boxes:
                try:
                    boxes[r.espn_id] = _player_box(r.sport, r.espn_id)
                except Exception:
                    boxes[r.espn_id] = {}
            v = boxes[r.espn_id].get((pl, stat))
            if v is None:   # player did not record the stat line (DNP -> void at most books)
                res.append(np.nan)
                continue
            d = (v - r.hline) if r.side == "over" else (r.hline - v)
            res.append(0 if d == 0 else (1 if d > 0 else -1))
            continue
        res.append(grade(r.market, r.side, r.hline if r.market != "ml" else None, hs, as_))
    L["result"] = res
    L["profit"] = np.where(L.result == 1, L.dec - 1, np.where(L.result == -1, -1.0, np.where(L.result == 0, 0.0, np.nan)))
    L["clv_note"] = ""
    S = L.dropna(subset=["result"])
    L.to_csv(SETTLED, index=False)
    if len(S):
        n, pnl = len(S), S.profit.sum()
        sd = S.profit.std(ddof=1) if n > 1 else np.nan
        z = pnl / (sd * math.sqrt(n)) if sd and sd == sd else np.nan
        print(f"settled +EV signals: n={n} units={pnl:+.2f} ROI={pnl / n:+.3%} z={z:.2f} "
              f"(expected ROI from fair_p {S.edge.mean():+.3%}); pending={L.result.isna().sum()}")
        for k, g in S.groupby("sport"):
            print(f"  {k}: n={len(g)} ROI={g.profit.mean():+.3%}")
    else:
        print(f"nothing settled yet; pending={len(L)}")
    return L



# ------------------------------------------------------------------ historical: Polymarket vs sportsbook close
PMH_CACHE = OUT / "arb_scanner_pm_history.csv"
PMH_SERIES = {"nba": 10345, "nhl": 10346, "ncaam": 10470, "nfl": 12185, "ncaaf": 12756}


def pm_closed_games(sport, max_pages=250):
    out, off = [], 0
    for _ in range(max_pages):
        try:
            page = _get("https://gamma-api.polymarket.com/events", {"series_id": PMH_SERIES[sport], "closed": "true",
                                                                    "limit": 100, "offset": off})
        except Exception:
            break
        if not page:
            break
        for e in page:
            if not e.get("gameId"):
                continue
            ml = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
            if not ml:
                continue
            m = ml[0]
            try:
                outs, toks = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])
                fin = json.loads(m.get("outcomePrices") or "[]")
            except Exception:
                continue
            if len(outs) != 2 or len(toks) != 2:
                continue
            out.append(dict(sport=sport, title=e.get("title"), away=outs[0], home=outs[1], token_away=toks[0],
                            start=m.get("gameStartTime") or e.get("startTime"), fee=json.dumps(m.get("feeSchedule")),
                            final_away=float(fin[0]) if fin else np.nan, volume=float(m.get("volumeNum") or 0)))
        off += 100
    return out


def _pm_hist_price(tok, start):
    """Last traded/marked price of token in the 3h before start, and at T-60m (CLOB prices-history)."""
    t = int(pd.Timestamp(start).timestamp())
    try:
        h = _get("https://clob.polymarket.com/prices-history",
                 {"market": tok, "startTs": t - 3 * 3600, "endTs": t, "fidelity": 5}).get("history", [])
    except Exception:
        return np.nan, np.nan, 0
    h = [(x["t"], x["p"]) for x in h if x["t"] <= t]
    if not h:
        return np.nan, np.nan, 0
    p_close = h[-1][1]
    p60 = next((p for tt, p in reversed(h) if tt <= t - 3600), np.nan)
    return p_close, p60, len(h)


def pm_history(sports=("nba", "nhl", "ncaam", "nfl", "ncaaf"), refresh=False):
    if PMH_CACHE.exists() and not refresh:
        return pd.read_csv(PMH_CACHE)
    G = []
    with ThreadPoolExecutor(5) as ex:
        for r in ex.map(pm_closed_games, sports):
            G += r
    G = pd.DataFrame(G)
    G["start"] = pd.to_datetime(G.start, utc=True, errors="coerce")
    G = G.dropna(subset=["start"])
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda r: _pm_hist_price(r[0], r[1]), zip(G.token_away, G.start)))
    G["pm_away_close"], G["pm_away_60"], G["n_hist"] = zip(*res)
    G.to_csv(PMH_CACHE, index=False)
    return G


def _book_games(sport):
    if sport == "nfl":
        from backtest import run_team as rt
        B = rt.load("nfl")
    else:
        B = pd.read_csv(ROOT / "data" / "processed" / f"{sport}.csv", low_memory=False)
    B["date"] = pd.to_datetime(B.date)
    live = ROOT / "data" / "live" / f"{sport}.csv"
    names = {}
    if live.exists():
        L = pd.read_csv(live, low_memory=False)
        for a, b in zip(L.get("home", []), L.get("home_name", [])):
            names[a] = b
        L["date"] = pd.to_datetime(L.date)
        L = L[L.date > B.date.max()]
        if "home_ml" in L:
            B = pd.concat([B, L], ignore_index=True)
    if sport in ("nfl", "nhl", "mlb"):          # dataset abbreviations -> ESPN display names
        from nightly.live import ABBR
        try:
            js = _get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/teams", {"limit": 100})
            for t in js["sports"][0]["leagues"][0]["teams"]:
                t = t["team"]
                names[ABBR.get(sport, {}).get(t["abbreviation"], t["abbreviation"])] = t["displayName"]
        except Exception:
            pass
        names.update({"ARI": names.get("ARI", "Arizona Coyotes") if sport == "nhl" else names.get("ARI", "ARI"),
                      "LA": "Los Angeles Rams" if sport == "nfl" else names.get("LA", "LA")})
    B["home_full"] = B.home.map(lambda x: names.get(x, x))
    B["away_full"] = B.away.map(lambda x: names.get(x, x))
    return B.dropna(subset=["home_ml", "away_ml", "home_score", "away_score"])


def pm_join(G, sport):
    B = _book_games(sport)
    B = B[B.date >= G.start.min().tz_localize(None).normalize() - pd.Timedelta(days=1)]
    G = G[G.sport == sport].copy()
    G["date"] = G.start.dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    rows = []
    idx = {d: g for d, g in B.groupby("date")}
    for r in G.itertuples():
        cand = idx.get(r.date)
        if cand is None:
            continue
        hit = None
        for b in cand.itertuples():
            hf, af = norm(b.home_full), norm(b.away_full)
            hp, ap = norm(r.home), norm(r.away)
            ok = lambda full, short: bool(short) and bool(full) and (
                full == short or full.endswith(short) or full.startswith(short) or short.startswith(full))
            if ok(hf, hp) and ok(af, ap):
                hit = (b, False)
                break
            if ok(hf, ap) and ok(af, hp):
                hit = (b, True)
                break
        if hit is None:
            continue
        b, flip = hit
        p_away = r.pm_away_close if not flip else 1 - r.pm_away_close
        rows.append(dict(sport=sport, date=r.date, home=b.home, away=b.away, home_ml=b.home_ml, away_ml=b.away_ml,
                         home_score=b.home_score, away_score=b.away_score, pm_home=1 - p_away,
                         pm_home_60=(1 - r.pm_away_60) if not flip else r.pm_away_60, fee=r.fee, volume=r.volume,
                         pm_final_home=(1 - r.final_away) if not flip else r.final_away))
    return pd.DataFrame(rows)


def pm_backtest(refresh=False):
    """Protocol test: does Polymarket's pre-game price diverging from the sportsbook close predict results?
    Candidates (all logged): side = the one Polymarket rates higher than the de-vigged book close by >= thr;
    bet A) at the sportsbook closing price (real price), B) at Polymarket (history price + 1c + taker fee;
    approximate - history is last trade, not the ask). Discovery = first 60% of dates, lockbox = last 40%."""
    from scipy.stats import norm as N
    G = pm_history(refresh=refresh)
    G["start"] = pd.to_datetime(G.start, utc=True)
    J = pd.concat([pm_join(G, s) for s in G.sport.unique()], ignore_index=True)
    J = J[(J.pm_home > 0.02) & (J.pm_home < 0.98) & (J.home_score != J.away_score)]
    ov = 1 / am2dec_v(J.home_ml) + 1 / am2dec_v(J.away_ml)
    J = J[(ov >= 1.0) & (ov <= 1.12)]
    J["p_book"], _ = devig(J.home_ml.values, J.away_ml.values, "odds_ratio")
    J["d"] = J.pm_home - J.p_book
    J["y"] = (J.home_score > J.away_score).astype(int)
    J = J.sort_values("date").reset_index(drop=True)
    cut = J.date.quantile(0.6)
    J["lockbox"] = J.date > cut
    fs = lambda f: json.loads(f) if isinstance(f, str) and f not in ("null", "") else {}
    cands = []
    for sport in ["ALL"] + sorted(J.sport.unique()):
        for thr in (0.02, 0.03, 0.05, 0.08):
            for venue in ("book", "pm"):
                cands.append((sport, thr, venue))
    rows, bets_all = [], {}
    for sport, thr, venue in cands:
        X = J if sport == "ALL" else J[J.sport == sport]
        X = X[X.d.abs() >= thr].copy()
        home = X.d > 0 if venue == "book" else X.d < 0      # pm venue: buy on PM the side PM rates LOWER
        X["home_side"] = home
        if venue == "book":
            X["dec"] = np.where(home, am2dec_v(X.home_ml), am2dec_v(X.away_ml))
        else:
            p = np.where(home, X.pm_home, 1 - X.pm_home) + 0.01
            fee = np.array([pm_fee(pp, fs(f)) for pp, f in zip(p, X.fee)])
            X["dec"] = 1 / (p + fee)
        X["win"] = np.where(home, X.y == 1, X.y == 0)
        X["pnl"] = np.where(X.win, X.dec - 1, -1.0)
        bets_all[(sport, thr, venue)] = X
        for samp, M in (("discovery", ~X.lockbox), ("lockbox", X.lockbox)):
            Z = X[M]
            n = len(Z)
            if n == 0:
                rows.append(dict(sport=sport, thr=thr, venue=venue, sample=samp, n=0))
                continue
            roi = Z.pnl.mean()
            sd = Z.pnl.std(ddof=1) if n > 1 else np.nan
            z = roi / (sd / math.sqrt(n)) if sd and sd == sd and sd > 0 else np.nan
            eq = Z.pnl.cumsum()
            rows.append(dict(sport=sport, thr=thr, venue=venue, sample=samp, n=n, roi=roi, units=Z.pnl.sum(), z=z,
                             p=1 - N.cdf(z) if z == z else np.nan, maxdd=float((eq.cummax() - eq).max()),
                             months_pos=float((Z.groupby(Z.date.dt.to_period("M")).pnl.sum() > 0).mean())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "arb_scanner_candidates.csv", index=False)
    J.to_csv(OUT / "arb_scanner_pm_joined.csv", index=False)
    return J, R, bets_all


def am2dec_v(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "scan":
        scan()
    elif cmd == "loop":
        n, gap = int(sys.argv[2]), int(sys.argv[3])
        for i in range(n):
            scan()
            if i < n - 1:
                time.sleep(gap)
    elif cmd == "settle":
        settle()
    elif cmd == "pmhist":
        J, R, _ = pm_backtest(refresh="refresh" in sys.argv)
        pd.set_option("display.width", 220)
        print(f"joined games: {len(J)} by sport {J.groupby('sport').size().to_dict()}  dates {J.date.min()} .. {J.date.max()}")
        print(R.round(4).to_string())
