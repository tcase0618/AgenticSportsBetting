"""
Free, key-less price sources for the live engine, normalized to American odds (+150 / -120).

  FanDuel     public web API (the JSON the FanDuel site itself loads; no account or key needed beyond the
              public `_ak` web key). State subdomain via env FANDUEL_STATE (default "tn").
  Polymarket  public Gamma API. Prices are share prices (0-1); we quote the cost to BUY (best ask) plus
              Polymarket's taker fee from each market's own feeSchedule, converted to American odds.
              fee/share = p * rate * (p * (1 - p)) ** exponent   (docs.polymarket.com/trading/fees)

Both return the same board schema as nightly.live.odds_board:
  home_name, away_name, commence, home_ml, away_ml, spread_home, spread_home_odds, spread_away_odds,
  total, over_odds, under_odds, source, plus NBA points-prop quotes via *_nba_points().
Note: FanDuel's terms restrict automated access to its site; this reads the same public JSON a browser
does at low volume (one request per sport per run). Use at your own discretion.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from backtest.core import american_to_decimal, decimal_to_american

FD_STATE = os.environ.get("FANDUEL_STATE", "tn")
FD = f"https://sbapi.{FD_STATE}.sportsbook.fanduel.com/api"
FD_AK = "FhMFpcPWXMeyZxOx"
FD_PAGE = {"nfl": "nfl", "ncaaf": "ncaaf", "nba": "nba", "ncaam": "ncaab", "mlb": "mlb", "nhl": "nhl"}
FD_MMA_EVENT_TYPE = 26420387
PM = "https://gamma-api.polymarket.com"
PM_SERIES = {"nba": 10345, "nfl": 12185, "ncaaf": 12756, "nhl": 10346, "mlb": 3, "ufc": 38, "ncaam": 10470}
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _get(url, params=None, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 401, 403, 404):
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _clean(name: str) -> str:
    """'Cleveland Guardians (G Williams)' -> 'Cleveland Guardians' (FanDuel MLB lists probable pitchers)."""
    return re.sub(r"\s*\(.*?\)\s*$", "", str(name)).strip()


def _am(r):
    return float(r["winRunnerOdds"]["americanDisplayOdds"]["americanOdds"])


# ------------------------------------------------------------------ FanDuel
def fanduel_board(sport: str) -> pd.DataFrame:
    if sport == "ufc":
        js = _get(f"{FD}/content-managed-page", {"page": "SPORT", "eventTypeId": FD_MMA_EVENT_TYPE, "_ak": FD_AK,
                                                 "timezone": "America/New_York"})
    else:
        js = _get(f"{FD}/content-managed-page", {"page": "CUSTOM", "customPageId": FD_PAGE[sport], "pbHorizontal": "false",
                                                 "_ak": FD_AK, "timezone": "America/New_York"})
    att = js.get("attachments", {})
    events, markets = att.get("events", {}), att.get("markets", {})
    rows = {}
    for m in markets.values():
        if m.get("marketStatus") != "OPEN" or m.get("inPlay"):
            continue
        ev = events.get(str(m["eventId"]))
        if not ev:
            continue
        mt, R = m.get("marketType"), m.get("runners", [])
        if sport == "ufc":
            if mt != "MATCH_BETTING" or len(R) != 2:
                continue
            # fighters have no home/away; keep listing order (first runner = "home" slot)
            row = rows.setdefault(ev["eventId"], dict(home_name=R[0]["runnerName"], away_name=R[1]["runnerName"],
                                                      commence=ev["openDate"], fd_event_id=ev["eventId"]))
            row["home_ml"], row["away_ml"] = _am(R[0]), _am(R[1])
            continue
        if "@" not in ev.get("name", ""):
            continue
        away, home = [_clean(s) for s in ev["name"].split("@", 1)]
        row = rows.setdefault(ev["eventId"], dict(home_name=home, away_name=away, commence=ev["openDate"],
                                                  fd_event_id=ev["eventId"]))
        side = {r.get("result", {}).get("type"): r for r in R}
        if mt == "MONEY_LINE" and {"HOME", "AWAY"} <= set(side):
            row["home_ml"], row["away_ml"] = _am(side["HOME"]), _am(side["AWAY"])
        elif mt == "MATCH_HANDICAP_(2-WAY)" and {"HOME", "AWAY"} <= set(side):
            row["spread_home"] = float(side["HOME"]["handicap"])
            row["spread_home_odds"], row["spread_away_odds"] = _am(side["HOME"]), _am(side["AWAY"])
        elif mt == "TOTAL_POINTS_(OVER/UNDER)":
            o = {r["runnerName"].split()[-1].lower(): r for r in R}
            if "over" in o and "under" in o:
                row["total"] = float(o["over"]["handicap"])
                row["over_odds"], row["under_odds"] = _am(o["over"]), _am(o["under"])
    df = pd.DataFrame(rows.values())
    if len(df):
        df["source"] = "FanDuel"
    return df


def fanduel_nba_points(board: pd.DataFrame) -> pd.DataFrame:
    """Player points O/U for each NBA game on the board (event page, 'player-points' tab)."""
    out = []
    for _, g in board.iterrows():
        try:
            js = _get(f"{FD}/event-page", {"_ak": FD_AK, "eventId": int(g.fd_event_id), "tab": "player-points"})
        except Exception:
            continue
        for m in js.get("attachments", {}).get("markets", {}).values():
            if not re.fullmatch(r"PLAYER_[A-Z]_TOTAL_POINTS", str(m.get("marketType"))) or m.get("marketStatus") != "OPEN":
                continue
            o = {r["runnerName"].rsplit(" ", 1)[-1].lower(): r for r in m.get("runners", [])}
            if "over" in o and "under" in o:
                out.append(dict(player=m["marketName"].split(" - ")[0].strip(), line=float(o["over"]["handicap"]),
                                over_odds=_am(o["over"]), under_odds=_am(o["under"]), book="fanduel",
                                home_team=g.home_name, away_team=g.away_name, commence=g.commence))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ Polymarket
def pm_cost(p, fee_schedule) -> float:
    """All-in cost per $1-payout share when buying at price p (taker)."""
    if p is None or not (0 < p < 1):
        return np.nan
    fs = fee_schedule or {}
    rate, ex = float(fs.get("rate", 0) or 0), float(fs.get("exponent", 1) or 1)
    return p + p * rate * (p * (1 - p)) ** ex


def pm_american(p, fee_schedule) -> float:
    c = pm_cost(p, fee_schedule)
    return float(decimal_to_american(1.0 / c)) if c == c and c < 1 else np.nan


PM_MAX_SPREAD = 0.05  # skip thin books: a 3c bid / 97c ask is not a real price


def _pm_buy_prices(m):
    """(cost to buy outcome[0], cost to buy outcome[1]) from the order book top; NaN if the book is thin."""
    ask, bid = m.get("bestAsk"), m.get("bestBid")
    if ask in (None, 0) or bid in (None, 0) or float(ask) - float(bid) > PM_MAX_SPREAD:
        return np.nan, np.nan
    a0 = float(ask) if ask not in (None, 0) else np.nan
    a1 = 1 - float(bid) if bid not in (None, 0) else np.nan
    return a0, a1


def _pm_mid(m):
    try:
        return float(json.loads(m["outcomePrices"])[0])
    except Exception:
        return np.nan


def polymarket_events(sport: str) -> list:
    sid = PM_SERIES.get(sport)
    if not sid:
        return []
    return [e for e in _get(f"{PM}/events", {"series_id": sid, "active": "true", "closed": "false", "limit": 200})
            if e.get("gameId")]


def polymarket_board(sport: str) -> pd.DataFrame:
    rows = []
    for e in polymarket_events(sport):
        title = e.get("title", "")
        mk = [m for m in e.get("markets", []) if m.get("active") and not m.get("closed") and m.get("acceptingOrders")]
        ml = [m for m in mk if m.get("sportsMarketType") == "moneyline"]
        if not ml:
            continue
        m0 = ml[0]
        if np.isnan(_pm_buy_prices(m0)[0]):
            continue
        teams = json.loads(m0["outcomes"])
        if len(teams) != 2:
            continue
        # Polymarket titles are "Away vs. Home"
        away, home = teams[0], teams[1]
        fs = m0.get("feeSchedule")
        b0, b1 = _pm_buy_prices(m0)
        row = dict(home_name=home, away_name=away, commence=m0.get("gameStartTime") or e.get("endDate"),
                   home_ml=pm_american(b1, fs), away_ml=pm_american(b0, fs), pm_slug=e.get("slug"),
                   pm_home_mid=1 - _pm_mid(m0), source="Polymarket")
        # main spread / total = the line whose price is closest to 50/50 (Polymarket lists alternates)
        sp = [m for m in mk if m.get("sportsMarketType") == "spreads" and m.get("line") is not None]
        if sp:
            m = min(sp, key=lambda x: abs(_pm_mid(x) - 0.5))
            fav = json.loads(m["outcomes"])[0]                    # "Spread: Knicks (-5.5)" -> outcome[0] is the favorite
            s0, s1 = _pm_buy_prices(m)
            line = float(m["line"])
            if fav == home:
                row.update(spread_home=line, spread_home_odds=pm_american(s0, m.get("feeSchedule")),
                           spread_away_odds=pm_american(s1, m.get("feeSchedule")))
            else:
                row.update(spread_home=-line, spread_home_odds=pm_american(s1, m.get("feeSchedule")),
                           spread_away_odds=pm_american(s0, m.get("feeSchedule")))
        tt = [m for m in mk if m.get("sportsMarketType") == "totals" and m.get("line") is not None
              and "Over" in str(m.get("outcomes"))]
        if tt:
            m = min(tt, key=lambda x: abs(_pm_mid(x) - 0.5))
            o0, o1 = _pm_buy_prices(m)
            row.update(total=float(m["line"]), over_odds=pm_american(o0, m.get("feeSchedule")),
                       under_odds=pm_american(o1, m.get("feeSchedule")))
        rows.append(row)
    return pd.DataFrame(rows)


def polymarket_nba_points() -> pd.DataFrame:
    """Polymarket player points markets ('<Player>: Points O/U 24.5', outcomes Over/Under)."""
    out = []
    for e in polymarket_events("nba"):
        for m in e.get("markets", []):
            if m.get("sportsMarketType") != "points" or m.get("line") is None or not m.get("acceptingOrders"):
                continue
            q = m.get("question", "")
            player = re.split(r":| - ", q)[0].strip()
            o0, o1 = _pm_buy_prices(m)
            outs = json.loads(m["outcomes"])
            over_first = outs[0].lower().startswith("over")
            po, pu = (o0, o1) if over_first else (o1, o0)
            out.append(dict(player=player, line=float(m["line"]), over_odds=pm_american(po, m.get("feeSchedule")),
                            under_odds=pm_american(pu, m.get("feeSchedule")), book="polymarket",
                            commence=m.get("gameStartTime"), question=q))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ combine
def _nick(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _same_team(full: str, short: str) -> bool:
    f, s = _nick(full), _nick(short)
    return bool(s) and (f == s or f.endswith(s) or s.endswith(f) or s in f)


def combined_board(sport: str) -> pd.DataFrame:
    """FanDuel lines (named like ESPN) + Polymarket matched by teams & start time.
    consensus columns (home_ml/... ) = FanDuel when present else Polymarket; `best` = best price per side
    across both, with its source, in American odds."""
    fd, pm, errs = pd.DataFrame(), pd.DataFrame(), []
    try:
        fd = fanduel_board(sport)
    except Exception as e:
        errs.append(f"FanDuel {sport}: {e}")
    try:
        pm = polymarket_board(sport)
    except Exception as e:
        errs.append(f"Polymarket {sport}: {e}")
    if fd.empty and pm.empty:
        out = pd.DataFrame()
        out.attrs["errors"] = errs
        return out
    base = fd.copy() if len(fd) else pm.copy()
    cols = ["home_ml", "away_ml", "spread_home_odds", "spread_away_odds", "over_odds", "under_odds"]
    bests = []
    for i, r in base.iterrows():
        cand = {c: [(r.get(c), r.get("source"))] for c in cols}
        if len(fd) and len(pm):
            t = pd.to_datetime(r["commence"], utc=True)
            for _, q in pm.iterrows():
                if abs((pd.to_datetime(q["commence"], utc=True) - t).total_seconds()) > 6 * 3600:
                    continue
                if _same_team(r.home_name, q.home_name) and _same_team(r.away_name, q.away_name):
                    for c in cols:
                        same_line = (c.startswith("spread") and q.get("spread_home") == r.get("spread_home")) or \
                                    (c in ("over_odds", "under_odds") and q.get("total") == r.get("total")) or \
                                    c.endswith("_ml")
                        if same_line:
                            cand[c].append((q.get(c), "Polymarket"))
                    base.loc[i, "pm_slug"] = q.get("pm_slug")
                    break
        best = {}
        for c, vs in cand.items():
            vs = [(float(v), s) for v, s in vs if v is not None and v == v]
            if vs:
                v, s = max(vs, key=lambda x: float(american_to_decimal(x[0])))
                best[c.replace("_odds", "")] = [round(v), s]
        bests.append(json.dumps(best))
    base["best"] = bests
    base.attrs["errors"] = errs
    return base
