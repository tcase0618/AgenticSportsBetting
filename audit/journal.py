"""
Append-only bet journal.

Storage: data/journal/events.csv — one row per EVENT (place / close / settle), never rewritten.
The current state of every bet is the fold of its events (last non-null value per field wins),
so the full audit trail (what we knew, when) is always recoverable.  `snapshot()` materialises
the folded view to data/journal/journal.parquet (or .csv) for convenience; it is derived data.

Schema of the folded journal (COLUMNS):
  bet_id, placed_at, sport, strategy, segment, event, market, pick, line, price_taken, book,
  model_prob, market_prob_at_bet (de-vigged), stake, closing_line, closing_price, closing_prob,
  clv_prob (closing_prob - market_prob_at_bet, probability points), result (+1/0/-1), pnl, settled_at

Typical nightly use
  rows = journal_from_card_legs(legs, placed_at=now)   # legs from nightly/card.py
  record(rows)
  ... at tip-off:  attach_close(bet_id, closing_price=..., closing_line=..., closing_opp_price=...)
  ... next day:    settle(bet_id, result=+1/0/-1)
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.core import american_to_decimal, american_to_prob

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DIR = ROOT / "data" / "journal"
EVENTS = "events.csv"

COLUMNS = ["bet_id", "placed_at", "sport", "strategy", "segment", "event", "market", "pick", "line",
           "price_taken", "book", "model_prob", "market_prob_at_bet", "stake", "closing_line",
           "closing_price", "closing_prob", "clv_prob", "result", "pnl", "settled_at"]
EVENT_COLS = ["event_type", "event_ts"] + COLUMNS

# Card tier -> stake (mirrors nightly/card.py STAKE; duplicated so importing this never runs card.py)
TIER_STAKE = {"A": 1.0, "B": 0.75, "C": 0.5, "prop": 0.5, "parlay2": 0.25, "parlay3": 0.15}

# Typical two-way overround when the opposite price is unknown (multiplicative de-vig).
DEFAULT_OVERROUND = {"ats": 1.0476, "tot": 1.0476, "ml": 1.04, "prop": 1.07}

# Std-dev of (outcome - closing line) in points, used to convert line moves into probability
# points: dP ~= phi(0)/sigma * d(points).  Empirical sd of (result - closing line) in data/processed & data/raw/nfl_games.csv.
LINE_SIGMA = {("nba", "tot"): 17.8, ("nba", "ats"): 12.7, ("ncaaf", "ats"): 15.4, ("ncaaf", "tot"): 16.3,
              ("nfl", "ats"): 13.2, ("nfl", "tot"): 13.4, ("ncaam", "ats"): 11.0, ("ncaam", "tot"): 16.0,
              ("nba-prop", "prop"): 7.0, ("mlb", "tot"): 4.4, ("nhl", "tot"): 2.3}
PHI0 = 0.3989422804014327


# ------------------------------------------------------------------ helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def devig_prob(price, opp_price=None, market: str = "ml") -> float:
    """De-vigged probability of OUR side. Two-way multiplicative de-vig when the opposite
    price is known, otherwise divide by a typical overround for the market."""
    if price is None or pd.isna(price):
        return np.nan
    p = float(american_to_prob(price))
    if opp_price is not None and not pd.isna(opp_price):
        q = float(american_to_prob(opp_price))
        return p / (p + q)
    return p / DEFAULT_OVERROUND.get(market, 1.045)


def points_to_prob(sport: str, market: str, pts: float) -> float:
    """Probability-point value of `pts` points of line (normal approximation near the mean)."""
    sig = LINE_SIGMA.get((sport, market))
    if sig is None or pts is None or pd.isna(pts):
        return 0.0
    return PHI0 / sig * float(pts)


def line_advantage(pick: str, market: str, line_taken, closing_line) -> float:
    """Points by which OUR line beats the closing line (positive = we got the better number).
    Lines are expressed from the picked side (spread '+4.5' for the dog, total for O/U)."""
    if line_taken is None or closing_line is None or pd.isna(line_taken) or pd.isna(closing_line):
        return 0.0
    if market == "tot" or (market == "prop" and re.search(r"\b(OVER|UNDER)\b", str(pick).upper())):
        over = "OVER" in str(pick).upper()
        return (closing_line - line_taken) if over else (line_taken - closing_line)
    # spreads (picked-side line, higher is better for us)
    return float(line_taken) - float(closing_line)


def parse_line(pick: str, market: str):
    """Extract the numeric line from a card pick string ('MIN +4.5', 'UNDER 39.0', 'X OVER 15.5 pts')."""
    if market == "ml":
        return np.nan
    s = str(pick)
    m = re.search(r"(?:OVER|UNDER)\s+([0-9.]+)", s, re.I) if market in ("tot", "prop") else None
    if m is None:
        m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:pts)?\s*(?:@.*)?$", s)
    return float(m.group(1)) if m else np.nan


def price_bucket(price) -> str:
    p = float(price)
    if p <= -200:
        return "heavy_fav"
    if p < -120:
        return "fav"
    if p <= 120:
        return "pickem"
    return "dog"


def default_segment(sport: str, market: str, pick: str, price) -> str:
    side = ""
    up = str(pick).upper()
    if "OVER" in up:
        side = "over"
    elif "UNDER" in up:
        side = "under"
    elif market == "ats":
        side = "dog" if re.search(r"\+\d", up) else "fav"
    elif market == "ml":
        side = "dog" if float(price) > 0 else "fav"
    return f"{sport}|{market}|{side}|{price_bucket(price)}"


def make_bet_id(placed_at, sport, strategy, event, pick, book="") -> str:
    day = str(pd.Timestamp(placed_at).date())
    key = "|".join(str(x) for x in (day, sport, strategy, event, pick, book))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ storage
def _path(journal_dir: Path | None) -> Path:
    d = Path(journal_dir) if journal_dir else JOURNAL_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / EVENTS


def _append(rows: list[dict], journal_dir: Path | None) -> None:
    if not rows:
        return
    fp = _path(journal_dir)
    df = pd.DataFrame(rows).reindex(columns=EVENT_COLS)
    df.to_csv(fp, mode="a", header=not fp.exists(), index=False)


def events(journal_dir: Path | None = None) -> pd.DataFrame:
    fp = _path(journal_dir)
    if not fp.exists():
        return pd.DataFrame(columns=EVENT_COLS)
    return pd.read_csv(fp, low_memory=False)


def load(journal_dir: Path | None = None) -> pd.DataFrame:
    """Folded current state: one row per bet_id (last non-null value of each field)."""
    ev = events(journal_dir)
    if ev.empty:
        return pd.DataFrame(columns=COLUMNS)
    folded = ev.groupby("bet_id", sort=False)[COLUMNS[1:]].last().reset_index()
    return folded.reindex(columns=COLUMNS)


def snapshot(journal_dir: Path | None = None) -> Path:
    j = load(journal_dir)
    d = _path(journal_dir).parent
    try:
        fp = d / "journal.parquet"
        j.to_parquet(fp, index=False)
    except Exception:  # no pyarrow/fastparquet
        fp = d / "journal.csv"
        j.to_csv(fp, index=False)
    return fp


# ------------------------------------------------------------------ API
def record(rows: list[dict] | pd.DataFrame, journal_dir: Path | None = None) -> list[str]:
    """Append 'place' events. Idempotent on bet_id (already-journaled bets are skipped).
    Returns bet_ids written."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    have = set(events(journal_dir).query("event_type == 'place'")["bet_id"]) if _path(journal_dir).exists() else set()
    out, ids = [], []
    ts = _now()
    for r in rows:
        r = dict(r)
        r.setdefault("placed_at", ts)
        r.setdefault("market", "ml")
        r.setdefault("book", "")
        r.setdefault("line", parse_line(r.get("pick", ""), r["market"]))
        r.setdefault("segment", default_segment(r.get("sport", ""), r["market"], r.get("pick", ""), r["price_taken"]))
        if pd.isna(r.get("market_prob_at_bet", np.nan)):
            r["market_prob_at_bet"] = devig_prob(r["price_taken"], r.pop("opp_price", None), r["market"])
        r.setdefault("bet_id", make_bet_id(r["placed_at"], r.get("sport"), r.get("strategy"), r.get("event"),
                                           r.get("pick"), r.get("book", "")))
        if r["bet_id"] in have:
            continue
        have.add(r["bet_id"])
        out.append(dict({k: r.get(k, np.nan) for k in COLUMNS}, event_type="place", event_ts=ts))
        ids.append(r["bet_id"])
    _append(out, journal_dir)
    return ids


def attach_close(bet_id: str, closing_price=None, closing_line=None, closing_opp_price=None,
                 closing_prob=None, journal_dir: Path | None = None) -> dict:
    """Record the closing market for a bet and its CLV in probability points.
    closing_prob (de-vigged, for OUR side at the CLOSING line) can be passed directly; otherwise it is
    computed from closing_price (+ closing_opp_price).  If the closing line differs from the line we
    took, the line difference is converted to probability with a normal approximation, so clv_prob
    measures 'fair probability of OUR exact bet at close' minus 'fair probability when we placed it'."""
    j = load(journal_dir).set_index("bet_id")
    if bet_id not in j.index:
        raise KeyError(bet_id)
    b = j.loc[bet_id]
    market = b["market"]
    if closing_prob is None:
        cp = closing_price if closing_price is not None else b["price_taken"]
        closing_prob = devig_prob(cp, closing_opp_price, market)
    adv = line_advantage(b["pick"], market, b["line"], closing_line)
    closing_prob_ours = float(closing_prob) + points_to_prob(str(b["sport"]), market, adv)
    clv = closing_prob_ours - float(b["market_prob_at_bet"])
    row = dict(bet_id=bet_id, closing_line=closing_line, closing_price=closing_price,
               closing_prob=closing_prob_ours, clv_prob=clv, event_type="close", event_ts=_now())
    _append([row], journal_dir)
    return row


def settle(bet_id: str, result: int, settled_at=None, journal_dir: Path | None = None) -> dict:
    """result: +1 win, 0 push/void, -1 loss.  pnl in units = stake * profit."""
    j = load(journal_dir).set_index("bet_id")
    if bet_id not in j.index:
        raise KeyError(bet_id)
    b = j.loc[bet_id]
    stake = float(b["stake"]) if pd.notna(b["stake"]) else 0.0
    prof = float(american_to_decimal(b["price_taken"])) - 1.0
    pnl = stake * (prof if result > 0 else (-1.0 if result < 0 else 0.0))
    row = dict(bet_id=bet_id, result=int(result), pnl=pnl, settled_at=settled_at or _now(),
               event_type="settle", event_ts=_now())
    _append([row], journal_dir)
    return row


def journal_from_card_legs(legs: list[dict], placed_at, stake_override: dict | None = None) -> list[dict]:
    """Convert nightly/card.py leg dicts (sport, strategy, tier, event, market, pick, price, [book, ev,
    model_prob, opp_price, stake]) into journal rows ready for record().  Private keys (_hs, _win...)
    are ignored.  model_prob: taken from the leg if present, else implied by its EV ((1+ev)/dec)."""
    rows = []
    for L in legs:
        market = L.get("market", "ml")
        price = float(L["price"])
        dec = float(american_to_decimal(price))
        mp = L.get("model_prob", L.get("p"))
        if mp is None and L.get("ev") is not None:
            mp = (1.0 + float(L["ev"])) / dec
        tier = L.get("tier", "C")
        stake = L.get("stake", (stake_override or TIER_STAKE).get(tier, 0.5))
        r = dict(placed_at=str(pd.Timestamp(placed_at)), sport=L["sport"], strategy=L["strategy"],
                 event=L["event"], market=market, pick=L["pick"], line=parse_line(L["pick"], market),
                 price_taken=price, book=L.get("book", ""), model_prob=mp if mp is not None else np.nan,
                 market_prob_at_bet=devig_prob(price, L.get("opp_price"), market), stake=float(stake))
        r["segment"] = L.get("segment") or default_segment(r["sport"], market, r["pick"], price)
        r["bet_id"] = make_bet_id(r["placed_at"], r["sport"], r["strategy"], r["event"], r["pick"], r["book"])
        rows.append(r)
    return rows
