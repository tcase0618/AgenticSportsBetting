"""
Nightly bet card: every qualifying bet for tonight from the 7 keys (+ validated extras),
NBA points-prop legs, and parlay tickets — with stakes.

  python -m nightly.card --date 2026-01-17 --replay     # rebuild a past night using only info known that morning
  python -m nightly.card --live                          # tonight (needs network; see nightly/live.py)

Replay mode hides every result on the target date, rebuilds all features, applies the rules, and
then grades the card against what actually happened. That's the proof the nightly logic matches
the backtest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np

from backtest.core import decimal_to_american, fmt_american
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import run_team as rt
from backtest.core import american_to_decimal
from backtest.features import build
from backtest.keys import KEYS
from backtest.ledger import EXTRA_LEGS
from backtest.strategies import side_masks, total_mask

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CARDS = RES / "cards"

TIER = {"ufc": "A", "nfl": "A", "ncaaf": "B", "nhl": "B", "nba": "B", "ncaam": "C", "mlb": "C"}
EXTRA_TIER = {"Bounce-back dogs ATS": "A"}
# stake in units (1u = 1% of bankroll). Fractional-Kelly-style: size ~ edge confidence.
STAKE = {"A": 1.0, "B": 0.75, "C": 0.5, "prop": 0.5, "parlay2": 0.25, "parlay3": 0.15}
PROP_EV_MIN = 0.04       # chosen on 2024-25 discovery season, validated on 2025-26
PROP_PARLAY_WINDOW = (-135, 115)

PRICE_COL = {"ats": ("spread_home_odds", "spread_away_odds"), "ml": ("home_ml", "away_ml"),
             "tot": ("over_odds", "under_odds")}


# ------------------------------------------------------------------ game-level keys
def _specs():
    out = [(sp, d["name"], tuple(d["key"]), TIER[sp]) for sp, d in KEYS.items()]
    out += [(sp, n, k, EXTRA_TIER.get(n, TIER[sp])) for sp, n, k in EXTRA_LEGS]
    return out


def team_legs(sport: str, date: pd.Timestamp, history: pd.DataFrame | None = None) -> list[dict]:
    """Qualifying legs on `date` for every key in `sport`. history: full game table incl. the
    target date (scores on/after date are hidden here)."""
    raw = history if history is not None else rt.load(sport)
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw.date <= date]
    today = raw.date == date
    if not today.any():
        return []
    truth = raw.loc[today, ["home", "away", "home_score", "away_score"]].copy()
    raw.loc[today, ["home_score", "away_score"]] = np.nan          # nothing known about tonight
    g = build(raw, rt.ELO.get(sport), keep_upcoming=True)
    meta = json.loads((RES / f"{sport}_meta.json").read_text())
    S = meta["S"]
    tq = g.loc[g.season < meta["split"], "total"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    G = g[g.date == date]
    legs = []
    for sp, name, key, tier in _specs():
        if sp != sport:
            continue
        mkt = key[0]
        if mkt == "tot":
            m = total_mask(g, S, sport, tq, key[1]).loc[G.index]
            sides = [(i, key[3] == "over") for i in G.index[m]]
        else:
            mh, ma = side_masks(g, S, sport, key[1], key[2], key[3], key[4])
            ok = G[PRICE_COL[mkt][0]].notna() if mkt != "ats" else G["spread_home"].notna()
            sides = [(i, True) for i in G.index[mh.loc[G.index] & ok]] + \
                    [(i, False) for i in G.index[ma.loc[G.index] & ok]]
        for i, home_side in sides:
            r = G.loc[i]
            hc, ac = PRICE_COL[mkt]
            price = r[hc] if home_side else r[ac]
            if mkt == "tot":
                pick = f"{'OVER' if home_side else 'UNDER'} {r['total']:g}"
            elif mkt == "ats":
                line = r["spread_home"] if home_side else -r["spread_home"]
                pick = f"{r['home'] if home_side else r['away']} {line:+g}"
            else:
                pick = f"{r['home'] if home_side else r['away']} ML"
            p_elo_side = r["p_home_elo"] if home_side else 1 - r["p_home_elo"]
            p_mkt_side = r["p_home_mkt"] if home_side else 1 - r["p_home_mkt"]
            elo_agree = bool(p_elo_side >= p_mkt_side) if pd.notna(p_elo_side) and pd.notna(p_mkt_side) else None
            legs.append(dict(sport=sport, strategy=name, tier=tier, event=f"{r['away']} @ {r['home']}", _elo_agree=elo_agree,
                             market=mkt, pick=pick, price=float(price) if pd.notna(price) else -110.0,
                             _home=r["home"], _away=r["away"], _home_name=r.get("home_name"), _away_name=r.get("away_name"),
                             _home_side=home_side, _line=r.get("spread_home"),
                             _total=r.get("total")))
    # attach truth for replay grading
    for L in legs:
        t = truth[(truth.home == L["_home"]) & (truth.away == L["_away"])]
        L["_hs"], L["_as"] = (float(t.home_score.iloc[0]), float(t.away_score.iloc[0])) if len(t) else (np.nan, np.nan)
    return legs


def ufc_legs(date: pd.Timestamp) -> list[dict]:
    from backtest.run_ufc import prep, corner_view, conditions, phases
    f = prep(pd.read_csv(ROOT / "data" / "processed" / "ufc.csv"))
    f = f[(f.date == date) & f.a_ml.notna() & f.b_ml.notna()]
    if f.empty:
        return []
    key = KEYS["ufc"]["key"]
    va, vb = corner_view(f, "a"), corner_view(f, "b")
    cf = conditions()[key[1]]
    ca, cb = cf(va).fillna(False), cf(vb).fillna(False)
    ca, cb = ca & ~cb, cb & ~ca
    ph = phases(f)[key[4]]
    legs = []
    for side, c, v in (("a", ca, va), ("b", cb, vb)):
        m = c & v.fav & ph
        for i in f.index[m]:
            r = f.loc[i]
            name = r[f"fighter_{side}"]
            legs.append(dict(sport="ufc", strategy=KEYS["ufc"]["name"], tier="A",
                             event=f"{r.fighter_a} vs {r.fighter_b}", market="ml", pick=f"{name} ML",
                             price=float(r[f"{side}_ml"]), _winner=r.winner, _side=side.upper()))
    return legs


def nba_prop_legs(date: pd.Timestamp) -> list[dict]:
    """Replay: pre-computed walk-forward frame (projection + leave-one-book-out consensus)."""
    from backtest.props_nba import pick_bets
    fp = RES / "nba_props_frame.csv.gz"
    if not fp.exists():
        return []
    p = pd.read_csv(fp, parse_dates=["date"], low_memory=False)
    p = p[p.date == date]
    if p.empty:
        return []
    p["p_blend"] = 0.75 * p.p_mkt + 0.25 * p.p_mod
    B = pick_bets(p, "p_blend", PROP_EV_MIN)
    legs = []
    for _, r in B.iterrows():
        legs.append(dict(sport="nba-prop", strategy="NBA points: blend EV", tier="prop",
                         event=f"{r.team} vs {r.opp}", market="prop",
                         pick=f"{r.player} {r.side.upper()} {r.line:g} pts @{r.book}", price=float(r.price),
                         ev=float(r.ev), main=bool(r.alt_line == 0), book=r.book, game_id=r.game_id,
                         _win=bool(r.win)))
    return legs


# ------------------------------------------------------------------ grading (replay)
def grade(L: dict) -> int:
    if L["sport"] == "ufc":
        return 0 if L["_winner"] not in ("A", "B") else (1 if L["_winner"] == L["_side"] else -1)
    if L["sport"] == "nba-prop":
        return 1 if L["_win"] else -1
    hs, as_ = L.get("_hs"), L.get("_as")
    if hs is None or np.isnan(hs):
        return 0
    margin = hs - as_
    if L["market"] == "tot":
        x = (hs + as_ - L["_total"]) * (1 if L["pick"].startswith("OVER") else -1)
    elif L["market"] == "ats":
        x = (margin + L["_line"]) * (1 if L["_home_side"] else -1)
    else:
        x = margin * (1 if L["_home_side"] else -1)
    return int(np.sign(x))


# ------------------------------------------------------------------ parlays
def parlay_tickets(legs: list[dict]) -> list[dict]:
    """2-leg round robin across Tier A+B game legs (different events). The backtest showed this is
    where edges compound: A-tier 2-leg RR = +15% ROI on holdout seasons."""
    core = [L for L in legs if L["tier"] in ("A", "B")]
    core = list({L["event"]: L for L in sorted(core, key=lambda x: x["tier"], reverse=True)}.values())
    tickets = []
    for a, b in itertools.combinations(core, 2):
        dec = float(american_to_decimal(a["price"]) * american_to_decimal(b["price"]))
        tickets.append(dict(type="2-leg", legs=[a, b], dec=dec, stake=STAKE["parlay2"]))
    if 3 <= len(core) <= 6:
        for c in itertools.combinations(core, 3):
            dec = float(np.prod([american_to_decimal(x["price"]) for x in c]))
            tickets.append(dict(type="3-leg", legs=list(c), dec=dec, stake=STAKE["parlay3"]))
    return tickets


def ticket_pnl(t, grades):
    gs = [grades[id(L)] for L in t["legs"]]
    if any(g < 0 for g in gs):
        return -t["stake"]
    live = [L for L, g in zip(t["legs"], gs) if g > 0]
    if not live:
        return 0.0
    return t["stake"] * (float(np.prod([american_to_decimal(L["price"]) for L in live])) - 1)


# ------------------------------------------------------------------ card
def fmt_price(p):
    return fmt_american(p)


def make_card(date: pd.Timestamp, replay: bool, sports=None) -> tuple[str, pd.DataFrame]:
    sports = sports or ["nfl", "ncaaf", "nba", "ncaam", "mlb", "nhl"]
    legs = []
    for sp in sports:
        legs += team_legs(sp, date)
    legs += ufc_legs(date)
    props = nba_prop_legs(date)
    tickets = parlay_tickets(legs)
    for L in legs + props:
        L["stake"] = STAKE[L["tier"]]
    lines = [f"# Bet Card: {date:%A %b %d, %Y}", "",
             f"1u = 1% of bankroll · {len(legs)} strategy bets · {len(props)} prop bets · {len(tickets)} parlay tickets", ""]
    rows = []
    if replay:
        grades = {id(L): grade(L) for L in legs + props}
    lines += ["## Strategy bets", "", "| Tier | Sport | Strategy | Game | Bet | Price | Stake |" + (" Result |" if replay else ""),
              "|---|---|---|---|---|---|---|" + ("---|" if replay else "")]
    for L in sorted(legs, key=lambda x: (x["tier"], x["sport"])):
        res = ""
        if replay:
            g = grades[id(L)]
            pnl = L["stake"] * (american_to_decimal(L["price"]) - 1) if g > 0 else (-L["stake"] if g < 0 else 0)
            res = f" {'WIN' if g > 0 else 'LOSS' if g < 0 else 'PUSH'} {pnl:+.2f}u |"
            rows.append(dict(kind="single", tier=L["tier"], sport=L["sport"], bet=L["pick"], price=L["price"],
                             stake=L["stake"], pnl=pnl))
        lines.append(f"| {L['tier']} | {L['sport'].upper()} | {L['strategy']} | {L['event']} | **{L['pick']}** | "
                     f"{fmt_price(L['price'])} | {L['stake']}u |{res}")
    if props:
        lines += ["", "## NBA points props (blend model, EV ≥ 4%)", "",
                  "| Bet | Price | Model EV | Stake |" + (" Result |" if replay else ""), "|---|---|---|---|" + ("---|" if replay else "")]
        for L in sorted(props, key=lambda x: -x["ev"]):
            res = ""
            if replay:
                g = grades[id(L)]
                pnl = L["stake"] * (american_to_decimal(L["price"]) - 1) if g > 0 else -L["stake"]
                res = f" {'WIN' if g > 0 else 'LOSS'} {pnl:+.2f}u |"
                rows.append(dict(kind="prop", tier="prop", sport="nba-prop", bet=L["pick"], price=L["price"],
                                 stake=L["stake"], pnl=pnl))
            lines.append(f"| {L['pick']} | {fmt_price(L['price'])} | {L['ev']*100:+.1f}% | {L['stake']}u |{res}")
    if tickets:
        lines += ["", "## Parlays (Tier A+B legs only, different games, place all legs at ONE book)", "",
                  "| Ticket | Legs | Payout | Stake |" + (" Result |" if replay else ""), "|---|---|---|---|" + ("---|" if replay else "")]
        for t in tickets:
            res = ""
            if replay:
                pnl = ticket_pnl(t, grades)
                res = f" {pnl:+.2f}u |"
                rows.append(dict(kind=t["type"], tier="parlay", sport="multi", bet=" + ".join(L["pick"] for L in t["legs"]),
                                 price=t["dec"], stake=t["stake"], pnl=pnl))
            lines.append(f"| {t['type']} | {' + '.join(L['pick'] for L in t['legs'])} | {fmt_american(decimal_to_american(t['dec']))} | {t['stake']}u |{res}")
    if not legs and not props:
        lines += ["", "_No qualifying bets tonight. Passing is a position._"]
    R = pd.DataFrame(rows)
    if replay and len(R):
        lines += ["", f"**Night result:** {R.pnl.sum():+.2f}u on {R.stake.sum():.2f}u risked "
                      f"(singles {R[R.kind=='single'].pnl.sum():+.2f}u · props {R[R.kind=='prop'].pnl.sum():+.2f}u · "
                      f"parlays {R[R.tier=='parlay'].pnl.sum():+.2f}u)"]
    return "\n".join(lines), R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live:
        from nightly.live import live_card
        md = live_card()
        date = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    else:
        date = pd.Timestamp(a.date)
        md, R = make_card(date, replay=a.replay)
    CARDS.mkdir(parents=True, exist_ok=True)
    out = CARDS / f"card_{date:%Y-%m-%d}{'_replay' if a.replay else ''}.md"
    out.write_text(md)
    print(md)
    print(f"\nsaved -> {out}")
