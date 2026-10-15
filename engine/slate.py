"""
Nightly SLATE BUILDER: a same-game parlay for every game, the best ones per sport on the card, every
candidate in Excel, plus cross-game parlays (weekend round robins, every-game parlay, 20-leg moonshot).

  python -m engine.slate                 # tonight (live FanDuel + Polymarket + ESPN)
  python -m engine.slate --days 2        # include tomorrow's games (e.g. build Saturday+Sunday on Saturday)

Probability of each leg
  base     = de-vigged FanDuel price (averaged with Polymarket's mid when both list the same line)
  edge     = a validated strategy fires on this leg -> p = (1 + ROI_holdout/2) / fair_decimal
             (half the holdout ROI: deliberate shrinkage) | NBA points: 0.25 model + 0.75 market blend
Joint probability: Gaussian copula with correlations fit on real games (engine/correlation.py), used only
where they beat independence on holdout seasons; NBA prop correlations (player vs game total / team side /
teammates) are stable across both seasons and always used.

What we can and cannot know
  * Cross-game parlays: sportsbooks pay the product of the legs' prices -> exact EV.
  * Same-game parlays: FanDuel re-prices correlated legs and does not publish that price via its web API,
    so each SGP comes with FAIR odds and a "bet only if FanDuel shows >= X" threshold (fair + 5% edge).
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from backtest.core import american_to_decimal, decimal_to_american, fmt_american, no_vig_probs
from engine.correlation import joint_prob, RHO_FILE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "slates"
JOURNAL = ROOT / "data" / "journal"
SPORTS = ["nfl", "ncaaf", "nba", "ncaam", "nhl", "mlb", "ufc"]
EDGE_TO_BET = 0.05           # require 5% expected edge over fair on SGPs before betting
MAX_SGP_LEGS = 4
STRATEGY_ROI = {}            # filled from results/keys.json (holdout ROI per key)


def _strategy_roi():
    if STRATEGY_ROI:
        return STRATEGY_ROI
    try:
        K = json.loads((ROOT / "results" / "keys.json").read_text())
        for sp, r in K.items():
            STRATEGY_ROI[r["name"]] = float(r["holdout"].get("roi", 0) or 0)
    except Exception:
        pass
    STRATEGY_ROI.setdefault("Bounce-back dogs ATS", 0.129)
    return STRATEGY_ROI


# ------------------------------------------------------------------ legs
def _local_day(ts) -> str:
    try:
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        return str(t.tz_convert("America/New_York").date())
    except Exception:
        return ""

def _fair(a, b):
    if a is None or b is None or a != a or b != b:
        return np.nan, np.nan
    pa, pb, _ = no_vig_probs(a, b)
    return float(pa), float(pb)


def game_legs(sport: str, g: pd.Series) -> list[dict]:
    """All standard legs for one game with best price + fair probability."""
    best = json.loads(g["best"]) if isinstance(g.get("best"), str) else {}
    ev = f"{g.away_name} @ {g.home_name}" if sport != "ufc" else f"{g.away_name} vs {g.home_name}"
    legs = []

    def add(kind, side, label, price_col, fair_p, line=None):
        b = best.get(price_col.replace("_odds", ""))
        price, src = (b[0], b[1]) if b else (g.get(price_col), g.get("source", "FanDuel"))
        fd = g.get(price_col) if g.get("source", "FanDuel") == "FanDuel" else None
        if price is None or price != price or fair_p != fair_p:
            return
        legs.append(dict(sport=sport, event=ev, game_key=ev, kind=kind, side=side, pick=label, line=line,
                         price=float(price), book=src, p_fair=fair_p, p=fair_p, edge_src="",
                         fd_price=float(fd) if fd is not None and fd == fd else None,
                         home=g.home_name, away=g.away_name, day=_local_day(g.get("commence"))))
    ph, pa = _fair(g.get("home_ml"), g.get("away_ml"))
    add("ml", "home", f"{g.home_name} ML", "home_ml", ph)
    add("ml", "away", f"{g.away_name} ML", "away_ml", pa)
    if sport != "ufc" and g.get("spread_home") == g.get("spread_home"):
        sh, sa = _fair(g.get("spread_home_odds"), g.get("spread_away_odds"))
        s = float(g.spread_home)
        add("spread", "home", f"{g.home_name} {s:+g}", "spread_home_odds", sh, s)
        add("spread", "away", f"{g.away_name} {-s:+g}", "spread_away_odds", sa, -s)
    if g.get("total") == g.get("total") and g.get("total") is not None:
        po, pu = _fair(g.get("over_odds"), g.get("under_odds"))
        t = float(g.total)
        short = f"{str(g.away_name).split()[-1]}/{str(g.home_name).split()[-1]}"
        add("total", "over", f"{short} OVER {t:g}", "over_odds", po, t)
        add("total", "under", f"{short} UNDER {t:g}", "under_odds", pu, t)
    fav_home = (g.get("home_ml") or 0) < (g.get("away_ml") or 0)
    for L in legs:
        L["fav"] = (L["side"] == "home") == fav_home if L["kind"] in ("ml", "spread") else None
        L["spread_abs"] = abs(float(g.get("spread_home") or 0)) if sport != "ufc" else 0
    return legs


def apply_strategy_edges(legs: list[dict], key_legs: list[dict]):
    """Mark legs a validated strategy fires on: p = (1 + ROI/2) / fair decimal (shrunk)."""
    from nightly.live import norm_name
    roi = _strategy_roi()
    for k in key_legs:
        hs = {norm_name(x) for x in (k.get("_home"), k.get("_home_name")) if isinstance(x, str)}
        as_ = {norm_name(x) for x in (k.get("_away"), k.get("_away_name")) if isinstance(x, str)}
        for L in legs:
            if L["sport"] == "ufc":
                same_game = k["event"] == L["event"]
            else:   # exact team identity on BOTH sides (no substring matches)
                same_game = norm_name(L.get("home", "")) in hs and norm_name(L.get("away", "")) in as_
            if not same_game:
                continue
            kind = {"ml": "ml", "ats": "spread", "tot": "total"}.get(k["market"])
            side = ("over" if k["pick"].startswith("OVER") else "under") if kind == "total" else \
                   ("home" if k.get("_home_side", True) else "away")
            if L["kind"] == kind and L["side"] == side:
                e = max(roi.get(k["strategy"], 0.0), 0) / 2
                L["p"] = min(0.97, (1 + e) * L["p_fair"])
                L["edge_src"] = k["strategy"]


def prop_legs_nba(board: pd.DataFrame) -> list[dict]:
    """FanDuel (+ liquid Polymarket) NBA points props with blended model probability."""
    from nightly.providers import fanduel_nba_points, polymarket_nba_points
    try:
        q = fanduel_nba_points(board)
    except Exception:
        q = pd.DataFrame()
    try:
        pmq = polymarket_nba_points()
        pmq = pmq[pmq.over_odds.notna() & pmq.under_odds.notna()]
    except Exception:
        pmq = pd.DataFrame()
    if q.empty:
        return []
    proj = _nba_projections(q)
    legs = []
    for _, r in q.iterrows():
        po, pu = _fair(r.over_odds, r.under_odds)
        m = pmq[(pmq.player.str.lower() == r.player.lower()) & (pmq.line == r.line)] if len(pmq) else pmq
        if len(m):
            po2, _ = _fair(m.over_odds.iloc[0], m.under_odds.iloc[0])
            po = (po + po2) / 2
            pu = 1 - po
        p_over = po
        pr = proj.get(r.player.lower())
        if pr:
            from scipy.stats import norm
            p_mod = 1 - norm.cdf((r.line - pr[0]) / pr[1])
            p_over = 0.75 * po + 0.25 * p_mod
        ev = f"{r.away_team} @ {r.home_team}"
        team_home = None
        if pr and len(pr) > 2:
            team_home = pr[2]
        for side, price, p, pf in (("over", r.over_odds, p_over, po), ("under", r.under_odds, 1 - p_over, pu)):
            legs.append(dict(sport="nba", event=ev, game_key=ev, kind="prop", side=side, day=_local_day(r.commence),
                             pick=f"{r.player} {side.upper()} {r.line:g} pts", line=float(r.line), price=float(price),
                             book="FanDuel", fd_price=float(price), p_fair=pf, p=p, edge_src="props model" if pr else "",
                             player=r.player, team_home=team_home))
    return legs


def _nba_projections(q: pd.DataFrame) -> dict:
    """player(lower) -> (proj mean, sd, plays_for_home_team) using the walk-forward points model."""
    try:
        from nightly.live import live_projection_table
        return live_projection_table(q)
    except Exception:
        return {}


# ------------------------------------------------------------------ correlations
_RHO = None


def _rho():
    global _RHO
    if _RHO is None:
        _RHO = json.loads(RHO_FILE.read_text()) if RHO_FILE.exists() else {}
    return _RHO


def pair_rho(sport: str, a: dict, b: dict) -> float:
    R = _rho()
    kinds = {a["kind"], b["kind"]}
    if a["kind"] == "prop" and b["kind"] == "prop":
        P = R.get("nba_props", {}).get("hold", {})
        same_team = a.get("team_home") is not None and a.get("team_home") == b.get("team_home")
        r = P.get("player_over__teammate_over", -0.05) if same_team else P.get("player_over__opponent_over", 0.0)
        return r * (1 if a["side"] == b["side"] else -1)
    if "prop" in kinds:
        p, o = (a, b) if a["kind"] == "prop" else (b, a)
        P = R.get("nba_props", {}).get("hold", {})
        s_p = 1 if p["side"] == "over" else -1
        if o["kind"] == "total":
            return P.get("player_over__game_over", 0.16) * s_p * (1 if o["side"] == "over" else -1)
        if o["kind"] in ("spread", "ml") and p.get("team_home") is not None:
            same = (o["side"] == "home") == bool(p["team_home"])
            return P.get("player_over__team_cover", 0.13) * s_p * (1 if same else -1)
        return 0.0
    if kinds == {"total"} or kinds == {"spread"} or kinds == {"ml"}:
        return 0.0
    if "total" in kinds and kinds & {"spread", "ml"}:
        S = R.get(sport, {})
        if not S.get("use", False):
            return 0.0
        side, tot = (a, b) if a["kind"] != "total" else (b, a)
        bucket = "<=3" if side["spread_abs"] <= 3 else "3.5-7" if side["spread_abs"] <= 7 else \
                 "7.5-14" if side["spread_abs"] <= 14 else "14.5+"
        rb = S.get(bucket, S.get("all", {}))
        r = rb.get("cover_over" if side["kind"] == "spread" else "win_over", 0.0)
        sgn = (1 if side.get("fav") else -1) * (1 if tot["side"] == "over" else -1)
        return r * sgn
    if kinds == {"spread", "ml"}:
        return 0.6 * (1 if a["side"] == b["side"] else -1)   # same-team ML & spread: strongly linked
    return 0.0


def compatible(legs: tuple) -> bool:
    kinds = [L["kind"] for L in legs if L["kind"] != "prop"]
    if len(kinds) != len(set(kinds)):
        return False                                   # one ML, one spread, one total max
    sides = {L["kind"]: L["side"] for L in legs if L["kind"] in ("ml", "spread")}
    if "ml" in sides and "spread" in sides:
        return False                                   # ML + spread on one game: redundant or contradictory
    players = [L.get("player") for L in legs if L["kind"] == "prop"]
    if len(players) != len(set(players)) or len(players) > 2:
        return False
    return True


def price_sgp(sport: str, legs: tuple) -> dict:
    p = [L["p"] for L in legs]
    pf = [L["p_fair"] for L in legs]
    k = len(legs)
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            R[i, j] = R[j, i] = pair_rho(sport, legs[i], legs[j])
    pj, pj_fair = joint_prob(p, R), joint_prob(pf, R)
    pj = min(0.99, pj * calibration_factor())       # learned from settled SGPs (1.0 until enough history)
    naive_dec = float(np.prod([american_to_decimal(L["price"]) for L in legs]))
    fair_dec = 1 / max(pj, 1e-9)
    need_dec = fair_dec * (1 + EDGE_TO_BET)
    return dict(sport=sport, event=legs[0]["event"], legs=" + ".join(L["pick"] for L in legs), n_legs=k,
                p_model=pj, p_market=pj_fair, edge_vs_market=pj / max(pj_fair, 1e-9) - 1,
                fair_odds=fmt_american(decimal_to_american(fair_dec)),
                bet_if_fanduel_at_least=fmt_american(decimal_to_american(need_dec)),
                naive_parlay_odds=fmt_american(decimal_to_american(naive_dec)),
                edge_legs=sum(bool(L["edge_src"]) for L in legs),
                edge_sources=", ".join(sorted({L["edge_src"] for L in legs if L["edge_src"]})),
                _legs=legs)


def best_sgps(sport: str, legs: list[dict], top=3) -> list[dict]:
    cands = []
    for k in range(2, MAX_SGP_LEGS + 1):
        for combo in itertools.combinations(legs, k):
            if compatible(combo):
                cands.append(price_sgp(sport, combo))
    # score: model edge over market-fair joint prob, then probability (prefer hittable scripts)
    cands.sort(key=lambda c: (c["edge_vs_market"], c["p_model"]), reverse=True)
    return cands[:top] if top else cands


# ------------------------------------------------------------------ cross-game parlays
PLAYBOOK = {
    # backtested on holdout seasons, real closing prices (engine/weekend_backtest.py, backtest/parlay.py)
    "weekend": [("best3", 0.5, "Best 3 legs (top+2nd tier): hit 31%, ROI +26.9%, ~+366"),
                ("rr2", 0.25, "2-leg round robin, all top+2nd tier legs: hit 41%, ROI +13.9% (most reliable)"),
                ("all", 0.25, "All edge legs in one: hit 30%, ROI +17.5%, ~+600 (upside backup)")],
    "weekday": [("rr2", 0.25, "2-leg round robin: hit 34%, ROI +11.4% (thin sample; singles are the core)")],
}


def cross_game_day(legs_day: list[dict], day: str, max_legs=20) -> dict:
    """One calendar day's tickets per the backtested playbook. Exact pricing (product of prices)."""
    from backtest.parlay import TIER_OF
    weekend = pd.Timestamp(day).weekday() >= 5 if day else False
    per_game = {}
    legs_day = [L for L in legs_day if L.get("fd_price") is not None and L["kind"] != "prop"]
    for L in legs_day:
        e = L["p"] * american_to_decimal(L["fd_price"]) - 1
        key = L["game_key"]
        if key not in per_game or (bool(L["edge_src"]), e) > (bool(per_game[key][1]["edge_src"]), per_game[key][0]):
            per_game[key] = (e, L)

    def ticket(sel, stake=None, note=""):
        # one book: every parlay leg at its FanDuel price (Polymarket has no parlays)
        dec = float(np.prod([american_to_decimal(L["fd_price"]) for _, L in sel]))
        p = float(np.prod([L["p"] for _, L in sel]))
        return dict(day=day, book="FanDuel", n_legs=len(sel),
                    legs=" + ".join(f"{L['pick']} ({fmt_american(L['fd_price'])})" for _, L in sel),
                    odds=fmt_american(decimal_to_american(dec)), hit_prob=p, ev=p * dec - 1,
                    one_in=round(1 / max(p, 1e-12)), stake=stake, note=note)
    edge = [(e, L) for e, L in per_game.values() if L["edge_src"] and TIER_OF.get(L["edge_src"], "C") in ("A", "B")]
    edge.sort(key=lambda x: (TIER_OF.get(x[1]["edge_src"], "C"), american_to_decimal(x[1]["fd_price"])))
    plan = []
    for kind, stake, note in PLAYBOOK["weekend" if weekend else "weekday"]:
        if kind == "best3" and len(edge) >= 3:
            plan.append(dict(type="Best 3", **ticket(edge[:3], stake, note)))
        elif kind == "best3" and len(edge) == 2:
            plan.append(dict(type="Best 2", **ticket(edge, stake, "Only 2 edge legs: 2-leg parlay (hit 46%, ROI +12%)")))
        elif kind == "all" and len(edge) >= 4:   # backtested full-day parlays averaged 3-4 legs: cap at 5
            plan.append(dict(type=f"All edge legs (top {min(len(edge), 5)})", **ticket(edge[:5], stake, note)))
        elif kind == "rr2" and len(edge) >= 3:
            for a, b in itertools.combinations(edge, 2):
                plan.append(dict(type="RR2", **ticket([a, b], stake, note)))
    # moonshot: the 20 MOST LIKELY legs on different games that are not negative-EV (no longshot chasing)
    best_p = {}
    for L in legs_day:   # most likely leg per game whose FanDuel price is not worse than fair
        e = L["p"] * american_to_decimal(L["fd_price"]) - 1
        if e >= -0.03 and L["fd_price"] >= -600 and (L["game_key"] not in best_p or L["p"] > best_p[L["game_key"]][1]["p"]):
            best_p[L["game_key"]] = (e, L)
    safe = sorted(best_p.values(), key=lambda x: -x[1]["p"])
    moon = ticket(safe[:max_legs], 0.1, "Lottery: 20 most-likely non-negative-EV legs. Tiny stake only.") \
        if len(safe) >= max_legs else None
    every = []
    for sp in sorted({L["sport"] for _, L in per_game.values()}):
        g = sorted([x for x in per_game.values() if x[1]["sport"] == sp], key=lambda x: -x[1]["p"])
        if 2 <= len(g) <= 16:
            every.append(dict(type=f"Every {sp.upper()} game", **ticket(g, 0, "Backtest: NFL every-game Sunday parlay -82% ROI. For fun only.")))
    return dict(plan=plan, edge_legs=edge, moonshot=moon, every_game=every)


def cross_game(legs_all: list[dict]) -> dict:
    days = sorted({L.get("day", "") for L in legs_all if L.get("day")})
    return {d: cross_game_day([L for L in legs_all if L.get("day") == d], d) for d in days}


# ------------------------------------------------------------------ build
def build(days=1) -> dict:
    from nightly import live, providers
    from nightly.card import team_legs
    today = pd.Timestamp(datetime.now().date())
    horizon = pd.Timestamp(datetime.now(), tz="UTC") + pd.Timedelta(days=days)
    all_legs, sgps, notes = [], [], []
    for sport in SPORTS:
        try:
            board = providers.combined_board(sport)
            notes += board.attrs.get("errors", [])
            if board.empty:
                continue
            t = pd.to_datetime(board.commence, utc=True, errors="coerce")
            board = board[(t > pd.Timestamp.now(tz="UTC")) & (t < horizon)]
            key_legs = []
            if sport != "ufc":
                try:
                    days_list = [datetime.now() + timedelta(days=d) for d in range(days)]
                    slates = [live.today_slate(sport, day=d, board=board) for d in days_list]
                    slates = [x for x in slates if len(x)]
                    if slates:
                        H = live.history_with_today(sport, pd.concat(slates, ignore_index=True))
                        for d in days_list:
                            key_legs += team_legs(sport, pd.Timestamp(d.date()), history=H)
                except Exception as e:
                    notes.append(f"{sport} strategies: {e}")
            else:
                key_legs = live.live_ufc_legs(today, notes)
            props = prop_legs_nba(board) if sport == "nba" else []
            for _, g in board.iterrows():
                legs = game_legs(sport, g)
                apply_strategy_edges(legs, key_legs)
                ev = legs[0]["event"] if legs else None
                legs += [P for P in props if P["event"] == ev]
                all_legs += legs
                if len(legs) >= 2:
                    sgps += best_sgps(sport, legs, top=5)
        except Exception as e:
            notes.append(f"{sport}: {e}")
    return dict(date=today, legs=all_legs, sgps=sgps, cross=cross_game(all_legs), notes=notes)


def to_excel(res: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"slate_{res['date']:%Y-%m-%d}.xlsx"
    S = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in res["sgps"]])
    L = pd.DataFrame([{**{k: v for k, v in x.items()}, "price": fmt_american(x["price"]),
                       "edge": x["p"] * american_to_decimal(x["price"]) - 1} for x in res["legs"]])
    with pd.ExcelWriter(f, engine="openpyxl") as xw:
        top = S.sort_values(["sport", "edge_vs_market", "p_model"], ascending=[True, False, False]).groupby("sport").head(3) if len(S) else S
        top.to_excel(xw, sheet_name="Top SGPs", index=False)
        S.to_excel(xw, sheet_name="All SGPs", index=False)
        L.to_excel(xw, sheet_name="All Legs", index=False)
        rows = []
        for day, X in res["cross"].items():
            rows += X.get("plan", []) + X.get("every_game", []) + ([dict(type="Moonshot", **X["moonshot"])] if X.get("moonshot") else [])
        X = pd.DataFrame(rows)
        X.to_excel(xw, sheet_name="Cross-game Parlays", index=False)
    return f


def log_for_learning(res: dict):
    """Append every SGP and cross-game ticket with its predicted probability; engine.super settle grades them
    and the calibration of p_model vs realized hits drives future correlation/edge refits."""
    JOURNAL.mkdir(parents=True, exist_ok=True)
    f = JOURNAL / "sgp_log.csv"
    rows = []
    for s in res["sgps"]:
        rows.append(dict(date=str(res["date"].date()), sport=s["sport"], event=s["event"], legs=s["legs"],
                         legs_json=json.dumps([{k: L.get(k) for k in ("kind", "side", "line", "pick", "player", "p")}
                                               for L in s["_legs"]]),
                         p_model=s["p_model"], p_market=s["p_market"], fair_odds=s["fair_odds"], result=""))
    df = pd.DataFrame(rows)
    df.to_csv(f, mode="a", header=not f.exists(), index=False)


def render(res: dict, per_sport=2) -> str:
    out = [f"# Parlay Slate: {res['date']:%A %b %d, %Y}", "",
           "SGP odds are FAIR odds from correlations fit on real games. **Only bet an SGP if FanDuel's SGP builder "
           "shows at least the 'Bet if ≥' price.** Cross-game parlays use the actual listed prices.", ""]
    S = pd.DataFrame([{k: v for k, v in s.items()} for s in res["sgps"]])
    if len(S):
        out += ["## Best same-game parlays by sport", "", "| Sport | Game | Legs | Hit prob | Fair | Bet if ≥ | Edge source |",
                "|---|---|---|---|---|---|---|"]
        S = S[S.edge_legs > 0].sort_values(["sport", "edge_vs_market", "p_model"], ascending=[True, False, False])
        S = S.drop_duplicates("event")
        for _, s in S.groupby("sport").head(per_sport).iterrows():
            out.append(f"| {s.sport.upper()} | {s.event} | {s.legs} | {s.p_model*100:.1f}% | {s.fair_odds} | "
                       f"**{s.bet_if_fanduel_at_least}** | {s.edge_sources} |")
        if S.empty:
            out += ["| – | – | No SGP contains a validated edge leg tonight | – | – | – | – |"]
    for day, X in res["cross"].items():
        wkend = pd.Timestamp(day).weekday() >= 5
        out += ["", f"## {pd.Timestamp(day):%A %b %d}: parlay playbook ({'WEEKEND' if wkend else 'weekday'} rules, exact prices)", ""]
        plan = X.get("plan", [])
        if plan:
            out += ["| Ticket | Legs | Odds | Hit prob | Stake | Why |", "|---|---|---|---|---|---|"]
            rr = [t for t in plan if t["type"] == "RR2"]
            for t in [t for t in plan if t["type"] != "RR2"] + rr[:10]:
                out.append(f"| **{t['type']}** | {t['legs']} | **{t['odds']}** | {t['hit_prob']*100:.1f}% | {t['stake']}u | {t['note']} |")
            if len(rr) > 10:
                out.append(f"| … | {len(rr) - 10} more round-robin tickets in the Excel file | | | | |")
        else:
            out += [f"{len(X.get('edge_legs', []))} top/second-tier edge leg(s): bet as singles, no parlay forced."]
        if X.get("moonshot"):
            t = X["moonshot"]
            out += ["", f"🚀 20-leg moonshot: **{t['odds']}**: true hit chance 1 in {t['one_in']:,} "
                    f"(model EV {t['ev']*100:+.1f}%). {t['stake']}u max.", f"> {t['legs']}"]
    if res["notes"]:
        out += ["", "_Feed notes: " + "; ".join(res["notes"]) + "_"]
    return "\n".join(out)


# ------------------------------------------------------------------ learning loop
CAL_FILE = ROOT / "results" / "sgp_calibration.json"


def _grade_leg(L: dict, game: pd.Series | None, box: pd.DataFrame | None) -> int | None:
    """+1 win, -1 loss, 0 push/void, None = not gradeable yet."""
    if L["kind"] == "prop":
        if box is None or box.empty:
            return None
        from nightly.live import norm_name
        r = box[box.player.map(norm_name) == norm_name(L.get("player", ""))]
        if r.empty or pd.isna(r.pts.iloc[0]):
            return 0                                   # player didn't play -> void leg
        d = float(r.pts.iloc[0]) - float(L["line"])
        return int(np.sign(d if L["side"] == "over" else -d))
    if game is None:
        return None
    hs, as_ = float(game.home_score), float(game.away_score)
    if L["kind"] == "total":
        d = hs + as_ - float(L["line"])
        return int(np.sign(d if L["side"] == "over" else -d))
    m = (hs - as_) * (1 if L["side"] == "home" else -1)
    if L["kind"] == "spread":
        return int(np.sign(m + float(L["line"])))
    return int(np.sign(m))


def settle_sgp_log():
    """Grade logged SGPs from ESPN finals (+ NBA box scores for props), then refit calibration."""
    from nightly import live
    from nightly.providers import _same_team
    f = JOURNAL / "sgp_log.csv"
    if not f.exists():
        return
    log = pd.read_csv(f, dtype={"result": "object"})
    todo = log[(log.result.isna() | (log.result == "")) & (pd.to_datetime(log.date) < pd.Timestamp(datetime.now().date()))]
    finals, box = {}, None
    for i, r in todo.iterrows():
        d = pd.Timestamp(r.date)
        if (r.sport, d) not in finals:
            try:
                finals[(r.sport, d)] = live.espn_games(r.sport, d.to_pydatetime()) if r.sport != "ufc" else pd.DataFrame()
            except Exception:
                finals[(r.sport, d)] = pd.DataFrame()
        G = finals[(r.sport, d)]
        away, home = (r.event.split(" @ ") + [""])[:2]
        game = None
        if len(G):
            m = G[G.completed & G.home_name.map(lambda x: _same_team(x, home)) & G.away_name.map(lambda x: _same_team(x, away))]
            game = m.iloc[0] if len(m) else None
        legs = json.loads(r.legs_json)
        if any(L["kind"] == "prop" for L in legs) and box is None:
            from backtest.props_nba import proc_path
            box = pd.read_csv(proc_path("nba_player_box"), usecols=["date", "player", "pts"], low_memory=False)
        b = box[box.date == str(d.date())] if box is not None else None
        res = [_grade_leg(L, game, b) for L in legs]
        if any(x is None for x in res):
            continue
        log.at[i, "result"] = "loss" if any(x < 0 for x in res) else ("win" if any(x > 0 for x in res) else "void")
    log.to_csv(f, index=False)
    refit_calibration(log)


def refit_calibration(log: pd.DataFrame, min_n=200):
    """Bucketed predicted-vs-realized hit rates. Once >= min_n settled, store a shrunk multiplicative
    correction (realized / predicted, shrunk toward 1 with weight n/(n+500)) applied to future p_model."""
    s = log[log.result.isin(["win", "loss"])].copy()
    if s.empty:
        return
    s["hit"] = s.result == "win"
    s["bucket"] = pd.cut(s.p_model, [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1])
    tab = s.groupby("bucket").agg(n=("hit", "size"), predicted=("p_model", "mean"), realized=("hit", "mean")).dropna()
    n = len(s)
    ratio = s.hit.mean() / max(s.p_model.mean(), 1e-9)
    w = n / (n + 500)
    factor = 1 + w * (ratio - 1) if n >= min_n else 1.0
    CAL_FILE.write_text(json.dumps(dict(n=n, raw_ratio=ratio, factor=factor,
                                        buckets=tab.reset_index().astype(str).to_dict("records")), indent=1))
    print(f"SGP calibration: n={n} realized/predicted={ratio:.3f} -> factor {factor:.3f}")


def calibration_factor() -> float:
    try:
        return float(json.loads(CAL_FILE.read_text())["factor"])
    except Exception:
        return 1.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--settle", action="store_true", help="grade logged SGPs and refit calibration, then exit")
    a = ap.parse_args()
    if a.settle:
        settle_sgp_log()
        raise SystemExit
    res = build(a.days)
    md = render(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"slate_{res['date']:%Y-%m-%d}.md").write_text(md)
    x = to_excel(res)
    log_for_learning(res)
    print(md)
    print(f"\nExcel -> {x}  ({len(res['sgps'])} SGPs, {len(res['legs'])} legs)")
