"""
THE SUPER ENGINE: one entry point for the whole nightly loop.

  python -m engine.super card   [--date YYYY-MM-DD --replay]   # build + grade + size tonight's card (journal it if live)
  python -m engine.super close                                  # ~1h before games: capture closing lines -> CLV
  python -m engine.super settle                                 # next morning: grade results -> learner/calibrator update
  python -m engine.super status                                 # learner state, CLV to date, calibration

Layers
  1. Strategy legs  : the 7 validated keys + extras (nightly.card), same masks as the backtest
  2. NBA props      : 25/75 model/market blend, EV >= 4% (backtest.props_nba)
  3. Parlays        : 2-leg round robins of non-retired Tier A/B legs, different events, one book
  4. Auditor        : Bayesian edge posterior per strategy (kill switch, not a throttle: static tier stakes
                      won the out-of-sample replay), pre-bet grade (calibrated p, P(edge>0), expected CLV),
                      journal at placement -> closing-line value -> settlement -> learning
Learning signals, most to least informative: CLV (known before the result), calibration, then slowly
the realized P&L posterior. Single wins/losses never move the engine much, by design.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from backtest.core import decimal_to_american, fmt_american
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "results" / "cards"


# ------------------------------------------------------------------ auditor glue
def auditor():
    """Tonight's learner + calibrator + CLV model, built only from already-settled information."""
    from audit import journal as J
    from audit.replay import warm_learner, load_bets
    from audit.calibration import OnlineRecalibrator
    from audit.confidence import CLVModel
    lr = warm_learner()
    cal = OnlineRecalibrator("platt")
    try:
        P = load_bets()
        P = P[P.strategy.str.contains("NBA points", na=False)]
        jr = J.load()
        if len(jr) and "model_prob" in jr:
            s = jr.dropna(subset=["result", "model_prob"])
            s = s[s.strategy.str.contains("NBA points", na=False)]
            if len(s):
                P = pd.concat([P[["model_prob", "y"]], pd.DataFrame({"model_prob": s.model_prob, "y": (s.result > 0).astype(int)})])
        cal.fit(P.model_prob, P.y)
    except Exception:
        cal = None
    try:
        clv = CLVModel().fit(J.load().dropna(subset=["clv_prob"]))
    except Exception:
        clv = None
    return lr, cal, clv


def grade_legs(legs: list[dict], lr, cal, clv) -> list[dict]:
    from audit import journal as J
    from audit.confidence import confidence
    from nightly.card import STAKE
    for L in legs:
        seg = J.default_segment(L["sport"], L["market"], L["pick"], L["price"])
        try:
            c = confidence(L, lr, cal, clv, segment=seg)
        except Exception as e:  # grading must never block the card
            c = {"grade": "?", "note": str(e)}
        L.update({f"aud_{k}": v for k, v in c.items()})
        base = STAKE[L["tier"]]
        state = lr.strategy_posterior(L["strategy"]).get("state", "ACTIVE") if hasattr(lr, "strategy_posterior") else "ACTIVE"
        L["state"] = state
        # kill switch: static tier stake unless the strategy has been RETIRED on evidence
        L["stake"] = 0.0 if state == "RETIRED" else base
    return legs


# ------------------------------------------------------------------ card
def build_card(date: pd.Timestamp, replay: bool) -> tuple[str, list, list, list]:
    from nightly.card import team_legs, ufc_legs, nba_prop_legs, parlay_tickets
    legs = []
    for sp in ["nfl", "ncaaf", "nba", "ncaam", "mlb", "nhl"]:
        legs += team_legs(sp, date)
    legs += ufc_legs(date)
    props = nba_prop_legs(date)
    return legs, props


def render(date, legs, props, tickets, replay, grades=None) -> str:
    from nightly.card import fmt_price
    from backtest.core import american_to_decimal
    out = [f"# Super Engine Card: {date:%A %b %d, %Y}{' (replay)' if replay else ''}", "",
           "Stake = static tier size (the winner in the out-of-sample audit), zeroed if the auditor has RETIRED "
           "the strategy. Grade = pre-bet confidence (calibrated p, P(edge>0), expected CLV).", "",
           "| Grade | Tier | Sport | Strategy | Game | Bet | Price | Stake |" + (" Result |" if replay else ""),
           "|---|---|---|---|---|---|---|---|" + ("---|" if replay else "")]
    total = 0.0
    for L in sorted(legs, key=lambda x: (x["tier"], str(x.get("aud_grade")))):
        res = ""
        if replay:
            g = grades[id(L)]
            pnl = L["stake"] * (american_to_decimal(L["price"]) - 1) if g > 0 else (-L["stake"] if g < 0 else 0)
            total += pnl
            res = f" {'W' if g > 0 else 'L' if g < 0 else 'P'} {pnl:+.2f}u |"
        out.append(f"| {L.get('aud_grade', '?')} | {L['tier']} | {L['sport'].upper()} | {L['strategy']} | {L['event']} | "
                   f"**{L['pick']}** | {fmt_price(L['price'])} | {L['stake']}u{' (RETIRED)' if L.get('state') == 'RETIRED' else ''} |{res}")
    if props:
        out += ["", "## NBA points props", "", "| Bet | Price | EV | Stake |" + (" Result |" if replay else ""),
                "|---|---|---|---|" + ("---|" if replay else "")]
        for L in sorted(props, key=lambda x: -x["ev"]):
            res = ""
            if replay:
                g = grades[id(L)]
                pnl = L["stake"] * (american_to_decimal(L["price"]) - 1) if g > 0 else -L["stake"]
                total += pnl
                res = f" {'W' if g > 0 else 'L'} {pnl:+.2f}u |"
            out.append(f"| {L['pick']} | {fmt_price(L['price'])} | {L['ev']*100:+.1f}% | {L['stake']}u |{res}")
    if tickets:
        out += ["", "## Parlays (one book, different games)", "", "| Legs | Payout | Stake |" + (" Result |" if replay else ""),
                "|---|---|---|" + ("---|" if replay else "")]
        from nightly.card import ticket_pnl
        for t in tickets:
            res = ""
            if replay:
                pnl = ticket_pnl(t, grades)
                total += pnl
                res = f" {pnl:+.2f}u |"
            out.append(f"| {' + '.join(L['pick'] for L in t['legs'])} | {fmt_american(decimal_to_american(t['dec']))} | {t['stake']}u |{res}")
    if replay:
        out += ["", f"**Night total: {total:+.2f}u**"]
    if not legs and not props:
        out += ["", "_No qualifying bets. Passing is a position._"]
    return "\n".join(out)


def cmd_card(date: pd.Timestamp | None, replay: bool):
    from nightly.card import parlay_tickets, grade, STAKE
    lr, cal, clv = auditor()
    if replay:
        legs, props = build_card(date, replay=True)
    else:
        from nightly import live
        date = pd.Timestamp(datetime.now().date())
        legs, props = live_legs(date)
    for P in props:
        P["stake"] = STAKE["prop"]
    legs = grade_legs(legs, lr, cal, clv)
    tickets = parlay_tickets([L for L in legs if L["stake"] > 0])
    grades = {id(L): grade(L) for L in legs + props} if replay else None
    md = render(date, legs, props, tickets, replay, grades)
    CARDS.mkdir(parents=True, exist_ok=True)
    out = CARDS / f"super_{date:%Y-%m-%d}{'_replay' if replay else ''}.md"
    out.write_text(md)
    if not replay:
        from audit import journal as J
        now = datetime.now().isoformat(timespec="seconds")
        rows = J.journal_from_card_legs([L for L in legs + props if L.get("stake", 0) > 0], placed_at=now)
        ids = J.record(rows)
        (CARDS / f"super_{date:%Y-%m-%d}_bets.json").write_text(json.dumps(
            [dict(bet_id=i, **{k: v for k, v in L.items() if not k.startswith("_") and _jsonable(v)})
             for i, L in zip(ids, [L for L in legs + props if L.get("stake", 0) > 0])], indent=1, default=str))
    print(md)
    print(f"\nsaved -> {out}")


def _jsonable(v):
    return isinstance(v, (str, int, float, bool, type(None)))


def live_legs(date):
    """Live strategy legs + props via nightly.live (network required)."""
    from nightly import live
    from nightly.card import team_legs
    legs = []
    for sport in ["nfl", "ncaaf", "nba", "ncaam", "mlb", "nhl"]:
        try:
            live.update_results(sport)
            slate = live.today_slate(sport)
            if slate.empty:
                continue
            L = team_legs(sport, date, history=live.history_with_today(sport, slate))
            for x in L:
                row = slate[(slate.home == x["_home"]) & (slate.away == x["_away"])]
                x["espn_id"] = row.iloc[0]["espn_id"] if len(row) else None
                x["game_date"] = str(date.date())
            legs += L
        except Exception as e:
            print(f"[warn] {sport}: {e}")
    legs += live.live_ufc_legs(date, [])
    props = []
    try:
        B = live.live_nba_props(date)
        for _, r in B.iterrows():
            props.append(dict(sport="nba-prop", strategy="NBA points: blend EV", tier="prop", event=f"{r.home_team} vs {r.away_team}",
                              market="prop", pick=f"{r.player} {r.side.upper()} {r.line:g} pts @{r.book}", price=float(r.price),
                              ev=float(r.ev), model_prob=float(r.p_blend if r.side == "over" else 1 - r.p_blend),
                              player=r.player, line=float(r.line), side=r.side, book=r.book, game_date=str(date.date())))
    except Exception as e:
        print(f"[warn] props: {e}")
    return legs, props


# ------------------------------------------------------------------ close + settle (live loop)
def cmd_close():
    """Re-price every open journaled bet from the current board -> closing line & CLV (before the result)."""
    from audit import journal as J
    from nightly import live
    jr = J.load()
    open_ = jr[jr.closing_price.isna() & jr.result.isna()] if len(jr) else jr
    boards = {}
    n = 0
    for _, b in open_.iterrows():
        sp = b.sport
        if sp not in live.SPORTS or sp == "nba-prop":
            continue
        if sp not in boards:
            try:
                boards[sp] = live.odds_board(sp)
            except Exception:
                boards[sp] = pd.DataFrame()
        B = boards[sp]
        if B.empty:
            continue
        away, home = [s.strip() for s in str(b.event).split(" @ ")] if " @ " in str(b.event) else (None, None)
        row = B[B.home_name.map(live.norm_name).str.contains(live.norm_name(home or "~"))] if home else B.iloc[0:0]
        if row.empty:
            continue
        r = row.iloc[0]
        home_side = str(b.pick).startswith(str(home)) or str(b.pick).upper().startswith("OVER")
        if b.market == "ml":
            price, opp = (r.home_ml, r.away_ml) if home_side else (r.away_ml, r.home_ml)
            J.attach_close(b.bet_id, closing_price=price, closing_opp_price=opp)
        elif b.market == "tot":
            over = str(b.pick).upper().startswith("OVER")
            J.attach_close(b.bet_id, closing_price=r.over_odds if over else r.under_odds, closing_line=r.total,
                           closing_opp_price=r.under_odds if over else r.over_odds)
        elif b.market == "ats":
            line = r.spread_home if home_side else -r.spread_home
            J.attach_close(b.bet_id, closing_price=r.spread_home_odds if home_side else r.spread_away_odds, closing_line=line,
                           closing_opp_price=r.spread_away_odds if home_side else r.spread_home_odds)
        n += 1
    print(f"closing lines attached: {n}")


def cmd_settle(days_back=3):
    """Grade journaled bets from ESPN finals (team sports) and refreshed box scores (NBA props)."""
    from audit import journal as J
    from nightly import live
    jr = J.load()
    if not len(jr):
        print("journal empty")
        return
    todo = jr[jr.result.isna()]
    finals = {}
    n = 0
    for _, b in todo.iterrows():
        d = pd.Timestamp(str(b.placed_at)[:10])
        sp = b.sport
        if sp in live.SPORTS and sp != "ufc":
            key = (sp, d)
            if key not in finals:
                try:
                    finals[key] = live.espn_games(sp, d.to_pydatetime())
                except Exception:
                    finals[key] = pd.DataFrame()
            G = finals[key]
            if G.empty or " @ " not in str(b.event):
                continue
            away, home = [s.strip() for s in str(b.event).split(" @ ")]
            g = G[(G.home == home) & (G.away == away) & G.completed]
            if g.empty:
                continue
            g = g.iloc[0]
            res = _grade_pick(b, g.home_score, g.away_score, home)
            J.settle(b.bet_id, res)
            n += 1
        elif sp == "nba-prop":
            from backtest import props_nba as P
            box = pd.read_csv(P.proc_path("nba_player_box"), usecols=["date", "player", "pts"], low_memory=False)
            import re
            m = re.match(r"(.+?) (OVER|UNDER) ([\d.]+) pts", str(b.pick))
            if not m:
                continue
            name, side, line = m.group(1), m.group(2), float(m.group(3))
            row = box[(box.date == str(d.date())) & (box.player.map(live.norm_name) == live.norm_name(name))]
            if len(row) and pd.notna(row.pts.iloc[0]):
                pts = float(row.pts.iloc[0])
                over = side == "OVER"
                J.settle(b.bet_id, int(np.sign((pts - line) * (1 if over else -1))))
                n += 1
    print(f"settled: {n}")


def _grade_pick(b, hs, as_, home):
    import re
    pick = str(b.pick)
    if b.market == "tot":
        tot = float(re.findall(r"[-\d.]+", pick)[-1])
        return int(np.sign((hs + as_ - tot) * (1 if pick.upper().startswith("OVER") else -1)))
    home_side = pick.startswith(home)
    margin = (hs - as_) * (1 if home_side else -1)
    if b.market == "ats":
        line = float(re.findall(r"[-+]\d+(?:\.\d+)?", pick)[-1])
        return int(np.sign(margin + line))
    return int(np.sign(margin))


def cmd_status():
    lr, cal, clv = auditor()
    from audit import journal as J
    print(lr.table().to_string() if hasattr(lr, "table") else lr)
    jr = J.load()
    if len(jr):
        s = jr.dropna(subset=["result"])
        print(f"\njournal: {len(jr)} bets, {len(s)} settled, P&L {s.pnl.sum():+.2f}u, "
              f"mean CLV {jr.clv_prob.mean()*100 if 'clv_prob' in jr else float('nan'):+.2f} pts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["card", "close", "settle", "status"])
    ap.add_argument("--date")
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    if a.cmd == "card":
        cmd_card(pd.Timestamp(a.date) if a.date else None, a.replay)
    elif a.cmd == "close":
        cmd_close()
    elif a.cmd == "settle":
        cmd_settle()
    else:
        cmd_status()
