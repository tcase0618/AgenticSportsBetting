"""
No-arbitrage / ladder-consistency finder for the arb scanner (called from arb_scanner.scan).

Every two-way quote is mapped onto a threshold on one integer outcome Y of the event:
  spread family  Y = home margin. home at home-line h wins if Y > -h ("up", c = -h); away at home-line h wins if
                 Y < -h ("down", c = -h). Moneylines are rungs of the same ladder: home ML = up at c = +0.5 and away
                 ML = down at c = -0.5 (no-tie sports); NFL ML = c = 0 (a tie pushes 2-way ML).
  totals family  Y = total (game total, 1H total, one team's total, or a player's stat). over = up at c = line,
                 under = down at c = line.
So P(up at c) must be non-increasing in c. That single relation covers (a) alt-spread ladders, incl.
P(team -k) + P(opp +k) = 1, (b) alt-total ladders, (c) Kalshi/Polymarket rungs vs sportsbook lines and (d) ML vs
spread (P(win) >= P(-1.5) and <= P(+1.5)).

Checks
  pairs     up at c_u + down at c_d with c_u <= c_d: at least one leg always wins, both win inside (c_u, c_d).
            With equal-payout stakes the worst case is 1/cost - 1, so cost < 1 is a guaranteed ARB (plus a middle
            bonus when c_u < c_d). 1 <= cost <= MIDDLE_MAX with an integer inside the window is a MIDDLE; its
            expected value (p_up + p_down)/cost - 1 is computed from the sharp-weighted fair probabilities at each
            leg's own line. Same-venue pairs are included (a venue whose own ladder is crossed).
  monotone  per-venue de-vigged ladders and the sharp-weighted fair ladder must be non-increasing in c; violations
            larger than MONO_TOL are reported (kind 'violation', informational: the bettable side, if any, shows up
            as a pair or as +EV).
  tt_total  (e) sum of the two team-total median lines vs the game-total median line (per venue and consensus);
            |difference| >= TT_TOL points is reported as a violation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIDDLE_MAX = 1.03
MONO_TOL_VENUE = 0.02
MONO_TOL_FAIR = 0.015
TT_TOL = 1.5
EXCHANGES = {"Polymarket", "Kalshi"}


def mbase(m):
    return str(m).split(":", 1)[0]


def family(mkt):
    if mkt in ("ml", "spread"):
        return "spread"
    return mkt                       # spread:1h, total, total:1h, team_total:home, prop:...


def _rungs(N):
    """Quotes as ladder rungs: columns fam, dirn ('up'/'down'), c."""
    X = N.copy()
    X["fam"] = X.market.map(family)
    sp = X.fam.str.startswith("spread")
    up = np.where(sp, X.side == "home", X.side == "over")
    X["dirn"] = np.where(up, "up", "down")
    ml_c = np.where(X.sport == "nfl", 0.0, 0.5)
    c = np.where(sp, -X.hline.astype(float), X.hline.astype(float))
    c = np.where(X.market == "ml", np.where(X.side == "home", ml_c, -ml_c), c)
    X["c"] = c.astype(float)
    return X[X.c == X.c]


def _check(a, b):
    if a["market"] == "ml" or b["market"] == "ml":
        return "d:ml-vs-spread"
    if mbase(a["market"]) == "team_total":
        return "e:team-total-ladder"
    ex = (a["venue"] in EXCHANGES) != (b["venue"] in EXCHANGES)
    if ex and a["hkey"] != b["hkey"]:
        return "c:exchange-rung-vs-book"
    if a["venue"] == b["venue"]:
        return ("a" if a["fam"].startswith("spread") else "b") + ":same-venue-ladder"
    return ("a:spread" if a["fam"].startswith("spread") else "b:total") + "-ladder"


def find(N: pd.DataFrame, fair, min_dec=1.2, max_dec=6.0) -> list:
    """N: clean quotes (stale/suspect removed). fair(espn_id, market, hkey, side) -> fair prob or nan.
    Returns signal dicts compatible with arb_scanner (kind arb/middle/violation, legs, cost, margin, ...)."""
    out = []
    if N.empty:
        return out
    R = _rungs(N)
    for (eid, fam), g in R.groupby(["espn_id", "fam"]):
        U = g[g.dirn == "up"].sort_values("dec", ascending=False)
        D = g[g.dirn == "down"].sort_values("dec", ascending=False)
        if U.empty or D.empty:
            continue
        U = U.groupby("c").head(2)
        D = D.groupby("c").head(2)
        for _, x in U.iterrows():
            for _, y in D.iterrows():
                if x.c > y.c:
                    continue
                if x.c == y.c and x.venue != y.venue and x.market == y.market:
                    continue                       # plain same-line cross-venue arb: arb_scanner.find_signals
                cost = 1 / x.dec + 1 / y.dec
                if cost >= MIDDLE_MAX or not (min_dec <= x.dec <= max_dec and min_dec <= y.dec <= max_dec):
                    continue
                ints = [k for k in range(math.floor(x.c) + 1, math.ceil(y.c)) if x.c < k < y.c]
                if cost >= 1 and not ints:
                    continue
                pu, pd_ = fair(eid, x.market, x.hkey, x.side), fair(eid, y.market, y.hkey, y.side)
                ev = (pu + pd_) / cost - 1 if pu == pu and pd_ == pd_ else np.nan
                pw = pu + pd_ - 1 if ev == ev else np.nan
                if cost >= 1 and not (ev == ev and ev > 0) and cost > 1.01:
                    continue
                out.append(dict(kind="arb" if cost < 1 else "middle", legs=[x, y], cost=cost, margin=1 - cost,
                                worst_case=1 / cost - 1, window=(float(x.c), float(y.c)), ints=ints,
                                gap=float(y.c - x.c), p_window=pw, ev=ev, check=_check(x, y), source="consistency"))
    out += monotone(R, fair)
    out += tt_vs_total(R, fair)
    return out


def _venue_fair(R):
    """Per-venue odds_ratio-free quick de-vig (multiplicative) of each two-sided rung -> P(up)."""
    P = R.pivot_table(index=["espn_id", "fam", "venue", "market", "c"], columns="dirn", values="dec", aggfunc="max")
    P = P.dropna(subset=["up", "down"]) if {"up", "down"} <= set(P.columns) else P.iloc[0:0]
    if P.empty:
        return pd.Series(dtype=float)
    iu, idn = 1 / P["up"], 1 / P["down"]
    s = iu + idn
    P = P[(s > 0.99) & (s < 1.25)]
    return (iu / s)[P.index]


def monotone(R, fair):
    out = []
    VF = _venue_fair(R)
    if len(VF):
        for (eid, fam, v), s in VF.groupby(level=["espn_id", "fam", "venue"]):
            s = s.droplevel(["espn_id", "fam", "venue"]).groupby(level="c").mean().sort_index()
            out += _mono_scan(eid, fam, v, s, MONO_TOL_VENUE, R)
    # sharp-weighted fair ladder (all venues), one prob per c
    rows = []
    for (eid, fam, c, mkt, hk), g in R[R.dirn == "up"].groupby(["espn_id", "fam", "c", "market", "hkey"]):
        p = fair(eid, mkt, hk, g.side.iloc[0])
        if p == p:
            rows.append((eid, fam, c, p))
    if rows:
        F = pd.DataFrame(rows, columns=["espn_id", "fam", "c", "p"])
        for (eid, fam), g in F.groupby(["espn_id", "fam"]):
            s = g.groupby("c").p.mean().sort_index()
            out += _mono_scan(eid, fam, "FAIR", s, MONO_TOL_FAIR, R)
    return out


def _mono_scan(eid, fam, venue, s, tol, R):
    out = []
    cs, ps = list(s.index), list(s.values)
    for i in range(len(cs) - 1):
        for j in (i + 1,):                   # adjacent rungs
            if ps[j] - ps[i] > tol:          # P(up at higher c) exceeds P(up at lower c)
                ref = R[(R.espn_id == eid) & (R.fam == fam)].iloc[0]
                out.append(dict(kind="violation", legs=[ref], cost=np.nan, margin=np.nan, ev=np.nan,
                                check=("d:ml-vs-spread" if fam == "spread" and (0.5 in (cs[i], cs[j]) or -0.5 in (cs[i], cs[j]))
                                       else ("a" if fam.startswith("spread") else ("e" if fam.startswith("team_total") else "b"))
                                       + ":monotone"),
                                source="consistency", venue_tag=venue, fam=fam,
                                note=f"{venue} {fam}: P(up>{cs[j]:g})={ps[j]:.3f} > P(up>{cs[i]:g})={ps[i]:.3f}"))
    return out


def _median_line(s):
    """s: P(over at c) by c (one venue or fair). Median by linear interpolation; None if not bracketed / unusable."""
    s = s.sort_index()
    cs, ps = list(s.index), list(s.values)
    if len(cs) == 1:
        return cs[0] if abs(ps[0] - 0.5) <= 0.05 else None
    for i in range(len(cs) - 1):
        if (ps[i] - 0.5) * (ps[i + 1] - 0.5) <= 0 and ps[i] != ps[i + 1]:
            return cs[i] + (ps[i] - 0.5) * (cs[i + 1] - cs[i]) / (ps[i] - ps[i + 1])
    k = int(np.argmin(np.abs(np.array(ps) - 0.5)))
    return cs[k] if abs(ps[k] - 0.5) <= 0.05 else None


def tt_vs_total(R, fair):
    """(e) game total vs team totals: sum of the team-total medians vs the game-total median."""
    out = []
    VF = _venue_fair(R)
    if not len(VF):
        return out
    VF = VF.reset_index()
    VF.columns = ["espn_id", "fam", "venue", "market", "c", "p"]
    fams = ("total", "team_total:home", "team_total:away")
    VF = VF[VF.fam.isin(fams)]
    for eid, g in VF.groupby("espn_id"):
        med = {}
        for (fam, v), h in g.groupby(["fam", "venue"]):
            m = _median_line(h.groupby("c").p.mean())
            if m is not None:
                med.setdefault(fam, {})[v] = m
        if not all(f in med for f in fams):
            continue
        cons = {f: float(np.median(list(med[f].values()))) for f in fams}
        diff = cons["team_total:home"] + cons["team_total:away"] - cons["total"]
        per_v = []
        for v in set(med["total"]) & set(med["team_total:home"]) & set(med["team_total:away"]):
            d = med["team_total:home"][v] + med["team_total:away"][v] - med["total"][v]
            per_v.append((v, d))
        bad_v = [(v, d) for v, d in per_v if abs(d) >= TT_TOL]
        if abs(diff) >= TT_TOL or bad_v:
            ref = R[(R.espn_id == eid)].iloc[0]
            side = "game OVER / team-total UNDERs cheap" if diff > 0 else "game UNDER / team-total OVERs cheap"
            out.append(dict(kind="violation", legs=[ref], cost=np.nan, margin=np.nan, ev=np.nan,
                            check="e:team-totals-vs-total", source="consistency", venue_tag="CONSENSUS",
                            fam="team_total", note=(f"TT medians {cons['team_total:home']:.1f}+{cons['team_total:away']:.1f}"
                                                    f" vs total {cons['total']:.1f} (diff {diff:+.1f}; {side}); per-venue: "
                                                    + ", ".join(f"{v} {d:+.1f}" for v, d in sorted(per_v, key=lambda t: -abs(t[1]))[:6]))))
    return out
