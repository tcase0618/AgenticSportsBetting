"""Simulate running the nightly card every night: tier-staked singles + NBA props + 2-leg parlays
(A+B legs, different events) exactly as nightly/card.py builds them. Units: 1u = 1% of starting bankroll."""
import itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from .core import american_to_decimal
from .parlay import load_ledger
OUT = Path(__file__).resolve().parents[1] / "results"
STAKE = {"A": 1.0, "B": 0.75, "C": 0.5, "prop": 0.5, "parlay2": 0.25}

def nightly(L, props=None, use_parlays=True):
    rows = []
    P = props.groupby("date") if props is not None else {}
    dates = sorted(set(L.date) | (set(props.date) if props is not None else set()))
    for d in dates:
        day = L[L.date == d]
        singles = float((day.pnl * day.tier.map(STAKE)).sum())
        risk = float(day.tier.map(STAKE).sum())
        par = 0.0
        if use_parlays:
            core = day[day.tier.isin(["A", "B"])]
            for (_, a), (_, b) in itertools.combinations(core.iterrows(), 2):
                if (a.outcome < 0) or (b.outcome < 0):
                    par -= STAKE["parlay2"]
                else:
                    live = [x for x in (a, b) if x.outcome > 0]
                    par += STAKE["parlay2"] * (np.prod([x.dec for x in live]) - 1) if live else 0
                risk += STAKE["parlay2"]
        pr = 0.0
        if props is not None and d in P.groups:
            x = P.get_group(d)
            pr = float(x.pnl.sum() * STAKE["prop"]); risk += len(x) * STAKE["prop"]
        rows.append(dict(date=d, singles=singles, parlays=par, props=pr, total=singles + par + pr, risk=risk))
    return pd.DataFrame(rows)

def summary(N, label):
    c = N.total.cumsum()
    by_m = N.set_index("date").total.resample("ME").sum()
    return dict(window=label, nights=len(N), units=round(N.total.sum(), 1), risked=round(N.risk.sum(), 1),
                roi=round(N.total.sum() / N.risk.sum(), 4), win_nights=round((N.total > 0).mean(), 3),
                best_night=round(N.total.max(), 2), worst_night=round(N.total.min(), 2),
                max_dd=round(float((c.cummax() - c).max()), 1), pos_months=f"{(by_m > 0).sum()}/{(by_m != 0).sum()}",
                singles=round(N.singles.sum(), 1), parlays=round(N.parlays.sum(), 1), props=round(N.props.sum(), 1))

if __name__ == "__main__":
    L = load_ledger()
    props = pd.read_csv(OUT / "nba_props_bets_p_blend_0.04.csv", parse_dates=["date"])
    out = []
    H = L[L.holdout]
    N = nightly(H); out.append(summary(N, "All holdout seasons, game strategies + parlays (no props)"))
    N.to_csv(OUT / "nightly_sim_holdout.csv", index=False)
    W = L[(L.date >= "2024-10-01") & (L.date <= "2026-06-30")]
    P = props[(props.date >= "2024-10-01")]
    N2 = nightly(W, P); out.append(summary(N2, "Oct 2024 - Jun 2026, full card incl. NBA props"))
    N2.to_csv(OUT / "nightly_sim_2024_2026.csv", index=False)
    N3 = nightly(W, P, use_parlays=False); out.append(summary(N3, "Oct 2024 - Jun 2026, singles + props only"))
    S = pd.DataFrame(out); S.to_csv(OUT / "nightly_sim_summary.csv", index=False)
    pd.set_option("display.width", 250); print(S.T.to_string())
