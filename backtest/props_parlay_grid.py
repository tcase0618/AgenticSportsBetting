"""Grid: leg filters (main-line only, price window) x prob model x EV threshold, for singles and
same-book cross-game parlays. Operating point is chosen on DISCOVERY (2024-25) only."""
import itertools, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from .core import perf
from .props_nba import pick_bets, DISC, HOLD
from .props_nba_deep import fit_stack, parlays
OUT = Path(__file__).resolve().parents[1] / "results"

def legs(p, col, thr, lo, hi):
    q = p[(p.alt_line == 0)]
    B = pick_bets(q, col, thr)
    B = B[(B.price >= lo) & (B.price <= hi)]
    B["edge"] = np.where(B.side == "over", B[col], 1 - B[col]) - 1 / B.dec
    return B.sort_values("edge", ascending=False)

def parl(B, k, rr):
    B = B.copy(); B["ev"] = B["edge"]  # rank by probability edge
    return parlays(B, k, 1, rr)

if __name__ == "__main__":
    p = pd.read_csv(OUT / "nba_props_frame.csv.gz", parse_dates=["date"], low_memory=False)
    p, _ = fit_stack(p)
    p["p_blend"] = 0.75 * p.p_mkt + 0.25 * p.p_mod
    rows = []
    for col in ("p_blend", "p_stack"):
        for thr in (0.02, 0.04, 0.06):
            for lo, hi in ((-160, 130), (-135, 115)):
                B = legs(p, col, thr, lo, hi)
                for name, T in (("single", B.assign(dec=B.dec)), ("rr2", parl(B, 2, True)), ("rr3", parl(B, 3, True)),
                                ("top2", parl(B, 2, False)), ("top3", parl(B, 3, False))):
                    for s in (DISC, HOLD):
                        x = T[T.season == s]
                        if len(x) < 30: continue
                        r = perf(x.pnl)
                        rows.append(dict(prob=col, thr=thr, window=f"{lo}/{hi}", structure=name, season=s, n=r["n"],
                                         hit=(x.pnl > 0).mean(), roi=r["roi"], z=r["z"], dd=r["max_dd"],
                                         avg_dec=x.dec.mean()))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "nba_props_parlay_grid.csv", index=False)
    W = R.pivot_table(index=["prob", "thr", "window", "structure"], columns="season", values=["n", "roi", "z", "avg_dec"])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(W.round(3).to_string())
