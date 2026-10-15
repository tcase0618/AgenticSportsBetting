"""DFS pick'em (PrizePicks / Underdog / Pick6) legs vs sportsbook consensus, NFL 2026.
Fair P(over DFS line) comes from the de-vigged median of sportsbook quotes, shifted to the DFS line
with a normal model (sd ~ 0.6*line + 8 for yardage, 1.3 for receptions). Entries use standard power
payouts: 2-pick 3x, 3-pick 5x, 4-pick 10x, 5-pick 20x (break-even per leg 57.7/58.5/56.2/54.9%).
FORWARD TEST ONLY: ~5 weeks of data."""
import itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
from .core import american_to_prob
ROOT = Path(__file__).resolve().parents[1]
PAY = {2: 3, 3: 5, 4: 10, 5: 20}

def sd(market, line):
    return np.where(market == "receptions", 1.3, 0.55 * line + 8)

if __name__ == "__main__":
    d = pd.read_csv(ROOT / "data/processed/nfl_props.csv")
    d = d[(d.season == 2026) & d.result.isin(["over", "under"])]
    k = ["week", "player", "market"]
    sb = d[(d.source_type == "sportsbook") & d.over_odds.notna() & d.under_odds.notna()].copy()
    po, pu = american_to_prob(sb.over_odds), american_to_prob(sb.under_odds)
    sb["mu"] = sb.line + sd(sb.market, sb.line) * norm.ppf((po / (po + pu)).clip(.02, .98))
    cons = sb.groupby(k).agg(mu=("mu", "median"), nb=("book", "nunique")).reset_index()
    f = d[d.source_type == "dfs"].merge(cons, on=k)
    f = f[f.nb >= 3]
    f["p_over"] = 1 - norm.cdf((f.line - f.mu) / sd(f.market, f.line))
    f["side"] = np.where(f.p_over >= 0.5, "over", "under")
    f["p"] = np.maximum(f.p_over, 1 - f.p_over)
    f["win"] = (f.side == f.result)
    print(f"DFS legs matched to >=3-book consensus: {len(f)} ({f.book.value_counts().to_dict()})")
    for t in (0.52, 0.55, 0.57, 0.60):
        x = f[f.p >= t]
        print(f"  legs p>={t}: n={len(x):4d} predicted={x.p.mean():.3f} actual hit={x.win.mean():.3f}")
    # entries: each week/book, combine the top legs (distinct players) into k-pick power entries
    rows = []
    for t in (0.55, 0.57):
        for kk in (2, 3, 4, 5):
            res = []
            for (w, b), x in f[f.p >= t].groupby(["week", "book"]):
                x = x.sort_values("p", ascending=False).drop_duplicates("player").head(10)
                for c in itertools.combinations(range(len(x)), kk):
                    L = x.iloc[list(c)]
                    if L.team.nunique() < 2: continue          # apps require 2+ teams
                    res.append(PAY[kk] - 1 if L.win.all() else -1)
            if res:
                r = np.array(res, float)
                rows.append(dict(min_p=t, picks=kk, entries=len(r), hit=(r > 0).mean(), roi=r.mean()))
    print(pd.DataFrame(rows).round(3).to_string())
