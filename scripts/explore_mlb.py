"""Ad-hoc MLB angle screen with the same train(<2019)/holdout(2019+) split."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import pandas as pd, numpy as np
from backtest import run_team as rt
from backtest.features import build
from backtest.strategies import market_pnl
from backtest.core import perf
g = build(rt.load("mlb"), rt.ELO["mlb"]); P = market_pnl(g); SPLIT = 2019
g["month"] = g.date.dt.month
def show(name, s):
    s = s.dropna(); se = g.season.loc[s.index]
    a, b, c = perf(s[se < SPLIT]), perf(s[se >= SPLIT]), perf(s, seasons=se)
    if a.get("n", 0) < 150: return
    print(f"{name:55s} tr n={a['n']:5d} roi={a['roi']:+.3f} p={a['p']:.3f} | ho n={b.get('n',0):5d} roi={b.get('roi',np.nan):+.3f} z={b.get('z',np.nan):+.2f} | all {c['roi']:+.3f} pos={c['pct_pos_seasons']:.2f}")
for m in range(3, 11):
    k = g.month == m
    show(f"tot under month={m}", P["tot_u"][k]); show(f"tot over month={m}", P["tot_o"][k])
pop = ["NYY", "LAD", "BOS", "CHC", "NYM", "SF", "PHI", "ATL", "STL"]
print(sorted(g.home.unique())[:40])
for t in pop:
    hf = (g.home == t) & (g.p_home_mkt > .5); af = (g.away == t) & (g.p_away_mkt > .5)
    show(f"fade {t} as fav", pd.concat([P["ml_a"][hf], P["ml_h"][af]]))
hf = g.home.isin(pop) & (g.p_home_mkt > .5); af = g.away.isin(pop) & (g.p_away_mkt > .5)
show("fade popular-team favs (all)", pd.concat([P["ml_a"][hf], P["ml_h"][af]]))
for lo, hi in [(0.7, 1), (0.65, 0.7), (0.6, 0.65), (0.55, 0.6), (0.5, 0.55)]:
    show(f"home fav p in [{lo},{hi})", P["ml_h"][(g.p_home_mkt >= lo) & (g.p_home_mkt < hi)])
    show(f"away fav p in [{lo},{hi})", P["ml_a"][(g.p_away_mkt >= lo) & (g.p_away_mkt < hi)])
    show(f"home dog vs fav p in [{lo},{hi})", P["ml_h"][(g.p_away_mkt >= lo) & (g.p_away_mkt < hi)])
    show(f"away dog vs fav p in [{lo},{hi})", P["ml_a"][(g.p_home_mkt >= lo) & (g.p_home_mkt < hi)])
if "dh_game" in g:
    for d in (1, 2):
        show(f"dh game {d} under", P["tot_u"][g.dh_game == d]); show(f"dh game {d} over", P["tot_o"][g.dh_game == d])
for t in sorted(g.home.unique()):
    k = g.home == t
    for side in ("o", "u"):
        s = P[f"tot_{side}"][k]; se = g.season.loc[s.dropna().index]
        a = perf(s.dropna()[se < SPLIT])
        if a.get("p", 1) < 0.02 and a["roi"] > 0: show(f"park {t} {side}", s)
for tq in [(0, 7.5), (7.5, 8.5), (8.5, 9.5), (9.5, 30)]:
    k = (g.total >= tq[0]) & (g.total < tq[1])
    show(f"under total in {tq}", P["tot_u"][k]); show(f"over total in {tq}", P["tot_o"][k])
show("playoff home dog", P["ml_h"][(g.playoff == 1) & (g.p_home_mkt < .5)])
show("playoff under", P["tot_u"][g.playoff == 1])
show("playoff road dog", P["ml_a"][(g.playoff == 1) & (g.p_away_mkt < .5)])
