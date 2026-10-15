"""Ad-hoc NCAAM angle screen, train <2020 / holdout 2020+."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import pandas as pd, numpy as np
from backtest import run_team as rt
from backtest.features import build
from backtest.strategies import market_pnl
from backtest.core import perf
g = build(rt.load("ncaam"), rt.ELO["ncaam"]); P = market_pnl(g); SPLIT = 2020
g["month"] = g.date.dt.month
def show(name, s):
    s = s.dropna(); se = g.season.loc[s.index]
    a, b, c = perf(s[se < SPLIT]), perf(s[se >= SPLIT]), perf(s, seasons=se)
    if a.get("n", 0) < 100: return
    print(f"{name:50s} tr n={a['n']:5d} roi={a['roi']:+.3f} p={a['p']:.3f} | ho n={b.get('n',0):5d} roi={b.get('roi',np.nan):+.3f} z={b.get('z',np.nan):+.2f} | all {c['roi']:+.3f} n={c['n']} pos={c['pct_pos_seasons']:.2f}")
neu = g.neutral == 1
for nm, m in {"neutral": neu, "neutral_march": neu & (g.month == 3), "neutral_nov_dec": neu & g.month.isin([11, 12]),
              "postseason": g.playoff == 1, "march_all": g.month == 3, "nonneutral": ~neu}.items():
    show(f"{nm} under", P["tot_u"][m]); show(f"{nm} over", P["tot_o"][m])
    show(f"{nm} dog ATS", pd.concat([P["ats_h"][m & (g.spread_home > 0)], P["ats_a"][m & (g.spread_home < 0)]]).sort_index())
    show(f"{nm} fav ATS", pd.concat([P["ats_h"][m & (g.spread_home < 0)], P["ats_a"][m & (g.spread_home > 0)]]).sort_index())
    show(f"{nm} big dog 10+ ATS", pd.concat([P["ats_h"][m & (g.spread_home >= 10)], P["ats_a"][m & (g.spread_home <= -10)]]).sort_index())
hp, ap = g.get("h_power"), g.get("a_power")
if hp is not None:
    pm_h = (hp == 0) & (ap == 1); pm_a = (ap == 0) & (hp == 1)
    show("mid-major HOME vs power ATS", P["ats_h"][pm_h])
    show("mid-major AWAY vs power ATS", P["ats_a"][pm_a])
    show("mid-major vs power ATS (any)", pd.concat([P["ats_h"][pm_h], P["ats_a"][pm_a]]).sort_index())
    show("mid vs power neutral ATS", pd.concat([P["ats_h"][pm_h & neu], P["ats_a"][pm_a & neu]]).sort_index())
    show("mid vs power march ATS", pd.concat([P["ats_h"][pm_h & (g.month == 3)], P["ats_a"][pm_a & (g.month == 3)]]).sort_index())
    show("power vs power under", P["tot_u"][(hp == 1) & (ap == 1)])
    show("mid vs mid over", P["tot_o"][(hp == 0) & (ap == 0)])
for lo, hi in [(0, 125), (125, 135), (135, 145), (145, 155), (155, 250)]:
    m = (g.total >= lo) & (g.total < hi)
    show(f"total [{lo},{hi}) under", P["tot_u"][m]); show(f"total [{lo},{hi}) over", P["tot_o"][m])
for lo, hi in [(0.5, 3), (3, 7), (7, 12), (12, 20), (20, 60)]:
    show(f"home dog +[{lo},{hi})", P["ats_h"][(g.spread_home >= lo) & (g.spread_home < hi) & ~neu])
    show(f"road dog +[{lo},{hi})", P["ats_a"][(g.spread_home <= -lo) & (g.spread_home > -hi) & ~neu])
