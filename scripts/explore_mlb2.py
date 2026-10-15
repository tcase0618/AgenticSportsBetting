import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import pandas as pd, numpy as np
from backtest import run_team as rt
from backtest.features import build
from backtest.strategies import market_pnl
from backtest.model import walk_forward, model_bets
from backtest.core import perf, no_vig_probs
g = build(rt.load("mlb"), rt.ELO["mlb"]); P = market_pnl(g)
def show(name, s, split):
    s = s.dropna(); se = g.season.loc[s.index]
    a, b, c = perf(s[se < split]), perf(s[se >= split]), perf(s, seasons=se)
    if not a.get("n"): return
    print(f"{name:45s} tr n={a['n']:5d} roi={a['roi']:+.3f} z={a['z']:+.2f} | ho n={b.get('n',0):5d} roi={b.get('roi',np.nan):+.3f} z={b.get('z',np.nan):+.2f} | all {c['roi']:+.3f} n={c['n']}")
# ---- line movement 2021+
po_h, po_a, _ = no_vig_probs(g.home_ml_open, g.away_ml_open)
mv = g.p_home_mkt - po_h  # >0 money came in on home
for t in (0.02, 0.03, 0.05, 0.08):
    show(f"follow ML move >= {t}", pd.concat([P['ml_h'][mv >= t], P['ml_a'][mv <= -t]]).sort_index(), 2024)
    show(f"fade ML move >= {t}", pd.concat([P['ml_a'][mv >= t], P['ml_h'][mv <= -t]]).sort_index(), 2024)
    show(f"follow move, dog only >= {t}", pd.concat([P['ml_h'][(mv >= t) & (g.p_home_mkt < .5)], P['ml_a'][(mv <= -t) & (g.p_away_mkt < .5)]]).sort_index(), 2024)
# ---- walk-forward model with starter features
extra = ["h_st_wvm10", "a_st_wvm10", "h_st_ra5", "a_st_ra5", "h_st_starts", "a_st_starts"]
pr = walk_forward(g, P, 2014, extra=extra)
for k, s in model_bets(g, P, pr).items():
    show(f"{k[0]} {k[1]}", s, 2019)
