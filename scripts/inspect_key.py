"""Quick per-season inspection of candidate strategies: python scripts/inspect_key.py nfl 'tot|wind15|game|under|all' ..."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import pandas as pd
from backtest import run_team as rt
from backtest.features import build
from backtest.strategies import market_pnl, side_strategies, total_strategies
from backtest.core import perf
sport = sys.argv[1]
g = build(rt.load(sport), rt.ELO.get(sport)); meta = json.load(open(f"results/{sport}_meta.json"))
pnl = market_pnl(g); tq = g.loc[g.season < meta["split"], "total"].quantile([.2,.4,.6,.8]).to_dict()
S = side_strategies(g, pnl, meta["S"], sport); S.update(total_strategies(g, pnl, meta["S"], sport, tq))
for arg in sys.argv[2:]:
    k = tuple(arg.split("|")); s = S[k]; se = g.season.loc[s.index]
    p = perf(s, seasons=se); r = perf(s[se >= se.max() - 2])
    print(k, {x: round(p[x], 3) for x in ["n", "roi", "z", "win_pct", "units", "max_dd", "pct_pos_seasons"]}, "last3:", r["n"], round(r["roi"], 3))
    print(s.groupby(se.values).agg(["count", "sum"]).round(1).T.to_string())
