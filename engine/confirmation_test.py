"""Does a rating model's agreement sharpen the validated strategies? (generalizing the UFC/OpenSkill finding)
For every historical strategy bet, compute pre-game Elo's opinion of the side bet vs the de-vigged market:
agree = Elo gives the bet side >= the market's probability. Split ROI by discovery/holdout."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from backtest import run_team as rt
from backtest.features import build
from backtest.core import perf
ROOT = Path(__file__).resolve().parents[1]

def main():
    L = pd.read_csv(ROOT / "results" / "ledger.csv", parse_dates=["date"])
    rows = []
    for sp in ["nfl", "ncaaf", "nhl", "nba", "ncaam", "mlb"]:
        split = json.loads((ROOT / "results" / f"{sp}_meta.json").read_text())["split"]
        g = build(rt.load(sp), rt.ELO[sp])
        g["event"] = g.away + " @ " + g.home
        x = L[(L.sport == sp)].merge(g[["date", "event", "home", "p_home_elo", "p_home_mkt"]], on=["date", "event"], how="inner")
        x = x[~x.pick.str.contains("OVER|UNDER")]          # totals: Elo has no opinion
        if x.empty:
            continue
        home_side = x.apply(lambda r: r.pick.startswith(str(r.home)), axis=1)
        elo = np.where(home_side, x.p_home_elo, 1 - x.p_home_elo)
        mkt = np.where(home_side, x.p_home_mkt, 1 - x.p_home_mkt)
        x["agree"] = elo >= mkt
        for strat, y in x.groupby("strategy"):
            for agree, z in y.groupby("agree"):
                for name, m in (("discovery", z.season < split), ("holdout", z.season >= split)):
                    p = perf(z.pnl[m])
                    if p.get("n"):
                        rows.append(dict(strategy=strat, elo_agrees=agree, sample=name, n=p["n"], roi=round(p["roi"], 4), z=round(p["z"], 2)))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results" / "confirmation_test.csv", index=False)
    print(R.pivot_table(index=["strategy", "elo_agrees"], columns="sample", values=["n", "roi"]).to_string())

if __name__ == "__main__":
    main()
