"""
Rebuild each sport's chosen KEY strategy from raw data and write a full
performance dossier (results/KEYS.md + results/keys.json).

The key for each sport is chosen in keys.py from the validated shortlist in
results/<sport>_strategies.csv (discovered on train seasons, positive on holdout).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .core import perf, american_to_decimal
from .keys import KEYS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def rebuild(sport: str):
    if sport == "ufc":
        from .run_ufc import prep, strategies
        f = prep(pd.read_csv(ROOT / "data" / "processed" / "ufc.csv"))
        f = f[f.a_ml.notna() & f.b_ml.notna()]
        S, _, _ = strategies(f)
        return f, S
    from . import run_team as rt
    from .features import build
    from .strategies import market_pnl, side_strategies, total_strategies
    from .model import walk_forward, model_bets
    g = build(rt.load(sport), rt.ELO.get(sport))
    meta = json.loads((OUT / f"{sport}_meta.json").read_text())
    split, Sc = meta["split"], meta["S"]
    pnl = market_pnl(g)
    tq = g.loc[g.season < split, "total"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    S = side_strategies(g, pnl, Sc, sport)
    S.update(total_strategies(g, pnl, Sc, sport, tq))
    if any(k[2] == "model" for k in [KEYS[sport]["key"]]):
        first = int(np.sort(g.season.unique())[min(5, g.season.nunique() - 1)])
        extra = [c for c in rt.MODEL_EXTRA.get(sport, []) if c in g]
        S.update(model_bets(g, pnl, walk_forward(g, pnl, first, extra=extra)))
    return g, S


def fmt_pct(x):
    return f"{x * 100:+.1f}%"


def dossier(sport: str) -> tuple[str, dict]:
    spec = KEYS[sport]
    key = tuple(spec["key"])
    g, S = rebuild(sport)
    s = S[key]
    meta = json.loads((OUT / f"{sport}_meta.json").read_text())
    split = meta["split"]
    season = g["season"].loc[s.index]
    A = perf(s, seasons=season)
    tr, ho = perf(s[season.values < split]), perf(s[season.values >= split])
    last_full = sorted(season.unique())
    recent_seasons = last_full[-3:]
    rc = perf(s[season.isin(recent_seasons).values])
    by = s.groupby(season.values).agg(bets="count", units="sum", roi="mean")
    by["record"] = s.groupby(season.values).apply(lambda x: f"{(x > 0).sum()}-{(x < 0).sum()}-{(x == 0).sum()}")
    # average price actually bet
    rec = dict(sport=sport, name=spec["name"], rule=spec["rule"], why=spec["why"], key=list(key),
               split=split, all=A, train=tr, holdout=ho, recent=rc, recent_seasons=[int(x) for x in recent_seasons],
               by_season={int(k): dict(bets=int(v.bets), units=round(float(v.units), 2), roi=round(float(v.roi), 4),
                                       record=v.record) for k, v in by.iterrows()})
    md = [f"## {spec['name']}  ({sport.upper()})",
          f"**Rule:** {spec['rule']}",
          f"**Why it works:** {spec['why']}",
          "",
          "| Sample | Bets | W-L-P | Win% | ROI | Units | z | p (one-sided) |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, p in (("All seasons", A), (f"Discovery (<{split})", tr), (f"Holdout ({split}+)", ho),
                   (f"Last 3 seasons ({recent_seasons[0]}-{recent_seasons[-1]})", rc)):
        if p.get("n"):
            md.append(f"| {lab} | {p['n']} | {p['wins']}-{p['losses']}-{p['pushes']} | {p['win_pct']*100:.1f}% | "
                      f"{fmt_pct(p['roi'])} | {p['units']:+.1f} | {p['z']:.2f} | {p['p']:.4f} |")
    md += ["", f"Profitable seasons: **{A['pos_seasons']}/{A['seasons']}** · Max drawdown: **{A['max_dd']:.1f}u** · "
               f"95% CI on ROI: {fmt_pct(A['ci_lo'])} to {fmt_pct(A['ci_hi'])}", "",
           "<details><summary>Season by season</summary>", "", "| Season | Bets | Record | Units | ROI |", "|---|---|---|---|---|"]
    for k, v in rec["by_season"].items():
        md.append(f"| {k} | {v['bets']} | {v['record']} | {v['units']:+.2f} | {fmt_pct(v['roi'])} |")
    md += ["", "</details>", ""]
    if spec.get("runners_up"):
        md.append("**Runners-up that also survived holdout:** " + "; ".join(spec["runners_up"]))
        md.append("")
    return "\n".join(md), rec


if __name__ == "__main__":
    import sys
    sports = sys.argv[1:] or list(KEYS)
    parts, recs = [], {}
    for sp in sports:
        md, r = dossier(sp)
        parts.append(md)
        recs[sp] = r
        print(md)
    (OUT / "keys.json").write_text(json.dumps(recs, indent=2, default=float))
    (OUT / "KEYS_body.md").write_text("\n\n".join(parts))
