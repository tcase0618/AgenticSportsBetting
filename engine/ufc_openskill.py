"""
OpenSkill (github.com/vivekjoshy/openskill.py) fighter ratings for UFC, tested against the market.

Each fighter carries (mu, sigma): skill and uncertainty. Ratings are built chronologically; a fight only
sees results before it. Questions, answered on discovery (<2020) then holdout (2020+):
  1. Does the rating model beat or add to the de-vigged market probability (log loss)?
  2. Is there a profitable 'model disagrees with market' strategy?
  3. Does it sharpen the validated Youth Over Experience strategy (agree vs disagree)?

  python -m engine.ufc_openskill
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from backtest.core import american_to_decimal, perf, settle
from engine.devig import devig

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SPLIT = 2020


def ratings(f: pd.DataFrame) -> pd.DataFrame:
    from openskill.models import PlackettLuce
    model = PlackettLuce()
    R = {}
    pa, sa, sb, ma, mb = [], [], [], [], []
    for r in f.itertuples(index=False):
        A = R.setdefault(r.fighter_a, model.rating(name=r.fighter_a))
        B = R.setdefault(r.fighter_b, model.rating(name=r.fighter_b))
        pa.append(model.predict_win([[A], [B]])[0])
        ma.append(A.mu); mb.append(B.mu); sa.append(A.sigma); sb.append(B.sigma)
        if r.winner in ("A", "B"):
            ranks = [1, 2] if r.winner == "A" else [2, 1]
            [[A2], [B2]] = model.rate([[A], [B]], ranks=ranks)
            R[r.fighter_a], R[r.fighter_b] = A2, B2
    f = f.copy()
    f["p_os"], f["mu_a"], f["mu_b"], f["sig_a"], f["sig_b"] = pa, ma, mb, sa, sb
    pd.DataFrame([dict(name=k, mu=v.mu, sigma=v.sigma) for k, v in R.items()]).to_csv(
        RES / "ufc_openskill_current.csv", index=False)          # ratings after every fight to date
    return f


def main():
    f = pd.read_csv(ROOT / "data" / "processed" / "ufc.csv", parse_dates=["date"]).sort_values("date")
    f = ratings(f)
    o = f[f.a_ml.notna() & f.b_ml.notna() & f.winner.isin(["A", "B"])].copy()
    o["season"] = o.date.dt.year
    o["p_mkt"], _ = devig(o.a_ml.values, o.b_ml.values, "odds_ratio")
    o["y"] = (o.winner == "A").astype(int)
    out = []
    # 1. log loss: market, model, blends (blend weight picked on discovery)
    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    disc, hold = o[o.season < SPLIT], o[o.season >= SPLIT]
    best_w, best = 0.0, 9
    for w in np.linspace(0, 0.5, 11):
        v = ll((1 - w) * disc.p_mkt + w * disc.p_os, disc.y)
        if v < best:
            best, best_w = v, w
    for name, d in (("discovery", disc), ("holdout", hold)):
        out.append(dict(test="log_loss", sample=name, market=ll(d.p_mkt, d.y), openskill=ll(d.p_os, d.y),
                        blend=ll((1 - best_w) * d.p_mkt + best_w * d.p_os, d.y), blend_weight=best_w, n=len(d)))
    print(pd.DataFrame(out).round(5).to_string(index=False))
    # 2. disagreement strategy: bet the side the model likes by >= t (vs the vigged price)
    rows = []
    o["pnl_a"] = settle(np.where(o.y == 1, 1, -1), o.a_ml)
    o["pnl_b"] = settle(np.where(o.y == 0, 1, -1), o.b_ml)
    for t in (0.03, 0.05, 0.08, 0.10, 0.15):
        for sig_filter in ("any", "uncertain_opp"):
            a = (o.p_os - o.p_mkt >= t)
            b = ((1 - o.p_os) - (1 - o.p_mkt) >= t)
            if sig_filter == "uncertain_opp":   # model's edge comes from the opponent being unproven
                a &= o.sig_b > o.sig_a + 1
                b &= o.sig_a > o.sig_b + 1
            s = pd.concat([o.pnl_a[a], o.pnl_b[b]])
            se = o.season.loc[s.index]
            for name, m in (("discovery", se < SPLIT), ("holdout", se >= SPLIT)):
                p = perf(s[m.values])
                if p.get("n"):
                    rows.append(dict(threshold=t, filter=sig_filter, sample=name, n=p["n"], roi=p["roi"], z=p["z"]))
    D = pd.DataFrame(rows)
    print("\nmodel-vs-market disagreement bets:")
    print(D.pivot_table(index=["filter", "threshold"], columns="sample", values=["n", "roi", "z"]).round(3).to_string())
    # 3. Youth Over Experience, split by whether OpenSkill agrees
    L = pd.read_csv(RES / "ledger.csv", parse_dates=["date"])
    Y = L[L.strategy == "Youth Over Experience"].copy()
    o["event"] = o.fighter_a + " vs " + o.fighter_b
    Y = Y.merge(o[["date", "event", "fighter_a", "p_os", "p_mkt"]], on=["date", "event"], how="left")
    Y["side_a"] = Y.pick.str.replace(" ML", "") == Y.fighter_a
    Y["model_edge"] = np.where(Y.side_a, Y.p_os - Y.p_mkt, (1 - Y.p_os) - (1 - Y.p_mkt))
    Y["season"] = Y.date.dt.year
    print("\nYouth Over Experience split by OpenSkill agreement:")
    for lab, m in (("model agrees (edge>=0)", Y.model_edge >= 0), ("model disagrees", Y.model_edge < 0)):
        for name, s in (("discovery", Y.season < SPLIT), ("holdout", Y.season >= SPLIT)):
            p = perf(Y.pnl[m & s])
            print(f"  {lab:24s} {name:9s} n={p.get('n',0):4d} roi={p.get('roi',0)*100:+.1f}% z={p.get('z',0):+.2f}")
    D.to_csv(RES / "ufc_openskill_test.csv", index=False)
    f[["date", "fighter_a", "fighter_b", "p_os", "mu_a", "mu_b", "sig_a", "sig_b"]].to_csv(RES / "ufc_openskill_ratings.csv", index=False)


def p_win(a: str, b: str, table: pd.DataFrame | None = None) -> float | None:
    """Current OpenSkill win probability for fighter a over b (None if either is unrated)."""
    from openskill.models import PlackettLuce
    from nightly.live import norm_name
    t = table if table is not None else pd.read_csv(RES / "ufc_openskill_current.csv")
    t = t.assign(k=t.name.map(norm_name)).drop_duplicates("k").set_index("k")
    ka, kb = norm_name(a), norm_name(b)
    if ka not in t.index or kb not in t.index:
        return None
    m = PlackettLuce()
    A, B = m.rating(mu=t.loc[ka, "mu"], sigma=t.loc[ka, "sigma"]), m.rating(mu=t.loc[kb, "mu"], sigma=t.loc[kb, "sigma"])
    return float(m.predict_win([[A], [B]])[0])


if __name__ == "__main__":
    main()
