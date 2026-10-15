"""
MLB pitcher strikeout props: Statcast (pybaseball) projection + market blend, tested on real 2026 K lines.

Projection for a starter's next game, from games BEFORE it only:
  K/BF rate   : shrunk blend of last-5-start, season and career rates toward league average
  whiff, CSW  : per-pitch rates (the strongest public predictors of K rate)
  batters faced expectation: recent BF per start
  opponent K% : the opposing lineup's strikeout rate over its previous 30 days
Count distribution: negative binomial (strikeouts are overdispersed vs Poisson), dispersion fit on training
residuals. Market probability: de-vigged DraftKings/FanDuel line (odds-ratio method).

Walk-forward: model fit only on games before 2026-07-01. Blend weight and EV threshold chosen on the
discovery window (Jul-Aug 2026), reported on holdout (Sep-Oct 2026).

  python -m engine.mlb_k_model
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
from backtest.core import american_to_decimal, perf
from engine.devig import devig

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
TRAIN_END = "2026-07-01"
DISC_END = "2026-09-01"
LEAGUE_K = 0.22


def norm(s: str) -> str:
    s = str(s)
    if "," in s:                                     # Statcast "Last, First" -> "First Last"
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower())
    return re.sub(r"[^a-z]", "", s)


def features() -> pd.DataFrame:
    g = pd.read_csv(PROC / "mlb_statcast_pitcher_games.csv.gz", parse_dates=["game_date"])
    g = g.sort_values(["game_date", "game_pk"])
    # opponent lineup K rate over the previous 30 days (all pitchers they faced), shift-safe by date
    team_day = g.groupby(["opp", "game_date"]).agg(k=("k", "sum"), bf=("bf", "sum")).reset_index().sort_values("game_date")
    def roll30(x):
        x = x.set_index("game_date")
        k = x.k.rolling("30D").sum().shift(1)
        b = x.bf.rolling("30D").sum().shift(1)
        return pd.DataFrame({"opp_k30": (k / b).values}, index=x.index)
    ok = team_day.groupby("opp", group_keys=True).apply(roll30).reset_index()
    s = g[g.starter == 1].copy()
    s = s.merge(ok, on=["opp", "game_date"], how="left")
    s = s.sort_values(["pitcher", "game_date"])
    gp = s.groupby("pitcher", sort=False)
    for c in ("k", "bf", "pitches", "whiffs", "csw"):
        s[f"{c}_l5"] = gp[c].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        s[f"{c}_car"] = gp[c].transform(lambda x: x.expanding().sum().shift(1))
    s["season"] = s.game_date.dt.year
    gs = s.groupby(["pitcher", "season"], sort=False)
    s["k_szn"] = gs.k.transform(lambda x: x.expanding().sum().shift(1))
    s["bf_szn"] = gs.bf.transform(lambda x: x.expanding().sum().shift(1))
    s["starts_car"] = gp.cumcount()
    # shrunk rates: add 'prior batters' at league average
    prior = 150
    s["k_rate"] = (s.k_car.fillna(0) + LEAGUE_K * prior) / (s.bf_car.fillna(0) + prior)
    s["k_rate_l5"] = (s.k_l5.fillna(0) + LEAGUE_K * 40) / (s.bf_l5.fillna(0) + 40)
    s["k_rate_szn"] = (s.k_szn.fillna(0) + LEAGUE_K * 80) / (s.bf_szn.fillna(0) + 80)
    s["whiff_rate"] = (s.whiffs_car.fillna(0) + 0.11 * 600) / (s.pitches_car.fillna(0) + 600)
    s["csw_rate"] = (s.csw_car.fillna(0) + 0.29 * 600) / (s.pitches_car.fillna(0) + 600)
    s["whiff_l5"] = (s.whiffs_l5.fillna(0) + 0.11 * 200) / (s.pitches_l5.fillna(0) + 200)
    s["bf_per_start_l5"] = s.bf_l5 / gp.cumcount().clip(upper=5).replace(0, np.nan)
    s["pitches_per_start_l5"] = s.pitches_l5 / gp.cumcount().clip(upper=5).replace(0, np.nan)
    s["opp_k30"] = s.opp_k30.fillna(LEAGUE_K)
    s["exp_k_simple"] = s.k_rate_szn * s.bf_per_start_l5.fillna(22) * (s.opp_k30 / LEAGUE_K)
    s["key"] = s.player_name.map(norm)
    return s


FEATS = ["k_rate", "k_rate_l5", "k_rate_szn", "whiff_rate", "csw_rate", "whiff_l5", "bf_per_start_l5",
         "pitches_per_start_l5", "opp_k30", "home", "starts_car", "exp_k_simple"]


def fit_predict(s: pd.DataFrame) -> pd.DataFrame:
    tr = s[(s.game_date < TRAIN_END) & (s.starts_car >= 1)]
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, min_samples_leaf=80,
                                      l2_regularization=1.0).fit(tr[FEATS], tr.k)
    s = s.copy()
    s["mu"] = m.predict(s[FEATS])
    # negative binomial dispersion: var = mu + mu^2 / r, fit by moments on training residuals
    t = s[s.game_date < TRAIN_END]
    excess = ((t.k - t.mu) ** 2 - t.mu).mean()
    r = float(max((t.mu ** 2).mean() / max(excess, 1e-6), 2.0))
    s["nb_r"] = r
    print(f"train starts {len(tr)}, NB dispersion r={r:.1f}, train MAE {np.abs(t.k - t.mu).mean():.3f}")
    return s


def p_over(mu, line, r):
    p = r / (r + mu)
    return 1 - nbinom.cdf(np.floor(line), r, p)


def evaluate(s: pd.DataFrame):
    props = pd.read_csv(PROC / "mlb_props.csv", parse_dates=["date"])
    props = props[props.result.isin(["over", "under"]) & props.over_odds.notna() & props.under_odds.notna()]
    props["key"] = props.player.map(norm)
    q = props.merge(s[["game_date", "key", "mu", "nb_r", "k"]].rename(columns={"game_date": "date"}),
                    on=["date", "key"], how="inner")
    print(f"props matched to projections: {len(q)} / {len(props)}")
    q["p_mod"] = p_over(q.mu, q.line, q.nb_r)
    q["p_mkt"], _ = devig(q.over_odds.values, q.under_odds.values, "odds_ratio")
    q["y"] = (q.result == "over").astype(int)
    q["window"] = np.where(q.date < DISC_END, "discovery", "holdout")
    q["dec_o"], q["dec_u"] = american_to_decimal(q.over_odds), american_to_decimal(q.under_odds)
    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    for w in ("discovery", "holdout"):
        x = q[q.window == w]
        print(f"{w}: n={len(x)} logloss market={ll(x.p_mkt, x.y):.4f} model={ll(x.p_mod, x.y):.4f} "
              f"blend25={ll(0.75*x.p_mkt+0.25*x.p_mod, x.y):.4f} over-rate={x.y.mean():.3f}")
    rows = []
    for wgt in (0.0, 0.25, 0.5, 1.0):
        q["pb"] = (1 - wgt) * q.p_mkt + wgt * q.p_mod
        for thr in (0.02, 0.04, 0.06, 0.08):
            eo, eu = q.pb * q.dec_o - 1, (1 - q.pb) * q.dec_u - 1
            side_over = eo >= eu
            ev = np.where(side_over, eo, eu)
            win = np.where(side_over, q.y == 1, q.y == 0)
            dec = np.where(side_over, q.dec_o, q.dec_u)
            b = q.assign(ev=ev, pnl=np.where(win, dec - 1, -1.0), over=side_over)[ev >= thr]
            b = b.sort_values("ev", ascending=False).drop_duplicates(["date", "key"])   # best book per pitcher
            for w in ("discovery", "holdout"):
                x = b[b.window == w]
                p = perf(x.pnl) if len(x) else dict(n=0)
                rows.append(dict(model_weight=wgt, ev_thr=thr, window=w, n=p.get("n", 0), roi=p.get("roi"),
                                 z=p.get("z"), over_share=float(x.over.mean()) if len(x) else None))
    R = pd.DataFrame(rows)
    R.to_csv(RES / "mlb_k_model_test.csv", index=False)
    print(R.pivot_table(index=["model_weight", "ev_thr"], columns="window", values=["n", "roi", "z"]).round(3).to_string())
    d = R[(R.window == "discovery") & (R.n >= 40)].sort_values("z", ascending=False).head(1)
    if len(d):
        d = d.iloc[0]
        h = R[(R.window == "holdout") & (R.model_weight == d.model_weight) & (R.ev_thr == d.ev_thr)].iloc[0]
        print(f"\nchosen on discovery: weight {d.model_weight}, EV>={d.ev_thr}: discovery {d.roi*100:+.1f}% (n={d.n}) "
              f"-> holdout {h.roi*100 if h.roi == h.roi else float('nan'):+.1f}% (n={h.n})")


if __name__ == "__main__":
    s = fit_predict(features())
    s[["game_date", "game_pk", "pitcher", "player_name", "pitch_team", "opp", "mu", "nb_r", "k"]].to_csv(
        RES / "mlb_k_projections.csv.gz", index=False, compression="gzip")
    evaluate(s)
