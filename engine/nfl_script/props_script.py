"""
Script-conditional NFL player prop projections.

  projection = player share (rolling, shift-safe EWMA of prior games)
             x team projected volume (walk-forward game-script model, 'mkt' variant)
             x player efficiency (shrunk EWMA) x opponent adjustment (defense allowed vs league, pre-game)

Mean is linearly recalibrated and the outcome distribution is the empirical
distribution of actual/projection in the same projection bin, both fitted on
prior seasons only. Evaluated against real prop lines in data/processed/nfl_props.csv.

usage: python -m engine.nfl_script.props_script
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.core import american_to_decimal, no_vig_probs, perf, settle
from .tendencies import ROOT, CACHE, tendencies

OUT = ROOT / "results"
HL = 4.0                      # EWMA half-life in games
DEC = 0.5 ** (1 / HL)
MARKETS = ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "completions"]
OPP_EXP = 0.5                 # partial opponent adjustment (fixed a priori)
EV_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
W_GRID = [0.25, 0.5, 0.75, 1.0]
STARTER_SHARE = 0.96   # mean starter share of team attempts, 2006-2015


def _ewsum(df, col, key="player_id"):
    """Decayed sum of a player's PRIOR games (excludes current row)."""
    m = df.groupby(key)[col].transform(lambda s: s.ewm(halflife=HL).mean().shift(1))
    n = df.groupby(key).cumcount()
    w = (1 - DEC ** n) / (1 - DEC)
    return (m * w).fillna(0.0)


def player_frame() -> pd.DataFrame:
    w = pd.read_csv(ROOT / "data" / "processed" / "nfl_player_week.csv.gz")
    w = w[w.season >= 2004]
    w["position"] = w["position"].replace({"FB": "RB"})
    team = w.groupby(["game_id", "team"])[["attempts", "carries", "targets"]].sum().add_prefix("team_").reset_index()
    w = w.merge(team, on=["game_id", "team"], how="left")
    w = w[w.position.isin(["QB", "RB", "WR", "TE"])].copy()
    # listed starting QB (nflverse games table, known at kickoff): QB props are conditional on starting
    g = pd.read_csv(ROOT / "data" / "raw" / "nfl_games.csv", usecols=["game_id", "home_qb_id", "away_qb_id"])
    st = pd.concat([g[["game_id", "home_qb_id"]].set_axis(["game_id", "player_id"], axis=1),
                    g[["game_id", "away_qb_id"]].set_axis(["game_id", "player_id"], axis=1)]).dropna()
    st["qb_start"] = 1
    w = w.merge(st.drop_duplicates(), on=["game_id", "player_id"], how="left")
    w["qb_start"] = w["qb_start"].fillna(0)
    w["gameday"] = pd.to_datetime(w["gameday"])
    w = w.sort_values(["player_id", "gameday", "game_id"]).reset_index(drop=True)
    cols = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards", "attempts", "completions",
            "passing_yards", "team_attempts", "team_carries"]
    E = {c: _ewsum(w, c) for c in cols}
    w["n_prior"] = w.groupby("player_id").cumcount()
    w["tgt_share"] = E["targets"] / E["team_attempts"].replace(0, np.nan)
    w["car_share"] = E["carries"] / E["team_carries"].replace(0, np.nan)
    w["att_share"] = E["attempts"] / E["team_attempts"].replace(0, np.nan)
    # position priors (constants from pre-2016 seasons: no look-ahead into any test season)
    pr = w[w.season < 2016]
    def prior(num, den, pos):
        x = pr[pr.position == pos]
        return x[num].sum() / max(x[den].sum(), 1)
    for pos in ("WR", "TE", "RB", "QB"):
        m = w.position == pos
        w.loc[m, "ypt"] = (E["receiving_yards"][m] + 25 * prior("receiving_yards", "targets", pos)) / (E["targets"][m] + 25)
        w.loc[m, "cr"] = (E["receptions"][m] + 25 * prior("receptions", "targets", pos)) / (E["targets"][m] + 25)
        w.loc[m, "ypc"] = (E["rushing_yards"][m] + 40 * prior("rushing_yards", "carries", pos)) / (E["carries"][m] + 40)
    q = w.position == "QB"
    w.loc[q, "ypa"] = (E["passing_yards"][q] + 80 * prior("passing_yards", "attempts", "QB")) / (E["attempts"][q] + 80)
    w.loc[q, "cmp"] = (E["completions"][q] + 80 * prior("completions", "attempts", "QB")) / (E["attempts"][q] + 80)
    return w


def add_script(w: pd.DataFrame) -> pd.DataFrame:
    P = pd.read_parquet(CACHE / "script_mkt.parquet")[["game_id", "team", "p_pass_att", "p_rush_att", "p_pts"]]
    Pp = pd.read_parquet(CACHE / "script_pure.parquet")[["game_id", "team", "p_pass_att", "p_rush_att"]]
    P = P.merge(Pp.rename(columns={"p_pass_att": "pp_pass_att", "p_rush_att": "pp_rush_att"}), on=["game_id", "team"])
    # mkt variant needs a closing line; fall back to the pure script when missing
    P["p_pass_att"] = P["p_pass_att"].fillna(P["pp_pass_att"])
    P["p_rush_att"] = P["p_rush_att"].fillna(P["pp_rush_att"])
    T = tendencies()
    d = [c for c in T.columns if c.startswith("def_") and (c.endswith("_std") or c.endswith("_lg"))
         and any(k in c for k in ("ypt_", "cr_", "ypc", "ypa", "cmp"))]
    D = T[["game_id", "team"] + d].rename(columns={"team": "opp"})
    w = w.merge(P, on=["game_id", "team"], how="left").merge(D, on=["game_id", "opp"], how="left")
    return w


def raw_projection(w: pd.DataFrame) -> pd.DataFrame:
    def adj(stat):
        return (w[f"def_{stat}_std"] / w[f"def_{stat}_lg"]).clip(0.6, 1.6).fillna(1.0) ** OPP_EXP
    pos = w["position"].where(w["position"].isin(["WR", "TE", "RB"]), "WR")
    a_ypt = pd.Series(1.0, index=w.index)
    a_cr = pd.Series(1.0, index=w.index)
    for p in ("WR", "TE", "RB"):
        m = pos == p
        a_ypt[m] = adj(f"ypt_{p}")[m]
        a_cr[m] = adj(f"cr_{p}")[m]
    tg = w["tgt_share"].fillna(0) * w["p_pass_att"]
    w["raw_receptions"] = tg * w["cr"] * a_cr
    w["raw_receiving_yards"] = tg * w["ypt"] * a_ypt
    w["raw_rushing_yards"] = w["car_share"].fillna(0) * w["p_rush_att"] * w["ypc"] * adj("ypc")
    # starting QB: share of team attempts a starter takes (pre-2016 constant), not his backup history
    att = np.where(w["qb_start"].eq(1), STARTER_SHARE, w["att_share"].fillna(0)) * w["p_pass_att"]
    w["raw_passing_yards"] = att * w["ypa"] * adj("ypa")
    w["raw_completions"] = att * w["cmp"] * adj("cmp")
    return w


def eligible(w: pd.DataFrame, market: str) -> pd.Series:
    base = (w["n_prior"] >= 1) & w["p_pass_att"].notna()
    if market in ("receiving_yards", "receptions"):
        return base & w.position.isin(["WR", "TE", "RB"]) & (w["tgt_share"] >= 0.05)
    if market == "rushing_yards":
        return base & w.position.isin(["RB", "QB"]) & (w["car_share"] >= 0.08)
    return base & w.position.eq("QB") & w["qb_start"].eq(1)


class Dist:
    """Walk-forward calibration (linear) + empirical ratio distribution by projection bin."""

    def __init__(self, tr: pd.DataFrame, market: str, nbins: int = 10):
        x, y = tr[f"raw_{market}"].values, tr[market].values
        A = np.c_[np.ones_like(x), x]
        self.b = np.linalg.lstsq(A, y, rcond=None)[0]
        mu = self.mean(x)
        self.edges = np.unique(np.quantile(mu, np.linspace(0, 1, nbins + 1)[1:-1]))
        bins = np.digitize(mu, self.edges)
        self.ratio = {k: (y[bins == k] / np.maximum(mu[bins == k], 0.5)) for k in np.unique(bins)}

    def mean(self, x):
        return np.maximum(self.b[0] + self.b[1] * np.asarray(x, float), 0.5)

    def sd_p(self, x, line):
        mu = self.mean(x)
        bins = np.digitize(mu, self.edges)
        sd, p = np.empty(len(mu)), np.empty(len(mu))
        for i, (m, k, l) in enumerate(zip(mu, bins, line)):
            r = self.ratio[k] * m
            sd[i] = r.std()
            win, lose = (r > l).sum(), (r < l).sum()
            p[i] = win / max(win + lose, 1)   # P(over | not push)
        return mu, sd, p


def market_frame() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data" / "processed" / "nfl_props.csv")
    p = p[p.game_id.notna() & p.player_id.notna()].copy()
    p["q_id"] = np.arange(len(p))
    sb = p.source_type.eq("sportsbook") & p.over_odds.notna() & p.under_odds.notna()
    po, _, _ = no_vig_probs(p.loc[sb, "over_odds"], p.loc[sb, "under_odds"])
    p.loc[sb, "mkt_p_over_book"] = po
    # medians must be taken on DECIMAL odds (American odds are discontinuous at +/-100)
    p.loc[sb, "over_dec"] = american_to_decimal(p.loc[sb, "over_odds"])
    p.loc[sb, "under_dec"] = american_to_decimal(p.loc[sb, "under_odds"])
    g = p[sb].groupby(["game_id", "player_id", "market", "line"])
    cons = g.agg(mkt_p_over=("mkt_p_over_book", "median"), n_books=("book", "nunique"),
                 med_over_dec=("over_dec", "median"), med_under_dec=("under_dec", "median")).reset_index()
    p = p.merge(cons, on=["game_id", "player_id", "market", "line"], how="left")
    return p


def project(w: pd.DataFrame, season: int, market: str, rows: pd.DataFrame, lines) -> tuple:
    tr = w[(w.season >= season - 6) & (w.season < season) & eligible(w, market) & w[market].notna()]
    D = Dist(tr, market)
    return D.sd_p(rows[f"raw_{market}"].values, lines)


def betting(B: pd.DataFrame, p_col: str, ev: float) -> pd.DataFrame:
    """One bet per (game, player, market) at the consensus line, at the median book price."""
    d = B.copy()
    po, pu = d[p_col], 1 - d[p_col]
    ev_o = po * d["med_over_dec"] - 1
    ev_u = pu * d["med_under_dec"] - 1
    d["side"] = np.where(ev_o >= ev_u, "over", "under")
    d["ev"] = np.maximum(ev_o, ev_u)
    d = d[d["ev"] >= ev].copy()
    s = np.sign(d["actual"] - d["line"])
    d["outcome"] = np.where(d.side == "over", s, -s)
    d["dec"] = np.where(d.side == "over", d["med_over_dec"], d["med_under_dec"])
    d["pnl"] = np.where(d["outcome"] > 0, d["dec"] - 1, np.where(d["outcome"] < 0, -1.0, 0.0))
    return d


def fmt(x: pd.DataFrame) -> str:
    q = perf(x["pnl"])
    return f"n={q['n']}, ROI={q.get('roi', float('nan')):+.1%}, z={q.get('z', float('nan')):+.2f}" if q["n"] else "n=0"


def run():
    w = raw_projection(add_script(player_frame()))
    p = market_frame()
    p = p.merge(w[["game_id", "player_id", "n_prior", "tgt_share", "car_share", "att_share", "p_pass_att",
                   "p_rush_att"] + [f"raw_{m}" for m in MARKETS]], on=["game_id", "player_id"], how="left")
    p["model_mean"] = np.nan
    p["model_sd"] = np.nan
    p["model_p_over"] = np.nan
    for (season, market), idx in p.groupby(["season", "market"]).groups.items():
        r = p.loc[idx]
        r = r[r[f"raw_{market}"].notna()]
        if not len(r):
            continue
        mu, sd, po = project(w, season, market, r, r["line"].values)
        p.loc[r.index, ["model_mean", "model_sd", "model_p_over"]] = np.c_[mu, sd, po]
    keep = ["season", "week", "game_id", "player", "player_id", "position", "team", "opp", "market", "book",
            "source_type", "line", "over_odds", "under_odds", "mkt_p_over_book", "mkt_p_over", "n_books",
            "p_pass_att", "p_rush_att", "model_mean", "model_sd", "model_p_over", "actual", "result"]
    OUT.mkdir(exist_ok=True)
    p[keep].round(4).to_csv(OUT / "nfl_script_props.csv", index=False)

    # ---- one bettable unit per player-game-market: consensus line = line with most books
    sb = p[p.source_type.eq("sportsbook") & p.mkt_p_over.notna() & p.model_p_over.notna() & p.actual.notna()]
    B = (sb.sort_values(["n_books", "line"], ascending=[False, True])
         .drop_duplicates(["game_id", "player_id", "market"]))
    B = B[["season", "week", "game_id", "player", "market", "line", "n_books", "mkt_p_over", "med_over_dec",
           "med_under_dec", "model_mean", "model_sd", "model_p_over", "actual"]].copy()
    out = []
    # accuracy vs the line
    for (s, m), x in B.groupby([B.season.where(B.season < 2026, 2026), "market"]):
        mae_model = (x.actual - x.model_mean).abs().mean()
        mae_line = (x.actual - x.line).abs().mean()
        brier_m = ((x.model_p_over - (x.actual > x.line)) ** 2)[x.actual != x.line].mean()
        brier_k = ((x.mkt_p_over - (x.actual > x.line)) ** 2)[x.actual != x.line].mean()
        out.append(f"ACC {s} {m}: n={len(x)}  MAE model {mae_model:.2f} vs line {mae_line:.2f}; "
                   f"Brier model {brier_m:.4f} vs market {brier_k:.4f}")
    # model-only at fixed EV thresholds, by market & side
    for ev in (0.05, 0.10):
        bo = betting(B.assign(p=B.model_p_over), "p", ev)
        for (s, m, sd), x in bo.groupby([bo.season.where(bo.season < 2026, 2026), "market", "side"]):
            out.append(f"MODEL-ONLY ev>={ev:.2f} {s} {m} {sd}: {fmt(x)}")
    # blend: weight & EV threshold chosen on a calibration window, tested out of sample
    def choose(cal):
        best = None
        for wgt in W_GRID:
            for ev in EV_GRID:
                b = betting(cal.assign(p=wgt * cal.model_p_over + (1 - wgt) * cal.mkt_p_over), "p", ev)
                if len(b) < 30:
                    continue
                if best is None or b.pnl.mean() > best[2]:
                    best = (wgt, ev, b.pnl.mean(), len(b))
        return best
    for lab, cal, test in (
        ("completions: choose on 2017, test 2018", B[(B.season == 2017)], B[(B.season == 2018)]),
        ("2026: choose on wk1-2, test wk3-5", B[(B.season == 2026) & (B.week <= 2)], B[(B.season == 2026) & B.week.between(3, 5)]),
    ):
        best = choose(cal)
        if best is None:
            out.append(f"BLEND {lab}: no config with n>=30")
            continue
        wgt, ev = best[0], best[1]
        b = betting(test.assign(p=wgt * test.model_p_over + (1 - wgt) * test.mkt_p_over), "p", ev)
        flag = "" if best[2] > 0 else "  [no config profitable on calibration -> rule says NO BETS; test shown for transparency]"
        out.append(f"BLEND {lab}: w={wgt}, ev>={ev} (cal ROI {best[2]:+.1%}, n={best[3]}) -> TEST {fmt(b)}{flag}")
        for (m, sd), x in b.groupby(["market", "side"]):
            out.append(f"    test {m} {sd}: {fmt(x)}")
    return p, B, out


if __name__ == "__main__":
    p, B, out = run()
    print("\n".join(out))
