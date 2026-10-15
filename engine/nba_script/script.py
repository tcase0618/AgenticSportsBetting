"""
NBA game-script model: both teams' pre-game tendencies (+ optionally the market) ->
expected pace, expected points for each side, P(blowout = |margin| >= 15).

Walk-forward by season: the model for season S is fit only on seasons < S (from 2008, the first season
with lines in data/processed/nba.csv).  Three variants are fit side by side:
  none  : tendencies only (no betting line) -> tests whether tendencies alone carry information
  pre   : market = OPENING spread/total when present, else closing.  This is what player_script uses,
          because NBA prop snapshots are taken ~12h before tip.  NOTE: nba.csv has no opening lines for
          seasons 2023-2025 (and 19% of 2026), so for those games 'pre' falls back to the CLOSE, which is
          set after the prop snapshot (it can contain late injury news) -> a mild optimistic bias for
          2025 (discovery) and part of 2026.  The 'none' variant is the leak-proof robustness check.
  close : market = closing line (fallback open) -> used to validate against / bet totals at the close.

Validation (main): MAE of predicted totals/margins vs actual and vs the closing number, the
residual correlation corr(model - close, actual - close), and a totals betting check at -110 with the
threshold chosen on seasons 2012-2018 and reported on 2019-2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from backtest.core import perf
from backtest.props_nba import OUT, PROC
from .tendencies import tendencies

BLOWOUT = 15
FIRST_LINE_SEASON = 2008
FIRST_PRED_SEASON = 2010
DISC_END = 2018   # totals check: discovery seasons <= 2018, holdout >= 2019


def game_frame(T: pd.DataFrame | None = None) -> pd.DataFrame:
    T = tendencies() if T is None else T
    g = pd.read_csv(PROC / "nba.csv")
    g["date"] = pd.to_datetime(g["date"])
    g = g[g.season >= FIRST_LINE_SEASON].rename(columns={"espn_game_id": "game_id"})
    keep = ["game_id", "team", "team_name", "pace_ewm", "pace_r10", "ortg_ewm", "drtg_ewm", "ortg_r10", "drtg_r10",
            "net_ewm", "net_r10", "lg_pace", "lg_ortg", "rest", "b2b", "g_last7", "gp_season",
            "o_fg3a_rate_ewm", "d_fg3a_rate_ewm", "o_ft_rate_ewm", "d_ft_rate_ewm", "o_tov_pct_ewm",
            "d_tov_pct_ewm", "o_oreb_pct_ewm", "d_oreb_pct_ewm", "poss", "pace"]
    tt = T[keep]
    h = tt.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team_name": "home"})
    a = tt.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team_name": "away"})
    g = g.merge(h, on=["game_id", "home"], how="inner").merge(a, on=["game_id", "away"], how="inner")
    g = g.drop(columns=["a_lg_pace", "a_lg_ortg"]).rename(columns={"h_lg_pace": "lg_pace", "h_lg_ortg": "lg_ortg",
                                                                  "h_poss": "poss", "h_pace": "pace_act"})
    g["y_total"] = g.home_score + g.away_score
    g["y_margin"] = g.home_score - g.away_score
    g["y_blow"] = (g.y_margin.abs() >= BLOWOUT).astype(int)
    # market lines
    g["close_total"] = g.total.fillna(g.total_open)
    g["close_spread"] = g.spread_home.fillna(g.spread_open_home)
    g["pre_total"] = g.total_open.fillna(g.total)
    g["pre_spread"] = g.spread_open_home.fillna(g.spread_home)
    g["pre_src"] = np.where(g.total_open.notna(), "open", np.where(g.total.notna(), "close", "none"))
    # naive tendency script (pace & efficiency matchup, relative to league level strictly before the date)
    for k, sfx in (("", "_ewm"), ("10", "_r10")):
        pace = g[f"h_pace{sfx}"] + g[f"a_pace{sfx}"] - g.lg_pace
        hp = pace * (g[f"h_ortg{sfx}"] + g[f"a_drtg{sfx}"] - g.lg_ortg) / 100
        ap = pace * (g[f"a_ortg{sfx}"] + g[f"h_drtg{sfx}"] - g.lg_ortg) / 100
        g[f"pace_naive{k}"], g[f"tot_naive{k}"], g[f"mar_naive{k}"] = pace, hp + ap, hp - ap
    g["rest_diff"] = g.h_rest - g.a_rest
    g["b2b_diff"] = g.h_b2b - g.a_b2b
    g["fg3_env"] = g.h_o_fg3a_rate_ewm + g.a_o_fg3a_rate_ewm + g.h_d_fg3a_rate_ewm + g.a_d_fg3a_rate_ewm
    g["ft_env"] = g.h_o_ft_rate_ewm + g.a_o_ft_rate_ewm + g.h_d_ft_rate_ewm + g.a_d_ft_rate_ewm
    g["tov_env"] = g.h_o_tov_pct_ewm + g.a_o_tov_pct_ewm + g.h_d_tov_pct_ewm + g.a_d_tov_pct_ewm
    g["early"] = (np.minimum(g.h_gp_season, g.a_gp_season) < 10).astype(int)
    need = ["tot_naive", "mar_naive", "pace_naive", "tot_naive10", "mar_naive10"]
    return g.dropna(subset=need).sort_values(["date", "game_id"]).reset_index(drop=True)


BASE_TOT = ["tot_naive", "tot_naive10", "pace_naive", "fg3_env", "ft_env", "tov_env", "h_b2b", "a_b2b", "playoff",
            "early"]
BASE_MAR = ["mar_naive", "mar_naive10", "rest_diff", "b2b_diff", "playoff", "neutral"]
BASE_PACE = ["pace_naive", "pace_naive10", "tov_env", "h_b2b", "a_b2b", "playoff"]


def _fit_predict(tr, te, X, y):
    m = LinearRegression().fit(tr[X], tr[y])
    return m.predict(te[X]), m


def script_predictions(g: pd.DataFrame, variants=("none", "pre", "close")) -> pd.DataFrame:
    """Walk-forward predictions (season S fit on seasons < S) for every variant."""
    out = []
    for S in range(FIRST_PRED_SEASON, int(g.season.max()) + 1):
        tr_all, te = g[g.season < S], g[g.season == S].copy()
        for v in variants:
            mk_t, mk_s = (None, None) if v == "none" else (f"{v}_total", f"{v}_spread")
            Xt = BASE_TOT + ([mk_t] if mk_t else [])
            Xm = BASE_MAR + ([mk_s] if mk_s else [])
            Xp = BASE_PACE + ([mk_t] if mk_t else [])
            tr = tr_all.dropna(subset=Xt + Xm + ["poss"])
            ok = te[Xt + Xm].notna().all(axis=1)
            tot, mar, pace = (np.full(len(te), np.nan) for _ in range(3))
            tot[ok.values] = _fit_predict(tr, te[ok], Xt, "y_total")[0]
            mar[ok.values] = _fit_predict(tr, te[ok], Xm, "y_margin")[0]
            pace[ok.values] = _fit_predict(tr, te[ok], Xp, "poss")[0]
            # blowout: logistic on the predicted script; training rows need in-sample script preds
            trp = tr.copy()
            trp["p_tot"] = LinearRegression().fit(tr[Xt], tr.y_total).predict(tr[Xt])
            trp["p_mar"] = LinearRegression().fit(tr[Xm], tr.y_margin).predict(tr[Xm])
            te_ = te.assign(p_tot=tot, p_mar=mar)
            Xb = ["abs_mar", "mar2", "p_tot", "playoff"]
            for d in (trp, te_):
                d["abs_mar"], d["mar2"] = d.p_mar.abs(), d.p_mar ** 2
            lb = LogisticRegression(C=1.0, max_iter=500).fit(trp[Xb], trp.y_blow)
            bp = np.full(len(te), np.nan)
            bp[ok.values] = lb.predict_proba(te_.loc[ok, Xb])[:, 1]
            te[f"tot_{v}"], te[f"mar_{v}"], te[f"pace_{v}"], te[f"blow_{v}"] = tot, mar, pace, bp
        # games without a market line: fall back to the tendency-only script
        for v in variants:
            if v != "none":
                for c in ("tot", "mar", "pace", "blow"):
                    te[f"{c}_{v}"] = te[f"{c}_{v}"].fillna(te[f"{c}_none"])
        out.append(te)
    P = pd.concat(out, ignore_index=True)
    for v in variants:
        P[f"hpts_{v}"] = (P[f"tot_{v}"] + P[f"mar_{v}"]) / 2
        P[f"apts_{v}"] = (P[f"tot_{v}"] - P[f"mar_{v}"]) / 2
    return P


def team_script(P: pd.DataFrame, v: str = "pre") -> pd.DataFrame:
    """Long (game_id, team) view of one variant's script for the player model."""
    cols = ["game_id", "date", "season", f"tot_{v}", f"mar_{v}", f"pace_{v}", f"blow_{v}", f"hpts_{v}",
            f"apts_{v}", "pre_spread", "pre_total", "pre_src", "lg_pace"]
    base = P[cols + ["h_team", "a_team"]]
    h = base.assign(team=base.h_team, team_pts_pred=base[f"hpts_{v}"], opp_pts_pred=base[f"apts_{v}"],
                    margin_pred=base[f"mar_{v}"], spread_team=base.pre_spread)
    a = base.assign(team=base.a_team, team_pts_pred=base[f"apts_{v}"], opp_pts_pred=base[f"hpts_{v}"],
                    margin_pred=-base[f"mar_{v}"], spread_team=-base.pre_spread)
    L = pd.concat([h, a], ignore_index=True)
    L = L.rename(columns={f"tot_{v}": "total_pred", f"pace_{v}": "pace_pred", f"blow_{v}": "blowout_p"})
    return L[["game_id", "team", "total_pred", "margin_pred", "team_pts_pred", "opp_pts_pred", "pace_pred",
              "blowout_p", "spread_team", "pre_total", "pre_src", "lg_pace"]]


# ------------------------------------------------------------------ validation
def _bet_totals(d, pred_col, line_col, thr):
    diff = d[pred_col] - d[line_col]
    side = np.where(diff >= thr, 1, np.where(diff <= -thr, -1, 0))
    res = np.sign(d.y_total - d[line_col])
    m = side != 0
    out = np.where(res[m] == 0, 0.0, np.where(res[m] == side[m], 100 / 110, -1.0))
    return pd.Series(out)


def validate(P: pd.DataFrame) -> dict:
    d = P[P.total.notna() & P.spread_home.notna()].copy()       # true closing lines (2012+)
    d["close_mar"] = -d.spread_home
    rows = []
    for s, x in d.groupby("season"):
        r = dict(season=s, n=len(x),
                 mae_tot_close=(x.y_total - x.total).abs().mean(),
                 mae_tot_none=(x.y_total - x.tot_none).abs().mean(),
                 mae_tot_pre=(x.y_total - x.tot_pre).abs().mean(),
                 mae_tot_close_model=(x.y_total - x.tot_close).abs().mean(),
                 mae_mar_close=(x.y_margin - x.close_mar).abs().mean(),
                 mae_mar_none=(x.y_margin - x.mar_none).abs().mean(),
                 mae_mar_close_model=(x.y_margin - x.mar_close).abs().mean(),
                 rc_tot_none=np.corrcoef(x.tot_none - x.total, x.y_total - x.total)[0, 1],
                 rc_tot_close=np.corrcoef(x.tot_close - x.total, x.y_total - x.total)[0, 1],
                 rc_mar_none=np.corrcoef(x.mar_none - x.close_mar, x.y_margin - x.close_mar)[0, 1],
                 mae_pace=(x.poss - x.pace_pre).abs().mean(),
                 blow_rate=x.y_blow.mean(),
                 brier_blow=np.mean((x.blow_pre - x.y_blow) ** 2),
                 brier_blow_clim=np.mean((x.y_blow.mean() - x.y_blow) ** 2))
        rows.append(r)
    V = pd.DataFrame(rows)
    # totals betting at -110: choose threshold & variant on 2012-2018, report 2019+
    B = []
    for v in ("none", "close"):
        for thr in (0.5, 1, 1.5, 2, 3, 4, 5):
            for per, x in (("disc", d[d.season <= DISC_END]), ("hold", d[d.season > DISC_END])):
                r = perf(_bet_totals(x, f"tot_{v}", "total", thr))
                B.append(dict(variant=v, thr=thr, period=per, n=r.get("n", 0), roi=r.get("roi"), z=r.get("z")))
    B = pd.DataFrame(B)
    disc = B[(B.period == "disc") & (B.n >= 300)].sort_values("roi", ascending=False).iloc[0]   # min sample
    chosen = B[(B.variant == disc.variant) & (B.thr == disc.thr)]
    return dict(by_season=V, totals_grid=B, chosen=chosen)


if __name__ == "__main__":
    import time
    t0 = time.time()
    g = game_frame()
    P = script_predictions(g)
    P.to_csv(OUT / "nba_script_games.csv.gz", index=False)
    R = validate(P)
    pd.set_option("display.width", 220)
    print(R["by_season"].round(3).to_string())
    print(R["totals_grid"].pivot_table(index=["variant", "thr"], columns="period", values=["n", "roi", "z"]).round(3))
    print("chosen on 2012-18:\n", R["chosen"].round(3).to_string())
    print(f"{time.time() - t0:.0f}s")
