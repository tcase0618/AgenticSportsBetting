"""
NCAAF adapter for the script-then-learn backtest (see engine/scripts/SPEC.md).

Pipeline (strictly sequential, one game at a time in date order):
  1. CORE SCRIPT  - base.ScriptEngine (Kalman off/def team ratings, learned level/HFA/noise/rho), warmed up from
                    2008. Subclassed only to give a never-seen FCS team a sensible prior (mean of FCS ratings so far).
  2. ENRICHMENTS  - pre-game features built only from earlier games:
        tempo     EWMA scrimmage plays per game (cfbfastR play-by-play) -> expected plays in this game
        eff       opponent-adjusted EWMA offensive / defensive EPA per play and success rate
        passrush  pass-rate x pass-EPA and rush-EPA matchup terms (offense tendency vs opposing defense)
        conf      online conference strength (learned from non-conference script errors)
        rest      days of rest, bye week, short week
        site      neutral site, postseason/bowl, early-season (ratings still uncertain)
        p5g5      Power-5 vs Group-of-5 / FBS vs FCS mismatch
        scale     core margin / total (lets the learner fix systematic over/under-shrinkage)
  3. LEARNED CORRECTION - recursive least squares (forgetting 0.999) of the core's margin and total errors on the
                    enrichment features, updated after every game ("learning from mistakes").
  4. MARKET BLEND - Bates-Granger weight of the enriched script vs the closing line, learned on past errors.

Outputs: results/scripts/ncaaf_games.parquet, results/scripts/ncaaf_learning.md
Ablation (2021+): drop each enrichment group, re-run the correction layer, compare MAE.

  python -m engine.scripts.ncaaf
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine.scripts.base import Params, ScriptEngine

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ncaaf" / "pbp"
OUT = ROOT / "results" / "scripts"
WARM, START, TUNE_END, HOLD = 2008, 2016, 2020, 2021

P5 = {"SEC", "Big Ten", "ACC", "Big 12", "Pac-12", "Pac-10", "Big East"}


def is_p5(conf, season, team):
    if team == "Notre Dame":
        return True
    if conf == "Pac-12" and season >= 2024:
        return False
    if conf == "Big East" and season >= 2013:
        return False
    return conf in P5


# ------------------------------------------------------------------ data
def pbp_team_games() -> pd.DataFrame:
    """Per game & side (home/away offense): plays, pass rate, EPA/play pass & rush, success rate. Cached."""
    cache = RAW / "team_games.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(RAW.glob("pbp_*.parquet")):
        names = pq.ParquetFile(f).schema.names
        if "is_home" in names:                                   # cfbfastR-data schema (<=2021)
            d = pq.read_table(f, columns=["game_id", "is_home", "EPA", "rush", "pass"]).to_pandas()
            d["home_off"] = d.is_home.astype(bool)
        else:                                                    # sportsdataverse release schema (2022+)
            d = pq.read_table(f, columns=["game_id", "pos_team", "home", "EPA", "rush", "pass"]).to_pandas()
            d["home_off"] = d.pos_team == d.home
        d["rush"], d["pass"] = d["rush"].fillna(0).astype(bool), d["pass"].fillna(0).astype(bool)
        d = d[(d.rush | d["pass"]) & d.EPA.notna()]
        d["succ"] = (d.EPA > 0).astype(float)
        d["epa_p"] = np.where(d["pass"], d.EPA, np.nan)
        d["epa_r"] = np.where(d.rush & ~d["pass"], d.EPA, np.nan)
        g = d.groupby(["game_id", "home_off"]).agg(plays=("EPA", "size"), pass_rate=("pass", "mean"),
                                                   epa=("EPA", "mean"), epa_pass=("epa_p", "mean"),
                                                   epa_rush=("epa_r", "mean"), succ=("succ", "mean")).reset_index()
        rows.append(g)
        print("pbp", f.name, len(g) // 2, "games")
    t = pd.concat(rows, ignore_index=True)
    t["game_id"] = t.game_id.astype("int64")
    t = t.drop_duplicates(["game_id", "home_off"])
    t.to_parquet(cache)
    return t


def load_games() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data" / "processed" / "ncaaf.csv", low_memory=False)
    l = pd.read_csv(ROOT / "data" / "live" / "ncaaf.csv", low_memory=False)
    l = l[l.completed.astype(bool)]
    # 2026 live rows: conference = team's latest known conference; never-seen team = FCS
    last = pd.concat([p[["date", "home", "home_conference", "home_division"]].set_axis(["date", "team", "conf", "div"], axis=1),
                      p[["date", "away", "away_conference", "away_division"]].set_axis(["date", "team", "conf", "div"], axis=1)])
    last = last.sort_values("date").groupby("team").last()
    def conf(t): return last.conf.get(t, "FCS")
    def div(t): return last["div"].get(t, "fcs")
    lv = pd.DataFrame(dict(date=l.date, season=2026, home=l.home, away=l.away, home_score=l.home_score,
                           away_score=l.away_score, neutral=l.neutral, spread_home=l.spread_home, total=l.total,
                           home_ml=l.home_ml, away_ml=l.away_ml, spread_open_home=np.nan, total_open=np.nan,
                           game_id=l.espn_id, week=np.nan, postseason=l.playoff, conference_game=l.conference_game,
                           home_conference=l.home.map(conf), away_conference=l.away.map(conf),
                           home_division=l.home.map(div), away_division=l.away.map(div)))
    g = pd.concat([p, lv], ignore_index=True)
    g = g[g.season >= WARM].copy()
    g["date"] = pd.to_datetime(g.date)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    for s in ("home", "away"):
        g[f"{s}_conference"] = g[f"{s}_conference"].fillna("FCS")
        fcs = (g[f"{s}_division"].fillna("fcs") == "fcs")
        g.loc[fcs, f"{s}_conference"] = "FCS"
    g["week"] = g.groupby("season").date.transform(lambda d: ((d - d.min()).dt.days // 7) + 1).where(g.week.isna(), g.week)
    return g


# ------------------------------------------------------------------ core engine with FCS prior
class CFBEngine(ScriptEngine):
    """Same Kalman core; a never-seen FCS team starts at the running mean rating of FCS teams (not at 0)."""
    fcs: set = set()

    def _team(self, t, season):
        new = t not in self.teams
        s = super()._team(t, season)
        if new and t in self.fcs:
            xs = [v.x for k, v in self.teams.items() if k in self.fcs and v.games >= 1]
            s.x = np.mean(xs, axis=0) if len(xs) >= 5 else np.array([-10.0, 10.0])
        return s


CORE = dict(init_level=27.0, init_hfa=3.0, init_sd=11.0, prior_var=40.0, q_game=0.6, carry=0.65,
            season_var=4.0, lr_level=0.002)


def run_core(g: pd.DataFrame, **kw) -> pd.DataFrame:
    p = Params(**{**CORE, **kw})
    e = CFBEngine(p)
    fcs = set(g.loc[g.home_conference == "FCS", "home"]) | set(g.loc[g.away_conference == "FCS", "away"])
    fbs = set(g.loc[g.home_conference != "FCS", "home"]) | set(g.loc[g.away_conference != "FCS", "away"])
    e.fcs = fcs - fbs
    r = e.run(g[["date", "season", "home", "away", "home_score", "away_score", "neutral", "spread_home", "total"]])
    # base.run re-sorts by date (unstable within a date): realign rows to g's order by (date, home, away)
    key = lambda d: pd.to_datetime(d.date).astype("int64").astype(str) + "|" + d.home + "|" + d.away
    r.index = key(r)
    r = r[~r.index.duplicated()].reindex(key(g).values)
    r.index = g.index
    return r


# ------------------------------------------------------------------ enrichment features (online)
class Ewma:
    def __init__(self, alpha, init):
        self.a, self.init, self.v, self.season = alpha, init, {}, {}

    def get(self, k, season, league=None):
        if k not in self.v:
            return self.init if league is None else league
        if self.season[k] != season:                       # regress halfway to league at a new season
            base = self.init if league is None else league
            self.v[k] = 0.6 * self.v[k] + 0.4 * base
            self.season[k] = season
        return self.v[k]

    def upd(self, k, x, season):
        if x is None or not np.isfinite(x):
            return
        if k not in self.v:
            self.v[k], self.season[k] = x, season
        else:
            self.get(k, season)
            self.v[k] += self.a * (x - self.v[k])


GROUPS = {
    "tempo": ["plays_exp", "plays_diff"],
    "eff": ["epa_net", "epa_sum", "succ_net"],
    "passrush": ["pass_match", "rush_match", "pass_match_sum", "rush_match_sum", "passrate_sum"],
    "conf": ["conf_diff"],
    "rest": ["rest_diff", "bye_h", "bye_a", "short_h", "short_a"],
    "site": ["neutral_f", "post", "early"],
    "p5g5": ["p5_vs_g5", "fbs_vs_fcs"],
    "scale": ["core_margin", "core_total"],
}
FEATS = [f for v in GROUPS.values() for f in v]


def build_features(g: pd.DataFrame, core: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One pass in date order. Features for game i use only games with an earlier date (same-day games are
    processed as a batch: all scripted before any is learned)."""
    pb = pbp.set_index(["game_id", "home_off"])
    plays = Ewma(0.25, 70.0); prate = Ewma(0.2, 0.47)
    off = {k: Ewma(0.2, 0.0) for k in ("epa", "epa_pass", "epa_rush", "succ")}
    dfn = {k: Ewma(0.2, 0.0) for k in ("epa", "epa_pass", "epa_rush", "succ")}
    lg = dict(plays=70.0, epa=0.0, epa_pass=0.05, epa_rush=-0.02, succ=0.42)
    conf_r = Ewma(0.03, 0.0)
    last_date = {}
    feats = []
    core = core.set_index(g.index)
    for d, block in g.groupby("date", sort=True):
        for i, r in block.iterrows():
            s = int(r.season)
            h, a = r.home, r.away
            ph, pa = plays.get(h, s, lg["plays"]), plays.get(a, s, lg["plays"])
            o = {k: (off[k].get(h, s), off[k].get(a, s)) for k in off}
            df = {k: (dfn[k].get(h, s), dfn[k].get(a, s)) for k in dfn}
            prh, pra = prate.get(h, s), prate.get(a, s)
            ch, ca = r.home_conference, r.away_conference
            nonconf = ch != ca
            cd = (conf_r.get(ch, s) - conf_r.get(ca, s)) if nonconf else 0.0
            rh = (d - last_date[h]).days if h in last_date else np.nan
            ra = (d - last_date[a]).days if a in last_date else np.nan
            rh_ = min(rh, 21) if rh == rh and rh < 60 else 7
            ra_ = min(ra, 21) if ra == ra and ra < 60 else 7
            p5h, p5a = is_p5(ch, s, h), is_p5(ca, s, a)
            fh, fa = ch != "FCS", ca != "FCS"
            feats.append(dict(
                idx=i,
                plays_exp=ph + pa - 2 * lg["plays"], plays_diff=ph - pa,
                plays_h=ph, plays_a=pa,
                epa_net=(o["epa"][0] + df["epa"][1]) - (o["epa"][1] + df["epa"][0]),
                epa_sum=(o["epa"][0] + df["epa"][1]) + (o["epa"][1] + df["epa"][0]),
                succ_net=(o["succ"][0] + df["succ"][1]) - (o["succ"][1] + df["succ"][0]),
                pass_match=prh * (o["epa_pass"][0] + df["epa_pass"][1]) - pra * (o["epa_pass"][1] + df["epa_pass"][0]),
                rush_match=(1 - prh) * (o["epa_rush"][0] + df["epa_rush"][1]) - (1 - pra) * (o["epa_rush"][1] + df["epa_rush"][0]),
                pass_match_sum=prh * (o["epa_pass"][0] + df["epa_pass"][1]) + pra * (o["epa_pass"][1] + df["epa_pass"][0]),
                rush_match_sum=(1 - prh) * (o["epa_rush"][0] + df["epa_rush"][1]) + (1 - pra) * (o["epa_rush"][1] + df["epa_rush"][0]),
                passrate_sum=prh + pra - 2 * 0.47, pass_rate_h=prh, pass_rate_a=pra,
                conf_diff=cd,
                rest_diff=rh_ - ra_, bye_h=float(rh_ >= 13), bye_a=float(ra_ >= 13),
                short_h=float(rh_ <= 5), short_a=float(ra_ <= 5),
                rest_h=rh, rest_a=ra,
                neutral_f=float(r.neutral or 0), post=float(r.postseason or 0), early=float((r.week or 9) <= 2),
                p5_vs_g5=float(p5h and fh and not p5a and fa) - float(p5a and fa and not p5h and fh),
                fbs_vs_fcs=float(fh and not fa) - float(fa and not fh),
                core_margin=core.at[i, "margin"], core_total=core.at[i, "total"] - 2 * core.at[i, "level"],
            ))
        # ---- reveal & learn (after the whole date is scripted)
        for i, r in block.iterrows():
            s = int(r.season)
            h, a, gid = r.home, r.away, r.game_id
            if r.home_score != r.home_score:
                continue
            last_date[h] = last_date[a] = d
            ch, ca = r.home_conference, r.away_conference
            if ch != ca:  # conference strength learns from the core's non-conference margin error
                e = (r.home_score - r.away_score) - core.at[i, "margin"]
                conf_r.upd(ch, conf_r.get(ch, s) + e / 2, s)
                conf_r.upd(ca, conf_r.get(ca, s) - e / 2, s)
            for side, t, opp in ((True, h, a), (False, a, h)):
                try:
                    x = pb.loc[(int(gid), side)]
                except KeyError:
                    continue
                plays.upd(t, x.plays, s)
                prate.upd(t, x.pass_rate, s)
                for k in off:  # opponent adjustment: observed minus opponent's current tendency
                    v = x[k]
                    if v == v:
                        off[k].upd(t, v - dfn[k].get(opp, s) - lg[k] * 0, s)
                        dfn[k].upd(opp, v - off[k].get(t, s), s)
    return pd.DataFrame(feats).set_index("idx").loc[g.index]


# ------------------------------------------------------------------ RLS correction
class RLS:
    def __init__(self, k, lam=0.999, delta=10.0):
        self.th = np.zeros(k); self.P = np.eye(k) * delta; self.lam = lam

    def pred(self, x):
        return float(self.th @ x)

    def upd(self, x, y):
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        self.th = self.th + k * (y - self.th @ x)
        self.P = (self.P - np.outer(k, Px)) / self.lam


class Norm:
    """Online feature standardisation from past games only."""
    def __init__(self, k):
        self.n, self.m, self.s = 0, np.zeros(k), np.zeros(k)

    def z(self, x):
        if self.n < 50:
            return np.zeros_like(x)
        sd = np.sqrt(self.s / (self.n - 1)); sd[sd < 1e-9] = 1
        return np.clip((x - self.m) / sd, -5, 5)

    def upd(self, x):
        self.n += 1; dlt = x - self.m; self.m += dlt / self.n; self.s += dlt * (x - self.m)


def correct(g, core, F, feats, lam=0.999, delta=1.0, shrink=1.0):
    """Online RLS of core margin/total errors on features; same-date games scripted before any update."""
    X = F[feats].fillna(0).values.astype(float) if feats else np.zeros((len(g), 0))
    k = X.shape[1] + 1
    rm, rt, nz = RLS(k, lam, delta), RLS(k, lam, delta), Norm(X.shape[1])
    cm = np.zeros(len(g)); ct = np.zeros(len(g))
    ym = (g.home_score - g.away_score).values - core.margin.values
    yt = (g.home_score + g.away_score).values - core.total.values
    dates = g.date.values
    i = 0
    n = len(g)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        Z = [np.r_[1.0, nz.z(X[t])] for t in range(i, j)]
        for t, z in zip(range(i, j), Z):
            cm[t], ct[t] = shrink * rm.pred(z), shrink * rt.pred(z)
        for t, z in zip(range(i, j), Z):
            if np.isfinite(ym[t]):
                rm.upd(z, ym[t]); rt.upd(z, yt[t]); nz.upd(X[t])
        i = j
    return cm, ct, (rm.th, rt.th)


def learned_blend(own, mkt, act, window=1500):
    """Bates-Granger combination weight learned from past errors only (same as base.ScriptEngine.blend)."""
    w = np.full(len(own), 0.5); eo, em = [], []
    cur = 0.5
    for i in range(len(own)):
        w[i] = cur
        if np.isfinite(mkt[i]) and np.isfinite(act[i]):
            eo.append(own[i] - act[i]); em.append(mkt[i] - act[i])
            if len(eo) > window:
                eo, em = eo[-window:], em[-window:]
            if len(eo) >= 100:
                a, b = np.array(eo), np.array(em)
                den = np.mean((a - b) ** 2)
                if den > 1e-9:
                    cur = float(np.clip(np.mean((a - b) * a) / den, 0, 1))
    return w


def devig_ml(h, a):
    def ip(o):
        o = np.where(np.abs(o) < 100, np.nan, o)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))
    ph, pa = ip(h), ip(a)
    return ph / (ph + pa)


def assemble(g, core, F, cm, ct):
    df = pd.DataFrame(dict(game_id=g.game_id.values, date=g.date.values, season=g.season.values,
                           week=g.week.values, home=g.home.values, away=g.away.values,
                           home_conference=g.home_conference.values, away_conference=g.away_conference.values,
                           neutral=g.neutral.values, postseason=g.postseason.values))
    df["margin_core"], df["total_core"] = core.margin.values, core.total.values
    df["margin"], df["total"] = core.margin.values + cm, core.total.values + ct
    df["mu_home"], df["mu_away"] = (df.total + df.margin) / 2, (df.total - df.margin) / 2
    sd_m, sd_t = core.sd_margin.values, core.sd_total.values
    df["sd_margin"], df["sd_total"] = sd_m, sd_t
    df["sd_home"], df["sd_away"], df["rho"] = core.sd_home.values, core.sd_away.values, core.rho.values
    df["spread_home"], df["total_line"] = g.spread_home.values, g.total.values
    df["spread_open_home"], df["total_open"] = g.spread_open_home.values, g.total_open.values
    df["home_ml"], df["away_ml"] = g.home_ml.values, g.away_ml.values
    df["home_score"], df["away_score"] = g.home_score.values, g.away_score.values
    act_m = (g.home_score - g.away_score).values.astype(float)
    act_t = (g.home_score + g.away_score).values.astype(float)
    mkt_m = -g.spread_home.values.astype(float); mkt_t = g.total.values.astype(float)
    w = learned_blend(df.margin.values, mkt_m, act_m)
    df["w_mkt"] = np.where(np.isfinite(mkt_m) & np.isfinite(mkt_t), w, np.nan)
    ww = np.nan_to_num(df.w_mkt.values, nan=0.0)
    df["margin_b"] = np.where(np.isfinite(mkt_m), ww * mkt_m + (1 - ww) * df.margin, df.margin)
    df["total_b"] = np.where(np.isfinite(mkt_t), ww * mkt_t + (1 - ww) * df.total, df.total)
    for k in ["plays_h", "plays_a", "pass_rate_h", "pass_rate_a", "rest_h", "rest_a", "conf_diff", "p5_vs_g5",
              "fbs_vs_fcs", "epa_net", "epa_sum", "pass_match", "rush_match"]:
        df[k] = F[k].values
    df["plays_script"] = df.plays_h + df.plays_a
    df["p_home_win"] = norm.cdf(df.margin / df.sd_margin)
    df["p_home_win_b"] = norm.cdf(df.margin_b / df.sd_margin)
    df["p_home_mkt"] = devig_ml(df.home_ml.values, df.away_ml.values)
    df["err_margin"], df["err_total"] = df.margin - act_m, df.total - act_t
    df["err_margin_core"], df["err_total_core"] = df.margin_core - act_m, df.total_core - act_t
    df["err_margin_b"], df["err_total_b"] = df.margin_b - act_m, df.total_b - act_t
    df["err_margin_mkt"], df["err_total_mkt"] = mkt_m - act_m, mkt_t - act_t
    return df


# ------------------------------------------------------------------ reporting helpers
def mae(x):
    return float(np.nanmean(np.abs(x)))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def holdout_mae(df, cm, ct, core, g, mask):
    act_m = (g.home_score - g.away_score).values; act_t = (g.home_score + g.away_score).values
    return mae((core.margin.values + cm - act_m)[mask]), mae((core.total.values + ct - act_t)[mask])


def ats_test(df, col, line_col, act_col, tune, hold, kind):
    """Bet side when script disagrees with the closing line by > tau (tau chosen on 2016-2020) at -110."""
    res = []
    d = df.dropna(subset=[line_col, act_col])
    for tau in [1, 2, 3, 4, 5, 6, 8]:
        for nm, m in (("tune", tune), ("hold", hold)):
            x = d[m.loc[d.index]]
            edge = x[col] - x[line_col]
            bet = edge.abs() > tau
            res_side = np.sign(x[act_col] - x[line_col])
            win = (np.sign(edge) == res_side)[bet & (res_side != 0)]
            n = len(win)
            hit = win.mean() if n else np.nan
            res.append(dict(kind=kind, tau=tau, split=nm, bets=n, hit=hit, roi=(hit * 1.909 - 1) if n else np.nan))
    return pd.DataFrame(res)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_games()
    pbp = pbp_team_games()
    print(f"games {len(g)}  pbp coverage {g.game_id.isin(pbp.game_id).mean():.2f}  {time.time()-t0:.0f}s")

    # ---- tune core Kalman params on 2016-2020 margin+total MAE (small grid)
    tune_m = ((g.season >= START) & (g.season <= TUNE_END)).values
    hold_m = (g.season >= HOLD).values
    act_m = (g.home_score - g.away_score).values; act_t = (g.home_score + g.away_score).values
    best, grid = None, []
    for pv in (60.0, 120.0):
        for qg in (0.15, 0.3):
            for car in (0.75, 0.85):
                c = run_core(g, prior_var=pv, q_game=qg, carry=car)
                s = mae((c.margin.values - act_m)[tune_m]) + mae((c.total.values - act_t)[tune_m])
                grid.append(dict(prior_var=pv, q_game=qg, carry=car, tune_mae_sum=s))
                if best is None or s < best[0]:
                    best = (s, dict(prior_var=pv, q_game=qg, carry=car))
    print("core tuned", best, f"{time.time()-t0:.0f}s")
    core = run_core(g, **best[1])
    core.index = g.index
    F = build_features(g, core, pbp)
    print(f"features built {time.time()-t0:.0f}s")

    # ---- tune RLS forgetting on 2016-2020
    bestr = None
    for lam in (0.995, 0.998, 0.999, 0.9995):
        for delta in (0.1, 1.0):
            cm, ct, _ = correct(g, core, F, FEATS, lam, delta)
            s = sum(holdout_mae(None, cm, ct, core, g, tune_m))
            if bestr is None or s < bestr[0]:
                bestr = (s, lam, delta)
    lam, delta = bestr[1], bestr[2]
    print("rls tuned", bestr)
    cm, ct, (thm, tht) = correct(g, core, F, FEATS, lam, delta)
    df = assemble(g, core, F, cm, ct)

    # ---- ablation: drop each group (and core-only) ; reported 2016-2020 and 2021+
    abl = []
    base_t = holdout_mae(None, cm, ct, core, g, tune_m); base_h = holdout_mae(None, cm, ct, core, g, hold_m)
    abl.append(dict(variant="all enrichments", tune_mae_margin=base_t[0], tune_mae_total=base_t[1],
                    hold_mae_margin=base_h[0], hold_mae_total=base_h[1]))
    z = np.zeros(len(g))
    abl.append(dict(variant="core only (no correction)", **dict(zip(
        ["tune_mae_margin", "tune_mae_total"], holdout_mae(None, z, z, core, g, tune_m))), **dict(zip(
        ["hold_mae_margin", "hold_mae_total"], holdout_mae(None, z, z, core, g, hold_m)))))
    for grp, cols in GROUPS.items():
        fs = [f for f in FEATS if f not in cols]
        a, b, _ = correct(g, core, F, fs, lam, delta)
        t_, h_ = holdout_mae(None, a, b, core, g, tune_m), holdout_mae(None, a, b, core, g, hold_m)
        abl.append(dict(variant=f"drop {grp}", tune_mae_margin=t_[0], tune_mae_total=t_[1],
                        hold_mae_margin=h_[0], hold_mae_total=h_[1]))
    for grp, cols in GROUPS.items():
        a, b, _ = correct(g, core, F, cols, lam, delta)
        h_ = holdout_mae(None, a, b, core, g, hold_m)
        abl.append(dict(variant=f"only {grp}", hold_mae_margin=h_[0], hold_mae_total=h_[1]))
    abl = pd.DataFrame(abl)
    abl["hold_d_margin_vs_all"] = abl.hold_mae_margin - base_h[0]
    abl["hold_d_total_vs_all"] = abl.hold_mae_total - base_h[1]
    print(abl.round(3).to_string())

    out = df[df.season >= START].copy()
    out["date"] = pd.to_datetime(out.date)
    out.to_parquet(OUT / "ncaaf_games.parquet", index=False)
    report(out, abl, best, grid, (lam, delta), thm, tht, pbp, g)
    print(f"done {time.time()-t0:.0f}s")


def report(df, abl, best, grid, rls, thm, tht, pbp, g):
    d = df.dropna(subset=["err_margin"]).copy()
    L = []
    L.append("# NCAAF script-then-learn: learning report\n")
    L.append(f"Games scripted 2016-2026: {len(d):,} (warm-up from {WARM}). Tuning window 2016-2020, holdout 2021-2026 "
             f"(2026 = live season through {d.date.max().date()}, PickCenter lines). Play-by-play from cfbfastR "
             f"(raw.githubusercontent cfbfastR-data <=2021, sportsdataverse-data releases 2022+): "
             f"{g.game_id.isin(pbp.game_id).mean():.0%} of games covered.\n")
    L.append(f"Core Kalman params chosen on 2016-2020: {best[1]} ; RLS forgetting/delta: {rls}.\n")
    L.append("## Per-season learning curve (MAE, points)\n")
    rows = []
    for s, x in d.groupby("season"):
        m = x.err_margin_mkt.notna()
        ml = x.p_home_mkt.notna() & (x.home_score != x.away_score)
        y = (x.home_score > x.away_score).astype(float)
        rows.append(dict(season=s, games=len(x), w_mkt=round(x.w_mkt.mean(), 2),
                         margin_core=mae(x.err_margin_core[m]), margin_script=mae(x.err_margin[m]),
                         margin_blend=mae(x.err_margin_b[m]), margin_market=mae(x.err_margin_mkt[m]),
                         total_core=mae(x.err_total_core[m]), total_script=mae(x.err_total[m]),
                         total_blend=mae(x.err_total_b[m]), total_market=mae(x.err_total_mkt[m]),
                         ll_script=ll(x.p_home_win[ml], y[ml]), ll_blend=ll(x.p_home_win_b[ml], y[ml]),
                         ll_market=ll(x.p_home_mkt[ml], y[ml]) if ml.sum() else np.nan))
    cur = pd.DataFrame(rows)
    L.append("(games with a closing line; ll = home-win log loss vs de-vigged closing moneyline)\n")
    L.append(cur.round(3).to_markdown(index=False))
    for nm, m in (("2016-2020", d.season <= TUNE_END), ("2021-2026 holdout", d.season >= HOLD)):
        x = d[m & d.err_margin_mkt.notna()]
        L.append(f"\n**{nm}** margin MAE core {mae(x.err_margin_core):.3f} / script {mae(x.err_margin):.3f} / "
                 f"blend {mae(x.err_margin_b):.3f} / market {mae(x.err_margin_mkt):.3f};  total MAE core "
                 f"{mae(x.err_total_core):.3f} / script {mae(x.err_total):.3f} / blend {mae(x.err_total_b):.3f} / "
                 f"market {mae(x.err_total_mkt):.3f}")
    L.append("\n## Ablation (RLS correction re-run without each enrichment group)\n")
    L.append("Positive `hold_d_*` = dropping the group made the 2021+ holdout WORSE, i.e. the group helps.\n")
    L.append(abl.round(3).to_markdown(index=False))
    L.append("\nFinal RLS coefficients (standardised features; margin / total):\n")
    names = ["intercept"] + FEATS
    L.append(pd.DataFrame(dict(feature=names, margin=thm, total=tht)).round(3).to_markdown(index=False))

    # ---- error attribution
    L.append("\n## What the engine got wrong most\n")
    h = d[(d.season >= HOLD) & d.err_margin_mkt.notna()].copy()
    pb = pbp.pivot_table(index="game_id", columns="home_off", values="plays")
    h["plays_act"] = h.game_id.map(pb.sum(axis=1))
    k = h.dropna(subset=["plays_act", "plays_script"])
    k = k[(k.plays_act > 80) & (k.home_score + k.away_score > 0)]
    act_t = k.home_score + k.away_score
    lp = np.log(k.plays_act / k.plays_script)
    le = np.log(act_t / k.plays_act) - np.log(k.total.clip(lower=10) / k.plays_script)
    tot = np.log(act_t.clip(lower=3) / k.total.clip(lower=10))
    L.append(f"Total-points error split (log scale, 2021+, {len(k):,} games with pbp): var(pace error) "
             f"{lp.var():.4f}, var(points-per-play error) {le.var():.4f}, 2cov {2*np.cov(lp, le)[0,1]:.4f}; "
             f"total var {tot.var():.4f}. -> {'efficiency (points per play)' if le.var() > lp.var() else 'pace'} "
             f"dominates the total error; the scripted play count misses by MAE {mae(k.plays_act-k.plays_script):.1f} plays.\n")
    def seg(name, m):
        x = h[m]
        return dict(segment=name, games=len(x), script_mae=mae(x.err_margin), market_mae=mae(x.err_margin_mkt),
                    gap=mae(x.err_margin) - mae(x.err_margin_mkt), script_total=mae(x.err_total),
                    market_total=mae(x.err_total_mkt))
    p5 = h.p5_vs_g5 != 0
    segs = [seg("all", h.season > 0), seg("conference game", h.home_conference == h.away_conference),
            seg("non-conference", h.home_conference != h.away_conference), seg("P5 vs G5", p5),
            seg("FBS vs FCS", h.fbs_vs_fcs != 0), seg("neutral site", h.neutral == 1),
            seg("postseason/bowl", h.postseason == 1), seg("weeks 1-2", h.week <= 2),
            seg("weeks 3-8", (h.week > 2) & (h.week <= 8)), seg("weeks 9+", h.week > 8),
            seg("|spread| >= 21", h.spread_home.abs() >= 21), seg("|spread| < 7", h.spread_home.abs() < 7),
            seg("2026 live", h.season == 2026)]
    L.append("Margin MAE by matchup type, 2021+ (gap = script minus market; positive = market better):\n")
    L.append(pd.DataFrame(segs).round(3).to_markdown(index=False))

    # ---- beat the close?
    L.append("\n## Does anything beat the closing line out of sample?\n")
    tune = (d.season >= START) & (d.season <= TUNE_END); hold = d.season >= HOLD
    d["mkt_margin"] = -d.spread_home; d["act_margin"] = d.home_score - d.away_score
    d["act_total"] = d.home_score + d.away_score
    res = []
    for col in ("margin", "margin_b"):
        r = ats_test(d, col, "mkt_margin", "act_margin", tune, hold, f"ATS via {col}")
        res.append(r)
    for col in ("total", "total_b"):
        res.append(ats_test(d, col, "total_line", "act_total", tune, hold, f"O/U via {col}"))
    res = pd.concat(res)
    pick = []
    for kind, x in res.groupby("kind"):
        t = x[(x.split == "tune") & (x.bets >= 100)]
        if len(t) == 0:
            continue
        tau = t.sort_values("roi").iloc[-1].tau
        tr = t[t.tau == tau].iloc[0]; hr = x[(x.split == "hold") & (x.tau == tau)].iloc[0]
        pick.append(dict(strategy=kind, tau_chosen_2016_20=tau, tune_bets=tr.bets, tune_hit=tr.hit, tune_roi=tr.roi,
                         hold_bets=hr.bets, hold_hit=hr.hit, hold_roi=hr.roi))
    L.append("Bet the side when the script disagrees with the close by more than tau points (tau picked on "
             "2016-2020 by ROI, min 100 bets), flat -110 (break-even 52.4%):\n")
    L.append(pd.DataFrame(pick).round(3).to_markdown(index=False))
    # line-movement test: does the script anticipate open->close moves?
    o = d.dropna(subset=["spread_open_home", "spread_home"])
    o = o[o.season >= HOLD]
    if len(o) > 200:
        mv = (-o.spread_home) - (-o.spread_open_home)
        sig = o.margin - (-o.spread_open_home)
        L.append(f"\nLine-move test 2021+ ({len(o):,} games with opening spreads): corr(script - open, close - open) = "
                 f"{np.corrcoef(sig, mv)[0,1]:.3f}; script MAE vs result {mae(o.err_margin):.2f}, open "
                 f"{mae(-o.spread_open_home - (o.home_score-o.away_score)):.2f}, close {mae(o.err_margin_mkt):.2f}.")
    hh = d[hold & d.err_margin_mkt.notna()]
    verdict = ("No. The closing spread and total are more accurate than the script on 2021+; the learned blend weight "
               f"sits at {hh.w_mkt.mean():.2f} on the market." if mae(hh.err_margin) > mae(hh.err_margin_mkt) else
               "The script matched or beat the close on margin MAE in the holdout.")
    L.append("\n**Verdict:** " + verdict + " Any ATS/O-U edge above must clear 52.4% on the holdout to matter; see "
             "the table (holdout numbers are the honest ones).")
    (OUT / "ncaaf_learning.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
