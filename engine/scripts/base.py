"""
Script-then-learn core shared by every sport.

For each game, in date order:
  1. SCRIPT (before the game): from each team's current ratings, write the expected game:
       mu_home, mu_away   expected points/goals/runs for each side
       sd_home, sd_away   uncertainty of each
       rho                correlation of the two scores (pace / game-environment effect)
     plus the derived margin & total, and (optionally) a market-informed version that blends the script
     with the de-vigged closing line using a weight the engine LEARNS from its own past errors.
  2. REVEAL: compare with the final score.
  3. LEARN: Kalman-filter update of the four ratings involved (home off/def, away off/def), league scoring
     level, home advantage, the noise levels, the score correlation and the market-blend weight. Each
     update is proportional to how wrong the script was ("innovation") and how uncertain the ratings were.

Model (per game, home team h vs away team a):
  E[pts_home] = m + hfa/2 + off_h + dfn_a          dfn = points ALLOWED above average (positive = bad D)
  E[pts_away] = m - hfa/2 + off_a + dfn_h
State per team: [off, dfn] with covariance; ratings drift between games (process noise q_game) and regress
toward 0 between seasons (keep `carry` of the rating, inflate variance).

Sports with low counts (NHL goals, MLB runs) use the same linear model on the mean; simulation then draws
counts from a gamma-Poisson (negative binomial) with the script's means and a shared game-level factor that
produces the learned correlation.

Every sport adapter calls `ScriptEngine.run(games_df)` and gets back one row per game with the pre-game
script, the outcome, the errors, and the running state needed for props/SGP layers. See SPEC.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    init_level: float              # league average points per team per game
    init_hfa: float                # home advantage (points)
    init_sd: float                 # per-team score noise sd
    prior_var: float               # variance of a new team's off/dfn rating
    q_game: float                  # rating drift variance added before each game
    carry: float = 0.7             # fraction of rating carried into a new season
    season_var: float = 0.0        # extra variance added at new season
    lr_level: float = 0.002        # learning rate for league level / hfa
    count_model: bool = False      # True for NHL / MLB style scores
    min_sd: float = 0.5


@dataclass
class TeamState:
    x: np.ndarray = field(default_factory=lambda: np.zeros(2))     # [off, dfn]
    P: np.ndarray = None
    season: int | None = None
    games: int = 0


class ScriptEngine:
    def __init__(self, p: Params):
        self.p = p
        self.m = p.init_level
        self.hfa = p.init_hfa
        self.sd = p.init_sd                         # learned observation noise
        self.rho = 0.0                              # learned correlation of home/away residuals
        self.teams: dict[str, TeamState] = {}
        self.w_mkt = 0.5                            # learned weight on market vs own script
        self._err_own, self._err_mkt, self._err_blend = [], [], []
        self._res = []                              # recent residual pairs for sd / rho

    # ---------------------------------------------------------------- state
    def _team(self, t: str, season: int) -> TeamState:
        s = self.teams.get(t)
        if s is None:
            s = TeamState(P=np.eye(2) * self.p.prior_var, season=season)
            self.teams[t] = s
        elif s.season != season:                    # new season: regress & widen
            s.x = s.x * self.p.carry
            s.P = s.P * self.p.carry ** 2 + np.eye(2) * (self.p.season_var + self.p.prior_var * (1 - self.p.carry ** 2))
            s.season = season
        return s

    # ---------------------------------------------------------------- script
    def script(self, h: str, a: str, season: int, neutral: bool = False) -> dict:
        H, A = self._team(h, season), self._team(a, season)
        H.P = H.P + np.eye(2) * self.p.q_game
        A.P = A.P + np.eye(2) * self.p.q_game
        adv = 0.0 if neutral else self.hfa / 2
        mu_h = self.m + adv + H.x[0] + A.x[1]
        mu_a = self.m - adv + A.x[0] + H.x[1]
        # predictive variance = rating uncertainty + observation noise
        var_h = H.P[0, 0] + A.P[1, 1] + self.sd ** 2
        var_a = A.P[0, 0] + H.P[1, 1] + self.sd ** 2
        cov = H.P[0, 1] + A.P[0, 1] + self.rho * self.sd ** 2
        return dict(mu_home=mu_h, mu_away=mu_a, sd_home=math.sqrt(var_h), sd_away=math.sqrt(var_a),
                    rho=cov / math.sqrt(var_h * var_a), margin=mu_h - mu_a, total=mu_h + mu_a,
                    sd_margin=math.sqrt(var_h + var_a - 2 * cov), sd_total=math.sqrt(var_h + var_a + 2 * cov),
                    rating_unc=float(np.trace(H.P) + np.trace(A.P)))

    # ---------------------------------------------------------------- learn
    def learn(self, h: str, a: str, season: int, y_h: float, y_a: float, neutral: bool = False):
        """Kalman update of [off_h, dfn_h, off_a, dfn_a] from the two observed scores."""
        H, A = self.teams[h], self.teams[a]
        x = np.r_[H.x, A.x]
        P = np.zeros((4, 4)); P[:2, :2] = H.P; P[2:, 2:] = A.P
        # y_h = m + adv + off_h + dfn_a ; y_a = m - adv + off_a + dfn_h
        Hm = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], float)
        adv = 0.0 if neutral else self.hfa / 2
        pred = np.array([self.m + adv, self.m - adv]) + Hm @ x
        R = np.array([[1, self.rho], [self.rho, 1]]) * self.sd ** 2
        S = Hm @ P @ Hm.T + R
        K = P @ Hm.T @ np.linalg.inv(S)
        innov = np.array([y_h, y_a]) - pred
        x = x + K @ innov
        P = (np.eye(4) - K @ Hm) @ P
        H.x, A.x = x[:2], x[2:]
        H.P, A.P = P[:2, :2], P[2:, 2:]
        H.games += 1; A.games += 1
        # league level and home advantage learn from the mean innovation
        self.m += self.p.lr_level * innov.mean()
        if not neutral:
            self.hfa += self.p.lr_level * (innov[0] - innov[1])
        # noise level and score correlation learn from recent residuals
        self._res.append(innov)
        if len(self._res) > 3000:
            self._res = self._res[-3000:]
        if len(self._res) >= 200:
            r = np.array(self._res[-2000:])
            exc = max(np.mean(r ** 2) - np.mean(np.diag(Hm @ P @ Hm.T)) / 2, self.p.min_sd ** 2)
            self.sd = 0.98 * self.sd + 0.02 * math.sqrt(exc)
            c = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
            if np.isfinite(c):
                self.rho = 0.98 * self.rho + 0.02 * float(np.clip(c, -0.5, 0.8))
        return innov

    # ---------------------------------------------------------------- market blend (learned)
    def blend(self, sc: dict, mkt_margin: float | None, mkt_total: float | None) -> dict:
        """Market-informed script: margin/total = w*market + (1-w)*own, w learned from past squared errors."""
        if mkt_margin is None or not np.isfinite(mkt_margin) or mkt_total is None or not np.isfinite(mkt_total):
            return dict(margin_b=sc["margin"], total_b=sc["total"], w_mkt=np.nan)
        w = self.w_mkt
        return dict(margin_b=w * mkt_margin + (1 - w) * sc["margin"], total_b=w * mkt_total + (1 - w) * sc["total"], w_mkt=w)

    def learn_blend(self, sc: dict, mkt_margin, mkt_total, act_margin, act_total, window=1500):
        if mkt_margin is None or not np.isfinite(mkt_margin) or not np.isfinite(mkt_total):
            return
        self._err_own.append((sc["margin"] - act_margin, sc["total"] - act_total))
        self._err_mkt.append((mkt_margin - act_margin, mkt_total - act_total))
        if len(self._err_own) > window:
            self._err_own, self._err_mkt = self._err_own[-window:], self._err_mkt[-window:]
        if len(self._err_own) >= 100:
            eo, em = np.array(self._err_own), np.array(self._err_mkt)
            # optimal combination weight for two forecasts (Bates-Granger) on recent errors
            num = np.mean((eo - em) * eo)
            den = np.mean((eo - em) ** 2)
            if den > 1e-9:
                self.w_mkt = float(np.clip(num / den, 0.0, 1.0))

    # ---------------------------------------------------------------- run
    def run(self, g: pd.DataFrame) -> pd.DataFrame:
        """g: date, season, home, away, home_score, away_score, [neutral, spread_home, total, home_ml, away_ml].
        Returns one row per game: the pre-game script, the outcome and the errors."""
        g = g.sort_values("date").reset_index(drop=True)
        out = []
        for r in g.itertuples(index=False):
            neutral = bool(getattr(r, "neutral", 0) or 0)
            sc = self.script(r.home, r.away, int(r.season), neutral)
            sp = getattr(r, "spread_home", np.nan)
            tot = getattr(r, "total", np.nan)
            mkt_margin = -sp if sp == sp and sp is not None else np.nan
            b = self.blend(sc, mkt_margin, tot)
            row = dict(date=r.date, season=r.season, home=r.home, away=r.away, neutral=int(neutral), **sc, **b,
                       spread_home=sp, total_line=tot, home_score=r.home_score, away_score=r.away_score,
                       hfa=self.hfa, level=self.m, noise_sd=self.sd, score_rho=self.rho,
                       games_home=self.teams[r.home].games, games_away=self.teams[r.away].games)
            if r.home_score == r.home_score and r.away_score == r.away_score:
                act_m, act_t = r.home_score - r.away_score, r.home_score + r.away_score
                row.update(err_margin=sc["margin"] - act_m, err_total=sc["total"] - act_t,
                           err_margin_b=b["margin_b"] - act_m, err_total_b=b["total_b"] - act_t,
                           err_margin_mkt=(mkt_margin - act_m) if mkt_margin == mkt_margin else np.nan,
                           err_total_mkt=(tot - act_t) if tot == tot else np.nan)
                self.learn_blend(sc, mkt_margin, tot, act_m, act_t)
                self.learn(r.home, r.away, int(r.season), float(r.home_score), float(r.away_score), neutral)
            out.append(row)
        return pd.DataFrame(out)


# ---------------------------------------------------------------- simulation
def simulate_scores(row: dict | pd.Series, n: int = 20000, count_model: bool = False, rng=None) -> np.ndarray:
    """Draw (home, away) scores from a script row. Normal for high-scoring sports; gamma-Poisson with a shared
    game factor for low-count sports (NHL/MLB) so the learned correlation carries into the simulation."""
    rng = rng or np.random.default_rng(0)
    mh, ma = max(row["mu_home"], 0.1), max(row["mu_away"], 0.1)
    if not count_model:
        sh, sa, r = row["sd_home"], row["sd_away"], float(np.clip(row["rho"], -0.9, 0.9))
        cov = [[sh ** 2, r * sh * sa], [r * sh * sa, sa ** 2]]
        return rng.multivariate_normal([mh, ma], cov, size=n)
    r = float(np.clip(row["rho"], 0.0, 0.6))
    shape = 1.0 / max(r, 1e-3) * 0.5                      # stronger shared factor -> more correlation
    game = rng.gamma(shape, 1.0 / shape, size=n)          # mean 1
    return np.c_[rng.poisson(mh * game), rng.poisson(ma * game)]


def learning_report(df: pd.DataFrame) -> pd.DataFrame:
    """How the scripts improved: mean absolute error per season, own script vs market vs learned blend."""
    d = df.dropna(subset=["err_margin"])
    rep = d.groupby("season").agg(
        games=("err_margin", "size"),
        mae_margin_script=("err_margin", lambda x: x.abs().mean()),
        mae_margin_blend=("err_margin_b", lambda x: x.abs().mean()),
        mae_margin_market=("err_margin_mkt", lambda x: x.abs().mean()),
        mae_total_script=("err_total", lambda x: x.abs().mean()),
        mae_total_blend=("err_total_b", lambda x: x.abs().mean()),
        mae_total_market=("err_total_mkt", lambda x: x.abs().mean()),
        w_mkt_end=("w_mkt", "last"))
    return rep.round(3)
