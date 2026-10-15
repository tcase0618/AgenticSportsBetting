"""
Calibration auditing + online recalibration of model probabilities.

  reliability(p, y)  -> binned table (mean predicted vs realised frequency, counts)
  brier, ece, log_loss
  OnlineRecalibrator: refit Platt (logistic on logit p) or isotonic on a trailing window of
  PAST settled bets only (`fit(history, asof)` uses rows with settled date < asof), then maps
  raw model_prob -> calibrated prob for EV and staking.

Note on selection bias: bets are only placed when the model thinks EV is high, so the model's
probabilities on *placed* bets are biased upward even if the model is well calibrated on all
quotes (winner's curse).  Recalibrating on placed bets corrects exactly that bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, bins=10, strategy="quantile") -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame(dict(b=idx, p=p, y=y)).groupby("b").agg(p_mean=("p", "mean"), y_rate=("y", "mean"),
                                                                n=("y", "size"))
    df["se"] = np.sqrt(df.y_rate * (1 - df.y_rate) / df.n)
    df["gap"] = df.y_rate - df.p_mean
    return df.reset_index(drop=True)


def ece(p, y, bins=10) -> float:
    r = reliability(p, y, bins)
    return float((r.n * r.gap.abs()).sum() / r.n.sum())


def calibration_summary(p, y, bins=10) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    # calibration-in-the-large and slope (logistic regression of y on logit p)
    slope = intercept = np.nan
    if len(np.unique(y)) == 2 and len(y) > 20:
        lr = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        slope, intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return dict(n=len(p), mean_p=float(p.mean()), win_rate=float(y.mean()), brier=brier(p, y),
                brier_const=brier(np.full_like(p, y.mean()), y), log_loss=log_loss(p, y), ece=ece(p, y, bins),
                cal_slope=slope, cal_intercept=intercept)


class OnlineRecalibrator:
    """Trailing-window recalibrator.  method: 'platt' (2-param logistic on logit p, regularised toward
    identity) or 'isotonic'.  With fewer than min_n past bets it returns the raw probability."""

    def __init__(self, method="platt", window=1000, min_n=200, shrink=50.0):
        self.method, self.window, self.min_n, self.shrink = method, window, min_n, shrink
        self.model = None
        self.n_fit = 0

    def fit(self, p_hist, y_hist):
        p, y = np.asarray(p_hist, float)[-self.window:], np.asarray(y_hist, float)[-self.window:]
        self.n_fit = len(p)
        if len(p) < self.min_n or len(np.unique(y)) < 2:
            self.model = None
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        else:
            # Platt with an identity offset: y ~ sigmoid(a + b*logit p); shrink (a,b) toward (0,1) by
            # fitting the residual model with L2 strength `shrink`/n (offset trick via augmented feature).
            x = _logit(p)
            lr = LogisticRegression(C=len(p) / self.shrink).fit(x.reshape(-1, 1), y)
            # blend toward identity by effective sample size
            w = len(p) / (len(p) + self.shrink)
            a = w * float(lr.intercept_[0])
            b = w * float(lr.coef_[0, 0]) + (1 - w) * 1.0
            self.model = (a, b)
        return self

    def fit_asof(self, history: pd.DataFrame, asof, p_col="model_prob", y_col="y", date_col="date"):
        """Fit using ONLY rows settled strictly before `asof`."""
        h = history[pd.to_datetime(history[date_col]) < pd.Timestamp(asof)].sort_values(date_col)
        return self.fit(h[p_col].values, h[y_col].values)

    def transform(self, p):
        p = np.asarray(p, float)
        if self.model is None:
            return p
        if self.method == "isotonic":
            return self.model.predict(p)
        a, b = self.model
        return 1 / (1 + np.exp(-(a + b * _logit(p))))
