#!/usr/bin/env python3
"""Build normalized NCAAF (FBS) game + betting-line dataset.

Sources (downloaded to data/raw/ncaaf/):
  cfbfastR/betting/cfb_line_odds.csv.gz
      https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz
      Multi-book lines per game (ESPN odds feed 2006-2019, CFBD lines 2020+).
  cfbfastR/schedules/cfb_schedules_{YYYY}.csv
      https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{YYYY}.csv
      CFBD schedules/results (game_id == ESPN id == betting game_id).
  jackschooley/ncaa football {2015..2019}.xlsx  (sportsbookreviewsonline mirror; validation only)
      https://raw.githubusercontent.com/jackschooley/cfb-betting/master/data/ncaa%20football%20{YYYY}.xlsx

Consensus line = median across books of each book's last (final) line.
spread_home: negative => home favored.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "raw", "ncaaf")
OUT = os.path.join(ROOT, "data", "processed", "ncaaf.csv")
FIRST, LAST = 2006, 2025


def load_schedules():
    fs = sorted(glob.glob(os.path.join(RAW, "cfbfastR", "schedules", "cfb_schedules_*.csv")))
    s = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)
    s = s[(s.season >= FIRST) & (s.season <= LAST)]
    s = s[(s.home_division == "fbs") | (s.away_division == "fbs")]
    s = s[s.home_points.notna() & s.away_points.notna()]
    s = s.drop_duplicates("game_id")
    return s


def load_lines():
    b = pd.read_csv(os.path.join(RAW, "cfbfastR", "betting", "cfb_line_odds.csv.gz"), low_memory=False)
    b = b[b.game_id.notna()].copy()
    b["game_id"] = b.game_id.astype("int64")
    b["book"] = b.book.replace({"Draft Kings": "DraftKings"})
    # source contains exact duplicate rows (esp. 2015); drop them
    b = b.drop_duplicates(["game_id", "id", "book", "market_type", "abbr", "lines", "odds", "opening_lines", "opening_odds"])
    return b


def map_abbr(b, sched):
    """Map each abbr string to a team id by intersecting team sets of the games it appears in."""
    ids = sched.set_index("game_id")[["home_id", "away_id"]]
    t = b[b.market_type.isin(["spread", "money_line"])][["game_id", "abbr"]].drop_duplicates()
    t = t.join(ids, on="game_id", how="inner")
    long = pd.concat([t[["abbr", "game_id", "home_id"]].rename(columns={"home_id": "tid"}),
                      t[["abbr", "game_id", "away_id"]].rename(columns={"away_id": "tid"})])
    cnt = long.groupby(["abbr", "tid"]).size().rename("n").reset_index()
    tot = t.groupby("abbr").size().rename("tot")
    cnt = cnt.join(tot, on="abbr").sort_values(["abbr", "n"], ascending=[True, False])
    best = cnt.groupby("abbr").head(2)
    out = {}
    for a, g in best.groupby("abbr"):
        n1 = g.n.iloc[0]
        n2 = g.n.iloc[1] if len(g) > 1 else 0
        # require the top id to appear in >=90% of games and clearly beat runner-up
        if n1 >= 0.9 * g.tot.iloc[0] and (n1 > n2 or g.tot.iloc[0] == 1 and False):
            out[a] = int(g.tid.iloc[0])
    return out


def am_to_dec(x):
    return np.where(x > 0, 1 + x / 100.0, 1 + 100.0 / np.abs(x))


def dec_to_am(d):
    return np.where(d >= 2, (d - 1) * 100.0, -100.0 / (d - 1))


def build():
    sched = load_schedules()
    b = load_lines()
    b = b[b.game_id.isin(sched.game_id)]
    # Older feed sometimes attaches a rematch's (e.g. bowl) lines to the regular-season game_id:
    # keep only rows whose line date is within 1 day of the scheduled game date.
    sdate = pd.to_datetime(sched.set_index("game_id").start_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    ds = b.date_time.astype("string")
    isz = ds.str.endswith("Z").fillna(False)
    bdt = pd.Series(pd.NaT, index=b.index, dtype="datetime64[ns]")
    bdt[isz] = pd.to_datetime(ds[isz], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.tz_localize(None)
    bdt[~isz] = pd.to_datetime(ds[~isz], errors="coerce")  # naive timestamps (2006-2019 feed), ~ET
    bdt = bdt.dt.normalize()
    dd = (bdt - b.game_id.map(sdate)).dt.days.abs()
    print("line rows dropped for date mismatch (>1 day):", int((dd > 1).sum()), file=sys.stderr)
    b = b[~(dd > 1)]
    amap = map_abbr(b, sched)
    ids = sched.set_index("game_id")[["home_id", "away_id"]]
    b = b.join(ids, on="game_id")
    b["tid"] = b.abbr.map(amap)
    b["side"] = np.where(b.tid == b.home_id, "home", np.where(b.tid == b.away_id, "away", None))
    # if one side of a (game, book, market) pair is resolved, infer the other
    sm = b[b.market_type.isin(["spread", "money_line"])].copy()
    sm["k"] = sm.groupby(["game_id", "book", "market_type"]).cumcount()
    grp = sm.groupby(["game_id", "book", "market_type"])
    sm["nrows"] = grp.abbr.transform("size")
    sm = sm[sm.nrows == 2]
    known = sm.groupby(["game_id", "book", "market_type"]).side.transform(lambda s: s.dropna().iloc[0] if s.notna().sum() == 1 else None)
    other = {"home": "away", "away": "home"}
    fill = sm.side.isna() & known.notna()
    sm.loc[fill, "side"] = known[fill].map(other)
    sm = sm[sm.side.notna()]
    # drop pairs where both rows got same side
    ok = sm.groupby(["game_id", "book", "market_type"]).side.transform("nunique") == 2
    sm = sm[ok]

    # ---- spreads: home line per book
    sp = sm[(sm.market_type == "spread") & (sm.side == "home")][["game_id", "book", "lines", "opening_lines"]]
    spa = sm[(sm.market_type == "spread") & (sm.side == "away")][["game_id", "book", "lines"]]
    sp = sp.merge(spa, on=["game_id", "book"], suffixes=("", "_away"))
    # internal consistency: home line == -away line
    sp = sp[np.isclose(sp["lines"], -sp["lines_away"]) | sp["lines"].isna()]
    # ---- moneylines per book
    ml = sm[sm.market_type == "money_line"].pivot_table(index=["game_id", "book"], columns="side", values="odds", aggfunc="first").reset_index()
    ml = ml.dropna(subset=["home", "away"])
    ml = ml[(ml.home.abs() >= 100) & (ml.away.abs() >= 100)]
    ml["hold"] = 1 / am_to_dec(ml.home.values) + 1 / am_to_dec(ml.away.values)
    ml = ml[(ml.hold > 1.0) & (ml.hold < 1.25)]
    # ---- sign-consistency check per book: spread fav must equal ML fav
    chk = sp.merge(ml, on=["game_id", "book"])
    bad = chk[((chk["lines"] < -0.5) & (chk.home > chk.away) & (chk.away < -105)) |
              ((chk["lines"] > 0.5) & (chk.home < chk.away) & (chk.home < -105))]
    print(f"book-level spread/ML favorite disagreements dropped: {len(bad)} of {len(chk)}", file=sys.stderr)
    badk = set(zip(bad.game_id, bad.book))
    sp = sp[[k not in badk for k in zip(sp.game_id, sp.book)]]
    ml = ml[[k not in badk for k in zip(ml.game_id, ml.book)]]

    # ---- totals
    tt = b[b.market_type == "total"].copy()
    tt = tt[(tt["lines"] > 20) & (tt["lines"] < 120)]
    tt = tt.groupby(["game_id", "book"]).agg(total=("lines", "first"), total_open=("opening_lines", "first")).reset_index()

    # spread sanity
    sp = sp[sp["lines"].abs() <= 70]
    agg_sp = sp.groupby("game_id").agg(spread_home=("lines", "median"), spread_open_home=("opening_lines", "median"),
                                       spread_nbooks=("lines", "count"), spread_std=("lines", "std"))
    agg_tt = tt.groupby("game_id").agg(total=("total", "median"), total_open=("total_open", "median"), total_nbooks=("total", "count"))
    ml["hp"] = 1 / am_to_dec(ml.home.values)
    ml["ap"] = 1 / am_to_dec(ml.away.values)
    agg_ml = ml.groupby("game_id").agg(hp=("hp", "median"), ap=("ap", "median"), ml_nbooks=("hp", "count"))
    agg_ml["home_ml"] = np.round(dec_to_am(1 / agg_ml.hp.values)).astype(int)
    agg_ml["away_ml"] = np.round(dec_to_am(1 / agg_ml.ap.values)).astype(int)
    # spread disagreement > 7 points across books -> outlier book likely; use median anyway but flag
    # pinnacle (2006-2019) separately as a sharp reference
    pin = sp[sp.book == "PINNACLE"].set_index("game_id")["lines"].rename("spread_home_pinnacle")

    g = sched.copy()
    g["date"] = pd.to_datetime(g.start_date, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "game_id": g.game_id.astype("int64"),
        "date": g.date,
        "season": g.season.astype(int),
        "week": g.week.astype("Int64"),
        "season_type": g.season_type,
        "postseason": (g.season_type == "postseason").astype(int),
        "home": g.home_team,
        "away": g.away_team,
        "home_score": g.home_points.astype(int),
        "away_score": g.away_points.astype(int),
        "neutral": g.neutral_site.map({True: 1, False: 0, "True": 1, "False": 0}).astype("Int64"),
        "conference_game": g.conference_game.map({True: 1, False: 0, "True": 1, "False": 0}).astype("Int64"),
        "home_conference": g.home_conference,
        "away_conference": g.away_conference,
        "home_division": g.home_division,
        "away_division": g.away_division,
        "notes": g.notes,
    })
    out = out.join(agg_sp, on="game_id").join(agg_tt, on="game_id").join(agg_ml[["home_ml", "away_ml", "ml_nbooks"]], on="game_id").join(pin, on="game_id")
    cols = ["date", "season", "home", "away", "home_score", "away_score", "neutral", "spread_home", "total",
            "home_ml", "away_ml", "spread_open_home", "total_open", "game_id", "week", "season_type", "postseason",
            "conference_game", "home_conference", "away_conference", "home_division", "away_division",
            "spread_nbooks", "spread_std", "total_nbooks", "ml_nbooks", "spread_home_pinnacle", "notes"]
    # schedule occasionally lists the same game under two ids; keep the one with lines
    out = out.sort_values(["spread_nbooks"], ascending=False, na_position="last")
    out = out.drop_duplicates(["date", "home", "away"])
    out = out[cols].sort_values(["date", "game_id"]).reset_index(drop=True)
    for c in ["home_ml", "away_ml", "spread_nbooks", "total_nbooks", "ml_nbooks"]:
        out[c] = out[c].astype("Int64")
    out["spread_std"] = out.spread_std.round(2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


def validate(df):
    print("rows", len(df), "dup game_id", df.game_id.duplicated().sum(),
          "dup (date,home,away)", df.duplicated(["date", "home", "away"]).sum())
    df = df.copy()
    df["margin"] = df.home_score - df.away_score
    s = df.dropna(subset=["spread_home"])
    print("corr(-spread_home, margin) =", round(np.corrcoef(-s.spread_home, s.margin)[0, 1], 3))
    print("ATS home cover rate (excl push):", round(((s.margin + s.spread_home) > 0).sum() / ((s.margin + s.spread_home) != 0).sum(), 3))
    fav = s[s.spread_home != 0]
    print("favorite SU win rate:", round((np.sign(-fav.spread_home) == np.sign(fav.margin)).mean(), 3))
    t = df.dropna(subset=["total"])
    print("corr(total, pts) =", round(np.corrcoef(t.total, t.home_score + t.away_score)[0, 1], 3))
    m = df.dropna(subset=["home_ml", "away_ml"]).copy()
    m["vig"] = 1 / am_to_dec(m.home_ml.astype(float).values) + 1 / am_to_dec(m.away_ml.astype(float).values)
    print("ML implied sum quantiles:", m.vig.quantile([.01, .5, .99]).round(3).tolist())
    mm = m.dropna(subset=["spread_home"])
    print("ML fav vs spread fav agree:", round((np.sign(mm.home_ml.astype(float) - mm.away_ml.astype(float)) == np.sign(mm.spread_home)).mean(), 3))
    per = df.groupby("season").agg(games=("game_id", "size"), spread=("spread_home", lambda x: x.notna().mean()),
                                   total=("total", lambda x: x.notna().mean()), ml=("home_ml", lambda x: x.notna().mean()),
                                   sp_open=("spread_open_home", lambda x: x.notna().mean()))
    fbs = df[(df.home_division == "fbs") & (df.away_division == "fbs")]
    per["fbs_vs_fbs"] = fbs.groupby("season").size()
    per["fbs_spread"] = fbs.groupby("season").spread_home.apply(lambda x: x.notna().mean())
    print(per.round(3).to_string())
    print("score range", df.home_score.min(), df.home_score.max(), df.away_score.min(), df.away_score.max())


def sbr_check(df):
    """Cross-check consensus spread vs sportsbookreviewsonline closing lines (2015-2019)."""
    res = []
    for y in range(2015, 2020):
        f = os.path.join(RAW, "jackschooley", f"ncaa football {y}.xlsx")
        if not os.path.exists(f):
            continue
        x = pd.read_excel(f)
        x = x[x.VH.isin(["V", "H", "N"])].reset_index(drop=True)
        for i in range(0, len(x) - 1, 2):
            v, h = x.iloc[i], x.iloc[i + 1]
            def num(z):
                try:
                    return float(str(z).replace("pk", "0").replace("PK", "0"))
                except ValueError:
                    return np.nan
            cv, ch = num(v.Close), num(h.Close)
            if np.isnan(cv) or np.isnan(ch):
                continue
            # smaller number is spread (favorite's line), larger is total
            if cv < ch:
                sp_home = cv  # visitor favored by cv -> home +cv
            else:
                sp_home = -ch
            res.append((y, int(v.Final), int(h.Final), sp_home))
    sbr = pd.DataFrame(res, columns=["season", "away_score", "home_score", "sbr_spread_home"])
    d = df[df.season.between(2015, 2019)].dropna(subset=["spread_home"])
    j = d.merge(sbr, on=["season", "away_score", "home_score"])
    # keep only unique score matches
    j = j[~j.duplicated(["game_id"], keep=False)]
    j = j[j.groupby(["season", "away_score", "home_score"]).game_id.transform("size") == 1]
    diff = (j.spread_home - j.sbr_spread_home).abs()
    flip = (j.spread_home + j.sbr_spread_home).abs()
    print(f"SBR cross-check (2015-19, unique score matches n={len(j)}): |diff|<=1: {(diff<=1).mean():.3f}, "
          f"median|diff| {diff.median():.2f}, sign-flip-looking {( (flip<diff) & (j.spread_home.abs()>2) ).mean():.3f}")


if __name__ == "__main__":
    df = build()
    validate(df)
    sbr_check(df)
