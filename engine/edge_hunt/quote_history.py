"""
Persistent per-quote price history for the arb scanner, and the stale / suspect quote flags built on it.

Store: results/edge_hunt/quote_history.sqlite (pruned to 48h)
  state        one row per (venue, espn_id, market, hkey, side): current price, `since` (first scan at which the
               current price was seen), first_seen, last_seen
  changes      change log of quote prices (only written when a price is new or changes)
  med_state /  cross-venue median implied probability per outcome (espn_id, market, hkey, side), change-logged
  med_changes  so the median at any past time can be looked up

Flags (all quotes stay in the scan's quote table, flagged; flagged quotes are excluded from arbs/+EV/alerts):
  STALE  (1) per-outcome: a non-exchange quote whose price has not changed for >= STALE_MIN minutes while the
             cross-venue median implied probability of that same outcome moved >= MED_MOVE over the same window
         (2) frozen feed: an aggregator-sourced venue (Action Network books, DraftKings via ESPN) none of whose
             main ml/spread/total quotes on an event changed for >= STALE_MIN minutes while >= 4 other sportsbooks
             (and >= 2/3 of the sportsbooks observed that long) changed a main quote on that event in that window
         (3) feed age: Action Network reports line_status == 'opener' (the book's quote has not been updated
             since the open). Action Network exposes no per-quote timestamp, so this is the only feed-age signal.
  SUSPECT  single-venue outlier: implied probability more than OUTLIER below every other venue quoting the same
           outcome/line (>= 2 others). A suspect quote only counts when it is on an exchange (Kalshi/Polymarket),
           where the order book is re-checked before anything is reported.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

STALE_MIN = 60.0
MED_MOVE = 0.015
OUTLIER = 0.04
KEEP_H = 48
EXCHANGES = {"Polymarket", "Kalshi"}
AGGREGATED = {"DraftKings", "BetMGM", "bet365", "Caesars", "Fanatics", "HardRock", "theScore", "BallyBet",
              "DraftKings(ESPN)"}


def _iso(ts):
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


class QuoteHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        c = self.db
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, venue TEXT, espn_id TEXT, market TEXT,
                     hkey REAL, side TEXT, am REAL, since TEXT, first_seen TEXT, last_seen TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS changes (ts TEXT, k TEXT, am REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS med_state (ok TEXT PRIMARY KEY, med REAL, n INTEGER, ts TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS med_changes (ts TEXT, ok TEXT, med REAL, n INTEGER)")
        c.execute("CREATE INDEX IF NOT EXISTS med_ch_ok ON med_changes(ok, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS ch_ts ON changes(ts)")
        c.commit()

    def close(self):
        self.db.close()

    # ------------------------------------------------------------------ main entry
    def update_and_flag(self, N: pd.DataFrame, now=None, stale_min=STALE_MIN, med_move=MED_MOVE,
                        outlier=OUTLIER) -> pd.DataFrame:
        """Record this scan's quotes and return N with columns stale, suspect, flag_why, unchanged_min."""
        N = N.copy()
        now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        nows = _iso(now)
        N["espn_id"] = N.espn_id.astype(str)
        hk = N.hkey.astype(float).round(2).astype(str)
        N["_k"] = N.venue + "|" + N.espn_id + "|" + N.market + "|" + hk + "|" + N.side
        N["_ok"] = N.espn_id + "|" + N.market + "|" + hk + "|" + N.side
        N["_p"] = 1.0 / N.dec
        # ---- prior state
        S = pd.read_sql("SELECT k, am, since, first_seen FROM state", self.db).set_index("k")
        prev_am = N._k.map(S.am)
        same = prev_am.notna() & ((prev_am - N.am).abs() < 1e-9)
        N["_since"] = np.where(same, N._k.map(S.since), nows)
        N["_first"] = N._k.map(S.first_seen).fillna(nows)
        since_ts = pd.to_datetime(N._since, utc=True)
        first_ts = pd.to_datetime(N._first, utc=True)
        N["unchanged_min"] = ((now - since_ts).dt.total_seconds() / 60).round(1)
        # ---- cross-venue median per outcome (now) and at each quote's `since`
        G = N.groupby("_ok")._p
        med_now, n_now = G.median(), G.size()
        N["_med_now"], N["_n"] = N._ok.map(med_now), N._ok.map(n_now)
        why = pd.Series("", index=N.index, dtype=object)
        stale = pd.Series(False, index=N.index)
        cand = N[(N.unchanged_min >= stale_min) & (~N.venue.isin(EXCHANGES)) & (N._n >= 3)]
        if len(cand):
            M = self._med_at(cand._ok.tolist(), cand._since.tolist())
            mt = pd.Series(M, index=cand.index)
            mv = (cand._med_now - mt).abs()
            hit = mv >= med_move
            for i in cand.index[hit.fillna(False).values]:
                stale[i] = True
                why[i] = (f"price unchanged {N.at[i, 'unchanged_min']:.0f}m while median moved "
                          f"{100 * (N.at[i, '_med_now'] - mt[i]):+.1f}pt")
        # ---- frozen aggregator feed (whole venue on an event)
        # compare like with like: main ml/spread/total quotes of sportsbooks (exchanges tick constantly)
        main = N[N.market.isin(["ml", "spread", "total"]) & (~N.alt.astype(bool)) & (~N.venue.isin(EXCHANGES))]
        chg = main[main._since.astype(str) > main._first.astype(str)]          # genuine price changes only
        last_move = chg.groupby(["espn_id", "venue"])._since.max()
        obs = main.groupby(["espn_id", "venue"]).agg(first=("_first", "min"), since=("_since", "max"))
        for (eid, v), r in obs.iterrows():
            if v not in AGGREGATED:
                continue
            frozen_since = pd.Timestamp(r["since"])
            if (now - frozen_since).total_seconds() / 60 < stale_min or (now - pd.Timestamp(r["first"])).total_seconds() / 60 < stale_min:
                continue
            others = obs.loc[eid]
            others = others[(others.index != v) & (pd.to_datetime(others["first"], utc=True) <= frozen_since)]
            if len(others) < 3:
                continue
            lm = last_move.get(eid)
            moved = 0 if lm is None else int(sum(1 for o in others.index if o in lm.index and lm[o] > r["since"]))
            if moved >= 4 and moved >= (2 / 3) * len(others):
                m = (N.espn_id == eid) & (N.venue == v)
                stale[m] = True
                why[m] = why[m].where(why[m] != "", f"feed frozen {(now - frozen_since).total_seconds() / 60:.0f}m "
                                                    f"while {moved}/{len(others)} venues moved")
        # ---- feed age (Action Network line_status)
        if "line_status" in N:
            op = N.line_status.astype(str).str.lower().eq("opener")
            stale |= op
            why[op & (why == "")] = "AN line_status=opener"
        # ---- single-venue outlier
        suspect = pd.Series(False, index=N.index)
        P = N.groupby(["_ok", "venue"])._p.min().reset_index()
        P["_rank_n"] = P.groupby("_ok").venue.transform("size")
        P = P[P._rank_n >= 3]
        if len(P):
            P = P.sort_values(["_ok", "_p"])
            first = P.groupby("_ok").head(1).set_index("_ok")
            second = P.groupby("_ok").nth(1).set_index("_ok")._p
            first["second"] = second
            bad = first[(first.second - first._p) > outlier]
            if len(bad):
                bk = set(zip(bad.index, bad.venue))
                m = pd.Series([(o, v) in bk for o, v in zip(N._ok, N.venue)], index=N.index)
                suspect = m
                for i in N.index[m.values]:
                    gap = 100 * (second[N.at[i, "_ok"]] - N.at[i, "_p"])
                    why[i] = (why[i] + "; " if why[i] else "") + f"outlier {gap:.1f}pt better than next venue"
        N["stale"], N["suspect"], N["flag_why"] = stale.values, suspect.values, why.values
        # ---- persist
        self._write(N, nows, same, med_now, n_now)
        return N.drop(columns=[c for c in N.columns if c.startswith("_")])

    def _med_at(self, oks, sinces):
        out = []
        cur = self.db.cursor()
        for ok, s in zip(oks, sinces):
            r = cur.execute("SELECT med FROM med_changes WHERE ok=? AND ts<=? ORDER BY ts DESC LIMIT 1", (ok, s)).fetchone()
            if r is None:
                r = cur.execute("SELECT med FROM med_changes WHERE ok=? ORDER BY ts LIMIT 1", (ok,)).fetchone()
            out.append(np.nan if r is None else r[0])
        return out

    def _write(self, N, nows, same, med_now, n_now):
        c = self.db
        rows = list(zip(N._k, N.venue, N.espn_id, N.market, N.hkey.astype(float), N.side, N.am, N._since, N._first,
                        [nows] * len(N)))
        c.executemany("INSERT OR REPLACE INTO state VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        ch = N[~same.values]
        c.executemany("INSERT INTO changes VALUES (?,?,?)", list(zip([nows] * len(ch), ch._k, ch.am)))
        MS = pd.read_sql("SELECT ok, med FROM med_state", c).set_index("ok").med
        prev = med_now.index.map(MS)
        newm = [(nows, ok, float(m), int(n_now[ok])) for ok, m, p in zip(med_now.index, med_now.values, prev)
                if p != p or abs(p - m) > 5e-4]
        c.executemany("INSERT INTO med_changes VALUES (?,?,?,?)", newm)
        c.executemany("INSERT OR REPLACE INTO med_state VALUES (?,?,?,?)",
                      [(ok, m, n, ts) for ts, ok, m, n in newm])
        cut = _iso(pd.Timestamp(nows) - pd.Timedelta(hours=KEEP_H))
        c.execute("DELETE FROM state WHERE last_seen < ?", (cut,))
        c.execute("DELETE FROM changes WHERE ts < ?", (cut,))
        c.execute("DELETE FROM med_changes WHERE ts < ?", (cut,))
        c.execute("DELETE FROM med_state WHERE ts < ?", (cut,))
        c.commit()


def backtest_snapshots(files, db_path):
    """Replay saved quote snapshots (results/edge_hunt/arb_quotes/quotes_*.csv.gz) through a fresh history DB.
    Returns the flagged quotes of every snapshot (with snapshot time)."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    H = QuoteHistory(db_path)
    out = []
    for f in sorted(files):
        ts = pd.Timestamp(Path(f).name.split("_")[1].split(".")[0], tz="UTC")
        Q = pd.read_csv(f, dtype={"espn_id": str})
        Q["hkey"] = Q.hline.fillna(-9999.0)
        F = H.update_and_flag(Q, now=ts)
        F["snapshot"] = ts
        out.append(F[F.stale | F.suspect])
    H.close()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
