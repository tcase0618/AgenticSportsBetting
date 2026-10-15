#!/usr/bin/env python3
"""Build data/processed/ufc.csv: one row per UFC fight (2010+), with odds + PRE-FIGHT features.

Sources (downloaded to data/raw/ufc/ by download()):
  greco/      Greco1899/scrape_ufc_stats  (ufcstats.com results, per-round stats, fighter tale-of-tape)
  ultimate/   shortlikeafox/ultimate_ufc_dataset ufc-master.csv (mdabbert "Ultimate UFC Dataset":
              BestFightOdds moneylines + method props, 2010-03-21 .. 2026-03-28)
  sudawg/     sudawg/UFC_Data (fork of andrew-couch/UFC_Data) moneyline.csv: BestFightOdds
              "closing range" per fighter page, 2007..2023-08 -> used only as FALLBACK odds.

Leakage policy: every a_/b_ career feature is computed from Greco per-fight data using ONLY fights
with date strictly before the fight's date (cumulative, shifted). Height/reach/stance/DOB are static
tale-of-tape attributes (current snapshot; reach/height rarely change). Ranks (a_rank/b_rank) come from
ufc-master, which uses the ranking published before the event.

Fighter A/B orientation: when ufc-master matches, A = Red corner, B = Blue corner. Otherwise A/B is
alphabetical (ufcstats lists the WINNER first, so its order must not be used).
"""
import os, re, sys, unicodedata, subprocess
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "raw", "ufc")
OUT = os.path.join(ROOT, "data", "processed", "ufc.csv")

URLS = {
    "greco": ("https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/",
              ["ufc_fight_results.csv", "ufc_fighter_tott.csv", "ufc_fight_stats.csv",
               "ufc_event_details.csv", "ufc_fighter_details.csv", "ufc_fight_details.csv"]),
    "ultimate": ("https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/",
                 ["ufc-master.csv", "upcoming.csv", "README.md"]),
    "sudawg": ("https://raw.githubusercontent.com/sudawg/UFC_Data/main/Data/",
               ["moneyline.csv", "fight_data.csv"]),
}


def download(force=False):
    for d, (base, files) in URLS.items():
        os.makedirs(os.path.join(RAW, d), exist_ok=True)
        for f in files:
            p = os.path.join(RAW, d, f)
            if force or not os.path.exists(p):
                subprocess.run(["curl", "-sSf", "-o", p, base + f], check=True)


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


def of_pair(x):
    try:
        a, b = str(x).split(" of ")
        return float(a), float(b)
    except Exception:
        return np.nan, np.nan


def mmss(x):
    try:
        m, s = str(x).strip().split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def fight_seconds(rnd, t, fmt):
    m = re.search(r"\(([\d-]+)\)", str(fmt))
    lens = [int(v) * 60 for v in m.group(1).split("-")] if m else [300] * 5
    try:
        r = int(rnd)
    except Exception:
        return np.nan
    sec = mmss(t)
    if np.isnan(sec):
        return np.nan
    return sum(lens[: r - 1]) + sec


def sched_rounds(fmt):
    m = re.match(r"\s*(\d+) Rnd", str(fmt))
    return int(m.group(1)) if m else np.nan


METHOD_MAP = {
    "Decision - Unanimous": "U-DEC", "Decision - Split": "S-DEC", "Decision - Majority": "M-DEC",
    "KO/TKO": "KO/TKO", "TKO - Doctor's Stoppage": "KO/TKO", "Submission": "SUB", "DQ": "DQ",
    "Overturned": "Overturned", "Could Not Continue": "CNC", "Other": "Other",
}


def load_greco():
    g = os.path.join(RAW, "greco")
    ev = pd.read_csv(os.path.join(g, "ufc_event_details.csv"))
    ev["EVENT"] = ev.EVENT.str.strip()
    ev["date"] = pd.to_datetime(ev.DATE.str.strip(), format="%B %d, %Y")
    r = pd.read_csv(os.path.join(g, "ufc_fight_results.csv"))
    for c in ["EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD", "TIME FORMAT"]:
        r[c] = r[c].astype(str).str.strip()
    # the results file lists a few renamed events twice (old + new name): keep one row per fight URL,
    # preferring the name present in the event list
    r["_known"] = r.EVENT.isin(ev.EVENT)
    r = r.sort_values("_known", ascending=False).drop_duplicates("URL").sort_index().drop(columns="_known")
    r = r.merge(ev[["EVENT", "date"]], on="EVENT", how="left")
    # some events were renamed on ufcstats (e.g. "UFC Fight Night: X vs. Y" -> "Noche UFC: X vs. Y"):
    # fall back to matching the headline after the colon when it is unique
    suf = ev.assign(s=ev.EVENT.str.split(": ", n=1).str[-1])
    suf = suf[~suf.s.duplicated(keep=False)].set_index("s")
    miss = r.date.isna()
    s = r.loc[miss, "EVENT"].str.split(": ", n=1).str[-1]
    r["event_name"] = r.EVENT  # EVENT kept as-is for joining the per-round stats file
    r.loc[miss, "event_name"] = s.map(suf.EVENT).fillna(r.loc[miss, "EVENT"])
    r.loc[miss, "date"] = s.map(suf.date)
    if r.date.isna().any():
        print("WARN results without event date (dropped):", r.loc[r.date.isna(), "EVENT"].unique())
    r = r.dropna(subset=["date"])
    names = r.BOUT.str.split(r"\s+vs\.\s+", n=1, expand=True)
    r["f1"], r["f2"] = names[0].str.strip(), names[1].str.strip()
    o = r.OUTCOME.str.split("/", expand=True)
    r["o1"], r["o2"] = o[0], o[1]
    r["method"] = r.METHOD.map(METHOD_MAP).fillna(r.METHOD)
    r["finish_round"] = pd.to_numeric(r.ROUND, errors="coerce")
    r["scheduled_rounds"] = r["TIME FORMAT"].map(sched_rounds)
    r["secs"] = [fight_seconds(a, b, c) for a, b, c in zip(r.ROUND, r.TIME, r["TIME FORMAT"])]
    wc = r.WEIGHTCLASS
    r["title_bout"] = wc.str.contains("Title|Championship", case=False).astype(int)
    r["gender"] = np.where(wc.str.contains("Women"), "F", "M")
    r["weight_class"] = (wc.str.replace(r"^UFC |Interim |Ultimate Fighter \d+ |Road to UFC \d+ ", "", regex=True)
                         .str.replace(r"\s*(Tournament )?Title ?Bout|\s*Bout|Superfight Championship", "", regex=True)
                         .str.strip())
    r = r.reset_index(drop=True)
    r["fid"] = r.index

    s = pd.read_csv(os.path.join(g, "ufc_fight_stats.csv"))
    for c in ["EVENT", "BOUT", "FIGHTER"]:
        s[c] = s[c].astype(str).str.strip()
    s = s[s.ROUND.astype(str).str.startswith("Round")]
    s["ssl"], s["ssa"] = zip(*s["SIG.STR."].map(of_pair))
    s["tdl"], s["tda"] = zip(*s["TD"].map(of_pair))
    s["sub"] = pd.to_numeric(s["SUB.ATT"], errors="coerce")
    s["kd"] = pd.to_numeric(s["KD"], errors="coerce")
    s["ctrl"] = s.CTRL.map(mmss)
    agg = s.groupby(["EVENT", "BOUT", "FIGHTER"])[["ssl", "ssa", "tdl", "tda", "sub", "kd", "ctrl"]].sum(min_count=1).reset_index()
    return r, agg


def fighter_history(r, agg):
    """Long table: one row per (fight, fighter) with per-fight numbers, then cumulative PRE-fight stats."""
    rows = []
    for side, me, op, res in [(1, "f1", "f2", "o1"), (2, "f2", "f1", "o2")]:
        t = r[["fid", "date", "EVENT", "BOUT", "secs", "method", "title_bout"]].copy()
        t["fighter"], t["opp"], t["res"] = r[me], r[op], r[res]
        rows.append(t)
    L = pd.concat(rows, ignore_index=True)
    a = agg.rename(columns={"FIGHTER": "fighter"})
    L = L.merge(a, on=["EVENT", "BOUT", "fighter"], how="left")
    ao = a.rename(columns={"fighter": "opp", "ssl": "o_ssl", "ssa": "o_ssa", "tdl": "o_tdl", "tda": "o_tda",
                           "sub": "o_sub", "kd": "o_kd", "ctrl": "o_ctrl"})
    L = L.merge(ao, on=["EVENT", "BOUT", "opp"], how="left")
    L = L.sort_values(["fighter", "date", "fid"]).reset_index(drop=True)
    L["has_stats"] = L.ssl.notna() & L.secs.notna()
    L["mins"] = np.where(L.has_stats, L.secs / 60.0, 0.0)
    L["W"] = (L.res == "W").astype(int)
    L["Lo"] = (L.res == "L").astype(int)
    L["D"] = (L.res == "D").astype(int)
    L["KOW"] = ((L.res == "W") & (L.method == "KO/TKO")).astype(int)
    L["SUBW"] = ((L.res == "W") & (L.method == "SUB")).astype(int)
    L["DECW"] = ((L.res == "W") & L.method.str.contains("DEC")).astype(int)
    L["KOL"] = ((L.res == "L") & (L.method == "KO/TKO")).astype(int)
    for c in ["ssl", "ssa", "o_ssl", "tdl", "tda", "o_tdl", "o_tda", "sub", "kd", "ctrl"]:
        L[c + "_z"] = np.where(L.has_stats, L[c].fillna(0), 0.0)
    g = L.groupby("fighter", sort=False)
    def prior(col):  # cumulative sum EXCLUDING current fight
        return g[col].cumsum() - L[col]
    out = pd.DataFrame({"fid": L.fid, "fighter": L.fighter, "date": L.date})
    out["fights_ufc"] = g.cumcount()
    out["wins"] = prior("W"); out["losses"] = prior("Lo"); out["draws"] = prior("D")
    out["ko_wins"] = prior("KOW"); out["sub_wins"] = prior("SUBW"); out["dec_wins"] = prior("DECW")
    out["ko_losses"] = prior("KOL")
    out["title_bouts"] = prior("title_bout")
    mins = prior("mins")
    out["ufc_minutes"] = mins
    safe = mins.where(mins > 0)
    out["sig_str_landed_pm"] = prior("ssl_z") / safe
    out["sig_str_absorbed_pm"] = prior("o_ssl_z") / safe
    out["sig_str_acc"] = prior("ssl_z") / prior("ssa_z").where(lambda x: x > 0)
    out["td_avg"] = prior("tdl_z") / safe * 15
    out["td_acc"] = prior("tdl_z") / prior("tda_z").where(lambda x: x > 0)
    out["td_def"] = 1 - prior("o_tdl_z") / prior("o_tda_z").where(lambda x: x > 0)
    out["sub_avg"] = prior("sub_z") / safe * 15
    out["kd_avg"] = prior("kd_z") / safe * 15
    out["ctrl_share"] = prior("ctrl_z") / (safe * 60)
    out["days_since_last_fight"] = (L.date - g.date.shift(1)).dt.days
    # streaks before this fight
    ws, ls = [], []
    for _, grp in L.groupby("fighter", sort=False):
        w = l = 0
        for res in grp.res:
            ws.append(w); ls.append(l)
            if res == "W": w, l = w + 1, 0
            elif res == "L": w, l = 0, l + 1
            else: w = l = 0
    # L is sorted by fighter so groupby(sort=False) iteration order equals row order
    out["win_streak"] = ws
    out["lose_streak"] = ls
    return out


def load_tott():
    t = pd.read_csv(os.path.join(RAW, "greco", "ufc_fighter_tott.csv"))
    def inches(h):
        m = re.match(r"(\d+)' (\d+)", str(h))
        return (int(m.group(1)) * 12 + int(m.group(2))) * 2.54 if m else np.nan
    t["height_cm"] = t.HEIGHT.map(inches)
    t["reach_cm"] = pd.to_numeric(t.REACH.astype(str).str.replace('"', ""), errors="coerce") * 2.54
    t["weight_lbs"] = pd.to_numeric(t.WEIGHT.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    t["dob"] = pd.to_datetime(t.DOB, format="%b %d, %Y", errors="coerce")
    t["stance"] = t.STANCE
    t["FIGHTER"] = t.FIGHTER.str.strip()
    return t[["FIGHTER", "height_cm", "reach_cm", "weight_lbs", "dob", "stance"]]


WC_LBS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
          "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 245}


def wc_lbs(wc):
    for k in sorted(WC_LBS, key=len, reverse=True):
        if k in str(wc):
            return WC_LBS[k]
    return np.nan


def attach_tott(df, tott, side):
    """Tale of tape by name; for duplicate names pick the profile whose weight is closest to the bout's class."""
    m = df[["fid", f"{side}_name", "weight_class"]].merge(tott, left_on=f"{side}_name", right_on="FIGHTER", how="left")
    m["dw"] = (m.weight_lbs - m.weight_class.map(wc_lbs)).abs().fillna(999)
    m = m.sort_values(["fid", "dw"]).drop_duplicates("fid")
    m = m.set_index("fid")
    return m


def load_master():
    m = pd.read_csv(os.path.join(RAW, "ultimate", "ufc-master.csv"))
    m["date"] = pd.to_datetime(m.date)
    m = m.copy()
    m["kR"], m["kB"] = m.R_fighter.map(norm), m.B_fighter.map(norm)
    return m


def load_sudawg():
    ml = pd.read_csv(os.path.join(RAW, "sudawg", "moneyline.csv"))
    ml = ml[ml.Card.astype(str).str.startswith("UFC")].copy()
    d = ml.Card.str.extract(r"(\w{3}) (\d+)\w\w (\d{4})$")
    ml["date"] = pd.to_datetime(d[0] + " " + d[1] + " " + d[2], format="%b %d %Y", errors="coerce")
    for c in ["fighter_a_odds", "fighter_b_odds"]:
        ml[c] = pd.to_numeric(ml[c].astype(str).str.replace("+", "", regex=False), errors="coerce")
    ml = ml.dropna(subset=["fighter_a_odds", "fighter_b_odds", "date"])
    ml["ka"], ml["kb"] = ml.fighter_a.map(norm), ml.fighter_b.map(norm)
    return ml.drop_duplicates(["date", "ka", "kb"])


def main():
    if "--download" in sys.argv:
        download(force=True)
    else:
        download(force=False)
    r, agg = load_greco()
    hist = fighter_history(r, agg)
    r = r[r.date >= "2010-01-01"].copy()
    r["k1"], r["k2"] = r.f1.map(norm), r.f2.map(norm)

    # ---- join ufc-master (odds + corner orientation) on date(+-1 day) and unordered name pair
    M = load_master()
    mi = {}
    for i, row in M.iterrows():
        for dd in (0, -1, 1):
            mi.setdefault((row.date + pd.Timedelta(days=dd), frozenset((row.kR, row.kB))), i)
    by_date_name = {}
    for i, row in M.iterrows():
        for k in (row.kR, row.kB):
            by_date_name.setdefault((row.date, k), []).append(i)
    match, used = {}, set()
    for fid, d, k1, k2 in zip(r.fid, r.date, r.k1, r.k2):
        i = mi.get((d, frozenset((k1, k2))))
        if i is None:  # fallback: same date, exactly one fighter name matches and it is unique that date
            c = set(by_date_name.get((d, k1), [])) ^ set()
            c2 = set(by_date_name.get((d, k2), []))
            cand = (c | c2)
            if len(cand) == 1:
                i = cand.pop()
        if i is not None and i not in used:
            match[fid] = i; used.add(i)
    print(f"ufc-master rows matched: {len(used)}/{len(M)}")

    S = load_sudawg()
    si = {(row.date, row.ka, row.kb): (row.fighter_a_odds, row.fighter_b_odds) for row in S.itertuples()}

    recs = []
    for row in r.itertuples():
        i = match.get(row.fid)
        rec = {"fid": row.fid, "date": row.date, "event": row.event_name, "weight_class": row.weight_class,
               "gender": row.gender, "title_bout": row.title_bout, "method": row.method,
               "finish_round": row.finish_round, "scheduled_rounds": row.scheduled_rounds,
               "method_detail": row.DETAILS, "ufcstats_url": row.URL}
        res = {row.f1: row.o1, row.f2: row.o2}
        if i is not None:
            mm = M.loc[i]
            # orient by red corner; map master names to greco names
            if norm(mm.R_fighter) == row.k2 or norm(mm.B_fighter) == row.k1:
                a, b = row.f2, row.f1
            else:
                a, b = row.f1, row.f2
            rec.update(a_ml=mm.R_odds, b_ml=mm.B_odds, odds_source="ufc-master(BFO)" if pd.notna(mm.R_odds) else None,
                       a_dec_odds=mm.r_dec_odds, b_dec_odds=mm.b_dec_odds, a_sub_odds=mm.r_sub_odds,
                       b_sub_odds=mm.b_sub_odds, a_ko_odds=mm.r_ko_odds, b_ko_odds=mm.b_ko_odds,
                       a_rank=mm.R_match_weightclass_rank, b_rank=mm.B_match_weightclass_rank, corner="red/blue")
        else:
            a, b = sorted([row.f1, row.f2])
            rec.update(corner="alpha")
        rec["fighter_a"], rec["fighter_b"] = a, b
        if pd.isna(rec.get("a_ml")) or pd.isna(rec.get("b_ml")):
            ka, kb = norm(a), norm(b)
            for dd in (0, -1, 1):
                d = row.date + pd.Timedelta(days=dd)
                if (d, ka, kb) in si:
                    rec["a_ml"], rec["b_ml"] = si[(d, ka, kb)]; rec["odds_source"] = "sudawg(BFO closing range)"; break
                if (d, kb, ka) in si:
                    rec["b_ml"], rec["a_ml"] = si[(d, kb, ka)]; rec["odds_source"] = "sudawg(BFO closing range)"; break
        ra = res[a]
        rec["winner"] = {"W": "A", "L": "B", "D": "draw", "NC": "nc"}.get(ra, ra)
        recs.append(rec)
    df = pd.DataFrame(recs)

    # ---- pre-fight features
    H = hist.set_index(["fid", "fighter"])
    feat_cols = [c for c in hist.columns if c not in ("fid", "fighter", "date")]
    tott = load_tott()
    for side in ("a", "b"):
        df[f"{side}_name"] = df[f"fighter_{side}"]
        f = H.reindex(list(zip(df.fid, df[f"fighter_{side}"])))[feat_cols].reset_index(drop=True)
        f.columns = [f"{side}_{c}" for c in feat_cols]
        df = pd.concat([df, f], axis=1)
        t = attach_tott(df, tott, side).reindex(df.fid)
        df[f"{side}_height_cm"] = t.height_cm.values.round(1)
        df[f"{side}_reach_cm"] = t.reach_cm.values.round(1)
        df[f"{side}_stance"] = t.stance.values
        df[f"{side}_age"] = ((df.date - pd.to_datetime(t.dob.values)).dt.days / 365.25).round(2)
        df = df.drop(columns=[f"{side}_name"])

    def implied(ml):
        ml = pd.to_numeric(ml, errors="coerce")
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    for c in ["a_ml", "b_ml"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].abs() < 100, c] = np.nan  # invalid American odds
    df["a_imp"] = implied(df.a_ml); df["b_imp"] = implied(df.b_ml)
    df["vig_sum"] = df.a_imp + df.b_imp
    bad = (df.vig_sum < 0.98) | (df.vig_sum > 1.25)
    print("dropping implausible odds pairs:", int(bad.sum()))
    df.loc[bad, ["a_ml", "b_ml", "a_imp", "b_imp", "vig_sum", "odds_source"]] = np.nan
    df = df.drop(columns=["a_imp", "b_imp"])

    front = ["date", "event", "fighter_a", "fighter_b", "a_ml", "b_ml", "odds_source", "winner", "method",
             "finish_round", "scheduled_rounds", "weight_class", "gender", "title_bout", "corner", "vig_sum",
             "a_ko_odds", "b_ko_odds", "a_sub_odds", "b_sub_odds", "a_dec_odds", "b_dec_odds", "a_rank", "b_rank"]
    rest = sorted([c for c in df.columns if c not in front and c not in ("fid", "method_detail", "ufcstats_url")],
                  key=lambda c: (c[2:], c[0]))
    df = df[front + rest + ["method_detail", "ufcstats_url"]].sort_values(["date", "event", "fighter_a"])
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    print("wrote", OUT, df.shape)


if __name__ == "__main__":
    main()
