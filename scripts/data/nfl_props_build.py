#!/usr/bin/env python3
"""Build NFL player-week stats and NFL player-prop lines joined to results.

Inputs (downloaded beforehand, see README notes at bottom):
  data/raw/nflverse/            player_stats.csv (1999-2024), stats_player_week_<yr>.csv (2024+),
                                snap_counts_<yr>.csv, players.csv, games.csv
  data/raw/props_nfl/repos/     bare git clones of GitHub repos holding prop snapshots:
     zjhabib_BeatVegas.git                       2017-18 QB completions (propbets.football), -115-ish prices
     davidcantugtr_nfl-player-prop-opportunity.git  2026 The-Odds-API snapshots (multi-book + DFS)
     mack.git (mackman08/nfl-prop-data)          2026 DK/FD snapshots, one per git commit
Outputs:
  data/processed/nfl_player_week.csv
  data/processed/nfl_props.csv          main lines, one row per player/game/market/book (latest pre-kick snapshot)
  data/processed/nfl_props_alt.csv      alternate / ladder lines (same schema)
"""
import io, re, subprocess, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NV = ROOT / "data/raw/nflverse"
PR = ROOT / "data/raw/props_nfl"
REPOS = PR / "repos"
OUT = ROOT / "data/processed"

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}
NAME2ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
NICK2ABBR = {k.split()[-1]: v for k, v in NAME2ABBR.items()}

MARKET_STAT = {
    "passing_yards": "passing_yards", "passing_tds": "passing_tds", "completions": "completions",
    "pass_attempts": "attempts", "interceptions": "interceptions", "rushing_yards": "rushing_yards",
    "rush_attempts": "carries", "receiving_yards": "receiving_yards", "receptions": "receptions",
    "rush_rec_yards": "rush_rec_yards", "anytime_td": "anytime_td",
}


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`,-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def git_show(repo, rev_path):
    return subprocess.run(["git", "-C", str(repo), "show", rev_path], capture_output=True, check=True).stdout


# ----------------------------------------------------------------------------- stats
def build_player_week():
    games = pd.read_csv(NV / "games.csv", low_memory=False)
    old = pd.read_csv(NV / "player_stats.csv", low_memory=False)
    old = old[old.season <= 2023].rename(columns={"recent_team": "team"})
    new = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(NV.glob("stats_player_week_*.csv"))])
    new = new.rename(columns={"passing_interceptions": "interceptions"})
    keep = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week", "season_type",
            "completions", "attempts", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards",
            "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "target_share",
            "air_yards_share", "fantasy_points", "fantasy_points_ppr", "special_teams_tds"]
    df = pd.concat([old[[c for c in keep if c in old]], new[[c for c in keep if c in new]]], ignore_index=True)
    df = df.rename(columns={"player_display_name": "player", "opponent_team": "opp"})
    df["team"] = df.team.replace(TEAM_FIX); df["opp"] = df.opp.replace(TEAM_FIX)
    # game_id / home from nflverse schedule
    g = games[["game_id", "season", "week", "home_team", "away_team", "gameday", "gametime"]].copy()
    g["home_team"] = g.home_team.replace(TEAM_FIX); g["away_team"] = g.away_team.replace(TEAM_FIX)
    gh = g.rename(columns={"home_team": "team", "away_team": "opp"}).assign(home=1)
    ga = g.rename(columns={"away_team": "team", "home_team": "opp"}).assign(home=0)
    gg = pd.concat([gh, ga])
    df = df.merge(gg[["season", "week", "team", "opp", "game_id", "home", "gameday"]], on=["season", "week", "team", "opp"], how="left")
    # snap pct via pfr id
    pl = pd.read_csv(NV / "players.csv", usecols=["gsis_id", "pfr_id"]).dropna().drop_duplicates("pfr_id")
    sc = pd.concat([pd.read_csv(p, usecols=["game_id", "pfr_player_id", "offense_snaps", "offense_pct"])
                    for p in sorted(NV.glob("snap_counts_*.csv"))])
    sc = sc.merge(pl, left_on="pfr_player_id", right_on="pfr_id").rename(columns={"gsis_id": "player_id", "offense_pct": "snap_pct"})
    sc = sc.drop_duplicates(["game_id", "player_id"])
    df = df.merge(sc[["game_id", "player_id", "offense_snaps", "snap_pct"]], on=["game_id", "player_id"], how="left")
    cols = ["season", "week", "season_type", "game_id", "gameday", "player_id", "player", "position", "team", "opp", "home",
            "completions", "attempts", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards",
            "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "special_teams_tds",
            "target_share", "air_yards_share", "offense_snaps", "snap_pct", "fantasy_points", "fantasy_points_ppr"]
    df = df[cols].sort_values(["season", "week", "team", "player"])
    df.to_csv(OUT / "nfl_player_week.csv", index=False)
    print(f"player_week rows={len(df)} seasons={df.season.min()}-{df.season.max()} "
          f"game_id%={df.game_id.notna().mean():.3f} snap%(2012+)={df[df.season>=2012].snap_pct.notna().mean():.3f}")
    return df, games


# ----------------------------------------------------------------------------- props sources
def src_beatvegas():
    raw = git_show(REPOS / "zjhabib_BeatVegas.git", "HEAD:data/Prop_Betting_Data.csv")
    (PR / "beatvegas_Prop_Betting_Data.csv").write_bytes(raw)
    d = pd.read_csv(io.BytesIO(raw))
    wk = d.Week.str.extract(r"(\d\d)NFL(\d\d)")
    out = pd.DataFrame({
        "season": 2000 + wk[0].astype(int), "week": wk[1].astype(int), "date": pd.NA, "player": d.Player,
        "team": d.Team.replace(TEAM_FIX), "opp": d.Opp.replace(TEAM_FIX), "market": "completions", "line": d.Line,
        "over_odds": d.Over, "under_odds": d.Under, "book": "unknown (propbets.football via zjhabib/BeatVegas)",
        "source_type": "sportsbook", "source": "github:zjhabib/BeatVegas", "snapshot_utc": pd.NA, "line_type": "main",
        "src_actual": d.Res,
    })
    return out


ODDSAPI_MKT = {"player_reception_yds": "receiving_yards", "player_rush_yds": "rushing_yards",
               "player_pass_yds": "passing_yards", "player_receptions": "receptions", "player_pass_tds": "passing_tds",
               "player_anytime_td": "anytime_td", "player_rush_attempts": "rush_attempts",
               "player_pass_completions": "completions", "player_pass_attempts": "pass_attempts"}
DFS_BOOKS = {"prizepicks", "underdog", "pick6", "betr_us_dfs", "sleeper", "dabble_au"}


def src_oddsapi():
    repo = REPOS / "davidcantugtr_nfl-player-prop-opportunity.git"
    files = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD", "data/snapshots"],
                           capture_output=True, text=True, check=True).stdout.split()
    frames = []
    dst = PR / "oddsapi_snapshots"; dst.mkdir(exist_ok=True)
    for f in files:
        if not f.endswith("_player_props.csv"):
            continue
        raw = git_show(repo, "HEAD:" + f)
        (dst / Path(f).name).write_bytes(raw)
        d = pd.read_csv(io.BytesIO(raw))
        d["snapshot_utc"] = pd.to_datetime(Path(f).name.split("_")[0], format="%Y%m%dT%H%M%SZ", utc=True)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["commence"] = pd.to_datetime(d["Commence Time"], utc=True)
    d = d[d.snapshot_utc < d.commence]  # pre-game only
    d["mkt"] = d["Market Key"].map(ODDSAPI_MKT)
    d = d[d.mkt.notna() & (d["Market Window"].fillna("Full Game") == "Full Game")]
    # latest pre-kick snapshot per (event, book, player, market, line, side)
    d = d.sort_values("snapshot_utc").drop_duplicates(["Event ID", "Book Key", "Player", "mkt", "Line", "Side"], keep="last")
    piv = d.pivot_table(index=["Event ID", "Season", "Week", "commence", "Home", "Away", "Book Key", "Bookmaker", "Player", "mkt", "Line"],
                        columns="Side", values="American Odds", aggfunc="last").reset_index()
    piv = piv.merge(d.groupby(["Event ID", "Book Key", "Player", "mkt", "Line"]).snapshot_utc.max().reset_index(),
                    on=["Event ID", "Book Key", "Player", "mkt", "Line"])
    for c in ("Over", "Under"):
        if c not in piv: piv[c] = np.nan
    # main line per (event,book,player,market): both sides priced and vig-free prob closest to .5; others = alt
    def imp(o):
        return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))
    po, pu = imp(piv.Over.astype(float)), imp(piv.Under.astype(float))
    piv["dist"] = np.abs(po / (po + pu) - 0.5)
    piv["dist"] = piv.dist.fillna(9)
    piv = piv.sort_values("dist")
    piv["line_type"] = np.where(piv.duplicated(["Event ID", "Book Key", "Player", "mkt"]), "alt", "main")
    dfs = piv["Book Key"].isin(DFS_BOOKS)
    out = pd.DataFrame({
        "season": piv.Season, "week": piv.Week,
        "date": piv.commence.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"),
        "player": piv.Player, "team": pd.NA, "home_abbr": piv.Home.map(NAME2ABBR), "away_abbr": piv.Away.map(NAME2ABBR),
        "market": piv.mkt, "line": piv.Line,
        "over_odds": np.where(dfs, np.nan, piv.Over), "under_odds": np.where(dfs, np.nan, piv.Under),
        "dfs_over_odds": np.where(dfs, piv.Over, np.nan), "dfs_under_odds": np.where(dfs, piv.Under, np.nan),
        "book": piv.Bookmaker, "source_type": np.where(dfs, "dfs", "sportsbook"),
        "source": "github:davidcantugtr/nfl-player-prop-opportunity (The Odds API)",
        "snapshot_utc": piv.snapshot_utc.astype(str), "commence_utc": piv.commence.astype(str), "line_type": piv.line_type,
    })
    return out


MACK_MKT = {"Passing Yards": "passing_yards", "Rushing Yards": "rushing_yards", "Receiving Yards": "receiving_yards",
            "Receptions": "receptions", "Rush + Receiving Yards": "rush_rec_yards", "Passing Touchdowns": "passing_tds",
            "Anytime Touchdown Scorer": "anytime_td", "Anytime TD": "anytime_td"}


def src_mack(games):
    repo = REPOS / "mack.git"
    log = subprocess.run(["git", "-C", str(repo), "log", "--format=%H %cI", "--", "odds/nfl_player_props_latest.csv"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    frames = []
    dst = PR / "mack_snapshots"; dst.mkdir(exist_ok=True)
    for line in filter(None, log):
        sha, ts = line.split()
        try:
            raw = git_show(repo, f"{sha}:odds/nfl_player_props_latest.csv")
        except subprocess.CalledProcessError:
            continue
        snap = pd.Timestamp(ts).tz_convert("UTC")
        (dst / f"{snap:%Y%m%dT%H%M%SZ}_nfl_player_props.csv").write_bytes(raw)
        d = pd.read_csv(io.BytesIO(raw))
        d["snapshot_utc"] = snap
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["alt"] = d.market.str.startswith("Alt ")
    d["mkt"] = d.market.str.replace(r"^Alt ", "", regex=True).map(MACK_MKT)
    d = d[d.mkt.notna()]
    tm = d.game.str.split(" @ ", expand=True)
    def abbr(x):
        x = str(x).strip()
        if x in NAME2ABBR: return NAME2ABBR[x]
        a = x.split()[0]; a = TEAM_FIX.get(a, a)
        return a if a in NAME2ABBR.values() else NICK2ABBR.get(x.split()[-1])
    d["away_abbr"] = tm[0].map(abbr); d["home_abbr"] = tm[1].map(abbr)
    # attach nflverse game -> kickoff
    g = games[games.season >= 2026].copy()
    g["kick"] = pd.to_datetime(g.gameday + " " + g.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    d = d.merge(g[["season", "week", "home_team", "away_team", "kick", "gameday"]], left_on=["home_abbr", "away_abbr"],
                right_on=["home_team", "away_team"], how="inner")
    d = d[(d.snapshot_utc < d.kick) & (d.kick - d.snapshot_utc < pd.Timedelta(days=8))]
    d = d.sort_values("snapshot_utc").drop_duplicates(["season", "week", "home_abbr", "sportsbook", "player", "mkt", "line", "alt"], keep="last")
    # DK publishes only a ladder of 'over' prices (no under) -> alt; main = book line with both sides priced
    d["line_type"] = np.where(~d.alt & d.under_odds.notna(), "main", "alt")
    m = d[d.line_type == "main"].sort_values("snapshot_utc")
    dup = m.duplicated(["season", "week", "home_abbr", "sportsbook", "player", "mkt"], keep="last")
    d.loc[m.index[dup], "line_type"] = "alt"
    out = pd.DataFrame({
        "season": d.season, "week": d.week, "date": d.gameday, "player": d.player, "team": pd.NA,
        "home_abbr": d.home_abbr, "away_abbr": d.away_abbr, "market": d.mkt, "line": d.line,
        "over_odds": d.over_odds, "under_odds": d.under_odds, "book": d.sportsbook, "source_type": "sportsbook",
        "source": "github:mackman08/nfl-prop-data (git history)", "snapshot_utc": d.snapshot_utc.astype(str),
        "commence_utc": d.kick.astype(str), "line_type": d.line_type,
    })
    return out


# ----------------------------------------------------------------------------- join
def join_actuals(props, pw):
    pw = pw[pw.season_type.isin(["REG", "POST"])].copy()
    pw["rush_rec_yards"] = pw.rushing_yards.fillna(0) + pw.receiving_yards.fillna(0)
    pw["anytime_td"] = pw.rushing_tds.fillna(0) + pw.receiving_tds.fillna(0) + pw.special_teams_tds.fillna(0)
    pw["nkey"] = pw.player.map(norm_name)
    pw["stats_imputed_zero"] = 0
    # players who took offensive snaps but recorded no stat have no nflverse stat row -> actual = 0
    sc = pd.concat([pd.read_csv(p, usecols=["game_id", "season", "game_type", "week", "player", "pfr_player_id", "position",
                                            "team", "opponent", "offense_snaps"]) for p in sorted(NV.glob("snap_counts_*.csv"))])
    sc = sc[(sc.offense_snaps > 0) & sc.position.isin(["QB", "RB", "FB", "WR", "TE"])]
    sc["team"] = sc.team.replace(TEAM_FIX); sc["nkey"] = sc.player.map(norm_name)
    have = set(zip(pw.game_id, pw.nkey))
    z = sc[[(g, k) not in have for g, k in zip(sc.game_id, sc.nkey)]]
    ids = pd.read_csv(NV / "players.csv", usecols=["gsis_id", "pfr_id"]).dropna().drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    z = pd.DataFrame({"season": z.season, "week": z.week, "season_type": np.where(z.game_type == "REG", "REG", "POST"),
                      "game_id": z.game_id, "player_id": z.pfr_player_id.map(ids), "player": z.player, "position": z.position,
                      "team": z.team, "opp": z.opponent.replace(TEAM_FIX), "nkey": z.nkey, "stats_imputed_zero": 1})
    for c in set(MARKET_STAT.values()) | {"rushing_tds", "receiving_tds", "interceptions"}:
        z[c] = 0.0
    z["player_id"] = z.player_id.fillna("pfr:" + sc.loc[z.index, "pfr_player_id"].astype(str) if False else z.player_id)
    pw = pd.concat([pw, z], ignore_index=True)
    pw = pw[pw.player_id.notna()]
    props = props.copy()
    ALIAS = {"hollywood brown": "marquise brown", "gabe davis": "gabriel davis", "chigoziem okonkwo": "chig okonkwo",
             "joshua palmer": "josh palmer", "nick mullins": "nick mullens", "cameron ward": "cam ward"}
    pw["nkey"] = pw.nkey.replace(ALIAS)
    props["nkey"] = props.player.map(norm_name).replace(ALIAS)
    props["_i"] = range(len(props))
    # A) with team known
    a = props[props.team.notna()].merge(pw, left_on=["season", "week", "team", "nkey"], right_on=["season", "week", "team", "nkey"],
                                         how="left", suffixes=("", "_s"))
    # B) team unknown: candidate players from either side of the game
    b0 = props[props.team.isna()]
    bh = b0.merge(pw, left_on=["season", "week", "home_abbr", "nkey"], right_on=["season", "week", "team", "nkey"], suffixes=("", "_s"))
    ba = b0.merge(pw, left_on=["season", "week", "away_abbr", "nkey"], right_on=["season", "week", "team", "nkey"], suffixes=("", "_s"))
    b = pd.concat([bh, ba])
    if "team_s" in b:
        b["team"] = b["team_s"].fillna(b["team"])
    m = pd.concat([a, b], ignore_index=True).drop_duplicates("_i")
    m = props[["_i"]].merge(m, on="_i", how="left", suffixes=("_x", ""))  # keep unmatched
    miss = m.season.isna()
    m.loc[miss, props.columns.drop("_i")] = props.set_index("_i").loc[m.loc[miss, "_i"], props.columns.drop("_i")].values
    def act(r):
        c = MARKET_STAT.get(r.market)
        return r[c] if c in r and pd.notna(r.get("player_id")) else np.nan
    m["actual"] = m.apply(act, axis=1)
    m["result"] = np.select([m.actual > m.line, m.actual < m.line, m.actual == m.line], ["over", "under", "push"], default="")
    if "src_actual" in m:
        m["src_actual_mismatch"] = (m.src_actual.notna() & m.actual.notna() & (m.src_actual != m.actual)).astype(int)
    if "opp_s" in m:
        m["opp"] = m["opp_s"].fillna(m["opp"])
    m = m.rename(columns={"player_x": "player"}) if "player_x" in m else m
    m["date"] = m.date.fillna(m.gameday) if "gameday" in m else m.date
    cols = ["season", "week", "date", "game_id", "player", "player_id", "position", "team", "opp", "market", "line", "over_odds",
            "under_odds", "dfs_over_odds", "dfs_under_odds", "book", "source_type", "line_type", "actual", "result",
            "snapshot_utc", "commence_utc", "source", "src_actual", "src_actual_mismatch", "stats_imputed_zero"]
    for c in cols:
        if c not in m: m[c] = np.nan
    return m[cols]


def main():
    pw, games = build_player_week()
    props = pd.concat([src_beatvegas(), src_oddsapi(), src_mack(games)], ignore_index=True)
    # player name in stats has 'player' column; ensure prop name preserved
    j = join_actuals(props, pw)
    j = j.sort_values(["season", "week", "player", "market", "book", "line"])
    j[j.line_type == "main"].drop(columns="line_type").to_csv(OUT / "nfl_props.csv", index=False)
    j[j.line_type == "alt"].drop(columns="line_type").to_csv(OUT / "nfl_props_alt.csv", index=False)
    mn = j[j.line_type == "main"]
    print("\n=== main-line props: rows by season x market ===")
    print(mn.pivot_table(index="season", columns="market", values="line", aggfunc="size", fill_value=0))
    print("\nrows by book/source_type:"); print(mn.groupby(["source_type", "book"]).size())
    played = mn[mn.season.lt(2026) | mn.week.le(5)]
    print(f"match rate on completed weeks: {played.player_id.notna().mean():.3f} (n={len(played)}); unmatched are mostly inactive players (void bets)")
    print(f"\nmatch rate (player_id): {mn.player_id.notna().mean():.3f}  "
          f"graded(actual present): {mn.actual.notna().mean():.3f}  "
          f"sportsbook rows w/ both prices: {mn[mn.source_type=='sportsbook'][['over_odds','under_odds']].notna().all(axis=1).mean():.3f}")
    gr = mn[mn.result.isin(["over", "under"])]
    print("\nover-hit rate by market (excl push):"); print(gr.groupby("market").result.apply(lambda s: f"{(s=='over').mean():.3f} n={len(s)}"))
    print("\nover-hit by season/source:"); print(gr.groupby(["season", "source_type"]).result.apply(lambda s: f"{(s=='over').mean():.3f} n={len(s)}"))
    if mn.src_actual_mismatch.notna().any():
        print("BeatVegas src actual mismatches vs nflverse:", int(mn.src_actual_mismatch.sum()))
    um = mn[mn.player_id.isna()]
    print("\nunmatched sample:", um[["season", "week", "player", "book"]].drop_duplicates("player").head(15).to_string())
    pre = mn.dropna(subset=["snapshot_utc", "commence_utc"])
    if len(pre):
        print("all snapshots pre-kick:", bool((pd.to_datetime(pre.snapshot_utc) < pd.to_datetime(pre.commence_utc)).all()))


if __name__ == "__main__":
    main()
