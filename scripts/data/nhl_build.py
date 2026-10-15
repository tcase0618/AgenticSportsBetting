"""Build data/processed/nhl.csv (one row per game, seasons 2007-08 .. 2025-26; season = END year).

Sources (all in data/raw/nhl/):
  fdaproj_nhl_games_odds.csv   - github.com/ethanbell528-cmd/fda-project-1 data/nhl.csv
        NHL Stats API scores/OT/SO/playoff flags + SportsbookReviewsOnline (SBRO) closing ML,
        closing total, puck line (2014-15+) for 2007-08 .. Nov 2022 (via Internet Archive).
  fdaproj_nhl_goalie_starts.csv - same repo, starting goalies (NHL Stats API)
  sbro_nhl_archive_10Y.json    - github.com/flancast90/sportsbookreview-scraper data/nhl_archive_10Y.json
        SBRO 2011-12..2021-22 parsed; used ONLY for opening ML and puck-line odds when its closing
        ML agrees with the primary closing ML (and the puck line side agrees).
  oddsapi_noon_snapshots/odds_YYYY-MM-DD.csv - github.com/nielsenz/odds-api-current-save
        odds-data/historical/: The Odds API snapshots at 17:00 UTC (noon ET) for BetMGM & Caesars,
        Oct 2023 .. Jan 2026 (+ a few Feb 2026 days from data/). NOT true closing lines.
"""
import glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'nhl')
OUT = os.path.join(ROOT, 'data', 'processed', 'nhl.csv')
BOOKS = ['caesars', 'betmgm']

FLAN = {'Lightning': 'TBL', 'Tampa Bay': 'TBL', 'Tampa': 'TBL', 'Penguins': 'PIT', 'Rangers': 'NYR', 'Bruins': 'BOS',
        'St.Louis': 'STL', 'Capitals': 'WSH', 'Blackhawks': 'CHI', 'Sharks': 'SJS', 'Predators': 'NSH', 'Canadiens': 'MTL',
        'Ducks': 'ANA', 'Kings': 'LAK', 'Avalanche': 'COL', 'Islanders': 'NYI', 'NY Islanders': 'NYI', 'Stars': 'DAL',
        'Wild': 'MIN', 'Hurricanes': 'CAR', 'Flyers': 'PHI', 'Maple Leafs': 'TOR', 'Senators': 'OTT', 'Jets': 'WPG',
        'WinnipegJets': 'WPG', 'Flames': 'CGY', 'Blue Jackets': 'CBJ', 'Red Wings': 'DET', 'Oilers': 'EDM', 'Panthers': 'FLA',
        'Canucks': 'VAN', 'Devils': 'NJD', 'Sabres': 'BUF', 'Coyotes': 'ARI', 'Phoenix': 'ARI', 'Arizonas': 'ARI',
        'Golden Knights': 'VGK', 'SeattleKraken': 'SEA', 'Kraken': 'SEA'}
FULL = {'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS', 'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY',
        'Carolina Hurricanes': 'CAR', 'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
        'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM', 'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK',
        'Minnesota Wild': 'MIN', 'Montréal Canadiens': 'MTL', 'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH',
        'New Jersey Devils': 'NJD', 'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT',
        'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS', 'Seattle Kraken': 'SEA',
        'St Louis Blues': 'STL', 'St. Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR',
        'Utah Hockey Club': 'UTA', 'Utah Mammoth': 'UTA', 'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
        'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG'}


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def base_games():
    m = pd.read_csv(os.path.join(RAW, 'fdaproj_nhl_games_odds.csv'), low_memory=False)
    m = m[(m.season >= 2007) & (m.season <= 2025)]  # start years 2007..2025 -> 2007-08 .. 2025-26
    h = m[m.home_away == 'H'].set_index('game_id')
    a = m[m.home_away == 'A'].set_index('game_id')
    assert h.index.is_unique and a.index.is_unique and len(h) == len(a)
    a = a.loc[h.index]
    g = pd.DataFrame({
        'game_id': h.index.astype(str), 'date': h.date.values, 'season': h.season.values + 1,
        'home': h.team.values, 'away': a.team.values, 'home_score': h.score_for.values, 'away_score': h.score_against.values,
        'home_ml': h.moneyline.values, 'away_ml': a.moneyline.values, 'total': h.total_line.values, 'spread_home': h.line.values,
        'playoff': (h.game_type == 'playoff').astype(int).values, 'decided_in': h.decided_in.values,
    })
    g['went_ot'] = g.decided_in.isin(['OT', 'SO']).astype(int)
    g['odds_source'] = np.where(g.home_ml.notna(), 'SBRO', '')
    g['book'] = np.where(g.home_ml.notna(), 'SBRO consensus', '')
    for c in ['home_ml_open', 'away_ml_open', 'over_odds', 'under_odds', 'spread_home_odds', 'spread_away_odds']:
        g[c] = np.nan
    gs = pd.read_csv(os.path.join(RAW, 'fdaproj_nhl_goalie_starts.csv'))
    gs = gs[gs.started == 1].drop_duplicates(['game_id', 'team'])
    gs = gs.set_index([gs.game_id.astype(str), gs.team]).goalie
    g['home_goalie'] = [gs.get((i, t), '') for i, t in zip(g.game_id, g.home)]
    g['away_goalie'] = [gs.get((i, t), '') for i, t in zip(g.game_id, g.away)]
    return g


def enrich_flancast(g):
    f = json.load(open(os.path.join(RAW, 'sbro_nhl_archive_10Y.json')))
    rows = []
    for r in f:
        h, a = FLAN.get(str(r['home_team'])), FLAN.get(str(r['away_team']))
        if not h or not a or h == a:
            continue
        d = str(int(r['date']))
        rows.append(dict(k=f'{d[:4]}-{d[4:6]}-{d[6:]}|{h}|{a}|{int(num(r["home_final"]))}|{int(num(r["away_final"]))}' if num(r['home_final']) == num(r['home_final']) and num(r['away_final']) == num(r['away_final']) else None,
                         hcl=num(r['home_close_ml']), acl=num(r['away_close_ml']), hop=num(r['home_open_ml']), aop=num(r['away_open_ml']),
                         hsp=num(r['home_close_spread']), asp=num(r['away_close_spread']), hso=num(r['home_close_spread_odds']), aso=num(r['away_close_spread_odds'])))
    f = pd.DataFrame(rows).dropna(subset=['k']).drop_duplicates('k', keep=False).set_index('k')
    key = g.date + '|' + g.home + '|' + g.away + '|' + g.home_score.astype(str) + '|' + g.away_score.astype(str)
    n = ok = rl = 0
    for i in np.where((g.odds_source == 'SBRO') & key.isin(f.index))[0]:
        n += 1
        r = f.loc[key.iat[i]]
        if abs(r.hcl - g.home_ml.iat[i]) > 1 or abs(r.acl - g.away_ml.iat[i]) > 1:
            continue
        ok += 1
        idx = g.index[i]
        if abs(r.hop) >= 100 and abs(r.aop) >= 100:
            g.at[idx, 'home_ml_open'], g.at[idx, 'away_ml_open'] = r.hop, r.aop
        sp = g.spread_home.iat[i]
        if pd.notna(sp) and r.hsp == sp and r.asp == -sp and abs(r.hso) >= 100 and abs(r.aso) >= 100:
            g.at[idx, 'spread_home_odds'], g.at[idx, 'spread_away_odds'] = r.hso, r.aso
            rl += 1
    print(f'flancast NHL: key matches {n}, closing-ML agreement {ok}, puck-line odds taken {rl}')
    return g


def add_oddsapi(g):
    frames = []
    for p in sorted(glob.glob(os.path.join(RAW, 'oddsapi_noon_snapshots', 'odds_*.csv'))):
        d = pd.read_csv(p)
        d = d[d.sport.astype(str).str.upper() == 'NHL']
        if d.empty:
            continue
        d['snap'] = pd.Timestamp(os.path.basename(p)[5:15] + 'T17:00:00Z')
        frames.append(d)
    d = pd.concat(frames)
    d['commence'] = pd.to_datetime(d.commence_time, utc=True)
    d = d[d.commence > d.snap]  # pre-game prices only (drop in-play)
    d = d[d.bookmaker.isin(BOOKS)]
    d['home'] = d.home_team.map(FULL); d['away'] = d.away_team.map(FULL)
    assert d.home.notna().all() and d.away.notna().all(), d[d.home.isna() | d.away.isna()].home_team.unique()
    d['date'] = d.commence.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    d = d.sort_values('snap').groupby(['game_id', 'bookmaker']).tail(1)  # latest pre-game snapshot per event/book
    d['rank'] = d.bookmaker.map({b: i for i, b in enumerate(BOOKS)})
    d = d.dropna(subset=['ml_home', 'ml_away']).sort_values('rank').groupby(['date', 'home', 'away']).head(1)
    d = d.set_index(d.date + '|' + d.home + '|' + d.away)
    key = g.date + '|' + g.home + '|' + g.away
    filled = 0
    for i in np.where((g.odds_source != 'SBRO') & key.isin(d.index))[0]:
        r = d.loc[key.iat[i]]; idx = g.index[i]
        g.loc[idx, ['home_ml', 'away_ml', 'total', 'over_odds', 'under_odds', 'spread_home', 'spread_home_odds', 'spread_away_odds']] = [
            r.ml_home, r.ml_away, r.total_line, r.total_over_odds, r.total_under_odds, r.spread_home, r.spread_home_odds, r.spread_away_odds]
        g.at[idx, 'odds_source'] = 'TheOddsAPI noon-ET snapshot'
        g.at[idx, 'book'] = r.bookmaker
        filled += 1
    print(f'Odds API: {len(d)} games with prices, filled {filled}')
    return g


def main():
    g = add_oddsapi(enrich_flancast(base_games()))
    ip = lambda x: np.where(x < 0, -x / (-x + 100), 100 / (x + 100))
    for c in ['home_ml', 'away_ml', 'home_ml_open', 'away_ml_open', 'over_odds', 'under_odds', 'spread_home_odds', 'spread_away_odds']:
        g[c] = pd.to_numeric(g[c], errors='coerce')
        g.loc[g[c].abs() < 100, c] = np.nan
    for a, b in [('home_ml', 'away_ml'), ('home_ml_open', 'away_ml_open'), ('spread_home_odds', 'spread_away_odds'), ('over_odds', 'under_odds')]:
        v = ip(g[a]) + ip(g[b]); bad = (v < 1.0) | (v > 1.12)
        print(f'blanking {int(bad.sum())} rows with implausible {a}/{b} vig')
        g.loc[bad, [a, b]] = np.nan
    cols = ['date', 'season', 'home', 'away', 'home_score', 'away_score', 'home_ml', 'away_ml', 'home_ml_open', 'away_ml_open',
            'total', 'over_odds', 'under_odds', 'spread_home', 'spread_home_odds', 'spread_away_odds', 'playoff', 'went_ot',
            'decided_in', 'home_goalie', 'away_goalie', 'odds_source', 'book', 'game_id']
    g = g.sort_values(['date', 'game_id'])[cols]
    for c in ['home_ml', 'away_ml', 'home_ml_open', 'away_ml_open', 'over_odds', 'under_odds', 'spread_home_odds', 'spread_away_odds']:
        g[c] = g[c].round().astype('Int64')
    g.to_csv(OUT, index=False)
    print('wrote', OUT, len(g))


if __name__ == '__main__':
    main()
