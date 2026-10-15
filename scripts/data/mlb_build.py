"""Build data/processed/mlb.csv (one row per game, 2010-2025) from downloaded raw files.

Sources (all in data/raw/mlb/):
  fdaproj_mlb_games_odds.csv  - github.com/ethanbell528-cmd/fda-project-1 data/mlb.csv
        Retrosheet/MLB scores (1990-2026) + SportsbookReviewsOnline (SBRO) closing ML,
        closing total, run line (2014+) for 2010-2021, matched on date+teams+score.
  fdaproj_mlb_starters.csv    - same repo, data/mlb_starters.csv (Retrosheet starting pitchers)
  sbro_mlb_archive_10Y.json   - github.com/flancast90/sportsbookreview-scraper (downloaded, REJECTED:
        rows misaligned vs. real results; kept only for reference)
  sbr_mlb_odds_2021_2025.json - github.com/ArnavSaraogi/mlb-odds-scraper release 'dataset'
        SportsBookReview per-book opening/closing ML, run line, total (2021-04-01..2025-08-16).
"""
import json, sys, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'mlb')
OUT = os.path.join(ROOT, 'data', 'processed', 'mlb.csv')
BOOK_PRIORITY = ['draftkings', 'fanduel', 'caesars', 'bet365', 'betmgm', 'bet_rivers_ny']

RETRO = {'ANA': 'LAA', 'CAL': 'LAA', 'CHA': 'CHW', 'CHN': 'CHC', 'KCA': 'KC', 'LAN': 'LAD', 'FLO': 'MIA',
         'NYA': 'NYY', 'NYN': 'NYM', 'OAK': 'ATH', 'SDN': 'SD', 'SFN': 'SF', 'SLN': 'STL', 'TBA': 'TB',
         'WAS': 'WSH', 'MON': 'WSH'}
SBR_CODE = {'AZ': 'ARI', 'OAK': 'ATH', 'WAS': 'WSH'}
FLAN = {'Angels': 'LAA', 'Astros': 'HOU', 'Athletics': 'ATH', 'Blue Jays': 'TOR', 'Braves': 'ATL',
        'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC', 'CUB': 'CHC', 'Diamondbacks': 'ARI',
        'Dodgers': 'LAD', 'Indians': 'CLE', 'KAN': 'KC', 'Mariners': 'SEA', 'Marlins': 'MIA', 'Mets': 'NYM',
        'Nationals': 'WSH', 'Orioles': 'BAL', 'Phillies': 'PHI', 'Pirates': 'PIT', 'Rangers': 'TEX',
        'Red Sox': 'BOS', 'Reds': 'CIN', 'Rockies': 'COL', 'SDG': 'SD', 'SFO': 'SF', 'TAM': 'TB',
        'Tigers': 'DET', 'Twins': 'MIN', 'White Sox': 'CHW', 'Yankees': 'NYY'}


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def base_games():
    m = pd.read_csv(os.path.join(RAW, 'fdaproj_mlb_games_odds.csv'), low_memory=False)
    m = m[(m.season >= 2010) & (m.season <= 2025)]
    m['retro_home'] = m.game_id.str[:3].map(lambda c: RETRO.get(c, c))
    is_home = (m.home_away == 'H') | ((m.home_away == 'N') & (m.team == m.retro_home))
    h = m[is_home].set_index('game_id')
    a = m[~is_home].set_index('game_id')
    assert h.index.is_unique and a.index.is_unique
    g = pd.DataFrame({
        'game_id': h.index, 'date': h.date.values, 'season': h.season.values,
        'home': h.team.values, 'away': a.loc[h.index, 'team'].values,
        'home_score': h.score_for.values, 'away_score': h.score_against.values,
        'home_ml': h.moneyline.values, 'away_ml': a.loc[h.index, 'moneyline'].values,
        'total': h.total_line.values, 'spread_home': h.line.values,
        'playoff': (h.game_type == 'playoff').astype(int).values,
        'neutral_site': (h.home_away == 'N').astype(int).values,
        'dh_game': h.index.str[-1].astype(int),
    })
    g['odds_source'] = np.where(g.home_ml.notna(), 'SBRO', '')
    g['book'] = np.where(g.home_ml.notna(), 'SBRO consensus', '')
    s = pd.read_csv(os.path.join(RAW, 'fdaproj_mlb_starters.csv'))
    s = s.drop_duplicates(['game_id', 'team']).set_index(['game_id', 'team']).starter_name
    g['home_sp'] = [s.get((i, t), '') for i, t in zip(g.game_id, g.home)]
    g['away_sp'] = [s.get((i, t), '') for i, t in zip(g.game_id, g.away)]
    for c in ['home_ml_open', 'away_ml_open', 'over_odds', 'under_odds', 'spread_home_odds', 'spread_away_odds']:
        g[c] = np.nan
    return g


def enrich_flancast(g):
    f = json.load(open(os.path.join(RAW, 'sbro_mlb_archive_10Y.json')))
    rows = []
    for r in f:
        h, a = FLAN.get(str(r['home_team'])), FLAN.get(str(r['away_team']))
        if not h or not a or h == a:
            continue
        d = str(int(r['date']))
        rows.append(dict(date=f'{d[:4]}-{d[4:6]}-{d[6:]}', home=h, away=a, hs=num(r['home_final']), as_=num(r['away_final']),
                         hcl=num(r['home_close_ml']), acl=num(r['away_close_ml']), hop=num(r['home_open_ml']),
                         aop=num(r['away_open_ml']), hsp=num(r['home_close_spread']), asp=num(r['away_close_spread']),
                         hso=num(r['home_close_spread_odds']), aso=num(r['away_close_spread_odds'])))
    f = pd.DataFrame(rows).dropna(subset=['hs', 'as_'])
    f = f.drop_duplicates(['date', 'home', 'away', 'hs', 'as_'], keep=False)
    key = g.date + '|' + g.home + '|' + g.away + '|' + g.home_score.astype(str) + '|' + g.away_score.astype(str)
    fk = f.date + '|' + f.home + '|' + f.away + '|' + f.hs.astype(int).astype(str) + '|' + f.as_.astype(int).astype(str)
    f = f.set_index(fk)
    n_match = n_ok = n_rl = 0
    for i in np.where(g.odds_source == 'SBRO')[0]:
        k = key.iat[i]
        if k not in f.index:
            continue
        n_match += 1
        r = f.loc[k]
        if abs(r.hcl - g.home_ml.iat[i]) > 1 or abs(r.acl - g.away_ml.iat[i]) > 1:
            continue  # disagreement -> do not trust this parsed row
        n_ok += 1
        if abs(r.hop) >= 100 and abs(r.aop) >= 100:
            g.at[g.index[i], 'home_ml_open'] = r.hop
            g.at[g.index[i], 'away_ml_open'] = r.aop
        sp = g.spread_home.iat[i]
        if pd.notna(sp) and r.hsp == sp and r.asp == -sp and abs(r.hso) >= 100 and abs(r.aso) >= 100:
            g.at[g.index[i], 'spread_home_odds'] = r.hso
            g.at[g.index[i], 'spread_away_odds'] = r.aso
            n_rl += 1
    print(f'flancast: matched {n_match}, closing ML agrees {n_ok}, RL odds taken {n_rl}')
    return g


def pick(lst, field_ok):
    by = {x['sportsbook']: x for x in (lst or [])}
    for b in BOOK_PRIORITY:
        x = by.get(b)
        if x and x.get('currentLine') and field_ok(x['currentLine']):
            return b, x
    return None, None


def add_sbr(g):
    d = json.load(open(os.path.join(RAW, 'sbr_mlb_odds_2021_2025.json')))
    rows = []
    for day, games in d.items():
        for x in games:
            gv = x['gameView']
            if gv.get('gameType') in ('S', 'A') or not str(gv.get('gameStatusText', '')).startswith('Final'):
                continue
            h = SBR_CODE.get(gv['homeTeam']['shortName'], gv['homeTeam']['shortName'])
            a = SBR_CODE.get(gv['awayTeam']['shortName'], gv['awayTeam']['shortName'])
            o = x.get('odds') or {}
            b, ml = pick(o.get('moneyline'), lambda c: num(c.get('homeOdds')) == num(c.get('homeOdds')) and abs(num(c.get('homeOdds'))) >= 100 and abs(num(c.get('awayOdds'))) >= 100)
            if not b:
                continue
            r = dict(date=day, start=gv['startDate'], home=h, away=a, hs=gv['homeTeamScore'], as_=gv['awayTeamScore'], book=b,
                     home_ml=num(ml['currentLine']['homeOdds']), away_ml=num(ml['currentLine']['awayOdds']))
            op = ml.get('openingLine') or {}
            r['home_ml_open'], r['away_ml_open'] = num(op.get('homeOdds')), num(op.get('awayOdds'))
            by = {y['sportsbook']: y for y in (o.get('totals') or [])}
            t = by.get(b) or (pick(o.get('totals'), lambda c: c.get('total') is not None)[1])
            if t and t.get('currentLine'):
                c = t['currentLine']; r['total'], r['over_odds'], r['under_odds'] = num(c.get('total')), num(c.get('overOdds')), num(c.get('underOdds'))
            by = {y['sportsbook']: y for y in (o.get('pointspread') or [])}
            s = by.get(b) or (pick(o.get('pointspread'), lambda c: c.get('homeSpread') is not None)[1])
            if s and s.get('currentLine'):
                c = s['currentLine']; r['spread_home'], r['spread_home_odds'], r['spread_away_odds'] = num(c.get('homeSpread')), num(c.get('homeOdds')), num(c.get('awayOdds'))
            rows.append(r)
    s = pd.DataFrame(rows).sort_values('start')
    # SBR dates are US-local; startDate is UTC. Match on SBR date key + teams + score.
    s['k'] = s.date + '|' + s.home + '|' + s.away + '|' + s.hs.astype(str) + '|' + s.as_.astype(str)
    s = s.drop_duplicates('k', keep=False)
    gk = g.date + '|' + g.home + '|' + g.away + '|' + g.home_score.astype(str) + '|' + g.away_score.astype(str)
    s = s.set_index('k')
    filled = 0
    cols = ['home_ml', 'away_ml', 'home_ml_open', 'away_ml_open', 'total', 'over_odds', 'under_odds', 'spread_home', 'spread_home_odds', 'spread_away_odds']
    for i in np.where((g.odds_source != 'SBRO') & gk.isin(s.index))[0]:
        r = s.loc[gk.iat[i]]
        for c in cols:
            g.at[g.index[i], c] = r.get(c, np.nan)
        g.at[g.index[i], 'odds_source'] = 'SBR(sportsbookreview.com)'
        g.at[g.index[i], 'book'] = r.book
        filled += 1
    print(f'SBR 2021-25: {len(s)} usable games, filled {filled}')
    return g


def main():
    g = base_games()
    # enrich_flancast(g) intentionally NOT used: spot checks showed that file's team/score
    # assignments are misaligned (e.g. 2015-04-05 lists Rockies 10 CUB 0; actual STL 3 @ CHC 0).
    g = add_sbr(g)
    # sanity: drop odds that are not valid American prices
    for c in ['home_ml', 'away_ml', 'home_ml_open', 'away_ml_open', 'over_odds', 'under_odds', 'spread_home_odds', 'spread_away_odds']:
        g.loc[g[c].abs() < 100, c] = np.nan
    # blank closing MLs whose two-way implied probability sum is impossible (<1.0) or absurd (>1.12);
    # these are source data-entry errors (e.g. SBRO 2020-09-18 has both sides negative / positive)
    ip = lambda x: np.where(x < 0, -x / (-x + 100), 100 / (x + 100))
    for a, b in [('home_ml', 'away_ml'), ('home_ml_open', 'away_ml_open'), ('spread_home_odds', 'spread_away_odds'), ('over_odds', 'under_odds')]:
        v = ip(g[a]) + ip(g[b])
        bad = (v < 1.0) | (v > 1.12)
        print(f'blanking {int(bad.sum())} rows with implausible {a}/{b} vig')
        g.loc[bad, [a, b]] = np.nan
    cols = ['date', 'season', 'home', 'away', 'home_score', 'away_score', 'home_ml', 'away_ml', 'home_ml_open', 'away_ml_open',
            'total', 'over_odds', 'under_odds', 'spread_home', 'spread_home_odds', 'spread_away_odds', 'playoff',
            'home_sp', 'away_sp', 'dh_game', 'neutral_site', 'odds_source', 'book', 'game_id']
    g = g.sort_values(['date', 'game_id'])[cols]
    for c in ['home_ml', 'away_ml', 'home_ml_open', 'away_ml_open', 'over_odds', 'under_odds', 'spread_home_odds', 'spread_away_odds']:
        g[c] = g[c].round().astype('Int64')
    g.to_csv(OUT, index=False)
    print('wrote', OUT, len(g))


if __name__ == '__main__':
    main()
