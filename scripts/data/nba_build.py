"""Build data/processed/nba.csv (one row per NBA game, 2007-08 .. latest) with closing/opening lines.

Sources (all in data/raw/nba/):
  espn_schedules/nba_schedule_YYYY.parquet  sportsdataverse-data release `espn_nba_schedules` (game list, scores, season type)
  espn_pickcenter/nba_YYYY.jsonl            header + pickcenter extracted from sportsdataverse/hoopR-nba-raw ESPN game JSON
                                            (scripts/data/fetch_espn_pickcenter.py)
  kyleskom_OddsData.sqlite                  kyleskom/NBA-Machine-Learning-Sports-Betting Data/OddsData.sqlite
                                            (SBR lines, 2007-08 .. Jan 2026; see NOTE)
  nba_archive_10Y.json                      flancast90/sportsbookreview-scraper data/ (SBR archive 2011-12 .. 2021-22,
                                            opening + closing spread/total, closing ML)
NOTE: kyleskom tables before 2022-23 hold SBR OPENING spread/total (unsigned spread; sign taken from ML favourite)
and closing ML -- verified against flancast90 open/close columns. They are used only as opening lines.
Closing spread/total priority: flancast90 close > kyleskom (2022-23+) > ESPN pickcenter sportsbook > teamrankings >
consensus. ML: kyleskom > flancast90 > ESPN (sportsbook/consensus; numberfire projections ignored).
Opening: flancast90 > kyleskom(pre-2022-23) > ESPN pickcenter open fields.
"""
import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'nba')
OUT = os.path.join(ROOT, 'data', 'processed', 'nba.csv')

# ---------- canonical names (current franchise name) ----------
CANON_FIX = {
    'Charlotte Bobcats': 'Charlotte Hornets', 'New Jersey Nets': 'Brooklyn Nets',
    'Seattle SuperSonics': 'Oklahoma City Thunder', 'New Orleans Hornets': 'New Orleans Pelicans',
    'New Orleans/Oklahoma City Hornets': 'New Orleans Pelicans', 'Los Angeles Clippers': 'LA Clippers',
}
FLAN = {
    'Hawks': 'Atlanta Hawks', 'Celtics': 'Boston Celtics', 'Nets': 'Brooklyn Nets', 'NewJersey': 'Brooklyn Nets',
    'Hornets': 'Charlotte Hornets', 'Bulls': 'Chicago Bulls', 'Cavaliers': 'Cleveland Cavaliers',
    'Mavericks': 'Dallas Mavericks', 'Nuggets': 'Denver Nuggets', 'Pistons': 'Detroit Pistons',
    'Warriors': 'Golden State Warriors', 'Golden State': 'Golden State Warriors', 'Rockets': 'Houston Rockets',
    'Pacers': 'Indiana Pacers', 'Clippers': 'LA Clippers', 'LA Clippers': 'LA Clippers', 'Lakers': 'Los Angeles Lakers',
    'Grizzlies': 'Memphis Grizzlies', 'Heat': 'Miami Heat', 'Bucks': 'Milwaukee Bucks',
    'Timberwolves': 'Minnesota Timberwolves', 'Pelicans': 'New Orleans Pelicans', 'Knicks': 'New York Knicks',
    'Thunder': 'Oklahoma City Thunder', 'Oklahoma City': 'Oklahoma City Thunder', 'Magic': 'Orlando Magic',
    'Seventysixers': 'Philadelphia 76ers', 'Suns': 'Phoenix Suns', 'Trailblazers': 'Portland Trail Blazers',
    'Kings': 'Sacramento Kings', 'Spurs': 'San Antonio Spurs', 'Raptors': 'Toronto Raptors', 'Jazz': 'Utah Jazz',
    'Wizards': 'Washington Wizards',
}


def canon(n):
    return CANON_FIX.get(n, n)


def num(x):
    try:
        if x is None or x == '':
            return np.nan
        return float(str(x).replace('+', '').lstrip('ou'))
    except ValueError:
        return np.nan


# ---------- ESPN base game list ----------
def load_espn():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, 'espn_pickcenter', 'nba_*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if 'error' in r or r.get('status') not in (None, 'STATUS_FINAL'):
                continue
            c = {x['homeAway']: x for x in r['competitors']}
            if 'home' not in c or 'away' not in c:
                continue
            st = r.get('season_type')
            row = dict(game_id=r['game_id'], season=r['season'], utc=r['date'], neutral=r.get('neutral'),
                       season_type=st, home_id=c['home']['id'], away_id=c['away']['id'],
                       home_name=c['home']['displayName'], away_name=c['away']['displayName'],
                       home_score=num(c['home']['score']), away_score=num(c['away']['score']))
            # pickcenter: sportsbook > teamrankings > consensus for spread/total; ML from sportsbook/consensus only
            cands = []
            for p in r['pickcenter']:
                name = p.get('provider', {}).get('name', '')
                if not name or name.lower() == 'numberfire':  # numberfire = model projections
                    continue
                cands.append((3 if name == 'consensus' else 2 if name == 'teamrankings' else 1, name, p))
            cands.sort(key=lambda t: t[0])
            for key, field in (('espn_spread', 'spread'), ('espn_total', 'overUnder')):
                for _, name, p in cands:
                    v = num(p.get(field))
                    if not np.isnan(v):
                        row[key] = v
                        if key == 'espn_spread':
                            row['espn_provider'] = name
                        break
            for _, name, p in cands:
                h, a = num(p.get('homeTeamOdds', {}).get('moneyLine')), num(p.get('awayTeamOdds', {}).get('moneyLine'))
                if name != 'teamrankings' and not (np.isnan(h) or np.isnan(a)):
                    row['espn_home_ml'], row['espn_away_ml'] = h, a
                    break
            for _, name, p in cands:
                ps, tt = p.get('pointSpread') or {}, p.get('total') or {}
                so = num(ps.get('home', {}).get('open', {}).get('line'))
                to = num(tt.get('over', {}).get('open', {}).get('line'))
                if not (np.isnan(so) and np.isnan(to)):
                    row['espn_spread_open'], row['espn_total_open'] = so, to
                    break
            rows.append(row)
    d = pd.DataFrame(rows)
    for c in ('espn_spread', 'espn_total', 'espn_home_ml', 'espn_away_ml', 'espn_spread_open', 'espn_total_open',
              'espn_provider'):
        if c not in d:
            d[c] = np.nan
    d['date'] = (pd.to_datetime(d.utc, utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d'))
    # canonical names via ESPN team id -> most recent display name
    latest = pd.concat([d[['season', 'home_id', 'home_name']].rename(columns={'home_id': 'id', 'home_name': 'n'}),
                        d[['season', 'away_id', 'away_name']].rename(columns={'away_id': 'id', 'away_name': 'n'})])
    latest = latest.sort_values('season').groupby('id').n.last().map(canon).to_dict()
    d['home'] = d.home_id.map(latest)
    d['away'] = d.away_id.map(latest)
    # drop All-Star / exhibition games (non-NBA franchises)
    valid = set(FLAN.values())
    d = d[d.home.isin(valid) & d.away.isin(valid) & d.season_type.isin([2, 3, 5])]
    d = d.drop_duplicates('game_id')
    return d


# ---------- kyleskom (SBR) ----------
def load_kyleskom():
    c = sqlite3.connect(os.path.join(RAW, 'kyleskom_OddsData.sqlite'))
    tabs = pd.read_sql("select name from sqlite_master where type='table'", c).name.tolist()
    use = [t for t in tabs if t.endswith('_new')] + ['2023-24', '2024-25', 'odds_2025-26']
    use = [t for t in use if t != 'odds_2023-24_new']
    parts = []
    for t in use:
        x = pd.read_sql(f'select * from "{t}"', c)
        x['src_table'] = t
        sp = pd.to_numeric(x.Spread.replace({'PK': 0, 'pk': 0}), errors='coerce')
        x['unsigned'] = bool((sp < 0).sum() == 0)  # pre-2022-23 tables store |spread| only
        parts.append(x)
    k = pd.concat(parts, ignore_index=True)
    k['Spread'] = k.Spread.replace({'PK': 0, 'pk': 0})
    for col in ['OU', 'Spread', 'ML_Home', 'ML_Away', 'Points', 'Win_Margin']:
        k[col] = pd.to_numeric(k[col].astype(str).str.strip().str.replace('+', '', regex=False), errors='coerce')
    k['date'] = pd.to_datetime(k.Date).dt.strftime('%Y-%m-%d')
    k['home'] = k.Home.map(canon)
    k['away'] = k.Away.map(canon)
    k['k_home_score'] = (k.Points + k.Win_Margin) / 2
    k['k_away_score'] = (k.Points - k.Win_Margin) / 2
    k['k_spread'] = -k.Spread  # kyleskom Spread = points home is favored by (signed tables)
    # unsigned tables: sign from the moneyline favourite; ambiguous (equal/missing ML) -> blank unless PK
    u = k.unsigned
    home_fav = k.ML_Home < k.ML_Away
    away_fav = k.ML_Home > k.ML_Away
    k.loc[u, 'k_spread'] = np.where(home_fav[u], -k.Spread[u].abs(), np.where(away_fav[u], k.Spread[u].abs(), np.nan))
    k.loc[u & (k.Spread == 0), 'k_spread'] = 0.0
    k = k.rename(columns={'OU': 'k_total', 'ML_Home': 'k_home_ml', 'ML_Away': 'k_away_ml'})
    k['k_is_open'] = k.unsigned.astype(int)
    return k[['date', 'home', 'away', 'k_home_score', 'k_away_score', 'k_spread', 'k_total', 'k_home_ml',
              'k_away_ml', 'k_is_open']].drop_duplicates(['date', 'home', 'away'])


# ---------- flancast90 (SBR) ----------
def load_flancast():
    f = pd.read_json(os.path.join(RAW, 'nba_archive_10Y.json'))
    f['home'] = f.home_team.astype(str).map(FLAN)
    f['away'] = f.away_team.astype(str).map(FLAN)
    f = f.dropna(subset=['home', 'away'])
    f['date'] = pd.to_datetime(f.date.astype(int).astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
    f = f.rename(columns={'home_final': 'f_home_score', 'away_final': 'f_away_score',
                          'home_close_spread': 'f_spread', 'home_open_spread': 'f_spread_open',
                          'close_over_under': 'f_total', 'open_over_under': 'f_total_open',
                          'home_close_ml': 'f_home_ml', 'away_close_ml': 'f_away_ml'})
    for col in ['f_home_score', 'f_away_score', 'f_spread', 'f_spread_open', 'f_total', 'f_total_open',
                'f_home_ml', 'f_away_ml']:
        f[col] = pd.to_numeric(f[col], errors='coerce')
    return f[['date', 'home', 'away', 'f_home_score', 'f_away_score', 'f_spread', 'f_spread_open', 'f_total',
              'f_total_open', 'f_home_ml', 'f_away_ml']].drop_duplicates(['date', 'home', 'away'])


def attach(base, src, prefix, score_cols, flip_cols, ml_pair):
    """Match src rows to base games by (date, then +/-1 day, unordered team pair); orient to base home team."""
    src = src.copy().reset_index(drop=True)
    src['_sid'] = src.index
    src['_d'] = pd.to_datetime(src.date)
    src['key'] = [tuple(sorted(p)) for p in zip(src.home, src.away)]
    b = base[['game_id', 'date', 'home', 'away']].copy()
    b['_d'] = pd.to_datetime(b.date)
    b['key'] = [tuple(sorted(p)) for p in zip(b.home, b.away)]
    matched = []
    used_g, used_s = set(), set()
    for off in (0, -1, 1):
        s = src[~src._sid.isin(used_s)].copy()
        s['_d'] = s._d + pd.Timedelta(days=off)
        m = b[~b.game_id.isin(used_g)].merge(s.drop(columns=['date']), on=['_d', 'key'], suffixes=('', '_s'))
        m = m.drop_duplicates('game_id').drop_duplicates('_sid')
        used_g |= set(m.game_id)
        used_s |= set(m._sid)
        matched.append(m)
    out = pd.concat(matched)
    swapped = out.home != out.home_s
    for c in flip_cols:
        out.loc[swapped, c] = -out.loc[swapped, c]
    a, bb = ml_pair
    out.loc[swapped, [a, bb]] = out.loc[swapped, [bb, a]].values
    sa, sb = score_cols
    out.loc[swapped, [sa, sb]] = out.loc[swapped, [sb, sa]].values
    out[prefix + 'swapped'] = swapped.astype(int)
    keep = [c for c in src.columns if c.startswith(prefix)]
    print(prefix, 'source rows', len(src), 'matched', len(out), 'home/away swapped', int(swapped.sum()))
    return base.merge(out[['game_id'] + keep], on='game_id', how='left')


def main():
    g = load_espn()
    k = load_kyleskom()
    f = load_flancast()
    g = attach(g, k, 'k_', ('k_home_score', 'k_away_score'), ['k_spread'], ('k_home_ml', 'k_away_ml'))
    g = attach(g, f, 'f_', ('f_home_score', 'f_away_score'), ['f_spread', 'f_spread_open'],
               ('f_home_ml', 'f_away_ml'))

    # reject source rows whose scores disagree with ESPN (bad match)
    for p in ('k_', 'f_'):
        bad = ((g[p + 'home_score'] - g.home_score).abs() > 1) | ((g[p + 'away_score'] - g.away_score).abs() > 1)
        cols = [c for c in g.columns if c.startswith(p)]
        g.loc[bad, cols] = np.nan
        g[p + 'score_mismatch'] = bad.astype(int)

    # kyleskom pre-2022-23 tables = SBR *opening* spread/total (verified: identical to flancast90 open columns,
    # higher MAE vs. result than the close) + closing ML.  2022-23+ tables (sbrscrape) are treated as closing.
    isopen = g.k_is_open == 1
    g['k_spread_open'] = g.k_spread.where(isopen)
    g['k_total_open'] = g.k_total.where(isopen)
    g['k_spread_close'] = g.k_spread.where(~isopen)
    g['k_total_close'] = g.k_total.where(~isopen)

    # per-source sanity before picking (so a corrupt value falls through to the next source)
    for c in ('k_spread_open', 'k_spread_close', 'f_spread', 'f_spread_open', 'espn_spread', 'espn_spread_open'):
        g.loc[g[c].abs() > 25, c] = np.nan
    for c in ('k_total_open', 'k_total_close', 'f_total', 'f_total_open', 'espn_total', 'espn_total_open'):
        g.loc[~g[c].between(150, 280), c] = np.nan
    for a, b in (('k_home_ml', 'k_away_ml'), ('f_home_ml', 'f_away_ml'), ('espn_home_ml', 'espn_away_ml')):
        bad = (g[a].abs() < 100) | (g[b].abs() < 100) | ((g[a] > 0) & (g[b] > 0) & ((g[a] > 105) | (g[b] > 105))) \
            | ((g[a] < 0) & (g[b] < 0) & ((g[a] < -125) | (g[b] < -125)))
        g.loc[bad, [a, b]] = np.nan
    # flancast90 close spread must agree in sign with its own closing ML favourite when the gap is large
    fav_conflict = ((g.f_spread < -1.5) & (g.f_home_ml > 100)) | ((g.f_spread > 1.5) & (g.f_away_ml > 100))
    g.loc[fav_conflict, 'f_spread'] = np.nan

    def pick(*cols):
        r = pd.Series(np.nan, index=g.index)
        src = pd.Series('', index=g.index)
        for c in cols:
            take = r.isna() & g[c].notna()
            r[take] = g.loc[take, c]
            src[take] = c
        return r, src

    SRC = {'f_spread': 'sbr_flancast90', 'k_spread_close': 'sbr_kyleskom', 'espn_spread': 'espn',
           'f_total': 'sbr_flancast90', 'k_total_close': 'sbr_kyleskom', 'espn_total': 'espn'}
    g['spread_home'], s1 = pick('f_spread', 'k_spread_close', 'espn_spread')
    g['total'], s2 = pick('f_total', 'k_total_close', 'espn_total')
    g['line_source'] = s1.map(SRC).fillna('')
    g.loc[g.line_source == 'espn', 'line_source'] = 'espn_' + g.espn_provider.fillna('')
    g['total_source'] = s2.map(SRC).fillna('')
    g.loc[g.total_source == 'espn', 'total_source'] = 'espn_' + g.espn_provider.fillna('')
    # moneylines must be taken as a pair from one source
    g['home_ml'] = np.nan
    g['away_ml'] = np.nan
    g['ml_source'] = ''
    for src, a, b in (('sbr_kyleskom', 'k_home_ml', 'k_away_ml'), ('sbr_flancast90', 'f_home_ml', 'f_away_ml'),
                      ('espn', 'espn_home_ml', 'espn_away_ml')):
        ok = g.home_ml.isna() & g[a].notna() & g[b].notna()
        g.loc[ok, 'home_ml'] = g.loc[ok, a]
        g.loc[ok, 'away_ml'] = g.loc[ok, b]
        g.loc[ok, 'ml_source'] = src
    g['spread_open_home'], _ = pick('f_spread_open', 'k_spread_open', 'espn_spread_open')
    g['total_open'], _ = pick('f_total_open', 'k_total_open', 'espn_total_open')
    g['playoff'] = (g.season_type == 3).astype(int)
    g['play_in'] = (g.season_type == 5).astype(int)
    g['neutral'] = g.neutral.map({True: 1, False: 0})
    g['home_score'] = g.home_score.astype(int)
    g['away_score'] = g.away_score.astype(int)
    g['espn_game_id'] = g.game_id

    cols = ['date', 'season', 'home', 'away', 'home_score', 'away_score', 'neutral', 'spread_home', 'total',
            'home_ml', 'away_ml', 'spread_open_home', 'total_open', 'playoff', 'play_in', 'line_source',
            'total_source', 'ml_source', 'espn_game_id']
    out = g[cols].sort_values(['date', 'home']).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print('wrote', OUT, len(out))


if __name__ == '__main__':
    main()
