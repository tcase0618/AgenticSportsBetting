"""Build data/processed/ncaam.csv (one row per NCAA men's basketball game) with betting lines.

Source (data/raw/ncaam/):
  espn_schedules/mbb_schedule_YYYY.parquet  sportsdataverse-data release `espn_mens_college_basketball_schedules`
  espn_pickcenter/mbb_YYYY.jsonl            header + pickcenter extracted from ESPN game-summary JSON mirrored in
                                            github.com/sportsdataverse/hoopR-mbb-raw (scripts/data/fetch_espn_pickcenter.py)
ESPN pickcenter carries lines from: a real sportsbook (Caesars NJ 2021-22, DraftKings 2025-26 ...), 'consensus'
and 'teamrankings'. 'numberfire' entries are model projections and are ignored.
Spread/total priority: sportsbook > teamrankings > consensus (teamrankings tracks the close better than ESPN
'consensus' in 2013-2018: lower MAE vs final margin). Moneyline: sportsbook > consensus only; teamrankings
moneylines have a constant ~2.9% hold (look formula-derived from the spread) so they go to tr_home_ml/tr_away_ml.
Seasons with no ESPN line data (2008-2012, 2024, 2025) have blank lines.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'ncaam')
OUT = os.path.join(ROOT, 'data', 'processed', 'ncaam.csv')


def num(x):
    try:
        if x is None or x == '':
            return np.nan
        return float(str(x).replace('+', '').lstrip('ou'))
    except ValueError:
        return np.nan


def provider_rank(name):
    n = (name or '').lower()
    if n == 'numberfire' or not n:
        return None
    if n == 'teamrankings':
        return 2
    if n == 'consensus':
        return 3
    return 1  # an actual sportsbook


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, 'espn_pickcenter', 'mbb_*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if 'error' in r or r.get('status') not in (None, 'STATUS_FINAL'):
                continue
            c = {x['homeAway']: x for x in r['competitors']}
            if 'home' not in c or 'away' not in c:
                continue
            row = dict(game_id=r['game_id'], season=r['season'], utc=r['date'], neutral=r.get('neutral'),
                       season_type=r.get('season_type'), notes='; '.join([n for n in r.get('notes') or [] if n]),
                       home_id=c['home']['id'], away_id=c['away']['id'],
                       home_loc=c['home'].get('location') or c['home'].get('displayName'),
                       away_loc=c['away'].get('location') or c['away'].get('displayName'),
                       home_score=num(c['home']['score']), away_score=num(c['away']['score']))
            cands = []
            for p in r['pickcenter']:
                name = p.get('provider', {}).get('name', '')
                rk = provider_rank(name)
                if rk is None:
                    continue
                cands.append((rk, name, p))
            cands.sort(key=lambda t: t[0])
            # spread/total: first provider that has the value; moneyline pair likewise
            for key, field in (('spread_home', 'spread'), ('total', 'overUnder')):
                for rk, name, p in cands:
                    v = num(p.get(field))
                    if not np.isnan(v):
                        row[key] = v
                        row[key + '_src'] = name
                        break
            for rk, name, p in cands:
                h, a = num(p.get('homeTeamOdds', {}).get('moneyLine')), num(p.get('awayTeamOdds', {}).get('moneyLine'))
                if np.isnan(h) or np.isnan(a):
                    continue
                if name == 'teamrankings':
                    row['tr_home_ml'], row['tr_away_ml'] = h, a
                elif 'home_ml' not in row:
                    row['home_ml'], row['away_ml'], row['ml_src'] = h, a, name
            for rk, name, p in cands:
                ps, tt = p.get('pointSpread') or {}, p.get('total') or {}
                so = num(ps.get('home', {}).get('open', {}).get('line'))
                to = num(tt.get('over', {}).get('open', {}).get('line'))
                if not (np.isnan(so) and np.isnan(to)):
                    row['spread_open_home'], row['total_open'] = so, to
                    break
            rows.append(row)
    g = pd.DataFrame(rows).drop_duplicates('game_id')
    g = g.dropna(subset=['home_score', 'away_score'])
    g = g[(g.home_score + g.away_score) > 0]
    g['date'] = pd.to_datetime(g.utc, utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')

    # canonical team name = most recent ESPN 'location' (school name) for each ESPN team id
    nm = pd.concat([g[['season', 'home_id', 'home_loc']].set_axis(['season', 'id', 'n'], axis=1),
                    g[['season', 'away_id', 'away_loc']].set_axis(['season', 'id', 'n'], axis=1)])
    latest = nm.sort_values('season').groupby('id').n.last()
    # disambiguate any duplicate names across different ids
    dup = latest[latest.duplicated(keep=False)]
    latest.loc[dup.index] = [f'{n} ({i})' for i, n in dup.items()]
    g['home'] = g.home_id.map(latest)
    g['away'] = g.away_id.map(latest)

    # sanity filters on lines
    for c in ('spread_home', 'spread_open_home'):
        if c in g:
            g.loc[g[c].abs() > 60, c] = np.nan
    for c in ('total', 'total_open'):
        if c in g:
            g.loc[(g[c] < 90) | (g[c] > 200), c] = np.nan
    for a, b in (('home_ml', 'away_ml'), ('tr_home_ml', 'tr_away_ml')):
        if a not in g:
            g[a], g[b] = np.nan, np.nan
        bad_ml = (g[a].abs() < 100) | (g[b].abs() < 100) | ((g[a] > 0) & (g[b] > 0) & ((g[a] > 105) | (g[b] > 105)))
        ip = lambda m: np.where(m < 0, -m / (100 - m), 100 / (m + 100))
        over = ip(g[a]) + ip(g[b])
        bad_ml = bad_ml | (g[a].notna() & ((over < 0.995) | (over > 1.2)))  # incoherent composite pairs
        g.loc[bad_ml, [a, b]] = np.nan

    # ESPN occasionally lists the same game under two ids: keep the one with the most line data
    g['_info'] = g[['spread_home', 'total', 'home_ml']].notna().sum(axis=1) if 'spread_home' in g else 0
    g = g.sort_values('_info', ascending=False).drop_duplicates(['date', 'home', 'away', 'home_score', 'away_score'])
    g['neutral'] = g.neutral.map({True: 1, False: 0})
    g['playoff'] = (g.season_type == 3).astype(int)  # NCAA tournament / NIT / other postseason
    g['home_score'] = g.home_score.astype(int)
    g['away_score'] = g.away_score.astype(int)
    g['espn_game_id'] = g.game_id
    g['espn_home_id'] = g.home_id
    g['espn_away_id'] = g.away_id
    cols = ['date', 'season', 'home', 'away', 'home_score', 'away_score', 'neutral', 'spread_home', 'total',
            'home_ml', 'away_ml', 'spread_open_home', 'total_open', 'playoff', 'notes', 'spread_home_src',
            'total_src', 'ml_src', 'tr_home_ml', 'tr_away_ml', 'espn_game_id', 'espn_home_id', 'espn_away_id']
    for c in cols:
        if c not in g:
            g[c] = np.nan
    out = g[cols].sort_values(['date', 'home']).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print('wrote', OUT, len(out))


if __name__ == '__main__':
    main()
