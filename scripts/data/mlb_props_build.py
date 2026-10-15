"""Build MLB player-prop datasets from GitHub-hosted raw files.

Source (only MLB player-prop line archive found on GitHub, 2026-10-15):
  github.com/jordantaylorkurzweil-hash/MLB-Kprop-Pipeline  workspace/
    model_results_<date>_{am,late}.json - pitcher strikeout lines, DraftKings + FanDuel (via The Odds API),
        with game-status at snapshot time ('Scheduled'/'Pre-Game' = pre-game)
    results_<date>.json   - actual starter K and IP (repo pulls MLB Stats API)
    MLB_K_Prop_Model_*.xlsx / backtest_*_partial.csv - same data, kept for reference only
  Coverage: 2026-07-10 .. 2026-10-14 (starting pitchers, strikeouts market only).

Outputs: data/processed/mlb_props.csv, data/processed/mlb_pitcher_games.csv
"""
import glob, json, os, re
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'props_mlb', 'kprop_pipeline')
OUT = os.path.join(ROOT, 'data', 'processed')
PREGAME = {'Scheduled', 'Pre-Game', 'Warmup', 'Delayed Start', ''}


def ip_to_outs(ip):
    try:
        w, _, f = str(ip).partition('.')
        return int(w) * 3 + int(f or 0)
    except ValueError:
        return np.nan


def norm(s):
    return re.sub(r'[^a-z]', '', str(s).lower())


def load_results():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, 'results_*.json'))):
        d = re.search(r'results_(\d{4}-\d{2}-\d{2})', f).group(1)
        for r in json.load(open(f)):
            away, home = [t.strip() for t in r['matchup'].split('@')]
            rows.append({'date': d, 'pitcher': r['name'], 'team': r['team'],
                         'opp': home if r['team'] == away else away, 'home': int(r['team'] == home),
                         'strikeouts': r.get('actual_ks'), 'ip': r.get('ip'), 'outs': ip_to_outs(r.get('ip')),
                         'final_status': r.get('final_status'), 'started': 1})
    g = pd.DataFrame(rows)
    g = g[g.final_status.isin(['Final', 'Game Over', 'Completed Early'])]
    g = g.drop_duplicates(['date', 'pitcher', 'team'])
    g['season'] = g.date.str[:4].astype(int)
    return g


def load_lines():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, 'model_results_*.json'))):
        m = re.search(r'model_results_(\d{4}-\d{2}-\d{2})_(am|late)', f)
        d, snap = m.groups()
        for r in json.load(open(f)):
            for book, pre in (('draftkings', 'DK'), ('fanduel', 'FD')):
                line = r.get(f'{pre} Line')
                if line is None or (isinstance(line, float) and np.isnan(line)):
                    continue
                rows.append({'date': d, 'snapshot': snap, 'player': r['name'], 'team': r['team'],
                             'matchup': r['matchup'], 'game_time_utc': r.get('time'),
                             'status_at_snapshot': r.get('status') or '', 'market': 'pitcher_strikeouts',
                             'line': float(line), 'over_odds': r.get(f'{pre} Over'),
                             'under_odds': r.get(f'{pre} Under'), 'book': book})
    L = pd.DataFrame(rows)
    # keep only snapshots taken while game not yet started; prefer the later (closer-to-close) snapshot
    L = L[L.status_at_snapshot.isin(PREGAME)]
    L['snap_rank'] = (L.snapshot == 'late').astype(int)
    L = L.sort_values('snap_rank').drop_duplicates(['date', 'player', 'team', 'book'], keep='last')
    return L.drop(columns='snap_rank')


def main():
    os.makedirs(OUT, exist_ok=True)
    g = load_results()
    g.drop(columns='final_status').to_csv(os.path.join(OUT, 'mlb_pitcher_games.csv'), index=False)
    L = load_lines()
    g['k'] = g.date + '|' + g.pitcher.map(norm) + '|' + g.team
    L['k'] = L.date + '|' + L.player.map(norm) + '|' + L.team
    L = L.merge(g[['k', 'strikeouts', 'opp', 'home']], on='k', how='left').drop(columns='k')
    L = L.rename(columns={'strikeouts': 'actual'})
    L['result'] = np.select([L.actual > L.line, L.actual < L.line, L.actual == L.line],
                            ['over', 'under', 'push'], '')
    L['source_type'] = 'sportsbook'
    L['source'] = 'MLB-Kprop-Pipeline (The Odds API snapshot)'
    L['player_id'] = ''  # source carries no MLBAM ids
    L['season'] = L.date.str[:4].astype(int)
    cols = ['date', 'season', 'player', 'player_id', 'team', 'opp', 'home', 'market', 'line', 'over_odds',
            'under_odds', 'book', 'source_type', 'snapshot', 'status_at_snapshot', 'game_time_utc', 'actual',
            'result', 'source']
    L[cols].sort_values(['date', 'player', 'book']).to_csv(os.path.join(OUT, 'mlb_props.csv'), index=False)
    m = L.actual.notna()
    print(f'pitcher games {len(g)}; props {len(L)}; matched {m.mean():.1%}')
    print(L[m].groupby(['season', 'market', 'book']).result.value_counts(normalize=True).unstack().round(3))
    print(L.groupby(['season', 'market', 'book']).size())


if __name__ == '__main__':
    main()
