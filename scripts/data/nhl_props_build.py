"""Build NHL player-prop datasets from GitHub-hosted raw files.

Sources:
  github.com/flagk/nhl-tracker  data/logs/alt_quotes/<run>.csv -> data/raw/props_nhl/flagk/alt_quotes/
      Append-only snapshots of player prop quotes (The Odds API; best decimal price across n_books,
      book name not recorded) for player_sog / player_points / player_assists / player_goals.
      Coverage: 2026-27 season, 2026-09-30 .. 2026-10-14.
  github.com/jjurgill14/nhl-props  data/boxscores/<game_id>.csv, data/games/<game_id>.json
      -> data/raw/props_nhl/jjurgill_box/  (per-player NHL API boxscores, 2026-27 season)

Outputs: data/processed/nhl_props.csv, nhl_skater_games.csv, nhl_goalie_games.csv
"""
import glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'props_nhl')
OUT = os.path.join(ROOT, 'data', 'processed')
STAT = {'player_sog': 'sog', 'player_points': 'points', 'player_assists': 'assists', 'player_goals': 'goals'}


def dec_to_amer(d):
    if pd.isna(d) or d <= 1:
        return np.nan
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def load_box():
    box = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, 'jjurgill_box', '*.csv'))])
    games = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(RAW, 'jjurgill_box', '*.json'))])
    games = games[games.state.isin(['OFF', 'FINAL'])]
    box = box[box.game_id.isin(games.game_id)]
    box['season'] = '2026-27'
    goal = box.pos.eq('G')
    sk = box[~goal][['game_id', 'season', 'date', 'player_id', 'full_name', 'team', 'opp', 'home', 'pos', 'goals',
                     'assists', 'points', 'sog', 'pp_goals', 'hits', 'blocked', 'pim', 'plus_minus', 'toi_s',
                     'pp_toi_s', 'shifts']]
    gl = box[goal][['game_id', 'season', 'date', 'player_id', 'full_name', 'team', 'opp', 'home', 'starter',
                    'decision', 'shots_against', 'saves', 'goals_against', 'save_pct', 'toi_s']]
    return sk, gl, games


def main():
    os.makedirs(OUT, exist_ok=True)
    sk, gl, games = load_box()
    sk.to_csv(os.path.join(OUT, 'nhl_skater_games.csv'), index=False)
    gl.to_csv(os.path.join(OUT, 'nhl_goalie_games.csv'), index=False)

    q = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, 'flagk', 'alt_quotes', '*.csv'))])
    q = q[q.market.isin(STAT)].copy()
    q['dir'] = q.side.str.split(':').str[0]
    q['player_id'] = q.player_id.astype('Int64')
    q['run_at'] = pd.to_datetime(q.run_at, utc=True)
    st = games.set_index('game_id').start_utc
    q['start_utc'] = pd.to_datetime(q.game_id.map(st), utc=True)
    # pre-game only: snapshot strictly before scheduled puck drop (unknown start -> dropped)
    q = q[q.start_utc.notna() & (q.run_at < q.start_utc)]
    q = q.sort_values('run_at').drop_duplicates(['game_id', 'market', 'player_id', 'point', 'dir'], keep='last')
    q['name'] = q.label.str.replace(r' (Over|Under) [\d.]+$', '', regex=True)
    key = ['game_id', 'game_date', 'market', 'player_id', 'point']
    p = q.pivot_table(index=key, columns='dir', values=['decimal', 'n_books', 'run_at'], aggfunc='first')
    p.columns = [f'{a}_{b}' for a, b in p.columns]
    p = p.reset_index().merge(q.drop_duplicates(key)[key + ['name']], on=key)
    p['over_odds'] = p.get('decimal_over').map(dec_to_amer)
    p['under_odds'] = p.get('decimal_under').map(dec_to_amer)
    p['snapshot_utc'] = p[['run_at_over', 'run_at_under']].max(axis=1)
    p['n_books'] = p[['n_books_over', 'n_books_under']].max(axis=1)

    s = sk.set_index(['game_id', 'player_id'])
    p = p.merge(sk[['game_id', 'player_id', 'full_name', 'team', 'opp', 'home'] + list(set(STAT.values()))],
                on=['game_id', 'player_id'], how='left')
    p['actual'] = [r[STAT[m]] if pd.notna(r['team']) else np.nan for m, (_, r) in zip(p.market, p.iterrows())]
    p['result'] = np.select([p.actual > p.point, p.actual < p.point, p.actual == p.point], ['over', 'under', 'push'], '')
    p['result'] = np.where(p.actual.isna(), '', p.result)
    p = p.rename(columns={'game_date': 'date', 'point': 'line'})
    p['player'] = p.full_name.fillna(p.name)
    p['market'] = p.market.map({'player_sog': 'shots_on_goal', 'player_points': 'points',
                                'player_assists': 'assists', 'player_goals': 'goals'})
    p['book'] = 'best_price_across_books'
    p['source_type'] = 'sportsbook'
    p['season'] = '2026-27'
    p['source'] = 'flagk/nhl-tracker alt_quotes (The Odds API)'
    cols = ['date', 'season', 'game_id', 'player', 'player_id', 'team', 'opp', 'home', 'market', 'line', 'over_odds',
            'under_odds', 'book', 'n_books', 'source_type', 'snapshot_utc', 'actual', 'result', 'source']
    p = p[cols].sort_values(['date', 'game_id', 'player', 'market', 'line'])
    p.to_csv(os.path.join(OUT, 'nhl_props.csv'), index=False)
    m = p.actual.notna()
    print(f'skater rows {len(sk)}, goalie rows {len(gl)}, games {sk.game_id.nunique()}')
    print(f'props {len(p)}; matched {m.mean():.1%}; both prices {(p.over_odds.notna() & p.under_odds.notna()).mean():.1%}')
    print(p[m].groupby(['market', 'line']).result.value_counts(normalize=True).unstack().round(3))
    print(p.groupby(['market']).size())


if __name__ == '__main__':
    main()
