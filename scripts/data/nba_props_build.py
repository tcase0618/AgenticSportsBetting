"""Build NBA player box scores + historical player prop lines joined to results.

Outputs
  data/processed/nba_player_box.csv  one row per player-game (ESPN, 2001-02 .. 2025-26, all season types)
  data/processed/nba_props.csv       one row per (game, player, market, book) prop line with actual/result

Raw sources (data/raw/props_nba/):
  player_box/player_box_YYYY.parquet
      github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_YYYY.parquet
  kamran/partner_odds_backtest.csv + kamran/events/<event_id>.json
      raw.githubusercontent.com/KamranSHussain/NBA-Propositions-Forecasting-App/main/betting data/...
      The Odds API historical event-odds snapshots (market player_points only), 2024-10-22 .. 2026-04-10,
      books: draftkings, fanduel, betmgm, williamhill_us (Caesars), bovada, betonlineag. One pre-game snapshot/event.
  vincedir/betting_data.csv
      raw.githubusercontent.com/VinceDiR/Prop_Betting_Regression_Project/main/betting_data.csv
      BettingPros points props 2021-10-21 .. 2022-01-26; one line + the price of BettingPros' projected side only.
No other free, downloadable multi-season NBA prop archive was found on GitHub (Oct 2026 search).
"""
import difflib
import glob
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'props_nba')
OUT_BOX = os.path.join(ROOT, 'data', 'processed', 'nba_player_box.csv')
OUT_PROPS = os.path.join(ROOT, 'data', 'processed', 'nba_props.csv')


# ---------------------------------------------------------------- box scores
def build_box():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, 'player_box', 'player_box_*.parquet'))):
        d = pd.read_parquet(f)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        'date': pd.to_datetime(d['game_date']).dt.strftime('%Y-%m-%d'),
        'tipoff_et': d['game_date_time'].astype(str),
        'season': d['season'],
        'season_type': d['season_type'],  # 1 pre, 2 regular, 3 post, (4/5 all-star/play-in per ESPN)
        'game_id': d['game_id'],
        'team': d['team_abbreviation'],
        'team_name': d['team_display_name'],
        'opp': d['opponent_team_abbreviation'],
        'home': (d['home_away'] == 'home').astype(int),
        'player_id': d['athlete_id'],
        'player': d['athlete_display_name'],
        'position': d['athlete_position_abbreviation'],
        'starter': d['starter'].fillna(False).astype(int),
        'did_not_play': d['did_not_play'].fillna(False).astype(int),
        'dnp_reason': d['reason'],
        'minutes': d['minutes'],
        'pts': d['points'], 'reb': d['rebounds'], 'oreb': d['offensive_rebounds'], 'dreb': d['defensive_rebounds'],
        'ast': d['assists'], 'fg3m': d['three_point_field_goals_made'], 'fg3a': d['three_point_field_goals_attempted'],
        'fgm': d['field_goals_made'], 'fga': d['field_goals_attempted'],
        'ftm': d['free_throws_made'], 'fta': d['free_throws_attempted'],
        'stl': d['steals'], 'blk': d['blocks'], 'tov': d['turnovers'], 'pf': d['fouls'],
        'plus_minus': pd.to_numeric(d['plus_minus'].astype(str).str.replace('+', '', regex=False), errors='coerce'),
        'team_score': d['team_score'], 'opp_score': d['opponent_team_score'],
    })
    played = out['did_not_play'].eq(0) & out['minutes'].fillna(0).gt(0)
    for c in ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov']:
        out.loc[~played, c] = np.nan  # DNP -> NaN, never 0
    out['pra'] = out['pts'] + out['reb'] + out['ast']
    out = out.drop_duplicates(['game_id', 'player_id']).sort_values(['date', 'game_id', 'team', 'player'])
    return out


# ---------------------------------------------------------------- helpers
SUFFIX = re.compile(r'\b(jr|sr|ii|iii|iv|v)\b')


def norm(name):
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[.\'`-]', ' ', s)
    s = SUFFIX.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip().replace(' ', '')


ALIAS = {  # odds-feed spelling -> ESPN spelling (normalised)
    'nicolasclaxton': 'nicclaxton', 'herbjones': 'herbertjones', 'camreddish': 'cameronreddish',
    'kenyonmartin': 'kjmartin', 'jakarisampson': 'jakarrsampson',
    'carltoncarrington': 'bubcarrington', 'gregjackson': 'ggjackson', 'moebamba': 'mohamedbamba',
    'ronholland': 'ronaldholland', 'bogdan': 'bogdanbogdanovic',
}


def dec_to_amer(x):
    if pd.isna(x) or x <= 1:
        return np.nan
    return round((x - 1) * 100) if x >= 2 else round(-100 / (x - 1))


def season_of(date):
    dt = pd.to_datetime(date)
    return dt.year + 1 if dt.month >= 8 else dt.year


# ---------------------------------------------------------------- prop sources
def load_kamran():
    base = os.path.join(RAW, 'kamran')
    csv = pd.read_csv(os.path.join(base, 'partner_odds_backtest.csv'), dtype={'event_id': str})
    rows, have = [], set()
    for f in glob.glob(os.path.join(base, 'events', '*.json')):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        ev = j['data']
        have.add(ev['id'])
        for b in ev.get('bookmakers', []):
            for m in b['markets']:
                if m['key'] != 'player_points':
                    continue
                tmp = {}
                for o in m['outcomes']:
                    k = (o['description'], o.get('point'))
                    tmp.setdefault(k, {})[o['name'].lower()] = o['price']
                for (pl, pt), px in tmp.items():
                    rows.append(dict(event_id=ev['id'], snapshot_ts=j['timestamp'], market_ts=m.get('last_update'),
                                     commence_time=ev['commence_time'], home_team=ev['home_team'],
                                     away_team=ev['away_team'], book=b['key'], player=pl, market='pts', line=pt,
                                     over_dec=px.get('over'), under_dec=px.get('under')))
    k = pd.DataFrame(rows)
    # fallback: CSV rows for events whose JSON is missing
    miss = csv[~csv.event_id.isin(have)]
    if len(miss):
        k = pd.concat([k, pd.DataFrame(dict(
            event_id=miss.event_id, snapshot_ts=miss.snapshot_ts, market_ts=np.nan,
            commence_time=pd.NaT, home_team=miss.home_team, away_team=miss.away_team, book=miss.bookmaker,
            player=miss.player_name, market='pts', line=miss.line, over_dec=miss.over_odds,
            under_dec=miss.under_odds, csv_date=miss.GAME_DATE))], ignore_index=True)
    ct = pd.to_datetime(k['commence_time'], utc=True)
    k['date'] = ct.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    if 'csv_date' in k:
        k['date'] = k['date'].fillna(k['csv_date'])
    k['minutes_before_tip'] = (ct - pd.to_datetime(k['snapshot_ts'], utc=True)).dt.total_seconds() / 60
    k['over_odds'] = k['over_dec'].map(dec_to_amer)
    k['under_odds'] = k['under_dec'].map(dec_to_amer)
    k['source'] = 'oddsapi_via_KamranSHussain'
    k['source_type'] = 'sportsbook'
    k['team'] = np.nan
    return k


def load_vincedir():
    v = pd.read_csv(os.path.join(RAW, 'vincedir', 'betting_data.csv'))
    side = v['over_under'].str.strip().str.lower()
    ml = pd.to_numeric(v['Moneyline'], errors='coerce')
    return pd.DataFrame(dict(
        date=v['Date'], player=v['Name'].str.strip(), team=v['Team'], market='pts', line=v['Line'],
        over_odds=np.where(side == 'o', ml, np.nan), under_odds=np.where(side == 'u', ml, np.nan),
        book='bettingpros_consensus', source='bettingpros_via_VinceDiR', source_type='sportsbook',
        snapshot_ts=np.nan, minutes_before_tip=np.nan, bp_pick_side=side, bp_projection=v['Projection']))


# ---------------------------------------------------------------- matching
def match(props, box):
    b = box[['date', 'game_id', 'team', 'opp', 'home', 'player_id', 'player', 'team_name', 'did_not_play',
             'minutes', 'pts', 'reb', 'ast', 'fg3m', 'pra']].copy()
    b['key'] = b['player'].map(norm)
    by_date = {d: g for d, g in b.groupby('date')}
    known = b.sort_values('date').groupby('key')['player_id'].last().to_dict()  # every name ESPN ever used
    out = []
    for date, g in props.groupby('date'):
        cand = by_date.get(date, b.iloc[:0])
        keys = dict(zip(cand['key'], cand.index))
        for name in g['player'].unique():
            n = norm(name)
            n = ALIAS.get(n, n)
            idx = keys.get(n)
            how = 'exact'
            if idx is None and n in known:
                # real player with a prop but no box row that date -> inactive / ruled out after the snapshot
                out.append(dict(date=date, player=name, box_idx=None, match_type='inactive', pid_known=known[n]))
                continue
            if idx is None:
                close = difflib.get_close_matches(n, list(keys), n=2, cutoff=0.85)
                if len(close) >= 1:
                    idx, how = keys[close[0]], 'fuzzy'
            out.append(dict(date=date, player=name, box_idx=idx, match_type=how if idx is not None else None))
    m = pd.DataFrame(out)
    props = props.merge(m, on=['date', 'player'], how='left')
    j = b.drop(columns='key').add_prefix('bx_')
    props = props.join(j, on='box_idx')
    return props


def main():
    box = build_box()
    os.makedirs(os.path.dirname(OUT_BOX), exist_ok=True)
    box.to_csv(OUT_BOX, index=False)
    print('box rows', len(box), 'seasons', box.season.min(), '-', box.season.max())

    props = pd.concat([load_kamran(), load_vincedir()], ignore_index=True)
    props = props[props['line'].notna()]
    props = match(props, box)
    # sanity: for odds-api rows the matched team must be one of the two event teams
    tn = props['bx_team_name'].replace({'LA Clippers': 'Los Angeles Clippers'})
    bad = props['home_team'].notna() & props['bx_team_name'].notna() & ~(
        tn.eq(props['home_team']) | tn.eq(props['away_team']))
    props.loc[bad, ['box_idx', 'match_type']] = [np.nan, None]
    props.loc[bad, [c for c in props if c.startswith('bx_')]] = np.nan
    if 'pid_known' not in props:
        props['pid_known'] = np.nan
    props['bx_player_id'] = props['bx_player_id'].fillna(props['pid_known'])

    # main vs alternate line: per (date, player, market, book) the line with the most balanced prices is main
    o, u = props['over_odds'], props['under_odds']
    imp = lambda a: np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
    props['_bal'] = np.abs(imp(o.astype(float)) - imp(u.astype(float)))
    props['_bal'] = props['_bal'].fillna(9)
    props['_r'] = props.groupby(['date', 'player', 'market', 'book'])['_bal'].rank(method='first')
    props['alt_line'] = (props['_r'] > 1).astype(int)

    mk = props['market']
    props['actual'] = np.select([mk.eq(x) for x in ['pts', 'reb', 'ast', 'fg3m', 'pra']],
                                [props['bx_' + x] for x in ['pts', 'reb', 'ast', 'fg3m', 'pra']], np.nan)
    props['result'] = np.where(props['actual'].isna(), np.where(props['box_idx'].notna(), 'void_dnp',
                                        np.where(props['match_type'].eq('inactive'), 'void_inactive', '')),
                               np.where(props['actual'] > props['line'], 'over',
                                        np.where(props['actual'] < props['line'], 'under', 'push')))
    props['team'] = props['bx_team'].fillna(props['team'])
    out = pd.DataFrame(dict(
        date=props['date'], season=props['date'].map(season_of), game_id=props['bx_game_id'],
        player=props['player'], player_box_name=props['bx_player'], player_id=props['bx_player_id'],
        team=props['team'], opp=props['bx_opp'], home=props['bx_home'], market=props['market'], line=props['line'],
        over_odds=props['over_odds'], under_odds=props['under_odds'], source_type=props['source_type'],
        alt_line=props['alt_line'], book=props['book'], source=props['source'], snapshot_ts=props['snapshot_ts'],
        minutes_before_tip=props['minutes_before_tip'].round(1), minutes=props['bx_minutes'],
        actual=props['actual'], result=props['result'], match_type=props['match_type'],
        bp_pick_side=props.get('bp_pick_side'), bp_projection=props.get('bp_projection')))
    for c in ['game_id', 'player_id', 'home', 'over_odds', 'under_odds']:
        out[c] = out[c].astype('Int64')
    out = out.drop_duplicates(['date', 'player', 'market', 'book', 'line']).sort_values(['date', 'player', 'book'])
    out.to_csv(OUT_PROPS, index=False)

    # ---------------- validation
    print('props rows', len(out))
    print('match rate (box row found) %.2f%% | player identified %.2f%%' % (
        100 * out['game_id'].notna().mean(), 100 * out['player_id'].notna().mean()))
    print('match rate excl. inactive %.2f%%' % (100 * out.loc[out.result != 'void_inactive', 'game_id'].notna().mean()))
    print(out.groupby(['source', 'season', 'market']).size().to_string())
    print('pct with both prices', round(100 * (out.over_odds.notna() & out.under_odds.notna()).mean(), 1),
          '| any price', round(100 * (out.over_odds.notna() | out.under_odds.notna()).mean(), 1))
    print('main-line rows by source/season:'); print(out[out.alt_line == 0].groupby(['source', 'season']).size().to_string())
    g = out[out.result.isin(['over', 'under']) & out.alt_line.eq(0)]
    print('over-hit rate by source/market:')
    print(g.assign(o=g.result.eq('over')).groupby(['source', 'market', 'book'])['o'].agg(['mean', 'size']).to_string())
    print('result counts', out.result.value_counts().to_dict())
    mbt = out['minutes_before_tip'].dropna()
    print('minutes_before_tip: min %.1f median %.1f; rows <=0: %d' % (mbt.min(), mbt.median(), (mbt <= 0).sum()))


if __name__ == '__main__':
    main()
