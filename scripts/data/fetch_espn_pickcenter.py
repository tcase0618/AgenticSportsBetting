"""Fetch ESPN game-summary JSONs mirrored in sportsdataverse hoopR-{mbb,nba}-raw (raw.githubusercontent.com)
and keep only header + pickcenter (betting lines) per game, written as JSONL.
Usage: python fetch_espn_pickcenter.py {mbb|nba} SEASON SCHEDULE_PARQUET OUT_JSONL
"""
import sys, json, urllib.request, concurrent.futures as cf, time, os
import pandas as pd

league, season, sched, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
REPO = {'mbb': 'hoopR-mbb-raw/main/mbb', 'nba': 'hoopR-nba-raw/main/nba'}[league]
DROP = {'links', 'logos', 'logo', 'link', 'header', 'team'}

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o

def fetch(gid):
    url = f'https://raw.githubusercontent.com/sportsdataverse/{REPO}/json/final/{gid}.json'
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                g = json.load(r)
            comp = g['header']['competitions'][0]
            rec = {'game_id': int(gid), 'season': season, 'date': comp.get('date'),
                   'neutral': comp.get('neutralSite'), 'conference': comp.get('conferenceCompetition'),
                   'season_type': g['header'].get('season', {}).get('type'),
                   'notes': [n.get('headline') for n in comp.get('notes', [])],
                   'status': comp.get('status', {}).get('type', {}).get('name'),
                   'competitors': [{'homeAway': c.get('homeAway'), 'id': c.get('id'),
                                    'location': c['team'].get('location'), 'name': c['team'].get('name'),
                                    'displayName': c['team'].get('displayName'),
                                    'abbreviation': c['team'].get('abbreviation'), 'score': c.get('score')}
                                   for c in comp['competitors']],
                   'pickcenter': clean(g.get('pickcenter') or []),
                   'odds': clean(g.get('odds') or [])}
            return rec
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {'game_id': int(gid), 'season': season, 'error': '404'}
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {'game_id': int(gid), 'season': season, 'error': 'failed'}

s = pd.read_parquet(sched)
s = s[s.game_json == True]
if 'status_type_completed' in s:
    s = s[s.status_type_completed == True]
ids = s.game_id.astype(int).unique().tolist()
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line)
        if 'error' not in r or r['error'] == '404':
            done.add(r['game_id'])
ids = [i for i in ids if i not in done]
print(league, season, 'to fetch', len(ids), flush=True)
with open(out, 'a') as f, cf.ThreadPoolExecutor(48) as ex:
    for n, rec in enumerate(ex.map(fetch, ids)):
        f.write(json.dumps(rec) + '\n')
        if n % 1000 == 0:
            print(season, n, flush=True)
print('done', season, flush=True)
