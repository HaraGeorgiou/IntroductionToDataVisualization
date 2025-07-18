#!/usr/bin/env python3
"""
epl_xpts_expected_points.py

Fetches xPTS (expected points) and actual points for EPL clubs from 2014/15–2023/24,
fits an OLS of points ~ xPTS, computes expected points (based on xPTS), and
exports season-level raw and summary JSON.

Usage:
    pip install understat aiohttp pandas nest_asyncio
    python epl_xpts_expected_points.py
"""
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import numpy as np
from understat import Understat

# allow nested event loop
nest_asyncio.apply()

# Configuration
LEAGUE         = "EPL"
START_SEASON   = 2014  # 2014/15
END_SEASON     = 2024  # 2023/24
RAW_OUTPUT     = "data/epl_xpts_points_raw.json"
SUMMARY_OUTPUT = "data/epl_xpts_expected_summary.json"

async def fetch_xpts():
    """
    Fetch per-season xPTS and actual points from Understat.
    Returns records list.
    """
    records = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in range(START_SEASON, END_SEASON + 1):
            print(f"Fetching season {season} data…")
            table = await understat.get_league_table(LEAGUE, str(season))
            if not table:
                continue
            # detect header row format
            if isinstance(table[0], list):
                headers = table[0]
                # find indices for team, xPTS, and actual points
                team_idx = next((i for i,h in enumerate(headers) if 'team' in h.lower() or 'squad' in h.lower()), None)
                xpts_idx = next((i for i,h in enumerate(headers) if 'xpts' == h.lower()), None)
                pts_idx  = next((i for i,h in enumerate(headers) if 'pts' == h.lower()), None)
                if team_idx is None or xpts_idx is None or pts_idx is None:
                    raise ValueError(f"Missing columns in season {season} headers: {headers}")
                for row in table[1:]:
                    team = row[team_idx]
                    try:
                        xpts = float(row[xpts_idx])
                    except:
                        xpts = 0.0
                    try:
                        pts = int(row[pts_idx])
                    except:
                        pts = 0
                    records.append({
                        'season_end_year': season,
                        'team': team,
                        'xPTS': xpts,
                        'points': pts
                    })
            else:
                # list of dicts
                sample = table[0]
                team_key = next((k for k in sample if 'team' in k.lower() or 'title' in k.lower()), None)
                xpts_key = next((k for k in sample if k.lower() == 'xpts'), None)
                pts_key  = next((k for k in sample if k.lower() == 'pts'), None)
                if team_key is None or xpts_key is None or pts_key is None:
                    raise ValueError(f"Missing keys in season {season} row: {sample.keys()}")
                for row in table:
                    try:
                        xpts = float(row.get(xpts_key, 0))
                    except:
                        xpts = 0.0
                    try:
                        pts = int(row.get(pts_key, 0))
                    except:
                        pts = 0
                    records.append({
                        'season_end_year': season,
                        'team': row.get(team_key),
                        'xPTS': xpts,
                        'points': pts
                    })
    return records


def main():
    records = asyncio.get_event_loop().run_until_complete(fetch_xpts())
    df = pd.DataFrame(records)
    df.to_json(RAW_OUTPUT, orient='records', indent=2)
    print(f"Raw xPTS/points data written to {RAW_OUTPUT}")

    # Fit OLS regression: points ~ xPTS
    m, b = np.polyfit(df['xPTS'], df['points'], 1)
    print(f"Fitted model: points = {m:.3f} * xPTS + {b:.2f}")

    # Compute expected and residuals
    df['expected_from_xpts'] = m * df['xPTS'] + b
    df['residual'] = df['points'] - df['expected_from_xpts']

    # Save raw with residuals
    df.to_json(RAW_OUTPUT, orient='records', indent=2)

    # Aggregate summary by team
    agg = (
        df.groupby('team')
          .agg(
            seasons_played           = ('season_end_year', 'count'),
            avg_xPTS                 = ('xPTS', 'mean'),
            avg_points               = ('points', 'mean'),
            avg_expected_from_xpts   = ('expected_from_xpts', 'mean'),
            mean_residual            = ('residual', 'mean'),
            std_residual             = ('residual', 'std')
          )
          .reset_index()
    )
    agg['std_residual'] = agg['std_residual'].fillna(0)
    agg.to_json(SUMMARY_OUTPUT, orient='records', indent=2)
    print(f"Summary written to {SUMMARY_OUTPUT}")

if __name__ == '__main__':
    main()