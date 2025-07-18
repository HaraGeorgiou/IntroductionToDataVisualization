#!/usr/bin/env python3
"""
fetch_epl_xg.py

Fetches per-club Expected Goals (xG) for all English Premier League seasons
from 2014/15 through 2023/24 using the understat package, and writes
out a CSV table with columns: season_end_year, team, xG.

Usage:
    pip install understat aiohttp pandas nest_asyncio
    python fetch_epl_xg.py
"""
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
from understat import Understat

# allow nested event loop in notebooks/conda
nest_asyncio.apply()

# --- Configuration ---
LEAGUE = "EPL"
START_SEASON = 2015  # for 2014/15
END_SEASON   = 2024  # for 2023/24
OUTPUT_CSV   = "epl_xg.csv"

async def fetch_xg():
    """
    Async fetch of xG data from Understat for each season.
    Returns list of records.
    """
    records = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in range(START_SEASON, END_SEASON + 1):
            print(f"Fetching xG for season ending {season}...")
            table = await understat.get_league_table(LEAGUE, str(season))
            if not table:
                continue
            # Detect if table is list-of-lists: first row is headers
            if isinstance(table[0], list):
                headers = table[0]
                # find index of team and xG columns (case-insensitive)
                team_idx = next((i for i,h in enumerate(headers) if 'team' in h.lower() or 'squad' in h.lower()), None)
                xg_idx   = next((i for i,h in enumerate(headers) if h.lower() == 'xg'), None)
                if team_idx is None or xg_idx is None:
                    raise ValueError(f"Couldn't find team or xG column in headers: {headers}")
                for values in table[1:]:
                    team = values[team_idx]
                    xg_val = values[xg_idx]
                    # convert xG string to float
                    try:
                        xg = float(xg_val)
                    except:
                        xg = 0.0
                    records.append({
                        "season_end_year": season,
                        "team": team,
                        "xG": xg
                    })
            else:
                # assume list of dicts
                # detect key for team and xG
                # pick first row to inspect keys
                sample = table[0]
                team_key = next((k for k in sample if 'team' in k.lower() or 'title' in k.lower()), None)
                xg_key   = next((k for k in sample if k.lower() == 'xg'), None)
                if team_key is None or xg_key is None:
                    raise ValueError(f"Couldn't find team or xG key in row: {sample.keys()}")
                for row in table:
                    try:
                        xg = float(row.get(xg_key, 0))
                    except:
                        xg = 0.0
                    records.append({
                        "season_end_year": season,
                        "team": row.get(team_key),
                        "xG": xg
                    })
    return records


def main():
    # run the async fetch
    records = asyncio.get_event_loop().run_until_complete(fetch_xg())

    # build DataFrame
    df = pd.DataFrame(records)
    # sort
    df = df.sort_values(["season_end_year", "team"]).reset_index(drop=True)

    # write CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved Expected Goals table to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
