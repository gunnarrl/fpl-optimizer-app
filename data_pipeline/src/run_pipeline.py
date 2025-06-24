# run_pipeline.py

# --- Standard Library Imports ---
import asyncio
import logging
import os
import shutil
import sys
import traceback
import warnings
from collections import defaultdict

# --- Third-party Imports ---
import aiohttp
import dask
import numpy as np
import pandas as pd
from thefuzz import fuzz
from understat import Understat

# --- Constants ---
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"

# --- Team Name Normalization ---
TEAM_NAME_MAPPINGS = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley",
    "Cardiff": "Cardiff", "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham", "Huddersfield": "Huddersfield",
    "Hull": "Hull", "Ipswich": "Ipswich", "Leeds": "Leeds", "Leicester": "Leicester",
    "Liverpool": "Liverpool", "Luton": "Luton", "Manchester City": "Man City",
    "Manchester United": "Man Utd", "Middlesbrough": "Middlesbrough",
    "Newcastle United": "Newcastle", "Norwich": "Norwich",
    "Nottingham Forest": "Nott'm Forest", "Sheffield United": "Sheffield Utd",
    "Southampton": "Southampton", "Stoke": "Stoke", "Sunderland": "Sunderland",
    "Swansea": "Swansea", "Tottenham": "Spurs", "Watford": "Watford",
    "West Bromwich Albion": "West Brom", "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}  # src/run_pipeline.py


def normalize_team_name(name):  # src/run_pipeline.py
    if not isinstance(name, str):
        return ""
    # Simplified the normalization from the original file for clarity
    return TEAM_NAME_MAPPINGS.get(name, name)


# Helper functions for data conversion
def to_float(value, default=0.0):  # src/run_pipeline.py
    if value is None: return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def to_int(value, default=0):  # src/run_pipeline.py
    if value is None: return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


async def get_2024_25_premier_league_player_ids(understat_client):  # src/run_pipeline.py
    """Fetches all player IDs for the 2024-25 EPL season."""
    print("Fetching player list for the 2024-25 season to use as a filter...")
    player_ids = set()
    try:
        league_players_2024 = await understat_client.get_league_players(
            league_name="epl", season=2024
        )
        for player in league_players_2024:
            player_id = player.get('id')
            if player_id:
                player_ids.add(int(player_id))
        print(f"Found {len(player_ids)} unique players in the 2024-25 Premier League season.")
    except Exception as e:
        print(f"Could not fetch 2024-25 player data: {e}. No player filtering will be applied.")
    return player_ids


async def fetch_understat_data(gw_reference_df):
    """
    Fetches detailed player-match data from Understat.com.

    This function uses a provided FPL data DataFrame as a reference to match
    Understat players and games to their FPL counterparts, ensuring data alignment.

    Args:
        gw_reference_df (pd.DataFrame): A DataFrame containing FPL game data.
                                        Must include columns: 'season_x', 'kickoff_time',
                                        'opp_team_name', 'was_home', 'GW', 'name', 'element'.

    Returns:
        pd.DataFrame: A DataFrame containing detailed Understat data for matched games,
                      or an empty DataFrame if no data could be fetched.
    """
    start_year = 2016
    end_year = 2025
    all_player_rows = []
    appended_rows_count = 0

    # Key: (season, fpl_element_id, kickoff_date) -> Value: List of [(opponent_name, was_home, gameweek)]
    fpl_player_game_lookup = defaultdict(list)  # src/run_pipeline.py
    fpl_player_name_to_id_lookup = {}  # src/run_pipeline.py

    try:
        # Create a date-only column for matching
        gw_reference_df['kickoff_time_date_only'] = pd.to_datetime(gw_reference_df['kickoff_time']).dt.strftime(
            '%Y-%m-%d')

        required_ref_cols = ['season_x', 'kickoff_time_date_only', 'opp_team_name', 'was_home', 'GW', 'name',
                             'element']  # src/run_pipeline.py
        missing_cols = [col for col in required_ref_cols if col not in gw_reference_df.columns]
        if missing_cols:
            print(
                f"CRITICAL Warning: The provided reference DataFrame is missing required columns: {missing_cols}. Cannot proceed.")
            return pd.DataFrame()

        # Build lookup tables from the reference DataFrame
        for _, row in gw_reference_df.iterrows():
            ref_season = str(row['season_x'])
            fpl_element_id = to_int(row['element'])
            kickoff_date = str(row['kickoff_time_date_only'])
            ref_opponent_name = normalize_team_name(str(row['opp_team_name']))
            ref_gw = to_int(row['GW'])
            fpl_player_name_norm = str(row['name']).lower().strip()
            was_home_raw = row['was_home']
            if pd.isna(was_home_raw) or not ref_opponent_name: continue
            ref_was_home = bool(was_home_raw)
            fpl_player_game_lookup[(ref_season, fpl_element_id, kickoff_date)].append(
                (ref_opponent_name, ref_was_home, ref_gw))  # src/run_pipeline.py
            fpl_player_name_to_id_lookup[(ref_season, fpl_player_name_norm)] = fpl_element_id  # src/run_pipeline.py
        print(f"Successfully built FPL player game and name lookups from the reference DataFrame.")

    except Exception as e:
        print(f"Error processing reference DataFrame: {e}")
        return pd.DataFrame()

    try:
        async with aiohttp.ClientSession() as session:
            understat_client = Understat(session)
            players_in_2024_season_ids = await get_2024_25_premier_league_player_ids(understat_client)  # src/run_pipeline.py

            for year in range(start_year, end_year + 1):
                season_formatted = f"{year}-{(year + 1) % 100:02d}"
                print(f"\nFetching Understat data for season: {season_formatted}")

                league_players_understat = []
                try:
                    league_players_understat = await understat_client.get_league_players(
                        league_name="epl", season=year
                    )
                except Exception as e:
                    print(f"  Could not fetch Understat league players for season {year}: {e}")
                    continue

                if not league_players_understat:
                    print(f"  No Understat players found for season {year}. Skipping.")
                    continue
                print(f"  Found {len(league_players_understat)} Understat players for season {season_formatted}.")

                for player_summary in league_players_understat:
                    player_id_str = player_summary.get('id')
                    player_name_understat = player_summary.get('player_name')

                    if not player_id_str or not player_name_understat:
                        continue

                    player_id = int(player_id_str)

                    if players_in_2024_season_ids and player_id not in players_in_2024_season_ids:
                        continue

                    print(f"  Processing Understat player: {player_name_understat} (ID: {player_id_str})")

                    normalized_understat_name_for_fpl = player_name_understat.lower().strip()
                    calculated_fpl_id = -1
                    best_fuzz_score_fpl = 0

                    if (season_formatted, normalized_understat_name_for_fpl) in fpl_player_name_to_id_lookup:
                        calculated_fpl_id = fpl_player_name_to_id_lookup[
                            (season_formatted, normalized_understat_name_for_fpl)]
                        print(
                            f"[MATCHED] Exact match found: '{normalized_understat_name_for_fpl}' for season {season_formatted}.")
                    else:
                        best_fuzz_name = "N/A"
                        for (s_key, fpl_name_key), fpl_id_val in fpl_player_name_to_id_lookup.items():
                            if s_key == season_formatted:
                                score = fuzz.token_set_ratio(normalized_understat_name_for_fpl, fpl_name_key)
                                if score > best_fuzz_score_fpl:
                                    best_fuzz_score_fpl = score
                                    best_fuzz_name = fpl_name_key
                                    if score >= 70:
                                        calculated_fpl_id = fpl_id_val
                        if calculated_fpl_id != -1:
                            print(
                                f"[FUZZY MATCH] Matched '{normalized_understat_name_for_fpl}' to '{best_fuzz_name}' with score {best_fuzz_score_fpl}.")
                        else:
                            print(
                                f"[NO MATCH] Could not match '{normalized_understat_name_for_fpl}' in season {season_formatted}. Best score was {best_fuzz_score_fpl} against '{best_fuzz_name}'.")

                    all_matches_for_player_raw_understat = []
                    try:
                        all_matches_for_player_raw_understat = await understat_client.get_player_matches(
                            player_id=player_id)
                    except Exception as e:
                        continue

                    if not all_matches_for_player_raw_understat:
                        continue

                    for m_data in all_matches_for_player_raw_understat:
                        understat_match_date_full = m_data.get('date')
                        if not understat_match_date_full: continue

                        try:
                            understat_match_date_only = pd.to_datetime(understat_match_date_full).strftime('%Y-%m-%d')
                        except Exception:
                            if isinstance(understat_match_date_full, str) and " " in understat_match_date_full:
                                understat_match_date_only = understat_match_date_full.split(" ")[0]
                            else:
                                continue

                        # --- REVISED MATCH VALIDATION LOGIC ---
                        validated_gw = -1
                        match_validated_as_epl = False

                        # Only proceed if we have a valid FPL ID for the player
                        if calculated_fpl_id != -1:
                            lookup_key = (season_formatted, calculated_fpl_id, understat_match_date_only)
                            possible_fpl_games = fpl_player_game_lookup.get(lookup_key)

                            if possible_fpl_games:
                                understat_h_team_raw = m_data.get('h_team')
                                understat_a_team_raw = m_data.get('a_team')
                                if not understat_h_team_raw or not understat_a_team_raw:
                                    continue

                                norm_understat_h = normalize_team_name(understat_h_team_raw)
                                norm_understat_a = normalize_team_name(understat_a_team_raw)

                                # Check against the specific FPL games for that player on that day
                                for ref_opponent_name, ref_was_home, ref_gw in possible_fpl_games:
                                    understat_opponent_name = norm_understat_a if ref_was_home else norm_understat_h
                                    opponent_match_score = fuzz.token_set_ratio(understat_opponent_name,
                                                                                ref_opponent_name)

                                    if opponent_match_score >= 90:
                                        validated_gw = ref_gw
                                        match_validated_as_epl = True
                                        break  # Match found, no need to check further

                        if match_validated_as_epl:
                            row_data = {
                                'goals': to_int(m_data.get('goals')), 'shots': to_int(m_data.get('shots')),
                                'xG': to_float(m_data.get('xG')), 'time': to_int(m_data.get('time')),
                                'position': m_data.get('position'), 'h_team': understat_h_team_raw,
                                'a_team': understat_a_team_raw, 'h_goals': to_int(m_data.get('h_goals')),
                                'a_goals': to_int(m_data.get('a_goals')), 'date': understat_match_date_full,
                                'id': player_id, 'season': year,
                                'roster_id': to_int(m_data.get('rID')), 'xA': to_float(m_data.get('xA')),
                                'assists': to_int(m_data.get('assists')),
                                'key_passes': to_int(m_data.get('key_passes')),
                                'npg': to_int(m_data.get('npg')), 'npxG': to_float(m_data.get('npxG')),
                                'xGChain': to_float(m_data.get('xGChain')),
                                'xGBuildup': to_float(m_data.get('xGBuildup')),
                                'match_identifier': to_int(m_data.get('id')),
                                'fpl_id': to_int(calculated_fpl_id, default=-1),
                                'season_formatted_for_idx': season_formatted,
                                'player_name_for_idx': player_name_understat,
                                'gameweek_for_idx': to_int(validated_gw, default=-1)
                            }
                            all_player_rows.append(row_data)
                            appended_rows_count += 1

                    if len(all_matches_for_player_raw_understat) > 0:
                        await asyncio.sleep(0.02)

    except aiohttp.ClientError as e:
        print(f"A client error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\n--- Understat Processing Complete. Total rows appended: {appended_rows_count} ---")
    if not all_player_rows:
        print("No Understat data collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_player_rows)
    if df.empty:
        return pd.DataFrame()

    # Set index and select columns as in the original script
    df = df.sort_values(by=['season_formatted_for_idx', 'player_name_for_idx', 'gameweek_for_idx'])
    df = df.set_index(['season_formatted_for_idx', 'player_name_for_idx', 'gameweek_for_idx'])
    df.index.names = ['Season', 'Player Name', 'GW']

    output_column_order = [
        'goals', 'shots', 'xG', 'time', 'position', 'h_team', 'a_team', 'h_goals', 'a_goals', 'date', 'id',
        'season', 'fpl_id', 'roster_id', 'xA', 'assists', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup',
        'match_identifier'
    ]  # src/run_pipeline.py
    final_output_columns = [col for col in output_column_order if col in df.columns]

    # Return the DataFrame instead of saving to CSV
    return df[final_output_columns]


async def load_and_process_data(base_url, start_season=2020, end_season=2023,
                                target_season_context='2024-25'):
    """
    Consolidates the entire data ingestion pipeline:
    1. Loads the pre-processed FPL data CSV.
    2. Filters FPL data to include only players active in the `target_season_context`.
    3. Uses the filtered FPL data as a reference to fetch and fuzzy-match Understat data.
    4. Merges the two datasets.
    5. Correctly handles missing Understat values.
    6. Merges in expected points (xP) data.
    """
    # --- MODIFIED: Part 1: Load the pre-processed FPL Data CSV ---
    print("--- Part 1: Loading pre-processed FPL data ---")
    fpl_data_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/cleaned/cleaned_merged_seasons_team_aggregated.csv"
    try:
        fpl_df = pd.read_csv(fpl_data_url)
        print(f"Successfully loaded FPL data from your specified file. Initial shape: {fpl_df.shape}")
    except Exception as e:
        print(f"CRITICAL: Could not load FPL data from {fpl_data_url}. Aborting. Error: {e}")
        return pd.DataFrame()

    # --- Part 1.5: Filter FPL Data Based on Current Season Players ---
    print(f"\n--- Part 1.5: Filtering for players active in {target_season_context} ---")
    player_list_url = f"{base_url}{target_season_context}/player_idlist.csv"
    try:
        player_list_df = pd.read_csv(player_list_url)
        player_list_df['full_name'] = player_list_df['first_name'] + " " + player_list_df['second_name']
        current_season_player_names = set(player_list_df['full_name'].str.strip())
        print(f"Loaded {len(current_season_player_names)} unique player names for filtering.")

        original_rows = len(fpl_df)
        fpl_df = fpl_df[fpl_df['name'].isin(current_season_player_names)].copy()
        print(f"Filtered FPL data from {original_rows} to {len(fpl_df)} rows. New shape: {fpl_df.shape}")

    except Exception as e:
        print(f"Could not load or process player list from {player_list_url}. Skipping filtering. Error: {e}")

    # --- Part 2: Fetch and Match Understat Data ---
    print("\n--- Part 2: Fetching and matching Understat data ---")
    all_player_rows = []
    fpl_player_game_lookup = defaultdict(list)
    fpl_player_name_to_id_lookup = {}

    # This step is no longer needed as the cleaned CSV should have the correct team names
    # team_list_df = pd.read_csv(f"{base_url}master_team_list.csv")
    # team_id_to_name = dict(zip(team_list_df['team'], team_list_df['team_name']))
    # fpl_df['opp_team_name'] = fpl_df['opponent_team'].map(team_id_to_name)

    fpl_df['kickoff_time_date_only'] = pd.to_datetime(fpl_df['kickoff_time']).dt.strftime('%Y-%m-%d')
    for _, row in fpl_df.iterrows():
        key = (str(row['season_x']), to_int(row['element']), str(row['kickoff_time_date_only']))
        val = (normalize_team_name(str(row['opp_team_name'])), bool(row['was_home']), to_int(row['GW']))
        fpl_player_game_lookup[key].append(val)
        fpl_player_name_to_id_lookup[(str(row['season_x']), str(row['name']).lower().strip())] = to_int(row['element'])
    print("Successfully built FPL player game and name lookups for Understat matching.")

    async with aiohttp.ClientSession() as session:
        understat_client = Understat(session)
        # We can narrow the years since we are using a specific file now, but keeping it broad is safer
        for year in range(start_season, end_season + 1):
            season_formatted = f"{year}-{(year + 1) % 100:02d}"
            print(f"Fetching Understat league players for season {season_formatted}...")
            league_players_understat = await understat_client.get_league_players("epl", year)
            for player_summary in league_players_understat:
                player_id = int(player_summary['id'])
                player_name_understat = player_summary['player_name']
                fpl_id = fpl_player_name_to_id_lookup.get((season_formatted, player_name_understat.lower().strip()),
                                                          -1)
                if fpl_id == -1:
                    best_score = 70
                    for (s_key, fpl_name_key), id_val in fpl_player_name_to_id_lookup.items():
                        if s_key == season_formatted:
                            score = fuzz.token_set_ratio(player_name_understat.lower().strip(), fpl_name_key)
                            if score > best_score:
                                best_score = score
                                fpl_id = id_val
                player_matches = await understat_client.get_player_matches(player_id)
                for m_data in player_matches:
                    match_date = pd.to_datetime(m_data['date']).strftime('%Y-%m-%d')
                    lookup_key = (season_formatted, fpl_id, match_date)
                    if lookup_key in fpl_player_game_lookup:
                        m_data['fpl_id'] = fpl_id
                        all_player_rows.append(m_data)

    understat_df = pd.DataFrame(all_player_rows)
    print(f"Fetched {len(understat_df)} matched Understat game rows.")

    # --- Part 3: Merge FPL and Understat Data ---
    print("\n--- Part 3: Merging FPL and Understat data ---")
    if not understat_df.empty:
        understat_df.rename(columns={'fpl_id': 'element'}, inplace=True)
        understat_df['merge_date'] = pd.to_datetime(understat_df['date']).dt.date.astype(str)
        fpl_df['merge_date'] = pd.to_datetime(fpl_df['kickoff_time']).dt.date.astype(str)

        understat_cols_to_merge = ['element', 'merge_date', 'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain',
                                   'xGBuildup']
        master_df = pd.merge(fpl_df, understat_df[understat_cols_to_merge], on=['element', 'merge_date'], how='left')
    else:
        master_df = fpl_df.copy()

    # --- Part 4: Handle Missing Values Correctly ---
    print("\n--- Part 4: Handling missing values ---")
    understat_value_columns = ['xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup']
    for col in understat_value_columns:
        if col not in master_df.columns:
            master_df[col] = np.nan
    master_df['understat_missing'] = ((master_df['minutes'] > 0) & (master_df['xG'].isnull())).astype(int)
    did_not_play_mask = (master_df['minutes'] == 0)
    master_df.loc[did_not_play_mask, understat_value_columns] = master_df.loc[
        did_not_play_mask, understat_value_columns].fillna(0)
    print("Missing Understat values for players with 0 minutes filled with 0.")
    print("Missing values for players with >0 minutes are left as NaN.")

    # --- Part 5: Add Expected Points (xP) ---
    print("\n--- Part 5: Adding expected points (xP) ---")
    master_df = add_expected_points(master_df, base_url)
    master_df.drop(columns=['merge_date', 'kickoff_time_date_only'], inplace=True, errors='ignore')
    print(f"\nFinal processed dataset shape: {master_df.shape}")
    return master_df


def add_expected_points(master_df, base_url):
    """
    Finds and combines seasonal expected points (xP) data from the raw data repo
    and merges it into the master dataset.
    """
    if master_df.empty:
        print("Master DataFrame is empty. Skipping adding expected points.")
        return master_df

    print("\n--- Searching for and loading Expected Points (xP) data ---")
    original_row_count = len(master_df)
    xp_data_frames = []
    for year in range(2020, 2024):
        season_folder_name = f"{year}-{(year + 1) % 100:02d}"
        xp_file_url = f"{base_url}{season_folder_name}/gws/merged_gw.csv"
        try:
            gw_df = pd.read_csv(xp_file_url)
            required_columns = ['element', 'kickoff_time', 'xP']
            if all(col in gw_df.columns for col in required_columns):
                temp_df = gw_df[required_columns].copy()
                xp_data_frames.append(temp_df)
                print(f"  > Successfully loaded and processed xP data for season {season_folder_name}")
            else:
                print(f"  > WARNING: Skipped file for season {season_folder_name}. Missing required columns.")
        except Exception as e:
            print(
                f"  > INFO: No data file found or error for season {season_folder_name} at '{xp_file_url}'. Reason: {e}")
    if not xp_data_frames:
        print("\n--- WARNING: No xP data was found. The 'xP' column will not be added. ---")
        master_df['xP'] = 0.0
    else:
        all_xp_df = pd.concat(xp_data_frames, ignore_index=True)
        print(f"\nSuccessfully combined all available xP data. Total xP entries: {len(all_xp_df)}")
        all_xp_df.drop_duplicates(subset=['element', 'kickoff_time'], keep='last', inplace=True)
        master_df['kickoff_time'] = pd.to_datetime(master_df['kickoff_time'])
        all_xp_df['kickoff_time'] = pd.to_datetime(all_xp_df['kickoff_time'])
        master_df['element'] = master_df['element'].astype(int)
        all_xp_df['element'] = all_xp_df['element'].astype(int)
        master_df = pd.merge(master_df, all_xp_df, on=['element', 'kickoff_time'], how='left')
        print("Merge operation complete.")
        if 'xP' in master_df.columns:
            master_df['xP'] = master_df['xP'].fillna(0.0)
            print("Filled missing 'xP' values with 0.")
        else:
            master_df['xP'] = 0.0
            print("No 'xP' data was merged. Created a default 'xP' column with 0s.")
    if len(master_df) != original_row_count:
        print(f"\n--- CRITICAL WARNING: The number of rows changed from {original_row_count} to {len(master_df)}. ---")
    else:
        print(f"\nRow count validation passed. The number of rows remains {original_row_count}.")
    return master_df


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting FPL Data Pipeline ---")
    final_df = asyncio.run(load_and_process_data(BASE_URL))
    try:
        output_filename = "test_output_corrected.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"\n--- TEST COMPLETE ---")
        print(f"Successfully saved intermediate results to '{output_filename}' for review.")
    except Exception as e:
        print(f"\nError saving test CSV file: {e}")
    print("\n--- Pipeline Paused for Review ---")