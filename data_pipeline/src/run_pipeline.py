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
    start_year = 2020
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


async def load_and_process_data(base_url, start_season=2020, end_season=2024,
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
    fpl_data_url = "cleaned_merged_seasons_team_aggregated.csv"
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
    # This call now uses the robust, self-contained `fetch_understat_data` function
    # which correctly validates matches using opponent names.
    understat_df = await fetch_understat_data(fpl_df)
    print(f"Fetched {len(understat_df)} matched Understat game rows.")

    # --- Part 3: Merge FPL and Understat Data ---
    print("\n--- Part 3: Merging FPL and Understat data ---")
    if not understat_df.empty:
        # Reset the index from the Understat DataFrame to prepare for merging
        understat_df_to_merge = understat_df.reset_index()
        understat_df_to_merge.rename(columns={'fpl_id': 'element'}, inplace=True)

        # Create consistent date-based keys for merging
        understat_df_to_merge['merge_date'] = pd.to_datetime(understat_df_to_merge['date']).dt.date.astype(str)
        fpl_df['merge_date'] = pd.to_datetime(fpl_df['kickoff_time']).dt.date.astype(str)

        understat_cols_to_merge = ['element', 'merge_date', 'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain',
                                   'xGBuildup']
        # Ensure all columns exist before trying to merge them
        understat_cols_to_merge = [col for col in understat_cols_to_merge if col in understat_df_to_merge.columns]
        master_df = pd.merge(fpl_df, understat_df_to_merge[understat_cols_to_merge], on=['element', 'merge_date'], how='left')
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
    master_df.drop(columns=['merge_date'], inplace=True, errors='ignore')
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


def calculate_aggregate_stats(df):
    """
    Calculates season-to-date aggregate statistics for each player
    on a gameweek-by-gameweek basis using an in-memory DataFrame.
    This version specifically excludes Understat data from aggregation if it's marked as missing.
    It also adds per 90 minute statistics for the aggregated columns.

    Args:
        df (pd.DataFrame): The master player data DataFrame, expected to contain gameweek data.

    Returns:
        pd.DataFrame: The DataFrame with added aggregate and per-90-minute columns.
    """
    print("\n--- Step 5: Calculating Aggregate Statistics ---")
    # --- 1. Validate the incoming DataFrame ---
    if df.empty:
        print("Input DataFrame is empty. Skipping aggregate calculations.")
        return df

    # Check if 'understat_missing' column exists, as it's essential for the logic
    if 'understat_missing' not in df.columns:
        print("Error: 'understat_missing' column not found in the input DataFrame. Aborting aggregation.")
        return df

    # It's also a good idea to ensure 'kickoff_time' is a datetime object for robust sorting
    if 'kickoff_time' in df.columns:
        # Ensure it's datetime, handling potential errors
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
    else:
        print("Error: 'kickoff_time' column not found, which is required for sorting. Aborting aggregation.")
        return df

    # --- 2. Sort Data for Chronological Aggregation ---
    df.sort_values(by=['season_x', 'element', 'kickoff_time'], inplace=True)
    print("Data sorted by season, player (element), and kickoff_time for aggregation.")

    # --- 3. Define Columns for Aggregation ---
    understat_stats_to_aggregate = ['xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup']
    other_stats_to_aggregate = [
        'xP', 'goals_scored', 'saves', 'penalties_saved', 'assists',
        'total_points', 'minutes', 'own_goals', 'penalties_missed', 'clean_sheets',
        'threat', 'ict_index', 'influence', 'creativity'
    ]

    verified_understat_cols = [col for col in understat_stats_to_aggregate if col in df.columns]
    verified_other_cols = [col for col in other_stats_to_aggregate if col in df.columns]
    all_requested = understat_stats_to_aggregate + other_stats_to_aggregate
    all_verified = verified_understat_cols + verified_other_cols
    missing_cols = [col for col in all_requested if col not in all_verified]
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {missing_cols}")

    if not all_verified:
        print("Error: None of the specified columns for aggregation exist in the input DataFrame. Aborting.")
        return df

    # --- FIX: Convert all aggregation columns to numeric to prevent 'object' dtype error ---
    # Coerce errors will turn non-numeric values into NaN, which we then fill with 0.
    for col in all_verified:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print("Ensured all columns for aggregation are numeric.")

    print("\n--- Calculating Aggregates ---")

    # --- 4. Calculate Cumulative Sums with Conditional Logic for Understat columns ---
    print("\nProcessing stats with 'understat_missing' condition:")
    for col in verified_understat_cols:
        aggregate_col_name = f'aggregate_{col}'
        temp_col_name = f'temp_for_sum_{col}'
        df[temp_col_name] = np.where(df['understat_missing'] == 1, 0, df[col])
        df[aggregate_col_name] = df.groupby(['season_x', 'element'])[temp_col_name].cumsum()
        df.drop(columns=[temp_col_name], inplace=True)
        print(f"  > Created '{aggregate_col_name}' column (conditional).")

    # --- 5. Calculate Standard Cumulative Sums for Other Stats ---
    print("\nProcessing standard stats:")
    for col in verified_other_cols:
        aggregate_col_name = f'aggregate_{col}'
        df[aggregate_col_name] = df.groupby(['season_x', 'element'])[col].cumsum()
        print(f"  > Created '{aggregate_col_name}' column.")

    # --- 6. Calculate "Per 90 Minutes" Statistics ---
    print("\n--- Calculating Per 90 Minute Statistics ---")
    if 'aggregate_minutes' in df.columns:
        # Calculate the number of 90-minute periods played
        df['num_90s'] = df['aggregate_minutes'] / 90.0

        # Define all columns that have been aggregated (excluding minutes itself)
        stats_for_per_90 = verified_understat_cols + [col for col in verified_other_cols if col != 'minutes']

        for col in stats_for_per_90:
            aggregate_col_name = f'aggregate_{col}'
            per_90_col_name = f'per_90_{col}'

            # Calculate the per-90 stat, handling cases with 0 minutes to avoid division by zero
            df[per_90_col_name] = np.where(
                df['num_90s'] > 0,
                df[aggregate_col_name] / df['num_90s'],
                0
            )
            print(f"  > Created '{per_90_col_name}' column.")

        # Clean up the intermediate 'num_90s' column
        df.drop(columns=['num_90s'], inplace=True)
    else:
        print("Warning: 'aggregate_minutes' column not found. Cannot calculate per 90 stats.")

    print(f"\nSuccessfully calculated aggregate stats. Final DataFrame shape: {df.shape}")
    return df

def calculate_form_stats(df):
    """
    Calculates rolling 3 and 5-gameweek form statistics from a DataFrame.

    This function handles Understat columns conditionally, excluding gameweeks from the average
    where data is marked as missing.

    Args:
        df (pd.DataFrame): The DataFrame containing aggregate player data.

    Returns:
        pd.DataFrame: The DataFrame with added form statistics columns.
    """
    print("\n--- Step 6: Calculating Rolling Form Statistics ---")
    # --- 1. Validate the incoming DataFrame ---
    if df.empty:
        print("Input DataFrame is empty. Skipping form calculations.")
        return df

    required_cols = ['understat_missing', 'kickoff_time', 'season_x', 'element']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns for form calculation: {required_cols}. Aborting.")
        return df

    # --- 2. Sort Data for Chronological Rolling Calculations ---
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df.sort_values(by=['season_x', 'element', 'kickoff_time'], inplace=True)
    print("Data sorted by season, player, and kickoff_time for form calculations.")

    # --- 3. Define Columns and Windows for Form Calculation ---
    stats_to_process = [
        'xP', 'goals_scored', 'saves', 'penalties_saved', 'assists', 'xG', 'xA',
        'total_points', 'minutes', 'key_passes', 'npg', 'npxG', 'xGChain',
        'xGBuildup', 'own_goals', 'penalties_missed', 'clean_sheets', 'bps',
        'threat', 'ict_index', 'influence', 'creativity'
    ]
    understat_related_stats = [
        'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup'
    ]
    rolling_windows = [3, 5]
    all_form_cols = []
    verified_stats = [col for col in stats_to_process if col in df.columns]

    # --- 4. Calculate Form using rolling windows ---
    for window in rolling_windows:
        print(f"\n--- Calculating {window}-Gameweek Rolling Form Statistics ---")
        for col in verified_stats:
            form_col_name = f'form_{window}gw_{col}'  # Changed name for clarity
            all_form_cols.append(form_col_name)

            if col in understat_related_stats:
                # Conditional calculation: set stat to NaN where understat data is missing
                temp_stat_col = f'temp_{col}'
                df[temp_stat_col] = np.where(df['understat_missing'] == 1, np.nan, df[col])
                # Rolling mean automatically ignores NaNs
                rolling_avg = df.groupby(['season_x', 'element'])[temp_stat_col].rolling(window=window,
                                                                                         min_periods=1).mean()
                df[form_col_name] = rolling_avg.reset_index(level=[0, 1], drop=True)
                df.drop(columns=[temp_stat_col], inplace=True)
            else:
                # Standard rolling average
                rolling_avg = df.groupby(['season_x', 'element'])[col].rolling(window=window, min_periods=1).mean()
                df[form_col_name] = rolling_avg.reset_index(level=[0, 1], drop=True)

    # --- 5. Clean up ---
    df[all_form_cols] = df[all_form_cols].fillna(0)
    print("\nFilled any potential NaN values in new form columns with 0.")

    print(f"Successfully calculated form stats. Final DataFrame shape: {df.shape}")
    return df


# <<< STEP 7: Refactored team_data.py >>>
def add_team_strength_and_form(player_df, base_url):
    """
    Adds team strength and form columns to the player data using in-memory DataFrames.

    This function uses kickoff_time to uniquely identify matches, correctly
    handling double gameweeks and providing more accurate stats.

    Args:
        player_df (pd.DataFrame): The main DataFrame with player and form stats.
        base_url (str): The base URL for fetching auxiliary data like the team list.

    Returns:
        pd.DataFrame: The DataFrame with added team strength and form columns.
    """
    print("\n--- Step 7: Adding Team Strength and Form ---")
    # --- 1. Load Auxiliary Dataset ---
    try:
        if 'kickoff_time' not in player_df.columns:
            raise ValueError("Input DataFrame must contain a 'kickoff_time' column.")
        player_df['kickoff_time'] = pd.to_datetime(player_df['kickoff_time'])

        team_list_url = f"{base_url}master_team_list.csv"
        team_list_df = pd.read_csv(team_list_url)
        print(f"Successfully loaded team list from '{team_list_url}'.")
    except Exception as e:
        print(f"An error occurred while preparing data: {e}")
        return player_df

    # --- 2. Map Team Names to Integer IDs ---
    team_list_df = team_list_df[['season', 'team', 'team_name']]
    team_list_df.rename(columns={'season': 'season_x', 'team_name': 'team_x'}, inplace=True)
    player_df = pd.merge(player_df, team_list_df, on=['season_x', 'team_x'], how='left')

    # --- 3. Calculate Match Points ---
    conditions = [
        (player_df['was_home'] & (player_df['team_h_score'] > player_df['team_a_score'])),
        (~player_df['was_home'] & (player_df['team_a_score'] > player_df['team_h_score'])),
        (player_df['team_h_score'] == player_df['team_a_score'])
    ]
    choices = [3, 3, 1]
    player_df['match_points'] = np.select(conditions, choices, default=0)

    # --- 4. Create and build the Team-Match DataFrame ---
    print("Creating and calculating all team-level statistics (per-match)...")
    team_match_df = player_df.groupby(['season_x', 'team', 'kickoff_time', 'GW']).agg(
        opponent_team=('opponent_team', 'first'),
        match_points=('match_points', 'first')
    ).reset_index()

    team_match_df.sort_values(by=['season_x', 'team', 'kickoff_time'], inplace=True)

    # Calculate cumulative points and form
    team_match_df['cumulative_points'] = team_match_df.groupby(['season_x', 'team'])['match_points'].cumsum()
    team_match_df['team_form'] = team_match_df.groupby(['season_x', 'team'])['match_points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # Calculate team_strength robustly for double gameweeks
    final_gw_stats = team_match_df.sort_values('kickoff_time').groupby(['season_x', 'team', 'GW']).last().reset_index()
    final_gw_stats['team_strength'] = final_gw_stats.groupby(['season_x', 'GW'])['cumulative_points'].rank(method='min',
                                                                                                           ascending=False)

    team_match_df = pd.merge(team_match_df, final_gw_stats[['season_x', 'team', 'GW', 'team_strength']],
                             on=['season_x', 'team', 'GW'], how='left')

    # Calculate opponent team strength
    strength_lookup_df = final_gw_stats[['season_x', 'GW', 'team', 'team_strength']].rename(
        columns={'team': 'opponent_team', 'team_strength': 'opp_team_strength'}
    )
    team_match_df = pd.merge(team_match_df, strength_lookup_df, on=['season_x', 'GW', 'opponent_team'], how='left')

    # Calculate upcoming fixture strength
    upcoming_strengths_df = pd.DataFrame(index=team_match_df.index)
    for i in range(1, 6):
        upcoming_strengths_df[f'next_opp_strength_{i}'] = team_match_df.groupby(['season_x', 'team'])[
            'opp_team_strength'].shift(-i)
    team_match_df['upcoming_strength'] = upcoming_strengths_df.mean(axis=1)
    team_match_df['upcoming_strength'] = team_match_df.groupby(['season_x', 'team'])['upcoming_strength'].ffill()

    # --- 5. Perform a Single, Final Merge ---
    print("Merging all team-level stats back to the player dataset...")
    cols_to_merge = ['season_x', 'kickoff_time', 'team', 'team_strength', 'team_form', 'opp_team_strength',
                     'upcoming_strength']
    final_df = pd.merge(player_df, team_match_df[cols_to_merge], on=['season_x', 'kickoff_time', 'team'], how='left')
    final_df.drop(columns=['match_points'], inplace=True, errors='ignore')

    # --- 6. Validation Step ---
    if len(final_df) > len(player_df):
        print(
            f"CRITICAL WARNING: Row count increased from {len(player_df)} to {len(final_df)}. Duplicates may have been created.")
    else:
        print(f"Row count validation passed. Final row count is {len(final_df)}.")

    print(f"Successfully added team data. Final DataFrame shape: {final_df.shape}")
    return final_df


# <<< STEP 8: Refactored reorder_and_add_labels.py >>>

# --- Helper functions for DGW processing ---
def _create_dgw_row(group: pd.DataFrame) -> pd.Series:
    """Helper function to process a single player's DGW into one row."""
    group = group.sort_values('kickoff_time', ascending=True)
    first_match = group.iloc[0]
    second_match = group.iloc[-1]
    new_row = {}
    sum_cols = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
        'goals_scored', 'ict_index', 'influence', 'minutes', 'own_goals',
        'penalties_missed', 'penalties_saved', 'red_cards', 'saves',
        'team_a_score', 'team_h_score', 'threat', 'total_points', 'yellow_cards',
        'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup', 'xP'
    ]
    first_match_cols = [
        'season_x', 'name', 'position', 'team_x', 'element', 'fixture', 'kickoff_time',
        'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home',
        'round', 'GW', 'team', 'team_form', 'team_strength', 'upcoming_strength'
    ]
    # All other columns are taken from the second match by default
    all_cols = set(group.columns)
    second_match_cols = list(all_cols - set(sum_cols) - set(first_match_cols))

    for col in sum_cols:
        if col in group.columns: new_row[col] = group[col].sum()
    for col in first_match_cols:
        if col in group.columns: new_row[col] = first_match[col]
    for col in second_match_cols:
        if col in group.columns: new_row[col] = second_match[col]

    new_row['opponent_team_1'] = first_match['opponent_team']
    new_row['opp_team_strength_1'] = first_match['opp_team_strength']
    new_row['opponent_team_2'] = second_match['opponent_team']
    new_row['opp_team_strength_2'] = second_match['opp_team_strength']
    new_row['is_dgw'] = 1
    return pd.Series(new_row)


def combine_dgw(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidates DGW rows in a DataFrame."""
    df = df.copy()
    dgw_identifiers = ['season_x', 'element', 'GW']
    dgw_mask = df.duplicated(subset=dgw_identifiers, keep=False)
    sgw_df = df[~dgw_mask].copy()
    dgw_df = df[dgw_mask].copy()

    sgw_df['is_dgw'] = 0
    sgw_df.rename(columns={'opponent_team': 'opponent_team_1', 'opp_team_strength': 'opp_team_strength_1'},
                  inplace=True)
    sgw_df['opponent_team_2'] = np.nan
    sgw_df['opp_team_strength_2'] = np.nan

    if not dgw_df.empty:
        processed_dgw_df = dgw_df.groupby(dgw_identifiers, as_index=False).apply(_create_dgw_row)
        final_df = pd.concat([sgw_df, processed_dgw_df], ignore_index=True, sort=False)
    else:
        final_df = sgw_df
    return final_df


# --- Main processing function for this step ---
def reorder_and_add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes FPL data by consolidating DGWs, creating future labels and flags,
    and reordering columns for modeling.

    Args:
        df: The DataFrame with all previously added features.

    Returns:
        The final, processed DataFrame ready for modeling.
    """
    print("\n--- Step 8: Consolidating DGWs, Adding Labels, and Reordering ---")
    PLANNING_HORIZON = 6

    # 1. Consolidate Double Gameweeks
    df = combine_dgw(df)
    print("Consolidated double gameweeks.")

    # 2. Filter out old seasons
    seasons_to_remove = ['2016-17', '2017-18']
    df = df[~df['season_x'].isin(seasons_to_remove)]
    print(f"Removed seasons {seasons_to_remove}. Remaining rows: {len(df)}.")

    # 3. Create complete timelines for each player to ensure accurate shifts
    df.sort_values(['season_x', 'element', 'GW'], inplace=True)

    def create_complete_timeline(group):
        full_gw_range = pd.RangeIndex(start=1, stop=38 + 1, name='GW')
        group = group.set_index('GW').reindex(full_gw_range)
        group['element'].fillna(method='ffill', inplace=True)
        group['season_x'].fillna(method='ffill', inplace=True)
        group['is_dgw'].fillna(0, inplace=True)
        return group.reset_index()

    df = df.groupby(['season_x', 'element'], group_keys=False).apply(create_complete_timeline)

    # 4. Create future DGW flags and target labels
    grouped = df.groupby(['season_x', 'element'])
    new_cols_data = {}
    label_cols = []
    dgw_flag_cols = []

    for i in range(1, PLANNING_HORIZON + 1):
        # DGW flags
        dgw_col = f'is_dgw_in_gw+{i}'
        new_cols_data[dgw_col] = grouped['is_dgw'].shift(-i).fillna(0).astype(int)
        dgw_flag_cols.append(dgw_col)
        # Target labels
        points_col = f'points_gw+{i}'
        minutes_col = f'minutes_gw+{i}'
        new_cols_data[points_col] = grouped['total_points'].shift(-i)
        new_cols_data[minutes_col] = grouped['minutes'].shift(-i)
        label_cols.extend([points_col, minutes_col])

    labels_df = pd.DataFrame(new_cols_data, index=df.index)
    df = pd.concat([df, labels_df], axis=1)
    print(f"Created target labels and DGW flags for a {PLANNING_HORIZON}-week horizon.")

    # 5. Clean up and reorder
    df.dropna(subset=['fixture'], inplace=True)  # Remove placeholder rows
    print("Removed placeholder rows used for label creation.")

    categorical_cols = ['team', 'GW', 'was_home', 'round', 'opponent_team_1', 'opponent_team_2', 'element', 'season_x',
                        'name', 'position', 'team_x', 'fixture']
    final_ordered_list = label_cols + dgw_flag_cols + ['is_dgw'] + categorical_cols

    # Add all remaining columns
    for col in df.columns:
        if col not in final_ordered_list:
            final_ordered_list.append(col)

    df = df[final_ordered_list]
    print("Successfully reordered all columns.")

    print(f"Final processing complete. DataFrame shape: {df.shape}")
    return df


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting FPL Data Pipeline ---")

    # --- Phase 1: Data Ingestion ---
    print("\n--- Phase 1: Running Data Ingestion ---")
    ingested_df = asyncio.run(load_and_process_data(BASE_URL))

    if not ingested_df.empty:
        print(f"--- Data Ingestion Complete. Shape: {ingested_df.shape} ---")

        # --- ADDED: Save pre-aggregation data for review as requested ---
        try:
            pre_agg_filename = "../../backend/pre_aggregation_data.csv"
            ingested_df.to_csv(pre_agg_filename, index=False)
            print(f"\n--- PIPELINE CHECKPOINT ---")
            print(f"Successfully saved data before aggregation to '{pre_agg_filename}' for review.")
        except Exception as e:
            print(f"\nError saving pre-aggregation CSV file: {e}")

        # --- Phase 2: Feature Engineering ---
        print("\n--- Phase 2: Running Feature Engineering ---")

        # Step 5: Calculate Aggregate Stats
        df_with_aggregates = calculate_aggregate_stats(ingested_df)

        # Step 6: Calculate Rolling Form Stats
        df_with_form = calculate_form_stats(df_with_aggregates)

        # Step 7: Add Team Strength and Form
        df_with_team_data = add_team_strength_and_form(df_with_form, BASE_URL)

        # Step 8: Reorder columns and add target labels
        final_features_df = reorder_and_add_labels(df_with_team_data)

        try:
            # Save the final result of the feature engineering pipeline
            output_filename = "../../backend/final_features_for_modeling.csv"
            final_features_df.to_csv(output_filename, index=False)
            print(f"\n--- PIPELINE CHECKPOINT ---")
            print(f"Successfully saved final processed data to '{output_filename}' for review.")
            print(f"Final shape before modeling: {final_features_df.shape}")

        except Exception as e:
            print(f"\nError saving final CSV file: {e}")
    else:
        print("\nData ingestion resulted in an empty DataFrame. Halting pipeline.")

    print("\n--- All Feature Engineering Steps Complete ---")
