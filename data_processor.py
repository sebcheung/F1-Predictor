import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Feature engineering pipeline for F1 race prediction, creating
# ML ready features from raw data
class F1FeatureEngineer:
    def __init__(self):
        self.driver_stats = {}
        self.constructor_stats = {}
        self.track_stats = {}
    
    # Load preprocessed F1 data
    def load_data(self, data_path: str = "data") -> Dict[str, pd.DataFrame]:
        import os

        data: Dict[str, pd.DataFrame] = {}
        for name in [
            "race_schedule",
            "qualifying_results",
            "race_results",
            "driver_standings",
            "constructor_standings",
        ]:
            path = os.path.join(data_path, f"{name}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                data[name] = df
                print(f"Loaded {name}: {len(df)} records")
        return data

    # Create driver-specific features based on historical performance
    def create_driver_features(self, race_results: pd.DataFrame, qualifying_results: pd.DataFrame) -> pd.DataFrame:
        # First sort by year and round to ensure chronological order
        race_results = race_results.sort_values(['year', 'round'])
        qualifying_results = qualifying_results.sort_values(['year', 'round'])

        driver_features = []
        drivers = race_results['driver_id'].unique()

        for driver in drivers:
            driver_races = race_results[race_results['driver_id'] == driver].copy()
            driver_quals = qualifying_results[qualifying_results['driver_id'] == driver].copy()

            for _, race in driver_races.iterrows():
                # Historical performance
                historical_races = driver_races[
                    (driver_races['year'] < race['year']) | 
                    ((driver_races['year'] == race['year']) & (driver_races['round'] < race['round']))
                ]
                historical_quals = driver_quals[
                    (driver_quals['year'] < race['year']) | 
                    ((driver_quals['year'] == race['year']) & (driver_quals['round'] < race['round']))
                ]

                # Current performance, last 5 races
                recent_races = historical_races.tail(5)
                recent_quals = historical_quals.tail(5)

                # Helper methods for reused code
                to_pos = lambda s: float(s) if str(s).isdigit() else 20.0
                podium = lambda df: df[df["position"].isin(["1", "2", "3"])]

                features = {
                    'year': race['year'],
                    'round': race['round'],
                    'driver_id': driver,
                    'driver_name': race['driver_name'],
                    'constructor_id': race['constructor_id'],

                    # Career stats
                    'driver_career_races': len(historical_races),
                    'driver_career_wins': len(historical_races[historical_races['position'] == '1']),
                    'driver_career_podiums': len(historical_races[historical_races['position'].isin(['1','2','3'])]),
                    'driver_career_points': historical_races['points'].sum(),
                    'driver_career_avg_pos': historical_races['position'].apply(to_pos).mean() 
                    if len(historical_races) > 0 
                    else 20,

                    # Recent stats
                    'driver_recent_avg_pos': recent_races['position'].apply(to_pos).mean() 
                    if len(recent_races) > 0 
                    else 20,
                    'driver_recent_points': recent_races['points'].sum(),
                    'driver_recent_wins': len(recent_races[recent_races['position'] == 1]),
                    'driver_recent_podiums': len(podium(recent_races)),

                    # Qualifying stats
                    'driver_avg_qual_pos': recent_quals['position'].apply(to_pos).mean() 
                    if len(recent_quals) > 0 
                    else 20,
                    'driver_qual_improvement': (
                        recent_quals['position'].apply(to_pos).mean() - driver_quals['position'].apply(to_pos).mean()
                    ) if len(driver_quals) > 0 and len(recent_quals) > 0 else 0,

                    # Track-specific performance
                    'circuit_id': race.get("circuit_id", "unknown"),

                    # Targets
                    'race_position': to_pos(race['position']),
                    'race_points': race['points'],
                    'race_win': 1 if race['position'] == '1' else 0,
                    'race_podium': 1 if race['position'] in ['1','2','3'] else 0,
                    'race_points_finish': 1 if race['points'] > 0 else 0
                }

                driver_features.append(features)
        
        return pd.DataFrame(driver_features)
    
    # Create constructor/team features
    def create_constructor_features(self, race_results: pd.DataFrame) -> pd.DataFrame:
        constructor_features = []
        constructors = race_results['constructor_id'].unique()

        for constructor in constructors:
            constructor_races = race_results[race_results['constructor_id'] == constructor].copy().sort_values(['year','round'])

            for year in constructor_races['year'].unique():
                for round_num in constructor_races[constructor_races['year'] == year]['round'].unique():
                    # Historical performance
                    historical_races = constructor_races[
                        (constructor_races['year'] < year) |
                        ((constructor_races['year'] == year) & (constructor_races['round'] < round_num))
                    ]

                    # Recent races (last 10)
                    recent_races = historical_races.tail(10)

                    # Helper methods
                    to_pos = lambda s: float(s) if str(s).isdigit() else 20.0
                    podium = lambda df: df[df["position"].isin(["1", "2", "3"])]

                    features = {
                        'year': year,
                        'round': round_num,
                        'constructor_id': constructor,

                        # Constructor performance
                        'constructor_career_wins': len(historical_races[historical_races['position'] == '1']),
                        'constructor_career_podiums': len(podium(historical_races)),
                        'constructor_career_points': historical_races['points'].sum(),
                        'constructor_avg_pos': historical_races['position'].apply(to_pos).mean() 
                        if len(historical_races) > 0 
                        else 20,

                        # Recent performance
                        'constructor_recent_avg_pos': historical_races['position'].apply(to_pos).mean() 
                        if len(recent_races) > 0 
                        else 20,
                        'constructor_recent_points': recent_races['points'].sum(),
                        'constructor_recent_wins': len(recent_races[recent_races['position'] == '1']),
                        'constructor_recent_podiums': len(podium(recent_races)),
                    }

                    constructor_features.append(features)
        return pd.DataFrame(constructor_features)
    
    # Create track-specific features
    def create_track_features(self, race_results: pd.DataFrame, race_schedule: pd.DataFrame) -> pd.DataFrame:
        # Add circuit info to race results
        race_results_with_circuit = race_results.merge(
            race_schedule[['year','round','circuit_id','circuit_name','country']],
            on=['year','round'],
            how='left'
        )

        track_features = []
        circuits = race_results_with_circuit['circuit_id'].dropna().unique()

        for circuit in circuits:
            circuit_races = race_results_with_circuit[race_results_with_circuit['circuit_id'] == circuit].sort_values(['year', 'round'])

            # iterate over every (year, round) the circuit hosted
            for _, race_row in circuit_races.iterrows():
                year      = int(race_row['year'])
                round_num = int(race_row['round'])

                # all earlier races at the SAME circuit
                historical = circuit_races[
                    (circuit_races['year'] < year) |
                    ((circuit_races['year'] == year) &
                    (circuit_races['round'] < round_num))
                ]

                # Helper method
                to_pos = lambda s: float(s) if str(s).isdigit() else 20.0
                
                overtaking = 0.0
                if len(historical):
                    overtaking = (
                        historical.groupby(["year", "round"])["grid"]
                        .apply(
                            lambda g: (g - historical.loc[g.index, "position"].apply(to_pos))
                            .abs()
                            .mean()
                        ).mean()
                    )

                features = {
                    "year": year,
                    "round": round_num,
                    "circuit_id": circuit,
                    "circuit_name": race_row["circuit_name"],
                    "country": race_row["country"],
                    "track_total_races": len(historical),
                    "track_avg_winner_pos": historical.loc[historical["position"] == "1", "grid"].mean()
                    if len(historical)
                    else 1.0,
                    "track_overtaking_difficulty": overtaking,
                    "track_repeat_winners": historical.loc[historical["position"] == "1", "driver_id"].value_counts().nunique()
                    if len(historical)
                    else 0,
                }
                track_features.append(features)

        return pd.DataFrame(track_features)
        
    # Create features based on race context (grid position, weather, etc)
    def create_race_context_features(self, race_results: pd.DataFrame, qualifyiing_results: pd.DataFrame) -> pd.DataFrame:
        race_context = []

        # Group by year and round to get race-specific context data
        for (year, round_num), race_group in race_results.groupby(['year','round']):
            qual_group = qualifyiing_results[
                (qualifyiing_results['year'] == year) &
                (qualifyiing_results['round'] == round_num)
            ]

            for _, driver_result in race_group.iterrows():
                num_pos = 20
                # Find corresponding qualifying result
                if not qual_group.empty:
                    driver_qual = qual_group[qual_group['driver_id'] == driver_result['driver_id']]
                    if not driver_qual.empty:
                        num_pos = float(driver_qual['position'].iloc[0]) if str(driver_qual['position'].iloc[0]).isdigit() else 20

                # Helper method
                to_pos = lambda s: float(s) if str(s).isdigit() else 20.0

                features = {
                    'year': year,
                    'round': round_num,
                    'driver_id': driver_result['driver_id'],

                    # Grid position features
                    'grid_position': driver_result['grid'],
                    'grid_position_normalized': driver_result['grid'] / 20.0,
                    'front_row_start': 1 if driver_result['grid'] <= 2 else 0,
                    'top_5_start': 1 if driver_result['grid'] <= 5 else 0,
                    'top_10_start': 1 if driver_result['grid'] <= 10 else 0,

                    # Qualifying performance
                    'qual_position': num_pos,
                    'qual_to_grid_diff': num_pos - driver_result['grid'],

                    # Race comp context
                    'field_size': len(race_group),
                    'points_available': 1 if driver_result['grid'] <= 10 else 0,

                    # Targets
                    'race_position': to_pos(driver_result['position']),
                    'race_points': driver_result['points'],
                    'position_change': driver_result['grid'] - to_pos(driver_result['position']),
                    'race_win': 1 if driver_result['position'] == '1' else 0,
                    'race_podium': 1 if driver_result['position'] in ['1','2','3'] else 0,
                    'points_finish': 1 if driver_result['points'] > 0 else 0,
                }

                race_context.append(features)

        return pd.DataFrame(race_context)
        
    # Combine all feature sets into one dataset
    def combine_all_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        print("Creating driver features...")
        driver_features = self.create_driver_features(
            data['race_results'],
            data['qualifying_results']
        )

        print("Creating constructor features...")
        constructor_features = self.create_constructor_features(data['race_results'])

        print("Creating track features...")
        track_features = self.create_track_features(
            data['race_results'],
            data['race_schedule']
        )

        print("Creating race context features...")
        race_context = self.create_race_context_features(
            data['race_results'],
            data['qualifying_results']
        )

        # Ensure consistent data types for merge keys
        for df in [driver_features, constructor_features, track_features, race_context]:
            df['year'] = df['year'].astype(int)
            df['round'] = df['round'].astype(int)
            if 'constructor_id' in df.columns:
                df['constructor_id'] = df['constructor_id'].astype(str)
            if 'driver_id' in df.columns:
                df['driver_id'] = df['driver_id'].astype(str)

        # Merge all of the features
        print("Merging all features...")
        final_features = (driver_features
            .merge(constructor_features, on = ['year', 'round', 'constructor_id'], how = 'left')
            .merge(track_features, on = ['year', 'round'], how = 'left')
            .merge(race_context, on = ['year', 'round', 'driver_id'], how = 'left')
        )

        # Fill any missing value (if any)
        final_features = final_features.fillna(0)

        print(f"Final dataset shape: {final_features.shape}")
        print(f"Features created: {final_features.columns.tolist()}")
        return final_features
    
    # Prepare training and validation datasets
    def prepare_datasets(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame | list]:
        # Resolve any _x/_y conflicts generated during merge
        cols_rename = {}
        for col in features_df.columns:
            if col.endswith("_x"):
                cols_rename[col] = col[:-2]  # drop _x
        features_df.rename(columns=cols_rename, inplace=True)
        features_df.drop(columns=[c for c in features_df.columns if c.endswith("_y")], inplace=True)
        
        # Clean - remove rows with missing target values
        features_df = features_df.dropna(subset=['race_position', 'race_points'])

        # Feature selection for our ML Model
        feature_cols = [
            # Driver features
            'driver_career_races', 'driver_career_wins', 'driver_career_podiums',
            'driver_career_points', 'driver_career_avg_pos', 'driver_recent_avg_pos',
            'driver_recent_points', 'driver_recent_wins', 'driver_recent_podiums',
            'driver_avg_qual_pos', 'driver_qual_improvement',
            
            # Constructor features
            'constructor_career_wins', 'constructor_career_podiums', 
            'constructor_career_points', 'constructor_avg_pos',
            'constructor_recent_avg_pos', 'constructor_recent_points',
            'constructor_recent_wins', 'constructor_recent_podiums',
            
            # Track features
            'track_total_races', 'track_avg_winner_pos', 'track_overtaking_difficulty',
            'track_repeat_winners',
            
            # Race context
            'grid_position', 'grid_position_normalized', 'front_row_start',
            'top_5_start', 'top_10_start', 'qual_position', 'qual_to_grid_diff',
            'field_size', 'points_available'
        ]

        target_cols = [
            'race_points', 'race_win', 'race_podium', 'points_finish'
        ]

        # Create feature matrix
        X = features_df[feature_cols].copy()
        y = features_df[target_cols].copy()

        # Add metadata
        meta_data = features_df[['year', 'round', 'driver_id', 'driver_name', 'constructor_id']].copy()

        # Split by year (older years will be used for training)
        train_years = sorted(features_df["year"].unique())[:-1]  # leave latest season for val
        train_mask = features_df["year"].isin(train_years)

        split = {
            "X_train": X[train_mask],
            "y_train": y[train_mask],
            "X_val": X[~train_mask],
            "y_val": y[~train_mask],
            "feature_names": feature_cols,
            "target_names": target_cols,
            "metadata_train": meta_data[train_mask],
            "metadata_val": meta_data[~train_mask],
        }

        print(f"Training set: {split['X_train'].shape}")
        print(f"Validation set: {split['X_val'].shape}")
        print(f"Features: {X.columns.tolist()}")

        return split

if __name__ == "__main__":
    fe = F1FeatureEngineer()

    # Load data
    data = fe.load_data()

    # Create features
    if data:
        features = fe.combine_all_features(data)
        
        # Save features
        features.to_csv('data/f1_features.csv', index=False)
        print("Features saved to data/f1_features.csv")
        
        # Prepare datasets
        ml_data = fe.prepare_datasets(features)
        
        # Save datasets
        ml_data['X_train'].to_csv('data/X_train.csv', index=False)
        ml_data['y_train'].to_csv('data/y_train.csv', index=False)
        ml_data['X_val'].to_csv('data/X_val.csv', index=False)
        ml_data['y_val'].to_csv('data/y_val.csv', index=False)
        
        print("ML datasets saved!")
        print(f"Training features shape: {ml_data['X_train'].shape}")
        print(f"Validation features shape: {ml_data['X_val'].shape}")
    else:
        print("No data found. Please run the data collector first.")