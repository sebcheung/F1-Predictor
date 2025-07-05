import requests
import pandas as pd
import time
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Collects F1 data using Ergast API and additional sources
class F1DataCollector: 
    def __init__(self):
        self.base_url = "https://api.jolpi.ca/ergast/f1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent' : 'F1-ML-Predictor/1.0'
        })

    # Retrieves the list of seasons to analyze the data
    def get_seasons(self, start_year: int = 2010, end_year: Optional[int] = None) -> List[int]:
        if end_year is None:
            end_year = datetime.now().year
        return list(range(start_year, end_year + 1))
    
    # Retrieves the race schedule info for a specific year
    def get_race_schedule(self, year: int) -> pd.DataFrame:
        url = f"{self.base_url}/{year}.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            races = []
            for race in data['MRData']['RaceTable']['Races']:
                race_info = {
                    'year': year,
                    'round': int(race['round']),
                    'race_name': race['raceName'],
                    'circuit_id': race['Circuit']['circuitId'],
                    'circuit_name': race['Circuit']['circuitName'],
                    'country': race['Circuit']['Location']['country'],
                    'date': race['date'],
                    'time': race.get('time', 'N/A')
                }
                races.append(race_info)

            return pd.DataFrame(races)
        
        except Exception as e:
            logger.error(f"Error fetching race schedule for {year}: {e}")
            return pd.DataFrame()
        
    # Retrieve qualifying results for a specific race
    def get_qualifying_results(self, year: int, round_num: int) -> pd.DataFrame:
        url = f"{self.base_url}/{year}/{round_num}/qualifying.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            qualifying = []
            for result in data['MRData']['RaceTable']['Races'][0]['QualifyingResults']:
                qual_info = {
                        'year': year,
                        'round': round_num,
                        'driver_id': result['Driver']['driverId'],
                        'driver_name': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                        'constructor_id': result['Constructor']['constructorId'],
                        'constructor_name': result['Constructor']['name'],
                        'position': int(result['position']),
                        'q1': result.get('Q1', 'N/A'),
                        'q2': result.get('Q2', 'N/A'),
                        'q3': result.get('Q3', 'N/A')
                }
                qualifying.append(qual_info)

            return pd.DataFrame(qualifying)
        
        except Exception as e:
            logger.error(f"Error fetching qualifying for {year} round {round_num}: {e}")
            return pd.DataFrame()
        
    # Retrieve race results for a specific race
    def get_race_results(self, year: int, round_num: int) -> pd.DataFrame:
        url = f"{self.base_url}/{year}/{round_num}/results.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data['MRData']['RaceTable']['Races'][0]['Results']:
                result_info = {
                    'year': year,
                    'round': round_num,
                    'driver_id': result['Driver']['driverId'],
                    'driver_name': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                    'constructor_id': result['Constructor']['constructorId'],
                    'constructor_name': result['Constructor']['name'],
                    'grid': int(result['grid']),
                    'position': result['position'],
                    'points': float(result['points']),
                    'laps': int(result['laps']),
                    'status': result['status'],
                    'time': result.get('Time', {}).get('time', 'N/A'),
                    'fastest_lap': result.get('FastestLap', {}).get('Time', {}).get('time', 'N/A')
                }
                results.append(result_info)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error fetching race results for {year} round {round_num}: {e}")
            return pd.DataFrame()
        
    # Retrieve driver championship standings for a year
    def get_driver_standings(self, year: int) -> pd.DataFrame:
        url = f"{self.base_url}/{year}/driverstandings.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            standings = []
            for standing in data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
                standing_info = {
                    'year': year,
                    'driver_id': standing['Driver']['driverId'],
                    'driver_name': f"{standing['Driver']['givenName']} {standing['Driver']['familyName']}",
                    'constructor_id': standing['Constructors'][0]['constructorId'],
                    'constructor_name': standing['Constructors'][0]['name'],
                    'position': int(standing['position']),
                    'points': float(standing['points']),
                    'wins': int(standing['wins'])
                }
                standings.append(standing_info)

            return pd.DataFrame(standings)
        
        except Exception as e:
            logger.error(f"Error fetching driver standings for {year}: {e}")
            return pd.DataFrame()
        
    # Retrieve constructor championship standings for a year
    def get_constructor_standings(self, year: int) -> pd.DataFrame:
        url = f"{self.base_url}/{year}/constructorstandings.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            standings = []
            for standing in data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
                standing_info = {
                    'year': year,
                    'constructor_id': standing['Constructor']['constructorId'],
                    'constructor_name': standing['Constructor']['name'],
                    'position': int(standing['position']),
                    'points': float(standing['points']),
                    'wins': int(standing['wins'])
                }
                standings.append(standing_info)
            
            return pd.DataFrame(standings)
            
        except Exception as e:
            logger.error(f"Error fetching constructor standings for {year}: {e}")
            return pd.DataFrame()
        
    # Retrieve F1 data for multiple years, which returns a dictionary with different data types
    def collect_comprehensive_data(self, years: List[int]) -> Dict[str, pd.DataFrame]:
        all_data = {
            'race_schedule': [],
            'qualifying_results': [],
            'race_results': [],
            'driver_standings': [],
            'constructor_standings': []
        }

        for year in years:
            logger.info(f"Collecting data for {year}")

            # Retrieve race schedule
            schedule = self.get_race_schedule(year)
            if not schedule.empty:
                all_data['race_schedule'].append(schedule)

            # Retrieve driver and constructor standings
            driver_standings = self.get_driver_standings(year)
            if not driver_standings.empty:
                all_data['driver_standings'].append(driver_standings) 

            constructor_standings = self.get_constructor_standings(year)
            if not constructor_standings.empty:
                all_data['constructor_standings'].append(constructor_standings)

            # Get race and qualifying results for each round
            if not schedule.empty:
                for _, race in schedule.iterrows():
                    round_num = race['round']
                    logger.info(f"Collecting race {round_num}: {race['race_name']}")

                    # Qualifying results
                    qualifying = self.get_qualifying_results(year, round_num)
                    if not qualifying.empty:
                        all_data['qualifying_results'].append(qualifying)

                    # Race results
                    results = self.get_race_results(year, round_num)
                    if not results.empty:
                        all_data['race_results'].append(results)

                    # Rate limiting to cool off API
                    time.sleep(5)

            # Combine all dataframes
            final_data = {}
            for key, df_list in all_data.items():
                if df_list:
                    final_data[key] = pd.concat(df_list, ignore_index=True)
                    logger.info(f"Combined {key}: {len(final_data[key])} records")
                else:
                    final_data[key] = pd.DataFrame()
        return final_data
        
    # Save collected data into a CSV file(s)
    def save_data(self, data: Dict[str, pd.DataFrame], base_path: str = "data"):
        import os
        os.makedirs(base_path, exist_ok=True)

        for key, df in data.items():
            if not df.empty:
                filepath = os.path.join(base_path, f"{key}.csv")
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {key} to {filepath}")


if __name__ == "__main__":
    collector = F1DataCollector()

    # TODO: testing, can change
    years = [2022, 2023, 2024, 2025]

    # Collect the data
    logger.info("Starting F1 data collection...")
    data = collector.collect_comprehensive_data(years)

    # Save the data
    collector.save_data(data)

    # Summary
    print("\n=== Data Collection Summary ===")
    for key, df in data.items():
        if not df.empty:
            print(f"{key}: {len(df)} records")
            print(f"Columns: {list(df.columns)}")
        else:
            print(f"{key}: No data collected")
        print()