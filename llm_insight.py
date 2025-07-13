import pandas as pd
from openai import OpenAI
import numpy as np
import os
import json
import xgboost as xgb
from joblib import load
from typing import List
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class F1LLMExplainer:
    def __init__(self, data_dir="data", model_path="models/xgb_race_points_model.joblib"):
        print("Loading model and data...")
        self.data_dir = data_dir
        self.model = xgb.XGBRegressor()
        self.model = load(model_path)
        self.features = pd.read_csv(os.path.join(data_dir, "f1_features.csv"))
        self.X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
        self.meta_data = pd.read_csv(os.path.join(data_dir, "metadata_val.csv")) if os.path.exists(os.path.join(data_dir, "metadata_val.csv")) else None
            
    # Search for the historical data of queried driver at specific circuit
    def query_driver_performance(self, driver_name: str, circuit_name: str):
        df = self.features
        match = df[
            (df["driver_name"].str.lower() == driver_name.lower()) &
            (df["circuit_name"].str.lower() == circuit_name.lower())
        ]
        print(f"Found {len(match)} rows for {driver_name} at {circuit_name}")
        return match
    
    # Finds the top N important features for a prediction
    def rank_feature_importance(self, X: pd.DataFrame, top_n=5):
        importance = self.model.feature_importances_
        feature_names = X.columns.tolist()
        top_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        return top_features
    
    # Generates a NL summary via LLM based on the stats
    def generate_response(self, driver: str, circuit: str):
        rows = self.query_driver_performance(driver, circuit)

        if rows.empty:
            return f"No data found for {driver} at {circuit}"

        avg_points = rows['race_points_finish'].mean()
        podiums = rows['race_podium'].sum()
        wins = rows['race_win'].sum()
        starts = len(rows)

        facts = f"""
        Driver: {driver}
        Circuit: {circuit}
        Total Races: {starts}
        Average Points: {avg_points:.2f}
        Podiums: {podiums}
        Wins: {wins}
        """

        prompt = f"""
        Given the following stats, explain in detail the driver's past performace at the track and what influences success. 
        
        {facts}

        Then answer: What are the top 3 factors contributing to the driver's success at this circuit?
        """

        print("Querying LLM...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are the best motorsport analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    
    # Allows the user to ask a question about a driver and circuit with context
    def ask_custom_question(self, question: str, context: str = "") -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are the best motorsport analyst."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
                ],
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return "[Error generating response]"

if __name__ == "__main__":
    explainer = F1LLMExplainer()

    print("\n=== F1 Race Insight Generator ===")
    driver = input("Enter driver name (e.g., Max Verstappen): ").strip()
    circuit = input("Enter circuit name (e.g., Silverstone Circuit): ").strip()

    custom_question = input("Optional: Ask your own question about this combo (or press enter to use default): ").strip()

    if not custom_question:
        output = explainer.generate_response(driver, circuit)
    else:
        # Load context from driver-circuit stats
        rows = explainer.query_driver_performance(driver, circuit)

        if rows.empty:
            print(f"No data found for {driver} at {circuit}")
        else:
            avg_points = rows['race_points'].mean()
            podiums = rows['race_podium'].sum()
            wins = rows['race_win'].sum()
            starts = len(rows)

            context = f"""
            Driver: {driver}
            Circuit: {circuit}
            Total Races: {starts}
            Average Points: {avg_points:.2f}
            Podiums: {podiums}
            Wins: {wins}
            """
            response = explainer.ask_custom_question(custom_question, context)
            output = response

    print("\n=== LLM Insight ===\n")
    print(output)
