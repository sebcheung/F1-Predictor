# 🏎️ F1 Race Outcome Predictor & LLM Explainer

This project combines traditional machine learning (XGBoost) with Large Language Models (LLMs) to **predict Formula 1 race outcomes** and **explain predictions in natural language**. It provides race-level insights using historical performance data, driver stats, constructor trends, and track characteristics.

---

## Features
- ML model with XGBoost trained on real live F1 data
- Makes predictions:
  - Race points
  - Podium finishes
  - Wins
- Data-set: drivers, constructors, tracks, grids, and qualifying context (i.e weather)
- OpenAI LLM integration with natural language responses for explanations

---

## Setup
### 1. Git Clone Repo
```bash
git clone https://github.com/yourusername/F1-Predictor.git
cd F1-Predictor
```

### 2. Set Up Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install requirements.txt
```

### 4. Add OpenAI Key (Optional)
If you want LLM explanations:
  1. Get an OpenAI API key from: https://platform.openai.com/account/api-keys
  2. Create a .env file in the root:
     ```bash
     OPENAI_API_KEY=sk-...
     ```
---

## How It Works
### Step 1: Data Collection
Run:
```bash
python data_collection.py
```
This fetches F1 data from the Jolpi API (successor of Ergast API) that consists of race results, qualifying results, standings, schedules, etc. It will save them as CSVs.

### Step 2: Feature Engineering
Run:
```bash
python data_processor.py
```
This builds features for drivers, constructors, circuits, and race context. It will merge all the datasets into one while handling duplicates, missing values, and merge conflicts.

### Step 3: Model Training
Run:
```bash
python model_trainer.py
```
This trains on XGBoost model to predict the following:
- Race points, race wins, race podiums, and point finishes
and prints evaluation metrics.

### Step 4: LLM Insights (Optional)
If you have an OpenAI key, run:
```bash
python llm_insights.py
```
You can also ask: What are Verstappen's typical performances at Silverstone?
The LLM will summarize based on matching driver, track, and historical context - backed by ML predictions.

