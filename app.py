from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import os
import shutil
import re
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "model.pkl"
DATA_PATH = "dataset.csv"

model = None
feature_columns = []


def extract_features_from_files(temp_dir):
    features = {}

    # --- 01_info.csv ---
    info = pd.read_csv(os.path.join(temp_dir, "01_info.csv"))
    ah_str = info[info['0'] == 'Pre-game AH']['1'].values[0]
    ou_str = info[info['0'] == 'Pre-game O/U']['1'].values[0]

    parts_ah = ah_str.split('/')
    home_ah_odds = float(parts_ah[0].strip().split()[-1])
    handicap_text = parts_ah[1].strip()
    away_ah_odds = float(parts_ah[2].strip().split()[-1])

    parts_ou = ou_str.split('/')
    over_odds = float(parts_ou[0].strip().split()[-1])
    ou_line = float(parts_ou[1].strip())
    under_odds = float(parts_ou[2].strip().split()[-1])

    features['ah_home_odds'] = home_ah_odds
    features['ah_away_odds'] = away_ah_odds
    features['ou_line'] = ou_line
    features['over_odds'] = over_odds
    features['under_odds'] = under_odds

    handicap_val = float(handicap_text.replace('-', '').replace('+', '').split('/')[0])
    if handicap_text.startswith('-'):
        features['handicap'] = -handicap_val
    else:
        features['handicap'] = handicap_val

    # --- 05_recent_stats.csv ---
    stats = pd.read_csv(os.path.join(temp_dir, "05_recent_stats.csv"))
    home_stats = stats[['Metric', 'Home_Last10']].set_index('Metric').T
    away_stats = stats[['Metric', 'Away_Last10']].set_index('Metric').T

    features['home_goals'] = float(home_stats['Goal'].values[0])
    features['away_goals'] = float(away_stats['Goal'].values[0])
    features['home_shots'] = float(home_stats['Shot(OT)'].values[0])
    features['away_shots'] = float(away_stats['Shot(OT)'].values[0])
    features['home_possession'] = float(home_stats['Possession'].values[0].replace('%', ''))
    features['away_possession'] = float(away_stats['Possession'].values[0].replace('%', ''))

    # --- 07_elo_home.csv & 08_elo_away.csv ---
    elo_home = pd.read_csv(os.path.join(temp_dir, "07_elo_home.csv"))
    elo_away = pd.read_csv(os.path.join(temp_dir, "08_elo_away.csv"))
    home_elo = elo_home.iloc[0]['ELO_H']
    away_elo = elo_away.iloc[0]['ELO_A']
    features['home_elo'] = home_elo
    features['away_elo'] = away_elo
    features['elo_diff'] = home_elo - away_elo

    # --- 03_home_form.csv & 04_away_form.csv ---
    form_home = pd.read_csv(os.path.join(temp_dir, "03_home_form.csv"))
    form_away = pd.read_csv(os.path.join(temp_dir, "04_away_form.csv"))

    def get_result(home_goals, away_goals, perspective='home'):
        if perspective == 'home':
            if home_goals > away_goals:
                return 'W'
            elif home_goals == away_goals:
                return 'D'
            else:
                return 'L'
        else:
            if away_goals > home_goals:
                return 'W'
            elif away_goals == home_goals:
                return 'D'
            else:
                return 'L'

    for name, df in [('home', form_home), ('away', form_away)]:
        df[['HG', 'AG']] = df['FT'].str.split('-', expand=True).astype(int)
        if name == 'home':
            df['Result'] = df.apply(lambda row: get_result(row['HG'], row['AG'], 'home'), axis=1)
        else:
            df['Result'] = df.apply(lambda row: get_result(row['HG'], row['AG'], 'away'), axis=1)

        last5 = df.head(5)
        features[f'{name}_avg_goals_scored'] = last5['HG'].mean() if name == 'home' else last5['AG'].mean()
        features[f'{name}_avg_goals_conceded'] = last5['AG'].mean() if name == 'home' else last5['HG'].mean()
        features[f'{name}_form_pts'] = last5['Result'].map({'W': 3, 'D': 1, 'L': 0}).sum()

    # --- 06_goals_time.csv ---
    goals_time = pd.read_csv(os.path.join(temp_dir, "06_goals_time.csv"))
    home_1h = (int(goals_time['Home_Scored'][0].replace('%', '')) +
               int(goals_time['Home_Scored'][1].replace('%', '')) +
               int(goals_time['Home_Scored'][2].replace('%', ''))) / 100
    away_1h = (int(goals_time['Away_Scored'][0].replace('%', '')) +
               int(goals_time['Away_Scored'][1].replace('%', '')) +
               int(goals_time['Away_Scored'][2].replace('%', ''))) / 100
    features['home_1h_goal_pct'] = home_1h
    features['away_1h_goal_pct'] = away_1h

    # --- master_match.csv ---
    master = pd.read_csv(os.path.join(temp_dir, "master_match.csv"))
    h2h_str = master['H2H_WDL_10'].values[0]
    nums = re.findall(r'\d+', h2h_str)
    w_away = int(nums[0])
    d = int(nums[1])
    w_home = int(nums[2])
    features['h2h_home_wins'] = w_home
    features['h2h_away_wins'] = w_away
    features['h2h_draws'] = d

    return features


def load_or_train_model():
    global model, feature_columns
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        feature_columns = [col for col in df.columns if col not in ['ah_winner', 'ou_result', 'btts', 'over_ht']]
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_columns = []


@app.on_event("startup")
async def startup():
    load_or_train_model()


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/predict")
async def predict(zip_file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, "uploaded.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    os.remove(zip_path)

    features = extract_features_from_files(temp_dir)

    global model, feature_columns
    if model is None or len(feature_columns) == 0:
        if features['handicap'] < 0:
            if features['elo_diff'] < -10:
                ah_choice = 'away'
            else:
                ah_choice = 'home'
        else:
            if features['elo_diff'] > 10:
                ah_choice = 'home'
            else:
                ah_choice = 'away'

        avg_goals = features['home_goals'] + features['away_goals']
        ou_choice = 'over' if avg_goals > 2.75 else 'under'

        btts = (features['home_goals'] >= 1.0 and features['away_goals'] >= 1.0)
        over_ht = (features['home_1h_goal_pct'] > 0.3 or features['away_1h_goal_pct'] > 0.3)
    else:
        input_df = pd.DataFrame([features])[feature_columns]
        preds = model.predict(input_df)[0]
        ah_choice = 'home' if preds[0] == 1 else 'away'
        ou_choice = 'over' if preds[1] == 1 else 'under'
        btts = bool(preds[2])
        over_ht = bool(preds[3])

    info = pd.read_csv(os.path.join(temp_dir, "01_info.csv"))
    ah_str = info[info['0'] == 'Pre-game AH']['1'].values[0]
    ou_str = info[info['0'] == 'Pre-game O/U']['1'].values[0]
    ah_parts = ah_str.split('/')
    handicap_display = ah_parts[1].strip()
    ou_parts = ou_str.split('/')
    ou_line_display = ou_parts[1].strip()

    shutil.rmtree(temp_dir)

    return JSONResponse({
        "ah": handicap_display,
        "ah_choice": ah_choice,
        "ou": ou_line_display,
        "ou_choice": ou_choice,
        "btts": btts,
        "over_ht": over_ht,
        "features": features
    })


@app.post("/feedback")
async def feedback(
    ah_winner: str = Form(...),
    ou_result: str = Form(...),
    btts: int = Form(...),
    over_ht: int = Form(...),
    features_json: str = Form(...)
):
    import json
    features = json.loads(features_json)
    features['ah_winner'] = 1 if ah_winner == 'home' else 0
    features['ou_result'] = 1 if ou_result == 'over' else 0
    features['btts'] = btts
    features['over_ht'] = over_ht

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame()

    new_df = pd.DataFrame([features])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    global model, feature_columns
    target_columns = ['ah_winner', 'ou_result', 'btts', 'over_ht']
    feature_columns = [col for col in df.columns if col not in target_columns]
    X = df[feature_columns]
    y = df[target_columns]

    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        acc = model.score(X_test, y_test)
        return JSONResponse({"message": "Model updated", "accuracy": acc})
    else:
        return JSONResponse({"message": "Data collected, need more samples for training"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)