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
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "model.pkl"
DATA_PATH = "dataset.csv"
PROFIT_HISTORY_PATH = "profit_history.csv"

model = None
feature_columns = []


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def safe_read_csv(file_path, required_columns=None, **kwargs):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {os.path.basename(file_path)} tidak ditemukan.")
    df = pd.read_csv(file_path, **kwargs)
    if df.empty:
        raise ValueError(f"File {os.path.basename(file_path)} kosong.")
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"File {os.path.basename(file_path)} kehilangan kolom: {missing}")
    return df


def parse_ah_line(ah_str):
    ah_str = ah_str.strip()
    if ' / ' in ah_str:
        parts = ah_str.split(' / ')
    else:
        parts = ah_str.split('/')
    if len(parts) < 3:
        raise ValueError(f"Format AH tidak valid: {ah_str}")
    home_odds = float(parts[0].strip().split()[-1])
    handicap_text = parts[1].strip()
    away_odds = float(parts[2].strip().split()[-1])
    return home_odds, handicap_text, away_odds


def parse_ou_line(ou_str):
    ou_str = ou_str.strip()
    if ' / ' in ou_str:
        parts = ou_str.split(' / ')
    else:
        parts = ou_str.split('/')
    if len(parts) < 3:
        raise ValueError(f"Format O/U tidak valid: {ou_str}")
    over_odds = float(parts[0].strip().split()[-1])
    line = float(parts[1].strip())
    under_odds = float(parts[2].strip().split()[-1])
    return over_odds, line, under_odds


def extract_features_from_files(temp_dir):
    features = {}

    # 01_info.csv
    info = safe_read_csv(os.path.join(temp_dir, "01_info.csv"), header=None)
    info = info.map(lambda x: x.strip() if isinstance(x, str) else x)

    mask_pre_ah = info[0].str.strip().str.lower() == 'pre-game ah'
    if not mask_pre_ah.any():
        raise ValueError("Baris 'Pre-game AH' tidak ditemukan di 01_info.csv")
    pre_ah_str = info.loc[mask_pre_ah, 1].values[0]
    pre_ou_str = info[info[0].str.strip().str.lower() == 'pre-game o/u'][1].values[0]
    live_ah_str = info[info[0].str.strip().str.lower() == 'live ah'][1].values[0]
    live_ou_str = info[info[0].str.strip().str.lower() == 'live o/u'][1].values[0]

    pre_home_ah, handicap_text, pre_away_ah = parse_ah_line(pre_ah_str)
    pre_over, ou_line, pre_under = parse_ou_line(pre_ou_str)
    live_home_ah, live_handicap_text, live_away_ah = parse_ah_line(live_ah_str)
    live_over, live_ou_line, live_under = parse_ou_line(live_ou_str)

    features['ah_home_odds'] = live_home_ah
    features['ah_away_odds'] = live_away_ah
    features['ou_line'] = live_ou_line
    features['over_odds'] = live_over
    features['under_odds'] = live_under

    handicap_val = float(handicap_text.replace('-', '').replace('+', '').split('/')[0])
    if handicap_text.startswith('-'):
        handicap = -handicap_val
    elif handicap_text.startswith('+'):
        handicap = handicap_val
    else:
        if live_home_ah > live_away_ah:
            handicap = handicap_val
        else:
            handicap = -handicap_val
    features['handicap'] = handicap

    features['delta_ah_home'] = live_home_ah - pre_home_ah
    features['delta_ah_away'] = live_away_ah - pre_away_ah
    features['delta_ou_over'] = live_over - pre_over
    features['delta_ou_under'] = live_under - pre_under

    # tim & tanggal
    home_team = str(info[info[0] == 'Home'][1].values[0])
    away_team = str(info[info[0] == 'Away'][1].values[0])
    match_date = str(info[info[0] == 'Tanggal'][1].values[0])
    features['home_team'] = home_team
    features['away_team'] = away_team
    features['match_date'] = match_date

    # 05_recent_stats.csv (dengan Last3 dan Last10)
    stats = safe_read_csv(os.path.join(temp_dir, "05_recent_stats.csv"),
                          required_columns=['Metric', 'Home_Last3', 'Home_Last10', 'Away_Last3', 'Away_Last10'])
    stats = stats.set_index('Metric')
    for metric, feat_name in [('Goal','goals'), ('Loss','loss'), ('Shot(OT)','shots'),
                              ('Corner Kicks','corners'), ('Yellow Cards','cards'),
                              ('Fouls','fouls'), ('Possession','possession')]:
        features[f'home_{feat_name}_l3'] = float(stats.loc[metric, 'Home_Last3'].replace('%',''))
        features[f'home_{feat_name}_l10'] = float(stats.loc[metric, 'Home_Last10'].replace('%',''))
        features[f'away_{feat_name}_l3'] = float(stats.loc[metric, 'Away_Last3'].replace('%',''))
        features[f'away_{feat_name}_l10'] = float(stats.loc[metric, 'Away_Last10'].replace('%',''))

    # ELO
    elo_home = safe_read_csv(os.path.join(temp_dir, "07_elo_home.csv"), required_columns=['ELO_H'])
    elo_away = safe_read_csv(os.path.join(temp_dir, "08_elo_away.csv"), required_columns=['ELO_A'])
    features['home_elo'] = float(elo_home.iloc[0]['ELO_H'])
    features['away_elo'] = float(elo_away.iloc[0]['ELO_A'])
    features['elo_diff'] = features['home_elo'] - features['away_elo']

    # Form
    form_home = safe_read_csv(os.path.join(temp_dir, "03_home_form.csv"), required_columns=['FT'])
    form_away = safe_read_csv(os.path.join(temp_dir, "04_away_form.csv"), required_columns=['FT'])

    def get_result(hg, ag, perspective):
        if perspective == 'home':
            return 'W' if hg > ag else ('D' if hg == ag else 'L')
        else:
            return 'W' if ag > hg else ('D' if ag == hg else 'L')

    for name, df in [('home', form_home), ('away', form_away)]:
        df[['HG', 'AG']] = df['FT'].str.split('-', expand=True).astype(int)
        df['Result'] = df.apply(lambda r: get_result(r['HG'], r['AG'], name), axis=1)
        last5 = df.head(min(5, len(df)))
        if name == 'home':
            features['home_avg_goals_scored_l5'] = float(last5['HG'].mean())
            features['home_avg_goals_conceded_l5'] = float(last5['AG'].mean())
        else:
            features['away_avg_goals_scored_l5'] = float(last5['AG'].mean())
            features['away_avg_goals_conceded_l5'] = float(last5['HG'].mean())
        features[f'{name}_form_pts'] = int(last5['Result'].map({'W': 3, 'D': 1, 'L': 0}).sum())

    # Goals time
    goals_time = safe_read_csv(os.path.join(temp_dir, "06_goals_time.csv"),
                               required_columns=['Home_Scored', 'Away_Scored'])
    home_1h = (int(goals_time['Home_Scored'][0].replace('%', '')) +
               int(goals_time['Home_Scored'][1].replace('%', '')) +
               int(goals_time['Home_Scored'][2].replace('%', ''))) / 100
    away_1h = (int(goals_time['Away_Scored'][0].replace('%', '')) +
               int(goals_time['Away_Scored'][1].replace('%', '')) +
               int(goals_time['Away_Scored'][2].replace('%', ''))) / 100
    features['home_1h_goal_pct'] = float(home_1h)
    features['away_1h_goal_pct'] = float(away_1h)

    # Master H2H
    master = safe_read_csv(os.path.join(temp_dir, "master_match.csv"), required_columns=['H2H_WDL_10'])
    h2h_str = master['H2H_WDL_10'].values[0]
    nums = re.findall(r'\d+', h2h_str)
    if len(nums) < 3:
        raise ValueError("Format H2H_WDL_10 tidak sesuai.")
    features['h2h_home_wins'] = int(nums[0])
    features['h2h_away_wins'] = int(nums[1])
    features['h2h_draws'] = int(nums[2])

    features['handicap_display'] = live_handicap_text
    features['ou_line_display'] = str(live_ou_line)

    return convert_numpy(features)


def load_or_train_model():
    global model, feature_columns
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        target_columns = ['ah_winner', 'ou_result', 'btts', 'over_ht']
        feature_columns = [c for c in df.columns if c not in target_columns]
    else:
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
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

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        os.remove(zip_path)

        features = extract_features_from_files(temp_dir)

        global model, feature_columns
        if model is None or len(feature_columns) == 0:
            # rule-based
            if features['handicap'] < 0:
                ah_choice = 'away' if features['elo_diff'] < -10 else 'home'
            else:
                ah_choice = 'home' if features['elo_diff'] > 10 else 'away'

            avg_goals = features['home_goals_l10'] + features['away_goals_l10']
            ou_choice = 'over' if avg_goals > 2.75 else 'under'

            btts = (features['home_goals_l10'] >= 1.0 and features['away_goals_l10'] >= 1.0)
            over_ht = (features['home_1h_goal_pct'] > 0.3 or features['away_1h_goal_pct'] > 0.3)
        else:
            input_df = pd.DataFrame([features])[feature_columns]
            preds = model.predict(input_df)[0]
            ah_choice = 'home' if preds[0] == 1 else 'away'
            ou_choice = 'over' if preds[1] == 1 else 'under'
            btts = bool(preds[2])
            over_ht = bool(preds[3])

        handicap_display = features.pop('handicap_display', '0/0.5')
        ou_line_display = features.pop('ou_line_display', '3')

        features['pred_ah'] = ah_choice
        features['pred_ou'] = ou_choice
        features['pred_btts'] = btts
        features['pred_over_ht'] = over_ht

        home_team = features.get('home_team', 'Home')
        away_team = features.get('away_team', 'Away')
        match_date = features.get('match_date', '')

        shutil.rmtree(temp_dir)

        return JSONResponse({
            "ah": handicap_display,
            "ah_choice": ah_choice,
            "ou": ou_line_display,
            "ou_choice": ou_choice,
            "btts": btts,
            "over_ht": over_ht,
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "features": features
        })
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/feedback")
async def feedback(
    ht_home: int = Form(...),
    ht_away: int = Form(...),
    ft_home: int = Form(...),
    ft_away: int = Form(...),
    features_json: str = Form(...)
):
    import json
    features = json.loads(features_json)

    pred_ah = features.get('pred_ah', 'home')
    pred_ou = features.get('pred_ou', 'over')
    pred_btts = features.get('pred_btts', False)
    pred_over_ht = features.get('pred_over_ht', False)

    handicap = features.get('handicap', 0)
    ou_line = features.get('ou_line', 2.5)

    effective_home = ft_home + handicap
    actual_ah = 'home' if effective_home > ft_away else ('away' if effective_home < ft_away else 'push')

    total_goals = ft_home + ft_away
    actual_ou = 'over' if total_goals > ou_line else ('under' if total_goals < ou_line else 'push')

    actual_btts = 1 if (ft_home > 0 and ft_away > 0) else 0
    actual_over_ht = 1 if (ht_home + ht_away) > 0.5 else 0

    features['ah_winner'] = None if actual_ah == 'push' else (1 if actual_ah == 'home' else 0)
    features['ou_result'] = None if actual_ou == 'push' else (1 if actual_ou == 'over' else 0)
    features['btts'] = actual_btts
    features['over_ht'] = actual_over_ht

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame()
    new_df = pd.DataFrame([features])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    # Profit
    ah_home_odds = features.get('ah_home_odds', 1.0)
    ah_away_odds = features.get('ah_away_odds', 1.0)
    over_odds = features.get('over_odds', 1.0)
    under_odds = features.get('under_odds', 1.0)

    profit_ah = profit_ou = profit_btts = profit_ht = 0
    if actual_ah != 'push':
        odds = ah_home_odds if pred_ah == 'home' else ah_away_odds
        profit_ah = (odds - 1) * 100 if pred_ah == actual_ah else -100
    if actual_ou != 'push':
        odds = over_odds if pred_ou == 'over' else under_odds
        profit_ou = (odds - 1) * 100 if pred_ou == actual_ou else -100
    profit_btts = 50 if pred_btts == bool(actual_btts) else -50
    profit_ht = 50 if pred_over_ht == bool(actual_over_ht) else -50
    total_profit = profit_ah + profit_ou + profit_btts + profit_ht

    # Simpan riwayat
    record = {
        'date': features.get('match_date', 'unknown'),
        'home': features.get('home_team', 'Home'),
        'away': features.get('away_team', 'Away'),
        'score': f"{ft_home}-{ft_away}",
        'profit_ah': profit_ah,
        'profit_ou': profit_ou,
        'profit_btts': profit_btts,
        'profit_ht': profit_ht,
        'total_profit': total_profit,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if os.path.exists(PROFIT_HISTORY_PATH):
        history_df = pd.read_csv(PROFIT_HISTORY_PATH)
    else:
        history_df = pd.DataFrame()
    history_df = pd.concat([history_df, pd.DataFrame([record])], ignore_index=True)
    history_df.to_csv(PROFIT_HISTORY_PATH, index=False)
    total_accumulated = history_df['total_profit'].sum()

    # Training
    global model, feature_columns
    target_columns = ['ah_winner', 'ou_result', 'btts', 'over_ht']
    clean_df = df.dropna(subset=target_columns)
    if len(clean_df) >= 10:
        feature_columns = [c for c in clean_df.columns if c not in target_columns]
        X = clean_df[feature_columns]
        y = clean_df[target_columns].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        acc = model.score(X_test, y_test)
        training_msg = f"Model updated. Accuracy: {acc:.4f}"
    else:
        training_msg = f"Data tersimpan ({len(clean_df)} sampel lengkap), butuh minimal 10 untuk training."

    return JSONResponse({
        "actual_ah": actual_ah,
        "actual_ou": actual_ou,
        "actual_btts": actual_btts,
        "actual_over_ht": actual_over_ht,
        "profit": {
            "ah": profit_ah,
            "ou": profit_ou,
            "btts": profit_btts,
            "over_ht": profit_ht,
            "total": total_profit
        },
        "total_accumulated": total_accumulated,
        "history": history_df[['home', 'away', 'score', 'total_profit']].tail(5).to_dict('records'),
        "message": training_msg
    })
