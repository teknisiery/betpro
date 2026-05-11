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
from sklearn.metrics import accuracy_score
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "model.pkl"
DATA_PATH = "dataset.csv"
PROFIT_HISTORY_PATH = "profit_history.csv"

model = None
feature_columns = []

# ==================== UTILS ====================
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

def parse_btts_line(btts_str):
    btts_str = btts_str.strip()
    if ' / ' in btts_str:
        parts = btts_str.split(' / ')
    else:
        parts = btts_str.split('/')
    yes_odds = float(parts[0].strip().split()[-1])
    no_odds = float(parts[1].strip().split()[-1])
    return yes_odds, no_odds

def parse_handicap_value(handicap_str):
    h = handicap_str.strip()
    clean = h.replace('-', '').replace('+', '')
    if '/' in clean:
        parts = clean.split('/')
        base = (float(parts[0]) + float(parts[1])) / 2
    else:
        base = float(clean)
    if h.startswith('-'):
        return base          # Home terima voor → positif
    else:
        return -base         # Home beri voor → negatif

def compute_ah_actual(ft_home, ft_away, handicap):
    effective = ft_home + handicap
    diff = effective - ft_away

    if diff > 0:
        if diff >= 0.5:
            return 'home', 1.0
        else:
            return 'home', 0.5
    elif diff < 0:
        if diff <= -0.5:
            return 'away', 1.0
        else:
            return 'away', 0.5
    else:
        return 'push', 0.0

def compute_ou_actual(ft_home, ft_away, ou_line):
    total = ft_home + ft_away
    # Garis bulat (2.0, 3.0, dst.)
    if ou_line % 1 == 0.0:
        if total > ou_line:
            return 'over', 1.0
        elif total < ou_line:
            return 'under', 1.0
        else:
            return 'push', 0.0

    # Garis pecahan (2.25, 2.75, 3.75, dst.)
    if (ou_line - int(ou_line)) == 0.25:
        lower_line = int(ou_line)
        upper_line = int(ou_line) + 0.5
    else:  # 0.75
        lower_line = int(ou_line) + 0.5
        upper_line = int(ou_line) + 1.0

    if total > upper_line:
        return 'over', 1.0
    elif total < lower_line:
        return 'under', 1.0
    elif total == upper_line:
        return 'over', 0.5
    elif total == lower_line:
        return 'under', 0.5
    else:
        return 'push', 0.0

def extract_features_from_files(temp_dir):
    features = {}

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

    mask_live_btts = info[0].str.strip().str.lower() == 'live btts'
    mask_live_over_ht = info[0].str.strip().str.lower() == 'live over ht'
    if mask_live_btts.any():
        btts_str = info.loc[mask_live_btts, 1].values[0]
        btts_yes, btts_no = parse_btts_line(btts_str)
    else:
        btts_yes, btts_no = 1.60, 2.00
    if mask_live_over_ht.any():
        ht_str = info.loc[mask_live_over_ht, 1].values[0]
        ht_yes, ht_no = parse_btts_line(ht_str)
    else:
        ht_yes, ht_no = 1.60, 2.00

    features['btts_yes_odds'] = btts_yes
    features['btts_no_odds'] = btts_no
    features['over_ht_yes_odds'] = ht_yes
    features['over_ht_no_odds'] = ht_no

    features['handicap_display_str'] = live_handicap_text
    features['ou_line_display_str'] = str(live_ou_line)

    handicap = parse_handicap_value(live_handicap_text)
    features['handicap'] = handicap

    features['delta_ah_home'] = live_home_ah - pre_home_ah
    features['delta_ah_away'] = live_away_ah - pre_away_ah
    features['delta_ou_over'] = live_over - pre_over
    features['delta_ou_under'] = live_under - pre_under

    home_team = str(info[info[0] == 'Home'][1].values[0])
    away_team = str(info[info[0] == 'Away'][1].values[0])
    match_date = str(info[info[0] == 'Tanggal'][1].values[0])
    features['home_team'] = home_team
    features['away_team'] = away_team
    features['match_date'] = match_date

    stats = safe_read_csv(os.path.join(temp_dir, "05_recent_stats.csv"),
                          required_columns=['Metric', 'Home_Last3', 'Home_Last10', 'Away_Last3', 'Away_Last10'])
    home_stats = stats[['Metric', 'Home_Last3', 'Home_Last10']].set_index('Metric').T
    away_stats = stats[['Metric', 'Away_Last3', 'Away_Last10']].set_index('Metric').T

    features['home_goals_l3'] = float(home_stats['Goal']['Home_Last3'])
    features['home_goals_l10'] = float(home_stats['Goal']['Home_Last10'])
    features['home_shots_l3'] = float(home_stats['Shot(OT)']['Home_Last3'])
    features['home_shots_l10'] = float(home_stats['Shot(OT)']['Home_Last10'])
    features['home_possession_l3'] = float(home_stats['Possession']['Home_Last3'].replace('%', ''))
    features['home_possession_l10'] = float(home_stats['Possession']['Home_Last10'].replace('%', ''))

    features['away_goals_l3'] = float(away_stats['Goal']['Away_Last3'])
    features['away_goals_l10'] = float(away_stats['Goal']['Away_Last10'])
    features['away_shots_l3'] = float(away_stats['Shot(OT)']['Away_Last3'])
    features['away_shots_l10'] = float(away_stats['Shot(OT)']['Away_Last10'])
    features['away_possession_l3'] = float(away_stats['Possession']['Away_Last3'].replace('%', ''))
    features['away_possession_l10'] = float(away_stats['Possession']['Away_Last10'].replace('%', ''))

    elo_home = safe_read_csv(os.path.join(temp_dir, "07_elo_home.csv"), required_columns=['ELO_H'])
    elo_away = safe_read_csv(os.path.join(temp_dir, "08_elo_away.csv"), required_columns=['ELO_A'])
    home_elo = elo_home.iloc[0]['ELO_H']
    away_elo = elo_away.iloc[0]['ELO_A']
    features['home_elo'] = float(home_elo)
    features['away_elo'] = float(away_elo)
    features['elo_diff'] = float(home_elo - away_elo)

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
            features['home_avg_goals_scored'] = float(last5['HG'].mean())
            features['home_avg_goals_conceded'] = float(last5['AG'].mean())
        else:
            features['away_avg_goals_scored'] = float(last5['AG'].mean())
            features['away_avg_goals_conceded'] = float(last5['HG'].mean())
        features[f'{name}_form_pts'] = int(last5['Result'].map({'W': 3, 'D': 1, 'L': 0}).sum())

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

    master = safe_read_csv(os.path.join(temp_dir, "master_match.csv"), required_columns=['H2H_WDL_10'])
    h2h_str = master['H2H_WDL_10'].values[0]
    nums = re.findall(r'\d+', h2h_str)
    if len(nums) < 3:
        raise ValueError("Format H2H_WDL_10 tidak sesuai (butuh 3 angka).")
    w_away = int(nums[0])
    d = int(nums[1])
    w_home = int(nums[2])
    features['h2h_home_wins'] = w_home
    features['h2h_away_wins'] = w_away
    features['h2h_draws'] = d

    features['handicap_display'] = live_handicap_text
    features['ou_line_display'] = str(live_ou_line)

    return convert_numpy(features)

def load_or_train_model():
    global model, feature_columns
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)
            df = pd.read_csv(DATA_PATH)
            target_columns = ['ah_winner', 'ou_result', 'btts', 'over_ht']
            feature_columns = [c for c in df.columns if c not in target_columns]
        except:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
            feature_columns = []
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

        use_ml = False
        if model is not None and len(feature_columns) > 0:
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(model)
                use_ml = True
            except:
                use_ml = False

        if not use_ml:
            handicap = features['handicap']
            elo_diff = features['elo_diff']
            if handicap < 0:
                ah_choice = 'away' if elo_diff < 0 else 'home'
            elif handicap > 0:
                ah_choice = 'home' if elo_diff > -10 else 'away'
            else:
                ah_choice = 'home' if elo_diff > 0 else 'away'

            avg_goals = features['home_goals_l10'] + features['away_goals_l10']
            ou_choice = 'over' if avg_goals > features['ou_line'] else 'under'

            btts = (features['home_goals_l10'] >= 1.0 and features['away_goals_l10'] >= 1.0)
            over_ht = (features['home_1h_goal_pct'] > 0.3 or features['away_1h_goal_pct'] > 0.3)
        else:
            input_df = pd.DataFrame([features])[feature_columns]
            preds = model.predict(input_df)[0]
            ah_choice = 'home' if preds[0] == 1 else 'away'
            ou_choice = 'over' if preds[1] == 1 else 'under'
            btts = bool(preds[2])
            over_ht = bool(preds[3])

        handicap_display = features.pop('handicap_display', '0')
        ou_line_display = features.pop('ou_line_display', '2.5')

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
    try:
        import json
        features = json.loads(features_json)

        hdcp_str = features.get('handicap_display_str', '0')
        handicap = parse_handicap_value(hdcp_str)

        ou_line_str = features.get('ou_line_display_str', '2.5')
        ou_line = float(ou_line_str)

        pred_ah = features.get('pred_ah', 'home')
        pred_ou = features.get('pred_ou', 'over')
        pred_btts = features.get('pred_btts', False)
        pred_over_ht = features.get('pred_over_ht', False)

        actual_ah, ah_factor = compute_ah_actual(ft_home, ft_away, handicap)
        actual_ou, ou_factor = compute_ou_actual(ft_home, ft_away, ou_line)

        actual_btts = 1 if (ft_home > 0 and ft_away > 0) else 0
        actual_over_ht = 1 if (ht_home + ht_away) > 0.5 else 0

        target_ah = None if actual_ah == 'push' else (1 if actual_ah == 'home' else 0)
        target_ou = None if actual_ou == 'push' else (1 if actual_ou == 'over' else 0)

        ah_home_odds = float(features.get('ah_home_odds', 1.0))
        ah_away_odds = float(features.get('ah_away_odds', 1.0))
        if actual_ah != 'push':
            if pred_ah == 'home':
                odds = ah_home_odds
                if actual_ah == 'home':
                    profit_ah = (odds - 1) * 100 * ah_factor
                else:
                    profit_ah = -100 * abs(ah_factor)
            else:
                odds = ah_away_odds
                if actual_ah == 'away':
                    profit_ah = (odds - 1) * 100 * ah_factor
                else:
                    profit_ah = -100 * abs(ah_factor)
        else:
            profit_ah = 0.0

        over_odds = float(features.get('over_odds', 1.0))
        under_odds = float(features.get('under_odds', 1.0))
        if actual_ou != 'push':
            if pred_ou == 'over':
                if actual_ou == 'over':
                    profit_ou = (over_odds - 1) * 100 * ou_factor
                else:
                    profit_ou = -100
            else:  # pred_ou == 'under'
                if actual_ou == 'under':
                    profit_ou = (under_odds - 1) * 100 * ou_factor
                else:
                    profit_ou = -100
        else:
            profit_ou = 0.0

        btts_yes_odds = float(features.get('btts_yes_odds', 1.60))
        btts_no_odds = float(features.get('btts_no_odds', 2.00))
        ht_yes_odds = float(features.get('over_ht_yes_odds', 1.60))
        ht_no_odds = float(features.get('over_ht_no_odds', 2.00))

        if pred_btts:
            profit_btts = (btts_yes_odds - 1) * 50 if actual_btts else -50
        else:
            profit_btts = (btts_no_odds - 1) * 50 if not actual_btts else -50

        if pred_over_ht:
            profit_ht = (ht_yes_odds - 1) * 50 if actual_over_ht else -50
        else:
            profit_ht = (ht_no_odds - 1) * 50 if not actual_over_ht else -50

        total_profit = profit_ah + profit_ou + profit_btts + profit_ht

        # Simpan dataset
        clean_features = {}
        non_numeric_keys = ['home_team', 'away_team', 'match_date',
                            'handicap_display_str', 'ou_line_display_str',
                            'handicap_display', 'ou_line_display']
        for k, v in features.items():
            if not k.startswith('pred_') and k not in non_numeric_keys:
                clean_features[k] = v
        clean_features['handicap'] = handicap
        clean_features['ah_winner'] = target_ah
        clean_features['ou_result'] = target_ou
        clean_features['btts'] = actual_btts
        clean_features['over_ht'] = actual_over_ht

        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            df = pd.DataFrame()

        new_row = pd.DataFrame([clean_features])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)

        home_team = features.get('home_team', 'Home')
        away_team = features.get('away_team', 'Away')
        match_date = features.get('match_date', 'unknown')

        record = {
            'date': match_date,
            'home': home_team,
            'away': away_team,
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

        total_accumulated = float(history_df['total_profit'].sum())

        global model, feature_columns
        target_columns = ['ah_winner', 'ou_result', 'btts', 'over_ht']
        clean_df = df.dropna(subset=target_columns)
        debug_info = (f"hdcp_str={hdcp_str}, handicap={handicap}, "
                      f"actual_ah={actual_ah}, clean_samples={len(clean_df)}")

        acc_ah = acc_ou = acc_btts = acc_ht = None
        if len(clean_df) >= 10:
            feature_columns = [c for c in clean_df.columns if c not in target_columns]
            X = clean_df[feature_columns]
            y = clean_df[target_columns].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_PATH)

            preds_test = model.predict(X_test)
            acc_ah = accuracy_score(y_test['ah_winner'], preds_test[:, 0])
            acc_ou = accuracy_score(y_test['ou_result'], preds_test[:, 1])
            acc_btts = accuracy_score(y_test['btts'], preds_test[:, 2])
            acc_ht = accuracy_score(y_test['over_ht'], preds_test[:, 3])

            training_msg = (f"Model updated. "
                            f"AH Acc: {acc_ah:.2f}, OU Acc: {acc_ou:.2f}, "
                            f"BTTS Acc: {acc_btts:.2f}, HT Acc: {acc_ht:.2f}")
        else:
            training_msg = f"Data tersimpan ({len(clean_df)} sampel lengkap), butuh minimal 10 untuk training."

        response_data = {
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
            "message": training_msg + " | Debug: " + debug_info,
            "acc_per_option": {
                "ah": acc_ah,
                "ou": acc_ou,
                "btts": acc_btts,
                "over_ht": acc_ht
            }
        }
        response_data = convert_numpy(response_data)
        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"error": f"Feedback failed: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)