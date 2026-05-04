from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import uvicorn
import glob

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Path untuk menyimpan model dan dataset
MODEL_PATH = "model.pkl"
DATA_PATH = "dataset.csv"

# Variabel global
model = None
feature_columns = []

# ------------------------------------------------------------
# Fungsi ekstraksi fitur dari 9 file CSV
# ------------------------------------------------------------
def extract_features_from_files(temp_dir):
    """
    Membaca semua file CSV di temp_dir, mengembalikan dictionary fitur.
    """
    features = {}

    # --- 01_info.csv ---
    info = pd.read_csv(os.path.join(temp_dir, "01_info.csv"))
    # Ambil odds AH dan OU
    # Format: "Home 2.03 / -0.5/1 / Milan 1.83" -> kita parse
    ah_str = info[info['0'] == 'Pre-game AH']['1'].values[0]
    ou_str = info[info['0'] == 'Pre-game O/U']['1'].values[0]

    # Parse AH
    parts_ah = ah_str.split('/')
    home_ah_odds = float(parts_ah[0].strip().split()[-1])
    handicap_text = parts_ah[1].strip()
    away_ah_odds = float(parts_ah[2].strip().split()[-1])

    # Parse OU
    parts_ou = ou_str.split('/')
    over_odds = float(parts_ou[0].strip().split()[-1])
    ou_line = float(parts_ou[1].strip())
    under_odds = float(parts_ou[2].strip().split()[-1])

    features['ah_home_odds'] = home_ah_odds
    features['ah_away_odds'] = away_ah_odds
    features['ou_line'] = ou_line
    features['over_odds'] = over_odds
    features['under_odds'] = under_odds

    # Handicap value untuk model: jika handicap negatif, away diunggulkan
    handicap_val = float(handicap_text.replace('-','').replace('+','').split('/')[0])
    if handicap_text.startswith('-'):
        features['handicap'] = -handicap_val
    else:
        features['handicap'] = handicap_val

    # --- 05_recent_stats.csv ---
    stats = pd.read_csv(os.path.join(temp_dir, "05_recent_stats.csv"))
    # Ambil statistik Home (Sassuolo) Last10 dan Away (Milan) Last10
    home_stats = stats[['Metric','Sassuolo_Last10']].set_index('Metric').T
    away_stats = stats[['Metric','Milan_Last10']].set_index('Metric').T

    # Goal, Loss, Shot(OT), Corner Kicks, Yellow Cards, Fouls, Possession
    features['home_goals'] = float(home_stats['Goal'].values[0])
    features['away_goals'] = float(away_stats['Goal'].values[0])
    features['home_shots'] = float(home_stats['Shot(OT)'].values[0])
    features['away_shots'] = float(away_stats['Shot(OT)'].values[0])
    features['home_possession'] = float(home_stats['Possession'].values[0].replace('%',''))
    features['away_possession'] = float(away_stats['Possession'].values[0].replace('%',''))

    # --- 07_elo_sassuolo.csv & 08_elo_milan.csv ---
    elo_home = pd.read_csv(os.path.join(temp_dir, "07_elo_sassuolo.csv"))
    elo_away = pd.read_csv(os.path.join(temp_dir, "08_elo_milan.csv"))
    home_elo = elo_home.iloc[0]['ELO_H']  # ELO terbaru (pertandingan terbaru di baris pertama)
    away_elo = elo_away.iloc[0]['ELO_A']
    features['home_elo'] = home_elo
    features['away_elo'] = away_elo
    features['elo_diff'] = home_elo - away_elo

    # --- 03_sassuolo_form.csv & 04_milan_form.csv ---
    form_home = pd.read_csv(os.path.join(temp_dir, "03_sassuolo_form.csv"))
    form_away = pd.read_csv(os.path.join(temp_dir, "04_milan_form.csv"))

    # Hitung rata2 gol dicetak/kebobolan dari form 5 terakhir
    for name, df in [('home', form_home), ('away', form_away)]:
        df[['HG','AG']] = df['FT'].str.split('-', expand=True).astype(int)
        df['total_goals'] = df['HG'] + df['AG']
        last5 = df.head(5)
        features[f'{name}_avg_goals_scored'] = last5['HG'].mean() if name=='home' else last5['AG'].mean()
        features[f'{name}_avg_goals_conceded'] = last5['AG'].mean() if name=='home' else last5['HG'].mean()
        features[f'{name}_form_pts'] = last5['Result_Sassuolo' if name=='home' else 'Result_Milan'].map({'W':3,'D':1,'L':0}).sum()

    # --- 06_goals_time.csv ---
    goals_time = pd.read_csv(os.path.join(temp_dir, "06_goals_time.csv"))
    # Rata2 gol babak pertama (1-15 + 16-30 + 31-45)
    home_1h = (int(goals_time['Sas_Scored'][0].replace('%','')) + 
               int(goals_time['Sas_Scored'][1].replace('%','')) + 
               int(goals_time['Sas_Scored'][2].replace('%',''))) / 100
    away_1h = (int(goals_time['Mil_Scored'][0].replace('%','')) + 
               int(goals_time['Mil_Scored'][1].replace('%','')) + 
               int(goals_time['Mil_Scored'][2].replace('%',''))) / 100
    features['home_1h_goal_pct'] = home_1h
    features['away_1h_goal_pct'] = away_1h

    # --- master_sassuolo_milan.csv ---
    master = pd.read_csv(os.path.join(temp_dir, "master_sassuolo_milan.csv"))
    # H2H_WDL_10 misal "Milan 4 - Draw 3 - Sassuolo 3"
    h2h_str = master['H2H_WDL_10'].values[0]
    # Parse
    import re
    nums = re.findall(r'\d+', h2h_str)
    w_away = int(nums[0])
    d = int(nums[1])
    w_home = int(nums[2])
    features['h2h_home_wins'] = w_home
    features['h2h_away_wins'] = w_away
    features['h2h_draws'] = d

    return features

# ------------------------------------------------------------
# Training awal (jika ada dataset)
# ------------------------------------------------------------
def load_or_train_model():
    global model, feature_columns
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        # Ambil feature columns dari dataset
        df = pd.read_csv(DATA_PATH)
        feature_columns = [col for col in df.columns if col not in ['ah_winner','ou_result','btts','over_ht']]
    else:
        # Jika tidak ada model, buat dummy classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Akan dilatih nanti saat ada data
        feature_columns = []

@app.on_event("startup")
async def startup():
    load_or_train_model()

# ------------------------------------------------------------
# Endpoint utama
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """
    Menerima 9 file CSV, ekstrak fitur, beri rekomendasi.
    """
    # Simpan file ke temporary directory
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Ekstrak fitur
    features = extract_features_from_files(temp_dir)

    # Jika model belum terlatih, gunakan rule-based
    global model, feature_columns
    if model is None or len(feature_columns) == 0:
        # Rekomendasi rule-based sederhana
        # AH: pilih tim dengan ELO lebih tinggi dan handicap menguntungkan
        if features['handicap'] < 0:  # Away diunggulkan
            if features['elo_diff'] < -10:
                ah_choice = 'away'
            else:
                ah_choice = 'home'
        else:
            if features['elo_diff'] > 10:
                ah_choice = 'home'
            else:
                ah_choice = 'away'

        # OU: jika rata2 gol kedua tim > 2.75 -> over
        avg_goals = features['home_goals'] + features['away_goals']
        ou_choice = 'over' if avg_goals > 2.75 else 'under'

        # BTTS: jika home dan away rata2 mencetak >=1 gol
        btts = (features['home_goals'] >= 1.0 and features['away_goals'] >= 1.0)

        # Over HT: jika persentase gol babak pertama tinggi
        over_ht = (features['home_1h_goal_pct'] > 0.3 or features['away_1h_goal_pct'] > 0.3)

    else:
        # Gunakan model machine learning
        input_df = pd.DataFrame([features])[feature_columns]
        preds = model.predict(input_df)[0]  # array 4 elemen
        ah_choice = 'home' if preds[0] == 1 else 'away'
        ou_choice = 'over' if preds[1] == 1 else 'under'
        btts = bool(preds[2])
        over_ht = bool(preds[3])

    # Parse handicap display string (misal "-0.5/1")
    info = pd.read_csv(os.path.join(temp_dir, "01_info.csv"))
    ah_str = info[info['0'] == 'Pre-game AH']['1'].values[0]
    ou_str = info[info['0'] == 'Pre-game O/U']['1'].values[0]

    # Bersihkan
    ah_parts = ah_str.split('/')
    handicap_display = ah_parts[1].strip()
    ou_parts = ou_str.split('/')
    ou_line_display = ou_parts[1].strip()

    # Hapus folder temp
    shutil.rmtree(temp_dir)

    return JSONResponse({
        "ah": handicap_display,
        "ah_choice": ah_choice,
        "ou": ou_line_display,
        "ou_choice": ou_choice,
        "btts": btts,
        "over_ht": over_ht
    })

@app.post("/feedback")
async def feedback(
    ah_winner: str = Form(...),
    ou_result: str = Form(...),
    btts: int = Form(...),
    over_ht: int = Form(...),
    features_json: str = Form(...)
):
    """
    Menerima hasil aktual dan menyimpan ke dataset, lalu melatih ulang model.
    """
    import json
    features = json.loads(features_json)
    features['ah_winner'] = 1 if ah_winner == 'home' else 0
    features['ou_result'] = 1 if ou_result == 'over' else 0
    features['btts'] = btts
    features['over_ht'] = over_ht

    # Baca dataset yang sudah ada, tambahkan
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame()

    new_df = pd.DataFrame([features])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    # Latih ulang model
    global model, feature_columns
    target_columns = ['ah_winner','ou_result','btts','over_ht']
    feature_columns = [col for col in df.columns if col not in target_columns]
    X = df[feature_columns]
    y = df[target_columns]

    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        # Hitung akurasi
        acc = model.score(X_test, y_test)
        return JSONResponse({"message": "Model updated", "accuracy": acc})
    else:
        return JSONResponse({"message": "Data collected, need more samples for training"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)    df = pd.read_csv(file_path)
    X = df.drop(columns=[target])
    y = df[target]

    X_encoded = X.copy()
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            encoders[col] = le
    encoder_dict = encoders

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        encoder_dict['target'] = le_target

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model = clf
    joblib.dump(model, "best_model.pkl")
    joblib.dump(encoder_dict, "encoders.pkl")

    return JSONResponse({"accuracy": round(acc, 4), "message": "Model siap!"})


@app.post("/predict")
async def predict(
    team_a: str = Form(...),
    team_b: str = Form(...),
    possession_a: float = Form(...),
    shots_a: int = Form(...),
    shots_b: int = Form(...),
    xg_a: float = Form(0.0),
    xg_b: float = Form(0.0),
):
    global model, encoder_dict, target_col
    if model is None:
        return JSONResponse({"error": "Model belum dilatih"}, status_code=400)

    input_data = pd.DataFrame([{
        "team_a": team_a,
        "team_b": team_b,
        "possession_a": possession_a,
        "shots_a": shots_a,
        "shots_b": shots_b,
        "xg_a": xg_a,
        "xg_b": xg_b,
    }])

    for col, le in encoder_dict.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    prediction = model.predict(input_data)[0]

    if 'target' in encoder_dict:
        prediction = encoder_dict['target'].inverse_transform([prediction])[0]

    return JSONResponse({"prediction": str(prediction)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
