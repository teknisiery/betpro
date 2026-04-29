from flask import Flask, jsonify
from fotmob_api import FotMob
from datetime import datetime

app = Flask(__name__)
fm = FotMob()

@app.route("/")
def home():
    return "OK - buka /json"

@app.route("/json")
def json_data():
    today = datetime.now().strftime("%Y%m%d")
    matches = fm.get_matches_by_date(today)
    # ambil 30 pertandingan pertama biar ringan
    out = []
    for m in matches[:30]:
        out.append({
            "home": m.get("home",{}).get("name"),
            "away": m.get("away",{}).get("name"),
            "score": f"{m.get('home',{}).get('score',0)}-{m.get('away',{}).get('score',0)}",
            "status": m.get("status",{}).get("reason",{}).get("short"),
            "league": m.get("leagueName")
        })
    return jsonify(out)

@app.route("/today")
def today():
    return json_data()
