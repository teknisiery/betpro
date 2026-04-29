from flask import Flask, jsonify
import requests, os
from datetime import datetime
import pytz

app = Flask(__name__)
KEY = os.getenv("API_KEY") or os.getenv("KEY", "")
HEADERS = {'x-apisports-key': KEY}

@app.route("/debug")
def debug():
    today = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d")
    url = f"https://v3.football.api-sports.io/fixtures?date={today}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    return jsonify({
        "key_terbaca": KEY[:8] + "..." if KEY else "KOSONG",
        "status_code": r.status_code,
        "api_response": r.json()
    })

@app.route("/")
def home():
    return "buka /debug"            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "kickoff": m["fixture"]["date"],
            "league": m["league"]["name"],
            "status": m["fixture"]["status"]["short"],
            "score": f"{m['goals']['home']}-{m['goals']['away']}"
        })
    return jsonify(out)
