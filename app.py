from flask import Flask, jsonify
import requests, os, time
from datetime import datetime
import pytz

app = Flask(__name__)
KEY = os.getenv("API_KEY") or os.getenv("KEY", "")
HEADERS = {'x-apisports-key': KEY}
JST = pytz.timezone('Asia/Tokyo')

cache = {"time":0, "data":[]}

def fetch_today():
    if time.time() - cache["time"] < 60:
        return cache["data"]
    today = datetime.now(JST).strftime("%Y-%m-%d")
    url = f"https://v3.football.api-sports.io/fixtures?date={today}"
    r = requests.get(url, headers=HEADERS, timeout=15).json()
    cache["data"] = r.get("response", [])
    cache["time"] = time.time()
    return cache["data"]

@app.route("/")
def home():
    return "OK - buka /json atau /today"

@app.route("/json")
def json_full():
    return jsonify(fetch_today())

@app.route("/today")
def today_simple():
    data = fetch_today()
    out = []
    for m in data[:30]:  # ambil 30 pertama biar ringan
        out.append({
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "kickoff": m["fixture"]["date"],
            "league": m["league"]["name"],
            "status": m["fixture"]["status"]["short"],
            "score": f"{m['goals']['home']}-{m['goals']['away']}"
        })
    return jsonify(out)
