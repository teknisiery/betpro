from flask import Flask, jsonify
import requests
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    return "OK - buka /today"

@app.route("/today")
def today():
    date = datetime.utcnow().strftime("%Y%m%d")
    # API FotMob langsung (tidak perlu key)
    url = f"https://www.fotmob.com/api/matches?date={date}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        matches = []
        for league in data.get("leagues", []):
            for m in league.get("matches", []):
                matches.append({
                    "home": m["home"]["name"],
                    "away": m["away"]["name"],
                    "score": f"{m['home'].get('score',0)}-{m['away'].get('score',0)}",
                    "status": m["status"]["short"],
                    "league": league["name"],
                    "time": m["time"]
                })
        return jsonify(matches[:40])
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/json")
def json_full():
    return today()
