import os
import requests
from flask import Flask, render_template_string, request
from datetime import datetime

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "e640f52564a146169b20d89e8259d518a12b0d03ea06e76c08785ada1926be52")
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

HTML = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BET PRO - AI Football</title>
<style>
body{font-family:system-ui;background:#0b1220;color:#e5e7eb;margin:0;padding:16px}
h1{color:#22d3ee}
.card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px;margin:12px 0}
.btn{background:#22d3ee;color:#001;padding:8px 12px;border:none;border-radius:8px;text-decoration:none}
.small{color:#9ca3af;font-size:12px}
.win{color:#10b981}
</style>
</head>
<body>
<h1>BET PRO ⚽</h1>
<p class="small">AI prediction powered by API-Football | {{today}}</p>

<form method="get">
<input name="date" type="date" value="{{date}}" style="padding:8px;border-radius:8px;border:1px solid #333;background:#000;color:#fff">
<button class="btn">Lihat</button>
</form>

{% for m in matches %}
<div class="card">
<b>{{m.teams.home.name}} vs {{m.teams.away.name}}</b><br>
<span class="small">{{m.league.name}} - {{m.fixture.date[11:16]}} WIB</span><br><br>
{% if m.pred %}
Home: {{m.pred.percent.home}} | Draw: {{m.pred.percent.draw}} | Away: {{m.pred.percent.away}}<br>
Prediksi: <b class="win">{{m.pred.advice or 'Analisis'}}</b><br>
<span class="small">Form: {{m.pred.form}}</span>
{% else %}
<a class="btn" href="/?date={{date}}&load={{m.fixture.id}}">Ambil Prediksi AI</a>
{% endif %}
</div>
{% endfor %}
</body>
</html>
"""

def get_fixtures(date_str):
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"date": date_str, "timezone": "Asia/Jakarta"}, timeout=15)
    return r.json().get("response", [])[:25]

def get_prediction(fixture_id):
    r = requests.get(f"{BASE_URL}/predictions", headers=HEADERS, params={"fixture": fixture_id}, timeout=15)
    data = r.json()
    if not data.get("response"): return None
    p = data["response"][0]["predictions"]
    return {
        "percent": p["percent"],
        "advice": p.get("advice"),
        "form": data["response"][0]["teams"]["home"]["last_5"] + " vs " + data["response"][0]["teams"]["away"]["last_5"]
    }

@app.route("/")
def index():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    load_id = request.args.get("load")
    matches = get_fixtures(date)
    for m in matches:
        if load_id and str(m["fixture"]["id"]) == load_id:
            m["pred"] = get_prediction(m["fixture"]["id"])
        else:
            if matches.index(m) < 3:
                try: m["pred"] = get_prediction(m["fixture"]["id"])
                except: m["pred"] = None
            else: m["pred"] = None
    return render_template_string(HTML, matches=matches, date=date, today=datetime.now().strftime("%d %b %Y %H:%M"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
