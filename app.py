from flask import Flask, jsonify, render_template_string
import requests
from datetime import datetime

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BET PRO - Today Matches</title>
<style>
body{font-family:system-ui;background:#0a0f1a;color:#fff;padding:20px}
h1{color:#4ade80}
table{width:100%;border-collapse:collapse;margin-top:20px}
th,td{padding:12px;border-bottom:1px solid #333;text-align:left}
th{background:#1e293b}
button{background:#4ade80;color:#000;border:none;padding:6px 12px;border-radius:6px;cursor:pointer}
button:hover{opacity:.8}
</style>
</head>
<body>
<h1>⚽ Pertandingan Hari Ini</h1>
<p id="date"></p>
<table id="tbl"><thead><tr><th>Liga</th><th>Home</th><th>Away</th><th>Skor</th><th>Status</th><th>Download</th></tr></thead><tbody></tbody></table>

<script>
async function load(){
  const res = await fetch('/api/today');
  const data = await res.json();
  document.getElementById('date').innerText = new Date().toLocaleString('id-ID');
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML='';
  data.forEach((m,i)=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${m.league}</td><td>${m.home}</td><td>${m.away}</td><td>${m.score}</td><td>${m.status}</td>
    <td><button onclick='download(${JSON.stringify(m).replace(/'/g,"&#39;")})'>JSON</button></td>`;
    tb.appendChild(tr);
  });
}
function download(obj){
  const blob = new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${obj.home}_vs_${obj.away}.json`;
  a.click();
}
load();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/api/today")
def api_today():
    date = datetime.utcnow().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/scoreboard?dates={date}"
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10).json()
    out = []
    for ev in r.get("events", []):
        comp = ev["competitions"][0]
        teams = comp["competitors"]
        home = next(t for t in teams if t["homeAway"]=="home")
        away = next(t for t in teams if t["homeAway"]=="away")
        # ambil statistik kalau live
        stats = {}
        if comp.get("status",{}).get("type",{}).get("state")=="in":
            for s in comp.get("statistics", []):
                stats[s["name"]] = { "home": s.get("home"), "away": s.get("away") }
        out.append({
            "match_id": ev["id"],
            "league": ev.get("league",{}).get("name"),
            "kickoff": ev["date"],
            "home": home["team"]["displayName"],
            "away": away["team"]["displayName"],
            "score": f"{home.get('score','0')}-{away.get('score','0')}",
            "status": comp["status"]["type"]["shortDetail"],
            "stats": stats,
            "prediction_input": {
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "league": ev.get("league",{}).get("name"),
                "minute": comp["status"].get("displayClock","0'"),
                "score_home": int(home.get('score',0)),
                "score_away": int(away.get('score',0))
            }
        })
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)    return today()
