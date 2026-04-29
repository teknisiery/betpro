import os, requests
from flask import Flask, render_template_string
from datetime import datetime
import pytz

app = Flask(__name__)
KEY = os.getenv("API_KEY","")
H = {"x-apisports-key": KEY}
WIB = pytz.timezone("Asia/Jakarta")

HTML = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>BET PRO LIVE</title>
<style>
body{font-family:system-ui;background:#fff;color:#111;margin:0}
.header{background:#ff6a00;color:#fff;padding:12px;font-weight:bold;text-align:center}
.match{border-bottom:1px solid #eee;padding:10px 12px}
.top{display:flex;justify-content:space-between;font-size:13px;color:#8a5a00}
.teams{font-size:16px;margin:4px 0}
.score{float:right;font-size:20px;font-weight:bold}
.live{color:#e60000}.ht{color:#007bff}
.ai{margin-top:6px;font-size:12px;color:#006400;background:#f0fff0;padding:4px;border-radius:4px}
</style></head><body>
<div class=header>BET PRO ⚽ LIVE — {{now}}</div>
{% for m in matches %}
<div class=match>
  <div class=top>
    <span>{{m.league}}</span>
    <span>{{m.time}}</span>
    <span class="{{'ht' if m.status=='HT' else 'live'}}">{{m.status}}</span>
  </div>
  <div class=teams>{{m.home}}<span class=score>{{m.gh}}</span></div>
  <div class=teams>{{m.away}}<span class=score>{{m.ga}}</span></div>
  <div class=ai>AI: Home {{m.ph}}% | Draw {{m.px}}% | Away {{m.pa}}% → <b>{{m.rec}}</b></div>
</div>
{% else %}<p style="padding:20px">Tidak ada live Jepang saat ini. Coba lagi nanti.</p>{% endfor %}
</body></html>"""

def fetch():
    if not KEY: return []
    r = requests.get("https://v3.football.api-sports.io/fixtures",
                     headers=H, params={"live":"all"}, timeout=20).json()
    out=[]
    for f in r.get("response",[]):
        if f["league"]["country"]!="Japan": continue
        fid=f["fixture"]["id"]
        pr = requests.get("https://v3.football.api-sports.io/predictions",
                          headers=H, params={"fixture":fid}, timeout=15).json()
        p = pr.get("response",[{}])[0].get("predictions",{}).get("percent",{})
        ph = int(p.get("home","0%").replace("%","") or 0)
        px = int(p.get("draw","0%").replace("%","") or 0)
        pa = int(p.get("away","0%").replace("%","") or 0)
        rec = "HOME" if ph>pa and ph>px else "AWAY" if pa>ph and pa>px else "DRAW"
        st=f["fixture"]["status"]
        status="HT" if st["short"]=="HT" else f"{st['elapsed']}'"
        out.append({
            "league":f"JPN {'D1' if 'J1' in f['league']['name'] else 'D2'}",
            "time":datetime.fromisoformat(f["fixture"]["date"]).astimezone(WIB).strftime("%H:%M"),
            "status":status,"home":f["teams"]["home"]["name"],"away":f["teams"]["away"]["name"],
            "gh":f["goals"]["home"] or 0,"ga":f["goals"]["away"] or 0,
            "ph":ph,"px":px,"pa":pa,"rec":rec
        })
    return out

@app.route("/")
def index():
    return render_template_string(HTML, matches=fetch(),
        now=datetime.now(WIB).strftime("%d %b %H:%M WIB"))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
