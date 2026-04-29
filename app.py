from flask import Flask
import requests, os
from datetime import datetime
import pytz
app=Flask(__name__)
KEY=os.getenv("KEY") or os.getenv("API_KEY","")
H={'x-apisports-key':KEY}
JST=pytz.timezone('Asia/Tokyo')
def ai(h,a):
 total=h+a+0.1; hp=(h/total)*100; ap=(a/total)*100
 if abs(hp-ap)<10: return f"AI: Draw {50+abs(hp-ap)/2:.0f}%"
 return f"AI: Home {hp:.0f}%" if hp>ap else f"AI: Away {ap:.0f}%"
@app.route("/")
def home():
 try:
  r=requests.get("https://v3.football.api-sports.io/fixtures?live=all",headers=H,timeout=10).json()
  games=[m for m in r.get("response",[]) if m["league"]["country"] in ["Japan","South Korea","China","Australia"]]
  now=datetime.now(JST).strftime("%H:%M JST")
 except: games=[]; now=""
 html=f"<html><head><meta name=viewport content='width=device-width'><style>body{{font-family:Arial;background:#fff;color:#000;padding:15px}} .m{{border-bottom:1px solid #ddd;padding:12px 0}} .t{{font-weight:bold}} .s{{color:#d00;font-weight:bold}} .a{{color:#080;font-size:13px}}</style></head><body><h2>⚽ BET PRO LIVE ({now})</h2>"
 if not games: html+="<p>Tidak ada laga Asia live saat ini.</p>"
 for m in games:
  h=m["teams"]["home"]["name"]; a=m["teams"]["away"]["name"]
  hs=m["goals"]["home"]; aw=m["goals"]["away"]; el=m["fixture"]["status"]["elapsed"]
  html+=f"<div class=m><div class=t>{h} vs {a}</div><div class=s>{hs}-{aw} ({el}')</div><div class=a>{ai(hs,aw)}</div></div>"
 html+="</body></html>"; return html
if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.environ.get("PORT",8080)))</body></html>"""

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
