from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BET PRO Today</title>
<style>body{background:#0a0f1a;color:#fff;font-family:system-ui;padding:16px}
h1{color:#4ade80}table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{padding:10px;border-bottom:1px solid #333}th{background:#1e293b}
button{background:#4ade80;color:#000;border:0;padding:5px 10px;border-radius:5px}</style>
</head><body>
<h1>⚽ Pertandingan Hari Ini</h1><div id="info"></div>
<table><thead><tr><th>Liga</th><th>Home</th><th>Away</th><th>Skor</th><th>Status</th><th>JSON</th></tr></thead><tbody id="tb"></tbody></table>
<script>
async function load(){
 const d=new Date().toISOString().slice(0,10).replace(/-/g,'');
 const url=`https://site.api.espn.com/apis/site/v2/sports/soccer/scoreboard?dates=${d}`;
 const res=await fetch(url); const j=await res.json();
 document.getElementById('info').innerText='Data: '+new Date().toLocaleString('id-ID');
 const tb=document.getElementById('tb'); tb.innerHTML='';
 (j.events||[]).forEach(ev=>{
  const c=ev.competitions[0]; const t=c.competitors;
  const h=t.find(x=>x.homeAway==='home'); const a=t.find(x=>x.homeAway==='away');
  const obj={match_id:ev.id, league:ev.league.name, kickoff:ev.date,
    home:h.team.displayName, away:a.team.displayName,
    score:`${h.score||0}-${a.score||0}`, status:c.status.type.shortDetail,
    prediction_input:{home_team:h.team.displayName,away_team:a.team.displayName,league:ev.league.name}};
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${obj.league}</td><td>${obj.home}</td><td>${obj.away}</td><td>${obj.score}</td><td>${obj.status}</td>
  <td><button onclick='dl(${JSON.stringify(obj)})'>DL</button></td>`;
  tb.appendChild(tr);
 });
}
function dl(o){const b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);
 a.download=`${o.home}_vs_${o.away}.json`;a.click();}
load();
</script></body></html>"""
