from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BET PRO - Today</title>
<style>
body{background:#0a0f1a;color:#fff;font-family:system-ui;margin:0;padding:16px}
h1{color:#4ade80;margin:0 0 8px;font-size:22px}
.info{opacity:.7;margin-bottom:12px;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:10px 8px;border-bottom:1px solid #222;text-align:left}
th{background:#1e293b;position:sticky;top:0}
button{background:#4ade80;color:#000;border:0;padding:6px 10px;border-radius:6px;font-weight:600}
</style></head><body>
<h1>⚽ Pertandingan Hari Ini</h1>
<div class="info" id="info">Loading...</div>
<table><thead><tr><th>Liga</th><th>Home</th><th>Away</th><th>Skor</th><th>Status</th><th>JSON</th></tr></thead>
<tbody id="tb"></tbody></table>
<script>
async function getData(){
  // coba 3 hari biar pasti ada data (kemarin-hari ini-besok)
  const dates=['20260428','20260429','20260430'];
  let all=[];
  for(const d of dates){
    try{
      const r=await fetch(`https://site.api.espn.com/apis/site/v2/sports/soccer/scoreboard?dates=${d}`);
      const j=await r.json();
      (j.events||[]).forEach(ev=>{
        const c=ev.competitions[0]; const t=c.competitors;
        const h=t.find(x=>x.homeAway==='home'); const a=t.find(x=>x.homeAway==='away');
        all.push({
          match_id:ev.id, league:ev.league?.name||'Unknown', kickoff:ev.date,
          home:h.team.displayName, away:a.team.displayName,
          score:`${h.score||0}-${a.score||0}`, status:c.status.type.shortDetail,
          prediction_input:{home_team:h.team.displayName,away_team:a.team.displayName,league:ev.league?.name}
        });
      });
    }catch(e){}
  }
  return all;
}
async function load(){
  const data=await getData();
  document.getElementById('info').innerText=`Ditemukan ${data.length} pertandingan • ${new Date().toLocaleString('id-ID')}`;
  const tb=document.getElementById('tb'); tb.innerHTML='';
  data.slice(0,100).forEach(o=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${o.league}</td><td>${o.home}</td><td>${o.away}</td><td>${o.score}</td><td>${o.status}</td>
    <td><button onclick='dl(${JSON.stringify(o).replace(/'/g,"")})'>DL</button></td>`;
    tb.appendChild(tr);
  });
}
function dl(o){const b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`${o.home}_vs_${o.away}.json`;a.click();}
load();
</script></body></html>""" const a=document.createElement('a');a.href=URL.createObjectURL(b);
 a.download=`${o.home}_vs_${o.away}.json`;a.click();}
load();
</script></body></html>"""
