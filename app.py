from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BET PRO</title>
<style>body{background:#0a0f1a;color:#fff;font-family:system-ui;padding:14px}
h1{color:#4ade80;font-size:20px}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}
th,td{padding:8px;border-bottom:1px solid #333}th{background:#1e293b}
button{background:#4ade80;color:#000;border:0;padding:5px 8px;border-radius:5px}</style>
</head><body>
<h1>⚽ Pertandingan Hari Ini</h1>
<div id="info">Loading...</div>
<table><thead><tr><th>Liga</th><th>Home</th><th>Away</th><th>Skor</th><th>Status</th><th>JSON</th></tr></thead>
<tbody id="tb"></tbody></table>
<script>
let DATA=[];
async function load(){
 const days=['20260428','20260429','20260430'];
 for(const d of days){
   try{
     const r=await fetch('https://site.api.espn.com/apis/site/v2/sports/soccer/scoreboard?dates='+d);
     const j=await r.json();
     (j.events||[]).forEach(ev=>{
       const c=ev.competitions[0]; const t=c.competitors;
       const h=t.find(x=>x.homeAway==='home'); const a=t.find(x=>x.homeAway==='away');
       DATA.push({id:ev.id,league:ev.league.name,home:h.team.displayName,away:a.team.displayName,
         score:(h.score||0)+'-'+(a.score||0),status:c.status.type.shortDetail,kickoff:ev.date});
     });
   }catch(e){}
 }
 document.getElementById('info').innerText='Total: '+DATA.length+' laga';
 const tb=document.getElementById('tb'); tb.innerHTML='';
 DATA.forEach((o,i)=>{
   const tr=document.createElement('tr');
   tr.innerHTML='<td>'+o.league+'</td><td>'+o.home+'</td><td>'+o.away+'</td><td>'+o.score+'</td><td>'+o.status+'</td><td><button onclick="dl('+i+')">DL</button></td>';
   tb.appendChild(tr);
 });
}
function dl(i){const o=DATA[i]; const b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download=o.home+'_vs_'+o.away+'.json'; a.click();}
load();
</script></body></html>'''          home:h.team.displayName, away:a.team.displayName,
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
