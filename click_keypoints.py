"""Hand-click 2D leg and neck keypoints on walking frames, to check the fitted joint angles.

    python click_keypoints.py [--per-track 3] [--port 7861] [--out new_output/keypoint_clicks]

The first run samples frames from the walking tracks in new_output/gait_all_tracks.csv (fixed seed, saved to
<out>/sample.json, so later runs show the same frames). It then serves a page at http://127.0.0.1:<port>. Every
frame is saved to <out>/clicks.json as soon as you leave it, so you can stop and resume at any time.

Per frame you click the 10 points below on the animal in the yellow box, on the side nearer the camera, and say
which side that is (the animal's own left or right). Coordinates are full-image pixels. Points you cannot see are
stored as null. The SMAL joint each point stands for is in clicks.json ("smal_joint"), for the later comparison
with the projected fitted joints.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NEW = ROOT / "new_output"

# (key, label, hint, SMAL joint index as used in animal3d/kinematics.py)
POINTS = [
    ("shoulder", "Shoulder joint", "point of the shoulder, top of the upper foreleg", 7),
    ("elbow", "Elbow", "back of the foreleg where it leaves the chest", 8),
    ("carpus", "Carpus (front 'knee')", "the joint halfway down the foreleg", 9),
    ("fore_fetlock", "Front fetlock", "the joint just above the front hoof / paw", 10),
    ("hip", "Hip joint", "top of the thigh, below the point of the hip", 17),
    ("stifle", "Stifle (hind 'knee')", "front of the thigh, where it meets the belly", 18),
    ("hock", "Hock", "the backward-pointing joint of the hind leg", 19),
    ("hind_fetlock", "Hind fetlock", "the joint just above the hind hoof / paw", 20),
    ("withers", "Withers / neck base", "top of the shoulders, where the neck starts", 15),
    ("poll", "Poll / head", "top of the head between the ears", 16),
]


def run_dir(run: str) -> Path:
    for base in (NEW / "labels" / "runs", NEW / "fits_existing"):
        if (base / run).is_dir():
            return base / run
    raise FileNotFoundError(run)


def image_dir(run: str) -> Path:
    if (NEW / "labels" / "inputs" / run / "rgb").is_dir():
        return NEW / "labels" / "inputs" / run / "rgb"
    name = json.loads((run_dir(run) / "summary.json").read_text())["input"]["name"]
    return ROOT / "input_30s" / name / "rgb"


def frame_file(folder: Path, frame: int) -> Path:
    """Frames are named by number, zero-padded in some clips, as .jpg or .png."""
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png") and path.stem.isdigit() and int(path.stem) == frame:
            return path
    raise FileNotFoundError(f"{folder}: frame {frame}")


def make_sample(per_track: int, seed: int = 0) -> list[dict]:
    """A few well-visible frames spread evenly over each walking bout."""
    rng = np.random.default_rng(seed)
    tracks = pd.read_csv(NEW / "gait_all_tracks.csv")
    tracks = tracks[tracks.walking.astype(bool)]
    items = []
    for t in tracks.itertuples():
        det = pd.read_csv(run_dir(t.run) / "detections.csv")
        det = det[(det.track_id == t.track_id) & det.name.between(t.first_frame, t.last_frame)]
        good = det[~det.truncated.astype(bool) & ~det.overlapping.astype(bool) & (det.box_h >= 100)]
        det = (good if len(good) >= per_track else det).sort_values("name")
        if det.empty:
            continue
        edges = np.linspace(0, len(det), per_track + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            row = det.iloc[int(rng.integers(a, b))]
            items.append({
                "id": f"{t.run}/{int(t.track_id)}/{int(row['name'])}", "run": t.run, "species": t.species,
                "track_id": int(t.track_id), "frame": int(row["name"]),
                "image": str(frame_file(image_dir(t.run), int(row["name"]))),
                "box_xyxy": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
            })
    order = rng.permutation(len(items))  # mixed species order, so tiredness does not bias one species
    return [items[i] for i in order]


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Keypoint clicks</title>
<style>
:root{--bg:#16181c;--panel:#20232a;--ink:#e8e8e8;--muted:#9aa0aa;--acc:#ffcc33;--done:#5fd38d;--skip:#e06c75}
body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#view{flex:1;position:relative;cursor:crosshair}
canvas{display:block;width:100%;height:100%}
#side{width:330px;background:var(--panel);padding:12px 14px;overflow:auto;box-sizing:border-box}
h2{margin:0 0 4px;font-size:16px} .muted{color:var(--muted)} .row{display:flex;gap:6px;margin:8px 0;flex-wrap:wrap}
button{background:#333844;color:var(--ink);border:1px solid #444a57;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}
button.on{background:var(--acc);color:#111;border-color:var(--acc)}
ol{padding-left:0;list-style:none;margin:8px 0}
li{padding:5px 8px;border-radius:6px;margin:2px 0;cursor:pointer;border:1px solid transparent}
li.cur{border-color:var(--acc);background:#2c2f37} li .h{display:block;font-size:12px;color:var(--muted)}
li.done::before{content:"\2713 ";color:var(--done)} li.skip::before{content:"\2715 ";color:var(--skip)}
kbd{background:#333844;border-radius:4px;padding:1px 5px;font-size:12px}
#msg{min-height:18px;color:var(--acc)} .bar{height:6px;background:#333844;border-radius:3px;overflow:hidden} .bar>div{height:100%;background:var(--done)}
</style></head><body>
<div id="view"><canvas id="c"></canvas></div>
<div id="side">
 <h2 id="title">…</h2><div class="muted" id="sub"></div>
 <div class="bar" style="margin-top:8px"><div id="prog"></div></div><div class="muted" id="count"></div>
 <div class="row"><span>Near side (animal's own):</span>
  <button id="sideL">Left <kbd>L</kbd></button><button id="sideR">Right <kbd>R</kbd></button></div>
 <ol id="pts"></ol>
 <div class="row"><button id="prev">&larr; Prev</button><button id="next">Save &amp; next &rarr;</button><button id="nextTodo">Next unfinished</button></div>
 <div class="row"><button id="bad">Frame unusable</button></div>
 <div id="msg"></div>
 <p class="muted">Left-click: place the highlighted point. <kbd>X</kbd> not visible. <kbd>Backspace</kbd> undo.
 Click a list entry to redo it. Wheel: zoom. Right-drag: pan. <kbd>F</kbd> refit view.
 <kbd>Enter</kbd>/<kbd>&rarr;</kbd> save &amp; next, <kbd>&larr;</kbd> previous. Saved on every frame change.</p>
</div>
<script>
const POINTS=__POINTS__; let items=[], clicks={}, idx=0, cur=0, img=new Image(), view={s:1,x:0,y:0}, drag=null;
const cv=document.getElementById('c'), ctx=cv.getContext('2d'), $=id=>document.getElementById(id);
function rec(){const it=items[idx]; if(!clicks[it.id]) clicks[it.id]={side:null,unusable:false,points:{}}; return clicks[it.id];}
function finished(r){return r && (r.unusable || (r.side && POINTS.every(p=>p[0] in r.points)));}
async function load(){const r=await (await fetch('state')).json(); items=r.items; clicks=r.clicks;
 idx=items.findIndex(it=>!finished(clicks[it.id])); if(idx<0) idx=0; show();}
function firstTodo(r){const k=POINTS.findIndex(p=>!(p[0] in r.points)); return k<0?0:k;}
function show(){const it=items[idx], r=rec(); cur=firstTodo(r);
 img=new Image(); img.onload=()=>{fit(); draw();}; img.src='img?i='+idx; ui();}
function fit(){const W=cv.clientWidth,H=cv.clientHeight,[x1,y1,x2,y2]=items[idx].box_xyxy, pad=0.35;
 const bw=(x2-x1)*(1+2*pad), bh=(y2-y1)*(1+2*pad); view.s=Math.min(W/bw,H/bh);
 view.x=(x1+x2)/2-W/2/view.s; view.y=(y1+y2)/2-H/2/view.s;}
function ui(){const it=items[idx], r=rec(); $('title').textContent=`${it.species} — ${it.run}`;
 $('sub').textContent=`track ${it.track_id}, frame ${it.frame}  (${idx+1} / ${items.length})`;
 const n=items.filter(t=>finished(clicks[t.id])).length; $('count').textContent=`${n} of ${items.length} frames finished`;
 $('prog').style.width=(100*n/items.length)+'%';
 $('sideL').className=r.side==='left'?'on':''; $('sideR').className=r.side==='right'?'on':''; $('bad').className=r.unusable?'on':'';
 $('pts').innerHTML=POINTS.map((p,k)=>{const v=r.points[p[0]]; const c=(k===cur?'cur ':'')+(v===undefined?'':v===null?'skip':'done');
  return `<li class="${c}" data-k="${k}">${k+1}. ${p[1]}<span class="h">${p[2]}</span></li>`}).join('');
 document.querySelectorAll('#pts li').forEach(li=>li.onclick=()=>{cur=+li.dataset.k; ui(); draw();});}
function draw(){const W=cv.width=cv.clientWidth, H=cv.height=cv.clientHeight; ctx.fillStyle='#111'; ctx.fillRect(0,0,W,H);
 ctx.setTransform(view.s,0,0,view.s,-view.x*view.s,-view.y*view.s); if(img.complete) ctx.drawImage(img,0,0);
 const [x1,y1,x2,y2]=items[idx].box_xyxy; ctx.lineWidth=2/view.s; ctx.strokeStyle='rgba(255,204,51,.8)'; ctx.setLineDash([8/view.s,6/view.s]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
 const r=rec(), P=k=>r.points[POINTS[k][0]];
 for(const chain of [[0,1,2,3],[4,5,6,7],[8,9]]){ctx.beginPath(); let on=false;
  for(const k of chain){const v=P(k); if(v){on?ctx.lineTo(v[0],v[1]):ctx.moveTo(v[0],v[1]); on=true;} else on=false;}
  ctx.strokeStyle='rgba(95,211,141,.9)'; ctx.lineWidth=2/view.s; ctx.stroke();}
 POINTS.forEach((p,k)=>{const v=P(k); if(!v) return; ctx.beginPath(); ctx.arc(v[0],v[1],5/view.s,0,7);
  ctx.fillStyle=k===cur?'#ffcc33':'#5fd38d'; ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=1.5/view.s; ctx.stroke();
  ctx.fillStyle='#fff'; ctx.font=`${13/view.s}px system-ui`; ctx.fillText(String(k+1),v[0]+7/view.s,v[1]-7/view.s);});
 ctx.setTransform(1,0,0,1,0,0);}
function setPoint(v){const r=rec(); r.points[POINTS[cur][0]]=v; r.unusable=false;
 const after=POINTS.findIndex((p,k)=>k>cur&&!(p[0] in r.points)), any=POINTS.findIndex(p=>!(p[0] in r.points));
 cur=after>=0?after:any>=0?any:cur; ui(); draw();}
async function save(){const it=items[idx]; const r=await fetch('save',{method:'POST',body:JSON.stringify({id:it.id,record:rec()})});
 $('msg').textContent=r.ok?`saved ${it.frame}`:'SAVE FAILED';}
async function go(d){await save(); idx=Math.max(0,Math.min(items.length-1,idx+d)); show();}
cv.addEventListener('mousedown',e=>{if(e.button===2||e.button===1){drag={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y}; return;}
 if(e.button!==0) return; const b=cv.getBoundingClientRect(); setPoint([view.x+(e.clientX-b.left)/view.s, view.y+(e.clientY-b.top)/view.s].map(v=>Math.round(v*10)/10));});
window.addEventListener('mousemove',e=>{if(!drag) return; view.x=drag.vx-(e.clientX-drag.x)/view.s; view.y=drag.vy-(e.clientY-drag.y)/view.s; draw();});
window.addEventListener('mouseup',()=>drag=null); cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{e.preventDefault(); const b=cv.getBoundingClientRect(), mx=e.clientX-b.left, my=e.clientY-b.top;
 const wx=view.x+mx/view.s, wy=view.y+my/view.s; view.s*=Math.exp(-e.deltaY*0.0015); view.x=wx-mx/view.s; view.y=wy-my/view.s; draw();},{passive:false});
window.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
 if(k==='x') setPoint(null); else if(k==='backspace'){e.preventDefault(); const r=rec(); const done=POINTS.map(p=>p[0]).filter(n=>n in r.points); if(done.length){const last=cur>0&&POINTS[cur-1][0] in r.points?POINTS[cur-1][0]:done[done.length-1]; delete r.points[last]; cur=POINTS.findIndex(p=>p[0]===last); ui(); draw();}}
 else if(k==='l'){rec().side='left'; ui();} else if(k==='r'){rec().side='right'; ui();}
 else if(k==='f'){fit(); draw();} else if(k==='enter'||k==='arrowright') go(1); else if(k==='arrowleft') go(-1);});
$('sideL').onclick=()=>{rec().side='left'; ui();}; $('sideR').onclick=()=>{rec().side='right'; ui();};
$('prev').onclick=()=>go(-1); $('next').onclick=()=>go(1);
$('nextTodo').onclick=async()=>{await save(); const j=items.findIndex((t,i)=>i>idx&&!finished(clicks[t.id])); const k=j>=0?j:items.findIndex(t=>!finished(clicks[t.id])); if(k>=0){idx=k; show();} else $('msg').textContent='all frames finished';};
$('bad').onclick=()=>{const r=rec(); r.unusable=!r.unusable; ui();};
window.addEventListener('resize',draw); window.addEventListener('beforeunload',()=>navigator.sendBeacon('save',JSON.stringify({id:items[idx].id,record:rec()})));
load();
</script></body></html>"""


def serve(out: Path, items: list[dict], port: int) -> None:
    clicks_path = out / "clicks.json"
    doc = json.loads(clicks_path.read_text()) if clicks_path.exists() else {}
    doc.setdefault("points", [{"key": k, "label": l, "smal_joint": j} for k, l, _, j in POINTS])
    doc.setdefault("note", "pixel coordinates in the full image; null = not visible; side = animal's own side nearer the camera")
    doc.setdefault("frames", {})
    page = PAGE.replace("__POINTS__", json.dumps([[k, l, h] for k, l, h, _ in POINTS])).encode()
    by_id = {it["id"]: it for it in items}

    def write() -> None:
        tmp = clicks_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, clicks_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def send(self, body: bytes, kind: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(page, "text/html; charset=utf-8")
            if self.path == "/state":
                state = {"items": items, "clicks": {k: v["record"] for k, v in doc["frames"].items()}}
                return self.send(json.dumps(state).encode(), "application/json")
            if self.path.startswith("/img?i="):
                i = int(self.path.split("=")[1])
                path = Path(items[i]["image"])
                return self.send(path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/jpeg")
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/save":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            item = by_id[body["id"]]
            doc["frames"][body["id"]] = {**{k: item[k] for k in ("run", "species", "track_id", "frame", "image", "box_xyxy")},
                                         "record": body["record"]}
            write()
            self.send(b"ok", "text/plain")

    write()
    print(f"open http://127.0.0.1:{port}  ({len(items)} frames, saving to {clicks_path})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-track", type=int, default=3)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--out", type=Path, default=NEW / "keypoint_clicks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sample_path = args.out / "sample.json"
    if not sample_path.exists():
        sample_path.write_text(json.dumps(make_sample(args.per_track), indent=1))
    items = json.loads(sample_path.read_text())
    serve(args.out, items, args.port)


if __name__ == "__main__":
    main()
