#!/usr/bin/env python3
"""
HubWeb — a browser console for the MOSS hub (the primary interface on the cluster,
where X11 GUIs are painful). Serves a self-contained page + crop images over a
stdlib ThreadingHTTPServer; the page POLLS /state (~1.5s) and re-renders. Control
actions (toggle / model / reset / discard / start / stop) are marshaled onto the
Qt main thread via a signal, because the backend touches QTimer/QThread which are
not safe to call from the HTTP worker thread.

No third-party deps. Open http://<host>:<port>/ in a browser.

Routes:
    GET  /                     the console page
    GET  /state                JSON snapshot (HubServer.web_state())
    GET  /crops/<uid>          JSON list of a user's crop filenames
    GET  /crop/<uid>/<name>    a crop image PNG
    GET  /mask/<uid>/<name>    a crop mask PNG
    POST /toggle {uid,included}    include/exclude a user's crops
    POST /model  {arch}            set the authoritative session model
    POST /reset                    reset the model
    POST /discard {uid,name}       discard a crop
    POST /train/start | /train/stop   operator start/stop training
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

from PyQt6.QtCore import QObject, pyqtSignal

_UID_RE = re.compile(r"^[A-Za-z0-9_]+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.png$")


class HubWeb(QObject):
    # Emitted from the HTTP thread; delivered (queued) to _dispatch on the main thread.
    _action = pyqtSignal(str, str)   # kind, json-arg

    def __init__(self, hub, host: str = "0.0.0.0", port: int = 8080, parent=None):
        super().__init__(parent)
        self.hub = hub
        self.host = host
        self.port = port
        self._httpd = None
        self._thread = None
        self._action.connect(self._dispatch)   # auto-queued to the main thread

    # -------------------------------------------------------------- lifecycle
    def start(self):
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        from ..network.session import get_local_ip
        ip = get_local_ip() if self.host in ("0.0.0.0", "::", "") else self.host
        print(f"[HubWeb] console at  http://{ip}:{self.port}/")

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()

    # ----------------------------------------------- control actions (main thread)
    def _dispatch(self, kind: str, arg: str):
        try:
            d = json.loads(arg) if arg else {}
        except Exception:
            d = {}
        try:
            if kind == "toggle":
                self.hub.set_user_included(d["uid"], bool(d["included"]))
            elif kind == "model":
                if d.get("arch"):
                    self.hub.set_prediction_model(d["arch"])
            elif kind == "reset":
                self.hub.reset_model()
            elif kind == "discard":
                self.hub.discard_crop(d["uid"], d["name"])
            elif kind == "start":
                self.hub.start_training()
            elif kind == "stop":
                self.hub.stop_training()
            elif kind == "select":
                self.hub.select_project(d.get("name", ""), fresh=False)
            elif kind == "new":
                self.hub.new_project()
        except Exception as e:
            print(f"[HubWeb] action {kind} failed: {e}")

    # ----------------------------------------------------------------- handler
    def _make_handler(self):
        hub = self.hub
        web = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # quiet

            def _send(self, code, ctype, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
                elif path == "/state":
                    self._send(200, "application/json",
                               json.dumps(hub.web_state()).encode("utf-8"))
                elif path.startswith("/crops/"):
                    uid = unquote(path[len("/crops/"):]).strip("/")
                    names = []
                    if _UID_RE.match(uid):
                        d = hub.data_dir / "incoming" / uid / "train_images"
                        if d.exists():
                            names = sorted((f.name for f in d.glob("*.png")), reverse=True)[:500]
                    self._send(200, "application/json", json.dumps(names).encode("utf-8"))
                elif path.startswith("/crop/") or path.startswith("/mask/"):
                    sub = "train_images" if path.startswith("/crop/") else "train_masks"
                    parts = unquote(path[6:]).split("/", 1)   # len('/crop/')==len('/mask/')==6
                    if len(parts) == 2 and _UID_RE.match(parts[0]) and _NAME_RE.match(parts[1]):
                        f = hub.data_dir / "incoming" / parts[0] / sub / parts[1]
                        if f.exists():
                            self._send(200, "image/png", f.read_bytes())
                            return
                    self._send(404, "text/plain", b"not found")
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                path = urlparse(self.path).path
                ln = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(ln).decode("utf-8") if ln else ""
                kind = {"/toggle": "toggle", "/model": "model",
                        "/reset": "reset", "/discard": "discard",
                        "/train/start": "start", "/train/stop": "stop",
                        "/select": "select", "/new": "new"}.get(path)
                if kind:
                    web._action.emit(kind, body)
                    self._send(200, "application/json", b'{"ok":true}')
                else:
                    self._send(404, "text/plain", b"not found")

        return Handler


# ============================================================ the page (inlined)
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOSS Hub</title>
<style>
  :root{
    --bg:#141417; --panel:#1c1c21; --inset:#232329; --border:#2d2d34;
    --ink:#ececed; --muted:#8b8b94; --faint:#5f5f68;
    --accent:#4d90e6; --amber:#f2c14e; --green:#5fbf6a; --danger:#d3574d;
    --r:12px; --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
    color-scheme:dark;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;}
  body{background:var(--bg);color:var(--ink);
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  .cap{color:var(--faint);font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;}
  .muted{color:var(--muted);} .mono{font-family:var(--mono);}
  button{font:inherit;border:none;border-radius:8px;cursor:pointer;color:#fff;
    background:var(--accent);padding:9px 14px;font-weight:600;transition:.12s;}
  button:hover{filter:brightness(1.12);}
  button.ghost{background:transparent;border:1px solid var(--border);color:var(--muted);}
  button.ghost:hover{color:var(--ink);border-color:var(--faint);background:transparent;filter:none;}
  button.danger{background:transparent;border:1px solid #5a3230;color:#e08a82;}
  button.danger:hover{background:#3a2220;filter:none;}
  button.mini{padding:3px 9px;font-size:11px;font-weight:600;background:var(--inset);color:var(--muted);border:1px solid var(--border);}
  button.mini:hover{color:var(--ink);filter:none;}

  /* top bar */
  #top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:28px;flex-wrap:wrap;
    padding:14px 20px;background:rgba(16,16,19,.92);backdrop-filter:blur(8px);
    border-bottom:1px solid var(--border);}
  .brand{font-size:17px;font-weight:800;letter-spacing:.5px;color:var(--accent);}
  .brand span{color:var(--muted);font-weight:600;}
  .addr-row{display:flex;align-items:center;gap:8px;}
  .addr{font-family:var(--mono);font-size:21px;font-weight:800;color:var(--amber);letter-spacing:.5px;}
  .code{font-family:var(--mono);font-size:14px;color:#a99340;letter-spacing:2px;}
  .proj{font-size:15px;font-weight:700;}
  .pill{margin-left:auto;display:flex;align-items:center;gap:7px;background:var(--inset);
    border:1px solid var(--border);border-radius:999px;padding:5px 13px;font-weight:600;font-size:13px;}
  .pill .dot{width:8px;height:8px;border-radius:50%;background:var(--green);}

  /* layout */
  #body{display:grid;grid-template-columns:1fr 320px;gap:16px;padding:16px;align-items:start;}
  @media(max-width:880px){#body{grid-template-columns:1fr;}}
  #left{display:flex;flex-direction:column;gap:16px;min-width:0;}
  #right{display:flex;flex-direction:column;gap:16px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:16px;}
  .card > h3{margin:0 0 12px;font-size:12px;font-weight:700;letter-spacing:.4px;color:var(--muted);text-transform:uppercase;}

  /* tiles */
  #tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;}
  .tile{background:var(--inset);border:1px solid var(--border);border-left:3px solid var(--faint);
    border-radius:10px;padding:10px;cursor:pointer;transition:.12s;position:relative;}
  .tile:hover{transform:translateY(-1px);border-color:var(--faint);}
  .tile .row{display:flex;align-items:center;gap:6px;height:16px;}
  .tile .crown{color:var(--amber);font-size:10px;font-weight:700;flex:1;overflow:hidden;white-space:nowrap;}
  .tile .dot{width:9px;height:9px;border-radius:50%;background:var(--faint);}
  .tile .dot.on{background:var(--green);animation:blink 1.5s ease-in-out infinite;}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
  .tile .av{display:block;margin:6px auto 4px;width:76px;height:76px;}
  .tile.off .av{filter:grayscale(1);opacity:.45;}
  .tile .nm{text-align:center;font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .tile.off .nm{color:var(--muted);}
  .tile .cc{text-align:center;color:var(--muted);font-size:11px;margin-top:1px;}
  .tile .inc{position:absolute;top:9px;right:9px;accent-color:var(--accent);cursor:pointer;}
  #empty{color:var(--faint);text-align:center;padding:26px;font-size:13px;}

  /* training panel */
  .stat{display:flex;justify-content:space-between;padding:3px 0;font-size:13px;}
  .stat b{font-family:var(--mono);font-weight:600;color:var(--ink);}
  .runchip{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;}
  .runchip .d{width:8px;height:8px;border-radius:50%;background:var(--faint);}
  .runchip.on .d{background:var(--green);animation:blink 1.5s infinite;}
  #trainbtn{width:100%;margin-top:12px;padding:11px;font-size:14px;}
  #resetbtn{width:100%;margin-top:8px;}
  select{width:100%;background:var(--inset);color:var(--ink);border:1px solid var(--border);
    padding:8px;border-radius:8px;font:inherit;}

  /* loss plot */
  .losshead{display:flex;align-items:baseline;gap:10px;margin-bottom:8px;}
  .lossnum{font-family:var(--mono);font-size:26px;font-weight:800;color:var(--green);}
  #loss{width:100%;height:230px;display:block;}

  /* review modal */
  #modal{position:fixed;inset:0;background:rgba(8,8,10,.8);display:none;align-items:center;justify-content:center;z-index:20;}
  #mcard{background:var(--panel);border:1px solid var(--border);border-radius:14px;width:min(92vw,720px);padding:18px;}
  #mhead{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
  #mhead h3{margin:0;flex:1;font-size:16px;}
  #mview{display:flex;gap:16px;justify-content:center;}
  .revcol{flex:1;max-width:300px;}
  .revlbl{text-align:center;color:var(--muted);font-size:11px;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px;}
  .review{width:100%;aspect-ratio:1;object-fit:contain;background:#000;border:1px solid var(--border);border-radius:8px;image-rendering:pixelated;}
  #mctrl{display:flex;gap:8px;justify-content:center;margin-top:14px;}
  #mhint{text-align:center;color:var(--faint);font-size:11px;margin-top:8px;}
  #picker{display:none;position:fixed;inset:0;background:var(--bg);z-index:50;align-items:flex-start;justify-content:center;padding:64px 16px;overflow:auto;}
  .pcard{width:100%;max-width:460px;background:var(--card);border:1px solid var(--border);border-radius:16px;padding:26px 28px;}
  .pcard h2{margin:0 0 2px;font-size:17px;font-weight:600;}
  .prow{display:flex;align-items:center;gap:12px;padding:12px 14px;border:1px solid var(--border);border-radius:10px;margin-bottom:8px;transition:border-color .12s;}
  .prow:hover{border-color:var(--accent);}
  .pinfo{display:flex;flex-direction:column;gap:2px;flex:1;min-width:0;}
  .pinfo b{font-size:14px;}
  .pinfo .muted{font-size:11px;}
  .prow button,.pnew button{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:7px 14px;font-size:12px;font-weight:600;cursor:pointer;}
  .prow button:hover,.pnew button:hover{filter:brightness(1.1);}
  .pnew{display:flex;gap:8px;margin-top:16px;}
  .pnew button{flex:1;padding:11px;font-size:13px;}
</style></head>
<body>
<div id="top">
  <div class="brand">MOSS<span>·HUB</span></div>
  <div><div class="cap">Connect address · share this</div>
    <div class="addr-row"><span class="addr" id="addr">—</span>
      <button class="mini" onclick="copyAddr()">copy</button></div></div>
  <div><div class="cap">Session code</div><div class="code" id="code">—</div></div>
  <div><div class="cap">Project</div><div class="proj" id="proj">—</div>
    <div class="muted" id="subproj" style="font-size:11px"></div></div>
  <div class="pill"><span class="dot"></span><span id="count">0 connected</span></div>
</div>

<div id="body">
  <div id="left">
    <div class="card"><h3>Connected users</h3>
      <div id="tiles"></div>
      <div id="empty">No users connected yet — share the connect address to invite collaborators.</div>
      <div class="muted" style="font-size:11px;margin-top:10px">Click a tile to review that user's crops. Uncheck to exclude them from training.</div>
    </div>
    <div class="card"><h3>Training loss</h3>
      <div class="losshead"><span class="lossnum" id="lossnum">—</span>
        <span class="muted" id="lossmeta"></span></div>
      <canvas id="loss"></canvas>
    </div>
  </div>

  <div id="right">
    <div class="card"><h3>Training</h3>
      <div class="stat"><span class="muted">Status</span>
        <span class="runchip" id="runchip"><span class="d"></span><span id="runtxt">Idle</span></span></div>
      <div class="stat"><span class="muted">Round (epoch)</span><b id="tround">0</b></div>
      <div class="stat"><span class="muted">Loss</span><b id="tloss">—</b></div>
      <div class="stat"><span class="muted">Contributing users</span><b id="tcontrib">0</b></div>
      <button id="trainbtn" onclick="toggleTrain()">Start training</button>
      <button id="resetbtn" class="danger" onclick="doReset()">Reset model</button>
    </div>
    <div class="card"><h3>Session model</h3>
      <div class="muted" style="font-size:12px;margin-bottom:8px">Trained by the hub; all clients predict with it.</div>
      <select id="model" onchange="setModel()"></select>
    </div>
    <div class="card"><h3>Storage</h3>
      <div class="muted" style="font-size:11px">Main folder (crops · models)</div>
      <div class="mono" id="datadir" style="font-size:11px;color:var(--ink);word-break:break-all;margin-top:3px">—</div>
    </div>
  </div>
</div>

<div id="picker">
  <div class="pcard">
    <h2>Select a project</h2>
    <div class="muted" id="pdir" style="font-size:11px;margin-bottom:14px"></div>
    <div id="plist"></div>
    <div class="pnew">
      <button onclick="newProj()">+ New session</button>
    </div>
    <div class="muted" style="font-size:11px;margin-top:10px">A new session starts empty and unnamed — the first person to join becomes the owner and defines its name, model and crop size.</div>
  </div>
</div>

<div id="modal" onclick="if(event.target.id==='modal')closeModal()"><div id="mcard">
  <div id="mhead"><h3 id="mtitle"></h3><span class="muted mono" id="mcount"></span>
    <button class="mini" onclick="closeModal()">close ✕</button></div>
  <div id="mview">
    <div style="max-width:380px;margin:0 auto;text-align:center">
      <canvas id="mcanvas" class="review" style="width:100%"></canvas>
      <label style="display:block;margin-top:8px;color:var(--muted);font-size:12px;cursor:pointer">
        <input type="checkbox" id="movl" checked onchange="showCrop()"> mask overlay</label>
    </div></div>
  <div id="mctrl">
    <button class="ghost" onclick="revPrev()">◀ Prev</button>
    <button class="danger" onclick="revDiscard()">Discard</button>
    <button class="ghost" onclick="revNext()">Next ▶</button></div>
  <div id="mhint">A / ← prev   ·   D / → next   ·   Del / X discard   ·   Esc close</div>
</div></div>

<script>
let MODEL_BUILT=false, TRAINING=false, REVIEW=null, CUR_NAMES={};
function copyAddr(){navigator.clipboard&&navigator.clipboard.writeText(document.getElementById('addr').textContent);}
function post(p,o){fetch(p,{method:'POST',body:o?JSON.stringify(o):''});}
function doReset(){if(confirm('Reset the model? Archives the checkpoint and clears the loss plot.'))post('/reset');}
function setModel(){post('/model',{arch:document.getElementById('model').value});}
function toggle(uid,inc){post('/toggle',{uid:uid,included:inc});}
function toggleTrain(){post(TRAINING?'/train/stop':'/train/start');}

// one-at-a-time crop reviewer (image + mask), like MOSS's Review Crops
function openModal(uid){fetch('/crops/'+uid).then(r=>r.json()).then(names=>{
  REVIEW={uid:uid,names:names,idx:0};
  document.getElementById('mtitle').textContent=(CUR_NAMES[uid]||uid);
  document.getElementById('modal').style.display='flex';showCrop();});}
function showCrop(){if(!REVIEW)return;const uid=REVIEW.uid,names=REVIEW.names;
  const cnt=document.getElementById('mcount'),cv=document.getElementById('mcanvas');
  if(!names.length){cnt.textContent='0 crops';cv.getContext('2d').clearRect(0,0,cv.width,cv.height);return;}
  const i=Math.max(0,Math.min(REVIEW.idx,names.length-1));REVIEW.idx=i;
  cnt.textContent=(i+1)+' / '+names.length;
  const raw=new Image(),mask=new Image();let n=0;
  const done=()=>{if(++n>=2)composite(raw,mask);};
  raw.onload=done;raw.onerror=done;mask.onload=done;mask.onerror=done;
  raw.src='/crop/'+uid+'/'+names[i];mask.src='/mask/'+uid+'/'+names[i];}
function composite(raw,mask){
  const cv=document.getElementById('mcanvas');
  const W=raw.naturalWidth||mask.naturalWidth||256,H=raw.naturalHeight||mask.naturalHeight||256;
  cv.width=W;cv.height=H;const x=cv.getContext('2d');x.clearRect(0,0,W,H);
  if(raw.naturalWidth)x.drawImage(raw,0,0,W,H);
  if(!document.getElementById('movl').checked||!mask.naturalWidth)return;
  const off=document.createElement('canvas');off.width=W;off.height=H;
  const ox=off.getContext('2d');ox.drawImage(mask,0,0,W,H);
  try{
    const md=ox.getImageData(0,0,W,H).data;
    const ri=x.getImageData(0,0,W,H),rd=ri.data;
    for(let p=0;p<md.length;p+=4){if(md[p]>127){
      rd[p]=(rd[p]*0.75)|0; rd[p+1]=(rd[p+1]*0.75+255*0.25)|0; rd[p+2]=(rd[p+2]*0.75)|0;}}
    x.putImageData(ri,0,0);
  }catch(e){}
}
function revPrev(){if(REVIEW&&REVIEW.idx>0){REVIEW.idx--;showCrop();}}
function revNext(){if(REVIEW&&REVIEW.idx<REVIEW.names.length-1){REVIEW.idx++;showCrop();}}
function revDiscard(){if(!REVIEW||!REVIEW.names.length)return;
  post('/discard',{uid:REVIEW.uid,name:REVIEW.names[REVIEW.idx]});
  REVIEW.names.splice(REVIEW.idx,1);showCrop();}
function closeModal(){REVIEW=null;document.getElementById('modal').style.display='none';}
document.addEventListener('keydown',e=>{
  if(!REVIEW||document.getElementById('modal').style.display!=='flex')return;
  if(e.key==='ArrowLeft'||e.key==='a')revPrev();
  else if(e.key==='ArrowRight'||e.key==='d')revNext();
  else if(e.key==='Backspace'||e.key==='Delete'||e.key==='x')revDiscard();
  else if(e.key==='Escape')closeModal();});

function render(s){
  const idle=(s.active===false);
  document.getElementById('picker').style.display=idle?'flex':'none';
  document.getElementById('body').style.display=idle?'none':'';
  if(idle){renderPicker(s);
    document.getElementById('addr').textContent='—';
    document.getElementById('code').textContent='—';
    document.getElementById('proj').textContent='— select a project —';
    document.getElementById('subproj').textContent='';
    document.getElementById('count').textContent='idle';
    return;}
  document.getElementById('addr').textContent=s.connect_address;
  document.getElementById('code').textContent=s.code;
  document.getElementById('proj').textContent=s.project_name||'— waiting for owner —';
  document.getElementById('subproj').textContent=s.session_subproject?(s.session_subproject+' · crop '+s.crop_size):'';
  document.getElementById('count').textContent=(s.online_count===s.total_count)?(s.online_count+' connected'):(s.online_count+' online · '+s.total_count+' total');
  document.getElementById('datadir').textContent=s.data_dir;
  const t=s.training||{};
  document.getElementById('tround').textContent=t.round||0;
  document.getElementById('tloss').textContent=t.loss?t.loss.toFixed(4):'—';
  document.getElementById('tcontrib').textContent=t.contributors||0;
  TRAINING=!!t.running;
  const rc=document.getElementById('runchip');rc.className='runchip'+(TRAINING?' on':'');
  document.getElementById('runtxt').textContent=TRAINING?'Training':'Idle';
  const tb=document.getElementById('trainbtn');
  tb.textContent=TRAINING?'Stop training':'Start training';
  tb.className=TRAINING?'danger':'';
  // model select (build options once; keep value synced)
  const sel=document.getElementById('model');
  if(!MODEL_BUILT&&s.models&&s.models.length){s.models.forEach(m=>{
    const o=document.createElement('option');o.value=m.id;o.textContent=m.name;sel.appendChild(o);});MODEL_BUILT=true;}
  if(s.model&&document.activeElement!==sel)sel.value=s.model;
  // tiles
  CUR_NAMES={};const tiles=document.getElementById('tiles');tiles.innerHTML='';
  document.getElementById('empty').style.display=s.users.length?'none':'block';
  s.users.forEach(u=>{CUR_NAMES[u.uid]=u.name;
    const d=document.createElement('div');d.className='tile'+(u.online?'':' off');
    d.style.borderLeftColor=u.included?u.color:'var(--faint)';
    d.innerHTML='<div class="row"><span class="crown">'+(u.is_owner?'♛ owner':'')+'</span>'+
      '<input class="inc" type="checkbox" '+(u.included?'checked':'')+' title="include in training">'+
      '<span class="dot '+(u.online?'on':'')+'"></span></div>'+
      '<div class="av">'+u.animal_svg+'</div>'+
      '<div class="nm">'+u.name+'</div><div class="cc">'+u.crops+' crop'+(u.crops===1?'':'s')+'</div>';
    d.querySelector('.inc').onclick=(e)=>{e.stopPropagation();toggle(u.uid,e.target.checked);};
    d.onclick=()=>openModal(u.uid);
    tiles.appendChild(d);});
  drawLoss(t.loss_history||[]);
}

function renderPicker(s){
  document.getElementById('pdir').textContent='in '+(s.projects_dir||'?');
  const list=document.getElementById('plist');list.innerHTML='';
  const ps=s.projects||[];
  if(!ps.length){list.innerHTML='<div class="muted" style="padding:12px 0">No projects here yet — create one below.</div>';return;}
  ps.forEach(p=>{
    const meta=p.has_session
      ?((p.project||p.name)+' · '+p.users+' user'+(p.users===1?'':'s')+(p.crop_size?(' · crop '+p.crop_size):''))
      :'empty · no session yet';
    const d=document.createElement('div');d.className='prow';
    d.innerHTML='<div class="pinfo"><b>'+p.name+'</b><span class="muted">'+meta+'</span></div>'+
      '<button>'+(p.has_session?'Resume':'Open')+'</button>';
    d.querySelector('button').onclick=()=>selectProj(p.name);
    list.appendChild(d);
  });
}
function selectProj(n){post('/select',{name:n});setTimeout(poll,400);}
function newProj(){post('/new');setTimeout(poll,400);}

function drawLoss(h){
  window._ls=h;
  const c=document.getElementById('loss'),dpr=window.devicePixelRatio||1;
  const W=c.clientWidth||600,H=230;
  if(c.width!==W*dpr||c.height!==H*dpr){c.width=W*dpr;c.height=H*dpr;}
  const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const num=document.getElementById('lossnum'),meta=document.getElementById('lossmeta');
  if(h.length<2){num.textContent='—';meta.textContent='';
    x.fillStyle='#5f5f68';x.font='13px sans-serif';x.textAlign='left';
    x.fillText(TRAINING?'warming up…':'waiting for training…',12,26);return;}
  // EMA smoothing (matches MOSS's widget)
  const a=2/(50+1);let e=null;const sm=h.map(p=>{e=(e==null)?p[1]:a*p[1]+(1-a)*e;return e;});
  const ys=h.map(p=>p[1]);let mn=Math.min.apply(null,ys),mx=Math.max.apply(null,ys);
  if(mx-mn<1e-6)mx=mn+1;const padY=(mx-mn)*0.08;mn-=padY;mx+=padY;
  num.textContent=sm[sm.length-1].toFixed(4);
  meta.textContent='smoothed · '+h.length+' steps';
  const L=44,R=10,T=12,B=24;
  const px=i=>L+(W-L-R)*(i/(h.length-1));
  const py=v=>T+(H-T-B)*(1-(v-mn)/(mx-mn));
  // grid + y labels (recessive)
  x.strokeStyle='rgba(255,255,255,.06)';x.fillStyle='#8b8b94';x.font='10px '+'ui-monospace,monospace';
  x.textAlign='right';x.textBaseline='middle';x.lineWidth=1;
  for(let g=0;g<=4;g++){const v=mn+(mx-mn)*g/4,yy=py(v);
    x.beginPath();x.moveTo(L,yy);x.lineTo(W-R,yy);x.stroke();
    x.fillText(v.toFixed(3),L-6,yy);}
  x.textAlign='left';x.textBaseline='alphabetic';x.fillStyle='#5f5f68';
  x.fillText('0',L,H-8);x.textAlign='right';x.fillText(''+h[h.length-1][0],W-R,H-8);
  x.fillText('batch',(L+W-R)/2,H-8);x.textAlign='left';
  // raw points (faded, downsampled)
  const step=Math.max(1,Math.floor(h.length/1200));
  x.fillStyle='rgba(95,191,106,.22)';
  for(let i=0;i<h.length;i+=step){x.beginPath();x.arc(px(i),py(h[i][1]),1.6,0,6.2832);x.fill();}
  // smoothed line (primary)
  x.strokeStyle='#5fbf6a';x.lineWidth=2;x.lineJoin='round';x.beginPath();
  sm.forEach((v,i)=>{i?x.lineTo(px(i),py(v)):x.moveTo(px(i),py(v));});x.stroke();
}

function poll(){fetch('/state').then(r=>r.json()).then(render).catch(()=>{});}
poll();setInterval(poll,1500);
window.addEventListener('resize',()=>{if(window._ls)drawLoss(window._ls);});
</script>
</body></html>
"""
