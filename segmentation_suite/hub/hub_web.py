#!/usr/bin/env python3
"""
HubWeb — a browser console for the MOSS hub (the primary interface on the cluster,
where X11 GUIs are painful). Serves a self-contained page + crop images over a
stdlib ThreadingHTTPServer; the page POLLS /state (~1.5s) and re-renders. Control
actions (toggle / model / reset / discard) are marshaled onto the Qt main thread
via a signal, because the backend touches QTimer/QThread which are not safe to
call from the HTTP worker thread.

No third-party deps. Open http://<host>:<port>/ in a browser.

Routes:
    GET  /                     the console page
    GET  /state                JSON snapshot (HubServer.web_state())
    GET  /crops/<uid>          JSON list of a user's crop filenames
    GET  /crop/<uid>/<name>    a crop PNG
    POST /toggle {uid,included}    include/exclude a user's crops
    POST /model  {arch}            set the authoritative session model
    POST /reset                    reset the model
    POST /discard {uid,name}       discard a crop
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
                elif path.startswith("/crop/"):
                    parts = unquote(path[len("/crop/"):]).split("/", 1)
                    if len(parts) == 2 and _UID_RE.match(parts[0]) and _NAME_RE.match(parts[1]):
                        f = hub.data_dir / "incoming" / parts[0] / "train_images" / parts[1]
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
                        "/reset": "reset", "/discard": "discard"}.get(path)
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
  :root{color-scheme:dark;}
  *{box-sizing:border-box;}
  body{margin:0;background:#161618;color:#e8e8e8;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;}
  a{color:#4d90e6;}
  #top{display:flex;gap:26px;align-items:center;padding:14px 18px;background:#0f0f10;border-bottom:1px solid #2a2a2a;flex-wrap:wrap;}
  .brand{font-size:20px;font-weight:800;color:#4d90e6;}
  .cap{color:#777;font-size:10px;font-weight:bold;letter-spacing:.5px;}
  .addr{font:800 20px monospace;color:#f4c542;}
  .code{font:700 15px monospace;color:#b9a24a;letter-spacing:2px;}
  .muted{color:#8a8a8a;font-size:11px;}
  button{background:#2f6fd0;color:#fff;border:none;padding:8px 12px;border-radius:6px;font-weight:600;cursor:pointer;}
  button:hover{background:#3a7ee0;}
  button.mini{background:#333;padding:2px 8px;font-size:11px;}
  button.danger{background:#b8433a;} button.danger:hover{background:#cc4d43;}
  .count{margin-left:auto;color:#5fbf6a;font-weight:700;font-size:14px;}
  #body{display:flex;gap:14px;padding:14px;align-items:flex-start;}
  #left{flex:3;min-width:0;}
  #right{flex:1;max-width:320px;display:flex;flex-direction:column;gap:14px;}
  .box{border:1px solid #333;border-radius:8px;padding:12px;}
  .box h3{margin:0 0 10px;font-size:13px;color:#c0c0c0;}
  #tiles{display:flex;flex-wrap:wrap;gap:14px;}
  .tile{width:150px;background:#2a2a2e;border:3px solid #555;border-radius:12px;padding:8px;cursor:pointer;position:relative;}
  .tile .hdr{display:flex;align-items:center;height:16px;}
  .tile .crown{color:#f4c542;font-size:10px;font-weight:bold;flex:1;}
  .tile .dot{width:11px;height:11px;border-radius:50%;background:#555;margin-right:6px;}
  .tile .dot.on{background:#3ddc5f;animation:blink 1.4s infinite;}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
  .tile .avatar{display:block;margin:4px auto;width:80px;height:80px;}
  .tile.off .avatar{filter:grayscale(1)opacity(.5);}
  .tile .nm{text-align:center;font-weight:600;font-size:13px;}
  .tile.off .nm{color:#888;}
  .tile .cc{text-align:center;color:#a0a0a0;font-size:11px;}
  .tile .inc{position:absolute;top:8px;right:8px;}
  #lossbox{margin-top:14px;}
  canvas{width:100%;height:200px;background:#101012;border-radius:6px;display:block;}
  select{width:100%;background:#2a2a2e;color:#eee;border:1px solid #444;padding:6px;border-radius:5px;}
  .kv{color:#ccc;font-size:12px;margin:2px 0;}
  /* review modal */
  #modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:10;}
  #modalinner{background:#1a1a1d;border-radius:10px;width:min(90vw,760px);max-height:86vh;display:flex;flex-direction:column;padding:14px;}
  #grid{display:flex;flex-wrap:wrap;gap:8px;overflow:auto;}
  .thumb{width:120px;border:2px solid #444;border-radius:8px;padding:5px;background:#242427;}
  .thumb img{width:106px;height:106px;object-fit:cover;border-radius:4px;display:block;}
  .thumb button{width:100%;margin-top:4px;background:#7a2f2a;font-size:10px;padding:3px;}
</style></head>
<body>
<div id="top">
  <div class="brand">MOSS · HUB</div>
  <div><div class="cap">CONNECT ADDRESS · share this</div><div><span class="addr" id="addr">—</span>
    <button class="mini" onclick="copyAddr()">copy</button></div>
    <div class="muted">clients: Multi-User → Join → LAN → paste this</div></div>
  <div><div class="cap">SESSION CODE</div><div class="code" id="code">—</div>
    <div class="muted">session label — not for joining</div></div>
  <div><div class="cap">PROJECT</div><div id="proj" style="font-weight:700">—</div>
    <div class="muted" id="subproj"></div></div>
  <div class="count" id="count">0 connected</div>
</div>
<div id="body">
  <div id="left">
    <div class="box"><h3>Connected users — click a tile to review crops</h3>
      <div id="tiles"></div>
      <div id="empty" class="muted" style="padding:20px;text-align:center">No users connected yet. Share the connect address.</div>
    </div>
    <div class="box" id="lossbox"><h3>Training loss (hub)</h3><canvas id="loss" width="900" height="200"></canvas></div>
  </div>
  <div id="right">
    <div class="box"><h3>Storage</h3><div class="muted">Main folder (crops, models):</div><div class="kv" id="datadir">—</div></div>
    <div class="box"><h3>Cluster training</h3>
      <div class="kv" id="tround">Round: 0</div><div class="kv" id="tloss">Loss: —</div>
      <div class="kv" id="tcontrib">Contributing users: 0</div><div class="kv" id="trun">Idle</div>
      <button class="danger" style="width:100%;margin-top:8px" onclick="doReset()">Reset model</button></div>
    <div class="box"><h3>Session model (training + prediction)</h3>
      <div class="muted">The hub trains this model; all clients predict with it:</div>
      <select id="model" onchange="setModel()"></select>
      <div class="muted" style="margin-top:6px">Locks clients' training + prediction (red).</div></div>
  </div>
</div>
<div id="modal" onclick="if(event.target.id==='modal')closeModal()"><div id="modalinner">
  <div style="display:flex;align-items:center;margin-bottom:8px"><h3 id="mtitle" style="margin:0;flex:1"></h3>
    <button class="mini" onclick="closeModal()">close</button></div>
  <div id="grid"></div></div></div>

<script>
let MODAL_UID=null, MODEL_BUILT=false;
function copyAddr(){navigator.clipboard&&navigator.clipboard.writeText(document.getElementById('addr').textContent);}
function post(path,obj){fetch(path,{method:'POST',body:obj?JSON.stringify(obj):''});}
function doReset(){if(confirm('Reset the model and restart training fresh?'))post('/reset');}
function setModel(){post('/model',{arch:document.getElementById('model').value});}
function toggle(uid,inc){post('/toggle',{uid:uid,included:inc});}
function discard(uid,name){post('/discard',{uid:uid,name:name});setTimeout(()=>openModal(uid),300);}

function openModal(uid){MODAL_UID=uid;document.getElementById('modal').style.display='flex';
  fetch('/crops/'+uid).then(r=>r.json()).then(names=>{
    const g=document.getElementById('grid');g.innerHTML='';
    document.getElementById('mtitle').textContent=(CUR_NAMES[uid]||uid)+' — '+names.length+' crops';
    if(!names.length)g.innerHTML='<div class="muted">No crops yet.</div>';
    names.forEach(n=>{const d=document.createElement('div');d.className='thumb';
      d.innerHTML='<img src="/crop/'+uid+'/'+n+'"><button>discard</button>';
      d.querySelector('button').onclick=()=>discard(uid,n);g.appendChild(d);});});}
function closeModal(){MODAL_UID=null;document.getElementById('modal').style.display='none';}

let CUR_NAMES={};
function render(s){
  document.getElementById('addr').textContent=s.connect_address;
  document.getElementById('code').textContent=s.code;
  document.getElementById('proj').textContent=s.project_name||'— waiting for owner —';
  document.getElementById('subproj').textContent=s.session_subproject?('subproject: '+s.session_subproject+' · crop '+s.crop_size):'';
  document.getElementById('count').textContent=(s.online_count===s.total_count)?(s.online_count+' connected'):(s.online_count+' online · '+s.total_count+' total');
  document.getElementById('datadir').textContent=s.data_dir;
  const t=s.training||{};
  document.getElementById('tround').textContent='Round: '+(t.round||0);
  document.getElementById('tloss').textContent='Loss: '+(t.loss?t.loss.toFixed(4):'—');
  document.getElementById('tcontrib').textContent='Contributing users: '+(t.contributors||0);
  document.getElementById('trun').textContent=t.running?'● training':'Idle';
  document.getElementById('trun').style.color=t.running?'#5fbf6a':'#888';
  // model select (build once)
  const sel=document.getElementById('model');
  if(!MODEL_BUILT&&s.models&&s.models.length){sel.innerHTML='';
    s.models.forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.name;sel.appendChild(o);});MODEL_BUILT=true;}
  if(s.model&&document.activeElement!==sel)sel.value=s.model;
  // tiles
  CUR_NAMES={};const tiles=document.getElementById('tiles');tiles.innerHTML='';
  document.getElementById('empty').style.display=s.users.length?'none':'block';
  s.users.forEach(u=>{CUR_NAMES[u.uid]=u.name;
    const d=document.createElement('div');d.className='tile'+(u.online?'':' off');d.style.borderColor=u.included?u.color:'#555';
    d.innerHTML='<div class="hdr"><span class="crown">'+(u.is_owner?'♛ owner':'')+'</span>'+
      '<span class="dot '+(u.online?'on':'')+'"></span>'+
      '<input class="inc" type="checkbox" '+(u.included?'checked':'')+' title="include in training"></div>'+
      '<div class="avatar">'+u.animal_svg+'</div>'+
      '<div class="nm">'+u.name+'</div><div class="cc">'+u.crops+' crop'+(u.crops===1?'':'s')+'</div>';
    d.querySelector('.inc').onclick=(e)=>{e.stopPropagation();toggle(u.uid,e.target.checked);};
    d.onclick=()=>openModal(u.uid);
    tiles.appendChild(d);});
  if(MODAL_UID&&document.getElementById('modal').style.display==='flex'){/* keep open; refreshed on discard */}
  drawLoss(t.loss_history||[]);
}
function drawLoss(h){const c=document.getElementById('loss'),x=c.getContext('2d');
  const W=c.width,H=c.height;x.clearRect(0,0,W,H);
  if(h.length<2){x.fillStyle='#666';x.font='13px sans-serif';x.fillText('waiting for training…',16,24);return;}
  const ys=h.map(p=>p[1]);let mn=Math.min(...ys),mx=Math.max(...ys);if(mx-mn<1e-6)mx=mn+1;
  const pad=28;const px=i=>pad+(W-2*pad)*i/(h.length-1);const py=v=>H-pad-(H-2*pad)*(v-mn)/(mx-mn);
  x.strokeStyle='#2a2a2a';x.beginPath();x.moveTo(pad,H-pad);x.lineTo(W-pad,H-pad);x.moveTo(pad,pad);x.lineTo(pad,H-pad);x.stroke();
  x.fillStyle='#888';x.font='10px sans-serif';x.fillText(mx.toFixed(3),2,pad+8);x.fillText(mn.toFixed(3),2,H-pad);
  x.strokeStyle='#5fbf6a';x.lineWidth=1.5;x.beginPath();h.forEach((p,i)=>{i?x.lineTo(px(i),py(p[1])):x.moveTo(px(i),py(p[1]));});x.stroke();}
function poll(){fetch('/state').then(r=>r.json()).then(render).catch(()=>{});}
poll();setInterval(poll,1500);
</script>
</body></html>
"""
