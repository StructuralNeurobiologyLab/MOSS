#!/usr/bin/env python3
"""
raw_transfer — client-side helpers to get a hub's raw volume onto a joinee.

The hub serves the session's raw zarr over its HTTP console (hub_web.py):
    GET /raw/meta                availability + shape/pyramid metadata (JSON)
    GET /raw/manifest            every file under the zarr: [{path, size}, ...]
    GET /raw/file/<relpath>      one file's bytes, byte-range capable
    GET /raw/tile?...            a single region (streaming fallback; see
                                 RemoteZarrImageSource in zarr_image_source.py)

PRIMARY path = download: replicate the whole zarr into a local directory, then
open it with the normal local ZarrImageSource (nothing remote once downloaded).
The download is resumable like rsync: files already present with the right size
are skipped, and a partially-fetched file resumes from where it stopped via an
HTTP Range request — so a dropped LAN connection never restarts from zero.

Stdlib only (urllib) — no new dependency on the client.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import quote


def _open(url: str, headers: dict | None = None, timeout: float = 120.0):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_meta(base_url: str, timeout: float = 30.0) -> dict:
    """GET /raw/meta. Returns {'available': False} if the hub has no raw volume."""
    with _open(base_url.rstrip("/") + "/raw/meta", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def hub_has_raw(base_url: str, timeout: float = 10.0) -> bool:
    """Cheap check for the join-time prompt gate."""
    try:
        return bool(fetch_meta(base_url, timeout=timeout).get("available"))
    except Exception:
        return False


def fetch_manifest(base_url: str, timeout: float = 300.0) -> dict:
    """GET /raw/manifest: {root, count, total_bytes, files:[{path,size}]}."""
    with _open(base_url.rstrip("/") + "/raw/manifest", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_upload_manifest(base_url: str, timeout: float = 300.0) -> dict:
    """GET /raw/upload_manifest: files already in the hub's UPLOAD TARGET (even
    before it's attached/serving). Used for resumable upload — skip what's there."""
    with _open(base_url.rstrip("/") + "/raw/upload_manifest", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def upload_raw(base_url: str, src_dir, progress_cb=None, should_stop=None,
               timeout: float = 300.0, block_size: int = 1 << 20,
               concurrency: int = 8) -> bool:
    """Upload a local zarr store (src_dir) TO the hub, for the case where the hub
    can't see the owner's data (different machine, no shared filesystem). Mirrors
    download_raw in reverse: POST each file to /raw/upload/<relpath>, streamed;
    resumable — skip files the hub already has at the right size (via /raw/manifest).

    Returns True when the whole store is on the hub; False if aborted (call again
    to resume). Raises on network/IO errors. Bulk goes over HTTP, never the WS.
    """
    import urllib.error
    from urllib.parse import quote
    base_url = base_url.rstrip("/")
    src_dir = Path(src_dir)

    files = [p for p in src_dir.rglob("*") if p.is_file()]
    total_files = len(files)
    total_bytes = sum(p.stat().st_size for p in files)

    # What does the hub already have? (resume) — the UPLOAD manifest lists files in
    # the target even before it's attached, so an interrupted upload skips what's
    # already there instead of re-sending everything.
    have = {}
    try:
        for f in fetch_upload_manifest(base_url, timeout=timeout).get("files", []):
            have[f["path"]] = int(f["size"])
    except Exception:
        have = {}

    import threading
    lock = threading.Lock()
    st = {"bytes": 0, "files": 0, "err": None, "stop": False}

    def _emit():
        if progress_cb:
            progress_cb(st["bytes"], total_bytes, st["files"], total_files)

    # Partition (resume): skip files the hub already has; queue the rest.
    todo = []
    for p in files:
        rel = p.relative_to(src_dir).as_posix()
        sz = p.stat().st_size
        if have.get(rel) == sz:
            st["bytes"] += sz
            st["files"] += 1
        else:
            todo.append((p, rel, sz))
    _emit()

    def _upload(item):
        if st["stop"] or (should_stop and should_stop()):
            return
        p, rel, sz = item
        url = base_url + "/raw/upload/" + quote(rel)
        with open(p, "rb") as fh:
            req = urllib.request.Request(url, data=fh, method="POST")
            req.add_header("Content-Type", "application/octet-stream")
            req.add_header("Content-Length", str(sz))
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
        with lock:
            st["bytes"] += sz
            st["files"] += 1
            _emit()

    _run_parallel(todo, _upload, concurrency, st, lock)
    if st["err"]:
        raise st["err"]
    return not st["stop"] and not (should_stop and should_stop())


def _run_parallel(items, fn, concurrency: int, st: dict, lock):
    """Run fn(item) over items with a bounded thread pool. First exception is
    captured into st['err'] and sets st['stop'] so the rest wind down. The hub's
    HTTP server is multi-threaded, so concurrent transfers are a big speedup on a
    many-small-files zarr."""
    import threading  # noqa: F401 (lock is a threading.Lock created by the caller)
    from concurrent.futures import ThreadPoolExecutor
    if not items:
        return
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                with lock:
                    if st["err"] is None:
                        st["err"] = e
                    st["stop"] = True


def download_raw(base_url: str, dest_dir, progress_cb=None, should_stop=None,
                 timeout: float = 300.0, block_size: int = 1 << 20,
                 concurrency: int = 8) -> bool:
    """Resumably download the hub's raw zarr into dest_dir (a local zarr store).

    Files are fetched CONCURRENTLY (`concurrency` workers) — a big speedup on a
    many-small-files zarr, since the transfer is otherwise per-file-latency bound.
    Per-file Range-resume; files already complete are skipped.

    Returns True when the whole volume is local; False if aborted mid-way (call
    again to resume). Raises on network/IO errors.
    """
    import threading
    base_url = base_url.rstrip("/")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest = fetch_manifest(base_url, timeout=timeout)
    files = manifest.get("files", [])
    total_files = len(files)
    total_bytes = int(manifest.get("total_bytes", sum(f["size"] for f in files)))

    lock = threading.Lock()
    st = {"bytes": 0, "files": 0, "err": None, "stop": False}

    def _emit():
        if progress_cb:
            progress_cb(st["bytes"], total_bytes, st["files"], total_files)

    # Partition (resume): count complete files, pre-count partials, queue the rest.
    todo = []
    for f in files:
        lp = dest_dir / f["path"]
        existing = lp.stat().st_size if lp.is_file() else 0
        if existing == f["size"]:
            st["bytes"] += f["size"]
            st["files"] += 1
        else:
            if 0 < existing < f["size"]:
                st["bytes"] += existing        # partial already on disk
            todo.append(f)
    _emit()

    def _fetch(f):
        if st["stop"] or (should_stop and should_stop()):
            return
        rel, size = f["path"], int(f["size"])
        lp = dest_dir / rel
        lp.parent.mkdir(parents=True, exist_ok=True)
        have = lp.stat().st_size if lp.is_file() else 0
        headers, mode = {}, "wb"
        if 0 < have < size:
            headers["Range"] = f"bytes={have}-"   # resume this file
            mode = "ab"
        elif have > size:
            have = 0                              # overshoot -> refetch clean
        url = base_url + "/raw/file/" + quote(rel)
        with _open(url, headers=headers, timeout=timeout) as r, open(lp, mode) as out:
            while True:
                if st["stop"] or (should_stop and should_stop()):
                    return
                block = r.read(block_size)
                if not block:
                    break
                out.write(block)
        with lock:
            st["bytes"] += (size - have)          # newly written bytes for this file
            st["files"] += 1
            _emit()

    _run_parallel(todo, _fetch, concurrency, st, lock)
    if st["err"]:
        raise st["err"]
    return not st["stop"] and not (should_stop and should_stop())
