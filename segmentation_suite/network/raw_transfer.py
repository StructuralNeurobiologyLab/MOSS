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


def download_raw(base_url: str, dest_dir, progress_cb=None, should_stop=None,
                 timeout: float = 300.0, block_size: int = 1 << 20) -> bool:
    """Resumably download the hub's raw zarr into dest_dir (a local zarr store).

    Args:
        base_url:    hub console origin, e.g. 'http://10.1.3.45:8080'
        dest_dir:    local directory to materialize the zarr into (created if needed)
        progress_cb: optional callable(done_bytes, total_bytes, done_files, total_files)
        should_stop: optional callable()->bool; True aborts, leaving a resumable partial
        block_size:  read/write block size

    Returns:
        True if the whole volume is present locally; False if aborted mid-way
        (call again to resume). Raises on network/IO errors.
    """
    base_url = base_url.rstrip("/")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest = fetch_manifest(base_url, timeout=timeout)
    files = manifest.get("files", [])
    total_files = len(files)
    total_bytes = int(manifest.get("total_bytes", sum(f["size"] for f in files)))

    # Account for anything already on disk (resume): count valid bytes and
    # completed files so the progress bar starts where the last run stopped.
    done_bytes = 0
    done_files = 0
    for f in files:
        lp = dest_dir / f["path"]
        if lp.is_file():
            existing = lp.stat().st_size
            if existing == f["size"]:
                done_files += 1
                done_bytes += f["size"]
            elif existing < f["size"]:
                done_bytes += existing        # partial, will resume
            # existing > size: corrupt overshoot -> refetch clean, count 0 now

    if progress_cb:
        progress_cb(done_bytes, total_bytes, done_files, total_files)

    for f in files:
        if should_stop and should_stop():
            return False
        rel, size = f["path"], int(f["size"])
        lp = dest_dir / rel
        have = lp.stat().st_size if lp.is_file() else 0
        if have == size:
            continue                          # already complete (counted above)
        lp.parent.mkdir(parents=True, exist_ok=True)

        headers, mode = {}, "wb"
        if 0 < have < size:
            headers["Range"] = f"bytes={have}-"   # resume this file
            mode = "ab"
        elif have > size:
            have = 0                              # overshoot -> start over

        url = base_url + "/raw/file/" + quote(rel)
        with _open(url, headers=headers, timeout=timeout) as r, open(lp, mode) as out:
            while True:
                if should_stop and should_stop():
                    return False
                block = r.read(block_size)
                if not block:
                    break
                out.write(block)
                done_bytes += len(block)
                if progress_cb:
                    progress_cb(done_bytes, total_bytes, done_files, total_files)
        done_files += 1
        if progress_cb:
            progress_cb(done_bytes, total_bytes, done_files, total_files)

    return True
