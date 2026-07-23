#!/usr/bin/env python3
"""
CLI entry point for the MOSS Hub.

    python -m segmentation_suite.hub --data-dir PATH   # web console + Qt window
    python -m segmentation_suite.hub --data-dir PATH --no-gui   # headless (web only, cluster)
    python -m segmentation_suite.hub --data-dir PATH --fresh    # ignore any saved session
    python -m segmentation_suite.hub --mock            # simulated backend (Qt visual dev)

The primary interface is the WEB CONSOLE (open http://<host>:<web-port>/ in a
browser) — ideal on the cluster where X11 GUIs are painful. The Qt window also
opens by default for laptop use; disable it with --no-gui (headless).

If --data-dir already contains a session.json, the hub RESUMES that session
(config + users + crops restored; users rejoin their roles). Pass --fresh to
start a new session instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MOSS Multi-User Hub")
    parser.add_argument("--data-dir", default="~/ceph/moss_hub_demo",
                        help="Main folder for crops and models (e.g. a ceph groups folder)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Address to bind the hub server to")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to listen on")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the saved session in --data-dir (auto if a session.json exists)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new session even if --data-dir has a saved session.json")
    parser.add_argument("--mock", action="store_true",
                        help="Use the simulated mock backend (Qt visual dev only, no network)")
    # Interface
    parser.add_argument("--web-port", type=int, default=8080,
                        help="Port for the browser console (0 to disable)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless: no Qt window, web console only (cluster)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable the web console (Qt window only)")
    # Trainer knobs
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (laptop dev)")
    parser.add_argument("--epochs", type=int, default=50000, help="Max training epochs")
    parser.add_argument("--broadcast-interval", type=int, default=5,
                        help="Broadcast trained weights every N epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    gui = not args.no_gui
    web = not args.no_web and args.web_port > 0

    # Qt needs an event loop either way (backend uses QTimer/signals). Use a
    # widget app when showing the window, else a headless core app.
    if gui:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
    else:
        from PyQt6.QtCore import QCoreApplication
        app = QCoreApplication(sys.argv)

    if args.mock:
        from .mock_backend import MockHubBackend
        backend = MockHubBackend(data_dir=args.data_dir)
        web = False  # the mock has no web_state()/controls
    else:
        # Auto-resume when a manifest is present, unless --fresh is given.
        manifest = Path(args.data_dir).expanduser() / "session.json"
        resume = (args.resume or manifest.exists()) and not args.fresh
        if resume and manifest.exists():
            print(f"[Hub] resuming saved session in {args.data_dir}")
        from .hub_server import HubServer
        backend = HubServer(data_dir=args.data_dir, host=args.host, port=args.port,
                            resume=resume)
        backend.force_cpu = args.cpu
        backend.train_epochs = args.epochs
        backend.broadcast_interval = args.broadcast_interval
        backend.train_batch_size = args.batch_size
        backend.train_lr = args.lr

    window = None
    if gui:
        from .hub_window import HubWindow
        window = HubWindow(backend)
        window.show()

    web_srv = None
    if web:
        from .hub_web import HubWeb
        web_srv = HubWeb(backend, host=args.host, port=args.web_port)
        web_srv.start()

    backend.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
