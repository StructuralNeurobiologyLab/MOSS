#!/usr/bin/env python3
"""
CLI entry point for the MOSS Hub.

The hub has two interchangeable interfaces — launch exactly ONE:

    python -m segmentation_suite.hub                     # web console (default)
    python -m segmentation_suite.hub --gui               # Qt window instead
    python -m segmentation_suite.hub --mock              # Qt window, simulated data

Project selection:

    # pick / resume a project IN the web console (no project named up front):
    python -m segmentation_suite.hub --projects-dir ~/ceph/moss_hub_projects

    # or point straight at one project folder (skips the picker):
    python -m segmentation_suite.hub --data-dir ~/ceph/moss_hub_projects/songbird_em
    python -m segmentation_suite.hub --data-dir PATH --fresh   # ignore saved session

When --data-dir is omitted, the web console opens a PROJECT PICKER listing the
folders under --projects-dir: resume an existing one, or create a new empty
project that waits for the first person to join (the owner) to define it.
If --data-dir already contains a session.json the hub RESUMES it (unless --fresh).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MOSS Multi-User Hub")
    # Interface: web console (default) OR the Qt window (--gui). Never both.
    parser.add_argument("--gui", action="store_true",
                        help="Launch the Qt window interface instead of the web console")
    parser.add_argument("--mock", action="store_true",
                        help="Qt window with a simulated backend (visual dev only, implies --gui)")
    parser.add_argument("--web-port", type=int, default=8080,
                        help="Port for the browser console (web mode)")
    # Project selection
    parser.add_argument("--data-dir", default=None,
                        help="Use THIS project folder directly (skips the picker). "
                             "Omit to choose/resume a project in the web console.")
    parser.add_argument("--projects-dir", default="~/moss_hub_projects",
                        help="Folder holding hub projects; the web console lists these "
                             "to pick or resume when --data-dir is omitted")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the saved session in --data-dir (auto if a session.json exists)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new session even if --data-dir has a saved session.json")
    # Network
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind the hub server to")
    parser.add_argument("--port", type=int, default=8765, help="Port the hub (WebSocket) listens on")
    # Trainer knobs
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (laptop dev)")
    parser.add_argument("--epochs", type=int, default=50000, help="Max training epochs")
    parser.add_argument("--broadcast-interval", type=int, default=5,
                        help="Broadcast trained weights every N epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    gui = args.gui or args.mock          # exactly one interface
    web = not gui

    # Qt needs an event loop either way (backend uses QTimer/signals). Use a
    # widget app for the window, else a headless core app for the web server.
    if gui:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
    else:
        from PyQt6.QtCore import QCoreApplication
        app = QCoreApplication(sys.argv)

    if args.mock:
        from .mock_backend import MockHubBackend
        backend = MockHubBackend(data_dir=args.data_dir or "~/moss_hub_demo")
    else:
        data_dir = args.data_dir
        # The Qt window has no project picker — it needs a concrete project.
        if gui and not data_dir:
            data_dir = "~/moss_hub_demo"
        resume = False
        if data_dir:
            manifest = Path(data_dir).expanduser() / "session.json"
            resume = (args.resume or manifest.exists()) and not args.fresh
            if resume and manifest.exists():
                print(f"[Hub] resuming saved session in {data_dir}")
        from .hub_server import HubServer
        backend = HubServer(data_dir=data_dir, projects_dir=args.projects_dir,
                            host=args.host, port=args.port, resume=resume)
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
    if web and args.web_port > 0:
        from .hub_web import HubWeb
        web_srv = HubWeb(backend, host=args.host, port=args.web_port)
        # Let the backend advertise the raw-volume HTTP URL with the right port.
        try:
            backend.web_port = args.web_port
        except Exception:
            pass
        web_srv.start()

    # Ctrl-C handling. Qt's event loop otherwise swallows SIGINT, so the hub can't
    # be stopped from the terminal. Install a handler (first Ctrl-C = graceful stop,
    # second = force) and a periodic timer so the loop wakes often enough for Python
    # to actually run the handler.
    import signal
    from PyQt6.QtCore import QTimer
    _sig = {"n": 0}

    def _on_sigint(*_a):
        _sig["n"] += 1
        if _sig["n"] >= 2:
            print("\n[Hub] force quit")
            os._exit(1)
        print("\n[Hub] shutting down… (Ctrl-C again to force)")
        try:
            backend.stop()
        except Exception:
            pass
        try:
            if web_srv:
                web_srv.stop()
        except Exception:
            pass
        app.quit()

    import os
    signal.signal(signal.SIGINT, _on_sigint)
    _wake = QTimer()
    _wake.timeout.connect(lambda: None)   # give Python a chance to see the signal
    _wake.start(300)

    backend.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
