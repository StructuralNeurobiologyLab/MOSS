#!/usr/bin/env python3
"""
CLI entry point for the MOSS Hub.

    python -m segmentation_suite.hub --data-dir PATH  # start / auto-resume
    python -m segmentation_suite.hub --data-dir PATH --fresh   # ignore any saved session
    python -m segmentation_suite.hub --mock           # simulated backend (visual dev)

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
                        help="Use the simulated mock backend (visual dev only, no network)")
    args = parser.parse_args()

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    if args.mock:
        from .mock_backend import MockHubBackend
        backend = MockHubBackend(data_dir=args.data_dir)
    else:
        # Auto-resume when a manifest is present, unless --fresh is given.
        manifest = Path(args.data_dir).expanduser() / "session.json"
        resume = (args.resume or manifest.exists()) and not args.fresh
        if resume and manifest.exists():
            print(f"[Hub] resuming saved session in {args.data_dir}")
        from .hub_server import HubServer
        backend = HubServer(data_dir=args.data_dir, host=args.host, port=args.port,
                            resume=resume)

    from .hub_window import HubWindow
    window = HubWindow(backend)
    window.show()
    backend.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
