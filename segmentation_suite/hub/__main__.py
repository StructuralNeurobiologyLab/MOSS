#!/usr/bin/env python3
"""
CLI entry point for the MOSS Hub.

    python -m segmentation_suite.hub                 # mock session (laptop dev)
    python -m segmentation_suite.hub --data-dir PATH # choose main folder
    python -m segmentation_suite.hub --port 8765     # (real server, later)

During local GUI development the Hub runs against a mock backend that simulates
users connecting and crops streaming in. The real WebSocket-backed HubServer is
selected with --live once implemented.
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="MOSS Multi-User Hub")
    parser.add_argument("--data-dir", default="~/ceph/moss_hub_demo",
                        help="Main folder for crops and models (e.g. a ceph groups folder)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Address to bind the hub server to (real mode)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to listen on (real mode)")
    parser.add_argument("--live", action="store_true",
                        help="Use the real WebSocket HubServer instead of the mock backend")
    args = parser.parse_args()

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    if args.live:
        print("Live HubServer not implemented yet — run without --live for the mock GUI.")
        return 1
    else:
        from .mock_backend import MockHubBackend
        backend = MockHubBackend(data_dir=args.data_dir)

    from .hub_window import HubWindow
    window = HubWindow(backend)
    window.show()
    backend.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
