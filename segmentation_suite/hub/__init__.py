"""
MOSS Multi-User Hub.

A dedicated, authoritative controller for joint multi-user training sessions.
Unlike the legacy "host = a full MOSS wizard secretly connected to its own
server" design, the Hub is a standalone app: it is the WebSocket server AND
the trainer. Connected users are pure clients — they annotate, send crops, and
receive weights + prediction directives.

This package is under active development on the `multiuser-hub` branch.
"""

__all__ = []
