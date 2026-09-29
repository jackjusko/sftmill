"""CLI logging for sftmill commands."""

from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | None = None, verbose: int = 0) -> None:
    if level is None:
        level = os.environ.get("SFTMILL_LOG", "").strip().upper()
    if not level:
        level = "DEBUG" if verbose else "INFO"
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("sftmill")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(numeric)
    root.propagate = False
