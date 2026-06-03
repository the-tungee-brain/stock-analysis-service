#!/usr/bin/env python3
"""CLI wrapper for data.download (use when ``python -m data.download`` is unavailable)."""

from __future__ import annotations

import sys

from data.download import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
