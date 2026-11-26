#!/usr/bin/env python3
"""Verify the partial-identification results in Figure 6(c)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.fig6.verify_fig6_panel import cli


if __name__ == "__main__":
    cli("c")
