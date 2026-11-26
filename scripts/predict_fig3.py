#!/usr/bin/env python3
"""Compatibility entry point for scripts/fig3/predict_fig3.py."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("fig3") / "predict_fig3.py"),
                   run_name="__main__")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.fig3 import predict_fig3 as implementation

    sys.modules[__name__] = implementation
