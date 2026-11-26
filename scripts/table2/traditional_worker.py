"""Fit one traditional Table 2 estimator in its model environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Baselines" / "src"))

from baselines.table2 import TRADITIONAL_MODELS, fit_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=TRADITIONAL_MODELS)
    parser.add_argument("realization", type=int)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--rscript")
    parser.add_argument("--r-libs", type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = dict(archive)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        prediction = fit_predict(args.model, arrays, args.realization, device=args.device,
                                 work=Path(temp), rscript=args.rscript, r_libs=args.r_libs)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(arrays["X_query"]),) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid query-aligned predictions for {args.model}")
    np.save(args.output, prediction, allow_pickle=False)


if __name__ == "__main__":
    main()
