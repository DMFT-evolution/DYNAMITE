#!/usr/bin/env python3
"""Plot a generic 2-column whitespace-separated time series produced by DYNAMITE.

Many DYNAMITE text outputs (e.g. rvec.txt, correlation.txt, qk0.txt) are written for quick
inspection/plotting and typically contain one row per saved time point.

This script assumes the simplest common case:
	col 1 = time
	col 2 = value

If your file has more columns, you can select one via --col (1-based index).

Usage:
	python3 scripts/plot_text_series.py /path/to/output/dir rvec.txt
	python3 scripts/plot_text_series.py /path/to/output/dir correlation.txt --col 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_txt(path: Path, col: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}; got {data.shape[1]}")
    if col < 2 or col > data.shape[1]:
        raise ValueError(f"--col must be in [2, {data.shape[1]}] for {path}")
    return data[:, 0], data[:, col - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot a DYNAMITE text time series")
    ap.add_argument("output_dir", type=Path, help="Run output directory")
    ap.add_argument("filename", type=str, help="Text file in the directory (e.g. rvec.txt)")
    ap.add_argument("--col", type=int, default=2, help="1-based column index to plot (default: 2)")
    ap.add_argument("--out", type=Path, default=None, help="Write figure to this file")
    ap.add_argument("--logx", action="store_true", help="Log-scale the time axis")
    ap.add_argument("--logy", action="store_true", help="Log-scale the y axis")
    args = ap.parse_args()

    path = args.output_dir / args.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    t, y = load_txt(path, args.col)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=130)
    ax.plot(t, y, lw=1.3)
    ax.set_xlabel("t")
    ax.set_ylabel(f"{args.filename} (col {args.col})")
    ax.grid(True, alpha=0.25)

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")

    fig.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out)
        print(f"Wrote {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
