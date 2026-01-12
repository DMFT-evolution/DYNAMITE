#!/usr/bin/env python3
"""Plot DYNAMITE `energy.txt`.

This script is intentionally lightweight and depends only on numpy + matplotlib.
It assumes `energy.txt` contains at least two whitespace-separated columns:
	time  energy

If your file contains comments, they should start with '#'.

Usage:
	python3 scripts/plot_energy.py /path/to/output/dir

Outputs:
	- displays an interactive plot
	- optionally writes a PNG with --out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_two_column_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError(f"Expected at least 2 columns in {path}")
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}; got {data.shape[1]}")
    return data[:, 0], data[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot DYNAMITE energy.txt")
    ap.add_argument("output_dir", type=Path, help="Run output directory containing energy.txt")
    ap.add_argument("--out", type=Path, default=None, help="Write figure to this file (e.g. energy.png)")
    ap.add_argument("--logx", action="store_true", help="Log-scale the time axis")
    ap.add_argument("--logy", action="store_true", help="Log-scale the energy axis")
    args = ap.parse_args()

    energy_path = args.output_dir / "energy.txt"
    if not energy_path.exists():
        raise FileNotFoundError(f"Missing {energy_path}")

    t, e = load_two_column_txt(energy_path)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=130)
    ax.plot(t, e, lw=1.3)
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
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
