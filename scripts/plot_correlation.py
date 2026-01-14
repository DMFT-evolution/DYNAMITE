#!/usr/bin/env python3
r"""Plot correlation curves from the *compressed* (QK/QR) outputs.

DYNAMITE can write a compact representation of the (K)eldysh correlation and
response histories as:

  - `QK_compressed` (binary)
  - `QR_compressed` (binary)
  - `t1_compressed.txt` (text)

This script focuses on the common use-case: plotting

	C(t_w + tau, t_w)

for a set of waiting times t_w.

## Coordinates

Let `t1_compressed.txt` contain a 1D grid of values `t1_points`.
Define:

	tlast = t1_points[-1]
	theta_points = t1_points / tlast

The 2D QK matrix is tabulated over the grid (t1, theta) with:

	t1    = theta_points * tlast
	theta = theta_points

so that QK[i, j] corresponds to (t1_i, theta_j).

To evaluate the physical curve, we query the interpolant at

	t1    = t_w + tau
	theta = t_w / (t_w + tau).

## Usage

	python3 scripts/plot_correlation.py /path/to/output/dir --out corr.png

By default this uses `QK_compressed`. Use `--which QR` to plot `QR_compressed`.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def load_t1_points(path: Path) -> np.ndarray:
	"""Load the `t1_compressed.txt` vector."""
	data = np.loadtxt(path, dtype=float)
	data = np.asarray(data, dtype=float).reshape(-1)
	if data.size < 2:
		raise ValueError(f"Expected at least 2 entries in {path}")
	if not np.all(np.isfinite(data)):
		raise ValueError(f"Non-finite values encountered in {path}")
	return data


def _try_read_header(fp, fmt: str) -> tuple[int, int] | None:
	"""Try reading (rows, cols) with a given struct format."""
	sz = struct.calcsize(fmt)
	header = fp.read(2 * sz)
	if len(header) != 2 * sz:
		return None
	# Build a valid repeat-count format string.
	# Example: fmt='<Q' -> '<2Q'
	rows, cols = struct.unpack(fmt[0] + "2" + fmt[1:], header)
	if not (isinstance(rows, int) and isinstance(cols, int)):
		return None
	if rows <= 0 or cols <= 0:
		return None
	return int(rows), int(cols)


def load_compressed_matrix(path: Path) -> np.ndarray:
	"""Load the binary matrix saved as: size_t rows, size_t cols, then float64 data.

The C++ code writes native-endian `size_t` and then raw `double` bytes.

In the C++ code, both the header (`size_t`) and the payload (`double`) are
written using the machine's native endianness.

In practice most runs are on little-endian machines.

Here we try, in order:
	1) native-endian uint64 header
	2) native-endian uint32 header
	3) little-endian uint64 / uint32
	4) big-endian uint64 / uint32  (rare, but cheap to support)

We then read a native-endian float64 payload.
"""

	file_bytes = path.stat().st_size
	if file_bytes < 16:
		raise ValueError(f"File too small to be a compressed matrix: {path}")

	with path.open("rb") as f:
		hdr: tuple[int, int] | None = None
		header_bytes = 0
		# (fmt, header_size)
		candidates: list[tuple[str, int]] = [
			("=Q", 16),
			("=I", 8),
			("<Q", 16),
			("<I", 8),
			(">Q", 16),
			(">I", 8),
		]
		for fmt, hb in candidates:
			f.seek(0)
			h = _try_read_header(f, fmt)
			if h is None:
				continue
			rows, cols = h
			cnt = rows * cols
			needed = hb + 8 * cnt
			if file_bytes < needed:
				# Header parsed but implies more bytes than present; try next.
				continue
			hdr = h
			header_bytes = hb
			break
		if hdr is None:
			raise ValueError(f"Could not parse header in {path}")

		rows, cols = hdr
		cnt = rows * cols
		needed = header_bytes + 8 * cnt
		if file_bytes < needed:
			raise ValueError(
				f"{path} is truncated: header says {rows}x{cols}={cnt} doubles "
				f"({needed} bytes) but file size is {file_bytes}"
			)

		f.seek(header_bytes)
		data = np.fromfile(f, dtype=np.dtype(float), count=cnt)
		if data.size != cnt:
			raise ValueError(f"Could not read {cnt} doubles from {path}")
		return data.reshape((rows, cols))


def build_interpolator(
	q: np.ndarray, t1_points: np.ndarray, *, order: int = 3
):
	"""Return a callable f(t1, theta) using spline interpolation.

We use SciPy if available (RectBivariateSpline). If SciPy isn't installed,
we fall back to a pure-numpy bilinear interpolation with a clear warning in
the error message.
"""

	tlast = float(t1_points[-1])
	theta_points = np.asarray(t1_points / tlast, dtype=float)
	t1_grid = np.asarray(theta_points * tlast, dtype=float)

	if q.shape[0] != t1_grid.size or q.shape[1] != theta_points.size:
		raise ValueError(
			"Matrix shape does not match t1 grid: "
			f"Q is {q.shape}, but t1 has {t1_grid.size} and theta has {theta_points.size}."
		)

	try:
		from scipy.interpolate import RectBivariateSpline  # type: ignore

		# RectBivariateSpline expects increasing x and y.
		# The compressed grids should be increasing; enforce check.
		if not (np.all(np.diff(t1_grid) > 0) and np.all(np.diff(theta_points) > 0)):
			raise ValueError("t1/theta grids must be strictly increasing for spline interpolation")

		k = int(order)
		k = max(1, min(5, k))
		spline = RectBivariateSpline(t1_grid, theta_points, q, kx=k, ky=k)

		def f(t1: np.ndarray, theta: np.ndarray) -> np.ndarray:
			t1a = np.asarray(t1, dtype=float)
			th = np.asarray(theta, dtype=float)
			# RectBivariateSpline returns shape (len(x), len(y)) when x,y arrays.
			# For pointwise evaluation we feed 1D arrays and take diagonal.
			vals = spline.ev(t1a, th)
			return np.asarray(vals, dtype=float)

		return f, tlast

	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"SciPy is required for spline interpolation. "
			"Install it (e.g. `pip install scipy`) or run this in an environment where SciPy is available."
		) from e


def parse_tw_list(s: str) -> list[float]:
	"""Parse `--tw` values.

	Accepted forms:
	  - comma-separated floats: '0,1,10,100'
	  - a power-range token: '100^0..3'
	  - mixtures: '0,100^0..3'
	"""

	s = s.strip()
	if not s:
		return []

	def parse_token(tok: str) -> list[float]:
		tok = tok.strip()
		if not tok:
			return []
		if "^" in tok and ".." in tok:
			base_raw, rng_raw = tok.split("^", 1)
			a_raw, b_raw = rng_raw.split("..", 1)
			base = float(base_raw)
			a = int(a_raw)
			b = int(b_raw)
			if b < a:
				a, b = b, a
			return [base**n for n in range(a, b + 1)]
		return [float(tok)]

	out: list[float] = []
	for part in s.split(","):
		out.extend(parse_token(part))
	return out


def main() -> int:
	ap = argparse.ArgumentParser(
		description="Plot C(tw+tau,tw) using compressed QK/QR outputs"
	)
	ap.add_argument(
		"output_dir",
		type=Path,
		help="Run output directory containing QK_compressed/QR_compressed and t1_compressed.txt",
	)
	ap.add_argument(
		"--which",
		choices=["QK", "QR"],
		default="QK",
		help="Which compressed matrix to use (default: QK)",
	)
	ap.add_argument(
		"--tw",
		type=str,
		default="0,100^0..3",
		help=(
			"Waiting times. Examples: '0,1,10,100' or '100^0..3'. "
			"Default: '0,100^0..3'."
		),
	)
	ap.add_argument(
		"--tau-min",
		type=float,
		default=0.01,
		help="Minimum tau (default 0.01)",
	)
	ap.add_argument(
		"--tau-max",
		type=float,
		default=None,
		help="Maximum tau; default is t_last from the compressed grid",
	)
	ap.add_argument(
		"--n-tau",
		type=int,
		default=800,
		help="Number of tau sample points (default 800)",
	)
	ap.add_argument(
		"--interp-order",
		type=int,
		default=3,
		help="Spline interpolation order (default 3)",
	)
	ap.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Write figure to this file (e.g. corr.png). If omitted, shows an interactive window.",
	)
	ap.add_argument(
		"--title",
		type=str,
		default=None,
		help="Plot title (optional)",
	)
	ap.add_argument(
		"--usetex",
		action="store_true",
		help=(
			"Enable full LaTeX rendering via matplotlib (requires a working LaTeX install). "
			"By default matplotlib's built-in mathtext is used."
		),
	)
	ap.add_argument(
		"--linear-x",
		action="store_true",
		help="Use a linear x-axis",
	)
	args = ap.parse_args()

	out_dir: Path = args.output_dir
	t1_path = out_dir / "t1_compressed.txt"
	q_path = out_dir / ("QK_compressed" if args.which == "QK" else "QR_compressed")

	t1_points = load_t1_points(t1_path)
	q = load_compressed_matrix(q_path)
	f, tlast = build_interpolator(q, t1_points, order=args.interp_order)

	tau_min = float(args.tau_min)
	tau_max = float(args.tau_max) if args.tau_max is not None else float(tlast)
	if tau_min <= 0:
		raise ValueError("tau-min must be > 0 (log plots need positive x)")
	if tau_max <= tau_min:
		raise ValueError("tau-max must be > tau-min")

	if args.linear_x:
		taus = np.linspace(tau_min, tau_max, int(args.n_tau))
	else:
		taus = np.geomspace(tau_min, tau_max, int(args.n_tau))

	tws = parse_tw_list(args.tw)
	if 0.0 not in tws:
		tws = [0.0] + tws

	# Compute curves
	curves: list[tuple[float, np.ndarray]] = []
	for tw in tws:
		tw = float(tw)
		t1 = tw + taus
		theta = tw / (tw + taus)
		y = f(t1, theta)

		# Do not show values beyond the simulated time window.
		# The compressed grid is defined only up to t_last; for t_w+tau > t_last
		# the correlation/response is not available and should not be plotted.
		mask = t1 > tlast
		if np.any(mask):
			y = np.asarray(y, dtype=float).copy()
			y[mask] = np.nan
		curves.append((tw, y))

	# Plot
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	# By default matplotlib uses mathtext ($...$) which does *not* need a LaTeX
	# installation. Only enable usetex on explicit request.
	use_tex = bool(args.usetex)
	if use_tex:
		mpl.rcParams.update({"text.usetex": True})

	fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
	for tw, y in curves:
		label = f"t_w={tw:g}"
		ax.plot(taus, y, label=label)

	if not args.linear_x:
		ax.set_xscale("log")

	# Labels:
	# - Default: plain Unicode text (robust even when mathtext/LaTeX isn't available)
	# - Optional: full LaTeX via --usetex
	if use_tex:
		ax.set_xlabel(r"$\\tau$")
		ax.set_ylabel(r"$C(t_w+\\tau,\\,t_w)$")
	else:
		ax.set_xlabel("τ")
		ax.set_ylabel("C(t_w+τ, t_w)")
	if args.title:
		ax.set_title(args.title)
	else:
		ax.set_title(f"{args.which}: compressed correlation slice")
	ax.grid(True, which="both", alpha=0.25)
	ax.legend(loc="best", fontsize=9)
	fig.tight_layout()

	if args.out is None:
		plt.show()
	else:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.out)
		print(f"Wrote {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
