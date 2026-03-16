"""
calc_pair_distance_ACF.py
---------------
Compute inter-Ca distance autocorrelation for residue pairs
listed in a pairs.dat file.

pairs.dat format (one pair per line, whitespace or comma separated):
    resid1  segid1  resid2  segid2

Output format (acf_distance.dat, whitespace-separated):
    #       tau(ps)        pair_1        pair_2  ...
         0.000000      1.000000      1.000000  ...
    ...

The ACF measures fluctuations around the mean distance:
    C(t) = <dd(t0) dd(t0+t)> / <dd(t0)^2>   where dd = d - <d>
"""

import argparse
from multiprocessing import Pool

import MDAnalysis as mda
import numpy as np
from numpy.fft import fft, ifft


# ─────────────────────────────────────────────
#  Core math
# ─────────────────────────────────────────────

def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """
    Normalised autocorrelation of a 1-D scalar time series via FFT.
    Mean is subtracted first so C(t) tracks fluctuations around <x>.

        C(t) = <dx(t0) dx(t0+t)> / <dx(t0)^2>   dx = x - <x>

    Returns array of length N with C(0) = 1.
    """
    N = len(x)
    xm = x - x.mean()                          # remove mean (fluctuation ACF)
    fx = fft(xm, n=2 * N)                      # zero-padded FFT
    power = np.real(ifft(fx * np.conj(fx)))[:N]
    power /= np.arange(N, 0, -1)               # correct for decreasing sample count
    return power / power[0]                    # normalise so C(0) = 1


# ─────────────────────────────────────────────
#  Per-pair worker (runs in a subprocess)
# ─────────────────────────────────────────────

def _compute_pair_distance_acf(args: tuple):
    """
    Worker: load trajectory slice, compute inter-Ca distance time series,
    return its normalised ACF.

    args = (label, topology, trajectory,
            resid1, segid1, resid2, segid2,
            start, stop, stride)
    """
    (label, topology, trajectory,
     resid1, segid1, resid2, segid2,
     start, stop, stride) = args

    u = mda.Universe(topology, trajectory)

    sel1 = u.select_atoms(f"segid {segid1} and resid {resid1} and name CA")
    sel2 = u.select_atoms(f"segid {segid2} and resid {resid2} and name CA")

    if len(sel1) == 0 or len(sel2) == 0:
        print(f"  [!] Pair {label}: atom(s) not found - skipping.")
        return label, None

    # stop=-1 in Python slice semantics stops *before* the last frame;
    # use None to include the last frame.
    traj_stop = None if stop == -1 else stop

    distances = []
    for _ts in u.trajectory[start:traj_stop:stride]:
        d = np.linalg.norm(sel2.positions[0] - sel1.positions[0])
        distances.append(d)

    if len(distances) < 2:
        print(f"  [!] Pair {label}: fewer than 2 frames collected - skipping.")
        return label, None

    C = autocorr_fft(np.array(distances))
    return label, C


# ─────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────

def read_pairs(path: str) -> list:
    """
    Parse pairs.dat. Each non-comment line must contain:
        resid1  segid1  resid2  segid2
    Both whitespace and comma delimiters are accepted.
    Returns a list of (resid1, segid1, resid2, segid2) tuples.
    """
    pairs = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) < 4:
                raise ValueError(
                    f"{path}:{lineno} - expected 4 fields "
                    f"(resid1 segid1 resid2 segid2), got: {raw!r}"
                )
            resid1, segid1, resid2, segid2 = tokens[:4]
            pairs.append((resid1, segid1, resid2, segid2))
    return pairs


def pair_label(resid1, segid1, resid2, segid2) -> str:
    return f"{segid1}:{resid1}-{segid2}:{resid2}"


# ─────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────

def run(
    topology: str,
    trajectory: str,
    pairs_file: str,
    output_file: str,
    start: int = 0,
    stop: int = -1,
    stride: int = 10,
    n_proc: int = 4,
):
    pairs = read_pairs(pairs_file)
    if not pairs:
        raise RuntimeError(f"No valid pairs found in {pairs_file}.")

    print(f"[i] {len(pairs)} pair(s) loaded from {pairs_file}")

    args_list = [
        (
            pair_label(r1, s1, r2, s2),
            topology, trajectory,
            r1, s1, r2, s2,
            start, stop, stride,
        )
        for r1, s1, r2, s2 in pairs
    ]

    with Pool(n_proc) as pool:
        results = pool.map(_compute_pair_distance_acf, args_list)

    # Collect valid results, preserving input order
    labels, C_all = [], []
    for label, C in results:
        if C is not None:
            labels.append(label)
            C_all.append(C)

    if not C_all:
        print("[!] No valid results - nothing written.")
        return

    # Trim all ACFs to the shortest (can differ if stop=None per universe)
    min_len = min(len(c) for c in C_all)
    C_all = [c[:min_len] for c in C_all]

    # dt is the time between adjacent frames in the file.
    # The effective lag between *sampled* frames is stride * dt.
    u_ref = mda.Universe(topology, trajectory)
    dt_eff = u_ref.trajectory.dt * stride      # lag per step (ps)
    tau = np.arange(min_len) * dt_eff

    data = np.column_stack([tau] + C_all)

    # np.savetxt prepends "# " (2 chars) to the header, so the first field
    # must be 2 chars shorter to stay aligned with fmt="%W.6f".
    W = 14
    header_cols = [f"{'tau(ps)':>{W - 2}}"] + [f"{lbl:>{W}}" for lbl in labels]
    header = " ".join(header_cols)
    np.savetxt(output_file, data, header=header, fmt=f"%{W}.6f")

    print(f"[+] Distance ACF saved -> {output_file}  "
          f"({len(labels)} pairs, {min_len} frames)")


# ─────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Pair Ca distance autocorrelation function"
    )
    p.add_argument("--top",    default="conf.psf",         help="Topology file (default: conf.psf)")
    p.add_argument("--traj",   default="system.xtc",       help="Trajectory file (default: system.xtc)")
    p.add_argument("--pairs",  default="pairs.dat",        help="Pairs file (default: pairs.dat)")
    p.add_argument("--out",    default="distance_acf.dat", help="Output file")
    p.add_argument("--start",  type=int, default=0,        help="First frame index (default: 0)")
    p.add_argument("--stop",   type=int, default=-1,       help="Last frame index, -1 = end (default: -1)")
    p.add_argument("--stride", type=int, default=10,       help="Frame stride (default: 10)")
    p.add_argument("--nproc",  type=int, default=4,        help="Parallel workers (default: 4)")
    a = p.parse_args()

    run(
        topology=a.top,
        trajectory=a.traj,
        pairs_file=a.pairs,
        output_file=a.out,
        start=a.start,
        stop=a.stop,
        stride=a.stride,
        n_proc=a.nproc,
    )


if __name__ == "__main__":
    _cli()
