#!/usr/bin/env python3
"""
calc_dssp.py
------------
Calculate per-residue helicity (H) and beta-sheet (E) content averaged
over an MD trajectory for every segment, using MDAnalysis DSSP.

Output files (whitespace-separated, aligned columns):
    helicity.dat    residue  seg1  seg2  ...
    beta.dat        residue  seg1  seg2  ...

Usage:
    python calc_dssp.py [--top conf.psf] [--traj system.xtc]
                        [--start 0] [--stop -1] [--stride 10]
                        [--nproc 4]
"""

import argparse
from multiprocessing import Pool

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP


# ─────────────────────────────────────────────
#  Per-segment worker (runs in a subprocess)
# ─────────────────────────────────────────────

def _compute_segment_dssp(args: tuple):
    """
    Worker: load trajectory, run DSSP on all CA-containing residues of one
    segment, return per-residue helicity and beta fractions.

    args = (segid, topology, trajectory, start, stop, stride)

    Returns
    -------
    segid : str
    resids : np.ndarray, shape (n_residues,)
        Residue IDs in the order DSSP sees them.
    helicity : np.ndarray, shape (n_residues,)
        Fraction of frames each residue is in helix (H).
    beta : np.ndarray, shape (n_residues,)
        Fraction of frames each residue is in beta-sheet (E).
    """
    segid, topology, trajectory, start, stop, stride = args

    u = mda.Universe(topology, trajectory)
    sel = u.select_atoms(f"segid {segid} and protein")

    if len(sel.residues) == 0:
        print(f"  [!] Segment {segid}: no protein residues found – skipping.")
        return segid, None, None, None

    # stop=-1 must become None so the MDAnalysis slice reaches the last frame
    traj_stop = None if stop == -1 else stop

    dssp = DSSP(sel)
    dssp.run(start=start, stop=traj_stop, step=stride)

    # dssp.results.dssp : (n_frames, n_residues) array of single-char codes
    codes = dssp.results.dssp          # shape (n_frames, n_residues)
    n_frames, n_residues = codes.shape

    if n_frames == 0:
        print(f"  [!] Segment {segid}: no frames analysed – skipping.")
        return segid, None, None, None

    # Vectorised counting: compare whole array at once
    helicity = np.sum(codes == "H", axis=0) / n_frames   # shape (n_residues,)
    beta     = np.sum(codes == "E", axis=0) / n_frames

    # Collect the residue IDs in the same order DSSP used them
    resids = sel.residues.resids.copy()

    print(f"  [+] Segment {segid}: {n_residues} residues, {n_frames} frames")
    return segid, resids, helicity, beta


# ─────────────────────────────────────────────
#  Output writer
# ─────────────────────────────────────────────

def _write_dat(output_file: str,
               all_resids: np.ndarray,
               values_by_seg: list,
               segids: list):
    """
    Write a whitespace-separated, column-aligned output file.

    Columns: residue  segid1  segid2  ...
    Missing values (segment does not have that residue) are written as NaN.
    """
    # Build a union of all resids seen, sorted
    all_res_sorted = np.array(sorted(set(all_resids)))
    n_res = len(all_res_sorted)
    n_seg = len(segids)

    # Fill data matrix; default NaN for absent residues
    mat = np.full((n_res, n_seg), np.nan)
    res_index = {r: i for i, r in enumerate(all_res_sorted)}

    for col, (resids, vals) in enumerate(values_by_seg):
        for r, v in zip(resids, vals):
            if r in res_index:
                mat[res_index[r], col] = v

    W = 12   # column width
    # Header: np.savetxt prepends "# " (2 chars), so first field is W-2 wide
    header_cols = ([f"{'residue':>{W - 2}}"] +
                   [f"{s:>{W}}" for s in segids])
    header = " ".join(header_cols)

    # Build full data array: residue id + value columns
    data = np.column_stack([all_res_sorted.astype(float), mat])
    np.savetxt(output_file, data, header=header,
               fmt=[f"%{W}.0f"] + [f"%{W}.6f"] * n_seg)

    print(f"  [✓] Saved {output_file}  ({n_res} residues, {n_seg} segments)")


# ─────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────

def run(
    topology: str,
    trajectory: str,
    helicity_file: str = "helicity.dat",
    beta_file: str     = "beta.dat",
    start: int  = 0,
    stop: int   = -1,
    stride: int = 10,
    n_proc: int = 4,
):
    # Discover all segments that contain protein residues
    u = mda.Universe(topology, trajectory)
    segments = [seg for seg in u.segments
                if len(seg.atoms.select_atoms("protein")) > 0]

    if not segments:
        raise RuntimeError("No protein segments found in topology.")

    segids = [seg.segid for seg in segments]
    print(f"[i] Found {len(segids)} protein segment(s): {segids}")
    print(f"[i] Frame range: start={start}, stop={stop}, stride={stride}")

    args_list = [
        (segid, topology, trajectory, start, stop, stride)
        for segid in segids
    ]

    with Pool(n_proc) as pool:
        results = pool.map(_compute_segment_dssp, args_list)

    # Collect valid results, preserving segment order
    valid_segids   = []
    helicity_pairs = []   # list of (resids, helicity_array)
    beta_pairs     = []   # list of (resids, beta_array)
    all_resids     = []

    for segid, resids, helicity, beta in results:
        if resids is None:
            continue
        valid_segids.append(segid)
        helicity_pairs.append((resids, helicity))
        beta_pairs.append((resids, beta))
        all_resids.extend(resids.tolist())

    if not valid_segids:
        print("[!] No valid results – nothing written.")
        return

    # Write output files
    _write_dat(helicity_file, all_resids, helicity_pairs, valid_segids)
    _write_dat(beta_file,     all_resids, beta_pairs,     valid_segids)


# ─────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Per-residue DSSP helicity and beta-sheet content"
    )
    p.add_argument("--top",      default="conf.psf",     help="Topology file (default: conf.psf)")
    p.add_argument("--traj",     default="system.xtc",   help="Trajectory file (default: system.xtc)")
    p.add_argument("--hout",     default="helicity.dat", help="Helicity output file (default: helicity.dat)")
    p.add_argument("--bout",     default="beta.dat",     help="Beta output file (default: beta.dat)")
    p.add_argument("--start",    type=int, default=0,    help="First frame index (default: 0)")
    p.add_argument("--stop",     type=int, default=-1,   help="Last frame index, -1 = end (default: -1)")
    p.add_argument("--stride",   type=int, default=10,   help="Frame stride (default: 10)")
    p.add_argument("--nproc",    type=int, default=4,    help="Parallel workers (default: 4)")
    a = p.parse_args()

    run(
        topology=a.top,
        trajectory=a.traj,
        helicity_file=a.hout,
        beta_file=a.bout,
        start=a.start,
        stop=a.stop,
        stride=a.stride,
        n_proc=a.nproc,
    )


if __name__ == "__main__":
    _cli()