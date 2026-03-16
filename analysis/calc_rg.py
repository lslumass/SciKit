#!/usr/bin/env python3
"""
calc_rg.py
----------
Calculate the radius of gyration (Rg) time series for every protein segment
in an MD trajectory.  An optional residue range can restrict the calculation
to a sub-region (e.g. N- or C-terminal); by default the full protein of each
segment is used.

Output file (whitespace-separated, aligned columns):
    rg.dat    frame  seg1  seg2  ...

Usage:
    python calc_rg.py [--top conf.psf] [--traj system.xtc]
                      [--resid FIRST:LAST]          # e.g. 1:38 or 98:141
                      [--out rg.dat]
                      [--start 0] [--stop -1] [--stride 10]
                      [--nproc 4]
"""

import argparse
from multiprocessing import Pool

import numpy as np
import MDAnalysis as mda


# ─────────────────────────────────────────────
#  Per-segment worker (runs in a subprocess)
# ─────────────────────────────────────────────

def _compute_segment_rg(args: tuple):
    """
    Worker: load trajectory, compute Rg time series for one segment.

    args = (segid, topology, trajectory, resid_range, start, stop, stride)

    resid_range : str or None
        MDAnalysis resid selection string, e.g. "1:38".
        None means the full protein of the segment.

    Returns
    -------
    segid : str
    rg : np.ndarray, shape (n_frames,)
    """
    segid, topology, trajectory, resid_range, start, stop, stride = args

    u = mda.Universe(topology, trajectory)

    if resid_range is not None:
        sel = u.select_atoms(f"segid {segid} and resid {resid_range} and protein")
    else:
        sel = u.select_atoms(f"segid {segid} and protein")

    if len(sel.atoms) == 0:
        print(f"  [!] Segment {segid}: no atoms selected – skipping.")
        return segid, None

    # stop=-1 must become None so the slice reaches the last frame
    traj_stop = None if stop == -1 else stop

    rg = []
    for _ts in u.trajectory[start:traj_stop:stride]:
        rg.append(sel.radius_of_gyration())

    if len(rg) == 0:
        print(f"  [!] Segment {segid}: no frames collected – skipping.")
        return segid, None

    print(f"  [+] Segment {segid}: {len(rg)} frames")
    return segid, np.array(rg)


# ─────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────

def run(
    topology: str,
    trajectory: str,
    resid_range: str  = None,
    output_file: str  = "rg.dat",
    start: int  = 0,
    stop: int   = -1,
    stride: int = 10,
    n_proc: int = 4,
):
    # Discover all segments that contain protein residues
    u = mda.Universe(topology, trajectory)
    if resid_range is not None:
        segments = [seg for seg in u.segments
                    if len(seg.atoms.select_atoms(
                        f"resid {resid_range} and protein")) > 0]
    else:
        segments = [seg for seg in u.segments
                    if len(seg.atoms.select_atoms("protein")) > 0]

    if not segments:
        raise RuntimeError("No matching protein segments found.")

    segids = [seg.segid for seg in segments]
    region_desc = f"resid {resid_range}" if resid_range else "full protein"
    print(f"[i] Found {len(segids)} segment(s): {segids}")
    print(f"[i] Region      : {region_desc}")
    print(f"[i] Frame range : start={start}, stop={stop}, stride={stride}")

    args_list = [
        (segid, topology, trajectory, resid_range, start, stop, stride)
        for segid in segids
    ]

    with Pool(n_proc) as pool:
        results = pool.map(_compute_segment_rg, args_list)

    # Collect valid results, preserving segment order
    valid_segids = []
    rg_all = []
    for segid, rg in results:
        if rg is not None:
            valid_segids.append(segid)
            rg_all.append(rg)

    if not rg_all:
        print("[!] No valid results – nothing written.")
        return

    # Align lengths (should be identical, but guard anyway)
    min_len = min(len(r) for r in rg_all)
    rg_all = [r[:min_len] for r in rg_all]

    frames = np.arange(min_len)
    data = np.column_stack([frames] + rg_all)

    # np.savetxt prepends "# " (2 chars) to the header line, so the first
    # field must be 2 chars shorter to stay aligned with fmt="%W..."
    W = 14
    header_cols = ([f"{'frame':>{W - 2}}"] +
                   [f"{s:>{W}}" for s in valid_segids])
    header = " ".join(header_cols)
    np.savetxt(output_file, data,
               header=header,
               fmt=[f"%{W}.0f"] + [f"%{W}.6f"] * len(valid_segids))

    print(f"[+] Rg saved -> {output_file}  "
          f"({len(valid_segids)} segments, {min_len} frames)")


# ─────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Per-segment radius of gyration time series"
    )
    p.add_argument("--top",    default="conf.psf",   help="Topology file (default: conf.psf)")
    p.add_argument("--traj",   default="system.xtc", help="Trajectory file (default: system.xtc)")
    p.add_argument("--resid",  default=None,
                   help="Residue range, e.g. '1:38' or '98:141'. "
                        "Omit for full protein (default)")
    p.add_argument("--out",    default="rg.dat",     help="Output file (default: rg.dat)")
    p.add_argument("--start",  type=int, default=0,  help="First frame index (default: 0)")
    p.add_argument("--stop",   type=int, default=-1, help="Last frame index, -1 = end (default: -1)")
    p.add_argument("--stride", type=int, default=10, help="Frame stride (default: 10)")
    p.add_argument("--nproc",  type=int, default=4,  help="Parallel workers (default: 4)")
    a = p.parse_args()

    run(
        topology=a.top,
        trajectory=a.traj,
        resid_range=a.resid,
        output_file=a.out,
        start=a.start,
        stop=a.stop,
        stride=a.stride,
        n_proc=a.nproc,
    )


if __name__ == "__main__":
    _cli()