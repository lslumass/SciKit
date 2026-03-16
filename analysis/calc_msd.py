#!/usr/bin/env python3
"""
calc_msd.py
-----------
Calculate the mean squared displacement (MSD) for Cα atoms of every protein
segment in an MD trajectory using MDAnalysis EinsteinMSD (FFT method).

Two output files are written per segment into --outdir:
    {segid}_msd.dat          lag_time(ps)  MSD(A^2)       — total MSD
    {segid}_CA_msd.dat       per-residue MSD matrix       — only with --per-residue

Optional residue range (--resid) restricts the Cα selection to a sub-region.
By default the full protein of each segment is used.

Usage:
    python calc_msd.py [--top conf.psf] [--traj system.xtc]
                       [--resid FIRST:LAST]
                       [--outdir ./diffusion]
                       [--start 0] [--stop -1] [--stride 1]
                       [--per-residue]
                       [--nproc 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd_mod


# ─────────────────────────────────────────────
#  Per-segment worker (runs in a subprocess)
# ─────────────────────────────────────────────

def _run_msd_for_segment(args: tuple):
    """
    Worker: run EinsteinMSD for one segment, write output files.

    args = (segid, topology, trajectory, resid_range,
            start, stop, stride, outdir, per_residue)
    """
    segid, topology, trajectory, resid_range, \
        start, stop, stride, outdir, per_residue = args

    u = mda.Universe(topology, trajectory)

    # Build atom selection
    if resid_range is not None:
        selection = f"name CA and segid {segid} and resid {resid_range}"
    else:
        selection = f"name CA and segid {segid}"

    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"No atoms matched selection: '{selection}'")

    # stop=-1 must become None so the MDAnalysis slice reaches the last frame
    traj_stop = None if stop == -1 else stop

    MSD = msd_mod.EinsteinMSD(u, select=selection, msd_type="xyz", fft=True)
    MSD.run(start=start, stop=traj_stop, step=stride)

    # Lag times and total MSD
    # delta_t_values is in ps and has the same length as timeseries
    lagtimes = MSD.results.delta_t_values   # shape (n_lags,)
    msds     = MSD.results.timeseries       # shape (n_lags,)

    # ── total MSD file ──────────────────────────────────────────────────────
    total_file = os.path.join(outdir, f"{segid}_msd.dat")
    W = 16
    header_total = f"{'lag_time(ps)':>{W - 2}} {'MSD(A^2)':>{W}}"
    np.savetxt(
        total_file,
        np.column_stack((lagtimes, msds)),
        header=header_total,
        fmt=f"%{W}.6f",
    )

    # ── per-residue MSD file (optional) ────────────────────────────────────
    if per_residue:
        # msds_by_particle : shape (n_lags, n_particles)
        # Write as: lag_time(ps)  res1  res2  ...
        msds_by_res = MSD.results.msds_by_particle   # (n_lags, n_CA)
        resids = ag.resids                            # residue IDs in selection order

        W2 = 14
        header_res_cols = ([f"{'lag_time(ps)':>{W2 - 2}}"] +
                           [f"{r:>{W2}}" for r in resids])
        header_res = " ".join(header_res_cols)
        per_res_file = os.path.join(outdir, f"{segid}_CA_msd.dat")
        np.savetxt(
            per_res_file,
            np.column_stack((lagtimes, msds_by_res)),
            header=header_res,
            fmt=[f"%{W2}.6f"] * (1 + len(resids)),
        )

    return segid


# ─────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────

def run(
    topology: str,
    trajectory: str,
    resid_range: str  = None,
    outdir: str       = "./diffusion",
    start: int  = 0,
    stop: int   = -1,
    stride: int = 1,
    per_residue: bool = False,
    n_proc: int = 4,
):
    os.makedirs(outdir, exist_ok=True)

    # Discover segments containing the target Cα atoms
    u = mda.Universe(topology, trajectory)
    if resid_range is not None:
        segments = [seg for seg in u.segments
                    if len(seg.atoms.select_atoms(
                        f"name CA and resid {resid_range}")) > 0]
    else:
        segments = [seg for seg in u.segments
                    if len(seg.atoms.select_atoms("name CA")) > 0]

    if not segments:
        raise RuntimeError("No matching segments found.")

    segids = [seg.segid for seg in segments]
    region_desc = f"resid {resid_range}" if resid_range else "full protein"
    print(f"[i] Found {len(segids)} segment(s): {segids}")
    print(f"[i] Region      : {region_desc} (Cα atoms)")
    print(f"[i] Frame range : start={start}, stop={stop}, stride={stride}")
    print(f"[i] Per-residue : {per_residue}")
    print(f"[i] Output dir  : {outdir}")
    del u

    args_list = [
        (segid, topology, trajectory, resid_range,
         start, stop, stride, outdir, per_residue)
        for segid in segids
    ]

    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futures = {
            executor.submit(_run_msd_for_segment, a): a[0]
            for a in args_list
        }
        for future in as_completed(futures):
            segid = futures[future]
            try:
                future.result()
                print(f"  [+] Done: {segid}")
            except Exception as exc:
                print(f"  [!] Failed: {segid} -> {exc}")

    print(f"[+] MSD results written to {outdir}/")


# ─────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Per-segment Cα MSD via MDAnalysis EinsteinMSD (FFT)"
    )
    p.add_argument("--top",          default="conf.psf",      help="Topology file (default: conf.psf)")
    p.add_argument("--traj",         default="system.xtc",    help="Trajectory file (default: system.xtc)")
    p.add_argument("--resid",        default=None,
                   help="Residue range, e.g. '1:38'. Omit for full protein (default)")
    p.add_argument("--outdir",       default="./diffusion",   help="Output directory (default: ./diffusion)")
    p.add_argument("--start",        type=int, default=0,     help="First frame index (default: 0)")
    p.add_argument("--stop",         type=int, default=-1,    help="Last frame index, -1 = end (default: -1)")
    p.add_argument("--stride",       type=int, default=1,     help="Frame stride (default: 1)")
    p.add_argument("--per-residue",  action="store_true",
                   help="Also write per-Cα MSD files (default: off)")
    p.add_argument("--nproc",        type=int, default=4,     help="Parallel workers (default: 4)")
    a = p.parse_args()

    run(
        topology=a.top,
        trajectory=a.traj,
        resid_range=a.resid,
        outdir=a.outdir,
        start=a.start,
        stop=a.stop,
        stride=a.stride,
        per_residue=a.per_residue,
        n_proc=a.nproc,
    )


if __name__ == "__main__":
    _cli()