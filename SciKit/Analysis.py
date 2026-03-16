#!/usr/bin/env python3
"""
Analysis.py
===========
Unified MD analysis toolkit — scikit-style package.

All analyses are exposed as Typer sub-commands through a single ``app``:

    python Analysis.py --help
    python Analysis.py msd        --top conf.psf --traj system.xtc
    python Analysis.py rg         --top conf.psf --traj system.xtc
    python Analysis.py dssp       --top conf.psf --traj system.xtc
    python Analysis.py distance   --top conf.psf --traj system.dcd  -f pairs.dat
    python Analysis.py distance-acf  --top conf.psf --traj system.xtc --pairs pairs.dat
    python Analysis.py vector-acf    --top conf.psf --traj system.xtc --pairs pairs.dat
    python Analysis.py contacts      --top system.psf --traj traj.dcd
    python Analysis.py aggr          --top conf.psf --traj system.xtc

Requires
--------
    mdanalysis  typer  numpy  scipy
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import numpy as np
import typer
from scipy.spatial import cKDTree
from typing_extensions import Annotated

# ─────────────────────────────────────────────────────────────────────────────
#  Typer application
# ─────────────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="analysis",
    help=(
        "MDAnalysis toolkit — "
        "MSD · Rg · DSSP · Distance · ACF · Contacts · Aggregation"
    ),
    add_completion=False,
    rich_markup_mode="markdown",
)


# =============================================================================
#  ░░  SHARED UTILITIES  ░░
# =============================================================================

def _traj_stop(stop: int) -> Optional[int]:
    """Convert CLI stop=-1 sentinel to None (MDA reads to the end)."""
    return None if stop == -1 else stop


# =============================================================================
#  ░░  1.  MSD  ░░
# =============================================================================

def _msd_worker(args: tuple):
    """Subprocess worker: run EinsteinMSD for one segment."""
    import MDAnalysis as mda
    import MDAnalysis.analysis.msd as msd_mod

    segid, topology, trajectory, resid_range, \
        start, stop, stride, outdir, per_residue = args

    u = mda.Universe(topology, trajectory)

    selection = (f"name CA and segid {segid} and resid {resid_range}"
                 if resid_range else f"name CA and segid {segid}")

    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"No atoms matched: '{selection}'")

    MSD = msd_mod.EinsteinMSD(u, select=selection, msd_type="xyz", fft=True)
    MSD.run(start=start, stop=_traj_stop(stop), step=stride)

    lagtimes = MSD.results.delta_t_values
    msds     = MSD.results.timeseries

    W = 16
    total_file = os.path.join(outdir, f"{segid}_msd.dat")
    np.savetxt(
        total_file,
        np.column_stack((lagtimes, msds)),
        header=f"{'lag_time(ps)':>{W - 2}} {'MSD(A^2)':>{W}}",
        fmt=f"%{W}.6f",
    )

    if per_residue:
        msds_by_res = MSD.results.msds_by_particle
        resids      = ag.resids
        W2          = 14
        header_res  = " ".join(
            [f"{'lag_time(ps)':>{W2 - 2}}"] + [f"{r:>{W2}}" for r in resids]
        )
        np.savetxt(
            os.path.join(outdir, f"{segid}_CA_msd.dat"),
            np.column_stack((lagtimes, msds_by_res)),
            header=header_res,
            fmt=[f"%{W2}.6f"] * (1 + len(resids)),
        )

    return segid


@app.command("msd")
def cmd_msd(
    top: Annotated[str, typer.Option("--top", help="Topology file")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj", help="Trajectory file")] = "system.xtc",
    resid: Annotated[Optional[str], typer.Option("--resid", help="Residue range e.g. '1:38'")] = None,
    outdir: Annotated[str, typer.Option("--outdir", help="Output directory")] = "./diffusion",
    start: Annotated[int, typer.Option("--start", help="First frame index")] = 0,
    stop: Annotated[int, typer.Option("--stop", help="Last frame index; -1=end")] = -1,
    stride: Annotated[int, typer.Option("--stride", help="Frame stride")] = 1,
    per_residue: Annotated[bool, typer.Option("--per-residue/--no-per-residue",
                                               help="Write per-Cα MSD files")] = False,
    nproc: Annotated[int, typer.Option("--nproc", help="Parallel workers")] = 4,
):
    """Calculate per-segment Cα **mean squared displacement** (FFT method)."""
    import MDAnalysis as mda

    os.makedirs(outdir, exist_ok=True)

    u = mda.Universe(top, traj)
    segments = [
        seg for seg in u.segments
        if len(seg.atoms.select_atoms(
            f"name CA{' and resid ' + resid if resid else ''}"
        )) > 0
    ]
    if not segments:
        typer.echo("[!] No matching segments found.", err=True)
        raise typer.Exit(1)

    segids = [seg.segid for seg in segments]
    typer.echo(f"[i] Segments    : {segids}")
    typer.echo(f"[i] Region      : {'resid ' + resid if resid else 'full protein'}")
    typer.echo(f"[i] Frames      : start={start} stop={stop} stride={stride}")
    typer.echo(f"[i] Per-residue : {per_residue}")
    typer.echo(f"[i] Output dir  : {outdir}")
    del u

    args_list = [
        (segid, top, traj, resid, start, stop, stride, outdir, per_residue)
        for segid in segids
    ]

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        futures = {executor.submit(_msd_worker, a): a[0] for a in args_list}
        for future in as_completed(futures):
            segid = futures[future]
            try:
                future.result()
                typer.echo(f"  [+] Done: {segid}")
            except Exception as exc:
                typer.echo(f"  [!] Failed: {segid} -> {exc}", err=True)

    typer.echo(f"[+] MSD results written to {outdir}/")


# =============================================================================
#  ░░  2.  Rg  ░░
# =============================================================================

def _rg_worker(args: tuple):
    """Subprocess worker: compute Rg time series for one segment."""
    import MDAnalysis as mda

    segid, topology, trajectory, resid_range, start, stop, stride = args
    u   = mda.Universe(topology, trajectory)
    sel = u.select_atoms(
        f"segid {segid} and resid {resid_range} and protein"
        if resid_range else f"segid {segid} and protein"
    )

    if len(sel.atoms) == 0:
        return segid, None

    rg = [sel.radius_of_gyration()
          for _ts in u.trajectory[start:_traj_stop(stop):stride]]

    if not rg:
        return segid, None

    typer.echo(f"  [+] Segment {segid}: {len(rg)} frames")
    return segid, np.array(rg)


@app.command("rg")
def cmd_rg(
    top: Annotated[str, typer.Option("--top")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj")] = "system.xtc",
    resid: Annotated[Optional[str], typer.Option("--resid", help="Residue range")] = None,
    out: Annotated[str, typer.Option("--out", help="Output file")] = "rg.dat",
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[int, typer.Option("--stop")] = -1,
    stride: Annotated[int, typer.Option("--stride")] = 10,
    nproc: Annotated[int, typer.Option("--nproc")] = 4,
):
    """Calculate per-segment **radius of gyration** time series."""
    import MDAnalysis as mda

    u = mda.Universe(top, traj)
    segments = [
        seg for seg in u.segments
        if len(seg.atoms.select_atoms(
            f"resid {resid} and protein" if resid else "protein"
        )) > 0
    ]
    if not segments:
        typer.echo("[!] No matching protein segments found.", err=True)
        raise typer.Exit(1)

    segids = [seg.segid for seg in segments]
    typer.echo(f"[i] Segments : {segids}")
    typer.echo(f"[i] Region   : {'resid ' + resid if resid else 'full protein'}")

    with Pool(nproc) as pool:
        results = pool.map(
            _rg_worker,
            [(s, top, traj, resid, start, stop, stride) for s in segids],
        )

    valid_segids, rg_all = [], []
    for segid, rg in results:
        if rg is not None:
            valid_segids.append(segid)
            rg_all.append(rg)

    if not rg_all:
        typer.echo("[!] No valid results — nothing written.", err=True)
        return

    min_len  = min(len(r) for r in rg_all)
    rg_all   = [r[:min_len] for r in rg_all]
    data     = np.column_stack([np.arange(min_len)] + rg_all)
    W        = 14
    header   = " ".join(
        [f"{'frame':>{W - 2}}"] + [f"{s:>{W}}" for s in valid_segids]
    )
    np.savetxt(out, data, header=header,
               fmt=[f"%{W}.0f"] + [f"%{W}.6f"] * len(valid_segids))
    typer.echo(f"[+] Rg saved -> {out}  ({len(valid_segids)} segments, {min_len} frames)")


# =============================================================================
#  ░░  3.  DSSP  ░░
# =============================================================================

def _dssp_worker(args: tuple):
    """Subprocess worker: run DSSP on one segment."""
    import MDAnalysis as mda
    from MDAnalysis.analysis.dssp import DSSP

    segid, topology, trajectory, start, stop, stride = args
    u   = mda.Universe(topology, trajectory)
    sel = u.select_atoms(f"segid {segid} and protein")

    if len(sel.residues) == 0:
        return segid, None, None, None

    dssp = DSSP(sel)
    dssp.run(start=start, stop=_traj_stop(stop), step=stride)
    codes = dssp.results.dssp        # (n_frames, n_residues)
    n_frames, _ = codes.shape

    if n_frames == 0:
        return segid, None, None, None

    helicity = np.sum(codes == "H", axis=0) / n_frames
    beta     = np.sum(codes == "E", axis=0) / n_frames
    resids   = sel.residues.resids.copy()

    typer.echo(f"  [+] Segment {segid}: {len(resids)} residues, {n_frames} frames")
    return segid, resids, helicity, beta


def _write_dssp_dat(output_file, all_resids, values_by_seg, segids):
    all_res_sorted = np.array(sorted(set(all_resids)))
    n_res, n_seg   = len(all_res_sorted), len(segids)
    mat            = np.full((n_res, n_seg), np.nan)
    res_index      = {r: i for i, r in enumerate(all_res_sorted)}

    for col, (resids, vals) in enumerate(values_by_seg):
        for r, v in zip(resids, vals):
            if r in res_index:
                mat[res_index[r], col] = v

    W      = 12
    header = " ".join(
        [f"{'residue':>{W - 2}}"] + [f"{s:>{W}}" for s in segids]
    )
    data = np.column_stack([all_res_sorted.astype(float), mat])
    np.savetxt(output_file, data, header=header,
               fmt=[f"%{W}.0f"] + [f"%{W}.6f"] * n_seg)
    typer.echo(f"  [✓] Saved {output_file}  ({n_res} residues, {n_seg} segments)")


@app.command("dssp")
def cmd_dssp(
    top: Annotated[str, typer.Option("--top")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj")] = "system.xtc",
    hout: Annotated[str, typer.Option("--hout", help="Helicity output file")] = "helicity.dat",
    bout: Annotated[str, typer.Option("--bout", help="Beta-sheet output file")] = "beta.dat",
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[int, typer.Option("--stop")] = -1,
    stride: Annotated[int, typer.Option("--stride")] = 10,
    nproc: Annotated[int, typer.Option("--nproc")] = 4,
):
    """Calculate per-residue **DSSP** helicity and beta-sheet content."""
    import MDAnalysis as mda

    u        = mda.Universe(top, traj)
    segments = [seg for seg in u.segments
                if len(seg.atoms.select_atoms("protein")) > 0]
    if not segments:
        typer.echo("[!] No protein segments found.", err=True)
        raise typer.Exit(1)

    segids = [seg.segid for seg in segments]
    typer.echo(f"[i] Segments : {segids}")

    with Pool(nproc) as pool:
        results = pool.map(
            _dssp_worker,
            [(s, top, traj, start, stop, stride) for s in segids],
        )

    valid_segids   = []
    helicity_pairs = []
    beta_pairs     = []
    all_resids     = []

    for segid, resids, helicity, beta in results:
        if resids is None:
            continue
        valid_segids.append(segid)
        helicity_pairs.append((resids, helicity))
        beta_pairs.append((resids, beta))
        all_resids.extend(resids.tolist())

    if not valid_segids:
        typer.echo("[!] No valid results — nothing written.", err=True)
        return

    _write_dssp_dat(hout, all_resids, helicity_pairs, valid_segids)
    _write_dssp_dat(bout, all_resids, beta_pairs,     valid_segids)


# =============================================================================
#  ░░  4.  Pair Distance  ░░
# =============================================================================

def _warn(msg: str):
    print(f"[WARNING] {msg}", file=sys.stderr)


def _load_pairs(filepath: str):
    pairs = []
    with open(filepath) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = ([x.strip() for x in line.split(",")]
                     if "," in line else line.split())
            if len(parts) != 4:
                _warn(f"{filepath}:{lineno} — expected 4 fields, skipping: {line!r}")
                continue
            resid1, segid1, resid2, segid2 = parts
            try:
                pairs.append((int(resid1), segid1, int(resid2), segid2))
            except ValueError:
                _warn(f"{filepath}:{lineno} — non-integer resid, skipping: {line!r}")
    return pairs


def _load_all_pairs(filepaths):
    file_pairs, seen, all_pairs = {}, set(), []
    for fp in filepaths:
        pairs = _load_pairs(fp)
        file_pairs[fp] = pairs
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                all_pairs.append(pair)
        typer.echo(f"  {fp}: {len(pairs)} pairs  ({len(all_pairs)} unique total)")
    return all_pairs, file_pairs


def _resolve_frame_range(n_total, start, stop, stride):
    start  = max(0, start if start is not None else 0)
    stop   = min(n_total, stop if stop is not None else n_total)
    stride = max(1, stride)
    if start >= stop:
        raise ValueError(f"start={start} >= stop={stop}; no frames selected.")
    return start, stop, stride


def _n_selected_frames(start, stop, stride):
    return max(0, (stop - start + stride - 1) // stride)


def _make_worker_slices(start, stop, stride, n_workers):
    sel      = np.arange(start, stop, stride)
    n_actual = min(n_workers, len(sel))
    slices   = []
    for chunk in np.array_split(sel, n_actual):
        if len(chunk) == 0:
            continue
        slices.append((int(chunk[0]), int(chunk[-1]) + stride, stride))
    return slices


def _resolve_ca_indices(universe, pairs):
    import MDAnalysis as mda
    unique_res = {}
    for r1, s1, r2, s2 in pairs:
        for resid, segid in ((r1, s1), (r2, s2)):
            if (resid, segid) not in unique_res:
                sel = universe.select_atoms(
                    f"resid {resid} and segid {segid} and name CA")
                if len(sel) == 0:
                    raise ValueError(f"No CA: resid={resid}, segid={segid}")
                if len(sel) > 1:
                    _warn(f"Multiple CAs for resid={resid} segid={segid}; using first.")
                unique_res[(resid, segid)] = sel[0].index
    return ([unique_res[(r1, s1)] for r1, s1, r2, s2 in pairs],
            [unique_res[(r2, s2)] for r1, s1, r2, s2 in pairs])


def _dist_worker(args):
    import MDAnalysis as mda
    topology, trajectory, idx1, idx2, traj_slice, worker_id = args
    w_start, w_stop, stride = traj_slice

    u      = mda.Universe(topology, trajectory)
    ag1    = u.atoms[idx1]
    ag2    = u.atoms[idx2]
    n_pairs = len(idx1)
    n_est   = _n_selected_frames(w_start, w_stop, stride)
    buf     = np.empty((n_pairs, n_est), dtype=np.float32)
    frames  = []

    for col, ts in enumerate(u.trajectory[w_start:w_stop:stride]):
        buf[:, col] = np.linalg.norm(ag1.positions - ag2.positions, axis=1)
        frames.append(ts.frame)

    n_actual = len(frames)
    typer.echo(f"  Worker {worker_id:02d}: traj[{w_start}:{w_stop}:{stride}] -> {n_actual} frames")
    return frames, buf[:, :n_actual].copy()


def _write_dist_output(out_path, pairs, dist_block, all_frames):
    float_buf = io.StringIO()
    np.savetxt(float_buf, dist_block, fmt="%.4f", delimiter=", ")
    float_buf.seek(0)
    float_lines = float_buf.read().splitlines()
    frame_header = ", ".join(f"frame_{fi}" for fi in all_frames)
    with open(out_path, "w") as fh:
        fh.write(f"# resid_1, segid_1, resid_2, segid_2, {frame_header}\n")
        for (r1, s1, r2, s2), fline in zip(pairs, float_lines):
            fh.write(f"{r1}, {s1}, {r2}, {s2}, {fline}\n")


@app.command("distance")
def cmd_dist(
    top: Annotated[str, typer.Option("-p", "--top")] = "conf.psf",
    traj: Annotated[str, typer.Option("-t", "--traj")] = "system.dcd",
    pair_files: Annotated[List[str], typer.Option("-f", "--pair-files",
                help="One or more residue-pair list files")] = ["./residue_pairs.dat"],
    start: Annotated[Optional[int], typer.Option("--start")] = None,
    stop: Annotated[Optional[int], typer.Option("--stop")] = None,
    stride: Annotated[int, typer.Option("--stride")] = 1,
    workers: Annotated[int, typer.Option("-n", "--workers",
               help="Parallel worker processes")] = max(1, cpu_count() or 1),
):
    """Calculate **Cα–Cα distances** for residue pairs over a trajectory."""
    import MDAnalysis as mda

    typer.echo("=" * 60)
    typer.echo(f"Topology  : {top}")
    typer.echo(f"Trajectory: {traj}")
    typer.echo(f"Pairs     : {pair_files}")
    typer.echo("=" * 60)

    u0      = mda.Universe(top, traj)
    n_total = len(u0.trajectory)
    s, e, st = _resolve_frame_range(n_total, start, stop, stride)
    typer.echo(f"Frames selected: {_n_selected_frames(s, e, st)}  [{s}:{e}:{st}]")

    all_pairs, file_pairs = _load_all_pairs(pair_files)
    if not all_pairs:
        typer.echo("[ERROR] No pairs loaded.", err=True)
        raise typer.Exit(1)

    idx1, idx2 = _resolve_ca_indices(u0, all_pairs)
    del u0

    slices   = _make_worker_slices(s, e, st, workers)
    work_args = [
        (top, traj, idx1, idx2, sl, wid)
        for wid, sl in enumerate(slices)
    ]

    with Pool(processes=len(slices)) as pool:
        results = pool.map(_dist_worker, work_args)

    results.sort(key=lambda r: r[0][0])
    all_frames, chunks = [], []
    for frames, data in results:
        all_frames.extend(frames)
        chunks.append(data)

    full         = np.hstack(chunks)
    pair_to_row  = {pair: i for i, pair in enumerate(all_pairs)}

    for fp in pair_files:
        out_path   = os.path.join(
            os.path.dirname(os.path.abspath(fp)),
            os.path.splitext(os.path.basename(fp))[0] + "_distance.dat",
        )
        pairs      = file_pairs[fp]
        rows       = [pair_to_row[p] for p in pairs]
        _write_dist_output(out_path, pairs, full[rows, :], all_frames)
        typer.echo(f"  -> {out_path}  ({len(pairs)} pairs)")

    typer.echo("\nAll done.")


# =============================================================================
#  ░░  5.  Pair Distance ACF  ░░
# =============================================================================

def _autocorr_fft_scalar(x: np.ndarray) -> np.ndarray:
    """Normalised fluctuation ACF of a 1-D scalar series via FFT."""
    N   = len(x)
    xm  = x - x.mean()
    fx  = np.fft.fft(xm, n=2 * N)
    pwr = np.real(np.fft.ifft(fx * np.conj(fx)))[:N]
    pwr /= np.arange(N, 0, -1)
    return pwr / pwr[0]


def _dist_acf_worker(args: tuple):
    import MDAnalysis as mda

    (label, topology, trajectory,
     resid1, segid1, resid2, segid2,
     start, stop, stride) = args

    u    = mda.Universe(topology, trajectory)
    sel1 = u.select_atoms(f"segid {segid1} and resid {resid1} and name CA")
    sel2 = u.select_atoms(f"segid {segid2} and resid {resid2} and name CA")

    if len(sel1) == 0 or len(sel2) == 0:
        typer.echo(f"  [!] Pair {label}: atom(s) not found — skipping.", err=True)
        return label, None

    distances = [
        np.linalg.norm(sel2.positions[0] - sel1.positions[0])
        for _ts in u.trajectory[start:_traj_stop(stop):stride]
    ]
    if len(distances) < 2:
        return label, None

    return label, _autocorr_fft_scalar(np.array(distances))


def _read_pairs_file(path: str) -> list:
    pairs = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) < 4:
                raise ValueError(f"{path}:{lineno} — expected 4 fields, got: {raw!r}")
            resid1, segid1, resid2, segid2 = tokens[:4]
            pairs.append((resid1, segid1, resid2, segid2))
    return pairs


def _pair_label(r1, s1, r2, s2) -> str:
    return f"{s1}:{r1}-{s2}:{r2}"


def _save_acf(output_file, labels, C_all, topology, trajectory, stride):
    import MDAnalysis as mda
    min_len = min(len(c) for c in C_all)
    C_all   = [c[:min_len] for c in C_all]
    u_ref   = mda.Universe(topology, trajectory)
    tau     = np.arange(min_len) * u_ref.trajectory.dt * stride
    data    = np.column_stack([tau] + C_all)
    W       = 14
    header  = " ".join(
        [f"{'tau(ps)':>{W - 2}}"] + [f"{lbl:>{W}}" for lbl in labels]
    )
    np.savetxt(output_file, data, header=header, fmt=f"%{W}.6f")
    typer.echo(f"[+] ACF saved -> {output_file}  ({len(labels)} pairs, {min_len} lags)")


@app.command("distance-acf")
def cmd_dist_acf(
    top: Annotated[str, typer.Option("--top")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj")] = "system.xtc",
    pairs: Annotated[str, typer.Option("--pairs", help="Pairs file")] = "pairs.dat",
    out: Annotated[str, typer.Option("--out", help="Output file")] = "distance_acf.dat",
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[int, typer.Option("--stop")] = -1,
    stride: Annotated[int, typer.Option("--stride")] = 10,
    nproc: Annotated[int, typer.Option("--nproc")] = 4,
):
    """Compute inter-Cα **distance autocorrelation** for residue pairs."""
    pair_list = _read_pairs_file(pairs)
    if not pair_list:
        typer.echo(f"[!] No valid pairs in {pairs}", err=True)
        raise typer.Exit(1)

    typer.echo(f"[i] {len(pair_list)} pair(s) from {pairs}")

    args_list = [
        (_pair_label(r1, s1, r2, s2), top, traj, r1, s1, r2, s2, start, stop, stride)
        for r1, s1, r2, s2 in pair_list
    ]

    with Pool(nproc) as pool:
        results = pool.map(_dist_acf_worker, args_list)

    labels, C_all = zip(*[(lbl, C) for lbl, C in results if C is not None]) \
        if any(C is not None for _, C in results) else ([], [])

    if not C_all:
        typer.echo("[!] No valid results — nothing written.", err=True)
        return

    _save_acf(out, list(labels), list(C_all), top, traj, stride)


# =============================================================================
#  ░░  6.  Pair Vector ACF  ░░
# =============================================================================

def _raw_acf_1d(x: np.ndarray) -> np.ndarray:
    N   = len(x)
    fx  = np.fft.fft(x, n=2 * N)
    pwr = np.real(np.fft.ifft(fx * np.conj(fx)))[:N]
    pwr /= np.arange(N, 0, -1)
    return pwr


def _autocorr_fft_vector(R: np.ndarray) -> np.ndarray:
    """
    C(t) = [<Rx(0)Rx(t)> + <Ry(0)Ry(t)> + <Rz(0)Rz(t)>]
           / <|R|^2>
    """
    C = _raw_acf_1d(R[:, 0]) + _raw_acf_1d(R[:, 1]) + _raw_acf_1d(R[:, 2])
    return C / C[0]


def _vec_acf_worker(args: tuple):
    import MDAnalysis as mda

    (label, topology, trajectory,
     resid1, segid1, resid2, segid2,
     start, stop, stride) = args

    u    = mda.Universe(topology, trajectory)
    sel1 = u.select_atoms(f"segid {segid1} and resid {resid1} and name CA")
    sel2 = u.select_atoms(f"segid {segid2} and resid {resid2} and name CA")

    if len(sel1) == 0 or len(sel2) == 0:
        typer.echo(f"  [!] Pair {label}: atom(s) not found — skipping.", err=True)
        return label, None

    R = [
        sel2.positions[0] - sel1.positions[0]
        for _ts in u.trajectory[start:_traj_stop(stop):stride]
    ]
    if len(R) < 2:
        return label, None

    return label, _autocorr_fft_vector(np.array(R))


@app.command("vector-acf")
def cmd_vector_acf(
    top: Annotated[str, typer.Option("--top")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj")] = "system.xtc",
    pairs: Annotated[str, typer.Option("--pairs")] = "pairs.dat",
    out: Annotated[str, typer.Option("--out")] = "vector_acf.dat",
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[int, typer.Option("--stop")] = -1,
    stride: Annotated[int, typer.Option("--stride")] = 10,
    nproc: Annotated[int, typer.Option("--nproc")] = 4,
):
    """Compute end-to-end Cα **vector autocorrelation** for residue pairs."""
    pair_list = _read_pairs_file(pairs)
    if not pair_list:
        typer.echo(f"[!] No valid pairs in {pairs}", err=True)
        raise typer.Exit(1)

    typer.echo(f"[i] {len(pair_list)} pair(s) from {pairs}")

    args_list = [
        (_pair_label(r1, s1, r2, s2), top, traj, r1, s1, r2, s2, start, stop, stride)
        for r1, s1, r2, s2 in pair_list
    ]

    with Pool(nproc) as pool:
        results = pool.map(_vec_acf_worker, args_list)

    labels, C_all = [], []
    for lbl, C in results:
        if C is not None:
            labels.append(lbl)
            C_all.append(C)

    if not C_all:
        typer.echo("[!] No valid results — nothing written.", err=True)
        return

    _save_acf(out, labels, C_all, top, traj, stride)


# =============================================================================
#  ░░  7.  Contact Map  ░░
# =============================================================================

HEAVY_ATOMS = "name CA CB CC CD CE CF"


def _contacts_worker(args):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    top, traj, cutoff, frame_indices, atom_to_local, atom_to_copy, N = args

    u       = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_hvy = u.select_atoms(HEAVY_ATOMS)
    partial_intra = np.zeros((N, N), dtype=np.float64)
    partial_inter = np.zeros((N, N), dtype=np.float64)

    for frame in frame_indices:
        u.trajectory[frame]
        pos_all = all_hvy.positions
        box     = u.dimensions

        pairs_raw, _ = capped_distance(pos_all, pos_all, cutoff,
                                       box=box, return_distances=True)
        if len(pairs_raw) == 0:
            continue

        ri = pairs_raw[:, 0].astype(np.int32)
        rj = pairs_raw[:, 1].astype(np.int32)
        li = atom_to_local[ri]
        lj = atom_to_local[rj]
        ci = atom_to_copy[ri]
        cj = atom_to_copy[rj]

        intra_mask = (ci == cj) & (np.abs(li - lj) >= 4)
        if intra_mask.any():
            key  = np.unique(np.stack([ci[intra_mask], li[intra_mask], lj[intra_mask]], axis=1), axis=0)
            flat = key[:, 1] * N + key[:, 2]
            m    = np.bincount(flat, minlength=N * N).reshape(N, N).astype(np.float64)
            partial_intra += m + m.T

        inter_mask = ci != cj
        if inter_mask.any():
            key2  = np.unique(np.stack([ci[inter_mask], cj[inter_mask],
                                        li[inter_mask], lj[inter_mask]], axis=1), axis=0)
            flat2 = key2[:, 2] * N + key2[:, 3]
            m2    = np.bincount(flat2, minlength=N * N).reshape(N, N).astype(np.float64)
            partial_inter += m2 + m2.T

    return partial_intra, partial_inter


@app.command("contacts")
def cmd_contacts(
    top: Annotated[str, typer.Option("--top", help="Topology file")] = "system.psf",
    traj: Annotated[Optional[str], typer.Option("--traj", help="Trajectory (omit for single frame)")] = None,
    cutoff: Annotated[float, typer.Option("--cutoff", help="Distance cutoff in Å")] = 6.0,
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[Optional[int], typer.Option("--stop")] = None,
    stride: Annotated[int, typer.Option("--stride")] = 1,
    nproc: Annotated[int, typer.Option("--nproc")] = 1,
    out: Annotated[str, typer.Option("--out", help="Output stem (appended with _intra/_inter)")] = "contact_map.npy",
):
    """Calculate intra- and inter-chain **contact maps** (heavy-atom, parallel)."""
    import MDAnalysis as mda

    warnings.filterwarnings("ignore")

    u      = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_ca = u.select_atoms("name CA")
    segids = list(dict.fromkeys(all_ca.segids))
    n_copies = len(segids)
    n_traj   = len(u.trajectory)

    seg_ca_list = [u.select_atoms(f"segid {s} and name CA") for s in segids]
    N_list      = [len(ca) for ca in seg_ca_list]
    if len(set(N_list)) != 1:
        typer.echo(f"[!] Unequal residue counts across copies: {set(N_list)}", err=True)
        raise typer.Exit(1)
    N = N_list[0]

    all_hvy     = u.select_atoms(HEAVY_ATOMS)
    M           = len(all_hvy)
    all_hvy_idx = all_hvy.indices

    atom_to_local = np.empty(M, dtype=np.int32)
    atom_to_copy  = np.empty(M, dtype=np.int32)

    for c, ca in enumerate(seg_ca_list):
        for res_local, res in enumerate(ca.residues):
            res_hvy_idx = np.intersect1d(res.atoms.indices, all_hvy_idx)
            rows = np.searchsorted(all_hvy_idx, res_hvy_idx)
            atom_to_local[rows] = res_local
            atom_to_copy[rows]  = c

    resids = seg_ca_list[0].residues.resids
    del u

    frames   = list(range(start, stop or n_traj, stride))
    n_frames = len(frames)
    nprocs   = min(nproc, n_frames)

    typer.echo(f"Copies        : {n_copies}  ({segids[0]} … {segids[-1]})")
    typer.echo(f"Residues/copy : {N}")
    typer.echo(f"Heavy atoms   : {M}")
    typer.echo(f"Cutoff        : {cutoff} Å")
    typer.echo(f"Frames        : {n_frames}  (stride={stride})")
    typer.echo(f"Workers       : {nprocs}")

    chunks      = [frames[i::nprocs] for i in range(nprocs)]
    worker_args = [
        (top, traj, cutoff, chunk, atom_to_local, atom_to_copy, N)
        for chunk in chunks
    ]

    t0 = time.perf_counter()
    if nprocs == 1:
        results = [_contacts_worker(worker_args[0])]
    else:
        ctx = mp.get_context("forkserver")
        with ctx.Pool(processes=nprocs) as pool:
            results = pool.map(_contacts_worker, worker_args)

    intra_sum, inter_sum = zip(*results)
    denom     = n_copies * n_frames
    intra_map = np.sum(intra_sum, axis=0) / denom
    inter_map = np.sum(inter_sum, axis=0) / denom

    stem = out.removesuffix(".npy")
    np.save(f"{stem}_intra.npy",  intra_map)
    np.save(f"{stem}_inter.npy",  inter_map)
    np.save(f"{stem}.npy",        intra_map + inter_map)
    np.save(f"{stem}_resids.npy", resids)

    elapsed = time.perf_counter() - t0
    typer.echo(f"\nDone in {elapsed:.2f}s")
    typer.echo(f"Saved -> {stem}_intra.npy   (max {intra_map.max():.4f})")
    typer.echo(f"Saved -> {stem}_inter.npy   (max {inter_map.max():.4f})")
    typer.echo(f"Saved -> {stem}.npy         (max {(intra_map+inter_map).max():.4f})")
    typer.echo(f"Saved -> {stem}_resids.npy  (resids {resids[0]}–{resids[-1]})")


# =============================================================================
#  ░░  8.  Aggregation / Clustering / Density  ░░
# =============================================================================

AVOGADRO = 6.022e23


def _load_universe(top: str, traj: Optional[str] = None):
    """Helper: load a Universe with or without a trajectory."""
    import MDAnalysis as mda
    return mda.Universe(top, traj) if traj else mda.Universe(top)


def _grp_init(u, ref_atom: str = "CA", step: int = 1):
    """Initialise per-segment atom groups for clustering."""
    segs, grps = [], []
    for i, segid in enumerate(u.segments):
        cas = u.select_atoms(f"segid {segid.segid} and name {ref_atom}")
        seg = cas[::step].atoms.select_atoms(f"name {ref_atom}")
        segs.append(seg)
        grps.append(i)
    return segs, np.array(grps)


def _find_clusters_and_stats(segs, grps, r_cutoff: float = 8.0):
    """
    Union-find style cluster detection using kd-tree contact search.

    Returns
    -------
    clusters       : list of lists — each inner list holds segment indices
    monomer        : int  — number of isolated segments
    n_clusters     : int  — number of multi-segment clusters
    max_cluster_size : int
    grps           : updated group-ID array
    """
    nmol = len(segs)
    grps = grps.copy()

    for i in range(nmol - 1):
        tree_i = cKDTree(segs[i].positions)
        grp_i  = grps[i]
        for j in range(i + 1, nmol):
            grp_j = grps[j]
            if grp_i == grp_j:
                continue
            neighbors = tree_i.query_ball_tree(cKDTree(segs[j].positions), r=r_cutoff)
            if any(len(nb) > 0 for nb in neighbors):
                grps = np.where(grps == grp_j, grp_i, grps)
                grp_i = grps[i]   # may have been relabelled

    aggr = {}
    for g in grps:
        aggr[g] = aggr.get(g, 0) + 1

    monomer          = sum(1 for v in aggr.values() if v == 1)
    n_clusters       = len(aggr) - monomer
    max_cluster_size = max(aggr.values())

    clusters: dict = {}
    for seg_idx, cid in enumerate(grps):
        clusters.setdefault(cid, []).append(seg_idx)

    return list(clusters.values()), monomer, n_clusters, max_cluster_size, grps


def _unwrap_cluster(u, cluster_seg_indices: list, box: np.ndarray) -> dict:
    """
    BFS unwrap of a cluster split across periodic boundaries.
    Returns {seg_idx: shift_vector} for every segment in the cluster.
    """
    box_half = box[:3] / 2.0
    shifts   = {cluster_seg_indices[0]: np.zeros(3)}
    queue    = deque([cluster_seg_indices[0]])
    visited  = {cluster_seg_indices[0]}

    while queue:
        ref_idx  = queue.popleft()
        ref_com  = (u.segments[ref_idx].atoms.positions + shifts[ref_idx]).mean(axis=0)

        for seg_idx in cluster_seg_indices:
            if seg_idx in visited:
                continue
            seg_com = u.segments[seg_idx].atoms.positions.mean(axis=0)
            delta   = seg_com - ref_com
            shift   = np.zeros(3)
            for dim in range(3):
                if   delta[dim] >  box_half[dim]:
                    shift[dim] = -box[dim]
                elif delta[dim] < -box_half[dim]:
                    shift[dim] =  box[dim]
            shifts[seg_idx] = shifts[ref_idx] + shift
            visited.add(seg_idx)
            queue.append(seg_idx)

    return shifts


def _recenter_frame(u, largest_cluster: list, box: np.ndarray) -> np.ndarray:
    """
    Unwrap the largest cluster, translate its CoM to the box centre,
    then wrap every atom back into the primary cell.

    Returns a new position array for u.atoms (does not mutate u in-place).
    """
    box_center   = box[:3] / 2.0
    shifts       = _unwrap_cluster(u, largest_cluster, box)
    new_pos      = u.atoms.positions.copy()

    for seg_idx in largest_cluster:
        idx = u.segments[seg_idx].atoms.indices
        new_pos[idx] += shifts[seg_idx]

    cluster_indices = [
        i for seg_idx in largest_cluster
        for i in u.segments[seg_idx].atoms.indices
    ]
    center_shift  = box_center - new_pos[cluster_indices].mean(axis=0)
    new_pos      += center_shift

    for dim in range(3):
        new_pos[:, dim] %= box[dim]

    return new_pos


def _radial_density(u, droplet_center: np.ndarray, ca_per_segment: int,
                    ref_atom: str = "CA",
                    r_max: float = 100.0, dr: float = 1.0) -> tuple:
    """
    Radial monomer concentration profile (mM) centred on *droplet_center*.

    Returns
    -------
    r_bins          : ndarray (n_bins,)  — bin centres in Å
    concentration   : ndarray (n_bins,)  — concentration in mM
    """
    all_ca    = u.select_atoms(f"name {ref_atom}")
    distances = np.linalg.norm(all_ca.positions - droplet_center, axis=1)

    n_bins = int(r_max / dr)
    bins   = np.linspace(0, r_max, n_bins + 1)
    r_bins = (bins[:-1] + bins[1:]) / 2.0

    ca_counts, _    = np.histogram(distances, bins=bins)
    monomer_counts  = ca_counts / ca_per_segment
    shell_vols      = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    concentration   = (monomer_counts / shell_vols) * (1e30 / AVOGADRO)

    return r_bins, concentration


@app.command("aggr")
def cmd_aggr(
    top: Annotated[str,  typer.Option("--top",  help="Topology file (PSF)")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj", help="Trajectory file")] = "system.xtc",
    out: Annotated[str,  typer.Option("--out",  help="Aggregation statistics output")] = "aggr.dat",
    outtraj: Annotated[str, typer.Option("--outtraj", help="Recentered trajectory output")] = "recentered.xtc",
    profile: Annotated[str, typer.Option("--profile", help="Radial density profile output")] = "density_profile.dat",
    ref: Annotated[str,   typer.Option("--ref",    help="Reference atom name (CA for peptide, P for RNA)")] = "CA",
    rcut: Annotated[float, typer.Option("--rcut",  help="Clustering distance cutoff (Å)")] = 8.0,
    castep: Annotated[int,  typer.Option("--castep", help="Use every Nth ref atom for clustering")] = 1,
    start: Annotated[Optional[int], typer.Option("--start", help="First frame index")] = None,
    stop:  Annotated[Optional[int], typer.Option("--stop",  help="Last frame index (exclusive)")] = None,
    stride: Annotated[int,  typer.Option("--stride")] = 1,
    recenter: Annotated[bool, typer.Option("--recenter/--no-recenter",
                help="Unwrap & recenter the largest cluster each frame")] = False,
    density: Annotated[bool, typer.Option("--density/--no-density",
                help="Compute radial monomer concentration profile (requires --recenter)")] = False,
    dr: Annotated[float, typer.Option("--dr", help="Radial bin width for density profile (Å)")] = 2.0,
    n_frames_avg: Annotated[int, typer.Option("--n-frames-avg",
                  help="Number of last frames averaged for the density profile")] = 50,
):
    """
    Analyse protein **aggregation**: cluster detection, optional PBC recentering,
    and radial monomer concentration profile.

    Clusters are identified by kd-tree contact search on the chosen reference
    atoms (default: Cα).  Each frame produces counts of monomers, multi-chain
    clusters, and the largest cluster size.

    With **--recenter** the largest cluster is unwrapped across PBC and
    translated to the box centre; the result is written as a new trajectory.

    With **--density** (requires **--recenter**) the radial monomer
    concentration (mM) is averaged over the last *n-frames-avg* frames and
    written to *profile*.
    """
    import MDAnalysis as mda

    if density and not recenter:
        typer.echo(
            "[!] --density requires --recenter — density calculation disabled.",
            err=True,
        )
        density = False

    typer.echo("Loading trajectory...")
    u = _load_universe(top, traj)

    segs, grps_init = _grp_init(u, ref_atom=ref, step=castep)
    ca_per_segment  = len(u.segments[0].atoms.select_atoms(f"name {ref}"))

    n_frames = len(u.trajectory[start:stop:stride])
    typer.echo(f"[i] Segments      : {len(segs)}")
    typer.echo(f"[i] {ref} / segment  : {ca_per_segment}")
    typer.echo(f"[i] Frames        : {n_frames}")
    typer.echo(f"[i] Cutoff        : {rcut} Å")
    typer.echo(f"[i] Recenter      : {recenter}")
    typer.echo(f"[i] Density       : {density}")

    stats_data: list       = []
    density_profiles: list = []

    writer = mda.Writer(outtraj, n_atoms=u.atoms.n_atoms) if recenter else None

    try:
        for ts_idx, ts in enumerate(u.trajectory[start:stop:stride]):
            if ts_idx % 10 == 0:
                typer.echo(
                    f"  Frame {ts_idx + 1}/{n_frames}  (traj frame {ts.frame})"
                )

            box = ts.dimensions

            for seg in segs:
                seg.wrap()

            clusters, monomer, n_clusters, max_size, _ = _find_clusters_and_stats(
                segs, grps_init, r_cutoff=rcut
            )

            if ts_idx % 10 == 0:
                typer.echo(
                    f"    monomers={monomer}  clusters={n_clusters}  "
                    f"largest={max_size}"
                )

            if recenter:
                largest = max(clusters, key=len)
                u.atoms.positions = _recenter_frame(u, largest, box)
                writer.write(u.atoms)

                if density and ts_idx >= n_frames - n_frames_avg:
                    r_max = float(np.min(box[:3]) / 2.0)
                    r_bins, conc = _radial_density(
                        u,
                        droplet_center=box[:3] / 2.0,
                        ca_per_segment=ca_per_segment,
                        ref_atom=ref,
                        r_max=r_max,
                        dr=dr,
                    )
                    density_profiles.append(conc)

            stats_data.append((ts.frame, monomer, n_clusters, max_size))

    finally:
        if writer is not None:
            writer.close()

    # ── Write aggregation statistics ────────────────────────────────────────
    typer.echo(f"\nWriting aggregation statistics -> {out}")
    with open(out, "w") as fh:
        fh.write("# Frame  Monomers  Clusters  LargestClusterSize\n")
        for frame, monomer, n_clusters, max_size in stats_data:
            fh.write(f"{frame}  {monomer}  {n_clusters}  {max_size}\n")

    # ── Write density profile ────────────────────────────────────────────────
    if density and density_profiles:
        avg_conc = np.mean(density_profiles, axis=0)
        std_conc = np.std(density_profiles,  axis=0)
        typer.echo(
            f"Writing radial density profile ({len(density_profiles)} frames)"
            f" -> {profile}"
        )
        with open(profile, "w") as fh:
            fh.write(
                f"# Radial monomer concentration averaged over last "
                f"{len(density_profiles)} frames\n"
            )
            fh.write("# Radius(A)  Concentration(mM)  StdDev(mM)\n")
            for r, c, s in zip(r_bins, avg_conc, std_conc):
                fh.write(f"{r:.3f}  {c:.6f}  {s:.6f}\n")

    if recenter:
        typer.echo(f"Recentered trajectory -> {outtraj}")
    typer.echo("DONE!")


# =============================================================================
#  ░░  Entry-point  ░░
# =============================================================================

if __name__ == "__main__":
    app()