#!/usr/bin/env python3
"""
Analysis Module
===============
Unified MD trajectory analysis toolkit for the SciKit package.

All eight analyses are registered as sub-commands of the ``scical`` CLI
entry-point (powered by `Typer <https://typer.tiangolo.com>`_).  Each command
reads a topology and (optionally) a trajectory file, runs the analysis in
parallel where supported, and writes results to plain-text ``.dat`` or NumPy
``.npy`` files.

Get started — list all commands and options::

    scical --help

Per-command help::

    scical msd --help
    scical rg --help
    scical dssp --help
    scical distance --help
    scical distance-acf --help
    scical vector-acf --help
    scical contacts --help
    scical aggr --help
    scical time-s2 --help

Available commands
------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Command
     - Description
   * - ``msd``
     - Per-segment Cα mean squared displacement (FFT, parallel)
   * - ``rg``
     - Per-segment radius of gyration time series
   * - ``dssp``
     - Per-residue DSSP helicity and β-sheet content
   * - ``distance``
     - Cα–Cα distances for user-defined residue pairs over a trajectory
   * - ``distance-acf``
     - Normalised fluctuation ACF of inter-Cα distances
   * - ``vector-acf``
     - End-to-end Cα vector autocorrelation function
   * - ``contacts``
     - Intra- and inter-chain heavy-atom contact maps (parallel)
   * - ``aggr``
     - Aggregation analysis: clustering, PBC recentering, radial density
   * - ``time-s2``
     - Time-dependent S² order parameter

Dependencies
------------
mdanalysis, typer, numpy, scipy
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings
import tempfile
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Annotated

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
    """Convert the CLI ``stop=-1`` sentinel to ``None`` for MDAnalysis slicing.

    MDAnalysis trajectory slices use ``None`` to mean "read to the last frame".
    The CLI exposes ``-1`` as the user-facing sentinel because ``Optional[int]``
    defaults require a concrete value.

    Args:
        stop (int): Frame index supplied by the user, or ``-1`` to mean "end".

    Returns:
        Optional[int]: ``None`` if *stop* is ``-1``, otherwise *stop* unchanged.
    """
    return None if stop == -1 else stop


# =============================================================================
#  ░░  1.  MSD  ░░
# =============================================================================

def _msd_worker(args: tuple):
    """Subprocess worker: compute EinsteinMSD for one chain segment.

    Intended to be executed inside a ``ProcessPoolExecutor``.  Imports
    MDAnalysis locally so that the module can be imported without the package
    being installed in the parent process.

    Args:
        args (tuple): A packed argument tuple containing:

            - **segid** (*str*) — Segment identifier (e.g. ``"PROA"``).
            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **resid_range** (*Optional[str]*) — Residue range string
              (e.g. ``"1:38"``), or ``None`` for the full segment.
            - **start** (*int*) — Index of the first frame to analyse.
            - **stop** (*int*) — Index of the last frame (``-1`` = end).
            - **stride** (*int*) — Step between analysed frames.
            - **outdir** (*str*) — Directory where output files are written.
            - **per_residue** (*bool*) — Whether to write per-Cα MSD files.
            - **max_tau** (*Optional[float]*) — Maximum lag time in ps;
              ``None`` keeps all lag times.

    Returns:
        str: The *segid* that was processed, for logging in the parent process.

    Raises:
        ValueError: If no atoms match the constructed selection string.

    Output files:
        - ``<outdir>/<segid>_msd.dat`` — Two-column file: lag time (ps) and
          total MSD (Å²).
        - ``<outdir>/<segid>_CA_msd.dat`` — Per-Cα MSD file (only when
          *per_residue* is ``True``).
    """
    import MDAnalysis as mda
    import MDAnalysis.analysis.msd as msd_mod

    segid, topology, trajectory, resid_range, \
        start, stop, stride, outdir, per_residue, max_tau = args

    u = mda.Universe(topology, trajectory)

    selection = (f"name CA and segid {segid} and resid {resid_range}"
                 if resid_range else f"name CA and segid {segid}")

    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"No atoms matched: '{selection}'")

    effective_stop = _traj_stop(stop)

    MSD = msd_mod.EinsteinMSD(u, select=selection, msd_type="xyz", fft=True)
    MSD.run(start=start, stop=effective_stop, step=stride)

    lagtimes = MSD.results.delta_t_values
    msds     = MSD.results.timeseries

    if max_tau is not None:
        mask    = lagtimes <= max_tau
        lagtimes = lagtimes[mask]
        msds     = msds[mask]

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
        if max_tau is not None:
            msds_by_res = msds_by_res[mask]  
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
    max_tau: Annotated[Optional[float], typer.Option("--max-tau",
                                                      help="Maximum lag time (ps); "
                                                           "limits frames loaded, not just output")] = None,
    nproc: Annotated[int, typer.Option("--nproc", help="Parallel workers")] = 4,
):
    """Calculate per-segment Cα **mean squared displacement** (FFT method).

    Iterates over all segments in the topology that contain matching Cα atoms
    and dispatches each segment to a subprocess worker.  The MSD is computed
    via the Einstein relation using MDAnalysis' ``EinsteinMSD`` with FFT
    acceleration.

    Args:
        top (str): Path to the topology file (PSF, PDB, GRO, …).
        traj (str): Path to the trajectory file (XTC, DCD, …).
        resid (Optional[str]): Residue range to restrict the Cα selection,
            e.g. ``"1:38"``.  When omitted the full segment is used.
        outdir (str): Output directory; created automatically if absent.
        start (int): Index of the first trajectory frame to include.
        stop (int): Index of the last trajectory frame; ``-1`` means end.
        stride (int): Step between analysed frames.
        per_residue (bool): When ``True``, also write per-Cα MSD files.
        max_tau (Optional[float]): Truncate the analysis at this lag time
            (ps) by limiting the number of frames passed to ``EinsteinMSD``.
            The maximum achievable lag equals ``(n_frames - 1) × dt``, so
            only ``ceil(max_tau / dt) + 1`` frames are ever read from disk.
            ``None`` (default) uses all available frames.
        nproc (int): Number of parallel worker processes.

    Output:
        ``<outdir>/<segid>_msd.dat`` — two columns: lag time (ps) and MSD (Å²).
        ``<outdir>/<segid>_CA_msd.dat`` — per-Cα MSD table (``--per-residue`` only).

    Example::

            scical msd --top conf.psf --traj system.xtc --resid 1:50 \\
                       --max-tau 5000 --outdir ./msd_results --nproc 8
    """
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
    typer.echo(f"[i] Max tau     : {f'{max_tau} ps' if max_tau is not None else 'all'}")
    typer.echo(f"[i] Output dir  : {outdir}")
    del u

    args_list = [
        (segid, top, traj, resid, start, stop, stride, outdir, per_residue, max_tau)
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
    """Subprocess worker: compute radius-of-gyration time series for one segment.

    Selects protein atoms (and optionally a residue sub-range) in the given
    segment and evaluates ``AtomGroup.radius_of_gyration()`` frame by frame.

    Args:
        args (tuple): A packed argument tuple containing:

            - **segid** (*str*) — Segment identifier.
            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **resid_range** (*Optional[str]*) — Residue range string, or
              ``None`` for the full protein segment.
            - **start** (*int*) — First frame index.
            - **stop** (*int*) — Last frame index (``-1`` = end).
            - **stride** (*int*) — Frame stride.

    Returns:
        tuple[str, Optional[np.ndarray]]: A ``(segid, rg_array)`` pair, where
        *rg_array* has shape ``(n_frames,)`` in ångström units.
        Returns ``(segid, None)`` if no atoms matched or the trajectory slice
        is empty.
    """
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
    """Calculate per-segment **radius of gyration** time series.

    Computes Rg(t) for every protein-containing segment in parallel and writes
    a single multi-column ``.dat`` file with one column per segment, indexed
    by frame number.

    Args:
        top (str): Path to the topology file.
        traj (str): Path to the trajectory file.
        resid (Optional[str]): Residue range to restrict the selection
            (e.g. ``"1:100"``).  Omit to use the full protein.
        out (str): Path for the output ``.dat`` file.
        start (int): Index of the first trajectory frame.
        stop (int): Index of the last trajectory frame; ``-1`` means end.
        stride (int): Step between analysed frames.
        nproc (int): Number of parallel worker processes.

    Output:
        ``<out>`` — space-separated columns: ``frame  <segid1>  <segid2>  …``
        Units: frame index (dimensionless), Rg in ångström.

    Example::

            scical rg --top conf.psf --traj system.xtc --out rg.dat --stride 5 --nproc 4
    """
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
    """Subprocess worker: run DSSP secondary-structure assignment for one segment.

    Uses ``MDAnalysis.analysis.dssp.DSSP`` to assign secondary structure
    codes (``'H'`` = α-helix, ``'E'`` = β-strand, ``'C'`` = coil) to each
    residue for every frame, then computes the per-residue helicity and
    β-sheet occupancy as fractions over the analysed trajectory window.

    Args:
        args (tuple): A packed argument tuple containing:

            - **segid** (*str*) — Segment identifier.
            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **start** (*int*) — First frame index.
            - **stop** (*int*) — Last frame index (``-1`` = end).
            - **stride** (*int*) — Frame stride.

    Returns:
        tuple: ``(segid, resids, helicity, beta)`` where:

        - **segid** (*str*) — The processed segment identifier.
        - **resids** (*np.ndarray | None*) — 1-D array of residue IDs, or
          ``None`` if no protein residues were found or no frames were analysed.
        - **helicity** (*np.ndarray | None*) — Per-residue helix fraction in
          ``[0, 1]``, shape ``(n_residues,)``.
        - **beta** (*np.ndarray | None*) — Per-residue β-strand fraction in
          ``[0, 1]``, shape ``(n_residues,)``.
    """
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
    """Write a DSSP occupancy table to a space-separated ``.dat`` file.

    Constructs a matrix with rows = residues (union across all segments) and
    columns = segments.  Missing residues for a given segment are filled with
    ``NaN``.

    Args:
        output_file (str): Path to the output file.
        all_resids (list[int]): Complete set of residue IDs (across all
            segments) to include as rows.
        values_by_seg (list[tuple]): List of ``(resids, values)`` tuples, one
            per segment, where *resids* and *values* are 1-D arrays of the
            same length.
        segids (list[str]): Ordered list of segment identifiers used as column
            headers.
    """
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
    """Calculate per-residue **DSSP** helicity and beta-sheet content.

    Assigns DSSP secondary-structure codes to every protein residue in every
    analysed frame using MDAnalysis' built-in DSSP implementation.
    Per-residue fractions are averaged over time and written to two separate
    files — one for helicity (``'H'`` codes) and one for β-sheet content
    (``'E'`` codes).

    Args:
        top (str): Path to the topology file.
        traj (str): Path to the trajectory file.
        hout (str): Output path for the per-residue helicity table.
        bout (str): Output path for the per-residue β-sheet table.
        start (int): Index of the first trajectory frame.
        stop (int): Index of the last trajectory frame; ``-1`` means end.
        stride (int): Step between analysed frames.
        nproc (int): Number of parallel worker processes.

    Output:
        ``<hout>`` and ``<bout>`` — space-separated tables:
        ``residue  <segid1>  <segid2>  …``
        Values are per-residue helix / β-strand occupancy fractions in [0, 1].

    Example::

            scical dssp --top conf.psf --traj system.xtc --hout helicity.dat --bout beta.dat --stride 10 --nproc 4
    """
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
    """Print a warning message to stderr.

    Args:
        msg (str): Warning text to display (prefixed with ``[WARNING]``).
    """
    print(f"[WARNING] {msg}", file=sys.stderr)


def _load_pairs(filepath: str):
    """Parse a residue-pair list file into a list of ``(resid1, segid1, resid2, segid2)`` tuples.

    Each non-comment, non-blank line must contain exactly four fields
    (comma- or whitespace-separated):
    ``resid1  segid1  resid2  segid2``

    Malformed lines emit a warning and are skipped; they do not raise.

    Args:
        filepath (str): Path to the pairs file.

    Returns:
        list[tuple[int, str, int, str]]: Parsed pairs in the order they appear
        in the file.
    """
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
    """Load and deduplicate residue pairs from one or more pair files.

    Pairs that appear in more than one file are included only once in the
    returned *all_pairs* list; the original per-file lists are preserved in
    *file_pairs* for downstream per-file output.

    Args:
        filepaths (list[str]): Ordered list of pair-file paths to read.

    Returns:
        tuple[list, dict]:

        - **all_pairs** — Deduplicated list of
          ``(resid1, segid1, resid2, segid2)`` tuples.
        - **file_pairs** — ``{filepath: [pairs]}`` mapping preserving the
          original per-file pair lists (with duplicates).
    """
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
    """Clamp and validate a trajectory frame range.

    Args:
        n_total (int): Total number of frames in the trajectory.
        start (Optional[int]): Requested start frame, or ``None`` for frame 0.
        stop (Optional[int]): Requested stop frame (exclusive), or ``None``
            for the last frame.
        stride (int): Step between frames; clamped to at least 1.

    Returns:
        tuple[int, int, int]: The validated ``(start, stop, stride)`` triple.

    Raises:
        ValueError: If the resolved *start* is greater than or equal to *stop*.
    """
    start  = max(0, start if start is not None else 0)
    stop   = min(n_total, stop if stop is not None else n_total)
    stride = max(1, stride)
    if start >= stop:
        raise ValueError(f"start={start} >= stop={stop}; no frames selected.")
    return start, stop, stride


def _n_selected_frames(start, stop, stride):
    """Return the number of frames in the slice ``[start:stop:stride]``.

    Args:
        start (int): First frame index (inclusive).
        stop (int): Last frame index (exclusive).
        stride (int): Step between frames.

    Returns:
        int: Number of frames selected; ``0`` if the range is empty.
    """
    return max(0, (stop - start + stride - 1) // stride)


def _make_worker_slices(start, stop, stride, n_workers):
    """Partition a frame range into roughly equal sub-slices for parallel workers.

    Args:
        start (int): First frame index (inclusive).
        stop (int): Last frame index (exclusive).
        stride (int): Step between frames.
        n_workers (int): Desired number of worker processes; clamped to the
            actual number of selected frames if fewer frames are available.

    Returns:
        list[tuple[int, int, int]]: List of ``(w_start, w_stop, stride)``
        tuples, one per worker.  Empty chunks are omitted.
    """
    sel      = np.arange(start, stop, stride)
    n_actual = min(n_workers, len(sel))
    slices   = []
    for chunk in np.array_split(sel, n_actual):
        if len(chunk) == 0:
            continue
        slices.append((int(chunk[0]), int(chunk[-1]) + stride, stride))
    return slices


def _resolve_ca_indices(universe, pairs):
    """Resolve Cα atom indices for a list of residue pairs.

    For each unique ``(resid, segid)`` key encountered in *pairs*, selects
    the corresponding Cα atom from *universe* and records its global atom
    index.  The result is two parallel lists of indices that can be passed
    directly to NumPy distance calculations.

    Args:
        universe (MDAnalysis.Universe): Loaded MDAnalysis Universe.
        pairs (list[tuple[int, str, int, str]]): List of
            ``(resid1, segid1, resid2, segid2)`` tuples.

    Returns:
        tuple[list[int], list[int]]: Two parallel lists — *idx1* and *idx2* —
        each of length ``len(pairs)``, containing the global atom indices of
        the first and second Cα atoms in each pair.

    Raises:
        ValueError: If a ``(resid, segid)`` key matches no Cα atom.
    """
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
    """Subprocess worker: compute pairwise Cα distances over a trajectory sub-slice.

    Loads its own MDAnalysis Universe and iterates over the assigned frame
    window, accumulating distances into a pre-allocated buffer for efficiency.

    Args:
        args (tuple): A packed argument tuple containing:

            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **idx1** (*list[int]*) — Global atom indices for the first atoms
              of each pair.
            - **idx2** (*list[int]*) — Global atom indices for the second atoms
              of each pair.
            - **traj_slice** (*tuple[int, int, int]*) — ``(start, stop, stride)``
              for this worker's frame window.
            - **worker_id** (*int*) — Zero-based worker index for logging.

    Returns:
        tuple[list[int], np.ndarray]: ``(frames, distances)`` where *frames*
        is a list of absolute trajectory frame numbers and *distances* has
        shape ``(n_pairs, n_frames)`` with values in ångström.
    """
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
    """Write a pairwise distance matrix to a CSV-style ``.dat`` file.

    The output has one row per pair and one column per trajectory frame.
    The header line lists each frame number.  The first four columns identify
    the pair: ``resid_1, segid_1, resid_2, segid_2``.

    Args:
        out_path (str): Destination file path.
        pairs (list[tuple]): List of ``(resid1, segid1, resid2, segid2)``
            tuples — one per row.
        dist_block (np.ndarray): Distance matrix of shape
            ``(n_pairs, n_frames)`` in ångström.
        all_frames (list[int]): Ordered list of trajectory frame numbers
            corresponding to the columns of *dist_block*.
    """
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
    """Calculate **Cα–Cα distances** for residue pairs over a trajectory.

    Reads one or more pair-list files, resolves the corresponding Cα atom
    indices, and distributes the trajectory frames across parallel workers.
    Results are assembled and written to a ``_distance.dat`` file placed
    alongside each input pair file.

    **Pair file format** (comma- or whitespace-separated, ``#`` comments allowed):

        resid1  segid1  resid2  segid2
        10      PROA    45      PROA
        10      PROA    45      PROB

    Args:
        top (str): Path to the topology file.
        traj (str): Path to the trajectory file.
        pair_files (List[str]): One or more residue-pair list files.
            Duplicate pairs across files are computed only once.
        start (Optional[int]): First trajectory frame to include
            (``None`` = frame 0).
        stop (Optional[int]): Last trajectory frame, exclusive
            (``None`` = last frame).
        stride (int): Step between analysed frames.
        workers (int): Number of parallel worker processes; defaults to the
            number of logical CPU cores.

    Output:
        ``<pair_file_stem>_distance.dat`` — one row per pair, one column per
        frame, prefixed by ``resid_1, segid_1, resid_2, segid_2``.

    Example::

            scical distance --top conf.psf --traj system.dcd -f pairs.dat --stride 2 --workers 8
    """
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
    """Compute the normalised fluctuation autocorrelation function of a scalar time series.

    Uses the Wiener–Khinchin theorem (FFT-based) with bias correction for
    the varying number of available pairs at each lag.  The zero-lag value
    is normalised to 1.

    Args:
        x (np.ndarray): 1-D array of scalar observations, shape ``(N,)``.

    Returns:
        np.ndarray: Normalised ACF, shape ``(N,)``, with ``result[0] == 1.0``.
    """
    N   = len(x)
    xm  = x - x.mean()
    fx  = np.fft.fft(xm, n=2 * N)
    pwr = np.real(np.fft.ifft(fx * np.conj(fx)))[:N]
    pwr /= np.arange(N, 0, -1)
    return pwr / pwr[0]


def _dist_acf_worker(args: tuple):
    """Subprocess worker: compute the distance ACF for one residue pair.

    Extracts the Cα–Cα distance time series for the given pair and applies
    :func:`_autocorr_fft_scalar` to obtain the normalised fluctuation ACF.

    Args:
        args (tuple): A packed argument tuple containing:

            - **label** (*str*) — Human-readable pair label
              (``"segid1:resid1-segid2:resid2"``).
            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **resid1** (*str*) — Residue ID of the first atom.
            - **segid1** (*str*) — Segment ID of the first atom.
            - **resid2** (*str*) — Residue ID of the second atom.
            - **segid2** (*str*) — Segment ID of the second atom.
            - **start** (*int*) — First frame index.
            - **stop** (*int*) — Last frame index (``-1`` = end).
            - **stride** (*int*) — Frame stride.

    Returns:
        tuple[str, Optional[np.ndarray]]: ``(label, acf)`` where *acf* is a
        1-D normalised ACF array, or ``None`` if atoms were not found or
        fewer than 2 frames were available.
    """
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
    """Read a residue-pair file and return a list of ``(resid1, segid1, resid2, segid2)`` tuples.

    Lines beginning with ``#`` and blank lines are skipped.  Fields may be
    comma- or whitespace-separated; only the first four tokens of each line
    are used.

    Args:
        path (str): Path to the pairs file.

    Returns:
        list[tuple[str, str, str, str]]: Parsed pairs as strings (resids are
        **not** cast to ``int`` here — the caller is responsible if needed).

    Raises:
        ValueError: If a data line contains fewer than four fields.
    """
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
    """Construct a compact human-readable label for a residue pair.

    Args:
        r1: Residue ID of the first residue.
        s1: Segment ID of the first residue.
        r2: Residue ID of the second residue.
        s2: Segment ID of the second residue.

    Returns:
        str: Label formatted as ``"<s1>:<r1>-<s2>:<r2>"``.
    """
    return f"{s1}:{r1}-{s2}:{r2}"


def _save_acf(output_file, labels, C_all, topology, trajectory, stride):
    """Write a collection of ACF curves to a space-separated ``.dat`` file.

    All ACF arrays are truncated to the length of the shortest one before
    writing.  The first column contains lag times in picoseconds (computed
    from the trajectory time step and *stride*); subsequent columns contain
    the ACF for each labelled pair.

    Args:
        output_file (str): Destination file path.
        labels (list[str]): Column labels (one per ACF curve).
        C_all (list[np.ndarray]): List of 1-D normalised ACF arrays.
        topology (str): Path to the topology file (used to read ``dt``).
        trajectory (str): Path to the trajectory file.
        stride (int): Frame stride used when computing the ACF (scales ``dt``).
    """
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
    """Compute inter-Cα **distance autocorrelation** for residue pairs.

    For each pair in *pairs*, extracts the Cα–Cα distance time series and
    computes the normalised fluctuation ACF via FFT (Wiener–Khinchin theorem).
    All ACF curves are written to a single ``.dat`` file.

    Args:
        top (str): Path to the topology file.
        traj (str): Path to the trajectory file.
        pairs (str): Path to the residue-pair list file.
        out (str): Output file path for the ACF table.
        start (int): First trajectory frame to include.
        stop (int): Last trajectory frame; ``-1`` means end.
        stride (int): Step between analysed frames (also scales the lag-time axis).
        nproc (int): Number of parallel worker processes.

    Output:
        ``<out>`` — columns: lag time (ps), normalised ACF per pair.
        The zero-lag value is normalised to 1.

    Example::

            scical distance-acf --top conf.psf --traj system.xtc --pairs pairs.dat --out distance_acf.dat --stride 10 --nproc 4
    """
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
    """Compute the un-normalised, bias-corrected autocorrelation of a 1-D array.

    Uses FFT via the Wiener–Khinchin theorem.  No mean subtraction is applied;
    use :func:`_autocorr_fft_scalar` for fluctuation ACFs.

    Args:
        x (np.ndarray): 1-D input signal, shape ``(N,)``.

    Returns:
        np.ndarray: Bias-corrected ACF, shape ``(N,)``.  Element ``[0]`` is
        the mean square of *x*.
    """
    N   = len(x)
    fx  = np.fft.fft(x, n=2 * N)
    pwr = np.real(np.fft.ifft(fx * np.conj(fx)))[:N]
    pwr /= np.arange(N, 0, -1)
    return pwr


def _autocorr_fft_vector(R: np.ndarray) -> np.ndarray:
    """Compute the normalised vector autocorrelation function.

    Evaluates

    .. math::

        C(t) = \\frac{\\langle R_x(0)R_x(t) \\rangle
                     + \\langle R_y(0)R_y(t) \\rangle
                     + \\langle R_z(0)R_z(t) \\rangle}
                     {\\langle |\\mathbf{R}|^2 \\rangle}

    where :math:`\\mathbf{R}(t)` is the end-to-end inter-Cα vector at time *t*.

    Args:
        R (np.ndarray): 2-D array of inter-residue vectors, shape
            ``(n_frames, 3)``, in ångström.

    Returns:
        np.ndarray: Normalised vector ACF, shape ``(n_frames,)``, with
        ``result[0] == 1.0``.
    """
    C = _raw_acf_1d(R[:, 0]) + _raw_acf_1d(R[:, 1]) + _raw_acf_1d(R[:, 2])
    return C / C[0]


def _vec_acf_worker(args: tuple):
    """Subprocess worker: compute the vector ACF for one residue pair.

    Collects the inter-Cα displacement vector :math:`\\mathbf{R}(t)` over the
    specified trajectory window and calls :func:`_autocorr_fft_vector`.

    Args:
        args (tuple): A packed argument tuple containing:

            - **label** (*str*) — Human-readable pair label.
            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **resid1** (*str*) — Residue ID of the first atom.
            - **segid1** (*str*) — Segment ID of the first atom.
            - **resid2** (*str*) — Residue ID of the second atom.
            - **segid2** (*str*) — Segment ID of the second atom.
            - **start** (*int*) — First frame index.
            - **stop** (*int*) — Last frame index (``-1`` = end).
            - **stride** (*int*) — Frame stride.

    Returns:
        tuple[str, Optional[np.ndarray]]: ``(label, acf)`` where *acf* is a
        1-D normalised vector ACF, or ``None`` if atoms were not found or
        fewer than 2 frames were available.
    """
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
    """Compute end-to-end Cα **vector autocorrelation** for residue pairs.

    For each pair in *pairs*, collects the inter-Cα displacement vector
    **R**(t) and computes the normalised vector ACF:

    .. math::

        C(t) = \\frac{\\langle \\mathbf{R}(0) \\cdot \\mathbf{R}(t) \\rangle}
                     {\\langle |\\mathbf{R}|^2 \\rangle}

    All ACF curves are written to a single ``.dat`` file.

    Args:
        top (str): Path to the topology file.
        traj (str): Path to the trajectory file.
        pairs (str): Path to the residue-pair list file.
        out (str): Output file path for the ACF table.
        start (int): First trajectory frame to include.
        stop (int): Last trajectory frame; ``-1`` means end.
        stride (int): Step between analysed frames.
        nproc (int): Number of parallel worker processes.

    Output:
        ``<out>`` — columns: lag time (ps), normalised vector ACF per pair.

    Example::

            scical vector-acf --top conf.psf --traj system.xtc --pairs pairs.dat --out vector_acf.dat --stride 10 --nproc 4
    """
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
"""str: MDAnalysis selection string for common heavy side-chain atoms."""


def _parse_components(spec, segids):
    """Parse a component specification string into a mapping of label → segid list.

    Segids are taken from *segids* in order and grouped according to the counts
    in *spec*.  If *spec* is ``None``, all segids are placed in a single
    component labelled ``"A"``.

    Args:
        spec (Optional[str]): Component spec, e.g. ``"A:100 B:100 C:100"``.
        segids (list[str]): Ordered list of segids detected from the topology.

    Returns:
        dict[str, list[str]]: Mapping ``{label: [segid, ...]}``.

    Raises:
        ValueError: If the total count in *spec* does not match ``len(segids)``.
    """
    if spec is None:
        return {"A": segids}
    comp_map = {}
    idx = 0
    for token in spec.split():
        label, count = token.split(":")
        count = int(count)
        comp_map[label] = segids[idx: idx + count]
        idx += count
    if idx != len(segids):
        raise ValueError(
            f"Component spec accounts for {idx} segids but {len(segids)} found."
        )
    return comp_map


def _parse_pairs(spec, comp_labels):
    """Parse a pair specification string into a list of ``(labelX, labelY)`` tuples.

    Args:
        spec (str): Space-separated pair tokens, e.g. ``"A-A A-B"``.
        comp_labels (list[str]): Known component labels.

    Returns:
        list[tuple[str, str]]: Ordered list of requested component pairs.

    Raises:
        ValueError: If a token references an unknown component label.
    """
    pairs = []
    for token in spec.split():
        x, y = token.split("-")
        if x not in comp_labels or y not in comp_labels:
            raise ValueError(f"Unknown component in pair '{token}'. Known: {comp_labels}")
        pairs.append((x, y))
    return pairs


def _contacts_worker(args):
    """Subprocess worker: accumulate contact matrices for all requested pairs over a frame chunk.

    Uses ``MDAnalysis.lib.distances.capped_distance`` for efficient
    neighbour-list construction.  Supports multi-component systems with
    arbitrary same- and cross-component contact pair requests.

    For same-component pairs the worker accumulates:

    - **intra** — contacts within the same chain copy (sequence separation ≥ 4).
    - **inter** — contacts between different chain copies of the same component.

    For cross-component pairs only **inter** (cross-component) contacts are
    accumulated.

    Args:
        args (tuple): A packed argument tuple containing:

            - **top** (*str*) — Path to the topology file.
            - **traj** (*Optional[str]*) — Path to the trajectory file, or
              ``None`` for a single-frame topology.
            - **cutoff** (*float*) — Distance cutoff in ångström.
            - **frame_indices** (*list[int]*) — Absolute frame indices to
              process in this worker.
            - **atom_to_local** (*np.ndarray[int32]*) — Maps each heavy-atom
              row to its local residue index within its chain copy.
            - **atom_to_copy** (*np.ndarray[int32]*) — Maps each heavy-atom
              row to its copy index within its component.
            - **atom_to_comp** (*np.ndarray[int32]*) — Maps each heavy-atom
              row to its component index.
            - **pair_specs** (*list[tuple]*) — List of
              ``(comp_idx_X, comp_idx_Y, N_X, N_Y, n_copies_X, same_comp)``
              tuples, one per requested pair.

    Returns:
        dict[tuple[int,int], dict[str, Optional[np.ndarray]]]:
        ``{(cx, cy): {"inter": np.ndarray, "intra": np.ndarray or None}}``
        with partial contact counts accumulated over the assigned frames.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    (top, traj, cutoff, frame_indices,
     atom_to_local, atom_to_copy, atom_to_comp,
     pair_specs) = args

    u       = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_hvy = u.select_atoms(HEAVY_ATOMS)

    partials = {}
    for spec in pair_specs:
        cx, cy, N_X, N_Y, _, same = spec
        key = (cx, cy)
        partials[key] = {
            "inter": np.zeros((N_X, N_Y), dtype=np.float64),
            "intra": np.zeros((N_X, N_X), dtype=np.float64) if same else None,
        }

    for frame in frame_indices:
        u.trajectory[frame]
        pos_all = all_hvy.positions
        box     = u.dimensions

        pairs, _ = capped_distance(pos_all, pos_all, cutoff,
                                   box=box, return_distances=True)
        if len(pairs) == 0:
            continue

        ri   = pairs[:, 0].astype(np.int32)
        rj   = pairs[:, 1].astype(np.int32)
        li   = atom_to_local[ri]
        lj   = atom_to_local[rj]
        ci   = atom_to_copy[ri]
        cj   = atom_to_copy[rj]
        cmpi = atom_to_comp[ri]
        cmpj = atom_to_comp[rj]

        for spec in pair_specs:
            cx, cy, N_X, N_Y, _, same = spec
            key = (cx, cy)

            if same:
                # ── Intra (same copy, same component) ────────────────────
                intra_mask = (cmpi == cx) & (cmpj == cx) & (ci == cj) & (np.abs(li - lj) >= 4)
                if intra_mask.any():
                    k    = np.unique(np.stack([ci[intra_mask], li[intra_mask], lj[intra_mask]], axis=1), axis=0)
                    flat = k[:, 1] * N_X + k[:, 2]
                    m    = np.bincount(flat, minlength=N_X * N_X).reshape(N_X, N_X).astype(np.float64)
                    partials[key]["intra"] += m + m.T

                # ── Inter (different copies, same component) ──────────────
                inter_mask = (cmpi == cx) & (cmpj == cx) & (ci != cj)
                if inter_mask.any():
                    k2    = np.unique(np.stack([ci[inter_mask], cj[inter_mask],
                                                li[inter_mask], lj[inter_mask]], axis=1), axis=0)
                    flat2 = k2[:, 2] * N_X + k2[:, 3]
                    m2    = np.bincount(flat2, minlength=N_X * N_Y).reshape(N_X, N_Y).astype(np.float64)
                    partials[key]["inter"] += m2 + m2.T

            else:
                # ── Cross-component contacts (cx → cy and cy → cx) ───────
                mask_xy = (cmpi == cx) & (cmpj == cy)
                mask_yx = (cmpi == cy) & (cmpj == cx)

                li_cx = np.concatenate([li[mask_xy], lj[mask_yx]])
                lj_cy = np.concatenate([lj[mask_xy], li[mask_yx]])
                ci_cx = np.concatenate([ci[mask_xy], cj[mask_yx]])
                cj_cy = np.concatenate([cj[mask_xy], ci[mask_yx]])

                if len(li_cx) > 0:
                    k3    = np.unique(np.stack([ci_cx, cj_cy, li_cx, lj_cy], axis=1), axis=0)
                    flat3 = k3[:, 2] * N_Y + k3[:, 3]
                    m3    = np.bincount(flat3, minlength=N_X * N_Y).reshape(N_X, N_Y).astype(np.float64)
                    partials[key]["inter"] += m3

    return partials


@app.command("contacts")
def cmd_contacts(
    top: Annotated[str, typer.Option("--top", help="Topology file")] = "system.psf",
    traj: Annotated[Optional[str], typer.Option("--traj", help="Trajectory (omit for single frame)")] = None,
    cutoff: Annotated[float, typer.Option("--cutoff", help="Distance cutoff in Å")] = 6.0,
    components: Annotated[Optional[str], typer.Option("--components",
        help="Component definitions e.g. 'A:100 B:100 C:100'. "
             "Default: all segids treated as one component 'A'.")] = None,
    pairs: Annotated[Optional[str], typer.Option("--pairs",
        help="Contact pairs to compute e.g. 'A-A A-B B-B'. "
             "Default: all same-component pairs.")] = None,
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[Optional[int], typer.Option("--stop")] = None,
    stride: Annotated[int, typer.Option("--stride")] = 1,
    nproc: Annotated[int, typer.Option("--nproc")] = 1,
    out: Annotated[str, typer.Option("--out", help="Output stem")] = "contact_map.npy",
):
    """Calculate intra- and inter-chain **contact maps** for multi-component systems (heavy-atom, parallel).

    Identifies contacts between heavy atoms (CA, CB, CC, CD, CE, CF) within
    a user-defined distance cutoff, accounting for periodic boundary conditions.
    Supports systems containing multiple distinct peptide/protein components
    (e.g. 100 copies of chain A + 100 copies of chain B).

    Components are defined by grouping auto-detected segids::

        --components "A:100 B:100 C:100"   # first 100 segids → A, next 100 → B, …

    Contact pairs to compute::

        --pairs "A-A A-B B-B"   # default: all same-component pairs

    Contacts are classified as:

    - **Intra** — same chain copy, sequence separation ≥ 4 residues (same-component pairs only).
    - **Inter** — different chain copies of the same component, or cross-component contacts.

    Normalisation: sum over all copy pairs / (n_copies_X × n_frames), giving the
    average contacts that residue *i* of one copy makes with residue *j* per frame.

    Args:
        top (str): Path to the topology file (PSF).
        traj (Optional[str]): Path to the trajectory file.  When omitted,
            only the single frame in *top* is analysed.
        cutoff (float): Heavy-atom distance cutoff in ångström.
        components (Optional[str]): Component definitions e.g. ``"A:100 B:100"``.
            When omitted all segids are assigned to a single component ``"A"``.
        pairs (Optional[str]): Contact pairs to compute e.g. ``"A-A A-B"``.
            When omitted, all same-component pairs are computed.
        start (int): Index of the first trajectory frame.
        stop (Optional[int]): Index of the last trajectory frame (exclusive);
            ``None`` means use all frames.
        stride (int): Step between analysed frames.
        nproc (int): Number of parallel worker processes.
        out (str): Output file stem.  Per requested pair ``X-Y`` the following
            files are written:

            - ``<stem>_X-Y_intra.npy`` — intra-copy map (same-component only).
            - ``<stem>_X-Y_inter.npy`` — inter-copy map (same-component only).
            - ``<stem>_X-Y_total.npy`` — combined (or cross-component) map.
            - ``<stem>_X-Y_resids.npy`` / ``<stem>_X-Y_resids_X.npy`` / ``_resids_Y.npy``.

    Example::

            # Single-component system (original behaviour)
            scical contacts --top system.psf --traj traj.dcd --cutoff 8.0 --stride 5 --nproc 8

            # Multi-component: 100 A-chains + 100 B-chains, compute A-A and A-B maps
            scical contacts --top system.psf --traj traj.dcd \\
                --components "A:100 B:100" --pairs "A-A A-B" --nproc 8
    """
    import MDAnalysis as mda

    warnings.filterwarnings("ignore")

    u      = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_ca = u.select_atoms("name CA")
    segids = list(dict.fromkeys(all_ca.segids))
    n_traj = len(u.trajectory)

    # ── Component setup ────────────────────────────────────────────────────
    comp_map    = _parse_components(components, segids)
    comp_labels = list(comp_map.keys())

    req_pairs = _parse_pairs(pairs, comp_labels) if pairs else [(x, x) for x in comp_labels]

    comp_ca = {label: [u.select_atoms(f"segid {s} and name CA") for s in segs]
               for label, segs in comp_map.items()}
    comp_N  = {}
    for label, ca_list in comp_ca.items():
        Ns = [len(ca) for ca in ca_list]
        if len(set(Ns)) != 1:
            typer.echo(f"[!] Component {label} has unequal residue counts: {set(Ns)}", err=True)
            raise typer.Exit(1)
        comp_N[label] = Ns[0]

    comp_idx = {label: i for i, label in enumerate(comp_labels)}

    # ── Lookup arrays: heavy atom row → (local_res, copy_within_comp, comp_idx) ──
    all_hvy     = u.select_atoms(HEAVY_ATOMS)
    M           = len(all_hvy)
    all_hvy_idx = all_hvy.indices

    atom_to_local = np.empty(M, dtype=np.int32)
    atom_to_copy  = np.empty(M, dtype=np.int32)
    atom_to_comp  = np.empty(M, dtype=np.int32)

    for label, ca_list in comp_ca.items():
        cidx = comp_idx[label]
        for copy_within, ca in enumerate(ca_list):
            for res_local, res in enumerate(ca.residues):
                res_hvy_idx = np.intersect1d(res.atoms.indices, all_hvy_idx)
                rows = np.searchsorted(all_hvy_idx, res_hvy_idx)
                atom_to_local[rows] = res_local
                atom_to_copy[rows]  = copy_within
                atom_to_comp[rows]  = cidx

    comp_resids  = {label: comp_ca[label][0].residues.resids for label in comp_labels}
    comp_ncopies = {label: len(segs) for label, segs in comp_map.items()}

    pair_specs = [
        (comp_idx[x], comp_idx[y], comp_N[x], comp_N[y], comp_ncopies[x], x == y)
        for x, y in req_pairs
    ]

    del u

    frames   = list(range(start, stop or n_traj, stride))
    n_frames = len(frames)
    nprocs   = min(nproc, n_frames)

    typer.echo(f"Components : {', '.join(f'{l}×{comp_ncopies[l]}(N={comp_N[l]})' for l in comp_labels)}")
    typer.echo(f"Pairs      : {', '.join(f'{x}-{y}' for x, y in req_pairs)}")
    typer.echo(f"Cutoff     : {cutoff} Å  ({cutoff / 10:.2f} nm)")
    typer.echo(f"Frames     : {n_frames}  (stride={stride})")
    typer.echo(f"Workers    : {nprocs}")

    chunks      = [frames[i::nprocs] for i in range(nprocs)]
    worker_args = [
        (top, traj, cutoff, chunk, atom_to_local, atom_to_copy, atom_to_comp, pair_specs)
        for chunk in chunks
    ]

    t0 = time.perf_counter()
    if nprocs == 1:
        results = [_contacts_worker(worker_args[0])]
    else:
        ctx = mp.get_context("forkserver")
        with ctx.Pool(processes=nprocs) as pool:
            results = pool.map(_contacts_worker, worker_args)

    # ── Merge and save ─────────────────────────────────────────────────────
    stem = out.removesuffix(".npy")

    for (x, y), spec in zip(req_pairs, pair_specs):
        cx, cy, N_X, N_Y, n_X, same = spec
        key   = (cx, cy)
        denom = n_X * n_frames

        inter_sum = sum(r[key]["inter"] for r in results)
        inter_map = inter_sum / denom

        tag = f"{x}-{y}"
        if same:
            intra_sum = sum(r[key]["intra"] for r in results)
            intra_map = intra_sum / denom
            np.save(f"{stem}_{tag}_intra.npy",  intra_map)
            np.save(f"{stem}_{tag}_inter.npy",  inter_map)
            np.save(f"{stem}_{tag}_total.npy",  intra_map + inter_map)
            np.save(f"{stem}_{tag}_resids.npy", comp_resids[x])
            typer.echo(f"Saved -> {stem}_{tag}_intra.npy  (max {intra_map.max():.4f})")
            typer.echo(f"Saved -> {stem}_{tag}_inter.npy  (max {inter_map.max():.4f})")
            typer.echo(f"Saved -> {stem}_{tag}_total.npy  (max {(intra_map + inter_map).max():.4f})")
            typer.echo(f"Saved -> {stem}_{tag}_resids.npy")
        else:
            np.save(f"{stem}_{tag}_total.npy",      inter_map)
            np.save(f"{stem}_{tag}_resids_{x}.npy", comp_resids[x])
            np.save(f"{stem}_{tag}_resids_{y}.npy", comp_resids[y])
            typer.echo(f"Saved -> {stem}_{tag}_total.npy  (max {inter_map.max():.4f})")

    elapsed = time.perf_counter() - t0
    typer.echo(f"\nDone in {elapsed:.2f}s")


# =============================================================================
#  ░░  8.  Aggregation / Clustering / Density  ░░
# =============================================================================

AVOGADRO = 6.022e23
"""float: Avogadro's number (mol⁻¹), used for concentration unit conversion."""
 
 
# ---------------------------------------------------------------------------
#  Universe helpers
# ---------------------------------------------------------------------------
 
def _load_universe(top: str, traj: Optional[str] = None):
    """Load an MDAnalysis Universe with or without a trajectory.
 
    Args:
        top (str): Path to the topology file.
        traj (Optional[str]): Path to the trajectory file, or ``None`` to
            load only the single frame embedded in *top*.
 
    Returns:
        MDAnalysis.Universe: The loaded Universe object.
    """
    import MDAnalysis as mda
    return mda.Universe(top, traj) if traj else mda.Universe(top)
 
 
def _grp_init(u, ref_atom: str = "CA", step: int = 1):
    """Initialise per-segment reference-atom groups for clustering.
 
    Args:
        u (MDAnalysis.Universe): Loaded Universe.
        ref_atom (str): Name of the reference atom (default: ``"CA"``).
        step (int): Sub-sampling step applied within each segment's reference
            atoms.
 
    Returns:
        tuple[list[AtomGroup], np.ndarray]:
 
        - **segs** — List of ``AtomGroup`` objects, one per segment.
        - **grps** — Integer array of initial group IDs, shape
          ``(n_segments,)``.
    """
    segs, grps = [], []
    for i, segid in enumerate(u.segments):
        cas = u.select_atoms(f"segid {segid.segid} and name {ref_atom}")
        seg = cas[::step].atoms.select_atoms(f"name {ref_atom}")
        segs.append(seg)
        grps.append(i)
    return segs, np.array(grps)
 
 
# ---------------------------------------------------------------------------
#  Cluster detection
# ---------------------------------------------------------------------------
 
def _find_clusters_and_stats(segs, grps, r_cutoff: float = 8.0):
    """Detect aggregation clusters via union-find with kd-tree contact search.
 
    Args:
        segs (list[AtomGroup]): Reference-atom groups, one per segment.
        grps (np.ndarray): Initial group-ID array of shape ``(n_segments,)``.
            This array is **not** mutated; a copy is taken internally.
        r_cutoff (float): Contact distance cutoff in ångström.
 
    Returns:
        tuple:
 
        - **clusters** (*list[list[int]]*) — Each cluster is a list of
          segment indices.
        - **monomer** (*int*) — Number of isolated single-segment clusters.
        - **n_clusters** (*int*) — Number of multi-segment clusters.
        - **max_cluster_size** (*int*) — Size of the largest cluster.
        - **grps** (*np.ndarray*) — Updated group-ID array after merging.
    """
    nmol = len(segs)
    grps = grps.copy()          # never mutate the caller's array
 
    for i in range(nmol - 1):
        tree_i = cKDTree(segs[i].positions)
        grp_i  = grps[i]
        for j in range(i + 1, nmol):
            grp_j = grps[j]
            if grp_i == grp_j:
                continue
            neighbors = tree_i.query_ball_tree(cKDTree(segs[j].positions), r=r_cutoff)
            if any(len(nb) > 0 for nb in neighbors):
                grps  = np.where(grps == grp_j, grp_i, grps)
                grp_i = grps[i]
 
    aggr: dict = {}
    for g in grps:
        aggr[g] = aggr.get(g, 0) + 1
 
    monomer          = sum(1 for v in aggr.values() if v == 1)
    n_clusters       = len(aggr) - monomer
    max_cluster_size = max(aggr.values())
 
    clusters: dict = {}
    for seg_idx, cid in enumerate(grps):
        clusters.setdefault(cid, []).append(seg_idx)
 
    return list(clusters.values()), monomer, n_clusters, max_cluster_size, grps
 
 
# ---------------------------------------------------------------------------
#  PBC recentering
# ---------------------------------------------------------------------------
 
def _unwrap_cluster(u, cluster_seg_indices: list, box: np.ndarray) -> dict:
    """Unwrap a cluster split across periodic boundaries using BFS.
 
    Args:
        u (MDAnalysis.Universe): Universe at the current trajectory frame.
        cluster_seg_indices (list[int]): Indices of segments in the cluster.
        box (np.ndarray): Simulation cell dimensions, shape ``(6,)``.
 
    Returns:
        dict[int, np.ndarray]: ``{seg_index: shift_vector}`` — shift each
        segment's positions by this vector to place it in the same image as
        the reference segment (index 0 of the list).
    """
    box_half = box[:3] / 2.0
    shifts   = {cluster_seg_indices[0]: np.zeros(3)}
    queue    = deque([cluster_seg_indices[0]])
    visited  = {cluster_seg_indices[0]}
 
    while queue:
        ref_idx = queue.popleft()
        ref_com = (u.segments[ref_idx].atoms.positions + shifts[ref_idx]).mean(axis=0)
 
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
    """Unwrap the largest cluster, translate its CoM to the box centre, re-wrap.
 
    Args:
        u (MDAnalysis.Universe): Universe at the current trajectory frame.
        largest_cluster (list[int]): Segment indices of the largest cluster.
        box (np.ndarray): Simulation cell dimensions, shape ``(6,)``.
 
    Returns:
        np.ndarray: New position array for **all** atoms in *u*, shape
        ``(n_atoms, 3)``.  The Universe is **not** mutated by this function.
    """
    box_center = box[:3] / 2.0
    shifts     = _unwrap_cluster(u, largest_cluster, box)
    new_pos    = u.atoms.positions.copy()
 
    for seg_idx in largest_cluster:
        idx          = u.segments[seg_idx].atoms.indices
        new_pos[idx] += shifts[seg_idx]
 
    cluster_atom_indices = [
        i for seg_idx in largest_cluster
        for i in u.segments[seg_idx].atoms.indices
    ]
    center_shift  = box_center - new_pos[cluster_atom_indices].mean(axis=0)
    new_pos      += center_shift
 
    for dim in range(3):
        new_pos[:, dim] %= box[dim]
 
    return new_pos
 
 
# ---------------------------------------------------------------------------
#  Radial density
# ---------------------------------------------------------------------------
 
def _radial_density(u, droplet_center: np.ndarray, ca_per_segment: int,
                    ref_atom: str = "CA",
                    r_max: float = 100.0, dr: float = 1.0) -> tuple:
    """Compute the radial monomer concentration profile centred on a droplet.
 
    Args:
        u (MDAnalysis.Universe): Universe at the current (recentered) frame.
        droplet_center (np.ndarray): 3-D Cartesian coordinates of the droplet
            centre, in ångström.
        ca_per_segment (int): Number of reference atoms per monomer.
        ref_atom (str): Reference atom name (default: ``"CA"``).
        r_max (float): Maximum radius, in ångström.
        dr (float): Radial bin width, in ångström.
 
    Returns:
        tuple[np.ndarray, np.ndarray]:
 
        - **r_bins** — Bin centres, shape ``(n_bins,)``, in ångström.
        - **concentration** — Monomer concentration per shell, in mM.
    """
    all_ca    = u.select_atoms(f"name {ref_atom}")
    distances = np.linalg.norm(all_ca.positions - droplet_center, axis=1)
 
    n_bins = int(r_max / dr)
    bins   = np.linspace(0, r_max, n_bins + 1)
    r_bins = (bins[:-1] + bins[1:]) / 2.0
 
    ca_counts, _   = np.histogram(distances, bins=bins)
    monomer_counts = ca_counts / ca_per_segment
    shell_vols     = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    concentration  = (monomer_counts / shell_vols) * (1e30 / AVOGADRO)
 
    return r_bins, concentration
 
 
# ---------------------------------------------------------------------------
#  Parallel worker
# ---------------------------------------------------------------------------
 
def _worker(
    top: str,
    traj: str,
    traj_indices: list,          # absolute indices into the trajectory array
    ref: str,
    rcut: float,
    castep: int,
    recenter: bool,
    density: bool,
    dr: float,
    n_frames_avg: int,
    n_frames_total: int,         # total number of analysed frames across all workers
    chunk_start_rank: int,       # global rank of traj_indices[0] among analysed frames
    tmp_traj: Optional[str],     # path for this worker's temporary XTC, or None
) -> dict:
    """Analyse a contiguous subset of frames in a subprocess.
 
    Each worker opens its own Universe — ``mda.Universe`` is not picklable
    and cannot be shared across processes.  Frames are accessed by absolute
    trajectory index so that any start/stop/stride combination is handled
    correctly regardless of how frame numbers are stored in the XTC.
 
    Args:
        top (str): Topology file path.
        traj (str): Trajectory file path.
        traj_indices (list[int]): Absolute trajectory indices to process.
        ref (str): Reference atom name.
        rcut (float): Clustering distance cutoff (Å).
        castep (int): Sub-sampling step for reference atoms.
        recenter (bool): Whether to unwrap and recenter the largest cluster.
        density (bool): Whether to compute radial density profiles.
        dr (float): Radial bin width (Å).
        n_frames_avg (int): Number of final frames to include in density avg.
        n_frames_total (int): Total analysed frames (all workers combined).
        chunk_start_rank (int): Global rank of this chunk's first frame.
        tmp_traj (Optional[str]): Path to write the recentered trajectory
            fragment, or ``None`` if recentering is disabled.
 
    Returns:
        dict:
            ``stats_data``       — list of ``(frame_number, monomers, clusters, max_size)``
            ``density_profiles`` — list of ``(global_rank, r_bins, conc)``
    """
    import MDAnalysis as mda
 
    u              = _load_universe(top, traj)
    segs, _        = _grp_init(u, ref_atom=ref, step=castep)
    ca_per_segment = len(u.segments[0].atoms.select_atoms(f"name {ref}"))
 
    stats_data:       list = []
    density_profiles: list = []
 
    writer = mda.Writer(tmp_traj, n_atoms=u.atoms.n_atoms) if (recenter and tmp_traj) else None
 
    try:
        for local_rank, traj_idx in enumerate(traj_indices):
            u.trajectory[traj_idx]              # seek by absolute index
            ts  = u.trajectory.ts
            box = ts.dimensions.copy()          # copy — dimensions can be a mutable view
 
            # Reset group labels every frame; never carry state between frames.
            grps_frame = np.arange(len(segs))
 
            for seg in segs:
                seg.wrap()
 
            clusters, monomer, n_clusters, max_size, _ = _find_clusters_and_stats(
                segs, grps_frame, r_cutoff=rcut
            )
 
            if recenter:
                largest           = max(clusters, key=len)
                u.atoms.positions = _recenter_frame(u, largest, box)
                if writer:
                    writer.write(u.atoms)
 
                if density:
                    global_rank = chunk_start_rank + local_rank
                    if global_rank >= n_frames_total - n_frames_avg:
                        r_max = float(np.min(box[:3]) / 2.0)
                        r_bins, conc = _radial_density(
                            u,
                            droplet_center=box[:3] / 2.0,
                            ca_per_segment=ca_per_segment,
                            ref_atom=ref,
                            r_max=r_max,
                            dr=dr,
                        )
                        density_profiles.append((global_rank, r_bins, conc))
 
            stats_data.append((ts.frame, monomer, n_clusters, max_size))
 
    finally:
        if writer is not None:
            writer.close()
 
    return {
        "stats_data":       stats_data,
        "density_profiles": density_profiles,
    }
 
 
# ---------------------------------------------------------------------------
#  Trajectory merge
# ---------------------------------------------------------------------------
 
def _merge_trajectories(top: str, tmp_paths: list, out_path: str) -> None:
    """Concatenate per-worker XTC fragments into a single output trajectory.
 
    Workers receive **contiguous** frame chunks, so sequential concatenation
    preserves the original frame order exactly.
 
    ``writer.write(tmp_u.atoms)`` copies both atom positions and the current
    ``ts.dimensions`` into the output file — no manual dimension copy needed.
 
    Args:
        top (str): Topology file path (used only to determine atom count).
        tmp_paths (list[str]): Ordered list of per-worker XTC paths.
            Must be sorted in frame order (i.e. chunk order).
        out_path (str): Destination XTC path.
    """
    import MDAnalysis as mda
 
    # Load topology only to get n_atoms without opening any trajectory file.
    n_atoms = _load_universe(top).atoms.n_atoms
 
    with mda.Writer(out_path, n_atoms=n_atoms) as wout:
        for tmp_path in tmp_paths:
            tmp_u = _load_universe(top, tmp_path)
            for _ in tmp_u.trajectory:
                # write() reads positions AND ts.dimensions from tmp_u directly
                wout.write(tmp_u.atoms)
            os.remove(tmp_path)
 
 
# ---------------------------------------------------------------------------
#  CLI command
# ---------------------------------------------------------------------------
 
@app.command("aggr")
def cmd_aggr(
    top:     Annotated[str,  typer.Option("--top",     help="Topology file (PSF)")] = "conf.psf",
    traj:    Annotated[str,  typer.Option("--traj",    help="Trajectory file")] = "system.xtc",
    out:     Annotated[str,  typer.Option("--out",     help="Aggregation statistics output")] = "aggr.dat",
    outtraj: Annotated[str,  typer.Option("--outtraj", help="Recentered trajectory output")] = "recentered.xtc",
    profile: Annotated[str,  typer.Option("--profile", help="Radial density profile output")] = "density_profile.dat",
    ref:     Annotated[str,  typer.Option("--ref",     help="Reference atom name (CA or P)")] = "CA",
    rcut:    Annotated[float, typer.Option("--rcut",   help="Clustering distance cutoff (Å)")] = 8.0,
    castep:  Annotated[int,   typer.Option("--castep", help="Use every Nth ref atom for clustering")] = 1,
    start:   Annotated[Optional[int], typer.Option("--start",  help="First frame index")] = None,
    stop:    Annotated[Optional[int], typer.Option("--stop",   help="Last frame index (exclusive)")] = None,
    stride:  Annotated[int,  typer.Option("--stride",  help="Step between analysed frames")] = 1,
    recenter: Annotated[bool, typer.Option("--recenter/--no-recenter",
              help="Unwrap & recenter the largest cluster each frame")] = False,
    density:  Annotated[bool, typer.Option("--density/--no-density",
              help="Compute radial monomer concentration (requires --recenter)")] = False,
    dr:           Annotated[float, typer.Option("--dr",           help="Radial bin width (Å)")] = 2.0,
    n_frames_avg: Annotated[int,   typer.Option("--n-frames-avg", help="Last N frames averaged for density profile")] = 50,
    nproc:        Annotated[int,   typer.Option("--nproc",        help="Number of parallel worker processes (1 = serial)")] = 1,
):
    """Analyse protein **aggregation**: cluster detection, PBC recentering, and radial density.
 
    Processes each trajectory frame to:
 
    1. **Cluster** — detect multi-chain aggregates using kd-tree contact search
       on reference atoms (default: Cα), with a union-find merging scheme.
    2. **Recenter** *(optional, ``--recenter``)* — unwrap the largest cluster
       across PBC and translate it to the box centre; write as a new trajectory.
    3. **Density profile** *(optional, ``--density``, requires ``--recenter``)* —
       compute and average the radial monomer concentration (mM) over the last
       ``--n-frames-avg`` frames.
 
    Parallelism is over frames (``--nproc``): each worker process handles a
    contiguous chunk of the trajectory and writes its own temporary XTC fragment,
    which are merged in order at the end.
 
    Example::
 
        # Serial — cluster statistics only
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0
 
        # 8 workers — with PBC recentering and radial density
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 \\
            --recenter --density --dr 2.0 --n-frames-avg 100 --nproc 8
    """
    import MDAnalysis as mda
 
    # Guard: density requires recentering
    if density and not recenter:
        typer.echo("[!] --density requires --recenter — density disabled.", err=True)
        density = False
 
    typer.echo("Loading trajectory...")
    u = _load_universe(top, traj)
 
    # Build the list of absolute trajectory indices to analyse.
    # Using indices (not XTC frame numbers) guarantees correct random-access
    # seeking even when the XTC stores non-contiguous or non-zero-based frame
    # numbers.
    _start  = start  if start  is not None else 0
    _stop   = stop   if stop   is not None else len(u.trajectory)
    traj_indices = list(range(_start, _stop, stride))
    n_frames     = len(traj_indices)
 
    # Initialise segment groups once — workers create their own copies.
    segs, _        = _grp_init(u, ref_atom=ref, step=castep)
    ca_per_segment = len(u.segments[0].atoms.select_atoms(f"name {ref}"))
 
    typer.echo(f"[i] Segments      : {len(segs)}")
    typer.echo(f"[i] {ref} / segment  : {ca_per_segment}")
    typer.echo(f"[i] Frames        : {n_frames}")
    typer.echo(f"[i] Cutoff        : {rcut} Å")
    typer.echo(f"[i] nproc         : {nproc}")
    typer.echo(f"[i] Recenter      : {recenter}")
    typer.echo(f"[i] Density       : {density}")
 
    # Initialise output accumulators before either branch so the common output
    # section below is always guaranteed to find them defined.
    stats_data:       list = []   # list of (frame_number, monomers, clusters, max_size)
    density_profiles: list = []   # list of (r_bins, conc) — unified format for both paths
 
    # ------------------------------------------------------------------ #
    # Serial path  (nproc == 1)
    # ------------------------------------------------------------------ #
    if nproc == 1:
        writer = mda.Writer(outtraj, n_atoms=u.atoms.n_atoms) if recenter else None
        try:
            for ts_idx, abs_idx in enumerate(traj_indices):
                u.trajectory[abs_idx]
                ts  = u.trajectory.ts
                box = ts.dimensions.copy()
 
                if ts_idx % 10 == 0:
                    typer.echo(f"  Frame {ts_idx + 1}/{n_frames}  (traj frame {ts.frame})")
 
                # Reset group labels every frame — never carry state between frames.
                grps_frame = np.arange(len(segs))
 
                for seg in segs:
                    seg.wrap()
 
                clusters, monomer, n_clusters, max_size, _ = _find_clusters_and_stats(
                    segs, grps_frame, r_cutoff=rcut
                )
 
                if ts_idx % 10 == 0:
                    typer.echo(
                        f"    monomers={monomer}  clusters={n_clusters}  largest={max_size}"
                    )
 
                if recenter:
                    largest           = max(clusters, key=len)
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
                        density_profiles.append((r_bins, conc))
 
                stats_data.append((ts.frame, monomer, n_clusters, max_size))
 
        finally:
            if writer is not None:
                writer.close()
 
    # ------------------------------------------------------------------ #
    # Parallel path  (nproc > 1)
    # ------------------------------------------------------------------ #
    else:
        # Contiguous chunks — chunk k owns frames [k*sz .. (k+1)*sz - 1].
        # This is required so that per-worker XTC fragments can be merged in
        # list order to reconstruct the correct frame sequence.
        chunk_size        = (n_frames + nproc - 1) // nproc
        chunks            = [traj_indices[i : i + chunk_size]
                             for i in range(0, n_frames, chunk_size)]
        # Global rank of each chunk's first frame among all analysed frames.
        chunk_start_ranks = [i * chunk_size for i in range(len(chunks))]
 
        # tmp_paths is populated in chunk (== frame) order so that
        # _merge_trajectories can simply concatenate them sequentially.
        tmp_paths: list = []
        futures:   dict = {}
 
        with ProcessPoolExecutor(max_workers=nproc) as pool:
            for chunk, start_rank in zip(chunks, chunk_start_ranks):
                tmp = tempfile.mktemp(suffix=".xtc") if recenter else None
                if tmp:
                    tmp_paths.append(tmp)
 
                fut = pool.submit(
                    _worker,
                    top, traj, chunk, ref, rcut, castep,
                    recenter, density, dr,
                    n_frames_avg, n_frames, start_rank, tmp,
                )
                futures[fut] = chunk
 
            raw_stats:   list = []
            raw_density: list = []
 
            for fut in as_completed(futures):
                result = fut.result()
                raw_stats.extend(result["stats_data"])
                raw_density.extend(result["density_profiles"])
 
        # Restore frame order — as_completed() returns in completion order,
        # not submission order.
        stats_data = sorted(raw_stats, key=lambda x: x[0])
 
        # Merge per-worker XTC fragments in chunk order (== frame order).
        if recenter and tmp_paths:
            typer.echo(f"Merging {len(tmp_paths)} partial trajectories -> {outtraj}")
            _merge_trajectories(top, tmp_paths, outtraj)
 
        # Rebuild density_profiles in the same (r_bins, conc) format as the
        # serial path so the common output section below handles both uniformly.
        if density and raw_density:
            raw_density.sort(key=lambda x: x[0])          # sort by global_rank
            min_len      = min(len(conc) for _, _, conc in raw_density)
            r_bins_first = raw_density[0][1][:min_len]
            density_profiles = [
                (r_bins_first, c[:min_len]) for _, _, c in raw_density
            ]
 
    # ------------------------------------------------------------------ #
    # Write outputs  (common to both serial and parallel paths)
    # ------------------------------------------------------------------ #
    typer.echo(f"\nWriting aggregation statistics -> {out}")
    with open(out, "w") as fh:
        fh.write("# Frame  Monomers  Clusters  LargestClusterSize\n")
        for frame, monomer, n_clusters, max_size in stats_data:
            fh.write(f"{frame}  {monomer}  {n_clusters}  {max_size}\n")
 
    if density and density_profiles:
        # Truncate to the shortest profile before stacking.  r_max varies per
        # frame (it is clipped to half the box side), so array lengths can
        # differ by one bin between frames.
        min_len      = min(len(c) for _, c in density_profiles)
        r_bins_out   = density_profiles[0][0][:min_len]
        profiles_arr = np.array([c[:min_len] for _, c in density_profiles])
        avg_conc     = np.mean(profiles_arr, axis=0)
        std_conc     = np.std( profiles_arr, axis=0)
 
        typer.echo(
            f"Writing radial density profile ({len(density_profiles)} frames) -> {profile}"
        )
        with open(profile, "w") as fh:
            fh.write(
                f"# Radial monomer concentration averaged over last "
                f"{len(density_profiles)} frames\n"
            )
            fh.write("# Radius(A)  Concentration(mM)  StdDev(mM)\n")
            for r, c, s in zip(r_bins_out, avg_conc, std_conc):
                fh.write(f"{r:.3f}  {c:.6f}  {s:.6f}\n")
 
    if recenter:
        typer.echo(f"Recentered trajectory -> {outtraj}")
    typer.echo("DONE!")


# =============================================================================
#  ░░  9.  Time-dependent S²  ░░
# =============================================================================

def _s2_parse_index_selection(spec: str, max_index: int) -> list:
    """Parse a segment index selection string into a sorted list of 0-based indices.

    Accepted formats (inclusive on both ends):
        ``"3"``         → ``[3]``
        ``"0-4"``       → ``[0, 1, 2, 3, 4]``
        ``"0-2,5-7"``   → ``[0, 1, 2, 5, 6, 7]``
        ``"1,3,5"``     → ``[1, 3, 5]``

    Args:
        spec (str): Selection string from the ``--segments`` argument.
        max_index (int): Length of the available segment list (for bounds checking).

    Returns:
        list[int]: Sorted, deduplicated list of valid integer indices.

    Raises:
        ValueError: If any index is out of range or a range token is malformed.
    """
    indices = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range token: {token!r}")
            lo, hi = int(parts[0]), int(parts[1])
            if lo > hi:
                raise ValueError(f"Range start > end in token: {token!r}")
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(token))

    bad = [i for i in indices if i < 0 or i >= max_index]
    if bad:
        raise ValueError(
            f"Segment index/indices {bad} out of range "
            f"(available: 0 – {max_index - 1})"
        )
    return sorted(indices)


def _s2_calc_vs_lag(vectors: np.ndarray, lag_frames: list) -> np.ndarray:
    """Compute window-averaged S²(τ) for one residue from raw N-H bond vectors.

    For each lag τ in *lag_frames*:

    1. Slide a window of length τ over the trajectory collecting every
       possible origin ``t₀ ∈ [0, n_frames − τ)``.
    2. Compute ``S² = (1/2)(3·Σᵢⱼ<μᵢμⱼ>² − 1)`` inside each window.
    3. Average S² over all windows → S²(τ).

    τ = 0 is **not** handled here; the caller (``_s2_residue_worker``) always
    writes 1.0 for that row.

    All windows for a given τ are stacked into a single ``(n_origins, τ, 3)``
    array and processed with ``np.einsum`` — there is no Python loop over
    time origins.

    Args:
        vectors (np.ndarray): Raw (un-normalised) N-H bond vectors,
            shape ``(n_frames, 3)``.  Normalisation happens per-window.
        lag_frames (list[int]): Window lengths τ (in frames) to evaluate.
            All values must be ≥ 2; entries that exceed ``n_frames`` return
            ``nan``.

    Returns:
        np.ndarray: S²(τ) for each requested lag, shape ``(len(lag_frames),)``.
        Entries are in ``[0, 1]``; ``nan`` where τ > n_frames.
    """
    n      = len(vectors)
    s2_tau = np.full(len(lag_frames), np.nan)

    for i, tau in enumerate(lag_frames):
        if tau < 2 or tau > n:
            continue

        n_origins = n - tau
        if n_origins < 1:
            continue

        # Stack all windows: (n_origins, tau, 3)
        windows = np.stack(
            [vectors[t0 : t0 + tau] for t0 in range(n_origins)],
            axis=0,
        )

        # Normalise to unit vectors within every window frame
        norms   = np.linalg.norm(windows, axis=2, keepdims=True)  # (n_origins, tau, 1)
        windows = windows / norms

        # <μᵢμⱼ> averaged over τ frames for each window: (n_origins, 3, 3)
        M = np.einsum("wfi,wfj->wij", windows, windows) / tau

        # S²[w] = (3·Σᵢⱼ M[w,i,j]² − 1) / 2  for each window w
        s2_per_window = (3.0 * np.sum(M * M, axis=(1, 2)) - 1.0) / 2.0
        s2_per_window = np.clip(s2_per_window, 0.0, 1.0)

        s2_tau[i] = s2_per_window.mean()

    return s2_tau


def _s2_residue_worker(args: tuple) -> int:
    """Subprocess worker: compute S²(τ) for one residue and write its output file.

    τ = 0 is always written as S² = 1.0 (zero-length window — perfectly ordered
    by definition).  For all other lags ``_s2_calc_vs_lag`` is called.

    Args:
        args (tuple): Packed argument tuple containing:

            - **resid** (*int*) — Residue number.
            - **vectors_by_segid** (*dict[str, np.ndarray]*) — Mapping of
              segid → raw N-H bond vectors, shape ``(n_frames, 3)``.
            - **all_segids** (*list[str]*) — Ordered list of all selected
              segment IDs (determines column order in the output file).
            - **lag_frames** (*list[int]*) — Window lengths τ to evaluate
              (τ = 0 is prepended automatically; these are all ≥ 2).
            - **out_dir** (*str*) — Output directory path (string, not Path).

    Returns:
        int: The processed *resid*, for progress tracking in the parent process.

    Output:
        ``<out_dir>/resid_<resid>.dat`` — space-separated columns:
        ``lag_frame  s2_<segid1>  s2_<segid2>  …``
        The first row always has ``lag_frame = 0`` and all s2 columns = 1.0.
    """
    from pathlib import Path

    resid, vectors_by_segid, all_segids, lag_frames, out_dir_str = args

    lag_with_zero = [0] + lag_frames
    n_lags        = len(lag_with_zero)

    # Build columns: one per segid
    cols = []
    for segid in all_segids:
        vecs = vectors_by_segid.get(segid)
        s2   = np.full(n_lags, np.nan)
        s2[0] = 1.0                                            # τ = 0 → 1 by definition
        if vecs is not None:
            s2[1:] = _s2_calc_vs_lag(vecs, lag_frames)
        cols.append(s2)

    data   = np.column_stack([np.array(lag_with_zero, dtype=float)] + cols)
    header = "lag_frame " + " ".join(f"s2_{seg}" for seg in all_segids)
    fname  = Path(out_dir_str) / f"resid_{resid}.dat"

    np.savetxt(fname, data, header=header,
               comments="# ", fmt=["%.0f"] + ["%.8f"] * len(all_segids))
    return resid


def _s2_extract_nh_vectors(
    topology: str,
    trajectory: str,
    selected_segids: Optional[list] = None,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    align_traj: bool = False,
) -> dict:
    """Extract raw backbone N-H bond vectors per (segid, resid) across all frames.

    Proline residues are excluded (no backbone H).  Vectors are **not**
    normalised here; normalisation is deferred to per-window computation in
    ``_s2_calc_vs_lag`` so that the raw geometry is preserved.

    Args:
        topology (str): Path to the topology file (.pdb, .psf, .gro, …).
        trajectory (str): Path to the trajectory file (.xtc, .dcd, .trr, …).
        selected_segids (Optional[list[str]]): If given, only these segids are
            extracted; all others are skipped before any frame iteration.
        start (int): Index of the first frame to read.
        stop (Optional[int]): Index of the last frame (``None`` = all frames).
        step (int): Frame stride when reading the trajectory.
        align_traj (bool): If ``True``, align each frame to frame 0 on Cα
            atoms before extracting vectors.  Off by default — only needed
            when overall tumbling has not already been removed from the
            trajectory.

    Returns:
        dict: ``{(segid, resid): np.ndarray shape (n_frames, 3)}`` — raw
        N-H bond vectors (H position minus N position) for every matched
        backbone N-H pair.

    Raises:
        ValueError: If no backbone N-H pairs are found after applying any
            segid filter, or if the atom naming is not recognised.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import align as mda_align
    from collections import Counter

    u = mda.Universe(topology, trajectory)

    if align_traj:
        ref     = mda.Universe(topology, trajectory)
        aligner = mda_align.AlignTraj(
            u, ref, select="backbone and name CA", in_memory=True
        )
        aligner.run(start=start, stop=stop, step=step)

    sel_N = u.select_atoms("name N and not resname PRO")
    sel_H = u.select_atoms("name H and not resname PRO")

    n_by_key = {(a.segid, a.resid): a for a in sel_N}
    h_by_key = {(a.segid, a.resid): a for a in sel_H}
    shared_keys = sorted(set(n_by_key) & set(h_by_key))

    if not shared_keys:
        raise ValueError(
            "No backbone N-H pairs found.  Check atom naming "
            "('H' vs 'HN') and that segids are set in your topology."
        )

    if selected_segids is not None:
        selected_set = set(selected_segids)
        shared_keys  = [k for k in shared_keys if k[0] in selected_set]
        if not shared_keys:
            raise ValueError(
                f"No N-H pairs remain after filtering to segids: {selected_segids}."
            )

    seg_counts = Counter(seg for seg, _ in shared_keys)
    for seg, count in sorted(seg_counts.items()):
        typer.echo(f"  Segment {seg!r:>10s} : {count} N-H pairs")
    typer.echo(f"  Total             : {len(shared_keys)} N-H pairs")

    vectors: dict = {k: [] for k in shared_keys}
    for _ts in u.trajectory[start:stop:step]:
        for key in shared_keys:
            n_pos = n_by_key[key].position
            h_pos = h_by_key[key].position
            vectors[key].append(h_pos - n_pos)

    return {key: np.array(vecs) for key, vecs in vectors.items()}


@app.command("time-s2")
def cmd_time_s2(
    top: Annotated[str, typer.Option("--top", help="Topology file")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj", help="Trajectory file")] = "system.xtc",
    segments: Annotated[Optional[str], typer.Option("--segments",
        help="Segment indices to include, e.g. '0-4' or '0,2,5-7' (0-based). "
             "Omit to use all segments.  Run once without this flag to print "
             "the full index list, then rerun with a filter.")] = None,
    min_lag: Annotated[int, typer.Option("--min-lag", help="Minimum window length τ in frames (default: 2)")] = 2,
    max_lag: Annotated[int, typer.Option("--max-lag", help="Maximum window length τ in frames (-1 = n_frames // 2)")] = -1,
    stride: Annotated[int, typer.Option("--stride", help="Step between successive τ values written to output (default: 1)")] = 1,
    start: Annotated[int, typer.Option("--start", help="First trajectory frame to read (default: 0)")] = 0,
    stop: Annotated[int, typer.Option("--stop", help="Last trajectory frame to read; -1 = all")] = -1,
    step: Annotated[int, typer.Option("--step", help="Frame stride when reading trajectory (default: 1)")] = 1,
    align: Annotated[bool, typer.Option("--align/--no-align", help="Align each frame on Cα before computing S² (off by default)")] = False,
    outdir: Annotated[str, typer.Option("--outdir", help="Output directory (default: s2-time)")] = "s2-time",
    nproc: Annotated[int, typer.Option("--nproc", help="Number of parallel worker processes (default: 1)")] = 1,
):
    """Compute **time-dependent S²(τ)** order parameters for backbone N-H bonds.

    For each window length τ (in frames):

    1. Slide a window of length τ over the trajectory — every possible origin
       ``t₀ ∈ [0, n_frames − τ)`` is used (analogous to MSD).
    2. Compute ``S² = (1/2)(3·Σᵢⱼ<μᵢμⱼ>² − 1)`` inside each window using
       the Lipari-Szabo tensor formula.
    3. Average S² over all windows → S²(τ).

    τ = 0 is always written as S² = 1.0 (zero-length window, perfectly ordered
    by definition).  As τ increases, S²(τ) converges to the Lipari-Szabo
    plateau value.

    Proline residues are skipped (no backbone NH).  Results are written to one
    file per residue so that per-residue curves can be loaded and plotted
    independently.

    Args:
        top (str): Path to the topology file (.pdb, .psf, .gro, …).
        traj (str): Path to the trajectory file (.xtc, .dcd, .trr, …).
        segments (Optional[str]): Segment index selection string.  Segments
            are numbered 0-based in topology order.  Run without this flag
            first to see the full list with indices.
        min_lag (int): Smallest window length τ to evaluate (frames).
        max_lag (int): Largest window length τ (frames); ``-1`` uses
            ``n_frames // 2``, which gives statistically reliable estimates
            (at least half the trajectory contributes to every average).
        stride (int): Step between successive τ values in the output.  Use
            this to reduce output size when many lag points are not needed.
        start (int): Index of the first trajectory frame to read.
        stop (int): Index of the last trajectory frame; ``-1`` = all frames.
        step (int): Frame stride when reading the trajectory.
        align (bool): Align each frame on Cα atoms before computing S².
            Off by default; only enable if tumbling has not been removed.
        outdir (str): Output directory (created automatically if absent).
        nproc (int): Number of parallel worker processes.  Parallelism is
            over residues — each worker handles one residue independently.

    Output:
        ``<outdir>/resid_<N>.dat`` — one file per residue; space-separated
        columns: ``lag_frame  s2_<segid1>  s2_<segid2>  …``
        The first data row always has ``lag_frame = 0`` and all s2 = 1.0.

    Example::

        # See segment list first
        scical time-s2 --top conf.psf --traj system.xtc

        # Run on segments 0–4, every 10th τ value, 8 workers
        scical time-s2 --top conf.psf --traj system.xtc \\
            --segments 0-4 --min-lag 2 --max-lag 5000 --stride 10 \\
            --start 0 --stop -1 --step 1 --outdir s2-time --nproc 8
    """
    import MDAnalysis as mda
    from pathlib import Path

    os.makedirs(outdir, exist_ok=True)

    # ── Resolve segment list and optional filter ───────────────────────────
    u = mda.Universe(top, traj)
    all_segids = sorted(
        {a.segid for a in u.select_atoms("name N and not resname PRO")}
    )

    typer.echo(f"\nAll segments found ({len(all_segids)}):")
    for i, sid in enumerate(all_segids):
        typer.echo(f"  [{i}] {sid}")

    if segments is not None:
        idx_list        = _s2_parse_index_selection(segments, len(all_segids))
        selected_segids = [all_segids[i] for i in idx_list]
        typer.echo(f"\nSelected segments (--segments {segments!r}): {selected_segids}")
    else:
        selected_segids = None
        typer.echo("\nNo --segments filter; using all segments.")

    typer.echo(f"Alignment: {'ON (Cα)' if align else 'OFF (default)'}")

    # ── Extract N-H vectors ────────────────────────────────────────────────
    traj_stop = _traj_stop(stop)      # reuse module helper: -1 → None

    typer.echo("\nExtracting N-H vectors …")
    nh_vectors = _s2_extract_nh_vectors(
        top, traj,
        selected_segids=selected_segids,
        start=start,
        stop=traj_stop,
        step=step,
        align_traj=align,
    )

    # ── Build lag frame list ───────────────────────────────────────────────
    n_frames   = next(iter(nh_vectors.values())).shape[0]
    eff_max    = max_lag if max_lag != -1 else n_frames // 2
    lag_frames = list(range(min_lag, eff_max + 1, stride))   # τ=0 added in worker

    active_segids = sorted({seg for seg, _ in nh_vectors})
    all_resids    = sorted({rid for _, rid in nh_vectors})

    typer.echo(f"\n[i] n_frames     : {n_frames}")
    typer.echo(f"[i] lag range    : 0 (fixed=1.0), {min_lag} – {eff_max}, stride={stride}")
    typer.echo(f"[i] n_lag_points : {len(lag_frames) + 1}  (including τ=0)")
    typer.echo(f"[i] Segments     : {active_segids}")
    typer.echo(f"[i] Residues     : {len(all_resids)}")
    typer.echo(f"[i] Workers      : {nproc}")
    typer.echo(f"[i] Output dir   : {outdir}/")

    # ── Build per-residue task list ────────────────────────────────────────
    # Pass out_dir as str (avoids any platform pickle edge-cases with Path).
    tasks = []
    for resid in all_resids:
        vectors_by_segid = {
            segid: nh_vectors[(segid, resid)]
            for segid in active_segids
            if (segid, resid) in nh_vectors
        }
        tasks.append(
            (resid, vectors_by_segid, active_segids, lag_frames, outdir)
        )

    n_tasks      = len(tasks)
    report_every = max(1, n_tasks // 10)

    # ── Dispatch ───────────────────────────────────────────────────────────
    typer.echo("\nComputing S²(τ) …")

    if nproc == 1:
        for i, task in enumerate(tasks):
            rid = _s2_residue_worker(task)
            if (i + 1) % report_every == 0:
                typer.echo(f"  {i + 1}/{n_tasks} residues done (resid {rid})")
    else:
        with Pool(processes=nproc) as pool:
            for i, rid in enumerate(
                pool.imap_unordered(_s2_residue_worker, tasks)
            ):
                if (i + 1) % report_every == 0:
                    typer.echo(f"  {i + 1}/{n_tasks} residues done (resid {rid})")

    typer.echo(f"\n[+] S²(τ) results written to {outdir}/")


# =============================================================================
#  ░░  Entry-point  ░░
# =============================================================================

if __name__ == "__main__":
    app()