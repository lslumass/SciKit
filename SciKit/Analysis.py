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


def _suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore")

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
    _suppress_warnings()
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
    _suppress_warnings()
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

def _parse_seg_selection(sel_str: str, available_segids: list[str]) -> list[str]:
    """Parse a segment selection string into a list of segment IDs.

    Supports:
        - Single segment: ``"R001"``
        - Range of segments: ``"R001-R010"`` (matches numeric suffix)
        - Comma-separated mix: ``"R001,R003-R006,R010"``

    Args:
        sel_str (str): Segment selection expression.
        available_segids (list[str]): All segment IDs present in the system.

    Returns:
        list[str]: Filtered list of matching segment IDs, preserving the order
        they appear in *available_segids*.

    Raises:
        typer.Exit: If the range expression cannot be parsed or prefixes
        do not match.
    """
    import re

    selected = set()

    for token in sel_str.split(","):
        token = token.strip()
        if "-" in token and token.count("-") == 1:
            # Range: e.g. "R001-R010"
            start_seg, end_seg = token.split("-")
            start_seg, end_seg = start_seg.strip(), end_seg.strip()

            # Extract prefix and numeric parts
            m_start = re.match(r"^([A-Za-z]*)(\d+)$", start_seg)
            m_end   = re.match(r"^([A-Za-z]*)(\d+)$", end_seg)

            if not m_start or not m_end:
                typer.echo(f"[!] Cannot parse range: '{token}'", err=True)
                raise typer.Exit(1)

            prefix_s, num_s = m_start.group(1), int(m_start.group(2))
            prefix_e, num_e = m_end.group(1),   int(m_end.group(2))

            if prefix_s != prefix_e:
                typer.echo(
                    f"[!] Mismatched prefixes in range: '{prefix_s}' vs '{prefix_e}'",
                    err=True,
                )
                raise typer.Exit(1)

            if num_s > num_e:
                typer.echo(
                    f"[!] Invalid range: start ({num_s}) > end ({num_e})",
                    err=True,
                )
                raise typer.Exit(1)

            # Match available segids that fall within the numeric range
            for segid in available_segids:
                m = re.match(r"^([A-Za-z]*)(\d+)$", segid)
                if m and m.group(1) == prefix_s:
                    num = int(m.group(2))
                    if num_s <= num <= num_e:
                        selected.add(segid)
        else:
            # Single segment
            selected.add(token)

    # Preserve original ordering from available_segids
    filtered = [s for s in available_segids if s in selected]
    return filtered


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
    _suppress_warnings()
    import MDAnalysis as mda

    segid, topology, trajectory, resid_range, start, stop, stride = args
    u   = mda.Universe(topology, trajectory)
    sel = u.select_atoms(
        f"segid {segid} and resid {resid_range}"
        if resid_range else f"segid {segid}"
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
    sel: Annotated[Optional[str], typer.Option("--sel", help="Segment selection (e.g. R001, R001-R010, or R001,R003-R006)")] = None,
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
        sel (Optional[str]): Segment selection expression. Accepts a single
            segment (``"R001"``), a range (``"R001-R010"``), or a
            comma-separated combination (``"R001,R003-R006,R010"``).
            Omit to use all segments.
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

            scical rg --top conf.psf --traj system.xtc --sel R001-R010 --out rg.dat --stride 5 --nproc 4
    """
    _suppress_warnings()
    import MDAnalysis as mda

    u = mda.Universe(top, traj)
    segments = [
        seg for seg in u.segments
        if len(seg.atoms.select_atoms(
            f"resid {resid}" if resid else "all"
        )) > 0
    ]
    if not segments:
        typer.echo("[!] No matching protein segments found.", err=True)
        raise typer.Exit(1)

    segids = [seg.segid for seg in segments]

    # --- Apply --sel filter ---
    if sel:
        segids = _parse_seg_selection(sel, segids)
        if not segids:
            typer.echo(f"[!] No segments match selection: '{sel}'", err=True)
            raise typer.Exit(1)

    typer.echo(f"[i] Segments : {segids}")
    typer.echo(f"[i] Region   : {'resid ' + resid if resid else 'all'}")

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
    _suppress_warnings()
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
    _suppress_warnings()
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
    _suppress_warnings()
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
    _suppress_warnings()
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
    """Write a pairwise distance matrix to a space-separated ``.dat`` file.

    Output format mirrors the Rg output: one row per frame, one column per
    pair.  The first column is the frame number; subsequent columns are the
    Ca-Ca distances (Angstrom) for each pair.  Pair identities are encoded
    compactly in the header as ``<segid1>:<resid1>-<segid2>:<resid2>`` --
    no per-row pair annotation is written.

    Args:
        out_path (str): Destination file path.
        pairs (list[tuple]): List of ``(resid1, segid1, resid2, segid2)``
            tuples -- one per column (after the frame column).
        dist_block (np.ndarray): Distance matrix of shape
            ``(n_pairs, n_frames)`` in Angstrom.
        all_frames (list[int]): Ordered list of trajectory frame numbers
            corresponding to the columns of *dist_block*.
    """
    data   = np.column_stack([np.array(all_frames)] + [dist_block[i] for i in range(len(pairs))])
    W      = 14
    labels = [_pair_label(r1, s1, r2, s2) for r1, s1, r2, s2 in pairs]
    frame_col = "frame"
    header = " ".join(
        [f"{frame_col:>{W - 2}}"] + [f"{lbl:>{W}}" for lbl in labels]
    )
    np.savetxt(out_path, data, header=header,
               fmt=[f"%{W}.0f"] + [f"%{W}.4f"] * len(pairs))


@app.command("distance")
def cmd_dist(
    top: Annotated[str, typer.Option("-p", "--top")] = "conf.psf",
    traj: Annotated[str, typer.Option("-t", "--traj")] = "system.dcd",
    pair_files: Annotated[List[str], typer.Option("-f", "--pair-files",
                help="One or more residue-pair list files")] = ["./residue_pairs.dat"],
    start: Annotated[Optional[int], typer.Option("--start")] = None,
    stop: Annotated[Optional[int], typer.Option("--stop")] = None,
    stride: Annotated[int, typer.Option("--stride")] = 1,
    nproc: Annotated[int, typer.Option("-n", "--nproc",
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
        nproc (int): Number of parallel worker processes; defaults to the
            number of logical CPU cores.

    Output:
        ``<pair_file_stem>_distance.dat`` — one row per frame, one column per
        pair.  Header encodes pair labels as ``<segid1>:<resid1>-<segid2>:<resid2>``.

    Example::

            scical distance --top conf.psf --traj system.dcd -f pairs.dat --stride 2 --nproc 8
    """
    _suppress_warnings()
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

    slices   = _make_worker_slices(s, e, st, nproc)
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
    _suppress_warnings()
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
    _suppress_warnings()
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
    _suppress_warnings()
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

    Supports two modes determined by whether the value after ``:`` is an integer
    (count mode) or a glob pattern (pattern mode).  If *spec* is ``None``, all
    segids are placed in a single component labelled ``"A"``.

    **Count mode** (original behaviour) — values after ``:`` are integers;
    segids are consumed sequentially from *segids* in the order they appear::

        "A:12 B:10"   # first 12 segids → A, next 10 → B

    **Pattern mode** — values after ``:`` are ``fnmatch`` glob patterns matched
    against *segids*; useful when copies of different components are interleaved
    in the topology::

        "A:PA* B:PB*"   # all segids matching PA* → A, all matching PB* → B

    Args:
        spec (Optional[str]): Component spec string, e.g. ``"A:12 B:10"`` or
            ``"A:PA* B:PB*"``.  ``None`` places all segids into component ``"A"``.
        segids (list[str]): Ordered list of unique segids detected from the topology.

    Returns:
        dict[str, list[str]]: Mapping ``{label: [segid, ...]}``.

    Raises:
        ValueError: In count mode, if the total count does not match ``len(segids)``.
        ValueError: In pattern mode, if a pattern matches no segids, or if the
            union of all patterns does not cover every segid exactly once.
    """
    if spec is None:
        return {"A": segids}

    # Detect mode: pattern mode if any value after ':' is not a plain integer
    if any(not t.split(":")[1].isdigit() for t in spec.split()):
        import fnmatch
        comp_map = {}
        for token in spec.split():
            label, pattern = token.split(":", 1)
            matched = [s for s in segids if fnmatch.fnmatch(s, pattern)]
            if not matched:
                raise ValueError(f"Pattern '{pattern}' matched no segids.")
            comp_map[label] = matched
        assigned = sum(comp_map.values(), [])
        if set(assigned) != set(segids):
            raise ValueError(
                f"Patterns do not cover all segids. Unassigned: {set(segids) - set(assigned)}"
            )
        return comp_map

    # Count mode (original behaviour)
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
    _suppress_warnings()
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    # Added sel_str to the unpacked arguments
    (top, traj, cutoff, frame_indices,
     atom_to_local, atom_to_copy, atom_to_comp,
     pair_specs, sel_str) = args

    u       = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_hvy = u.select_atoms(sel_str) # Use the dynamic selection

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

        # Filter out garbage indices (unmapped atoms from np.full)
        valid = (li != -1) & (lj != -1)
        li, lj, ci, cj, cmpi, cmpj = li[valid], lj[valid], ci[valid], cj[valid], cmpi[valid], cmpj[valid]

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
                    partials[key]["intra"] += m

                # ── Inter (different copies, same component) ──────────────
                inter_mask = (cmpi == cx) & (cmpj == cx) & (ci != cj)
                if inter_mask.any():
                    k2    = np.unique(np.stack([ci[inter_mask], cj[inter_mask],
                                                li[inter_mask], lj[inter_mask]], axis=1), axis=0)
                    flat2 = k2[:, 2] * N_X + k2[:, 3]
                    m2    = np.bincount(flat2, minlength=N_X * N_Y).reshape(N_X, N_Y).astype(np.float64)
                    partials[key]["inter"] += m2

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
    ref: Annotated[str, typer.Option("--ref", 
        help="Selection string for the reference atom defining a residue sequence (default: 'name CA'). Give the unique atom name for each residue.")] = "name CA",
    sel: Annotated[str, typer.Option("--sel", 
        help="Selection string for atoms involved in the contact calculation.")] = "name CA CB CC CD CE CF",
    start: Annotated[int, typer.Option("--start")] = 0,
    stop: Annotated[Optional[int], typer.Option("--stop")] = None,
    stride: Annotated[int, typer.Option("--stride")] = 1,
    nproc: Annotated[int, typer.Option("--nproc")] = 1,
    out: Annotated[str, typer.Option("--out", help="Output stem")] = "contact_map.npy",
):
    """Calculate intra- and inter-chain **contact maps** for multi-component systems (parallel).

    Identifies contacts between target atoms (--sel) within a user-defined distance cutoff, 
    accounting for periodic boundary conditions. Supports systems containing multiple distinct 
    components (e.g., proteins, lipids, coarse-grained polymers).

    Components are defined by grouping auto-detected segids using a reference atom (--ref)::

        --components "A:100 B:100"   # first 100 segids → A, next 100 → B

    Contact pairs to compute::

        --pairs "A-A A-B B-B"   # default: all same-component pairs

    Normalisation: sum over all copy pairs / (n_copies_X × n_frames), giving the
    average contacts that residue i of one copy makes with residue j per frame.
    Warning: for A-B contacts, if number of copies is not symmetric, the normalization is not symmetric,
    meaning the contact map will not be symmetric A-B != B-A.
    """
    _suppress_warnings()
    import MDAnalysis as mda
    import time
    import multiprocessing as mp
    import numpy as np

    u = mda.Universe(top, traj) if traj else mda.Universe(top)
    
    # ── Component setup using --ref ────────────────────────────────────────
    all_ref = u.select_atoms(ref)
    if len(all_ref) == 0:
        typer.echo(f"[!] No atoms found matching reference selection: '{ref}'", err=True)
        raise typer.Exit(1)
        
    segids = list(dict.fromkeys(all_ref.segids))
    n_traj = len(u.trajectory)

    comp_map    = _parse_components(components, segids)
    comp_labels = list(comp_map.keys())

    req_pairs = _parse_pairs(pairs, comp_labels) if pairs else [(x, x) for x in comp_labels]

    comp_ref = {label: [u.select_atoms(f"segid {s} and ({ref})") for s in segs]
               for label, segs in comp_map.items()}
               
    comp_N  = {}
    for label, ref_list in comp_ref.items():
        Ns = [len(r) for r in ref_list]
        if len(set(Ns)) != 1:
            typer.echo(f"[!] Component {label} has unequal residue counts based on '{ref}': {set(Ns)}", err=True)
            raise typer.Exit(1)
        comp_N[label] = Ns[0]

    comp_idx = {label: i for i, label in enumerate(comp_labels)}

    # ── Lookup arrays using --sel ──────────────────────────────────────────
    all_hvy     = u.select_atoms(sel)
    if len(all_hvy) == 0:
        typer.echo(f"[!] No atoms found matching contact selection: '{sel}'", err=True)
        raise typer.Exit(1)
        
    M           = len(all_hvy)
    all_hvy_idx = all_hvy.indices

    # Initialize with -1 to safely handle atoms that match --sel but aren't in a --ref residue
    atom_to_local = np.full(M, -1, dtype=np.int32)
    atom_to_copy  = np.full(M, -1, dtype=np.int32)
    atom_to_comp  = np.full(M, -1, dtype=np.int32)

    for label, ref_list in comp_ref.items():
        cidx = comp_idx[label]
        for copy_within, ref_atoms in enumerate(ref_list):
            for res_local, res in enumerate(ref_atoms.residues):
                # Find which contact atoms (--sel) belong to this specific residue
                res_hvy_idx = np.intersect1d(res.atoms.indices, all_hvy_idx)
                if len(res_hvy_idx) > 0:
                    rows = np.searchsorted(all_hvy_idx, res_hvy_idx)
                    atom_to_local[rows] = res_local
                    atom_to_copy[rows]  = copy_within
                    atom_to_comp[rows]  = cidx

    comp_resids  = {label: comp_ref[label][0].residues.resids for label in comp_labels}
    comp_ncopies = {label: len(segs) for label, segs in comp_map.items()}

    pair_specs = [
        (comp_idx[x], comp_idx[y], comp_N[x], comp_N[y], comp_ncopies[x], x == y)
        for x, y in req_pairs
    ]

    del u  # Free memory before multiprocessing

    # ── Multiprocessing setup ──────────────────────────────────────────────
    frames   = list(range(start, stop or n_traj, stride))
    n_frames = len(frames)
    nprocs   = min(nproc, n_frames)

    typer.echo(f"Reference  : '{ref}'")
    typer.echo(f"Selection  : '{sel}'")
    typer.echo(f"Components : {', '.join(f'{l}×{comp_ncopies[l]} (N={comp_N[l]})' for l in comp_labels)}")
    typer.echo(f"Pairs      : {', '.join(f'{x}-{y}' for x, y in req_pairs)}")
    typer.echo(f"Cutoff     : {cutoff} Å  ({cutoff / 10:.2f} nm)")
    typer.echo(f"Frames     : {n_frames}  (stride={stride})")
    typer.echo(f"Workers    : {nprocs}")

    chunks      = [frames[i::nprocs] for i in range(nprocs)]
    worker_args = [
        (top, traj, cutoff, chunk, atom_to_local, atom_to_copy, atom_to_comp, pair_specs, sel)
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

import re

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
    _suppress_warnings()
    import MDAnalysis as mda
    return mda.Universe(top, traj) if traj else mda.Universe(top)


# ---------------------------------------------------------------------------
#  Segment-name parsing
# ---------------------------------------------------------------------------

def _parse_segname(name: str) -> tuple:
    """Split a segment name into (prefix, number, zero-pad width).

    The segment name must consist of an alphabetic prefix followed by a
    numeric suffix.  Names beginning with digits, containing hyphens, or
    with interleaved letters and digits (e.g. ``"H2O1"``) are not supported.

    Args:
        name (str): Segment name such as ``"R001"`` or ``"PRO099"``.

    Returns:
        tuple[str, int, int]:

        - **prefix** — Alphabetic prefix (e.g. ``"R"``).
        - **number** — Integer value of the numeric suffix.
        - **width** — Character width of the numeric suffix (for zero-padding).

    Raises:
        ValueError: If *name* does not match the expected pattern.

    Examples::

        _parse_segname("R001")    # → ("R", 1, 3)
        _parse_segname("PRO099")  # → ("PRO", 99, 3)
        _parse_segname("K10")     # → ("K", 10, 2)
    """
    match = re.match(r'^([A-Za-z]+)(\d+)$', name)
    if not match:
        raise ValueError(
            f"Segment name must be alphabetic prefix + numeric suffix "
            f"(e.g. 'R001', 'PRO099'), got: {name!r}"
        )
    prefix  = match.group(1)
    num_str = match.group(2)
    return prefix, int(num_str), len(num_str)


def _expand_range(start_name: str, end_name: str) -> list:
    """Expand a segment-name range into a list of segment name strings.

    Both endpoints must share the same alphabetic prefix.  The zero-padding
    width is taken from the **start** endpoint and applied uniformly to all
    generated names.  Python's format specification naturally handles numbers
    whose digit count exceeds the width (no truncation occurs).

    A consistency check verifies that the generated name for the end number
    matches the provided *end_name*; a mismatch indicates inconsistent
    zero-padding between endpoints.

    Args:
        start_name (str): First segment name in the range (e.g. ``"R001"``).
        end_name (str): Last segment name in the range (e.g. ``"R090"``).

    Returns:
        list[str]: Expanded segment names, inclusive of both endpoints.

    Raises:
        ValueError: If prefixes differ, start > end, or padding is
            inconsistent between endpoints.

    Examples::

        _expand_range("R001", "R090")  # → ["R001", "R002", …, "R090"]
        _expand_range("R01", "R09")    # → ["R01", "R02", …, "R09"]
        _expand_range("R1", "R20")     # → ["R1", "R2", …, "R9", "R10", …, "R20"]
        _expand_range("R1", "R9")      # → ["R1", "R2", …, "R9"]
    """
    start_prefix, start_num, start_width = _parse_segname(start_name)
    end_prefix,   end_num,   end_width   = _parse_segname(end_name)

    if start_prefix != end_prefix:
        raise ValueError(
            f"Range endpoints must share the same prefix: "
            f"'{start_name}' (prefix='{start_prefix}') vs "
            f"'{end_name}' (prefix='{end_prefix}')"
        )

    if start_num > end_num:
        raise ValueError(
            f"Range start > end: '{start_name}' > '{end_name}'"
        )

    # Validate that end_name matches what we would generate using start_width.
    # f"{num:0{width}d}" pads to at least `width` digits but never truncates.
    generated_end = f"{start_prefix}{end_num:0{start_width}d}"
    if generated_end != end_name:
        raise ValueError(
            f"Inconsistent zero-padding in range '{start_name}-{end_name}': "
            f"start uses width {start_width}, which would produce "
            f"'{generated_end}' for the end number, not '{end_name}'. "
            f"Use '{start_name}-{generated_end}' or "
            f"'{start_prefix}{start_num:0{end_width}d}-{end_name}'."
        )

    return [f"{start_prefix}{num:0{start_width}d}"
            for num in range(start_num, end_num + 1)]


def _parse_sel(sel: Optional[str], u) -> list:
    """Parse segment-name range string(s) into universe segment indices.

    Supports:

    - Single range: ``"R001-R099"``
    - Comma-separated ranges: ``"R001-R099,K001-K010"``
    - Single segment name: ``"R001"``
    - Comma-separated mix: ``"R001-R099,K001,K005"``

    Each range endpoint must consist of an alphabetic prefix followed by a
    zero-padded numeric suffix.  Both endpoints of a single range must share
    the same prefix.

    Segments in the specified range that do not exist in the universe are
    silently skipped (a warning is emitted if any are missing).

    Args:
        sel (Optional[str]): Segment-name range string, or ``None`` to select
            all segments.
        u (MDAnalysis.Universe): Loaded Universe (used to look up segids).

    Returns:
        list[int]: Sorted list of zero-based universe segment indices.

    Raises:
        ValueError: If *sel* is malformed, prefixes mismatch within a range,
            or no segments match.

    Examples::

        _parse_sel("R001-R099", u)            # segments R001 … R099
        _parse_sel("R001-R099,K001-K010", u)  # two ranges combined
        _parse_sel("R001", u)                 # single segment
        _parse_sel(None, u)                   # all segments
    """
    if sel is None:
        return list(range(len(u.segments)))

    # Build lookup: segid → universe segment index.
    segid_to_idx: dict = {}
    for i, seg in enumerate(u.segments):
        segid_to_idx[seg.segid] = i

    selected_indices: list = []
    missing_names:    list = []

    tokens = [t.strip() for t in sel.split(",")]
    for token in tokens:
        if not token:
            continue

        # Split on '-' to detect ranges vs single names.
        parts = token.split("-")

        if len(parts) == 2:
            # Range: "R001-R099"
            left  = parts[0].strip()
            right = parts[1].strip()
            names = _expand_range(left, right)
        elif len(parts) == 1:
            # Single segment name: "R001"
            _parse_segname(token)  # validate format; raises on failure
            names = [token]
        else:
            # More than one '-' — ambiguous
            raise ValueError(
                f"Ambiguous token '{token}': use comma-separated ranges "
                f"(e.g. 'R001-R099,K001-K010'), not 'R001-R099-K001-K010'."
            )

        for name in names:
            if name in segid_to_idx:
                selected_indices.append(segid_to_idx[name])
            else:
                missing_names.append(name)

    if missing_names:
        n_missing = len(missing_names)
        sample    = missing_names[:5]
        typer.echo(
            f"[w] {n_missing} segment name(s) in --sel not found in topology "
            f"(skipped). First few: {sample}",
            err=True,
        )

    if not selected_indices:
        available = sorted(segid_to_idx.keys())[:10]
        raise ValueError(
            f"No segments matched --sel '{sel}'. "
            f"Available segids (first 10): {available}"
        )

    return sorted(set(selected_indices))


def _grp_init(u, ref_atom: str = "CA", step: int = 1,
              sel_indices: Optional[list] = None):
    """Initialise per-segment reference-atom groups for clustering.

    Only the segments listed in *sel_indices* are included in the returned
    groups.  The caller uses the returned *sel_indices* to translate a local
    cluster index (position within *segs*) back to a universe-level segment
    index, which is what :func:`_recenter_frame` and :func:`_unwrap_cluster`
    expect.

    Args:
        u (MDAnalysis.Universe): Loaded Universe.
        ref_atom (str): Name of the reference atom (default: ``"CA"``).
        step (int): Sub-sampling step applied within each segment's reference
            atoms.
        sel_indices (Optional[list[int]]): Universe-level segment indices to
            include in clustering.  ``None`` selects every segment.

    Returns:
        tuple[list[AtomGroup], np.ndarray, list[int]]:

        - **segs** — List of ``AtomGroup`` objects, one per *selected* segment.
        - **grps** — Integer array of initial group IDs, shape
          ``(n_selected,)``.
        - **sel_indices** — The resolved universe-level segment indices.
          ``sel_indices[i]`` is the universe index of ``segs[i]``.
    """
    if sel_indices is None:
        sel_indices = list(range(len(u.segments)))

    segs, grps = [], []
    empty: list = []                         # segment indices with no ref atoms
    for i, seg_idx in enumerate(sel_indices):
        segid = u.segments[seg_idx].segid
        cas   = u.select_atoms(f"segid {segid} and name {ref_atom}")
        seg   = cas[::step].atoms.select_atoms(f"name {ref_atom}")
        if len(seg) == 0:
            empty.append(seg_idx)
        segs.append(seg)
        grps.append(i)

    if empty:
        n_empty      = len(empty)
        sample       = empty[:5]
        sample_names = [u.segments[idx].segid for idx in sample]
        raise ValueError(
            f"{n_empty} of the {len(sel_indices)} selected segment(s) contain no "
            f"atoms named '{ref_atom}' (segids: {sample_names}"
            f"{'…' if n_empty > 5 else ''}).\n"
            f"  • Check --ref (current: '{ref_atom}'). For proteins use 'CA'; "
            f"for nucleic acids / lipids use 'P'.\n"
            f"  • Check --sel: the selected range may include the wrong molecule type."
        )

    return segs, np.array(grps), sel_indices


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
          **local** indices into *segs* (i.e. positions within the list
          passed in, not universe segment indices).
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

def _format_clusters_line(frame: int, clusters: list, sel_indices: list, u) -> str:
    """Format one frame's cluster composition as a single pipe-delimited line.

    Format::

        <frame> | <size_list> | <count_list> | <details>

    - **size_list** — unique cluster sizes, descending, comma-separated
      (e.g. ``6,5,4,1``).
    - **count_list** — number of clusters at each size, comma-separated,
      positionally aligned with *size_list* (e.g. ``1,2,2,50`` means
      1 cluster of size 6, 2 of size 5, 2 of size 4, 50 of size 1).
    - **details** — one group per unique size (same order as *size_list*),
      groups joined by ``-``. Within a group:

        * size > 1: multiple clusters of that size are separated by ``;``,
          and segids within a single cluster are comma-separated, e.g.
          ``R011,R013,R044,R055,R061;R002,R009,R018,R022,R030`` for two
          size-5 clusters.
        * size == 1 (monomers): every monomer segid is joined directly by
          ``-`` (not comma), e.g. ``R001-R002-R003``, since each monomer
          is a singleton cluster and the per-cluster comma grouping would
          be redundant.

    Args:
        frame (int): Trajectory frame number (``ts.frame``).
        clusters (list[list[int]]): Clusters as **local** indices into the
            clustered segment list (as returned by
            :func:`_find_clusters_and_stats`).
        sel_indices (list[int]): Universe-level segment indices used for
            clustering; ``sel_indices[i]`` is the universe index of local
            index ``i``.
        u (MDAnalysis.Universe): Universe used to resolve segids.

    Returns:
        str: The formatted line (no trailing newline).

    Examples::

        # 1 cluster of size 6, 2 of size 5, 2 of size 4, 50 monomers
        "12 | 6,5,4,1 | 1,2,2,50 | R011,R013,R044,R055,R061,R017-" \\
        "R002,R009,R018,R022,R030;R014,R016,R019,R023,R031-" \\
        "R004,R006,R008,R026;R012,R015,R020,R027-" \\
        "R001-R003-R005-R007-...-R099"
    """
    # Sort clusters largest-first.
    sorted_clusters = sorted(clusters, key=len, reverse=True)

    # Group clusters by size, preserving descending order of first appearance.
    size_to_clusters: dict = {}
    order: list = []
    for cluster in sorted_clusters:
        sz = len(cluster)
        if sz not in size_to_clusters:
            size_to_clusters[sz] = []
            order.append(sz)
        size_to_clusters[sz].append(cluster)

    size_list  = order                                     # already descending
    count_list = [len(size_to_clusters[sz]) for sz in size_list]

    detail_groups = []
    for sz in size_list:
        group_clusters = size_to_clusters[sz]
        if sz == 1:
            # Monomers: dash-join every segid directly.
            segids = [u.segments[sel_indices[c[0]]].segid for c in group_clusters]
            detail_groups.append("-".join(segids))
        else:
            # Multiple clusters of this size: comma-join segids within a
            # cluster, semicolon-join multiple clusters of the same size.
            cluster_strs = [
                ",".join(u.segments[sel_indices[i]].segid for i in c)
                for c in group_clusters
            ]
            detail_groups.append(";".join(cluster_strs))

    size_str    = ",".join(str(s) for s in size_list)
    count_str   = ",".join(str(c) for c in count_list)
    details_str = "-".join(detail_groups)

    return f"{frame} | {size_str} | {count_str} | {details_str}"

# ---------------------------------------------------------------------------
#  PBC recentering
# ---------------------------------------------------------------------------

def _unwrap_cluster(u, cluster_seg_indices: list, box: np.ndarray) -> dict:
    """Unwrap a cluster split across periodic boundaries using BFS.

    Args:
        u (MDAnalysis.Universe): Universe at the current trajectory frame.
        cluster_seg_indices (list[int]): **Universe-level** indices of segments
            in the cluster.
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

    Always operates on **all** atoms in the universe — the selection only
    determines which segments define the centre-of-mass reference.

    Args:
        u (MDAnalysis.Universe): Universe at the current trajectory frame.
        largest_cluster (list[int]): **Universe-level** segment indices of the
            largest cluster.
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

    Always operates on **all** atoms matching *ref_atom* in the universe.

    Args:
        u (MDAnalysis.Universe): Universe at the current (recentered) frame.
        droplet_center (np.ndarray): 3-D Cartesian coordinates of the droplet
            centre, in ångström.
        ca_per_segment (int): Number of reference atoms per monomer (derived
            from the selected species, not the full universe).
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


def _slab_density(u, droplet_center: np.ndarray, ca_per_segment: int,
                   ref_atom: str, r_max: float, dr: float,
                   omitted_axis: int, box: np.ndarray) -> tuple:
    """Compute the 1-D slab monomer concentration profile for a flat (2D)
    aggregate — e.g. a membrane-bound or interfacially confined slab.

    Mirrors :func:`_radial_density`, but bins the *signed* displacement of
    ``ref_atom`` positions along a single axis (the one *omitted* from the
    plane of the slab) instead of the radial distance in 3-D.  Minimum-image
    convention is applied along that axis so PBC-wrapped atoms are still
    binned relative to the (possibly re-centred) droplet/slab centre.

    Always operates on **all** atoms matching *ref_atom* in the universe.

    Args:
        u (MDAnalysis.Universe): Universe at the current (recentered) frame.
        droplet_center (np.ndarray): 3-D Cartesian coordinates of the slab
            centre, in ångström (typically the box centre after recentering).
        ca_per_segment (int): Number of reference atoms per monomer.
        ref_atom (str): Reference atom name (e.g. ``"CA"``).
        r_max (float): Half-range of the profile, in ångström.  Bins span
            ``[-r_max, +r_max]`` relative to *droplet_center*.
        dr (float): Bin width, in ångström.
        omitted_axis (int): Axis index (0=X, 1=Y, 2=Z) along which the
            profile is computed — i.e. the axis normal to the slab plane.
        box (np.ndarray): Current frame's box dimensions, shape ``(6,)``
            (``[Lx, Ly, Lz, alpha, beta, gamma]``), used both for PBC
            wrapping along *omitted_axis* and for the in-plane cross-section
            area used to normalise counts into a volumetric concentration.

    Returns:
        tuple[np.ndarray, np.ndarray]:

        - **r_bins** — Signed bin centres relative to *droplet_center* along
          *omitted_axis*, shape ``(n_bins,)``, in ångström.
        - **concentration** — Monomer concentration per slab bin, in mM.
    """
    all_ca = u.select_atoms(f"name {ref_atom}")
    values = all_ca.positions[:, omitted_axis] - droplet_center[omitted_axis]

    # Minimum-image wrap along the profiled axis only.
    L = box[omitted_axis]
    if L > 0:
        values = values - L * np.round(values / L)

    n_bins = int(round(2 * r_max / dr))
    half   = n_bins * dr / 2.0
    bins   = np.linspace(-half, half, n_bins + 1)
    r_bins = 0.5 * (bins[:-1] + bins[1:])

    ca_counts, _   = np.histogram(values, bins=bins)
    monomer_counts = ca_counts / ca_per_segment

    # In-plane cross-section area (product of the two non-omitted axes).
    in_plane_axes = [ax for ax in range(3) if ax != omitted_axis]
    area          = box[in_plane_axes[0]] * box[in_plane_axes[1]]
    slab_vols     = area * dr
    concentration = (monomer_counts / slab_vols) * (1e30 / AVOGADRO)

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
    sel_indices: list,           # universe-level segment indices used for clustering
    save_clusters: bool,         # whether to record per-frame cluster-details lines
    slab: bool = False,          # slab (1D, flat-aggregate) density mode
    omitted_axis: Optional[int] = None,  # axis index normal to the slab plane
) -> dict:
    """Analyse a contiguous subset of frames in a subprocess.

    Each worker opens its own Universe — ``mda.Universe`` is not picklable
    and cannot be shared across processes.  Frames are accessed by absolute
    trajectory index so that any start/stop/stride combination is handled
    correctly regardless of how frame numbers are stored in the XTC.

    Clustering is restricted to *sel_indices*.  Recentering and density
    always act on the **full** universe.

    Args:
        top (str): Topology file path.
        traj (str): Trajectory file path.
        traj_indices (list[int]): Absolute trajectory indices to process.
        ref (str): Reference atom name.
        rcut (float): Clustering distance cutoff (Å).
        castep (int): Sub-sampling step for reference atoms.
        recenter (bool): Whether to unwrap and recenter the largest cluster.
        density (bool): Whether to compute density profiles (radial or slab).
        dr (float): Radial/slab bin width (Å).
        n_frames_avg (int): Number of final frames to include in density avg.
        n_frames_total (int): Total analysed frames (all workers combined).
        chunk_start_rank (int): Global rank of this chunk's first frame.
        tmp_traj (Optional[str]): Path to write the recentered trajectory
            fragment, or ``None`` if recentering is disabled.
        sel_indices (list[int]): Universe-level segment indices to use for
            clustering.  Passed straight through to :func:`_grp_init`.
        save_clusters (bool): Whether to build a formatted cluster-details
            line for each frame (see :func:`_format_clusters_line`).
        slab (bool): If ``True``, compute a 1-D slab density profile along
            *omitted_axis* (via :func:`_slab_density`) instead of the default
            3-D spherical radial profile (via :func:`_radial_density`).
        omitted_axis (Optional[int]): Axis index (0=X, 1=Y, 2=Z) normal to
            the slab plane.  Required when *slab* is ``True``.

    Returns:
        dict:
            ``stats_data``       — list of ``(frame_number, monomers, clusters, max_size)``
            ``density_profiles`` — list of ``(global_rank, r_bins, conc)``
            ``cluster_lines``    — list of ``(frame_number, line)``
    """
    _suppress_warnings()
    import MDAnalysis as mda

    u                    = _load_universe(top, traj)
    segs, _, sel_indices = _grp_init(u, ref_atom=ref, step=castep,
                                     sel_indices=sel_indices)

    # Use the first *selected* segment so ca_per_segment reflects the correct
    # molecule type even when sel_indices does not start at 0.
    ca_per_segment = len(
        u.segments[sel_indices[0]].atoms.select_atoms(f"name {ref}")
    )

    stats_data:       list = []
    density_profiles: list = []
    cluster_lines:    list = []

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

            # ── Clustering: selected segments only ─────────────────────────
            # clusters contains lists of LOCAL indices (0…len(segs)-1).
            clusters, monomer, n_clusters, max_size, _ = _find_clusters_and_stats(
                segs, grps_frame, r_cutoff=rcut
            )

            if save_clusters:
                cluster_lines.append((
                    ts.frame,
                    _format_clusters_line(ts.frame, clusters, sel_indices, u),
                ))

            if recenter:
                # Translate local cluster indices → universe-level segment
                # indices.  _recenter_frame / _unwrap_cluster both call
                # u.segments[seg_idx] directly and must receive universe
                # indices.  The functions themselves always move ALL atoms.
                largest_local = max(clusters, key=len)
                largest_univ  = [sel_indices[i] for i in largest_local]

                # ── Recentering: full universe ─────────────────────────────
                u.atoms.positions = _recenter_frame(u, largest_univ, box)
                if writer:
                    writer.write(u.atoms)

                if density:
                    global_rank = chunk_start_rank + local_rank
                    if global_rank >= n_frames_total - n_frames_avg:
                        # ── Density: full universe ─────────────────────────
                        if slab:
                            r_max = float(box[omitted_axis] / 2.0)
                            r_bins, conc = _slab_density(
                                u,
                                droplet_center=box[:3] / 2.0,
                                ca_per_segment=ca_per_segment,
                                ref_atom=ref,
                                r_max=r_max,
                                dr=dr,
                                omitted_axis=omitted_axis,
                                box=box,
                            )
                        else:
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
        "cluster_lines":    cluster_lines,
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
    _suppress_warnings()
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
              help="Compute density profile — spherical or slab (requires --recenter)")] = False,
    dimen:    Annotated[str, typer.Option(
                  "--dimen",
                  help=(
                      "'xyz' (default) = 3D spherical droplet profile; "
                      "'xy'/'xz'/'yz' = profile along the omitted axis when "
                      "--slab is given (has no effect without --slab)."
                  ),
              )] = "xyz",
    slab:     Annotated[bool, typer.Option(
                  "--slab/--no-slab",
                  help=(
                      "Treat the aggregate as a flat **slab** (e.g. a "
                      "membrane-bound or interfacially confined layer) "
                      "instead of a 3D droplet.  Requires a 2D --dimen "
                      "(xy, xz, or yz); profiles monomer concentration "
                      "along the omitted axis, symmetric about the "
                      "(recentered) box centre: -rmax … +rmax.  "
                      "  --dimen xy --slab  →  profile along Z  "
                      "  --dimen xz --slab  →  profile along Y  "
                      "  --dimen yz --slab  →  profile along X"
                  ),
              )] = False,
    dr:           Annotated[float, typer.Option("--dr",           help="Radial/slab bin width (Å)")] = 2.0,
    n_frames_avg: Annotated[int,   typer.Option("--n-frames-avg", help="Last N frames averaged for density profile")] = 50,
    nproc:        Annotated[int,   typer.Option("--nproc",        help="Number of parallel worker processes (1 = serial)")] = 1,
    sel:          Annotated[Optional[str], typer.Option(
                      "--sel",
                      help=(
                          "Segment name range(s) for clustering, inclusive. "
                          "Each token is 'PREFIXSTART-PREFIXEND' or a single "
                          "segment name. Multiple tokens are comma-separated. "
                          "Examples: 'R001-R099', 'R001-R099,K001-K010', "
                          "'R001'. Only matched segments contribute to "
                          "aggregation statistics. Recentering and density "
                          "profiles always use the full system. "
                          "Default: all segments."
                      ),
                  )] = None,
    clusters:     Annotated[bool, typer.Option(
                      "--clusters/--no-clusters",
                      help="Save per-frame cluster composition (size + segids) to --clusterdat",
                  )] = False,
    clusterdat:   Annotated[str, typer.Option(
                      "--clusterdat",
                      help="Cluster-details output file",
                  )] = "cluster_details.dat",
):
    """Analyse protein **aggregation**: cluster detection, PBC recentering, and
    radial *or slab* density.

    Processes each trajectory frame to:

    1. **Cluster** — detect multi-chain aggregates using kd-tree contact search
       on reference atoms (default: Cα), with a union-find merging scheme.
       Only the segments selected by ``--sel`` are considered.
    2. **Recenter** *(optional, ``--recenter``)* — unwrap the largest cluster
       across PBC and translate it to the box centre; write as a new trajectory.
       Always applied to the **full** system.
    3. **Density profile** *(optional, ``--density``, requires ``--recenter``)* —
       compute and average the monomer concentration (mM) over the last
       ``--n-frames-avg`` frames.  Always computed from the **full** system.
       Two profile modes are available, selected via ``--dimen``/``--slab``:

       - **Spherical droplet** *(default, ``--dimen xyz``)* — radial monomer
         concentration around the (recentered) box centre, normalised by
         spherical shell volume.  Suited to a compact 3D droplet.
       - **Flat slab** *(``--dimen xy|xz|yz --slab``)* — 1D monomer
         concentration along the axis omitted from ``--dimen`` (i.e. normal
         to the slab plane), symmetric about the box centre
         (``-rmax … +rmax``), normalised by the average in-plane
         cross-section area × ``--dr``.  Suited to a flat, membrane-like, or
         interfacially confined aggregate that spans the box in two
         dimensions.

    Parallelism is over frames (``--nproc``): each worker process handles a
    contiguous chunk of the trajectory and writes its own temporary XTC fragment,
    which are merged in order at the end.

    Example::

        # Serial — cluster statistics only, select by segment name
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 --sel R001-R099

        # Also save per-frame cluster composition details
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 --sel R001-R099 \\
            --clusters --clusterdat cluster_details.dat

        # Multiple segment types
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 \\
            --sel R001-R099,K001-K010

        # Single segment
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 --sel R001

        # 8 workers — with PBC recentering and spherical droplet density
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 --sel R001-R099 \\
            --recenter --density --dr 2.0 --n-frames-avg 100 --nproc 8

        # Flat slab (e.g. membrane-bound aggregate) — profile along Z
        scical aggr --top conf.psf --traj system.xtc --rcut 8.0 --sel R001-R099 \\
            --recenter --density --dimen xy --slab \\
            --dr 2.0 --n-frames-avg 100 --nproc 8
    """
    _suppress_warnings()
    import MDAnalysis as mda

    # The CLI flag is named `clusters`, but the per-frame cluster-membership
    # variable inside the loops below is also named `clusters` (a list).
    # Rename here to avoid shadowing.
    save_clusters = clusters

    # Guard: density requires recentering
    if density and not recenter:
        typer.echo("[!] --density requires --recenter — density disabled.", err=True)
        density = False

    # ── Validate --dimen / --slab (mirrors the `density` command) ──────────
    dimen_lower = dimen.lower()
    if dimen_lower not in _DIMEN_AXES:
        typer.echo(
            f"[!] Invalid --dimen '{dimen}'. Must be one of: xyz, xy, xz, yz.",
            err=True,
        )
        raise typer.Exit(1)

    is_3d = len(_DIMEN_AXES[dimen_lower]) == 3

    if slab and is_3d:
        typer.echo(
            "[!] --slab requires a 2D --dimen (xy, xz, or yz). "
            "For --dimen xyz there is no omitted axis.",
            err=True,
        )
        raise typer.Exit(1)

    if not slab and not is_3d:
        typer.echo(
            f"[!] --dimen {dimen_lower} without --slab is not supported for "
            f"aggregation density — 2D in-plane profiles have no well-defined "
            f"volumetric (mM) normalisation here. Add --slab for a slab "
            f"profile along the omitted axis, or use --dimen xyz for the "
            f"default spherical droplet profile.",
            err=True,
        )
        raise typer.Exit(1)

    omitted_axis = _DIMEN_OMITTED.get(dimen_lower)  # None for xyz

    typer.echo("Loading trajectory...")
    u = _load_universe(top, traj)

    # ── Resolve segment selection ──────────────────────────────────────────
    # sel_indices is a list of universe-level segment indices.
    # All downstream code uses this list to restrict clustering; recentering
    # and density always operate on the full universe.
    try:
        sel_indices = _parse_sel(sel, u)
    except ValueError as exc:
        typer.echo(f"[!] {exc}", err=True)
        raise typer.Exit(1)

    # Build the list of absolute trajectory indices to analyse.
    # Using indices (not XTC frame numbers) guarantees correct random-access
    # seeking even when the XTC stores non-contiguous or non-zero-based frame
    # numbers.
    _start       = start if start is not None else 0
    _stop        = stop  if stop  is not None else len(u.trajectory)
    traj_indices = list(range(_start, _stop, stride))
    n_frames     = len(traj_indices)

    # Initialise segment groups once from the selected segments.
    # _grp_init raises ValueError if any selected segment has no ref atoms,
    # which means --ref or --sel is wrong; surface that as a clean message.
    try:
        segs, _, sel_indices = _grp_init(u, ref_atom=ref, step=castep,
                                         sel_indices=sel_indices)
    except ValueError as exc:
        typer.echo(f"[!] {exc}", err=True)
        raise typer.Exit(1)

    # ca_per_segment: count ref atoms on the first selected segment.
    # _grp_init already guaranteed this is > 0, so no further guard needed.
    ca_per_segment = len(
        u.segments[sel_indices[0]].atoms.select_atoms(f"name {ref}")
    )

    # Display selected segids for clarity
    first_segid = u.segments[sel_indices[0]].segid
    last_segid  = u.segments[sel_indices[-1]].segid

    typer.echo(f"[i] Segments (total)   : {len(u.segments)}")
    typer.echo(f"[i] Segments (cluster) : {len(sel_indices)}  "
               f"({first_segid} … {last_segid})")
    typer.echo(f"[i] {ref} / segment       : {ca_per_segment}")
    typer.echo(f"[i] Frames             : {n_frames}")
    typer.echo(f"[i] Cutoff             : {rcut} Å")
    typer.echo(f"[i] nproc              : {nproc}")
    typer.echo(f"[i] Recenter           : {recenter}")
    typer.echo(f"[i] Density            : {density}")
    if density:
        axis_names = ["X", "Y", "Z"]
        mode_label = (
            f"slab along {axis_names[omitted_axis]}-axis"
            if slab else "3D spherical droplet"
        )
        typer.echo(f"[i] Density mode       : {mode_label}")
    typer.echo(f"[i] Cluster details    : {save_clusters}")

    # Initialise output accumulators before either branch so the common output
    # section below is always guaranteed to find them defined.
    stats_data:       list = []   # list of (frame_number, monomers, clusters, max_size)
    density_profiles: list = []   # list of (r_bins, conc) — unified format for both paths
    cluster_lines:    list = []   # list of (frame_number, line) — unified format for both paths

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

                # ── Clustering: selected segments only ─────────────────────
                # clusters contains lists of LOCAL indices (0…len(segs)-1).
                clusters, monomer, n_clusters, max_size, _ = _find_clusters_and_stats(
                    segs, grps_frame, r_cutoff=rcut
                )

                if ts_idx % 10 == 0:
                    typer.echo(
                        f"    monomers={monomer}  clusters={n_clusters}  largest={max_size}"
                    )

                if save_clusters:
                    cluster_lines.append((
                        ts.frame,
                        _format_clusters_line(ts.frame, clusters, sel_indices, u),
                    ))

                if recenter:
                    # Translate local cluster indices → universe-level segment
                    # indices.  _recenter_frame / _unwrap_cluster both call
                    # u.segments[seg_idx] directly and must receive universe
                    # indices.  The functions themselves always move ALL atoms.
                    largest_local = max(clusters, key=len)
                    largest_univ  = [sel_indices[i] for i in largest_local]

                    # ── Recentering: full universe ──────────────────────────
                    u.atoms.positions = _recenter_frame(u, largest_univ, box)
                    writer.write(u.atoms)

                    if density and ts_idx >= n_frames - n_frames_avg:
                        # ── Density: full universe ──────────────────────────
                        if slab:
                            r_max = float(box[omitted_axis] / 2.0)
                            r_bins, conc = _slab_density(
                                u,
                                droplet_center=box[:3] / 2.0,
                                ca_per_segment=ca_per_segment,
                                ref_atom=ref,
                                r_max=r_max,
                                dr=dr,
                                omitted_axis=omitted_axis,
                                box=box,
                            )
                        else:
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
                    sel_indices,          # universe-level indices for clustering
                    save_clusters,        # whether to record cluster-details lines
                    slab,                 # slab (1D flat-aggregate) density mode
                    omitted_axis,         # axis index normal to the slab plane
                )
                futures[fut] = chunk

            raw_stats:    list = []
            raw_density:  list = []
            raw_clusters: list = []

            for fut in as_completed(futures):
                result = fut.result()
                raw_stats.extend(result["stats_data"])
                raw_density.extend(result["density_profiles"])
                raw_clusters.extend(result["cluster_lines"])

        # Restore frame order — as_completed() returns in completion order,
        # not submission order.
        stats_data    = sorted(raw_stats, key=lambda x: x[0])
        cluster_lines = sorted(raw_clusters, key=lambda x: x[0])

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

    if save_clusters and cluster_lines:
        typer.echo(f"Writing cluster details -> {clusterdat}")
        with open(clusterdat, "w") as fh:
            fh.write("# frame | cluster sizes (largest-first) | counts per size | details\n")
            fh.write(
                "# details: size groups separated by '-'; within a size group, "
                "multiple clusters separated by ';', segids within a cluster "
                "comma-separated; monomers (size=1) are dash-joined directly.\n"
            )
            for _, line in cluster_lines:
                fh.write(line + "\n")

    if density and density_profiles:
        # Truncate to the shortest profile before stacking.  r_max varies per
        # frame (clipped to half the box side), so array lengths can differ by
        # one bin between frames.
        min_len      = min(len(c) for _, c in density_profiles)
        r_bins_out   = density_profiles[0][0][:min_len]
        profiles_arr = np.array([c[:min_len] for _, c in density_profiles])
        avg_conc     = np.mean(profiles_arr, axis=0)
        std_conc     = np.std( profiles_arr, axis=0)

        if slab:
            axis_names   = ["X", "Y", "Z"]
            prof_label   = f"slab along {axis_names[omitted_axis]}-axis"
            col1_label   = f"{axis_names[omitted_axis]}(A)"
            header_desc  = (
                f"# Slab monomer concentration along {axis_names[omitted_axis]}-axis "
                f"averaged over last {len(density_profiles)} frames\n"
            )
        else:
            prof_label   = "spherical droplet"
            col1_label   = "Radius(A)"
            header_desc  = (
                f"# Radial monomer concentration averaged over last "
                f"{len(density_profiles)} frames\n"
            )

        typer.echo(
            f"Writing {prof_label} density profile "
            f"({len(density_profiles)} frames) -> {profile}"
        )
        with open(profile, "w") as fh:
            fh.write(header_desc)
            fh.write(f"# {col1_label}  Concentration(mM)  StdDev(mM)\n")
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

    Accepts comma-separated integers and inclusive dash-ranges:

    ==================  ==========================
    Input               Result
    ==================  ==========================
    ``"3"``             ``[3]``
    ``"0-4"``           ``[0, 1, 2, 3, 4]``
    ``"0-2,5-7"``       ``[0, 1, 2, 5, 6, 7]``
    ``"1,3,5"``         ``[1, 3, 5]``
    ==================  ==========================

    Args:
        spec (str): Raw selection string from ``--segments``.
        max_index (int): Number of available segments; valid indices are
            ``0 … max_index − 1``.

    Returns:
        list[int]: Sorted, deduplicated list of valid integer indices.

    Raises:
        ValueError: If any index is outside ``[0, max_index)`` or a range
            token is syntactically malformed.
    """
    indices = set()
    for token in spec.split(","):
        token    = token.strip()
        # Search from position 1 so a hypothetical leading '-' is not
        # mistaken for a range separator (it would be caught by the bounds
        # check below, but the parse would be wrong).
        dash_pos = token.find("-", 1)
        if dash_pos != -1:
            lo_str, hi_str = token[:dash_pos], token[dash_pos + 1:]
            if not lo_str or not hi_str:
                raise ValueError(f"Invalid range token: {token!r}")
            lo, hi = int(lo_str), int(hi_str)
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
    """Compute window-averaged S²(τ) for one residue's backbone N-H bond vectors.

    **Formula**

    For a window starting at origin ``t₀`` with length τ:

    .. code-block:: text

        M(t₀) = (1/τ) · Σ_{t=t₀}^{t₀+τ-1}  v(t) ⊗ v(t)   [3×3 mean tensor]

        S²(t₀) = ( 3 · ‖M(t₀)‖²_F  −  1 ) / 2

        S²(τ)  = mean over all origins t₀ ∈ [0, n_frames − τ)

    ``‖·‖²_F`` is the Frobenius norm squared (= ``Σᵢⱼ <μᵢ μⱼ>²`` in
    Lipari-Szabo notation).  Every frame of the trajectory participates in at
    least one window for every τ ≤ ``n_frames // 2``.

    **Efficiency — prefix-sum approach**

    Naïve window averaging is O(n_origins · τ) per lag (must sum τ frames for
    each of the n_origins windows).  Here:

    1. Normalise all frame vectors **once** (outside the τ loop).
    2. Compute the 3×3 outer product ``P[t] = v(t) ⊗ v(t)`` for every frame,
       flattened to shape ``(n_frames, 9)``.
    3. Build a prefix-sum table ``cumP`` (shape ``(n_frames+1, 9)``) so that::

           Σ_{t=t₀}^{t₀+τ-1} P[t]  =  cumP[t₀+τ] − cumP[t₀]

    4. For a given τ, obtain all ``n_origins`` window sums with **one NumPy
       slice** — no Python loop over origins::

           M_sum = cumP[τ : τ+n_origins] − cumP[:n_origins]   # (n_origins, 9)

    Complexity: O(n_frames) pre-processing + O(n_origins) per τ.
    The speed-up vs. the naïve approach scales with τ; it is largest at the
    long lags that dominate total runtime.

    Args:
        vectors (np.ndarray): Raw (un-normalised) N-H bond vectors,
            shape ``(n_frames, 3)``.  Normalisation is performed internally.
        lag_frames (list[int]): Window lengths τ in frames.  Every entry must
            satisfy ``2 ≤ τ ≤ n_frames``; entries outside this range receive
            ``nan`` in the output.

    Returns:
        np.ndarray: S²(τ) values, shape ``(len(lag_frames),)``, dtype
        ``float64``, clipped to ``[0, 1]``.  ``nan`` for out-of-range τ.
    """
    n = len(vectors)

    # ── Pre-computation (done once, outside the τ loop) ──────────────────

    # Normalise every frame vector once; per-frame norms are independent of
    # which window the frame belongs to, so this never needs to be repeated.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)   # (n, 1)
    v     = vectors / norms                                   # (n, 3) unit vecs

    # Outer products P[t] = v(t) ⊗ v(t), flattened to (n, 9).
    # float64 for numerical safety when summing thousands of terms.
    P = (v[:, :, None] * v[:, None, :]).reshape(n, 9).astype(np.float64)

    # Prefix-sum table: cumP[k] = Σ_{t=0}^{k-1} P[t], cumP[0] = 0.
    # Shape (n+1, 9); cumP[n] is the sum over all frames (used at τ→n).
    cumP    = np.empty((n + 1, 9), dtype=np.float64)
    cumP[0] = 0.0
    np.cumsum(P, axis=0, out=cumP[1:])

    s2_tau = np.full(len(lag_frames), np.nan)

    # ── Per-τ loop (vectorised over all origins) ──────────────────────────
    for i, tau in enumerate(lag_frames):
        if tau < 2 or tau > n:
            continue
        n_origins = n - tau          # windows: t₀ = 0, 1, …, n_origins-1
        if n_origins < 1:
            continue

        # One slice gives all n_origins window sums simultaneously.
        # M_sum[t₀] = cumP[t₀+τ] − cumP[t₀]  (verified in-bounds:
        #   max index of cumP accessed = τ + n_origins - 1 + 1
        #                              = τ + (n - τ - 1) + 1 = n  ✓)
        M_sum = cumP[tau : tau + n_origins] - cumP[:n_origins]  # (n_origins, 9)
        M_avg = M_sum / tau                                      # mean tensor

        # ‖M_avg‖²_F = Σⱼ M_avg[w,j]² = Σᵢⱼ <μᵢμⱼ>² per window
        frob_sq = np.einsum("wi,wi->w", M_avg, M_avg)           # (n_origins,)

        s2_per_window = np.clip((3.0 * frob_sq - 1.0) / 2.0, 0.0, 1.0)
        s2_tau[i]     = s2_per_window.mean()

    return s2_tau


def _s2_residue_worker(args: tuple) -> int:
    """Subprocess worker: compute S²(τ) for one residue and write its output file.

    Designed to run inside ``multiprocessing.Pool.imap_unordered`` or
    directly in the parent process (serial mode).  Fully self-contained so
    it can be pickled and sent to worker processes without carrying any
    MDAnalysis state.

    τ = 0 is prepended to the output and written as S² = 1.0 — the bond
    vector cannot reorient in zero time so the window is perfectly ordered by
    definition.  For all other lags :func:`_s2_calc_vs_lag` is called.

    If a residue is absent in a particular segment (e.g. due to chain-length
    differences) its column is filled with ``nan`` for all lags except τ = 0
    which is always 1.0.

    Args:
        args (tuple): Packed argument tuple:

            - **resid** (*int*) — Residue sequence number.
            - **vectors_by_segid** (*dict[str, np.ndarray]*) — ``{segid:
              raw_vectors}`` where ``raw_vectors`` has shape
              ``(n_frames, 3)``.
            - **all_segids** (*list[str]*) — Ordered segment IDs; determines
              column order in the output file.
            - **lag_frames** (*list[int]*) — Window lengths τ ≥ 2 to
              evaluate; τ = 0 is prepended automatically inside this function.
            - **out_dir_str** (*str*) — Output directory as a plain string
              (avoids ``pathlib.Path`` pickle edge-cases on some platforms).

    Returns:
        int: The processed *resid*, returned to the parent for progress
        reporting via ``imap_unordered``.

    Output:
        ``<out_dir_str>/resid_<resid>.dat`` — space-separated plain text::

            # lag_frame  s2_PROA  s2_PROB  …
              0          1.00000000  1.00000000
              2          0.95123456  0.94801234
              …
    """
    from pathlib import Path

    resid, vectors_by_segid, all_segids, lag_frames, out_dir_str = args

    # τ = 0 always prepended; S² = 1.0 by definition.
    lag_with_zero = [0] + lag_frames
    n_lags        = len(lag_with_zero)

    cols = []
    for segid in all_segids:
        vecs       = vectors_by_segid.get(segid)
        s2         = np.full(n_lags, np.nan)
        s2[0]      = 1.0
        if vecs is not None:
            s2[1:] = _s2_calc_vs_lag(vecs, lag_frames)
        cols.append(s2)

    data   = np.column_stack([np.array(lag_with_zero, dtype=float)] + cols)
    header = "lag_frame " + " ".join(f"s2_{seg}" for seg in all_segids)
    fname  = Path(out_dir_str) / f"resid_{resid}.dat"

    np.savetxt(
        fname, data,
        header=header, comments="# ",
        fmt=["%.0f"] + ["%.8f"] * len(all_segids),
    )
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
    """Extract raw backbone N-H bond vectors per ``(segid, resid)`` across frames.

    Iterates the trajectory **once** and records the H − N displacement
    vector for every matched backbone N-H pair.  Proline residues are skipped
    (no backbone amide H).

    Vectors are stored un-normalised.  Normalisation is deferred to
    :func:`_s2_calc_vs_lag` where it is performed *once per residue* (outside
    the τ loop) rather than once per window — O(n_frames) instead of
    O(n_frames · n_lags).

    **Vectorised extraction** — rather than calling ``atom.position`` once per
    key per frame (a Python-level loop with one C round-trip per atom), two
    :class:`~MDAnalysis.core.groups.AtomGroup` objects are built upfront (one
    for all N atoms, one for all matched H atoms in the same order).  Each
    frame then requires exactly one ``h_group.positions − n_group.positions``
    call — a single contiguous NumPy operation regardless of the number of
    residues.

    Args:
        topology (str): Path to topology file (.psf, .pdb, .gro, …).
        trajectory (str): Path to trajectory file (.xtc, .dcd, .trr, …).
        selected_segids (Optional[list[str]]): If given, only extract vectors
            for these segment IDs; all others are skipped *before* the frame
            loop.  ``None`` uses every segment.
        start (int): Index of the first frame to read.
        stop (Optional[int]): Index of the last frame, exclusive
            (``None`` reads to the end of the trajectory).
        step (int): Frame stride; every ``step``-th frame is loaded.
        align_traj (bool): Rigidly superpose each frame onto frame 0 using
            Cα atoms before collecting vectors.  Off by default — leave
            disabled when overall rotation/translation has already been
            removed from the trajectory (the standard case for
            RMSD-fitted production runs).

    Returns:
        dict: ``{(segid, resid): np.ndarray shape (n_frames, 3)}`` of raw
        H − N displacement vectors in ångström.  Every key shares the same
        ``n_frames``.

    Raises:
        ValueError: If no backbone N-H pairs are found (check atom naming:
            try ``'HN'`` instead of ``'H'``), or if the segid filter leaves
            no pairs.
    """
    _suppress_warnings()
    import MDAnalysis as mda
    from MDAnalysis.analysis import align as mda_align
    from collections import Counter

    u = mda.Universe(topology, trajectory)

    if align_traj:
        ref     = mda.Universe(topology, trajectory)
        aligner = mda_align.AlignTraj(
            u, ref, select="backbone and name CA", in_memory=True,
        )
        aligner.run(start=start, stop=stop, step=step)

    sel_N = u.select_atoms("name N and not resname PRO")
    sel_H = u.select_atoms("name H and not resname PRO")

    n_by_key    = {(a.segid, a.resid): a for a in sel_N}
    h_by_key    = {(a.segid, a.resid): a for a in sel_H}
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

    # Build two AtomGroups in matched order so each frame requires exactly
    # one vectorised positions subtraction instead of len(shared_keys) individual
    # Python-level atom.position calls.
    n_indices = [n_by_key[k].index for k in shared_keys]
    h_indices = [h_by_key[k].index for k in shared_keys]
    n_group   = u.atoms[n_indices]   # AtomGroup, shape (n_pairs,)
    h_group   = u.atoms[h_indices]   # AtomGroup, shape (n_pairs,)

    # Accumulate: one (n_pairs, 3) array per frame
    frame_vecs = []
    for _ts in u.trajectory[start:stop:step]:
        # Single C-level call; returns a fresh (n_pairs, 3) array
        frame_vecs.append(h_group.positions - n_group.positions)

    # Stack to (n_frames, n_pairs, 3), then slice per key
    all_vecs = np.array(frame_vecs)                          # (n_frames, n_pairs, 3)
    return {key: all_vecs[:, i, :] for i, key in enumerate(shared_keys)}


@app.command("time-s2")
def cmd_time_s2(
    top: Annotated[str, typer.Option(
        "--top", help="Topology file (.psf, .pdb, .gro, …).")] = "conf.psf",
    traj: Annotated[str, typer.Option(
        "--traj", help="Trajectory file (.xtc, .dcd, .trr, …).")] = "system.xtc",
    segments: Annotated[Optional[str], typer.Option(
        "--segments",
        help=(
            "Segment indices to include, 0-based in topology order. "
            "Formats: '0'  '0-4'  '0-2,5-7'  '1,3,5'. "
            "Omit to use all segments. "
            "Run once without this flag to print the full index list."
        ),
    )] = None,
    min_lag: Annotated[int, typer.Option(
        "--min-lag",
        help="Minimum window length τ in frames (default: 2).")] = 2,
    max_lag: Annotated[int, typer.Option(
        "--max-lag",
        help=(
            "Maximum window length τ in frames. "
            "Always hard-capped at n_frames // 2 — at this cap every "
            "data point still averages n_frames // 2 windows drawn from "
            "the full trajectory. "
            "-1 uses the cap directly (default: -1)."
        ),
    )] = -1,
    lag_stride: Annotated[int, typer.Option(
        "--lag-stride",
        help=(
            "Step between successive τ values written to output "
            "(default: 1). The hard-cap endpoint is always included."
        ),
    )] = 1,
    start: Annotated[int, typer.Option(
        "--start",
        help="First trajectory frame to read (default: 0).")] = 0,
    stop: Annotated[int, typer.Option(
        "--stop",
        help="Last trajectory frame to read; -1 = all frames (default: -1).")] = -1,
    step: Annotated[int, typer.Option(
        "--step",
        help="Frame stride when reading the trajectory (default: 1).")] = 1,
    align: Annotated[bool, typer.Option(
        "--align/--no-align",
        help=(
            "Align each frame on Cα atoms before extracting vectors "
            "(default: off). Enable only when overall tumbling has not "
            "already been removed from the trajectory."
        ),
    )] = False,
    outdir: Annotated[str, typer.Option(
        "--outdir",
        help="Output directory; created automatically if absent (default: s2-time)."
    )] = "s2-time",
    nproc: Annotated[int, typer.Option(
        "--nproc",
        help="Number of parallel worker processes (default: 1).")] = 1,
):
    """Compute **time-dependent S²(τ)** order parameters for backbone N-H bonds.

    **Physical meaning**

    The Lipari-Szabo order parameter S² measures local backbone rigidity from
    a single number.  S²(τ) extends this: *how ordered does the bond vector
    appear if observed only over a time window of length τ?*

    - τ → 0 : S² = 1.0 (no reorientation in zero time).
    - τ → ∞ : S²(τ) converges to the plateau Lipari-Szabo value.

    The shape of the S²(τ) curve reveals which timescales carry the most
    orientational disorder.

    **Calculation — window averaging over the full trajectory**

    For each window length τ:

    1. Slide a window of length τ over the *entire* trajectory.  Every
       possible origin ``t₀ ∈ [0, n_frames − τ)`` contributes one window
       (the same multi-origin averaging used in MSD calculations).
    2. Inside each window compute the mean outer-product tensor::

           M(t₀) = (1/τ) · Σ_{t=t₀}^{t₀+τ-1}  v(t) ⊗ v(t)

       where ``v(t)`` is the unit N-H bond vector at frame ``t``.
    3. Evaluate the Lipari-Szabo formula per window::

           S²(t₀) = ( 3 · ‖M(t₀)‖²_F  −  1 ) / 2

    4. Average over all windows::

           S²(τ) = mean_{t₀} S²(t₀)

    **Hard cap and full-trajectory coverage**

    τ is hard-capped at ``n_frames // 2``.  At this cap there are still
    ``n_frames // 2`` windows and *every frame* of the trajectory participates
    in at least one window — the whole trajectory is used for every data point
    on the S²(τ) curve.  ``--max-lag`` values above the cap are silently
    clamped; use ``-1`` (default) to select the cap directly.

    **Efficiency**

    *Extraction* — two AtomGroups (all matched N atoms; all matched H atoms)
    are built once.  Each frame requires a single vectorised
    ``h_group.positions - n_group.positions`` call regardless of residue count.

    *Computation* — outer products are precomputed once per residue and stored
    in a prefix-sum table.  Window averages for all origins at a given τ are
    obtained with a single NumPy slice — O(n_origins) per lag instead of
    O(n_origins · τ).  Parallelism is over residues (``--nproc``); each
    worker is fully independent with no inter-process communication.

    **Segment selection**

    Segments are indexed 0-based in topology order.  Run without
    ``--segments`` to print the full list, then rerun with a filter::

        scical time-s2 --top conf.psf --traj system.xtc
        # → [0] PROA  [1] PROB  [2] PROC …

        scical time-s2 --top conf.psf --traj system.xtc --segments 0-1

    Args:
        top (str): Topology file path.
        traj (str): Trajectory file path.
        segments (Optional[str]): Segment index selection string (see above).
        min_lag (int): Smallest τ to evaluate in frames (must be ≥ 2).
        max_lag (int): Largest τ in frames; hard-capped at ``n_frames // 2``.
            ``-1`` selects the cap directly (default).
        lag_stride (int): Step between successive τ values written to output.
            The hard-cap endpoint is always included regardless of stride.
        start (int): First trajectory frame to read.
        stop (int): Last trajectory frame; ``-1`` = all frames.
        step (int): Frame stride when reading the trajectory.
        align (bool): Cα-alignment before vector extraction; off by default.
        outdir (str): Output directory (created if absent).
        nproc (int): Parallel worker processes; parallelism is over residues.

    Output:
        One file per residue in ``<outdir>/``, named ``resid_<N>.dat``.
        Space-separated columns with a ``#``-prefixed header::

            # lag_frame  s2_PROA  s2_PROB  …
              0          1.00000000  1.00000000
              2          0.95123456  0.94801234
              …

    Example::

        # Print segment list, then exit
        scical time-s2 --top conf.psf --traj system.xtc

        # Segments 0-1, every 10th τ up to the hard cap, 8 workers
        scical time-s2 --top conf.psf --traj system.xtc \\
            --segments 0-1 --min-lag 2 --lag-stride 10 \\
            --start 0 --stop -1 --step 1 --outdir s2-time --nproc 8
    """
    _suppress_warnings()
    import MDAnalysis as mda

    os.makedirs(outdir, exist_ok=True)

    # ── Segment list + optional filter ───────────────────────────────────
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
    del u   # release before spawning subprocesses

    # ── Extract N-H vectors ──────────────────────────────────────────────
    typer.echo("\nExtracting N-H vectors …")
    nh_vectors = _s2_extract_nh_vectors(
        top, traj,
        selected_segids=selected_segids,
        start=start,
        stop=_traj_stop(stop),    # module helper: converts -1 → None
        step=step,
        align_traj=align,
    )

    # ── Hard cap: max τ = n_frames // 2 ─────────────────────────────────
    # At the cap: n_origins = n_frames - n_frames//2 >= n_frames//2.
    # Every frame participates in ≥1 window, so the full trajectory
    # contributes to every point on the S²(τ) curve.
    n_frames = next(iter(nh_vectors.values())).shape[0]
    hard_cap = n_frames // 2

    if max_lag == -1:
        eff_max = hard_cap
    elif max_lag > hard_cap:
        typer.echo(
            f"[!] --max-lag {max_lag} exceeds n_frames // 2 = {hard_cap}; "
            f"clamped to {hard_cap}.",
            err=True,
        )
        eff_max = hard_cap
    else:
        eff_max = max_lag

    if eff_max < min_lag:
        typer.echo(
            f"[!] Effective max_lag ({eff_max}) < min_lag ({min_lag}); "
            "nothing to compute.",
            err=True,
        )
        raise typer.Exit(1)

    # Build lag list; always include eff_max as the final point so the
    # hard-cap plateau estimate is never accidentally omitted by stride.
    lag_frames = list(range(min_lag, eff_max, lag_stride))
    if not lag_frames or lag_frames[-1] != eff_max:
        lag_frames.append(eff_max)

    active_segids = sorted({seg for seg, _ in nh_vectors})
    all_resids    = sorted({rid for _, rid in nh_vectors})

    typer.echo(f"\n[i] n_frames      : {n_frames}")
    typer.echo(
        f"[i] hard cap      : n_frames // 2 = {hard_cap}  "
        f"(≥{n_frames - hard_cap} windows per point; full trajectory always used)"
    )
    typer.echo(
        f"[i] lag range     : 0 (S²=1.0 fixed), "
        f"{min_lag} – {eff_max}, lag-stride={lag_stride}"
    )
    typer.echo(f"[i] n_lag_points  : {len(lag_frames) + 1}  (including τ=0)")
    typer.echo(f"[i] Segments      : {active_segids}")
    typer.echo(f"[i] Residues      : {len(all_resids)}")
    typer.echo(f"[i] Workers       : {nproc}")
    typer.echo(f"[i] Output dir    : {outdir}/")

    # ── Per-residue task list ────────────────────────────────────────────
    tasks = []
    for resid in all_resids:
        vectors_by_segid = {
            segid: nh_vectors[(segid, resid)]
            for segid in active_segids
            if (segid, resid) in nh_vectors
        }
        tasks.append((resid, vectors_by_segid, active_segids, lag_frames, outdir))

    n_tasks      = len(tasks)
    report_every = max(1, n_tasks // 10)

    # ── Dispatch ─────────────────────────────────────────────────────────
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
#  ░░  10.  Intra-chain Form Factor (Debye)  ░░
# =============================================================================

def _intracf_worker(args: tuple):
    """Subprocess worker: compute intra-chain form factor W(q) over a frame chunk.

    For each frame in the assigned chunk, evaluates the single-chain Debye
    scattering formula for every selected segment:

    .. math::

        W(q) = \\frac{1}{N^2} \\sum_{i,j} \\frac{\\sin(q \\cdot r_{ij})}{q \\cdot r_{ij}}

    where *N* is the number of reference atoms per segment and *r_ij* is the
    pairwise distance (in nm) between atoms *i* and *j*.  The diagonal terms
    (i == j) each contribute 1 (the limit of sin(x)/x as x->0).  The formula
    exploits i<->j symmetry by computing only the upper triangle and doubling.

    Supports two evaluation modes controlled by the ``dr`` parameter:

    - **Exact mode** (dr = 0): evaluates sinc(q*r) for every pair individually.
      O(N^2 * n_q) per frame per segment.
    - **Histogram mode** (dr > 0): bins pairwise distances into a histogram
      with resolution dr, then evaluates the Debye sum as a matrix-vector
      product over histogram bins.  O(N^2 + n_r_bins * n_q) per frame per
      segment.

    Args:
        args (tuple): A packed argument tuple containing:

            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **frame_indices** (*list[int]*) — Absolute frame indices to
              process in this worker.
            - **segids** (*list[str]*) — Segment identifiers to analyse.
            - **ref_atom** (*str*) — Reference atom name for selection
              (e.g. ``"CA"``), or ``"all"`` for all atoms.
            - **qs** (*np.ndarray*) — Pre-computed q-value array (nm^-1),
              passed directly from the main process to guarantee consistency.
            - **dr** (*float*) — Histogram bin width in nm.  If 0, uses exact
              pairwise evaluation (no histogram).
            - **worker_id** (*int*) — Worker index for logging.

    Returns:
        tuple[int, np.ndarray]: ``(n_frames_processed, Wq_sum)`` where
        *Wq_sum* has shape ``(n_segments, n_q_bins)`` containing the
        accumulated (un-averaged) W(q) over the processed frames.
    """
    _suppress_warnings()
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances as mda_distances

    (topology, trajectory, frame_indices, segids,
     ref_atom, qs, dr, worker_id) = args

    u = mda.Universe(topology, trajectory)
    bin_num = len(qs)
    use_hist = dr > 0.0

    segs = []
    for segid in segids:
        if ref_atom == "all":
            seg = u.select_atoms(f"segid {segid}")
        else:
            seg = u.select_atoms(f"segid {segid} and name {ref_atom}")
        segs.append(seg)

    n_segs = len(segs)
    Wq_sum = np.zeros((n_segs, bin_num), dtype=np.float64)

    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        box = u.dimensions

        for s_idx, seg in enumerate(segs):
            n_atoms = len(seg)
            if n_atoms < 2:
                Wq_sum[s_idx] += 1.0
                continue

            dists = mda_distances.self_distance_array(seg.positions, box=box)
            dists_nm = dists / 10.0  # Angstrom -> nm

            if use_hist:
                # --- Histogram-based acceleration ---
                r_max = float(dists_nm.max()) + dr
                n_r_bins = max(int(np.ceil(r_max / dr)), 1)

                hist, bin_edges = np.histogram(
                    dists_nm, bins=n_r_bins, range=(0.0, n_r_bins * dr)
                )

                # Verify no pairs were lost (integer arithmetic — exact)
                n_pairs = n_atoms * (n_atoms - 1) // 2
                assert hist.sum() == n_pairs, (
                    f"Histogram lost pairs: {hist.sum()} != {n_pairs}"
                )

                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                # Only process non-empty bins
                mask = hist > 0
                h = hist[mask].astype(np.float64)
                r = bin_centres[mask]

                if len(r) > 0:
                    qr_matrix = np.outer(qs, r)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        sinc_matrix = np.where(
                            qr_matrix == 0.0, 1.0,
                            np.sin(qr_matrix) / qr_matrix
                        )
                    pair_sum = sinc_matrix @ h
                else:
                    pair_sum = np.zeros(bin_num, dtype=np.float64)

            else:
                # --- Exact pairwise evaluation ---
                qr = np.outer(qs, dists_nm)

                with np.errstate(invalid='ignore', divide='ignore'):
                    sinc_qr = np.where(
                        qr == 0.0, 1.0, np.sin(qr) / qr
                    )

                pair_sum = sinc_qr.sum(axis=1)

            # W(q) = (2 * Sigma_{i<j} sinc(q*r_ij) + N) / N^2
            Wq_seg = (pair_sum * 2.0 + n_atoms) / (n_atoms ** 2)
            Wq_sum[s_idx] += Wq_seg

    typer.echo(f"  Worker {worker_id:02d}: {len(frame_indices)} frames processed")
    return len(frame_indices), Wq_sum


def _intracf_worker_gpu(args: tuple):
    """GPU-accelerated worker: compute intra-chain form factor W(q).

    Computes pairwise distances and the full Debye sinc sum on GPU using
    CuPy with float32 arithmetic for element-wise operations and float64
    accumulation for summation.  This eliminates the need for histogram
    binning — the exact formula is evaluated directly with GPU parallelism.

    Handles periodic boundary conditions via minimum-image convention
    (orthorhombic boxes only).  Automatically chunks the pair dimension
    to stay within GPU memory limits.

    Args:
        args (tuple): A packed argument tuple containing:

            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **frame_indices** (*list[int]*) — Absolute frame indices to
              process in this worker.
            - **segids** (*list[str]*) — Segment identifiers to analyse.
            - **ref_atom** (*str*) — Reference atom name for selection
              (e.g. ``"CA"``), or ``"all"`` for all atoms.
            - **qs** (*np.ndarray*) — Pre-computed q-value array (nm^-1).
            - **worker_id** (*int*) — Worker index for logging.

    Returns:
        tuple[int, np.ndarray]: ``(n_frames_processed, Wq_sum)`` where
        *Wq_sum* has shape ``(n_segments, n_q_bins)`` containing the
        accumulated (un-averaged) W(q) over the processed frames.

    Raises:
        ValueError: If the simulation box is non-orthorhombic (angles != 90).

    Notes:
        - Peak GPU memory scales as O(N^2) for the distance matrix phase
          and O(n_q * chunk_size) for the sinc evaluation phase.
        - Practical limit: N ~ 10,000 atoms per segment on a 16 GB GPU.
    """
    _suppress_warnings()
    import MDAnalysis as mda
    import cupy as cp

    (topology, trajectory, frame_indices, segids,
     ref_atom, qs, worker_id) = args

    u = mda.Universe(topology, trajectory)
    bin_num = len(qs)

    segs = []
    for segid in segids:
        if ref_atom == "all":
            seg = u.select_atoms(f"segid {segid}")
        else:
            seg = u.select_atoms(f"segid {segid} and name {ref_atom}")
        segs.append(seg)

    n_segs = len(segs)
    Wq_sum = np.zeros((n_segs, bin_num), dtype=np.float64)

    # Transfer q-values to GPU (float32 for element-wise ops)
    qs_gpu = cp.asarray(qs, dtype=cp.float32)

    # Chunk size for sinc evaluation: target ~2 GB peak for qr + sinc_qr
    # Both arrays are float32: 2 * bin_num * chunk * 4 bytes <= 2 GB
    max_elements = 256 * 1024 * 1024  # elements per array (~1 GB in float32)
    chunk_size = max(max_elements // bin_num, 1024)

    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        box = u.dimensions

        # Prepare box for minimum-image convention (orthorhombic only)
        if box is not None and box[0] > 0:
            angles = box[3:6]
            if not np.allclose(angles, 90.0, atol=0.01):
                raise ValueError(
                    f"GPU mode requires orthorhombic box (angles=90), "
                    f"got angles={angles}. Use CPU mode instead."
                )
            box_gpu = cp.asarray(box[:3] / 10.0, dtype=cp.float32)  # nm
        else:
            box_gpu = None

        for s_idx, seg in enumerate(segs):
            n_atoms = len(seg)
            if n_atoms < 2:
                Wq_sum[s_idx] += 1.0
                continue

            # Transfer positions to GPU (Angstrom -> nm, float32)
            pos_gpu = cp.asarray(seg.positions, dtype=cp.float32) / 10.0

            # Pairwise distance vectors: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diff = pos_gpu[:, None, :] - pos_gpu[None, :, :]
            del pos_gpu

            # Minimum-image convention (orthorhombic)
            if box_gpu is not None:
                diff -= cp.rint(diff / box_gpu) * box_gpu

            # Distance matrix (N, N) in float32
            dist_matrix = cp.sqrt((diff * diff).sum(axis=2))
            del diff

            # Extract upper triangle (i < j)
            idx_i, idx_j = cp.triu_indices(n_atoms, k=1)
            dists_gpu = dist_matrix[idx_i, idx_j]
            del dist_matrix, idx_i, idx_j

            # Chunked sinc evaluation (float32 compute, float64 accumulation)
            n_pairs = len(dists_gpu)
            pair_sum = cp.zeros(bin_num, dtype=cp.float64)

            for p_start in range(0, n_pairs, chunk_size):
                p_end = min(p_start + chunk_size, n_pairs)
                dists_chunk = dists_gpu[p_start:p_end]

                # float32: (n_q, 1) * (1, n_chunk) -> (n_q, n_chunk)
                qr = qs_gpu[:, None] * dists_chunk[None, :]

                # float32 sinc; use cp.float32(1.0) to prevent dtype promotion
                sinc_qr = cp.where(
                    qr == 0.0, cp.float32(1.0), cp.sin(qr) / qr
                )

                # Accumulate in float64 to prevent summation roundoff
                pair_sum += sinc_qr.sum(axis=1, dtype=cp.float64)
                del qr, sinc_qr

            del dists_gpu

            # W(q) = (2 * Sigma_{i<j} sinc(q*r_ij) + N) / N^2
            Wq_seg = (pair_sum * 2.0 + n_atoms) / (n_atoms ** 2)
            Wq_sum[s_idx] += cp.asnumpy(Wq_seg)

    typer.echo(f"  Worker {worker_id:02d}: {len(frame_indices)} frames processed (GPU)")
    return len(frame_indices), Wq_sum


@app.command("intraCF")
def cmd_intracf(
    top: Annotated[str, typer.Option("--top", help="Topology file")] = "conf.psf",
    traj: Annotated[str, typer.Option("--traj", help="Trajectory file")] = "system.xtc",
    ref: Annotated[str, typer.Option(
        "--ref", help="Reference atom name (e.g. CA, P). Default: all atoms."
    )] = "all",
    sel: Annotated[Optional[str], typer.Option(
        "--sel",
        help="Segment selection (e.g. R001, R001-R010, or R001,R003-R006). "
             "Default: all segments."
    )] = None,
    out: Annotated[str, typer.Option("--out", help="Output file")] = "intraCF.dat",
    qmax: Annotated[float, typer.Option(
        "--qmax", help="Maximum q value (nm^-1)"
    )] = 15.0,
    dq: Annotated[float, typer.Option("--dq", help="q-spacing (nm^-1)")] = 0.02,
    dr: Annotated[float, typer.Option(
        "--dr", help="Distance histogram bin width (nm). "
                     "0 = exact pairwise (default); >0 enables histogram acceleration. "
                     "Ignored when --gpu is set."
    )] = 0.0,
    gpu: Annotated[bool, typer.Option(
        "--gpu/--no-gpu", help="Use GPU acceleration via CuPy (exact, float32 compute)."
    )] = False,
    start: Annotated[int, typer.Option("--start", help="First frame index")] = 0,
    stop: Annotated[int, typer.Option(
        "--stop", help="Last frame index; -1=end"
    )] = -1,
    stride: Annotated[int, typer.Option("--stride", help="Frame stride")] = 100,
    nproc: Annotated[int, typer.Option("--nproc", help="Parallel workers")] = 4,
):
    """Calculate per-segment **intra-chain form factor** W(q) via the Debye formula.

    Computes the single-chain static structure factor (form factor) for every
    segment (or a selected subset) in the topology.  The Debye scattering
    equation is evaluated for a range of wavevectors q:

    .. math::

        W(q) = \\frac{1}{N^2} \\sum_{i,j} \\frac{\\sin(q \\cdot r_{ij})}{q \\cdot r_{ij}}

    where *r_ij* is the pairwise distance in nm between reference atoms *i*
    and *j* within the same chain.  The calculation accounts for periodic
    boundary conditions via the simulation box dimensions.

    Three evaluation modes are available:

    - **Exact CPU** (``--dr 0 --no-gpu``, default): evaluates sinc(q*r) for
      every atom pair individually on CPU.  O(N^2 * n_q).
    - **Histogram CPU** (``--dr 0.001 --no-gpu``): bins pairwise distances
      and evaluates the Debye sum as a matrix-vector product.
      O(N^2 + n_r_bins * n_q).  Error < 0.001% at dr=0.001 nm.
    - **GPU** (``--gpu``): exact pairwise evaluation on GPU via CuPy with
      float32 element-wise operations and float64 accumulation.
      O(N^2 * n_q) but massively parallelised.  Requires orthorhombic box.
      Ignores --dr.

    If the system contains multiple chains (segments), each chain is computed
    independently by default.  Use ``--sel`` to restrict to a subset of
    segments.

    Results are averaged over all analysed trajectory frames.

    Args:
        top (str): Path to the topology file (PSF, PDB, GRO, ...).
        traj (str): Path to the trajectory file (XTC, DCD, ...).
        ref (str): Reference atom name used for distance calculations
            (e.g. ``"CA"`` for C-alpha, ``"P"`` for phosphorus).
            Use ``"all"`` (default) to include every atom in each segment.
        sel (Optional[str]): Segment selection expression.  Accepts a single
            segment (``"R001"``), a range (``"R001-R010"``), or a
            comma-separated combination (``"R001,R003-R006,R010"``).
            Omit to compute all segments.
        out (str): Output file path for the form factor table.
        qmax (float): Maximum wavevector magnitude in nm^-1.
        dq (float): Wavevector spacing in nm^-1.
        dr (float): Distance histogram bin width in nm.  Set to 0 (default)
            for exact pairwise evaluation.  Set to a positive value (e.g.
            0.001) to enable histogram-based acceleration.  Ignored when
            --gpu is used.
        gpu (bool): Use GPU acceleration via CuPy.  Performs exact pairwise
            evaluation with float32 compute and float64 accumulation.
            Requires CuPy, a CUDA GPU, and an orthorhombic simulation box.
        start (int): Index of the first trajectory frame to include.
        stop (int): Index of the last trajectory frame; ``-1`` means end.
        stride (int): Step between analysed frames.
        nproc (int): Number of parallel worker processes (CPU mode only;
            GPU mode uses a single process with one GPU).

    Output:
        ``<out>`` -- space-separated columns::

            # q(nm^-1)  wq_<segid1>  wq_<segid2>  ...

        The first column is the wavevector q; subsequent columns are the
        frame-averaged W(q) for each segment.

    Example::

            scical intraCF --top conf.psf --traj system.xtc --ref CA \\
                           --qmax 15 --dq 0.02 --start 10000 --stride 100 \\
                           --out intraCF.dat --nproc 8

            scical intraCF --top conf.psf --traj system.xtc --ref CA \\
                           --dr 0.001 --out intraCF.dat --nproc 8

            scical intraCF --top conf.psf --traj system.xtc --ref CA \\
                           --gpu --out intraCF.dat

            scical intraCF --top conf.psf --traj system.xtc \\
                           --sel R001-R010 --gpu --out intraCF.dat
    """
    _suppress_warnings()
    import MDAnalysis as mda

    # Validate GPU availability early
    if gpu:
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
        except ImportError:
            typer.echo("[!] CuPy is not installed. Install with: "
                       "pip install cupy-cuda12x", err=True)
            raise typer.Exit(1)
        except cp.cuda.runtime.CUDARuntimeError:
            typer.echo("[!] No CUDA GPU detected.", err=True)
            raise typer.Exit(1)

    u = mda.Universe(top, traj)
    n_total = len(u.trajectory)

    # Resolve frame range
    effective_stop = n_total if stop == -1 else min(stop, n_total)
    frame_indices = list(range(start, effective_stop, stride))
    n_frames = len(frame_indices)

    if n_frames == 0:
        typer.echo("[!] No frames selected.", err=True)
        raise typer.Exit(1)

    # Detect segments that contain matching reference atoms
    if ref == "all":
        segments = [seg for seg in u.segments if len(seg.atoms) > 0]
    else:
        segments = [
            seg for seg in u.segments
            if len(seg.atoms.select_atoms(f"name {ref}")) > 0
        ]

    if not segments:
        if ref == "all":
            typer.echo("[!] No non-empty segments found.", err=True)
        else:
            typer.echo(f"[!] No segments contain atoms named '{ref}'.", err=True)
        raise typer.Exit(1)

    segids = [seg.segid for seg in segments]

    # Apply --sel segment filter
    if sel:
        segids = _parse_seg_selection(sel, segids)
        if not segids:
            typer.echo(f"[!] No segments match selection: '{sel}'", err=True)
            raise typer.Exit(1)

    bin_num = round(qmax / dq)
    if bin_num < 1:
        typer.echo(f"[!] q-grid is empty (qmax={qmax}, dq={dq}).", err=True)
        raise typer.Exit(1)
    qs = np.arange(1, bin_num + 1) * dq

    # Determine mode string for display
    if gpu:
        mode_str = "GPU exact (float32 compute, float64 accumulation)"
    elif dr > 0:
        mode_str = f"CPU histogram (dr={dr} nm)"
    else:
        mode_str = "CPU exact pairwise"

    typer.echo(f"[i] Topology    : {top}")
    typer.echo(f"[i] Trajectory  : {traj}")
    typer.echo(f"[i] Ref atom    : {'all atoms' if ref == 'all' else ref}")
    typer.echo(f"[i] Segments    : {len(segids)}")
    typer.echo(f"[i] Frames      : {n_frames}  "
               f"(start={start} stop={effective_stop} stride={stride})")
    typer.echo(f"[i] q range     : {qs[0]:.4f} - {qs[-1]:.4f} nm^-1  ({bin_num} bins)")
    typer.echo(f"[i] Mode        : {mode_str}")
    if not gpu:
        typer.echo(f"[i] Workers     : {nproc}")
    typer.echo(f"[i] Output      : {out}")
    del u

    if gpu:
        # GPU mode: single process, all frames on GPU
        work_args = (top, traj, frame_indices, segids, ref, qs, 0)
        n_proc_frames, Wq_total = _intracf_worker_gpu(work_args)
        total_frames = n_proc_frames
    else:
        # CPU mode: parallel workers
        n_workers = min(nproc, n_frames)
        chunks = [frame_indices[i::n_workers] for i in range(n_workers)]

        work_args = [
            (top, traj, chunk, segids, ref, qs, dr, wid)
            for wid, chunk in enumerate(chunks)
        ]

        with Pool(n_workers) as pool:
            results = pool.map(_intracf_worker, work_args)

        n_segs = len(segids)
        Wq_total = np.zeros((n_segs, bin_num), dtype=np.float64)
        total_frames = 0
        for n_proc_frames, Wq_partial in results:
            Wq_total += Wq_partial
            total_frames += n_proc_frames

    # Average over frames
    Wq_avg = Wq_total / total_frames

    # Write output
    n_segs = len(segids)
    W = 16
    header = " ".join(
        [f"{'q(nm^-1)':>{W - 2}}"] + [f"{'wq_' + s:>{W}}" for s in segids]
    )

    data = np.column_stack([qs, Wq_avg.T])
    np.savetxt(
        out, data,
        header=header,
        fmt=[f"%{W}.6f"] * (1 + n_segs),
    )

    typer.echo(f"[+] Intra-chain form factor saved -> {out}  "
               f"({n_segs} segments, {bin_num} q-bins, "
               f"{total_frames} frames averaged)")


# =============================================================================
#  11.  RADIAL / SLAB NUMBER-DENSITY PROFILES (2D / 3D)  ░░
# =============================================================================
#
#  Known limitation
#  ----------------
#  PBC minimum-image is applied to atom–origin *displacement* vectors.
#  If the reference group itself straddles a periodic boundary, its
#  center_of_mass() / center_of_geometry() may be incorrect.  Make the
#  reference molecule whole (e.g. with MDAnalysis.transformations) before
#  running this command in that case.  Use --ref-dist-out to diagnose this:
#  if bead-to-center distances look anomalously large/inconsistent across
#  frames, the reference group is likely not whole.
#

# Mapping from dimension label to coordinate axes indices
_DIMEN_AXES = {
    "xyz": [0, 1, 2],
    "xy":  [0, 1],
    "xz":  [0, 2],
    "yz":  [1, 2],
}

# For 2D dimen: the single axis index NOT present in _DIMEN_AXES.
# Used by --slab to identify which axis to profile along.
_DIMEN_OMITTED = {
    "xy": 2,   # omitted = Z
    "xz": 1,   # omitted = Y
    "yz": 0,   # omitted = X
}


def _density_worker(args: tuple):
    """Subprocess worker: accumulate radial or slab density histograms.

    Two modes controlled by the *slab* flag:

    - ``slab=False``:  bins atoms by their 2D/3D radial distance from the
      origin in the plane/space defined by *axes*.  PBC minimum-image
      convention is applied on every selected axis before computing the
      Euclidean distance.

    - ``slab=True``:  bins atoms by their signed 1D displacement along
      *omitted_axis* relative to the origin.  PBC minimum-image is applied
      on that single axis.  Symmetric bins ``[-r_max, +r_max]`` are used.
      The cross-section area of the box (product of the two in-plane box
      lengths) is accumulated for correct volumetric normalisation.

    Bin edges are computed with ``np.linspace`` (not ``np.arange``) so the
    number of bins is always exactly ``int(round(...))`` regardless of
    floating-point rounding in *dr*.

    Args:
        args (tuple): packed tuple of:

            - **topology**     (*str*)            — topology file path
            - **trajectory**   (*str*)            — trajectory file path
            - **frame_indices** (*list[int]*)     — absolute frame indices
            - **atom_names**   (*list[str]*)      — atom names to histogram
            - **ref_segids**   (*Optional[list[str]]*) — segment IDs for
              reference origin; ``None`` → box center
            - **use_mass**     (*bool*)            — CoM vs CoG for origin
            - **axes**         (*list[int]*)       — in-plane axis indices
            - **dr**           (*float*)           — bin width (Å)
            - **r_max**        (*float*)           — max radius / half-range (Å)
            - **worker_id**    (*int*)             — index for logging
            - **slab**         (*bool*)            — slab mode flag
            - **omitted_axis** (*Optional[int]*)  — axis index for slab mode

    Returns:
        tuple[int, dict[str, np.ndarray], float]:
            ``(n_frames_processed, counts_dict, area_sum)``

            *area_sum* is the running sum of per-frame box cross-section
            areas in Å² (non-zero only in slab mode).
    """
    _suppress_warnings()
    import MDAnalysis as mda

    (topology, trajectory, frame_indices, atom_names,
     ref_segids, use_mass, axes, dr, r_max,
     worker_id, slab, omitted_axis) = args

    u = mda.Universe(topology, trajectory)

    # ── Bin edges — must match cmd_density exactly ─────────────────────────
    if slab:
        n_bins = int(round(2 * r_max / dr))
        half   = n_bins * dr / 2
        bins   = np.linspace(-half, half, n_bins + 1)
    else:
        n_bins    = int(round(r_max / dr))
        rmax_eff  = n_bins * dr
        bins      = np.linspace(0.0, rmax_eff, n_bins + 1)

    # ── Atom selections ────────────────────────────────────────────────────
    selections = {name: u.select_atoms(f"name {name}") for name in atom_names}

    # ── Reference group ────────────────────────────────────────────────────
    if ref_segids is not None:
        ref_group = u.select_atoms("segid " + " ".join(ref_segids))
        if len(ref_group) == 0:
            raise ValueError(
                f"No atoms found in reference segments: {ref_segids}"
            )
        use_ref = True
    else:
        ref_group = None
        use_ref   = False

    counts      = {name: np.zeros(n_bins, dtype=np.float64) for name in atom_names}
    area_sum    = 0.0
    n_processed = 0

    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        box = u.dimensions  # shape (6,): [Lx, Ly, Lz, alpha, beta, gamma]

        # ── Origin (3D coordinate) ─────────────────────────────────────────
        if use_ref:
            origin = (ref_group.center_of_mass() if use_mass
                      else ref_group.center_of_geometry())
        else:
            if box is not None and box[0] > 0:
                origin = box[:3] / 2.0
            else:
                raise ValueError(
                    f"No --ref given and no valid box dimensions found at "
                    f"frame {frame_idx}.  Cannot determine box center."
                )

        # ── Cross-section area for slab normalisation ──────────────────────
        if slab:
            if box is None or box[axes[0]] <= 0 or box[axes[1]] <= 0:
                raise ValueError(
                    f"Slab mode requires valid box lengths on both in-plane "
                    f"axes at frame {frame_idx}."
                )
            area_sum += box[axes[0]] * box[axes[1]]

        # ── Histogram each atom type ───────────────────────────────────────
        for name in atom_names:
            sel = selections[name]
            if len(sel) == 0:
                continue

            if slab:
                values = sel.positions[:, omitted_axis] - origin[omitted_axis]
                if box is not None and box[omitted_axis] > 0:
                    L      = box[omitted_axis]
                    values -= L * np.round(values / L)
            else:
                diff = sel.positions[:, axes] - origin[axes]
                if box is not None:
                    box_axes = box[axes]
                    valid    = box_axes > 0
                    diff[:, valid] -= (
                        box_axes[valid]
                        * np.round(diff[:, valid] / box_axes[valid])
                    )
                values = np.linalg.norm(diff, axis=1)

            hist, _ = np.histogram(values, bins=bins)
            counts[name] += hist.astype(np.float64)

        n_processed += 1

    typer.echo(f"  Worker {worker_id:02d}: {n_processed} frames processed")
    return n_processed, counts, area_sum


def _ref_distance_worker(args: tuple):
    """Subprocess worker: compute each bead's distance to the origin,
    for a chunk of frames.

    Mirrors the origin logic in `_density_worker`:

    - ``ref_segids`` given: origin = ref group's center_of_mass()/
      center_of_geometry() per frame; beads = the ref group's own atoms
      (diagnoses whether the ref group is whole / straddling PBC).
    - ``ref_segids`` is ``None``: origin = box center per frame;
      beads = atoms matching *atom_names* (the --sel selection), i.e.
      the same atoms being profiled in the density histograms.

    PBC minimum-image convention is applied to the bead–origin
    displacement on each axis independently.

    Args:
        args (tuple): packed tuple of:

            - **topology**     (*str*)
            - **trajectory**   (*str*)
            - **frame_indices** (*list[int]*)
            - **ref_segids**   (*Optional[list[str]]*) — None → box center
            - **atom_names**   (*list[str]*) — used as bead group only
              when ref_segids is None
            - **use_mass**     (*bool*)
            - **worker_id**    (*int*)

    Returns:
        tuple[list[int], list[str], np.ndarray]:
            ``(frame_indices, bead_ids, distances)``, distances shape
            ``(n_beads, len(frame_indices))``.
    """
    _suppress_warnings()
    import MDAnalysis as mda

    (topology, trajectory, frame_indices, ref_segids,
     atom_names, use_mass, worker_id) = args

    u = mda.Universe(topology, trajectory)

    use_ref = ref_segids is not None
    if use_ref:
        bead_group = u.select_atoms("segid " + " ".join(ref_segids))
        if len(bead_group) == 0:
            raise ValueError(f"No atoms found in reference segments: {ref_segids}")
    else:
        bead_group = u.select_atoms(
            " or ".join(f"name {n}" for n in atom_names)
        )
        if len(bead_group) == 0:
            raise ValueError(f"No atoms found for --sel names: {atom_names}")

    # Sequential bead labels 1..N in the bead group's original atom order.
    n_beads   = len(bead_group)
    bead_ids  = [str(i + 1) for i in range(n_beads)]
    distances = np.zeros((n_beads, len(frame_indices)), dtype=np.float64)

    for col, frame_idx in enumerate(frame_indices):
        u.trajectory[frame_idx]
        box = u.dimensions

        if use_ref:
            origin = (bead_group.center_of_mass() if use_mass
                      else bead_group.center_of_geometry())
        else:
            if box is not None and box[0] > 0:
                origin = box[:3] / 2.0
            else:
                raise ValueError(
                    f"No --ref given and no valid box dimensions found at "
                    f"frame {frame_idx}.  Cannot determine box center."
                )

        diff = bead_group.positions - origin
        if box is not None:
            L     = box[:3]
            valid = L > 0
            diff[:, valid] -= L[valid] * np.round(diff[:, valid] / L[valid])

        distances[:, col] = np.linalg.norm(diff, axis=1)

    typer.echo(f"  Worker {worker_id:02d}: {len(frame_indices)} frames processed")
    return frame_indices, bead_ids, distances


@app.command("density")
def cmd_density(
    top: Annotated[str, typer.Option(
        "--top", help="Topology file"
    )] = "conf.psf",
    traj: Annotated[str, typer.Option(
        "--traj", help="Trajectory file"
    )] = "system.xtc",
    sel: Annotated[str, typer.Option(
        "--sel",
        help="Comma-separated atom names to profile (e.g. 'P,C,Mg'). "
             "Case-sensitive: must match topology exactly.",
    )] = "P",
    ref: Annotated[Optional[str], typer.Option(
        "--ref",
        help="Segment name range(s) whose center defines the radial origin "
             "(e.g. 'P001-P009', 'R001-R002,K001-K003'). "
             "Omit to use the box center.",
    )] = None,
    dimen: Annotated[str, typer.Option(
        "--dimen",
        help=(
            "'xyz' = 3D spherical shells (particles/Å³); "
            "'xy'/'xz'/'yz' = 2D annular rings in that plane (particles/Å²), "
            "or slab profile along the omitted axis when --slab is given."
        ),
    )] = "xyz",
    slab: Annotated[bool, typer.Option(
        "--slab/--no-slab",
        help=(
            "(2D --dimen only) Profile along the axis omitted from --dimen "
            "instead of radially in the plane.  "
            "  --dimen xy --slab  →  profile along Z  "
            "  --dimen xz --slab  →  profile along Y  "
            "  --dimen yz --slab  →  profile along X  "
            "Bins run from -rmax to +rmax relative to the origin.  "
            "Normalised by avg box cross-section area × dr → particles/Å³."
        ),
    )] = False,
    out: Annotated[str, typer.Option(
        "--out", help="Output file path"
    )] = "density_profile.dat",
    dr: Annotated[float, typer.Option(
        "--dr", help="Bin width (Å)"
    )] = 3.0,
    r_max: Annotated[float, typer.Option(
        "--rmax",
        help="Max radius for radial mode, or half-range for slab mode (Å)",
    )] = 200.0,
    use_mass: Annotated[bool, typer.Option(
        "--mass/--no-mass",
        help=(
            "Use center_of_mass() (--mass, default) or center_of_geometry() "
            "(--no-mass) for the reference origin.  "
            "Use --no-mass for coarse-grained systems without masses."
        ),
    )] = True,
    start: Annotated[int, typer.Option(
        "--start", help="First frame index (0-based)"
    )] = 0,
    stop: Annotated[int, typer.Option(
        "--stop", help="Last frame index (exclusive); -1 = end of trajectory"
    )] = -1,
    stride: Annotated[int, typer.Option(
        "--stride", help="Step between analysed frames"
    )] = 1,
    nproc: Annotated[int, typer.Option(
        "--nproc", help="Number of parallel worker processes"
    )] = 4,
    ref_dist_out: Annotated[Optional[str], typer.Option(
        "--ref-dist-out",
        help=(
            "Write each bead's per-frame distance to the origin to this "
            "file.  Format: bead_id,distance_frame<i>,...  "
            "If --ref is given: beads = the ref group's atoms, origin = "
            "ref group's COM/COG (diagnoses whether the ref group is "
            "whole / straddling a PBC boundary).  "
            "If --ref is omitted: beads = the --sel atoms, origin = box "
            "center (mirrors the density command's own origin fallback).  "
            "Pass an empty string ('') to skip writing this file."
        ),
    )] = "bead_distances.dat",
):
    """Calculate radial or slab **number-density profiles** (2D or 3D).

    **Modes**

    - ``--dimen xyz``:
      3D spherical radial density.
      Normalised by spherical shell volume ``(4/3)π(r_out³−r_in³)``.
      Units: particles/Å³.

    - ``--dimen xy|xz|yz`` (no ``--slab``):
      2D cylindrical radial density projected onto the chosen plane.
      All atoms regardless of their position on the omitted axis are
      projected.  Normalised by annular ring area ``π(r_out²−r_in²)``.
      Units: particles/Å².

    - ``--dimen xy|xz|yz --slab``:
      1D slab density along the axis omitted from ``--dimen``.
      Bins run from ``-rmax`` to ``+rmax`` relative to the origin.
      Normalised by average box cross-section area × dr.
      Units: particles/Å³.

    **PBC**

    Minimum-image convention is applied to atom–origin displacement vectors
    in all modes.  If the reference group straddles a periodic boundary, make
    it whole before analysis.  Use ``--ref-dist-out`` to diagnose this.

    **Origin (--ref)**

    - ``--ref`` given: center of mass/geometry of the specified segments.
    - ``--ref`` omitted: geometric box center (requires valid box dimensions).

    Args:
        top (str): Topology file path.
        traj (str): Trajectory file path.
        sel (str): Comma-separated atom names (case-sensitive).
        ref (Optional[str]): Segment range(s) for the radial origin.
        dimen (str): Profile dimensionality: xyz, xy, xz, or yz.
        slab (bool): Slab-density mode (2D dimen only).
        out (str): Output file path.
        dr (float): Bin width in Å.
        r_max (float): Max radius / half-range in Å.
        use_mass (bool): CoM (True) or CoG (False) for reference origin.
        start (int): First frame index.
        stop (int): Last frame index; -1 = end.
        stride (int): Frame stride.
        nproc (int): Parallel worker processes.
        ref_dist_out (Optional[str]): Bead-distance diagnostic output path;
            "" disables it.

    Output:
        Space-separated columns written to ``<out>``::

            # r(A)  density_P  density_C   ...   (radial modes)
            # Z(A)  density_P  density_C   ...   (slab mode, --dimen xy)

        If ``--ref-dist-out`` is set (default), a comma-separated file::

            bead_id,distance_frame<i>,distance_frame<j>,...

    Examples::

        # 3D spherical profile around pore segments
        scical density --top conf.psf --traj system.xtc \\
            --sel P,C,Mg --ref P001-P009 --dimen xyz \\
            --dr 3.0 --rmax 200 --out density_3d.dat --nproc 8

        # 2D cylindrical profile in the XY plane
        scical density --top conf.psf --traj system.xtc \\
            --sel P,C --ref R001-R002 --dimen xy \\
            --dr 2.0 --rmax 150 --out density_xy.dat --nproc 4

        # Slab profile along Z (using --dimen xy --slab)
        scical density --top conf.psf --traj system.xtc \\
            --sel P,C --ref R001 --dimen xy --slab \\
            --dr 2.0 --rmax 100 --out slab_z.dat --nproc 4

        # Skip the bead-distance diagnostic file
        scical density --top conf.psf --traj system.xtc \\
            --sel M1 --dimen xyz --dr 3.0 --rmax 200 \\
            --out density_3d.dat --ref-dist-out ""
    """
    _suppress_warnings()
    import MDAnalysis as mda

    # ── Validate --dimen ───────────────────────────────────────────────────
    dimen_lower = dimen.lower()
    if dimen_lower not in _DIMEN_AXES:
        typer.echo(
            f"[!] Invalid --dimen '{dimen}'. "
            f"Must be one of: xyz, xy, xz, yz.",
            err=True,
        )
        raise typer.Exit(1)

    axes  = _DIMEN_AXES[dimen_lower]
    is_3d = len(axes) == 3

    # ── Guard: --slab is only meaningful for 2D dimen ──────────────────────
    if slab and is_3d:
        typer.echo(
            "[!] --slab requires a 2D --dimen (xy, xz, or yz).  "
            "For --dimen xyz there is no omitted axis.",
            err=True,
        )
        raise typer.Exit(1)

    omitted_axis = _DIMEN_OMITTED.get(dimen_lower)  # None for xyz

    # ── Parse atom names ───────────────────────────────────────────────────
    atom_names = [n.strip() for n in sel.split(",") if n.strip()]
    if not atom_names:
        typer.echo("[!] No atom names specified in --sel.", err=True)
        raise typer.Exit(1)

    # ── Validate dr / r_max ────────────────────────────────────────────────
    if dr <= 0:
        typer.echo(f"[!] --dr must be positive, got {dr}.", err=True)
        raise typer.Exit(1)
    if r_max <= dr:
        typer.echo(
            f"[!] --rmax ({r_max}) must be greater than --dr ({dr}).",
            err=True,
        )
        raise typer.Exit(1)

    # ── Load universe ──────────────────────────────────────────────────────
    u = mda.Universe(top, traj)
    n_total = len(u.trajectory)

    # ── Frame range ────────────────────────────────────────────────────────
    effective_stop = n_total if stop == -1 else min(stop, n_total)
    frame_indices  = list(range(start, effective_stop, stride))
    n_frames       = len(frame_indices)
    if n_frames == 0:
        typer.echo("[!] No frames selected.", err=True)
        raise typer.Exit(1)

    # ── Resolve --ref ──────────────────────────────────────────────────────
    ref_segids = None
    if ref is not None:
        try:
            ref_seg_indices = _parse_sel(ref, u)
        except ValueError as exc:
            typer.echo(f"[!] {exc}", err=True)
            raise typer.Exit(1)

        ref_segids = [u.segments[i].segid for i in ref_seg_indices]
        ref_group  = u.select_atoms("segid " + " ".join(ref_segids))
        if len(ref_group) == 0:
            typer.echo(
                f"[!] No atoms found in reference segments: {ref_segids}",
                err=True,
            )
            raise typer.Exit(1)

        # Early validation: confirm origin computation works on frame 0
        try:
            _ = (ref_group.center_of_mass() if use_mass
                 else ref_group.center_of_geometry())
        except Exception as exc:
            hint = (
                "\n    Hint: use --no-mass for coarse-grained systems."
                if use_mass else ""
            )
            typer.echo(
                f"[!] {'center_of_mass' if use_mass else 'center_of_geometry'}"
                f"() failed: {exc}{hint}",
                err=True,
            )
            raise typer.Exit(1)

        origin_method = "center_of_mass" if use_mass else "center_of_geometry"
        typer.echo(
            f"[i] Reference origin : {origin_method} of "
            f"{len(ref_segids)} segment(s) "
            f"({ref_segids[0]} ... {ref_segids[-1]}, "
            f"{len(ref_group)} atoms)"
        )
    else:
        typer.echo("[i] Reference origin : box center")
        box = u.dimensions
        if box is None or box[0] <= 0:
            typer.echo(
                "[!] No valid box dimensions found.  "
                "Cannot use box center; please provide --ref.",
                err=True,
            )
            raise typer.Exit(1)

    # ── Warn if r_max exceeds half-box on the profiled axis (slab) ─────────
    if slab:
        box = u.dimensions
        if box is not None and box[omitted_axis] > 0:
            half_box = box[omitted_axis] / 2.0
            if r_max > half_box:
                typer.echo(
                    f"[w] --rmax ({r_max} Å) > half box length on the omitted "
                    f"axis ({half_box:.1f} Å) at frame 0.  "
                    f"Bins beyond ±{half_box:.1f} Å will be empty after "
                    f"minimum-image wrapping.",
                )

    # ── Validate atom selections ───────────────────────────────────────────
    for name in atom_names:
        n_atoms = len(u.select_atoms(f"name {name}"))
        if n_atoms == 0:
            typer.echo(
                f"[!] No atoms found with name '{name}' "
                f"(atom names are case-sensitive).",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"[i] Atom '{name}': {n_atoms} atoms")

    # ── Mode / label setup ─────────────────────────────────────────────────
    axis_names  = ["X", "Y", "Z"]
    dimen_label = dimen_lower.upper()

    if slab:
        omitted_name = axis_names[omitted_axis]
        mode_label   = f"slab along {omitted_name}-axis"
        unit_label   = "particles/A^3"
        bin_label    = f"{omitted_name}(A)"
    elif is_3d:
        mode_label   = "3D spherical shells"
        unit_label   = "particles/A^3"
        bin_label    = "r(A)"
    else:
        in_plane     = "".join(axis_names[ax] for ax in axes)
        mode_label   = f"2D annular rings in {in_plane}-plane"
        unit_label   = "particles/A^2"
        bin_label    = "r(A)"

    typer.echo(f"[i] Mode        : {mode_label}  ({unit_label})")
    typer.echo(
        f"[i] Frames      : {n_frames}  "
        f"(start={start} stop={effective_stop} stride={stride})"
    )
    typer.echo(f"[i] dr          : {dr} A")
    typer.echo(
        f"[i] r_max       : {r_max} A"
        + ("  (slab bins: -rmax … +rmax)" if slab else "")
    )
    typer.echo(f"[i] Workers     : {nproc}")
    typer.echo(f"[i] Output      : {out}")
    del u  # free before spawning workers

    # ── Distribute frames across workers ───────────────────────────────────
    n_workers = min(nproc, n_frames)
    chunks    = [frame_indices[i::n_workers] for i in range(n_workers)]
    work_args = [
        (top, traj, chunk, atom_names, ref_segids, use_mass,
         axes, dr, r_max, wid, slab, omitted_axis)
        for wid, chunk in enumerate(chunks)
    ]

    if n_workers == 1:
        results = [_density_worker(work_args[0])]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_density_worker, work_args)

    # ── Merge results from all workers ─────────────────────────────────────
    if slab:
        n_bins = int(round(2 * r_max / dr))
        half   = n_bins * dr / 2
        bins   = np.linspace(-half, half, n_bins + 1)
    else:
        n_bins   = int(round(r_max / dr))
        rmax_eff = n_bins * dr
        bins     = np.linspace(0.0, rmax_eff, n_bins + 1)

    total_counts = {name: np.zeros(n_bins, dtype=np.float64)
                    for name in atom_names}
    total_frames = 0
    total_area   = 0.0

    for n_proc, counts, area_sum in results:
        total_frames += n_proc
        total_area   += area_sum
        for name in atom_names:
            total_counts[name] += counts[name]

    # ── Normalise ──────────────────────────────────────────────────────────
    r_inner  = bins[:-1]
    r_outer  = bins[1:]
    centres  = 0.5 * (r_inner + r_outer)

    if slab:
        avg_area   = total_area / total_frames
        shell_norm = avg_area * dr
        typer.echo(f"[i] Avg cross-section area: {avg_area:.2f} A^2")
    elif is_3d:
        shell_norm = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
    else:
        shell_norm = np.pi * (r_outer**2 - r_inner**2)

    densities = {}
    for name in atom_names:
        avg_counts      = total_counts[name] / total_frames
        densities[name] = avg_counts / shell_norm

    # ── Write output ───────────────────────────────────────────────────────
    W         = 16
    meta_line = (
        f"units: {unit_label} | mode: {mode_label} | "
        f"dr={dr} A | r_max={r_max} A | frames={total_frames}"
        + (f" | avg_area={avg_area:.2f} A^2" if slab else "")
    )
    col_header = " ".join(
        [f"{bin_label:>{W - 2}}"] +
        [f"{'density_' + name:>{W}}" for name in atom_names]
    )
    data = np.column_stack(
        [centres] + [densities[name] for name in atom_names]
    )
    np.savetxt(
        out, data,
        header=meta_line + "\n" + col_header,
        fmt=[f"%{W}.6f"] * (1 + len(atom_names)),
    )

    typer.echo(
        f"\n[+] Density profile saved -> {out}  "
        f"({len(atom_names)} atom type(s), {n_bins} bins, "
        f"{total_frames} frames, {mode_label}, {unit_label})"
    )

    # ── Optional: per-bead distance-to-center diagnostic ───────────────────
    if ref_dist_out:   # skip if empty string ("" = user opted out)
        origin_desc = "ref-group center" if ref is not None else "box center"
        typer.echo(
            f"\n[i] Computing bead distances to {origin_desc} -> {ref_dist_out}"
        )

        dist_work_args = [
            (top, traj, chunk, ref_segids, atom_names, use_mass, wid)
            for wid, chunk in enumerate(chunks)   # reuse chunks from above
        ]

        if n_workers == 1:
            dist_results = [_ref_distance_worker(dist_work_args[0])]
        else:
            with Pool(n_workers) as pool:
                dist_results = pool.map(_ref_distance_worker, dist_work_args)

        col_of_frame = {f: i for i, f in enumerate(frame_indices)}
        bead_ids     = dist_results[0][1]
        n_beads      = len(bead_ids)
        distances    = np.zeros((n_beads, n_frames), dtype=np.float64)

        for chunk_frames, chunk_bead_ids, chunk_dist in dist_results:
            if chunk_bead_ids != bead_ids:
                typer.echo(
                    "[!] Internal error: bead ordering mismatch across "
                    "workers.",
                    err=True,
                )
                raise typer.Exit(1)
            for local_col, frame_idx in enumerate(chunk_frames):
                distances[:, col_of_frame[frame_idx]] = chunk_dist[:, local_col]

        header = "bead_id," + ",".join(
            f"distance_frame{f}" for f in frame_indices
        )
        with open(ref_dist_out, "w") as fh:
            fh.write(header + "\n")
            for i in range(n_beads):
                row = ",".join(f"{d:.6f}" for d in distances[i])
                fh.write(f"{i + 1},{row}\n")

        typer.echo(
            f"[+] Bead distances saved -> {ref_dist_out}  "
            f"({n_beads} beads, {n_frames} frames, origin={origin_desc})"
        )
    
                   
# =============================================================================
#  ░░  OCF  ░░
# =============================================================================

def _ocf_worker(args: tuple):
    """Subprocess worker: compute the orientational correlation function (OCF)
    for one chain segment.

    The OCF measures how quickly bond-vector orientational memory decays along
    the backbone of a single chain.  For a chain with *N* reference atoms
    there are *N − 1* bond vectors **b**_i = (r_{i+1} − r_i) / |r_{i+1} − r_i|.
    The OCF at bond-index lag *k* (k = 1 … N−2) is:

    .. math::

        C(k) = \\left\\langle \\mathbf{b}_i \\cdot \\mathbf{b}_{i+k}
               \\right\\rangle_{i,\\,\\text{frames}}

    where the average runs over all valid pairs (i, i+k) within every
    analysed frame.  This exactly mirrors the double-loop in the original
    script::

        for i in range(len(bds) - 1):
            for j in range(i + 1, len(bds)):
                ij = j - i            # lag 1 … N-2
                ocfs[ij] += dot(bds[i], bds[j])

    Lag-0 (always 1.0, trivial self-dot) is never accumulated.  The maximum
    lag is *N−2* (not *N−1*) because the outer loop runs only to
    ``len(bds) - 1``.

    The worker opens its own independent Universe (one per subprocess) to
    avoid pickling issues.  A numpy-vectorised inner loop replaces the
    original Python double loop over atoms.

    Args:
        args (tuple): A packed argument tuple containing:

            - **segid** (*str*) — Segment identifier (e.g. ``"R001"``).
            - **ref_atom** (*str*) — Atom name used to trace the backbone
              (e.g. ``"CA"`` for proteins, ``"P"`` for nucleic acids/lipids).
            - **topology** (*str*) — Path to the topology file.
            - **trajectory** (*str*) — Path to the trajectory file.
            - **start** (*int*) — First frame index (inclusive).
            - **stop** (*int*) — Last frame index (inclusive); ``-1`` = end.
            - **stride** (*int*) — Step between analysed frames.

    Returns:
        tuple[str, Optional[np.ndarray]]:
            ``(segid, ocf_array)`` where *ocf_array* has shape ``(N-2,)``
            containing C(1) … C(N-2) in ascending lag order, matching the
            values written by the original script's normalisation loop
            ``ocfs[1] … ocfs[natom-2]``.
            Returns ``(segid, None)`` if the atom selection is empty, the
            chain has fewer than 3 reference atoms (need at least 2 bond
            vectors to form one lag-1 pair), or the trajectory slice is empty.
    """
    _suppress_warnings()
    import MDAnalysis as mda

    segid, ref_atom, topology, trajectory, start, stop, stride = args
    u   = mda.Universe(topology, trajectory)
    sel = u.select_atoms(f"segid {segid} and name {ref_atom}")

    if len(sel.atoms) == 0:
        typer.echo(
            f"  [!] Segment {segid}: no atoms match 'name {ref_atom}' — skipped.",
            err=True,
        )
        return segid, None

    natom = len(sel.atoms)   # number of ref atoms in this chain
    nbond = natom - 1        # number of bond vectors  (= len(bds) in original)

    # Need at least 2 bond vectors to form one pair at lag 1.
    # Original: range(len(bds)-1) requires len(bds) >= 2 i.e. natom >= 3.
    if nbond < 2:
        typer.echo(
            f"  [!] Segment {segid}: only {natom} ref atom(s) — need ≥ 3 "
            f"to form at least one lag-1 pair.",
            err=True,
        )
        return segid, None

    # Accumulators for lags 1 … nbond-1.
    # Array index k-1 stores lag k: index 0 → lag 1, index nbond-2 → lag nbond-1.
    n_lags  = nbond - 1
    ocf_sum = np.zeros(n_lags, dtype=np.float64)
    count   = np.zeros(n_lags, dtype=np.int64)

    n_frames = 0
    for _ts in u.trajectory[start : _traj_stop(stop) : stride]:
        pos  = sel.positions                            # (natom, 3)
        diff = pos[1:] - pos[:-1]                      # (nbond, 3)  bond vectors
        nrm  = np.linalg.norm(diff, axis=1, keepdims=True)
        # Guard against zero-length bonds (degenerate frames).
        nrm  = np.where(nrm == 0.0, 1.0, nrm)
        bds  = diff / nrm                              # (nbond, 3)  unit vectors

        # Vectorised lag accumulation.
        # Original double loop: i in range(len(bds)-1), j in range(i+1, len(bds))
        #   ij = j - i  runs from 1 to nbond-1.
        # Equivalent: for each lag k = 1..nbond-1,
        #   pairs are (bds[0],bds[k]), (bds[1],bds[k+1]), ..., (bds[nbond-1-k], bds[nbond-1])
        #   i.e. bds[:nbond-k]  dot  bds[k:]
        for k in range(1, nbond):        # k = lag = 1 … nbond-1
            dots = np.einsum("ij,ij->i", bds[: nbond - k], bds[k:])
            ocf_sum[k - 1] += dots.sum()
            count[k - 1]   += len(dots)

        n_frames += 1

    if n_frames == 0:
        typer.echo(
            f"  [!] Segment {segid}: trajectory slice is empty — skipped.",
            err=True,
        )
        return segid, None

    # Normalise: ocf[k-1] = ocf_sum[k-1] / count[k-1]  for k = 1..nbond-1.
    # Matches original:  for i in range(1, natom-1): ocfs[i] /= nums[i]
    ocf = ocf_sum / count                              # (n_lags,) = (nbond-1,)

    typer.echo(
        f"  [+] Segment {segid}: {natom} atoms, {n_lags} lags, {n_frames} frames"
    )
    return segid, ocf


@app.command("ocf")
def cmd_ocf(
    top:    Annotated[str, typer.Option("--top",  help="Topology file")]  = "conf.psf",
    traj:   Annotated[str, typer.Option("--traj", help="Trajectory file")] = "system.xtc",
    sel:    Annotated[Optional[str], typer.Option(
                "--sel",
                help=(
                    "Segment selection: single id ('R001'), range ('R001-R010'), "
                    "or comma-separated mix ('R001-R099,K001-K010').  "
                    "Omit to use all segments."
                ),
            )] = None,
    ref:    Annotated[str, typer.Option(
                "--ref",
                help="Reference atom name traced along the chain backbone "
                     "(e.g. 'CA' for proteins, 'P' for lipids/nucleic acids).",
            )] = "CA",
    out:    Annotated[str, typer.Option("--out",  help="Output .dat file")] = "ocf.dat",
    start:  Annotated[int, typer.Option("--start",  help="First frame index")]                = 0,
    stop:   Annotated[int, typer.Option("--stop",   help="Last frame index; -1 = end")]       = -1,
    stride: Annotated[int, typer.Option("--stride", help="Frame stride")]                     = 1,
    nproc:  Annotated[int, typer.Option("--nproc",  help="Parallel worker processes")]        = 4,
):
    """Compute the **orientational correlation function** (OCF) per chain.

    For every chain selected by ``--sel``, the OCF measures how quickly
    bond-vector orientation memory decays along the backbone:

    .. math::

        C(k) = \\langle \\mathbf{b}_i \\cdot \\mathbf{b}_{i+k}
               \\rangle_{i,\\,\\text{frames}}

    where **b**_i is the unit vector of the *i*-th bond formed by consecutive
    ``--ref`` atoms, and *k* is the bond-index lag (1 … N−1 for a chain with
    *N* reference atoms).

    Output is a ``.dat`` file with rows ``lag  chain1  chain2  …``, one row
    per lag value.  Each column is the OCF for that chain averaged over all
    valid *(i, i+k)* pairs across all analysed frames.

    Args:
        top (str): Topology file path (PSF, GRO, …).
        traj (str): Trajectory file path (XTC, DCD, …).
        sel (Optional[str]): Segment selection string.  Accepts a single
            segment (``"R001"``), a range (``"R001-R099"``), or a
            comma-separated mix (``"R001-R099,K001-K010"``).  Omit to include
            all segments in the topology.
        ref (str): Name of the backbone atom used to build bond vectors.
            ``"CA"`` for coarse-grained or all-atom proteins; ``"P"`` for
            lipid or nucleic-acid phosphate backbones.
        out (str): Output file path.
        start (int): Index of the first trajectory frame to include.
        stop (int): Index of the last trajectory frame (inclusive); ``-1``
            means read to the end of the trajectory.
        stride (int): Step size between analysed frames.
        nproc (int): Number of parallel worker processes.  Each worker handles
            one chain independently.

    Output:
        ``<out>`` — tab-separated columns: ``lag  <segid1>  <segid2>  …``
        *lag* counts bond-vector steps (1 = nearest neighbours, 2 = one bond
        apart, etc.).  Values are dimensionless cosines in [−1, 1].

    Example::

        # Protein backbone (Cα), all chains in R001-R240
        scical ocf --top conf.psf --traj system.xtc --sel R001-R240 \\
            --ref CA --out ocf.dat --stride 1 --nproc 8

        # Lipid phosphate backbone, last 1000 frames
        scical ocf --top mem.psf --traj mem.xtc --sel L001-L240 \\
            --ref P --out ocf_lipid.dat --start -1000 --nproc 8
    """
    _suppress_warnings()
    import MDAnalysis as mda

    # ── Load topology and resolve segment selection ─────────────────────────
    typer.echo("Loading topology...")
    u = _load_universe(top, traj)

    try:
        sel_indices = _parse_sel(sel, u)
    except ValueError as exc:
        typer.echo(f"[!] {exc}", err=True)
        raise typer.Exit(1)

    segids = [u.segments[i].segid for i in sel_indices]

    # ── Validate that --ref atoms actually exist in at least one segment ────
    probe = u.select_atoms(f"name {ref}")
    if len(probe) == 0:
        typer.echo(
            f"[!] No atoms named '{ref}' found in the topology.  "
            f"Check --ref (e.g. 'CA' for proteins, 'P' for lipids).",
            err=True,
        )
        raise typer.Exit(1)

    # ── Info banner ─────────────────────────────────────────────────────────
    typer.echo(f"[i] Segments  : {segids}")
    typer.echo(f"[i] Ref atom  : {ref}")
    typer.echo(f"[i] Frames    : start={start}  stop={stop}  stride={stride}")
    typer.echo(f"[i] Workers   : {nproc}")
    typer.echo(f"[i] Output    : {out}")

    # ── Dispatch one worker per chain ────────────────────────────────────────
    work_args = [
        (segid, ref, top, traj, start, stop, stride)
        for segid in segids
    ]

    if nproc == 1:
        results = [_ocf_worker(a) for a in work_args]
    else:
        with Pool(nproc) as pool:
            results = pool.map(_ocf_worker, work_args)

    # ── Collect valid results (preserve --sel ordering) ─────────────────────
    valid_segids: list = []
    ocf_all:      list = []
    for segid, ocf in results:
        if ocf is not None:
            valid_segids.append(segid)
            ocf_all.append(ocf)

    if not ocf_all:
        typer.echo("[!] No valid OCF results — nothing written.", err=True)
        raise typer.Exit(1)

    # ── Align lengths (chains may differ in natom → different nbond) ────────
    # Truncate every array to the shortest chain so the output matrix is
    # rectangular.  Chains with more atoms simply contribute fewer long-lag
    # columns than they could — warn the user if lengths differ.
    lengths = [len(o) for o in ocf_all]
    if len(set(lengths)) > 1:
        typer.echo(
            f"[w] Chains have different numbers of '{ref}' atoms "
            f"(lag lengths: {sorted(set(lengths))}).  "
            f"Truncating all columns to the shortest chain ({min(lengths)} lags).",
            err=True,
        )
    min_lags = min(lengths)
    ocf_all  = [o[:min_lags] for o in ocf_all]

    # ── Write output ─────────────────────────────────────────────────────────
    # Format: tab-separated, first column = lag (1-indexed), then one column
    # per chain.  Header line starts with '#'.
    typer.echo(
        f"\nWriting OCF -> {out}  "
        f"({len(valid_segids)} chains, {min_lags} lags)"
    )
    W = 14
    with open(out, "w") as fh:
        header_cols = ["lag"] + valid_segids
        fh.write("# " + "\t".join(f"{c:>{W}}" for c in header_cols) + "\n")
        for lag_idx in range(min_lags):
            lag_num  = lag_idx + 1
            row_vals = "\t".join(
                f"{ocf_all[chain_idx][lag_idx]:>{W}.6f}"
                for chain_idx in range(len(valid_segids))
            )
            fh.write(f"  {lag_num:>{W}}\t{row_vals}\n")

    typer.echo("[+] Done!")


# =============================================================================
#  ░░  Entry-point  ░░
# =============================================================================

if __name__ == "__main__":
    app()