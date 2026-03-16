#!/usr/bin/env python3
"""
Calculate CA-CA distances between residue pairs across a trajectory — PARALLEL.

Performance design
──────────────────
  1. select_atoms called ONCE per unique (resid, segid), not once per pair.
  2. Atom indices resolved in the main process; workers receive plain int lists.
  3. Each worker receives a CONTIGUOUS trajectory slice [w_start:w_stop:stride]
     and iterates sequentially — no random seeks (critical for DCD / XTC).
  4. Per-frame distance calculation is fully vectorised:
       np.linalg.norm(ag1.positions - ag2.positions, axis=1)  →  (n_pairs,)
  5. Output float formatting uses np.savetxt on the whole block at once.

Input:  one or more pair files
        Columns: resid_1  segid_1  resid_2  segid_2
        Delimiter: whitespace OR comma (auto-detected per file)
Output: one .dat per pair file, same folder, suffix _distance.dat
"""

import io
import os
import sys
import argparse
import numpy as np
import MDAnalysis as mda
from multiprocessing import Pool, cpu_count


# ── USER DEFAULTS (all overridden by CLI) ─────────────────────────────────────
DEFAULT_TOPOLOGY   = "conf.psf"
DEFAULT_TRAJECTORY = "system.dcd"
DEFAULT_PAIR_FILES = ["./residue_pairs.dat"]
DEFAULT_WORKERS    = cpu_count() or 1   # cpu_count() can return None in containers
# ─────────────────────────────────────────────────────────────────────────────


def _warn(msg):
    """Write a warning to stderr only — never pollutes stdout / pipes."""
    print(f"[WARNING] {msg}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calculate CA-CA distances for residue pairs over a trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-p", "--topology",   default=DEFAULT_TOPOLOGY,
                   help="Topology file (.prmtop / .psf / .tpr / .gro ...)")
    p.add_argument("-t", "--trajectory", default=DEFAULT_TRAJECTORY,
                   help="Trajectory file (.dcd / .xtc / .trr / .nc ...)")
    p.add_argument("-f", "--pair-files", nargs="+", default=DEFAULT_PAIR_FILES,
                   metavar="FILE",
                   help="One or more residue-pair list files")
    p.add_argument("--start",  type=int, default=None,
                   help="First frame (0-based, inclusive). Default: 0.")
    p.add_argument("--stop",   type=int, default=None,
                   help="Last frame (0-based, exclusive). Default: end.")
    p.add_argument("--stride", type=int, default=1,
                   help="Step between frames.")
    p.add_argument("-n", "--workers", type=int, default=DEFAULT_WORKERS,
                   help="Number of parallel worker processes.")
    return p.parse_args()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_pairs(filepath):
    """
    Parse one pair file -> list of (resid1, segid1, resid2, segid2).

    Delimiter auto-detection:
      - If a data line contains a comma  -> split on commas
      - Otherwise                        -> split on any whitespace
    Both '4  P002  1  P001' and '4, P002, 1, P001' are accepted.
    """
    pairs = []
    with open(filepath) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Auto-detect delimiter
            parts = [x.strip() for x in line.split(",")] if "," in line \
                    else line.split()
            if len(parts) != 4:
                _warn(f"{filepath}:{lineno} — expected 4 fields, "
                      f"got {len(parts)}, skipping: {line!r}")
                continue
            resid1, segid1, resid2, segid2 = parts
            try:
                pairs.append((int(resid1), segid1, int(resid2), segid2))
            except ValueError:
                _warn(f"{filepath}:{lineno} — non-integer resid, "
                      f"skipping: {line!r}")
    return pairs


def load_all_pairs(filepaths):
    """
    Merge multiple pair files.

    Returns
    -------
    all_pairs  : list — deduplicated union, preserving first-seen order
    file_pairs : dict — {filepath: [pair, ...]}
    """
    file_pairs = {}
    seen       = set()
    all_pairs  = []
    for fp in filepaths:
        pairs = load_pairs(fp)
        file_pairs[fp] = pairs
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                all_pairs.append(pair)
        print(f"  {fp}: {len(pairs)} pairs  "
              f"({len(all_pairs)} unique total so far)")
    return all_pairs, file_pairs


def make_output_path(pair_filepath):
    """<dir>/<stem>_distance.dat"""
    d    = os.path.dirname(os.path.abspath(pair_filepath))
    stem = os.path.splitext(os.path.basename(pair_filepath))[0]
    return os.path.join(d, f"{stem}_distance.dat")


# ── Frame helpers ─────────────────────────────────────────────────────────────

def resolve_frame_range(n_total, start, stop, stride):
    """
    Clip and validate start/stop/stride against the actual trajectory length.
    Returns (start, stop, stride) ready for u.trajectory[start:stop:stride].
    """
    start  = max(0,       start  if start  is not None else 0)
    stop   = min(n_total, stop   if stop   is not None else n_total)
    stride = max(1, stride)
    if start >= stop:
        raise ValueError(f"start={start} >= stop={stop}; no frames selected.")
    return start, stop, stride


def n_selected_frames(start, stop, stride):
    """Number of frames in range(start, stop, stride)."""
    return max(0, (stop - start + stride - 1) // stride)


def make_worker_slices(start, stop, stride, n_workers):
    """
    Divide [start:stop:stride] into n_workers contiguous sequential slices.
    Each slice (w_start, w_stop, stride) has no overlap and full coverage.
    Workers read forward through their section without seeking backwards.
    """
    sel      = np.arange(start, stop, stride)
    n_actual = min(n_workers, len(sel))
    chunks   = np.array_split(sel, n_actual)
    slices   = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        w_start = int(chunk[0])
        w_stop  = int(chunk[-1]) + stride   # exclusive; MDA clips to traj end
        slices.append((w_start, w_stop, stride))
    return slices


# ── Atom-index resolution (main process, done ONCE) ──────────────────────────

def resolve_ca_indices(universe, pairs):
    """
    Map every pair to two CA atom indices.

    select_atoms is called once per unique (resid, segid), not once per pair.
    Returns idx1, idx2 as plain Python lists of ints (picklable).
    """
    unique_res = {}
    all_keys   = set()
    for r1, s1, r2, s2 in pairs:
        all_keys.add((r1, s1))
        all_keys.add((r2, s2))

    for resid, segid in all_keys:
        sel = universe.select_atoms(
            f"resid {resid} and segid {segid} and name CA")
        if len(sel) == 0:
            raise ValueError(f"No CA atom found: resid={resid}, segid={segid}")
        if len(sel) > 1:
            _warn(f"Multiple CAs for resid={resid} segid={segid}; using first.")
        unique_res[(resid, segid)] = sel[0].index

    idx1 = [unique_res[(r1, s1)] for r1, s1, r2, s2 in pairs]
    idx2 = [unique_res[(r2, s2)] for r1, s1, r2, s2 in pairs]
    return idx1, idx2


# ── Worker ────────────────────────────────────────────────────────────────────

def worker(args):
    """
    Subprocess worker — owns its own Universe instance.

    Iterates the assigned trajectory slice sequentially (no random seeks).
    Distance calculation is fully vectorised per frame — no Python loops over pairs.

    args : (topology, trajectory, idx1, idx2, traj_slice, worker_id)
    Returns : (list[int] frame_numbers, ndarray shape=(n_pairs, n_local))
    """
    topology, trajectory, idx1, idx2, traj_slice, worker_id = args
    w_start, w_stop, stride = traj_slice

    u   = mda.Universe(topology, trajectory)
    ag1 = u.atoms[idx1]          # (n_pairs,) AtomGroup — CA set 1
    ag2 = u.atoms[idx2]          # (n_pairs,) AtomGroup — CA set 2

    n_pairs = len(idx1)
    n_est   = n_selected_frames(w_start, w_stop, stride)
    buf     = np.empty((n_pairs, n_est), dtype=np.float32)
    frames  = []

    for col, ts in enumerate(u.trajectory[w_start:w_stop:stride]):
        buf[:, col] = np.linalg.norm(ag1.positions - ag2.positions, axis=1)
        frames.append(ts.frame)

    n_actual = len(frames)
    result   = buf[:, :n_actual].copy()   # copy not view — IPC sends only used data

    print(f"  Worker {worker_id:02d}: "
          f"traj[{w_start}:{w_stop}:{stride}]  ->  {n_actual} frames")
    return frames, result


# ── Output helper ─────────────────────────────────────────────────────────────

def write_output(out_path, pairs, dist_block, all_frames):
    """
    Write one output file.

    dist_block : ndarray shape (n_file_pairs, n_frames).
    Float formatting uses np.savetxt on the entire block at once
    (~2.5x faster than a Python-level f-string join per row).
    """
    float_buf = io.StringIO()
    np.savetxt(float_buf, dist_block, fmt="%.4f", delimiter=", ")
    float_buf.seek(0)
    float_lines = float_buf.read().splitlines()

    assert len(float_lines) == len(pairs), (
        f"write_output: {len(float_lines)} rows but {len(pairs)} pairs — "
        "dist_block shape mismatch"
    )

    frame_header = ", ".join(f"frame_{fi}" for fi in all_frames)
    with open(out_path, "w") as fh:
        fh.write(f"# resid_1, segid_1, resid_2, segid_2, {frame_header}\n")
        for (r1, s1, r2, s2), float_line in zip(pairs, float_lines):
            fh.write(f"{r1}, {s1}, {r2}, {s2}, {float_line}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 64)
    print(f"Topology          : {args.topology}")
    print(f"Trajectory        : {args.trajectory}")
    print(f"Pair files        : {args.pair_files}")
    print(f"Start/Stop/Stride : {args.start} / {args.stop} / {args.stride}")
    print(f"Workers           : {args.workers}")
    print("=" * 64)

    # ── Probe trajectory ───────────────────────────────────────────────────────
    u0      = mda.Universe(args.topology, args.trajectory)
    n_total = len(u0.trajectory)
    print(f"\nTotal trajectory frames : {n_total}")

    start, stop, stride = resolve_frame_range(
        n_total, args.start, args.stop, args.stride)
    n_frames = n_selected_frames(start, stop, stride)
    print(f"Selected frames         : {n_frames}  [{start}:{stop}:{stride}]")

    # ── Load & merge pairs ─────────────────────────────────────────────────────
    print(f"\nLoading pair files ...")
    all_pairs, file_pairs = load_all_pairs(args.pair_files)
    n_pairs = len(all_pairs)
    print(f"Total unique pairs      : {n_pairs}")
    if n_pairs == 0:
        sys.exit("[ERROR] No pairs loaded — check pair file format.")

    # ── Validate atoms ─────────────────────────────────────────────────────────
    print(f"\nValidating CA atoms ...")
    idx1, idx2 = resolve_ca_indices(u0, all_pairs)
    del u0       # release file handles before forking
    print(f"  {n_pairs} pairs validated OK")

    # ── Parallel trajectory processing ────────────────────────────────────────
    slices   = make_worker_slices(start, stop, stride, args.workers)
    n_actual = len(slices)
    print(f"\nLaunching {n_actual} worker(s) ...")

    work_args = [
        (args.topology, args.trajectory, idx1, idx2, sl, wid)
        for wid, sl in enumerate(slices)
    ]

    with Pool(processes=n_actual) as pool:
        results = pool.map(worker, work_args)

    # ── Reassemble in frame order ──────────────────────────────────────────────
    results.sort(key=lambda r: r[0][0])

    all_frames, chunks = [], []
    for frames, data in results:
        all_frames.extend(frames)
        chunks.append(data)

    full = np.hstack(chunks)              # (n_all_pairs, n_frames)
    assert full.shape == (n_pairs, len(all_frames)), \
        f"Shape mismatch after reassembly: {full.shape}"

    pair_to_row = {pair: i for i, pair in enumerate(all_pairs)}

    # ── Write output files ─────────────────────────────────────────────────────
    print(f"\nWriting output files ...")
    for fp in args.pair_files:
        out_path   = make_output_path(fp)
        pairs      = file_pairs[fp]
        rows       = [pair_to_row[p] for p in pairs]
        dist_block = full[rows, :]

        write_output(out_path, pairs, dist_block, all_frames)
        print(f"  -> {out_path}  ({len(pairs)} pairs)")

    print("\nAll done.")


if __name__ == "__main__":
    main()