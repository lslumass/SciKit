"""
Contact Map Calculator — PSF / segid edition (parallel)
========================================================
Auto-detects peptide copies from unique segids.
Heavy atoms used: CA, CB, CC, CD, CE, CF (cutoff 6.0 Å = 0.6 nm).
A residue pair (i,j) is in contact if ANY heavy atom of residue i
is within cutoff of ANY heavy atom of residue j.
Multiple atom-atom hits within the same residue pair are deduplicated
so each (copy, residue-pair, frame) contributes at most 1 count.

Outputs (N x N):
  *_intra.npy  — average contacts per peptide from same-chain pairs
  *_inter.npy  — average contacts per peptide from different-chain pairs

Parallelisation: frames split into chunks, one chunk per worker process.
Each worker owns its own Universe (MDA is not safe to share across processes).

Usage
-----
python contact_map.py --top system.psf --traj traj.dcd
python contact_map.py --top system.psf --traj traj.dcd --cutoff 6.0 --stride 5 --nprocs 8
"""

import argparse
import time
import warnings
import multiprocessing as mp

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

warnings.filterwarnings("ignore")

HEAVY_ATOMS = "name CA CB CC CD CE CF"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top",    required=True)
    p.add_argument("--traj",   default=None)
    p.add_argument("--cutoff", type=float, default=6.0,   help="Distance cutoff in Å (default: 6.0 = 0.6 nm)")
    p.add_argument("--start",  type=int,   default=0,     help="First frame index (default: 0)")
    p.add_argument("--stop",   type=int,   default=None,  help="Last frame index exclusive (default: all)")
    p.add_argument("--stride", type=int,   default=1,     help="Frame stride (default: 1)")
    p.add_argument("--nprocs", type=int,   default=1,     help="Number of parallel worker processes (default: 1)")
    p.add_argument("--out",    default="contact_map.npy")
    return p.parse_args()


def process_chunk(args):
    """
    Worker: process a list of frame indices, return (partial_intra, partial_inter)
    each of shape (N, N).
    """
    top, traj, cutoff, frame_indices, atom_to_local, atom_to_copy, N = args

    u       = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_hvy = u.select_atoms(HEAVY_ATOMS)

    partial_intra = np.zeros((N, N), dtype=np.float64)
    partial_inter = np.zeros((N, N), dtype=np.float64)

    for frame in frame_indices:
        u.trajectory[frame]
        pos_all = all_hvy.positions
        box     = u.dimensions

        # self_capped_distance: same array → unique atom pairs i < j only.
        # No self-pairs, no duplicates.
        pairs, _ = capped_distance(pos_all, pos_all, cutoff,
                                   box=box, return_distances=True)
        if len(pairs) == 0:
            continue

        ri = pairs[:, 0].astype(np.int32)
        rj = pairs[:, 1].astype(np.int32)

        li = atom_to_local[ri]   # residue index (0..N-1) within its copy
        lj = atom_to_local[rj]
        ci = atom_to_copy[ri]    # copy index
        cj = atom_to_copy[rj]

        # ── Intra contacts (same copy) ────────────────────────────────────
        # Exclude 1-2, 1-3, 1-4 pairs (|li-lj| < 4).
        # Multiple atom-atom hits within the same residue pair are collapsed
        # to one by taking unique (ci, li, lj) triples before bincount.
        intra_mask = (ci == cj) & (np.abs(li - lj) >= 4)
        if intra_mask.any():
            # unique columns: (copy, res_i, res_j) — deduplicates multi-atom hits
            key   = np.stack([ci[intra_mask], li[intra_mask], lj[intra_mask]], axis=1)
            key   = np.unique(key, axis=0)
            flat  = key[:, 1] * N + key[:, 2]
            m     = np.bincount(flat, minlength=N * N).reshape(N, N).astype(np.float64)
            partial_intra += m + m.T

        # ── Inter contacts (different copy) ──────────────────────────────
        # Deduplicate multi-atom hits per (copy_i, copy_j, res_i, res_j) tuple.
        inter_mask = ci != cj
        if inter_mask.any():
            key2  = np.stack([ci[inter_mask], cj[inter_mask],
                              li[inter_mask], lj[inter_mask]], axis=1)
            key2  = np.unique(key2, axis=0)
            flat2 = key2[:, 2] * N + key2[:, 3]
            m2    = np.bincount(flat2, minlength=N * N).reshape(N, N).astype(np.float64)
            partial_inter += m2 + m2.T

    return partial_intra, partial_inter


def main():
    args = parse_args()

    # Single Universe load — extract everything before closing
    u       = mda.Universe(args.top, args.traj) if args.traj else mda.Universe(args.top)
    # Use CA only for residue/copy detection (one CA per residue, always present)
    all_ca  = u.select_atoms("name CA")
    segids  = list(dict.fromkeys(all_ca.segids))
    n_copies = len(segids)
    n_traj  = len(u.trajectory)

    seg_ca_list = [u.select_atoms(f"segid {s} and name CA") for s in segids]
    N_list = [len(ca) for ca in seg_ca_list]
    if len(set(N_list)) != 1:
        raise ValueError(f"Copies have unequal residue counts: {set(N_list)}")
    N = N_list[0]

    # Build lookup arrays over ALL heavy atoms: atom row → (local residue, copy).
    # select_atoms(HEAVY_ATOMS) returns only atoms that exist — residues lacking
    # CB/CC/CD/CE/CF (e.g. Gly has no CB) contribute fewer atoms automatically.
    all_hvy     = u.select_atoms(HEAVY_ATOMS)
    M           = len(all_hvy)
    all_hvy_idx = all_hvy.indices   # sorted global atom indices

    atom_to_local = np.empty(M, dtype=np.int32)
    atom_to_copy  = np.empty(M, dtype=np.int32)

    for c, ca in enumerate(seg_ca_list):
        for res_local, res in enumerate(ca.residues):
            # Intersect this residue's atoms with the global heavy-atom set.
            # Works correctly regardless of how many heavy atoms the residue has.
            res_hvy_idx = np.intersect1d(res.atoms.indices, all_hvy_idx)
            rows = np.searchsorted(all_hvy_idx, res_hvy_idx)
            atom_to_local[rows] = res_local
            atom_to_copy[rows]  = c

    # Extract residue IDs from the first copy — same for all copies
    resids = seg_ca_list[0].residues.resids   # shape (N,), actual resid numbers from PSF

    del u

    frames   = list(range(args.start, args.stop or n_traj, args.stride))
    n_frames = len(frames)
    nprocs   = min(args.nprocs, n_frames)

    print(f"Copies   : {n_copies}  ({segids[0]} … {segids[-1]})")
    print(f"Residues : {N} per copy")
    print(f"Heavy atoms per system: {M}  ({HEAVY_ATOMS})")
    print(f"Cutoff   : {args.cutoff} Å  ({args.cutoff/10:.2f} nm)")
    print(f"Frames   : {n_frames}  (stride={args.stride})")
    print(f"Workers  : {nprocs}\n")

    chunks = [frames[i::nprocs] for i in range(nprocs)]
    worker_args = [
        (args.top, args.traj, args.cutoff,
         chunk, atom_to_local, atom_to_copy, N)
        for chunk in chunks
    ]

    t0 = time.perf_counter()

    if nprocs == 1:
        results = [process_chunk(worker_args[0])]
    else:
        # forkserver avoids MKL/OpenBLAS deadlocks on HPC nodes
        ctx = mp.get_context("forkserver")
        with ctx.Pool(processes=nprocs) as pool:
            results = pool.map(process_chunk, worker_args)

    intra_sum, inter_sum = zip(*results)
    denom     = n_copies * n_frames
    intra_map = np.sum(intra_sum, axis=0) / denom
    inter_map = np.sum(inter_sum, axis=0) / denom

    stem = args.out.removesuffix('.npy')
    np.save(f"{stem}_intra.npy", intra_map)
    np.save(f"{stem}_inter.npy", inter_map)
    np.save(f"{stem}.npy", intra_map + inter_map)
    np.save(f"{stem}_resids.npy", resids)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s")
    print(f"Saved  → {stem}_intra.npy    (max {intra_map.max():.4f})")
    print(f"Saved  → {stem}_inter.npy    (max {inter_map.max():.4f})")
    print(f"Saved  → {stem}_combined.npy (max {(intra_map+inter_map).max():.4f})")
    print(f"Saved  → {stem}_resids.npy   (resids {resids[0]}–{resids[-1]})")


if __name__ == "__main__":
    main()
