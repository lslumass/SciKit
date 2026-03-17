"""
Contact Map Calculator — multi-component PSF / segid edition (parallel)
========================================================================
Supports systems with multiple peptide/protein components (e.g. 100 A + 100 B + 100 C).
Heavy atoms: CA CB CC CD CE CF  |  default cutoff 6.0 Å (0.6 nm)

Components are defined by grouping auto-detected segids:
  --components "A:100 B:100 C:100"   (first 100 segids → A, next 100 → B, etc.)

Contact pairs to compute:
  --pairs "A-A A-B B-B"  (default: all same-component pairs)

Output per requested pair X-Y:
  *_X-Y_inter.npy   N_X × N_Y   contacts between different copies
  *_X-X_intra.npy   N_X × N_X   contacts within the same copy  (only for X==Y)
  *_X-Y_resids_X.npy, *_X-Y_resids_Y.npy   residue IDs for each axis

Normalisation: sum over all (copyX, copyY) pairs / (n_X * n_frames)
  → "average number of contacts residue i of one X-copy makes with residue j
     across all Y-copies per frame"

Usage
-----
python contact_map.py --top system.psf --traj traj.dcd \\
    --components "A:100 B:100 C:100" --pairs "A-A A-B"
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


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top",        required=True)
    p.add_argument("--traj",       default=None)
    p.add_argument("--cutoff",     type=float, default=6.0)
    p.add_argument("--components", default=None,
                   help="Component definitions, e.g. 'A:100 B:100 C:100'. "
                        "Default: all segids treated as one component 'A'.")
    p.add_argument("--pairs",      default=None,
                   help="Pairs to compute, e.g. 'A-A A-B'. Default: all same-component pairs.")
    p.add_argument("--start",      type=int, default=0)
    p.add_argument("--stop",       type=int, default=None)
    p.add_argument("--stride",     type=int, default=1)
    p.add_argument("--nprocs",     type=int, default=1)
    p.add_argument("--out",        default="contact_map.npy")
    return p.parse_args()


def parse_components(spec, segids):
    """
    Parse 'A:100 B:100 C:100' into {label: [segid, ...], ...}.
    Segids are assigned in the order they are detected from the PSF.
    If spec is None, all segids are placed in a single component 'A'.
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
            f"Component spec accounts for {idx} segids but {len(segids)} found in PSF."
        )
    return comp_map


def parse_pairs(spec, comp_labels):
    """Parse 'A-A A-B' into list of (labelX, labelY) tuples."""
    pairs = []
    for token in spec.split():
        x, y = token.split("-")
        if x not in comp_labels or y not in comp_labels:
            raise ValueError(f"Unknown component in pair '{token}'. Known: {comp_labels}")
        pairs.append((x, y))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def process_chunk(args):
    """
    Worker: accumulate contact matrices for all requested pairs over a chunk of frames.
    Returns a dict  pair_key → (partial_intra_or_None, partial_inter)
    """
    (top, traj, cutoff, frame_indices,
     atom_to_local, atom_to_copy, atom_to_comp,
     pair_specs) = args
    # pair_specs: list of (compX_idx, compY_idx, N_X, N_Y, n_X, same_comp)

    u       = mda.Universe(top, traj) if traj else mda.Universe(top)
    all_hvy = u.select_atoms(HEAVY_ATOMS)

    # Initialise accumulators for each pair
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

        ri = pairs[:, 0].astype(np.int32)
        rj = pairs[:, 1].astype(np.int32)

        li    = atom_to_local[ri]
        lj    = atom_to_local[rj]
        ci    = atom_to_copy[ri]
        cj    = atom_to_copy[rj]
        cmpi  = atom_to_comp[ri]
        cmpj  = atom_to_comp[rj]

        for spec in pair_specs:
            cx, cy, N_X, N_Y, _, same = spec
            key = (cx, cy)

            if same:
                # ── Intra (same copy, same component) ────────────────────
                intra_mask = (cmpi == cx) & (cmpj == cx) & (ci == cj) & (np.abs(li - lj) >= 4)
                if intra_mask.any():
                    k     = np.unique(np.stack([ci[intra_mask], li[intra_mask], lj[intra_mask]], axis=1), axis=0)
                    flat  = k[:, 1] * N_X + k[:, 2]
                    m     = np.bincount(flat, minlength=N_X * N_X).reshape(N_X, N_X).astype(np.float64)
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
                # Collect pairs where one side is cx and other is cy
                mask_xy = (cmpi == cx) & (cmpj == cy)
                mask_yx = (cmpi == cy) & (cmpj == cx)

                # cx side is always the row axis (li → lj for xy, lj → li for yx)
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    u      = mda.Universe(args.top, args.traj) if args.traj else mda.Universe(args.top)
    all_ca = u.select_atoms("name CA")
    segids = list(dict.fromkeys(all_ca.segids))
    n_traj = len(u.trajectory)

    # ── Component setup ────────────────────────────────────────────────────
    comp_map    = parse_components(args.components, segids)
    comp_labels = list(comp_map.keys())

    # Pair selection
    if args.pairs:
        req_pairs = parse_pairs(args.pairs, comp_labels)
    else:
        req_pairs = [(x, x) for x in comp_labels]   # default: all same-component

    # Per-component CA lists and residue counts
    comp_ca   = {label: [u.select_atoms(f"segid {s} and name CA") for s in segs]
                 for label, segs in comp_map.items()}
    comp_N    = {}
    for label, ca_list in comp_ca.items():
        Ns = [len(ca) for ca in ca_list]
        if len(set(Ns)) != 1:
            raise ValueError(f"Component {label} has unequal residue counts: {set(Ns)}")
        comp_N[label] = Ns[0]

    # Global component index for each label
    comp_idx = {label: i for i, label in enumerate(comp_labels)}

    # ── Lookup arrays: heavy atom row → (local_res, copy_within_comp, comp_idx) ──
    all_hvy     = u.select_atoms(HEAVY_ATOMS)
    M           = len(all_hvy)
    all_hvy_idx = all_hvy.indices

    atom_to_local = np.empty(M, dtype=np.int32)
    atom_to_copy  = np.empty(M, dtype=np.int32)   # copy index within component
    atom_to_comp  = np.empty(M, dtype=np.int32)   # component index

    for label, ca_list in comp_ca.items():
        cidx = comp_idx[label]
        for copy_within, ca in enumerate(ca_list):
            for res_local, res in enumerate(ca.residues):
                res_hvy_idx = np.intersect1d(res.atoms.indices, all_hvy_idx)
                rows = np.searchsorted(all_hvy_idx, res_hvy_idx)
                atom_to_local[rows] = res_local
                atom_to_copy[rows]  = copy_within
                atom_to_comp[rows]  = cidx

    # Residue IDs per component (from first copy of each)
    comp_resids = {label: comp_ca[label][0].residues.resids for label in comp_labels}

    # Number of copies per component
    comp_ncopies = {label: len(segs) for label, segs in comp_map.items()}

    # pair_specs passed to workers
    pair_specs = [
        (comp_idx[x], comp_idx[y], comp_N[x], comp_N[y], comp_ncopies[x], x == y)
        for x, y in req_pairs
    ]

    del u

    frames   = list(range(args.start, args.stop or n_traj, args.stride))
    n_frames = len(frames)
    nprocs   = min(args.nprocs, n_frames)

    print(f"Components : {', '.join(f'{l}×{comp_ncopies[l]}(N={comp_N[l]})' for l in comp_labels)}")
    print(f"Pairs      : {', '.join(f'{x}-{y}' for x,y in req_pairs)}")
    print(f"Cutoff     : {args.cutoff} Å  ({args.cutoff/10:.2f} nm)")
    print(f"Frames     : {n_frames}  (stride={args.stride})")
    print(f"Workers    : {nprocs}\n")

    chunks = [frames[i::nprocs] for i in range(nprocs)]
    worker_args = [
        (args.top, args.traj, args.cutoff,
         chunk, atom_to_local, atom_to_copy, atom_to_comp, pair_specs)
        for chunk in chunks
    ]

    t0 = time.perf_counter()

    if nprocs == 1:
        results = [process_chunk(worker_args[0])]
    else:
        ctx = mp.get_context("forkserver")
        with ctx.Pool(processes=nprocs) as pool:
            results = pool.map(process_chunk, worker_args)

    # ── Merge and save ─────────────────────────────────────────────────────
    stem = args.out.removesuffix('.npy')

    for (x, y), spec in zip(req_pairs, pair_specs):
        cx, cy, N_X, N_Y, n_X, same = spec
        key    = (cx, cy)
        denom  = n_X * n_frames

        inter_sum = sum(r[key]["inter"] for r in results)
        inter_map = inter_sum / denom

        tag = f"{x}-{y}"
        if same:
            intra_sum = sum(r[key]["intra"] for r in results)
            intra_map = intra_sum / denom
            np.save(f"{stem}_{tag}_intra.npy",    intra_map)
            np.save(f"{stem}_{tag}_inter.npy",    inter_map)
            np.save(f"{stem}_{tag}_total.npy",    intra_map + inter_map)
            np.save(f"{stem}_{tag}_resids.npy",   comp_resids[x])
            print(f"Saved → {stem}_{tag}_intra.npy  (max {intra_map.max():.4f})")
            print(f"Saved → {stem}_{tag}_inter.npy  (max {inter_map.max():.4f})")
            print(f"Saved → {stem}_{tag}_total.npy  (max {(intra_map+inter_map).max():.4f})")
            print(f"Saved → {stem}_{tag}_resids.npy")
        else:
            # Cross-component: no intra possible, inter is the total
            np.save(f"{stem}_{tag}_total.npy",    inter_map)
            np.save(f"{stem}_{tag}_resids_{x}.npy", comp_resids[x])
            np.save(f"{stem}_{tag}_resids_{y}.npy", comp_resids[y])
            print(f"Saved → {stem}_{tag}_total.npy  (max {inter_map.max():.4f})")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.2f}s")


if __name__ == "__main__":
    main()