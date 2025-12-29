import typer
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from typing import Optional
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.dssp import DSSP
from collections import deque
from scipy.spatial import cKDTree


app = typer.Typer(help="HyRes/iConRNA analysis tools")


def load_universe(top: str, traj: str) -> mda.Universe:
    """Load MDAnalysis Universe with error handling."""
    try:
        return mda.Universe(top, traj)
    except Exception as e:
        typer.echo(f"Error loading trajectory: {e}", err=True)
        raise typer.Exit(code=1)


def write_statistics(values: list[float], metric_name: str, unit: str = "A") -> None:
    """Calculate and print statistics for a metric."""
    ave = np.mean(values)
    sd = np.std(values, ddof=1)
    typer.echo(f"Finished! {metric_name} = {ave:.2f} ± {sd:.2f} {unit}")


# calculate radius of gyration
@app.command()
def rg(
    top: str = typer.Option("conf.psf", help="Topology file"),
    traj: str = typer.Option("system.xtc", help="Trajectory file"),
    out: str = typer.Option("rgs.dat", help="Output file"),
    sel: str = typer.Option("all", help="Atom selection"),
    start: Optional[int] = typer.Option(None, help="Start frame"),
    step: int = typer.Option(1, help="Frame step"),
    stop: Optional[int] = typer.Option(None, help="Stop frame"),
) -> None:
    """Calculate radius of gyration (Rg) over trajectory."""
    u = load_universe(top, traj)
    
    try:
        selection = u.select_atoms(sel)
    except Exception as e:
        typer.echo(f"Error in selection '{sel}': {e}", err=True)
        raise typer.Exit(code=1)
    
    if len(selection) == 0:
        typer.echo(f"Warning: Selection '{sel}' returned 0 atoms", err=True)
        raise typer.Exit(code=1)
    
    rgs = []
    with open(out, "w") as f:
        f.write("# time(ps)  Rg(A)\n")
        for ts in u.trajectory[start:stop:step]:
            val = selection.radius_of_gyration()
            rgs.append(val)
            f.write(f"{ts.time:.2f}  {val:.2f}\n")
    
    from .utils import block_bootstrap as bb
    rgs = np.asarray(rgs)
    avg1, sd1 = np.mean(rgs), bb(rgs, statistic=np.mean)
    avg2, sd2 = np.sqrt(np.mean(rgs**2)), bb(rgs)
    typer.echo(f"Finished!\n  <Rg> = {avg1:.2f} ± {sd1:.2f} A\n  RMS Rg = {avg2:.2f} ± {sd2:.2f} A")
    typer.echo(f"Results written to {out}")


# calculate end-to-end distance
@app.command()
def re(
    top: str = typer.Option("conf.psf", help="Topology file"),
    traj: str = typer.Option("system.xtc", help="Trajectory file"),
    out: str = typer.Option("res.dat", help="Output file"),
    atom1: str = typer.Option("resid 1 and name CA", help="First atom selection"),
    atom2: str = typer.Option("resid -1 and name CA", help="Second atom selection"),
    start: Optional[int] = typer.Option(None, help="Start frame"),
    step: int = typer.Option(1, help="Frame step"),
    stop: Optional[int] = typer.Option(None, help="Stop frame"),
) -> None:
    """Calculate end-to-end distance (Re) between two atom selections."""
    u = load_universe(top, traj)
    
    try:
        sel1 = u.select_atoms(atom1)
        sel2 = u.select_atoms(atom2)
    except Exception as e:
        typer.echo(f"Error in atom selection: {e}", err=True)
        raise typer.Exit(code=1)
    
    if len(sel1) == 0 or len(sel2) == 0:
        typer.echo("Error: One or both selections returned 0 atoms", err=True)
        raise typer.Exit(code=1)
    
    res = []
    with open(out, "w") as f:
        f.write("# time(ps)  Re(A)\n")
        for ts in u.trajectory[start:stop:step]:
            val = distances.dist(sel1, sel2)[2][0]
            res.append(val)
            f.write(f"{ts.time:.2f}  {val:.2f}\n")
    
    write_statistics(res, "Re")
    typer.echo(f"Results written to {out}")


# DSSP calculate secondary structure
@app.command()
def ssp(
    top: str = typer.Option("conf.psf", help="Topology file"),
    traj: str = typer.Option("system.xtc", help="Trajectory file"),
    sel: str = typer.Option("all", help="Atom selection"),
    start: Optional[int] = typer.Option(None, help="Start frame"),
    step: int = typer.Option(1, help="Frame step"),
    stop: Optional[int] = typer.Option(None, help="Stop frame"),
    out_helix: str = typer.Option("helicity.dat", help="Helix output file"),
    out_beta: str = typer.Option("beta.dat", help="Beta sheet output file"),
) -> None:
    """Calculate secondary structure propensities (SSP) using DSSP."""
    u = load_universe(top, traj)
    
    try:
        selection = u.select_atoms(sel)
    except Exception as e:
        typer.echo(f"Error in selection: {e}", err=True)
        raise typer.Exit(code=1)
    
    resids = selection.residues.resids
    
    typer.echo("Running DSSP analysis...")
    dssp = DSSP(selection)
    dssp.run(start=start, step=step, stop=stop)
    
    n_frames = dssp.n_frames
    half = n_frames // 2
    
    # Calculate helicity
    helix_results = (dssp.results.dssp == "H").astype(int)
    avgs1 = np.mean(helix_results[:half], axis=0)
    avgs2 = np.mean(helix_results[half:], axis=0)
    
    with open(out_helix, "w") as f:
        f.write("# resid  avg_helicity  std_helicity\n")
        for res, avg1, avg2 in zip(resids, avgs1, avgs2):
            avg = np.mean([avg1, avg2])
            std = np.std([avg1, avg2], ddof=1)
            f.write(f"{res}  {avg:.4f}  {std:.4f}\n")
    
    # Calculate beta propensity
    beta_results = (dssp.results.dssp == "E").astype(int)
    avgs1 = np.mean(beta_results[:half], axis=0)
    avgs2 = np.mean(beta_results[half:], axis=0)
    
    with open(out_beta, "w") as f:
        f.write("# resid  avg_beta  std_beta\n")
        for res, avg1, avg2 in zip(resids, avgs1, avgs2):
            avg = np.mean([avg1, avg2])
            std = np.std([avg1, avg2], ddof=1)
            f.write(f"{res}  {avg:.4f}  {std:.4f}\n")
    
    typer.echo(f"Finished! Results written to {out_helix} and {out_beta}")


# ============================================
# Aggregation Analysis Functions
# clustering, recentering, density profile
# ============================================
# Constants
AVOGADRO = 6.022e23

def grp_init(u, ref_atom: str = "CA", step: int = 1):
    """Initialize segment groups for clustering."""
    segs = []
    grps = []
    segments = u.segments
    for i, segid in enumerate(segments):
        CAs = u.select_atoms(f'segid {segid.segid} and name {ref_atom}')
        seg = CAs[::step].atoms.select_atoms(f'name {ref_atom}')
        segs.append(seg)
        grps.append(i)
    return segs, np.array(grps)


def find_clusters_and_stats(segs, grps, r_cutoff: float = 8.0):
    """Find all clusters and calculate statistics."""
    nmol = len(segs)
    grps = grps.copy()
    
    # Build contact-based clusters
    for i in range(nmol - 1):
        seg_i = segs[i]
        grp_i = grps[i]
        tree_i = cKDTree(seg_i.positions)
        
        for j in range(i + 1, nmol):
            grp_j = grps[j]
            if grp_i == grp_j:
                continue
            
            seg_j = segs[j]
            tree_j = cKDTree(seg_j.positions)
            
            neighbors = tree_i.query_ball_tree(tree_j, r=r_cutoff)
            has_contact = any(len(nb) > 0 for nb in neighbors)
            
            if has_contact:
                grps = np.where(grps == grp_j, grp_i, grps)
    
    # Calculate statistics
    aggr = {}
    for grp in grps:
        aggr[grp] = aggr.get(grp, 0) + 1
    
    monomer = sum(1 for v in aggr.values() if v == 1)
    n_clusters = len(aggr) - monomer
    max_cluster_size = max(aggr.values())
    
    # Group segments by cluster
    clusters = {}
    for seg_idx, cid in enumerate(grps):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(seg_idx)
    
    return list(clusters.values()), monomer, n_clusters, max_cluster_size, grps


def unwrap_cluster(u, cluster_seg_indices, box):
    """Unwrap a cluster that may be split across periodic boundaries."""
    segments = u.segments
    box_half = box[:3] / 2.0
    
    shifts = {cluster_seg_indices[0]: np.array([0.0, 0.0, 0.0])}
    queue = deque([cluster_seg_indices[0]])
    visited = {cluster_seg_indices[0]}
    
    while queue:
        ref_idx = queue.popleft()
        ref_seg = segments[ref_idx]
        ref_atoms = ref_seg.atoms
        ref_pos = ref_atoms.positions + shifts[ref_idx]
        ref_com = ref_pos.mean(axis=0)
        
        for seg_idx in cluster_seg_indices:
            if seg_idx in visited:
                continue
            
            seg = segments[seg_idx]
            seg_atoms = seg.atoms
            seg_pos = seg_atoms.positions.copy()
            seg_com = seg_pos.mean(axis=0)
            
            delta = seg_com - ref_com
            shift = np.zeros(3)
            for dim in range(3):
                if delta[dim] > box_half[dim]:
                    shift[dim] = -box[dim]
                elif delta[dim] < -box_half[dim]:
                    shift[dim] = box[dim]
            
            shifts[seg_idx] = shifts[ref_idx] + shift
            visited.add(seg_idx)
            queue.append(seg_idx)
    
    return shifts


def recenter_frame(u, largest_cluster, box):
    """Recenter the largest cluster to the box center."""
    segments = u.segments
    box_center = box[:3] / 2.0
    
    shifts = unwrap_cluster(u, largest_cluster, box)
    new_positions = u.atoms.positions.copy()
    
    for seg_idx in largest_cluster:
        seg = segments[seg_idx]
        atom_indices = seg.atoms.indices
        new_positions[atom_indices] += shifts[seg_idx]
    
    cluster_indices = []
    for seg_idx in largest_cluster:
        cluster_indices.extend(segments[seg_idx].atoms.indices)
    
    cluster_com = new_positions[cluster_indices].mean(axis=0)
    center_shift = box_center - cluster_com
    new_positions += center_shift
    
    # Wrap back into box
    for dim in range(3):
        new_positions[:, dim] = np.where(
            new_positions[:, dim] < 0,
            new_positions[:, dim] + box[dim],
            new_positions[:, dim]
        )
        new_positions[:, dim] = np.where(
            new_positions[:, dim] >= box[dim],
            new_positions[:, dim] - box[dim],
            new_positions[:, dim]
        )
    
    return new_positions


def calculate_radial_density(u, droplet_center, segments, ca_per_segment, ref_atom: str = "CA",
                            r_max: float = 100.0, dr: float = 1.0, box=None):
    """Calculate radial monomer concentration profile from droplet center."""
    all_ca = u.select_atoms(f'name {ref_atom}')
    ca_positions = all_ca.positions
    distances = np.sqrt(np.sum((ca_positions - droplet_center)**2, axis=1))
    
    n_bins = int(r_max / dr)
    bins = np.linspace(0, r_max, n_bins + 1)
    r_bins = (bins[:-1] + bins[1:]) / 2.0
    
    ca_counts, _ = np.histogram(distances, bins=bins)
    monomer_counts = ca_counts / ca_per_segment
    
    bins_cubed = bins**3
    shell_volumes = (4.0 / 3.0) * np.pi * (bins_cubed[1:] - bins_cubed[:-1])
    concentration_mM = (monomer_counts / shell_volumes) * (1e30 / AVOGADRO)
    
    return r_bins, concentration_mM


@app.command()
def aggr(
    top: str = typer.Option("conf.psf", help="Topology file (PSF)"),
    traj: str = typer.Option("system.xtc", help="Trajectory file"),
    out: str = typer.Option("aggr.dat", help="Output aggregation statistics file"),
    outtraj: str = typer.Option("recentered.xtc", help="Output recentered trajectory"),
    profile: str = typer.Option("density_profile.dat", help="Output density profile"),
    ref: str = typer.Option("CA", help="Atoms used for calculation, CA for peptide, P for RNA"),
    rcut: float = typer.Option(8.0, help="Distance cutoff for clustering (Angstrom)"),
    CAstep: int = typer.Option(1, help="Use every Nth CA atom for clustering"),
    start: Optional[int] = typer.Option(None, help="Start frame"),
    stop: Optional[int] = typer.Option(None, help="Stop frame"),
    stride: int = typer.Option(1, help="Frame stride"),
    recenter: bool = typer.Option(False, help="Recenter largest cluster and write trajectory"),
    density: bool = typer.Option(False, help="Calculate radial density profile"),
    dr: float = typer.Option(2.0, help="Bin width for radial distribution (Angstrom)"),
    n_frames_avg: int = typer.Option(50, help="Number of last frames to average for density"),
) -> None:
    """
    Calculate aggregation statistics and optionally recenter droplet across PBC.
    
    Analyzes protein aggregation by identifying clusters based on CA atom distances.
    Can optionally recenter the largest cluster and calculate radial density profiles.
    """
    
    typer.echo("Loading trajectory...")
    u = load_universe(top, traj)
    
    # Initialize segments
    segs, grps_init = grp_init(u, ref_atom=ref, step=CAstep)
    
    # Precompute ref atoms per segment
    first_seg = u.segments[0]
    ca_per_segment = len(first_seg.atoms.select_atoms(f'name {ref}'))
    typer.echo(f"{ref} atoms per segment: {ca_per_segment}")
    
    n_frames = len(u.trajectory[start:stop:stride])
    typer.echo(f"Number of segments: {len(segs)}")
    typer.echo(f"Processing {n_frames} frames...")
    
    if recenter:
        typer.echo(f"Recentering enabled: will write trajectory to {outtraj}")
    else:
        typer.echo("Recentering disabled: only calculating statistics")
    
    if density and not recenter:
        typer.echo("Warning: calc_density requires recenter=True. Disabling density calculation.")
        density = False
    
    if density:
        typer.echo(f"Radial density calculation enabled: averaging last {n_frames_avg} frames")
    
    stats_data = []
    density_profiles = []
    
    writer = mda.Writer(outtraj, n_atoms=u.atoms.n_atoms) if recenter else None
    
    try:
        for ts_idx, ts in enumerate(u.trajectory[start:stop:stride]):
            
            if ts_idx % 10 == 0 or ts_idx == 0:
                typer.echo(f"Processing frame {ts_idx + 1}/{n_frames} (frame {ts.frame})")
            
            box = ts.dimensions
            
            # Update segment positions
            for seg in segs:
                seg.wrap()
            
            # Find clusters and calculate statistics
            clusters, monomer, n_clusters, max_size, _ = find_clusters_and_stats(
                segs, grps_init, r_cutoff=rcut
            )
            
            if ts_idx % 10 == 0 or ts_idx == 0:
                typer.echo(f"  Monomers: {monomer}, Clusters: {n_clusters}, Largest: {max_size}")
            
            if recenter:
                largest_cluster = max(clusters, key=len)
                new_positions = recenter_frame(u, largest_cluster, box)
                u.atoms.positions = new_positions
                writer.write(u.atoms)
                
                if density and ts_idx >= n_frames - n_frames_avg:
                    droplet_center = box[:3] / 2.0
                    r_max = min(box[:3]) / 2.0
                    
                    r_bins, concentration = calculate_radial_density(
                        u, droplet_center,
                        segments=u.segments,
                        ca_per_segment=ca_per_segment,
                        ref_atom=ref,
                        r_max=r_max, dr=dr, box=box
                    )
                    density_profiles.append(concentration)
            
            stats_data.append((ts.frame, monomer, n_clusters, max_size))
    
    finally:
        if writer is not None:
            writer.close()
    
    # Write aggregation statistics
    typer.echo("\nWriting aggregation statistics...")
    with open(out, 'w') as fout:
        fout.write("# Frame  Monomers  Clusters  LargestClusterSize\n")
        for frame, monomer, n_clusters, max_size in stats_data:
            fout.write(f"{frame}  {monomer}  {n_clusters}  {max_size}\n")
    
    # Write density profile
    if density and len(density_profiles) > 0:
        typer.echo("\nWriting radial monomer concentration profile...")
        avg_concentration = np.mean(density_profiles, axis=0)
        std_concentration = np.std(density_profiles, axis=0)
        
        with open(profile, 'w') as fout:
            fout.write(f"# Radial monomer concentration profile averaged over last {len(density_profiles)} frames\n")
            fout.write("# Radius(Angstrom)  Concentration(mM)  StdDev(mM)\n")
            for r, conc, std in zip(r_bins, avg_concentration, std_concentration):
                fout.write(f"{r:.3f}  {conc:.6f}  {std:.6f}\n")
        
        typer.echo(f"Density profile written to: {profile}")
    
    if recenter:
        typer.echo(f"\nRecentered trajectory written to: {outtraj}")
    typer.echo(f"Aggregation statistics written to: {out}")
    typer.echo("\nDONE!")




if __name__ == "__main__":
    app()