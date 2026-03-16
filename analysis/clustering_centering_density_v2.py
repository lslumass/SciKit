#!/usr/bin/python

'''
Combined script: Recenter droplet across PBC and calculate aggregation statistics
Outputs: aggr.dat file, optional recentered traj, optional density profile
author: Shanlong Li@UMass
date: Nov-17-2025
'''

from __future__ import division
import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree
from collections import deque


# Avogadro's constant
AVOGADRO = 6.022e23


def grp_init(u, step=5):
    """Initialize segment groups"""
    segs = []
    grps = []
    segments = u.segments
    for i, segid in enumerate(segments):
        CAs = u.select_atoms(f'segid {segid.segid} and name CA')
        seg = CAs[::step].atoms.select_atoms('name CA')
        segs.append(seg)
        grps.append(i)
    grps = np.array(grps)
    return segs, grps


def find_clusters_and_stats(segs, grps, r_cutoff=8.0):
    """
    Find all clusters and calculate statistics
    
    Returns:
    --------
    clusters : list of lists
        Each sublist contains segment indices in that cluster
    monomer : int
        Number of monomers (clusters of size 1)
    n_clusters : int
        Number of clusters with size > 1
    max_cluster_size : int
        Size of largest cluster
    grps : array
        Updated group assignments
    """
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
    """
    Unwrap a cluster that may be split across periodic boundaries
    
    Returns:
    --------
    shifts : dict
        Dictionary mapping segment index to shift vector [dx, dy, dz]
    """
    segments = u.segments
    box_half = box[:3] / 2.0
    
    # Start with first segment as reference (no shift)
    shifts = {cluster_seg_indices[0]: np.array([0.0, 0.0, 0.0])}
    
    # BFS to unwrap connected segments
    queue = deque([cluster_seg_indices[0]])
    visited = {cluster_seg_indices[0]}
    
    while queue:
        ref_idx = queue.popleft()
        ref_seg = segments[ref_idx]
        ref_atoms = ref_seg.atoms
        ref_pos = ref_atoms.positions + shifts[ref_idx]
        ref_com = ref_pos.mean(axis=0)
        
        # Check all other segments in cluster
        for seg_idx in cluster_seg_indices:
            if seg_idx in visited:
                continue
            
            seg = segments[seg_idx]
            seg_atoms = seg.atoms
            seg_pos = seg_atoms.positions.copy()
            seg_com = seg_pos.mean(axis=0)
            
            # Calculate distance considering PBC
            delta = seg_com - ref_com
            
            # Find the minimum image
            shift = np.zeros(3)
            for dim in range(3):
                if delta[dim] > box_half[dim]:
                    shift[dim] = -box[dim]
                elif delta[dim] < -box_half[dim]:
                    shift[dim] = box[dim]
            
            # Accumulate shifts
            shifts[seg_idx] = shifts[ref_idx] + shift
            visited.add(seg_idx)
            queue.append(seg_idx)
    
    return shifts


def recenter_frame(u, largest_cluster, box):
    """
    Recenter the largest cluster to the box center
    
    Returns:
    --------
    new_positions : array
        New atomic positions with recentered droplet
    """
    segments = u.segments
    box_center = box[:3] / 2.0
    
    # Unwrap the largest cluster
    shifts = unwrap_cluster(u, largest_cluster, box)
    
    # Apply shifts to unwrap cluster atoms
    new_positions = u.atoms.positions.copy()
    for seg_idx in largest_cluster:
        seg = segments[seg_idx]
        atom_indices = seg.atoms.indices
        new_positions[atom_indices] += shifts[seg_idx]
    
    # Calculate COM of unwrapped cluster
    cluster_indices = []
    for seg_idx in largest_cluster:
        cluster_indices.extend(segments[seg_idx].atoms.indices)
    
    cluster_com = new_positions[cluster_indices].mean(axis=0)
    
    # Calculate shift to center the cluster
    center_shift = box_center - cluster_com
    
    # Apply shift to all atoms
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


def calculate_radial_density(u, droplet_center, segments, ca_per_segment,
                            r_max=100.0, dr=1.0, box=None):
    """
    Calculate radial monomer concentration profile from droplet center
    
    Parameters:
    -----------
    u : MDAnalysis Universe
    droplet_center : array
        Center of the droplet [x, y, z]
    segments : list
        List of segments (used to count CA per segment)
    ca_per_segment : int
        Number of CA atoms per segment (precomputed)
    r_max : float
        Maximum radius for density calculation (Angstrom)
    dr : float
        Bin width for radial distribution (Angstrom)
    box : array
        Box dimensions
    
    Returns:
    --------
    r_bins : array
        Radial bin centers (Angstrom)
    concentration_mM : array
        Monomer concentration in mM at each radial bin
    """
    
    # Select all CA atoms
    all_ca = u.select_atoms('name CA')
    
    # Calculate distances from droplet center for each CA (vectorized)
    ca_positions = all_ca.positions
    distances = np.sqrt(np.sum((ca_positions - droplet_center)**2, axis=1))
    
    # Create radial bins
    n_bins = int(r_max / dr)
    bins = np.linspace(0, r_max, n_bins + 1)
    r_bins = (bins[:-1] + bins[1:]) / 2.0  # Bin centers
    
    # Calculate histogram of CA atoms (fast)
    ca_counts, _ = np.histogram(distances, bins=bins)
    
    # Convert CA counts to monomer counts
    monomer_counts = ca_counts / ca_per_segment
    
    # Calculate volume of each spherical shell (vectorized)
    bins_cubed = bins**3
    shell_volumes = (4.0 / 3.0) * np.pi * (bins_cubed[1:] - bins_cubed[:-1])
    
    # Convert to mM (millimolar) - vectorized
    # 1 molecule/Angstrom^3 = (1e30/Avogadro) mmol/L
    concentration_mM = (monomer_counts / shell_volumes) * (1e30 / AVOGADRO)
    
    return r_bins, concentration_mM


def process_trajectory(psf_file, xtc_file, output_traj='recentered.xtc', 
                      output_dat='aggr.dat', output_density='density_profile.dat',
                      r_cutoff=8.0, step=3, start=0, stop=None, stride=1, 
                      recenter=True, calc_density=True,
                      dr=1.0, n_frames_avg=50):
    """
    Main function: Calculate aggregation statistics and optionally recenter trajectory
    
    Parameters:
    -----------
    psf_file : str
        PSF topology file
    xtc_file : str
        Input trajectory file
    output_traj : str
        Output recentered trajectory (only used if recenter=True)
    output_dat : str
        Output aggregation statistics file
    output_density : str
        Output radial density profile file
    r_cutoff : float
        Distance cutoff for clustering
    step : int
        Step size for selecting CA atoms
    start, stop, stride : int
        Frame range to process
    recenter : bool
        If True, recenter the largest cluster and write trajectory
        If False, only calculate statistics (much faster)
    calc_density : bool
        If True, calculate radial monomer concentration profile (requires recenter=True)
    dr : float
        Bin width for radial distribution (Angstrom)
    n_frames_avg : int
        Number of last frames to average for density profile
    """
    
    # Load universe
    print("Loading trajectory...")
    u = mda.Universe(psf_file, xtc_file)
    
    # Initialize segments
    segs, grps_init = grp_init(u, step=step)
    
    # Precompute CA atoms per segment
    first_seg = u.segments[0]
    ca_per_segment = len(first_seg.atoms.select_atoms('name CA'))
    print(f"CA atoms per segment: {ca_per_segment}")
    
    # Preselect all CA atoms for density calculation (speeds up repeated selections)
    if calc_density:
        all_ca = u.select_atoms('name CA')
    
    n_frames = len(u.trajectory[start:stop:stride])
    print(f"Number of segments: {len(segs)}")
    print(f"Processing {n_frames} frames...")
    if recenter:
        print(f"Recentering enabled: will write trajectory to {output_traj}")
    else:
        print(f"Recentering disabled: only calculating statistics")
    
    if calc_density and not recenter:
        print("Warning: calc_density=True but recenter=False. Density calculation requires recentering.")
        calc_density = False
    
    if calc_density:
        print(f"Radial density calculation enabled: averaging last {n_frames_avg} frames")
        print(f"Calculating monomer concentration based on CA atoms")
        print(f"r_max will be set to half box size automatically")
    
    # Storage for statistics and density profiles
    stats_data = []
    density_profiles = []
    
    # Open trajectory writer only if recentering
    writer = mda.Writer(output_traj, n_atoms=u.atoms.n_atoms) if recenter else None
    
    try:
        for ts_idx, ts in enumerate(u.trajectory[start:stop:stride]):
            
            if ts_idx % 10 == 0 or ts_idx == 0:
                print(f"Processing frame {ts_idx + 1}/{n_frames} (frame {ts.frame})")
            
            box = ts.dimensions
            
            # Update segment positions (wrap first)
            for seg in segs:
                seg.wrap()
            
            # Find clusters and calculate statistics
            clusters, monomer, n_clusters, max_size, _ = find_clusters_and_stats(
                segs, grps_init, r_cutoff=r_cutoff
            )
            
            if ts_idx % 10 == 0 or ts_idx == 0:
                print(f"  Monomers: {monomer}, Clusters: {n_clusters}, Largest: {max_size}")
            
            # Only recenter if requested
            if recenter:
                # Find the largest cluster
                largest_cluster = max(clusters, key=len)
                
                # Recenter the largest cluster
                new_positions = recenter_frame(u, largest_cluster, box)
                
                # Update positions
                u.atoms.positions = new_positions
                
                # Write recentered frame
                writer.write(u.atoms)
                
                # Calculate radial density for last n_frames_avg frames
                if calc_density and ts_idx >= n_frames - n_frames_avg:
                    # Droplet center is now at box center
                    droplet_center = box[:3] / 2.0
                    # Use half of minimum box dimension as r_max
                    r_max = min(box[:3]) / 2.0
                    
                    r_bins, concentration = calculate_radial_density(
                        u, droplet_center, 
                        segments=u.segments,
                        ca_per_segment=ca_per_segment,
                        r_max=r_max, dr=dr, box=box
                    )
                    density_profiles.append(concentration)
            
            # Store statistics (frame, monomer, n_clusters, max_size)
            stats_data.append((ts.frame, monomer, n_clusters, max_size))
    
    finally:
        # Close writer if it was opened
        if writer is not None:
            writer.close()
    
    # Write aggregation statistics
    print("\nWriting aggregation statistics to file...")
    with open(output_dat, 'w') as fout:
        fout.write("# Frame  Monomers  Clusters  LargestClusterSize\n")
        for frame, monomer, n_clusters, max_size in stats_data:
            print(frame, monomer, n_clusters, max_size, file=fout)
    
    # Write average radial density profile
    if calc_density and len(density_profiles) > 0:
        print(f"\nWriting average radial monomer concentration profile to file...")
        avg_concentration = np.mean(density_profiles, axis=0)
        std_concentration = np.std(density_profiles, axis=0)
        
        with open(output_density, 'w') as fout:
            fout.write(f"# Radial monomer concentration profile averaged over last {len(density_profiles)} frames\n")
            fout.write(f"# Concentration calculated from CA atoms\n")
            fout.write("# Radius(Angstrom)  Concentration(mM)  StdDev(mM)\n")
            for r, conc, std in zip(r_bins, avg_concentration, std_concentration):
                print(f"{r:.3f}  {conc:.6f}  {std:.6f}", file=fout)
        
        print(f"Radial monomer concentration profile written to: {output_density}")
    
    if recenter:
        print(f"\nRecentered trajectory written to: {output_traj}")
    print(f"Aggregation statistics written to: {output_dat}")
    print("\nDONE!!")


# Main execution
if __name__ == '__main__':
    
    # Input/Output files
    psf_file = 'conf.psf'
    xtc_file = 'system.xtc'
    output_traj = 'recentered.xtc'
    output_dat = 'aggr.dat'
    output_density = 'density_profile.dat'
    
    # Parameters
    r_cutoff = 8.0          # Distance cutoff for clustering (Angstrom)
    step = 1                 # Use every 3rd CA atom for clustering
    recenter = False         # Set to False to only calculate statistics (faster)
    
    # Density calculation parameters
    calc_density = False     # Calculate radial monomer concentration profile
    dr = 2.0                # Bin width for radial distribution (Angstrom)
    n_frames_avg = 50       # Number of last frames to average for density profile
    
    # Frame range
    start_frame = 0
    stop_frame = None        # None means all frames
    stride = 100             # Process every frame (set to 100 for every 100th frame)
    
    # Run analysis
    process_trajectory(
        psf_file=psf_file,
        xtc_file=xtc_file,
        output_traj=output_traj,
        output_dat=output_dat,
        output_density=output_density,
        r_cutoff=r_cutoff,
        step=step,
        start=start_frame,
        stop=stop_frame,
        stride=stride,
        recenter=recenter,
        calc_density=calc_density,
        dr=dr,
        n_frames_avg=n_frames_avg
    )
