import typer
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from typing import Optional
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.dssp import DSSP


app = typer.Typer(help="Molecular dynamics trajectory analysis tools")


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


@app.command()
def cal_rg(
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
    
    write_statistics(rgs, "Rg")
    typer.echo(f"Results written to {out}")


@app.command()
def cal_re(
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


@app.command()
def cal_ssp(
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


if __name__ == "__main__":
    app()