import typer
import numpy as np
import MDAnalysis as mda


app = typer.Typer()

@app.command()
def rg(top: str='conf.psf', 
       traj: str='system.xtc',
       out: str='rgs.dat',
       sel: str='all',
       start: int=1,
       step: int=1,
       stop: int=-1,
       help="calculate Rg"
    ):
    
    u = mda.Universe(top, traj)
    selection = u.select_atoms(sel)
    rgs = []
    with open(out, 'w') as f:
        print("# time  Rg(A)", file=f)
        for ts in u.trajectory[start:stop:step]:
            val = selection.radius_of_gyration()
            rgs.append(val)
            print(ts.time, f"{val:.2f}", file=f)
        
    ave = np.mean(rgs)
    sd = np.std(rgs, ddof=1)
    print(f"Finish! Rg = {ave:.2f} +- {sd:.2f} A")

def re(top: str='conf.psf', 
       traj: str='system.xtc',
       out: str='res.dat',
       atom1: str='resid 1 and name CA',
       atom2: str='resid -1 and name CA',
       start: int=1,
       step: int=1,
       stop: int=-1,
       help="calculate end-to-end distance"
    ):
    
    from MDAnalysis.analysis import distances
    u = mda.Universe(top, traj)
    sel1 = u.select_atoms(atom1)
    sel2 = u.select_atoms(atom2)
    res = []
    with open(out, 'w') as f:
        print("# time  Re(A)", file=f)
        for ts in u.trajectory[start:stop:step]:
            val = distances.dist(sel1, sel2)[2]
            res.append(val)
            print(ts.time, f"{val:.2f}", file=f)
    
    ave = np.mean(res)
    sd = np.std(res, ddof=1)
    print(f"Finish! Re = {ave:.2f} +- {sd:.2f} A")

@app.command()
def ssp(top: str='conf.psf', 
        traj: str='system.xtc',
        sel: str='all',
        start: int=1,
        step: int=1,
        stop: int=-1,
        help="calculate SSP: helicity.dat and beta.dat"
    ):
    
    from MDAnalysis.analysis.dssp import DSSP
    u = mda.Universe(top, traj)
    selection = u.select_atoms(sel)
    resids = selection.residues.resids.tolist()
    dssp = DSSP(selection)
    dssp.run(start=start, step=step, stop=stop)

    # get helicity
    results = np.where(dssp.results.dssp == 'H', 1, 0)
    half = int(dssp.n_frames/2)
    avgs1 = np.mean(results[:half], axis=0)
    avgs2 = np.mean(results[half:], axis=0)
    with open('helicity.dat', 'w') as f:
        for res, avg1, avg2 in zip(resids, avgs1, avgs2):
            avg, std = np.mean([avg1, avg2]), np.std([avg1, avg2])
            print(res, avg, std, file=f)

    # get beta propensity
    results = np.where(dssp.results.dssp == 'E', 1, 0)
    half = int(dssp.n_frames/2)
    avgs1 = np.mean(results[:half], axis=0)
    avgs2 = np.mean(results[half:], axis=0)
    with open('beta.dat', 'w') as f:
        for res, avg1, avg2 in zip(resids, avgs1, avgs2):
            avg, std = np.mean([avg1, avg2]), np.std([avg1, avg2])
            print(res, avg, std, file=f)
    print('Finish!')


if __name__ == "__main__":
    app()