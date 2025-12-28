import typer
from typing import Optional
from HyresBuilder import utils
from openmm.app import *
from openmm import *
import parmed as chem


app = typer.Typer(help="Hyres/iConRNA tools")


# convert psf to top and gro files
@app.command()
def psf2top(
    pdb: str = typer.Option("conf.pdb", help="Input pdb file"),
    psf: str = typer.Option("conf.psf", help="Input psf file"),
    top: str = typer.Option("conf.top", help="Output top file"),
    gro: str = typer.Option("conf.gro", help="Output gro file"),
    box: float = typer.Option(100.0, help='Box size'),
) -> None:
    top_RNA, param_RNA = utils.load_ff('RNA')
    top_pro, param_pro = utils.load_ff('Protein')
    pdb = PDBFile(pdb)
    psf = CharmmPsfFile(psf)
    top = psf.topology
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)
    psf.setBox(box, box, box)
    system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds)

    structure = chem.openmm.load_topology(top, system)
    structure.save(f'./{top}', overwrite=True)
    structure.coordinates = pdb.positions
    structure.save(f'./{gro}', overwrite=True)
    typer.echo(f'Convert to {top} and {gro} files.')


@app.command()
def psftop(
    pdb: str = typer.Option("conf.pdb", help="Input pdb file"),
    psf: str = typer.Option("conf.psf", help="Input psf file"),
    top: str = typer.Option("conf.top", help="Output top file"),
    gro: str = typer.Option("conf.gro", help="Output gro file"),
    box: float = typer.Option(100.0, help='Box size'),
) -> None:
    top_RNA, param_RNA = utils.load_ff('RNA')
    top_pro, param_pro = utils.load_ff('Protein')
    pdb = PDBFile(pdb)
    psf = CharmmPsfFile(psf)
    top = psf.topology
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)
    psf.setBox(box, box, box)
    system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds)

    structure = chem.openmm.load_topology(top, system)
    structure.save(top, overwrite=True)
    structure.coordinates = pdb.positions
    structure.save(gro, overwrite=True)
    typer.echo(f'Convert to {top} and {gro} files.')


if __name__ == "__main__":
    app()