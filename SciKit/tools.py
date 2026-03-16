"""
Command-line interface for HyresBuilder / iConRNA utilities.

Provides Typer-based CLI commands for converting CHARMM structure files
(PSF + PDB) to GROMACS-compatible topology and coordinate files (.top / .gro)
using OpenMM and ParmEd.

Commands
--------
psf2top
    Convert a CHARMM PSF + PDB pair to a GROMACS .top and .gro file.
    Loads built-in RNA and Protein force-field parameters, builds an
    OpenMM system, and serialises the result via ParmEd.

psftop
    Equivalent to ``psf2top`` with abbreviated option names. Retained for
    backwards compatibility.

Usage
-----
Run any command with ``--help`` for a full option listing::

    python cli.py psf2top --help
    python cli.py psftop  --help
"""

import typer
from typing import Optional
from HyresBuilder import utils
from openmm.app import *
from openmm import *
import parmed as chem


app = typer.Typer(help="Hyres/iConRNA tools")


@app.command()
def psf2top(
    pdb: str = typer.Option("conf.pdb", help="Input PDB coordinate file."),
    psf: str = typer.Option("conf.psf", help="Input CHARMM PSF topology file."),
    outtop: str = typer.Option("conf.top", help="Output GROMACS topology file (.top)."),
    outgro: str = typer.Option("conf.gro", help="Output GROMACS coordinate file (.gro)."),
    box: float = typer.Option(100.0, help="Cubic periodic box side length in Angstroms."),
) -> None:
    """
    Convert a CHARMM PSF + PDB pair to GROMACS topology and coordinate files.

    Loads the built-in HyresBuilder RNA and Protein force-field parameters,
    constructs a periodic OpenMM system with ``CutoffPeriodic`` non-bonded
    interactions and H-bond constraints, then uses ParmEd to serialise the
    result to a GROMACS ``.top`` topology and ``.gro`` coordinate file.

    Parameters
    ----------
    pdb : str
        Path to the input PDB file containing atomic coordinates.
        Default: ``'conf.pdb'``.
    psf : str
        Path to the input CHARMM PSF file containing bonded topology.
        Default: ``'conf.psf'``.
    outtop : str
        Path for the output GROMACS topology file (``.top``).
        Default: ``'conf.top'``.
    outgro : str
        Path for the output GROMACS coordinate file (``.gro``).
        Default: ``'conf.gro'``.
    box : float
        Side length of the cubic periodic simulation box in Angstroms.
        Default: ``100.0``.

    Notes
    -----
    - Non-bonded method: ``CutoffPeriodic``.
    - Constraints: ``HBonds`` (bonds involving hydrogen are constrained).
    - Force-field files are resolved automatically via
      :func:`HyresBuilder.utils.load_ff` for both ``'RNA'`` and ``'Protein'``
      residue types.

    Examples
    --------
    .. code-block:: bash

        python cli.py psf2top \\
            --pdb input.pdb \\
            --psf input.psf \\
            --outtop system.top \\
            --outgro system.gro \\
            --box 120.0
    """
    top_RNA, param_RNA = utils.load_ff('RNA')
    top_pro, param_pro = utils.load_ff('Protein')
    pdb_obj = PDBFile(pdb)
    psf_obj = CharmmPsfFile(psf)
    topology = psf_obj.topology
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)
    psf_obj.setBox(box, box, box)

    system = psf_obj.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds)

    structure = chem.openmm.load_topology(topology, system)
    structure.save(outtop, overwrite=True)
    structure.coordinates = pdb_obj.positions
    structure.save(outgro, overwrite=True)
    typer.echo(f'Converted to {outtop} and {outgro} files.')


@app.command()
def psftop(
    pdb: str = typer.Option("conf.pdb", help="Input PDB coordinate file."),
    psf: str = typer.Option("conf.psf", help="Input CHARMM PSF topology file."),
    top: str = typer.Option("conf.top", help="Output GROMACS topology file (.top)."),
    gro: str = typer.Option("conf.gro", help="Output GROMACS coordinate file (.gro)."),
    box: float = typer.Option(100.0, help="Cubic periodic box side length in Angstroms."),
) -> None:
    """
    Convert a CHARMM PSF + PDB pair to GROMACS topology and coordinate files.

    Functionally identical to :func:`psf2top`; retained for backwards
    compatibility with scripts that use the shorter ``psftop`` command name.
    New workflows should prefer :func:`psf2top`, which uses unambiguous
    option names (``--outtop`` / ``--outgro``) to avoid confusion with the
    input ``--psf`` option.

    Parameters
    ----------
    pdb : str
        Path to the input PDB file containing atomic coordinates.
        Default: ``'conf.pdb'``.
    psf : str
        Path to the input CHARMM PSF file containing bonded topology.
        Default: ``'conf.psf'``.
    top : str
        Path for the output GROMACS topology file (``.top``).
        Default: ``'conf.top'``.
    gro : str
        Path for the output GROMACS coordinate file (``.gro``).
        Default: ``'conf.gro'``.
    box : float
        Side length of the cubic periodic simulation box in Angstroms.
        Default: ``100.0``.

    Notes
    -----
    - Non-bonded method: ``CutoffPeriodic``.
    - Constraints: ``HBonds`` (bonds involving hydrogen are constrained).
    - Force-field files are resolved automatically via
      :func:`HyresBuilder.utils.load_ff` for both ``'RNA'`` and ``'Protein'``
      residue types.

    Examples
    --------
    .. code-block:: bash

        python cli.py psftop \\
            --pdb input.pdb \\
            --psf input.psf \\
            --top system.top \\
            --gro system.gro \\
            --box 120.0
    """
    top_RNA, param_RNA = utils.load_ff('RNA')
    top_pro, param_pro = utils.load_ff('Protein')
    pdb_obj = PDBFile(pdb)
    psf_obj = CharmmPsfFile(psf)
    topology = psf_obj.topology
    params = CharmmParameterSet(top_RNA, param_RNA, top_pro, param_pro)
    psf_obj.setBox(box, box, box)

    system = psf_obj.createSystem(params, nonbondedMethod=CutoffPeriodic, constraints=HBonds)

    structure = chem.openmm.load_topology(topology, system)
    structure.save(top, overwrite=True)
    structure.coordinates = pdb_obj.positions
    structure.save(gro, overwrite=True)
    typer.echo(f'Converted to {top} and {gro} files.')


if __name__ == "__main__":
    app()