Quick Start
===========

**SciKit** is a Python package for scientific data analysis and visualization,
built on top of NumPy, SciPy, and Matplotlib. It provides four focused modules
covering analysis pipelines, plotting utilities, high-level tools, and helper
functions.

.. contents:: On this page
   :local:
   :depth: 1

----

Installation
------------

Install SciKit from source:

.. code-block:: bash

   git clone https://github.com/lslumass/SciKit.git
   cd SciKit
   pip install .

**Dependencies** — installed automatically:

- `numpy >= 1.20 <https://numpy.org>`_
- `scipy >= 1.7 <https://scipy.org>`_
- `matplotlib >= 3.3 <https://matplotlib.org>`_
- `typer >= 0.9 <https://typer.tiangolo.com>`_

----

Package Overview
----------------

SciKit is organised into four modules, each with a clear responsibility:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Module
     - Purpose
   * - :mod:`SciKit.Analysis`
     - MD trajectory analysis exposed as CLI sub-commands (MSD, Rg, DSSP, distances, contacts, aggregation)
   * - :mod:`SciKit.plots`
     - Ready-made Matplotlib figures for scientific data
   * - :mod:`SciKit.tools`
     - High-level workflow tools that combine analysis and plotting
   * - :mod:`SciKit.utils`
     - Shared helpers for I/O, data validation, and unit conversion

----

Analysis Module
---------------

The :mod:`SciKit.Analysis` module is a unified MD analysis toolkit.  All eight
analyses are registered as sub-commands of a single ``scical`` CLI entry-point
powered by `Typer <https://typer.tiangolo.com>`_.  List all available commands
and global options with:

.. code-block:: bash

   scical --help

Each sub-command has its own ``--help`` flag that describes every option:

.. code-block:: bash

   scical msd --help
   scical rg --help
   scical dssp --help
   # … and so on

Available sub-commands
~~~~~~~~~~~~~~~~~~~~~~

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

Usage examples
~~~~~~~~~~~~~~

**Mean squared displacement**

.. code-block:: bash

   scical msd --top conf.psf --traj system.xtc \
       --resid 1:50 --outdir ./msd_results --nproc 8

Output: ``./msd_results/<segid>_msd.dat`` — two columns: lag time (ps) and MSD (Å²).

**Radius of gyration**

.. code-block:: bash

   scical rg --top conf.psf --traj system.xtc \
       --out rg.dat --stride 5 --nproc 4

Output: ``rg.dat`` — columns: frame index, Rg (Å) per segment.

**DSSP secondary structure**

.. code-block:: bash

   scical dssp --top conf.psf --traj system.xtc \
       --hout helicity.dat --bout beta.dat --stride 10 --nproc 4

Output: ``helicity.dat`` and ``beta.dat`` — per-residue helix / β-sheet
fractions (0–1) for each segment.

**Cα–Cα pair distances**

Prepare a pair file (comma- or space-separated, ``#`` comments allowed):

.. code-block:: text

   # resid1  segid1  resid2  segid2
   10        PROA    45      PROA
   10        PROA    45      PROB

Then run:

.. code-block:: bash

   scical distance --top conf.psf --traj system.dcd \
       -f pairs.dat --stride 2 --workers 8

Output: ``pairs_distance.dat`` — one row per pair, one column per frame.

**Distance autocorrelation function**

.. code-block:: bash

   scical distance-acf --top conf.psf --traj system.xtc \
       --pairs pairs.dat --out distance_acf.dat --stride 10 --nproc 4

Output: ``distance_acf.dat`` — columns: lag time (ps), normalised ACF per pair.

**Vector autocorrelation function**

.. code-block:: bash

   scical vector-acf --top conf.psf --traj system.xtc \
       --pairs pairs.dat --out vector_acf.dat --stride 10 --nproc 4

Output: ``vector_acf.dat`` — columns: lag time (ps), normalised vector ACF per pair.

**Contact maps**

.. code-block:: bash

   scical contacts --top system.psf --traj traj.dcd \
       --cutoff 8.0 --stride 5 --nproc 8 --out contacts

Output: four NumPy binary files — ``contacts_intra.npy``, ``contacts_inter.npy``,
``contacts.npy`` (combined), and ``contacts_resids.npy``.

**Aggregation analysis**

.. code-block:: bash

   # Basic: cluster statistics only
   scical aggr --top conf.psf --traj system.xtc --rcut 8.0

   # With PBC recentering and radial density profile
   scical aggr --top conf.psf --traj system.xtc \
       --rcut 8.0 --recenter --density --dr 2.0 \
       --n-frames-avg 100 --outtraj recentered.xtc

Output: ``aggr.dat`` (per-frame monomer / cluster counts),
``recentered.xtc`` (recentered trajectory), and
``density_profile.dat`` (radial concentration in mM vs. radius in Å).

For the full API reference, see :doc:`api/analysis`.

----

Plots Module
------------

The :mod:`SciKit.plots` module wraps Matplotlib to produce publication-ready
figures with minimal boilerplate:

.. code-block:: python

   import numpy as np
   from SciKit import plots

   data = np.random.randn(200)

   # Generate a plot (replace with actual function names)
   fig, ax = plots.plot(data, title="My Dataset")
   fig.savefig("output.png", dpi=150)

All plotting functions return a ``(fig, ax)`` tuple so you can further
customise them with standard Matplotlib commands.

For the full API, see :doc:`api/plots`.

----

Tools Module
------------

The :mod:`SciKit.tools` module provides high-level convenience functions that
combine analysis and plotting into single calls:

.. code-block:: python

   from SciKit import tools

   # One-liner end-to-end pipeline (replace with actual function names)
   tools.run_pipeline("my_data.csv", output_dir="results/")

For the full API, see :doc:`api/tools`.

----

Utils Module
------------

The :mod:`SciKit.utils` module contains shared helpers used across the package.
You can also call them directly:

.. code-block:: python

   from SciKit import utils

   # File I/O helper (replace with actual function names)
   data = utils.load("my_data.csv")

   # Validation helper
   utils.validate(data)

For the full API, see :doc:`api/utils`.

----

Next Steps
----------

- Browse the full :doc:`api/analysis`, :doc:`api/plots`, :doc:`api/tools`, and :doc:`api/utils` references.
- Check the `GitHub repository <https://github.com/lslumass/SciKit>`_ for
  example notebooks and issue tracking.
- Found a bug? Open an issue on GitHub.