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

**Available sub-commands**

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