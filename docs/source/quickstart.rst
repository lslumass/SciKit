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

Install SciKit via pip:

.. code-block:: bash

   pip install scikit

Or install directly from source:

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
   * - :mod:`scikit.Analysis`
     - Core data analysis routines (statistics, signal processing, decomposition)
   * - :mod:`scikit.plots`
     - Ready-made Matplotlib figures for scientific data
   * - :mod:`scikit.tools`
     - High-level workflow tools that combine analysis and plotting
   * - :mod:`scikit.utils`
     - Shared helpers for I/O, data validation, and unit conversion

----

Analysis Module
---------------

The :mod:`scikit.Analysis` module provides the core numerical routines.
Import it as:

.. code-block:: python

   from scikit import Analysis

A typical workflow:

.. code-block:: python

   import numpy as np
   from scikit import Analysis

   # Load or generate your data
   data = np.loadtxt("my_data.csv", delimiter=",")

   # Run analysis (replace with actual function names)
   result = Analysis.run(data)
   print(result)

For the full API, see :doc:`api/analysis`.

----

Plots Module
------------

The :mod:`scikit.plots` module wraps Matplotlib to produce publication-ready
figures with minimal boilerplate:

.. code-block:: python

   import numpy as np
   from scikit import plots

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

The :mod:`scikit.tools` module provides high-level convenience functions that
combine analysis and plotting into single calls:

.. code-block:: python

   from scikit import tools

   # One-liner end-to-end pipeline (replace with actual function names)
   tools.run_pipeline("my_data.csv", output_dir="results/")

SciKit also exposes a **command-line interface** powered by Typer. After
installation you can run:

.. code-block:: bash

   scikit --help

For the full API, see :doc:`api/tools`.

----

Utils Module
------------

The :mod:`scikit.utils` module contains shared helpers used across the package.
You can also call them directly:

.. code-block:: python

   from scikit import utils

   # File I/O helper (replace with actual function names)
   data = utils.load("my_data.csv")

   # Validation helper
   utils.validate(data)

For the full API, see :doc:`api/utils`.

----

A Minimal End-to-End Example
-----------------------------

.. code-block:: python

   import numpy as np
   from scikit import Analysis, plots, tools, utils

   # 1. Load data
   data = utils.load("experiment.csv")

   # 2. Validate
   utils.validate(data)

   # 3. Analyse
   result = Analysis.run(data)

   # 4. Visualise
   fig, ax = plots.plot(result)
   fig.savefig("figure.png", dpi=150)

   # 5. Or run the full pipeline at once
   tools.run_pipeline("experiment.csv", output_dir="results/")

----

Next Steps
----------

- Browse the full :doc:`api/analysis`, :doc:`api/plots`, :doc:`api/tools`, and :doc:`api/utils` references.
- Check the `GitHub repository <https://github.com/lslumass/SciKit>`_ for
  example notebooks and issue tracking.
- Found a bug? Open an issue on GitHub.