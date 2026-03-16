Analysis Module
===============

.. automodule:: SciKit.Analysis

----

CLI Commands
------------

All functions below are registered as sub-commands of the ``scical``
entry-point, defined in ``pyproject.toml`` as:

.. code-block:: toml

   [project.scripts]
   scical = "SciKit.Analysis:app"

After installation, run ``scical --help`` for a full command summary, or
``scical <command> --help`` for the option list of any individual command:

.. code-block:: bash

   scical --help
   scical rg --help

Mean Squared Displacement
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_msd

----

Radius of Gyration
~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_rg

----

DSSP Secondary Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_dssp

----

Pairwise Cα Distances
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_dist

----

Distance Autocorrelation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_dist_acf

----

Vector Autocorrelation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_vector_acf

----

Contact Maps
~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_contacts

----

Aggregation Analysis
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SciKit.Analysis.cmd_aggr