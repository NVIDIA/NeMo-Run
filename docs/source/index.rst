.. NeMo-Run documentation master file, created by
   sphinx-quickstart on Thu Jul 25 17:57:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NeMo-Run documentation
======================

NeMo-Run is a powerful tool designed to streamline the configuration, execution and management of Machine Learning experiments across various computing environments. NeMo Run has three core responsibilities:

1. `Configuration <./guides/configuration.html>`_
2. `Execution <./guides/execution.html>`_
3. `Management <./guides/management.html>`_

Please click into each link to learn more.
This is also the typical order Nemo Run users will follow to setup and launch experiments.

.. toctree::
   :maxdepth: 1

   guides/index.rst
   API Reference <api/nemo_run/index>
   faqs

Installation
------------
To install the project, use the following command:

``pip install git+https://github.com/NVIDIA/NeMo-Run.git``

To install Skypilot, we have optional features available.

``pip install git+https://github.com/NVIDIA/NeMo-Run.git[skypilot]``
will install Skypilot w Kubernetes

``pip install git+https://github.com/NVIDIA/NeMo-Run.git[skypilot-all]``
will install Skypilot w all clouds

You can also manually install Skypilot from https://skypilot.readthedocs.io/en/latest/getting-started/installation.html

If using DGX Cloud Lepton, use the following command to install the Lepton CLI:

``pip install leptonai``

To authenticate with the DGX Cloud Lepton cluster, navigate to the **Settings > Tokens** page in the DGX Cloud Lepton UI and copy the ``lep login`` command shown on the page and run it in the terminal.

Make sure you have `pip` installed and configured properly.


Tutorials
---------

The ``hello_world`` tutorial series provides a comprehensive introduction to NeMo Run, demonstrating its capabilities through a simple example. The tutorial covers:

- Configuring Python functions using ``Partial`` and ``Config`` classes.
- Executing configured functions locally and on remote clusters.
- Visualizing configurations with ``graphviz``.
- Creating and managing experiments using ``run.Experiment``.

You can find the tutorial series below:

1. `Part 1 <../../../NeMo-Run/examples/hello-world/hello_world.ipynb>`
2. `Part 2 <../../../NeMo-Run/examples/hello-world/hello_experiments.ipynb>`
3. `Part 3 <../../../NeMo-Run/examples/hello-world/hello_scripts.py>`
