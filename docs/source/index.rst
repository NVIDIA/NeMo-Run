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
   :glob:
   :maxdepth: 1

   api/index.rst
   faq*

Installation
---------
To install the project, use the following command:

``pip install git+https://github.com/NVIDIA/NeMo-Run.git``

To install Skypilot, we have optional features available.

``pip install git+https://github.com/NVIDIA/NeMo-Run.git[skypilot]``
will install Skypilot w Kubernetes

``pip install git+https://github.com/NVIDIA/NeMo-Run.git[skypilot-all]``
will install Skypilot w all clouds

You can also manually install Skypilot from https://skypilot.readthedocs.io/en/latest/getting-started/installation.html

Make sure you have `pip` installed and configured properly.


Tutorials
---------

The ``hello_world`` tutorial series provides a comprehensive introduction to NeMo Run, demonstrating its capabilities through a simple example. The tutorial covers:

- Configuring Python functions using ``Partial`` and ``Config`` classes.
- Executing configured functions locally and on remote clusters.
- Visualizing configurations with ``graphviz``.
- Creating and managing experiments using ``run.Experiment``.

You can find the tutorial series below:

1. `Part 1 <examples/hello-world/hello_world.ipynb>`_
2. `Part 2 <examples/hello-world/hello_experiments.ipynb>`_
3. `Part 3 <examples/hello-world/hello_scripts.py>`_
