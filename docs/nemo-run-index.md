---
description: "NeMo-Run documentation - Streamline ML experiment configuration, execution and management"
tags: ["overview", "quickstart", "getting-started"]
categories: ["getting-started"]
---

(nemo-run-home)=

# NeMo-Run Documentation

NeMo-Run is a powerful tool designed to streamline the configuration, execution and management of Machine Learning experiments across various computing environments. NeMo Run has three core responsibilities:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`settings;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: ref
:link-alt: Configuration guide

Learn how to configure your ML experiments and environments.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: guides/execution
:link-type: ref
:link-alt: Execution guide

Execute your configured experiments across various computing environments.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: guides/management
:link-type: ref
:link-alt: Management guide

Manage and monitor your running experiments and results.
:::

::::

This is the typical order Nemo Run users will follow to setup and launch experiments.

## Installation

To install the project, use the following command:

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

To install Skypilot, we have optional features available:

::::{tab-set}

:::{tab-item} Kubernetes Support
:sync: sync-skypilot

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]
```

This will install Skypilot with Kubernetes support.
:::

:::{tab-item} All Cloud Support
:sync: sync-skypilot

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot-all]
```

This will install Skypilot with support for all cloud providers.
:::

::::

You can also manually install Skypilot from [https://skypilot.readthedocs.io/en/latest/getting-started/installation.html](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)

### DGX Cloud Lepton Setup

If using DGX Cloud Lepton, use the following command to install the Lepton CLI:

```bash
pip install leptonai
```

To authenticate with the DGX Cloud Lepton cluster:

1. Navigate to the **Settings > Tokens** page in the DGX Cloud Lepton UI
2. Copy the `lep login` command shown on the page
3. Run it in your terminal

:::{note}
Make sure you have `pip` installed and configured properly before proceeding.
:::

## Tutorials

The `hello_world` tutorial series provides a comprehensive introduction to NeMo Run, demonstrating its capabilities through a simple example. The tutorial covers:

- Configuring Python functions using `Partial` and `Config` classes
- Executing configured functions locally and on remote clusters
- Visualizing configurations with `graphviz`
- Creating and managing experiments using `run.Experiment`

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Part 1: Hello World
:link: ../../../NeMo-Run/examples/hello-world/hello_world.ipynb
:link-type: url
:link-alt: Hello World tutorial part 1

Basic configuration and execution setup.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Part 2: Hello Experiments
:link: ../../../NeMo-Run/examples/hello-world/hello_experiments.ipynb
:link-type: url
:link-alt: Hello World tutorial part 2

Experiment management and tracking.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Part 3: Hello Scripts
:link: ../../../NeMo-Run/examples/hello-world/hello_scripts.py
:link-type: url
:link-alt: Hello World tutorial part 3

Script-based execution and automation.
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/configuration
guides/execution
guides/management
guides/why-use-nemo-run
guides/ray
guides/cli
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
API Reference <api/nemo_run/index>
faqs
::::
