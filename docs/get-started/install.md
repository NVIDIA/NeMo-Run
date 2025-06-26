---
description: "Install NeMo Run and optional dependencies for different computing environments and cloud platforms."
tags: ["installation", "setup", "dependencies", "skypilot", "lepton"]
categories: ["get-started"]
---

(install-guide)=

# Installation

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

## DGX Cloud Lepton Setup

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
