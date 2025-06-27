---
description: "NeMo Run documentation - Streamline ML experiment configuration, execution and management"
tags: ["nemo-run", "ml", "experiments", "configuration", "execution", "management"]
categories: ["documentation"]
---

(nemo-run-home)=

# NeMo Run Documentation

NeMo Run is a powerful tool designed to streamline the configuration, execution and management of Machine Learning experiments across various computing environments. NeMo Run has three core responsibilities:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: doc
:link-alt: Configuration guide

Learn how to configure your ML experiments and environments.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: guides/execution
:link-type: doc
:link-alt: Execution guide

Execute your configured experiments across various computing environments.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: guides/management
:link-type: doc
:link-alt: Management guide

Manage and monitor your running experiments and results.
:::

::::

This is the typical order Nemo Run users will follow to setup and launch experiments.

---

## About

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` About NeMo Run
:link: about/index
:link-type: doc
:link-alt: About NeMo Run

Overview of NeMo Run's core concepts and architecture.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Key Features
:link: about/key-features
:link-type: doc
:link-alt: Key features

Explore the technical capabilities and implementation details.
:::

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Why Choose NeMo Run
:link: about/why-nemo-run
:link-type: doc
:link-alt: Why choose NeMo Run

Learn why NeMo Run is the preferred choice for ML experiment management.
:::

::::

::::{toctree}
:hidden:
:caption: About
:maxdepth: 2
about/index
about/key-features
about/why-nemo-run
::::

---

## Get Started

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index
get-started/install
get-started/tutorials
::::

::::{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/configuration
guides/execution
guides/management
guides/ray
guides/packaging
guides/cli
guides/admin
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
faqs
troubleshooting
::::

---

## Guides

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Interface
:link: guides/cli
:link-type: doc
:link-alt: CLI guide

Create command-line interfaces with rich argument parsing and factory functions.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deploy Ray Clusters and Jobs
:link: guides/ray
:link-type: doc
:link-alt: Deploy Ray Clusters and Jobs

Deploy and manage Ray clusters and jobs for scalable distributed computing.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: guides/packaging
:link-type: doc
:link-alt: Packaging guide

Deploy your code using Git archives, pattern matching, or hybrid packaging strategies.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Administration
:link: guides/admin
:link-type: doc
:link-alt: Administration guide

Deploy, maintain, and operate NeMo Run in production environments.
:::

::::
